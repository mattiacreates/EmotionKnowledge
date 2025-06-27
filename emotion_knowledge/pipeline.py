"""LangChain Runnables for German speech transcription and sentiment."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from langchain_core.runnables import Runnable
from pydub import AudioSegment
import chromadb
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import whisperx


class AudioTranscriber(Runnable):
    """Transcribes German speech with optional diarization using WhisperX."""

    def __init__(self, *, diarize: bool = False, model_size: str = "small") -> None:
        self.diarize = diarize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisperx.load_model(
            model_size, device=self.device, language="de", compute_type="int8"
        )
        if diarize:
            self.align_model, self.metadata = whisperx.load_align_model(
                language_code="de", device=self.device
            )
            token = os.getenv("HF_TOKEN")
            self.diarize_pipeline = whisperx.DiarizationPipeline(
                device=self.device, use_auth_token=token
            )

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> Dict[str, Any]:
        print("✅ Using updated AudioTranscriber.invoke")
        """Transcribe audio file and return text segments with timestamps."""
        audio_path = input if isinstance(input, str) else input.get("audio_path")
        assert os.path.exists(audio_path), f"File not found: {audio_path}"
        result = self.model.transcribe(audio_path)
        segments = result["segments"]

        if not self.diarize:
            utterances = [
                {
                    "speaker": "Speaker",
                    "text": seg["text"].strip(),
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                }
                for seg in segments
            ]
            return {"audio_path": audio_path, "segments": utterances}

        aligned = whisperx.align(
            segments, self.align_model, self.metadata, audio_path, device=self.device
        )
        diarized = self.diarize_pipeline(audio_path)
        with_speakers = whisperx.assign_word_speakers(diarized, aligned)
        words = with_speakers["word_segments"]

        utterances = []
        current_speaker = None
        current_line = ""
        start = 0.0
        end = 0.0
        for word in words:
            speaker = word.get("speaker", "Speaker")
            if speaker != current_speaker:
                if current_line:
                    utterances.append(
                        {
                            "speaker": current_speaker,
                            "text": current_line.strip(),
                            "start": start,
                            "end": end,
                        }
                    )
                    current_line = ""
                current_speaker = speaker
                start = word.get("start", 0.0)
            current_line += word.get("text", word.get("word", "")) + " "
            end = word.get("end", end)
        if current_line:
            utterances.append(
                {
                    "speaker": current_speaker,
                    "text": current_line.strip(),
                    "start": start,
                    "end": end,
                }
            )
        return {"audio_path": audio_path, "segments": utterances}


class SegmentDBWriter(Runnable):
    """Save segments to ChromaDB and split audio into per-speaker clips."""

    def __init__(
        self,
        *,
        db_path: str = "segment_db",
        collection_name: str = "segments",
        clip_dir: str = "clips",
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.clip_dir = clip_dir
        os.makedirs(clip_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def invoke(self, data: Dict[str, Any], config: Optional[dict] = None) -> str:
        """Persist segments in the DB and write audio clips."""
        audio_path = data["audio_path"]
        segments = data["segments"]
        audio = AudioSegment.from_file(audio_path)
        docs, metas, ids = [], [], []
        for i, seg in enumerate(segments):
            start_ms = int(seg.get("start", 0) * 1000)
            end_ms = int(seg.get("end", 0) * 1000)
            speaker = seg.get("speaker", "Speaker")
            clip_name = f"{i}_{speaker}.wav"
            clip_path = os.path.join(self.clip_dir, clip_name)
            audio[start_ms:end_ms].export(clip_path, format="wav")
            seg["clip_path"] = clip_path
            docs.append(seg["text"])
            metas.append(seg)
            ids.append(str(i))
        self.collection.add(documents=docs, metadatas=metas, ids=ids)
        return self.db_path


class DBEmotionAnnotator(Runnable):
    """Fetches segments from a Chroma DB and annotates them with emotions."""

    def __init__(
        self,
        *,
        db_path: str = "segment_db",
        collection_name: str = "segments",
        model_id: str = "oliverguhr/german-sentiment-bert",
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        if self.device >= 0:
            self.model = self.model.to(self.device)

    def invoke(self, _: Any = None, config: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Annotate all segments in the DB and return them."""
        docs = self.collection.get(include=["documents", "metadatas"])
        ids = docs["ids"]
        texts = docs["documents"]
        metas = docs["metadatas"]
        annotated = []
        for _id, text, meta in zip(ids, texts, metas):
            text_short = text[:200] if len(text) > 200 else text
            inputs = self.tokenizer(text_short, return_tensors="pt", truncation=True)
            if self.device >= 0:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                output = self.model(**inputs)
                if self.device >= 0:
                    torch.cuda.empty_cache()
            scores = output.logits.softmax(dim=-1)[0]
            label = self.model.config.id2label[int(scores.argmax())]
            meta["emotion"] = label
            annotated.append({**meta, "text": text})
            self.collection.update(ids=[_id], metadatas=[meta])
        return annotated


class TranscriptFormatter(Runnable):
    """Formats the annotated transcript."""

    def invoke(
        self, segments: List[Dict[str, Any]], config: Optional[dict] = None
    ) -> str:
        """Format annotated segments. The optional config dict is ignored."""
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Speaker")
            emotion = seg.get("emotion", "")
            line = f"[{speaker}][{emotion}] {seg['text']}"
            lines.append(line)
        return "\n".join(lines)


# Convenience pipeline using RunnableSequence
from langchain_core.runnables import RunnableSequence


def emotion_transcription_pipeline(
    *,
    diarize: bool = False,
    asr_model_size: str = "small",
    model_id: str = "oliverguhr/german-sentiment-bert",
    db_path: str = "segment_db",
    collection_name: str = "segments",
    clip_dir: str = "clips",
) -> RunnableSequence:
    """Build the ASR -> DB -> emotion pipeline."""

    transcriber = AudioTranscriber(diarize=diarize, model_size=asr_model_size)
    saver = SegmentDBWriter(
        db_path=db_path, collection_name=collection_name, clip_dir=clip_dir
    )
    annotator = DBEmotionAnnotator(
        db_path=db_path, collection_name=collection_name, model_id=model_id
    )
    formatter = TranscriptFormatter()
    return transcriber | saver | annotator | formatter
