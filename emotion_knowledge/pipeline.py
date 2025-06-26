"""LangChain Runnables for German speech transcription and sentiment."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from langchain_core.runnables import Runnable
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

    def invoke(self, audio_path: str) -> List[Dict[str, Any]]:
        assert os.path.exists(audio_path), f"File not found: {audio_path}"
        result = self.model.transcribe(audio_path)
        segments = result["segments"]

        if not self.diarize:
            return [
                {"speaker": "Speaker", "text": seg["text"].strip()} for seg in segments
            ]

        aligned = whisperx.align(
            segments, self.align_model, self.metadata, audio_path, device=self.device
        )
        diarized = self.diarize_pipeline(audio_path)
        with_speakers = whisperx.assign_word_speakers(diarized, aligned)
        words = with_speakers["word_segments"]

        utterances = []
        current_speaker = None
        current_line = ""
        for word in words:
            speaker = word.get("speaker", "Speaker")
            if speaker != current_speaker:
                if current_line:
                    utterances.append(
                        {"speaker": current_speaker, "text": current_line.strip()}
                    )
                    current_line = ""
                current_speaker = speaker
            current_line += word.get("text", word.get("word", "")) + " "
        if current_line:
            utterances.append({"speaker": current_speaker, "text": current_line.strip()})
        return utterances


class EmotionAnnotator(Runnable):
    """Annotates text segments with emotions using a lightweight model."""

    def __init__(self, model_id: str = "oliverguhr/german-sentiment-bert") -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        if device >= 0:
            self.model = self.model.to(device)
        self.device = device

    def invoke(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        annotated = []
        for seg in segments:
            inputs = self.tokenizer(seg["text"], return_tensors="pt", truncation=True)
            if self.device >= 0:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            scores = out.logits.softmax(dim=-1)[0]
            label_id = int(scores.argmax())
            label = self.model.config.id2label[label_id]
            annotated.append(
                {
                    "speaker": seg.get("speaker", "Speaker"),
                    "text": seg["text"],
                    "emotion": label,
                }
            )
        return annotated


class TranscriptFormatter(Runnable):
    """Formats the annotated transcript."""

    def invoke(self, segments: List[Dict[str, Any]]) -> str:
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
) -> RunnableSequence:
    """Build the ASR -> emotion pipeline."""

    transcriber = AudioTranscriber(diarize=diarize, model_size=asr_model_size)
    annotator = EmotionAnnotator(model_id=model_id)
    formatter = TranscriptFormatter()
    return transcriber | annotator | formatter
