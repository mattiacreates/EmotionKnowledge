"""LangChain Runnables for diarized emotion recognition."""

from __future__ import annotations

import os
from typing import List, Dict, Any

import torch
from langchain_core.runnables import Runnable
from transformers import AutoModelForSequenceClassification, AutoProcessor

from . import transcribe_diarize_whisperx


class AudioTranscriber(Runnable):
    """Transcribes audio and returns diarized segments."""

    def invoke(self, audio_path: str) -> List[Dict[str, Any]]:
        text = transcribe_diarize_whisperx.invoke(audio_path)
        # Each line from transcribe_diarize_whisperx has format: [Speaker] text
        segments = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.startswith("[") and "]" in line:
                speaker, utterance = line.split("]", 1)
                segments.append({
                    "speaker": speaker.strip("[]"),
                    "text": utterance.strip(),
                })
        if not segments:
            raise ValueError("No segments produced from transcription")
        return segments


class EmotionDetector(Runnable):
    """Annotates segments with emotions using a HF model."""

    def __init__(self, model_id: str = "ZebangCheng/Emotion-LLaMA") -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        if device >= 0:
            self.model = self.model.to(device)
        self.device = device

    def invoke(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        annotated = []
        for seg in segments:
            inputs = self.processor(text=seg["text"], return_tensors="pt")
            if self.device >= 0:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            scores = out.logits.softmax(dim=-1)[0]
            label_id = int(scores.argmax())
            label = self.model.config.id2label[label_id]
            seg = dict(seg)
            seg["emotion"] = label
            seg["confidence"] = float(scores[label_id])
            annotated.append(seg)
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


def emotion_transcription_pipeline(model_id: str = "ZebangCheng/Emotion-LLaMA") -> RunnableSequence:
    transcriber = AudioTranscriber()
    detector = EmotionDetector(model_id=model_id)
    formatter = TranscriptFormatter()
    return transcriber | detector | formatter
