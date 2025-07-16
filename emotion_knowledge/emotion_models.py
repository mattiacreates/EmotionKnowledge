from __future__ import annotations

from transformers import pipeline


class EmotionModel:
    """Wrapper around a Hugging Face audio-classification pipeline."""

    def __init__(self, model_name: str = "padmalcom/wav2vec2-large-emotion-detection-german") -> None:
        # load from local cache if available, no auth token required
        self.classifier = pipeline("audio-classification", model=model_name)

    def predict(self, audio_path: str) -> tuple[str, float]:
        """Return the top emotion label and confidence for the given audio file."""
        result = self.classifier(audio_path)
        if result:
            top = result[0]
            label = top.get("label", "")
            score = float(top.get("score", 0.0))
            return label, score
        return "", 0.0


class TextEmotionModel:
    """Wrapper around a text-classification pipeline for emotion detection."""

    def __init__(self, model_name: str = "oliverguhr/german-sentiment-bert") -> None:
        self.classifier = pipeline("text-classification", model=model_name)

    def predict(self, text: str) -> tuple[str, float]:
        """Return the top emotion label and confidence for the given text."""
        result = self.classifier(text)
        if result:
            top = result[0]
            label = top.get("label", "")
            score = float(top.get("score", 0.0))
            return label, score
        return "", 0.0
