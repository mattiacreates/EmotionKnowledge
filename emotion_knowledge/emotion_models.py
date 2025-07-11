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


class MultimodalEmotionModel:
    """Simple fusion of an audio and text emotion model."""

    def __init__(
        self,
        audio_model: EmotionModel,
        text_classifier,
        neutral_label: str = "neutral",
    ) -> None:
        self.audio_model = audio_model
        self.text_classifier = text_classifier
        self.neutral_label = neutral_label.lower()

    def predict(self, audio_path: str, text: str) -> str:
        audio_label, _ = self.audio_model.predict(audio_path)
        text_label = ""
        if text:
            text_res = self.text_classifier(text)
            if text_res:
                text_label = text_res[0].get("label", "")

        if audio_label and audio_label.lower() != self.neutral_label:
            return audio_label
        return text_label or audio_label
