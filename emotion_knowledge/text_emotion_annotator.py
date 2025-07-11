from __future__ import annotations
from typing import Any, Dict, Optional

try:
    from langchain_core.runnables import Runnable
except Exception:  # pragma: no cover - optional dependency
    class Runnable:  # minimal fallback so tests can import module
        def invoke(self, *args, **kwargs):  # pragma: no cover - not used in tests
            raise ImportError("langchain-core is required for this feature")

from .emotion_models import TextEmotionModel


class TextEmotionAnnotator(Runnable):
    """Annotate utterances with an emotion label predicted from text."""

    def __init__(
        self,
        emotion_model: Optional[TextEmotionModel] = None,
        label_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.emotion_model = emotion_model or TextEmotionModel()
        self.label_map = label_map or {}

    def invoke(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        text = entry.get("text", "")
        label = ""
        confidence = 0.0
        if text:
            raw_label, confidence = self.emotion_model.predict(text)
            if raw_label:
                label = self.label_map.get(raw_label.lower(), raw_label)
        entry["text_emotion_label"] = label
        entry["text_emotion_confidence"] = confidence
        return entry
