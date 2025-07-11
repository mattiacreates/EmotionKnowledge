import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from emotion_knowledge.text_emotion_annotator import TextEmotionAnnotator
from emotion_knowledge.emotion_models import TextEmotionModel


class FakeTextModel(TextEmotionModel):
    def __init__(self):
        pass

    def predict(self, text: str) -> tuple[str, float]:  # type: ignore[override]
        FakeTextModel.called_with = text
        return "Freude", 0.92


def test_text_emotion_annotator_keeps_other_fields():
    entry = {
        "text": "Hallo Welt",
        "emotion_annotated_text": "[anger] Hallo Welt",
        "emotion_confidence": 0.5,
    }
    annotator = TextEmotionAnnotator(emotion_model=FakeTextModel())
    result = annotator.invoke(entry)
    assert result["text_emotion_label"] == "Freude"
    assert result["text_emotion_confidence"] == pytest.approx(0.92)
    assert result["emotion_annotated_text"] == "[anger] Hallo Welt"
    assert result["emotion_confidence"] == pytest.approx(0.5)
    assert FakeTextModel.called_with == entry["text"]
