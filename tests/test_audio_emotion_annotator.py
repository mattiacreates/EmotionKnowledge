import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from emotion_knowledge.audio_emotion_annotator import AudioEmotionAnnotator
from emotion_knowledge.emotion_models import EmotionModel


class FakeModel(EmotionModel):
    def __init__(self):
        pass

    def predict(self, audio_path: str) -> tuple[str, float]:  # type: ignore[override]
        FakeModel.called_with = audio_path
        return "anger", 0.85


def test_audio_annotator_uses_injected_model(monkeypatch, tmp_path):
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    model = FakeModel()
    annotator = AudioEmotionAnnotator(emotion_model=model)
    entry = {"audio_clip_path": str(tmp_path / "audio.wav"), "text": "Hallo"}
    result = annotator.invoke(entry)
    assert result["audio_emotion_label"] == "anger"
    assert result["audio_emotion_confidence"] == pytest.approx(0.85)
    assert result["audio_text"] == "Hallo"
    assert FakeModel.called_with == entry["audio_clip_path"]

