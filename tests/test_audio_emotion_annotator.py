import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emotion_knowledge.audio_emotion_annotator import AudioEmotionAnnotator
from emotion_knowledge.emotion_models import EmotionModel


class FakeModel(EmotionModel):
    def __init__(self):
        pass

    def predict(self, audio_path: str) -> str:  # type: ignore[override]
        FakeModel.called_with = audio_path
        return "anger"


def test_audio_annotator_uses_injected_model(monkeypatch, tmp_path):
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    model = FakeModel()
    annotator = AudioEmotionAnnotator(emotion_model=model)
    entry = {"audio_clip_path": str(tmp_path / "audio.wav"), "text": "Hallo"}
    result = annotator.invoke(entry)
    assert result["emotion_annotated_text"] == "[Wut] Hallo"
    assert FakeModel.called_with == entry["audio_clip_path"]

