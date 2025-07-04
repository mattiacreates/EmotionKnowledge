import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emotion_knowledge.audio_emotion_annotator import AudioEmotionAnnotator
from emotion_knowledge.emotion_models import EmotionModel


class FakeModel(EmotionModel):
    def __init__(self):
        pass

    def predict_scores(self, audio_path: str) -> dict[str, float]:  # type: ignore[override]
        FakeModel.called_with = audio_path
        return {"anger": 0.9, "happy": 0.1}


def test_audio_annotator_uses_injected_model(monkeypatch, tmp_path):
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    model = FakeModel()
    annotator = AudioEmotionAnnotator(emotion_model=model)
    entry = {"audio_clip_path": str(tmp_path / "audio.wav"), "text": "Hallo"}
    result = annotator.invoke(entry)
    assert result["emotion_annotated_text"] == "[anger] Hallo"
    assert result["emotion_scores"] == {"anger": 0.9, "happy": 0.1}
    assert result["emotion_top_label"] == "anger"
    assert FakeModel.called_with == entry["audio_clip_path"]

