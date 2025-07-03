import os
import sys
import tempfile
from pydub import AudioSegment
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emotion_knowledge.emotion_tagger import MultimodalEmotionTagger
from emotion_knowledge.segment_saver import SegmentSaver




def test_multimodal_combines_text_and_audio(monkeypatch):
    calls = []

    def fake_pipeline(task, model=None):
        calls.append((task, model))
        if task == "text-classification":
            return lambda text: [{"label": "pos", "score": 0.6}]
        elif task == "audio-classification":
            return lambda path: [
                {"label": "neg", "score": 0.5},
                {"label": "pos", "score": 0.4},
            ]
        else:
            raise AssertionError("unexpected task")

    monkeypatch.setattr(
        "emotion_knowledge.emotion_tagger.pipeline", fake_pipeline
    )

    tagger = MultimodalEmotionTagger(
        text_model="text-model", audio_model="audio-model"
    )
    label = tagger.invoke("hi", "dummy.wav")

    assert label == "pos"
    assert ("text-classification", "text-model") in calls
    assert ("audio-classification", "audio-model") in calls


class FakeCollection:
    def __init__(self):
        self.add_calls = []

    def add(self, documents, metadatas, ids):
        self.add_calls.append((documents, metadatas, ids))


class FakeClient:
    def __init__(self, path):
        self.path = path
        self.collection = FakeCollection()

    def get_or_create_collection(self, name):
        return self.collection


def test_segment_saver_saves_audio_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "emotion_knowledge.segment_saver.chromadb.PersistentClient",
        FakeClient,
    )

    audio_path = tmp_path / "audio.wav"
    AudioSegment.silent(duration=1000).export(audio_path, format="wav")

    saver = SegmentSaver(db_path=str(tmp_path / "db"), output_dir=str(tmp_path))
    result = saver.invoke(
        {
            "audio_path": str(audio_path),
            "start": 0.0,
            "end": 0.5,
            "speaker": "spk",
            "text": "hello",
        }
    )

    assert os.path.exists(result["clip_path"])
    assert saver.collection.add_calls
    docs, metas, ids = saver.collection.add_calls[0]
    assert metas[0]["text"] == "hello"
    assert metas[0]["speaker"] == "spk"

