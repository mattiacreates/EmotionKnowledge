import os
import sys
from pydub import AudioSegment
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emotion_knowledge.segment_saver import SegmentSaver


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
            "duration": 0.5,
            "n_words": 1,
            "words_per_sec": 2.0,
            "mean_word_gap": 0.0,
            "p95_word_gap": 0.0,
            "overlaps_started": False,
        }
    )

    assert os.path.exists(result["clip_path"])
    assert saver.collection.add_calls
    docs, metas, ids = saver.collection.add_calls[0]
    assert metas[0]["text"] == "hello"
    assert metas[0]["speaker"] == "spk"
    assert metas[0]["duration"] == pytest.approx(0.5)
    assert metas[0]["n_words"] == 1
    assert metas[0]["overlaps_started"] is False

