import os
import sys
import logging
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import emotion_knowledge


def test_cli_whisperx_model_base(monkeypatch, tmp_path, caplog):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"")

    class FakeTranscribe:
        def invoke(self, params):
            model_size = params.get("model_size", "medium")
            emotion_knowledge.logger.info(
                "Starting WhisperX transcription using model '%s'", model_size
            )
            return {"text": "", "segments": []}

    monkeypatch.setattr(
        emotion_knowledge, "transcribe_diarize_whisperx", FakeTranscribe()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            str(audio_file),
            "--diarize",
            "--whisperx-model",
            "base",
        ],
    )

    with caplog.at_level(logging.INFO, logger="emotion_knowledge"):
        emotion_knowledge.main()

    assert "Starting WhisperX transcription using model 'base'" in caplog.text


def test_cli_max_gap_forwarded(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"")

    class FakeTranscribe:
        def invoke(self, params):
            return {"text": "", "segments": [{"speaker": "s1", "start": 0.0, "end": 0.5}]}

    class FakeSaver:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            pass

    captured = {}

    def fake_group(seg, max_gap=0.7, **kwargs):
        captured["max_gap"] = max_gap
        return []

    monkeypatch.setattr(
        emotion_knowledge, "transcribe_diarize_whisperx", FakeTranscribe()
    )
    monkeypatch.setattr(emotion_knowledge, "SegmentSaver", FakeSaver)
    monkeypatch.setattr(emotion_knowledge, "_group_utterances", fake_group)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            str(audio_file),
            "--diarize",
            "--max-gap",
            "1.23",
        ],
    )

    emotion_knowledge.main()

    assert captured["max_gap"] == pytest.approx(1.23)
