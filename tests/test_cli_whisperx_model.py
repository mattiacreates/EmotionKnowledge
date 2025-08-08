import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import emotion_knowledge


def test_cli_whisperx_model_base(monkeypatch, tmp_path, caplog):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"")

    def fake_transcribe(audio_path: str, model_size: str = "medium"):
        emotion_knowledge.logger.info(
            "Starting WhisperX transcription using model '%s'", model_size
        )
        return {"text": "", "segments": []}

    monkeypatch.setattr(emotion_knowledge, "transcribe_diarize_whisperx", fake_transcribe)
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
