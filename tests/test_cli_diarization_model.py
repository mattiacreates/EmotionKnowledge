import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import emotion_knowledge


def test_cli_diarization_model(monkeypatch, tmp_path, caplog):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"")

    class FakeTranscribe:
        def invoke(self, params):
            model = params.get("diarization_model")
            emotion_knowledge.logger.info("Using diarization model '%s'", model)
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
            "--diarization-model",
            "mistralai/Voxtral-Small-24B-2507",
        ],
    )

    with caplog.at_level(logging.INFO, logger="emotion_knowledge"):
        emotion_knowledge.main()

    assert (
        "Using diarization model 'mistralai/Voxtral-Small-24B-2507'" in caplog.text
    )
