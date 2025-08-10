import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import emotion_knowledge


def test_cli_sortformer_model(tmp_path, monkeypatch, caplog):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"")

    class FakeTranscribe:
        def invoke(self, params):
            model_name = params.get("model_name", "nvidia/diar_sortformer_4spk-v1")
            emotion_knowledge.logger.info(
                "Loading Sortformer diarization model '%s'", model_name
            )
            return {"text": "", "segments": []}

    monkeypatch.setattr(
        emotion_knowledge, "transcribe_diarize_sortformer", FakeTranscribe()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            str(audio_file),
            "--diarize",
            "--sortformer-model",
            "local_model",
        ],
    )

    with caplog.at_level(logging.INFO, logger="emotion_knowledge"):
        emotion_knowledge.main()

    assert "Loading Sortformer diarization model 'local_model'" in caplog.text
