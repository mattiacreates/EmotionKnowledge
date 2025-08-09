import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import emotion_knowledge


def test_cli_sets_temperature(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"")

    class FakeTranscribe:
        def invoke(self, params):
            assert params.get("temperature") == 0.7
            return ""

    monkeypatch.setattr(
        emotion_knowledge, "transcribe_audio_whisper", FakeTranscribe()
    )
    monkeypatch.setattr(
        sys, "argv", ["prog", str(audio_file), "--temperature", "0.7"]
    )

    emotion_knowledge.main()

