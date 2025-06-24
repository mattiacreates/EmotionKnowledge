import argparse
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
import whisper
import os


@tool
def transcribe_audio_whisper(audio_path: str) -> str:
    """Transkribiert deutsche Sprache aus einer Audiodatei mit Whisper."""
    assert os.path.exists(audio_path), f"Datei nicht gefunden: {audio_path}"
    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path, language=None, temperature=0.3)
    return result["text"].strip()


class TranscriptionOnlyWorkflow(Runnable):
    """Workflow zur reinen Transkription von Audio mit Whisper."""

    def invoke(self, audio_path: str) -> str:
        text = transcribe_audio_whisper.invoke(audio_path)
        print("📄 Transkribierter Text:\n")
        print(text)
        return text


def main():
    parser = argparse.ArgumentParser(description="Nur Transkription mit Whisper durchführen.")
    parser.add_argument("audio", help="Pfad zur Audiodatei (WAV/MP3)")
    args = parser.parse_args()

    workflow = TranscriptionOnlyWorkflow()
    _ = workflow.invoke(args.audio)


if __name__ == "__main__":
    main()
