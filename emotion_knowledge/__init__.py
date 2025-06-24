import argparse
import os
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
import whisper


@tool
def transcribe_diarize_whisperx(audio_path: str) -> str:
    """Transkribiert Audio auf Deutsch mit WhisperX und Speaker-Diarization."""
    import torch
    import whisperx

    assert os.path.exists(audio_path), f"Datei nicht gefunden: {audio_path}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """ Changed to int8, to see if it works on collab"""
    model = whisperx.load_model("large-v3", device=device, language="de", compute_type="int8")
    result = model.transcribe(audio_path)

    align_model, metadata = whisperx.load_align_model(
        language_code="de", device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio_path, device=device
    )

    diarize_model = whisperx.DiarizationPipeline(device=device)
    diarize_segments = diarize_model(audio_path)
    words = whisperx.assign_word_speakers(diarize_segments, result["word_segments"])

    lines = []
    current_speaker = None
    current_line = ""
    for word in words:
        speaker = word.get("speaker", "Speaker")
        if speaker != current_speaker:
            if current_line:
                lines.append(f"[{current_speaker}] {current_line.strip()}")
                current_line = ""
            current_speaker = speaker
        current_line += word["text"] + " "
    if current_line:
        lines.append(f"[{current_speaker}] {current_line.strip()}")
    return "\n".join(lines)


@tool
def transcribe_audio_whisper(audio_path: str) -> str:
    """Transkribiert deutsche Sprache aus einer Audiodatei mit Whisper."""
    assert os.path.exists(audio_path), f"Datei nicht gefunden: {audio_path}"
    """ Larger Model performed better """
    """model = whisper.load_model("large-v3")"""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=None, temperature=0.3)
    return result["text"].strip()


class TranscriptionOnlyWorkflow(Runnable):
    """Workflow zur reinen Transkription von Audio mit Whisper."""

    def invoke(self, audio_path: str) -> str:
        text = transcribe_audio_whisper.invoke(audio_path)
        print("📄 Transkribierter Text:\n")
        print(text)
        return text


class WhisperXDiarizationWorkflow(Runnable):
    """Workflow für Transkription und Speaker-Diarization mit WhisperX."""

    def invoke(self, audio_path: str) -> str:
        text = transcribe_diarize_whisperx.invoke(audio_path)
        print("📄 Transkription mit Sprecherlabels:\n")
        print(text)
        return text


def main():
    parser = argparse.ArgumentParser(
        description="Transkription deutscher Audiodateien mit optionaler Diarization."
    )
    parser.add_argument("audio", help="Pfad zur Audiodatei (WAV/MP3)")
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Speaker-Diarization mit WhisperX verwenden",
    )
    args = parser.parse_args()

    if args.diarize:
        workflow = WhisperXDiarizationWorkflow()
    else:
        workflow = TranscriptionOnlyWorkflow()
    _ = workflow.invoke(args.audio)


if __name__ == "__main__":
    main()
