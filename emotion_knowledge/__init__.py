import argparse
import os
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
import whisper

from .segment_saver import SegmentSaver


@tool
def transcribe_diarize_whisperx(audio_path: str):
    """Transkribiert Audio auf Deutsch mit WhisperX und Speaker-Diarization.

    Returns a dict with ``text`` and ``segments`` keys so downstream code can
    further process the diarized segments.
    """
    import torch
    import whisperx

    assert os.path.exists(audio_path), f"Datei nicht gefunden: {audio_path}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """ Changed to int8, to see if it works on collab"""
    model = whisperx.load_model("medium", device=device, language="de", compute_type="int8")
    result = model.transcribe(audio_path)

    align_model, metadata = whisperx.load_align_model(
        language_code="de", device=device
    )
    aligned_output = whisperx.align(
        result["segments"], align_model, metadata, audio_path, device=device
    )
    word_segments = aligned_output["word_segments"]
    print(type(word_segments))
    print(word_segments[:2])  # Now it's safe to preview

    token = os.getenv("HF_TOKEN")  # set this in Colab/terminal
    diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token=token)
    diarize_segments = diarize_model(audio_path)

    result_with_speakers = whisperx.assign_word_speakers(
        diarize_segments, aligned_output
    )

    words = result_with_speakers["word_segments"]

    # Build a formatted string while keeping the raw segment information so it
    # can be persisted elsewhere.
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
        word_text = word.get("text", word.get("word", ""))
        # ensure SegmentSaver can access the spoken text
        word["text"] = word_text
        current_line += word_text + " "
    if current_line:
        lines.append(f"[{current_speaker}] {current_line.strip()}")

    text = "\n".join(lines)
    return {"text": text, "segments": words}


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
    """Workflow für Transkription und Speaker-Diarization mit WhisperX.

    The workflow optionally stores each diarized segment to ChromaDB using
    :class:`SegmentSaver`.
    """

    def invoke(self, audio_path: str, db_path: str = "segment_db", clip_dir: str = "clips") -> str:
        result = transcribe_diarize_whisperx.invoke(audio_path)
        if isinstance(result, dict):
            text = result.get("text", "")
            segments = result.get("segments", [])
        else:
            text = str(result)
            segments = []

        print("📄 Transkription mit Sprecherlabels:\n")
        print(text)

        if segments:
            saver = SegmentSaver(db_path=db_path, output_dir=clip_dir)
            for segment in segments:
                segment["audio_path"] = audio_path
                saver.invoke(segment)

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
    parser.add_argument(
        "--db-path",
        default="segment_db",
        help="Pfad zum ChromaDB-Verzeichnis",
    )
    parser.add_argument(
        "--clip-dir",
        default="clips",
        help="Verzeichnis zum Speichern der Audio-Schnipsel",
    )
    args = parser.parse_args()

    if args.diarize:
        workflow = WhisperXDiarizationWorkflow()
        _ = workflow.invoke(args.audio, db_path=args.db_path, clip_dir=args.clip_dir)
    else:
        workflow = TranscriptionOnlyWorkflow()
        _ = workflow.invoke(args.audio)


if __name__ == "__main__":
    main()
