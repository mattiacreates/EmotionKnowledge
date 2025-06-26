import argparse
import os
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
import whisper

from .pipeline import emotion_transcription_pipeline



@tool
def transcribe_diarize_whisperx(audio_path: str) -> str:
    """Transkribiert Audio auf Deutsch mit WhisperX und Speaker-Diarization."""
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

    token = os.getenv("HF_TOKEN")  # set this in Colab/terminal
    diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token=token)
    diarize_segments = diarize_model(audio_path)

    result_with_speakers = whisperx.assign_word_speakers(
        diarize_segments, aligned_output
    )

    words = result_with_speakers["word_segments"]
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
        current_line += word_text + " "
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


def main() -> None:
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
        "--model-size",
        default="small",
        help="WhisperX ASR Modellgröße",
    )
    args = parser.parse_args()

    pipeline = emotion_transcription_pipeline(
        diarize=args.diarize, asr_model_size=args.model_size
    )
    transcript = pipeline.invoke(args.audio)
    print(transcript)


if __name__ == "__main__":
    main()
