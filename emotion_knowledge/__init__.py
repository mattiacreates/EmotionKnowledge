import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain_core.runnables import Runnable
    from langchain_core.tools import tool
except Exception:  # pragma: no cover - optional dependency
    class Runnable:  # minimal fallback so tests can import _group_utterances
        def invoke(self, *args, **kwargs):  # pragma: no cover - not used in tests
            raise ImportError("langchain-core is required for this feature")

    def tool(func):  # pragma: no cover - noop decorator
        return func

try:  # whisper is only needed for the CLI workflow, not for tests
    import whisper  # type: ignore
except Exception:  # pragma: no cover - allow tests without whisper installed
    whisper = None

# Import SegmentSaver lazily to avoid requiring optional deps during tests
try:
    from .segment_saver import SegmentSaver
except Exception:  # pragma: no cover - optional dependency
    SegmentSaver = None

try:
    from .audio_emotion_annotator import AudioEmotionAnnotator, annotate_chromadb
except Exception:  # pragma: no cover - optional dependency
    AudioEmotionAnnotator = None
    annotate_chromadb = None

try:
    from .emotion_models import (
        EmotionModel,
        MultimodalEmotionModel,
        TextEmotionModel,
    )
except Exception:  # pragma: no cover - optional dependency
    EmotionModel = None
    MultimodalEmotionModel = None
    TextEmotionModel = None

try:
    from .text_emotion_annotator import TextEmotionAnnotator
except Exception:  # pragma: no cover - optional dependency
    TextEmotionAnnotator = None


def _group_utterances(segments, max_gap: float = 0.7):
    """Merge word-level segments into full utterances.

    Parameters
    ----------
    segments : list of dict
        The word-level diarization results from WhisperX.
    max_gap : float
        Maximum allowed pause between words (in seconds) for them to be
        merged into the same utterance.
    """

    if not segments:
        return []

    norm_segments = []
    fallback_dur = 0.1
    for i, seg in enumerate(segments):
        start = float(seg.get("start", seg.get("start_time", 0)))
        end_val = seg.get("end")
        if end_val is None or float(end_val) == 0.0:
            end_val = seg.get("end_time")
        if end_val is None or float(end_val) == 0.0:
            if i + 1 < len(segments):
                nxt = segments[i + 1]
                end_val = nxt.get("start", nxt.get("start_time", start))
            else:
                end_val = start + fallback_dur
        end = float(end_val)
        norm_segments.append(
            {
                "speaker": seg.get("speaker") or "speaker",
                "start": start,
                "end": end,
                "text": seg.get("text", seg.get("word", "")),
            }
        )

    # WhisperX already returns the word segments in chronological order.
    # Sorting here can lead to problems when some words have missing or
    # zero timestamps (e.g. due to alignment issues).  Such words would be
    # moved to the beginning and end up in the wrong utterance.  We therefore
    # keep the original order instead of sorting by ``start`` time.

    grouped = []
    current = norm_segments[0].copy()

    # interjections shorter than this duration or consisting of a single word
    # will be merged back into the surrounding utterance
    interjection_dur = 1.0
    i = 1
    while i < len(norm_segments):
        seg = norm_segments[i]
        gap = seg["start"] - current["end"]

        # merge same-speaker segments when the pause is short
        if seg["speaker"] == current["speaker"] and gap <= max_gap:
            current["text"] += " " + seg["text"]
            current["end"] = seg["end"]
            i += 1
            continue

        # short interjection from another speaker followed by the original speaker
        if (
            seg["speaker"] != current["speaker"]
            and (
                seg["end"] - seg["start"] <= interjection_dur
                or " " not in seg["text"].strip()
            )
            and i + 1 < len(norm_segments)
            and norm_segments[i + 1]["speaker"] == current["speaker"]
            and norm_segments[i + 1]["start"] - current["end"] <= interjection_dur
        ):
            current["text"] += " " + seg["text"]
            next_seg = norm_segments[i + 1]
            current["text"] += " " + next_seg["text"]
            current["end"] = next_seg["end"]
            i += 2
            continue

        grouped.append(current)
        current = seg.copy()
        i += 1

    grouped.append(current)

    # extend each utterance to start of the following one so the audio clip
    # fully contains the spoken words even if WhisperX produced short end
    # timestamps.  The final utterance keeps its original end time.
    for i in range(len(grouped) - 1):
        grouped[i]["end"] = grouped[i + 1]["start"]

    return grouped


@tool
def transcribe_diarize_whisperx(audio_path: str, model_size: str = "medium"):
    """Transkribiert Audio auf Deutsch mit WhisperX und Speaker-Diarization.

    ``model_size`` controls which WhisperX model checkpoint is loaded
    (e.g. ``base``, ``small``, ``medium``, ``large``).

    Returns a dict with ``text`` and ``segments`` keys so downstream code can
    further process the diarized segments.
    """
    import torch
    import whisperx

    assert os.path.exists(audio_path), f"Datei nicht gefunden: {audio_path}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """ Changed to int8, to see if it works on collab"""
    model = whisperx.load_model(model_size, device=device, language="de", compute_type="int8")
    result = model.transcribe(audio_path)

    align_model, metadata = whisperx.load_align_model(
        language_code="de", device=device
    )
    aligned_output = whisperx.align(
        result["segments"], align_model, metadata, audio_path, device=device
    )
    word_segments = aligned_output["word_segments"]
    logger.debug(type(word_segments))
    logger.debug(word_segments[:2])  # Now it's safe to preview

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
        speaker = word.get("speaker")
        if speaker is None or speaker == "Speaker":
            speaker = current_speaker or "Speaker"
        word["speaker"] = speaker

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
        logger.info("\ud83d\udcc4 Transkribierter Text:\n%s", text)
        return text


class WhisperXDiarizationWorkflow(Runnable):
    """Workflow für Transkription und Speaker-Diarization mit WhisperX.

    The workflow optionally stores each diarized segment to ChromaDB using
    :class:`SegmentSaver`. The WhisperX model size can be configured via the
    ``model_size`` argument.
    """

    def invoke(
        self,
        audio_path: str,
        db_path: str = "segment_db",
        clip_dir: str = "clips",
        model_size: str = "medium",
    ) -> str:
        result = transcribe_diarize_whisperx.invoke(audio_path, model_size=model_size)
        if isinstance(result, dict):
            text = result.get("text", "")
            segments = result.get("segments", [])
        else:
            text = str(result)
            segments = []

        logger.info("\ud83d\udcc4 Transkription mit Sprecherlabels:\n%s", text)

        if segments:
            # Log the first segment for easier debugging
            logger.debug("First diarized segment: %s", segments[0])
            if SegmentSaver is None:
                raise ImportError("SegmentSaver requires optional dependencies")
            saver = SegmentSaver(db_path=db_path, output_dir=clip_dir)
            utterances = _group_utterances(segments)
            for utt in utterances:
                utt["audio_path"] = audio_path
                saver.invoke(utt)

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
    parser.add_argument(
        "--whisperx-model",
        default="medium",
        help="WhisperX model size to use (base, small, medium, large)",
    )
    args = parser.parse_args()

    if args.diarize:
        workflow = WhisperXDiarizationWorkflow()
        _ = workflow.invoke(
            args.audio,
            db_path=args.db_path,
            clip_dir=args.clip_dir,
            model_size=args.whisperx_model,
        )
    else:
        workflow = TranscriptionOnlyWorkflow()
        _ = workflow.invoke(args.audio)


if __name__ == "__main__":
    main()
