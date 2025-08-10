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


def _group_utterances(
    segments,
    max_gap: float = 0.7,
    segments_info=None,
    merge_sentences: bool = False,
):
    """Merge word-level segments into full utterances.

    Parameters
    ----------
    segments : list of dict
        The word-level diarization results from the diarization step.
    max_gap : float
        Maximum allowed pause between words (in seconds) for them to be
        merged into the same utterance.
    merge_sentences : bool, optional
        When ``True`` merge consecutive utterances from the same speaker into a
        single entry. This is useful for sentence-level grouping.
    """

    if not segments:
        return []

    logger.info("Grouping %d word segments into utterances", len(segments))

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
                "segment": seg.get("segment"),
            }
        )

    use_segment_ids = False
    if segments_info is not None:
        use_segment_ids = True
    elif all(seg.get("segment") is not None for seg in norm_segments):
        use_segment_ids = True

    # If we have segment ids we simply merge by those first
    if use_segment_ids:
        ordered_groups = []
        if segments_info is not None:
            for idx, _info in enumerate(segments_info):
                words_for_seg = [w for w in norm_segments if w.get("segment") == idx]
                if words_for_seg:
                    ordered_groups.append(words_for_seg)
        else:
            current_id = norm_segments[0].get("segment")
            buf = []
            for w in norm_segments:
                if w.get("segment") != current_id:
                    if buf:
                        ordered_groups.append(buf)
                    buf = [w]
                    current_id = w.get("segment")
                else:
                    buf.append(w)
            if buf:
                ordered_groups.append(buf)

        grouped = []
        for group in ordered_groups:
            speaker_counts = {}
            for w in group:
                sp = w["speaker"]
                speaker_counts[sp] = speaker_counts.get(sp, 0) + 1
            majority_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
            grouped.append(
                {
                    "speaker": majority_speaker,
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "text": " ".join(w["text"] for w in group),
                }
            )

        if merge_sentences and grouped:
            merged = [grouped[0].copy()]
            for utt in grouped[1:]:
                if utt["speaker"] == merged[-1]["speaker"]:
                    merged[-1]["text"] += " " + utt["text"]
                    merged[-1]["end"] = utt["end"]
                else:
                    merged.append(utt.copy())
            grouped = merged

        for i in range(len(grouped) - 1):
            grouped[i]["end"] = grouped[i + 1]["start"]

        logger.info("Created %d utterances based on segment ids", len(grouped))
        for idx, utt in enumerate(grouped, 1):
            logger.debug(
                "Utterance %d: speaker=%s start=%.2f end=%.2f text=%s",
                idx,
                utt.get("speaker"),
                utt.get("start"),
                utt.get("end"),
                utt.get("text"),
            )
        return grouped

    # Word segments are expected to be in chronological order.
    # Sorting can lead to problems when some words have missing or
    # zero timestamps (e.g. due to alignment issues). Such words would be
    # moved to the beginning and end up in the wrong utterance. We therefore
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
    if merge_sentences and grouped:
        merged = [grouped[0].copy()]
        for utt in grouped[1:]:
            if utt["speaker"] == merged[-1]["speaker"]:
                merged[-1]["text"] += " " + utt["text"]
                merged[-1]["end"] = utt["end"]
            else:
                merged.append(utt.copy())
        grouped = merged

    # extend each utterance to start of the following one so the audio clip
    # fully contains the spoken words even if the diarization produced short end
    # timestamps.  The final utterance keeps its original end time.
    for i in range(len(grouped) - 1):
        grouped[i]["end"] = grouped[i + 1]["start"]

    logger.info("Created %d utterances", len(grouped))
    for idx, utt in enumerate(grouped, 1):
        logger.debug(
            "Utterance %d: speaker=%s start=%.2f end=%.2f text=%s",
            idx,
            utt.get("speaker"),
            utt.get("start"),
            utt.get("end"),
            utt.get("text"),
        )
    return grouped


@tool
def transcribe_diarize_sortformer(
    audio_path: str, model_name: str = "nvidia/diar_sortformer_4spk-v1"
):
    """Transkribiert Audio und führt Speaker-Diarization mit Sortformer durch.

    ``model_name`` kann eine Hugging-Face-Modell-ID oder der Pfad zu einer
    ``.nemo``-Datei sein.

    Returns a dict with ``text`` and ``segments`` keys so downstream code can
    further process the diarized segments.
    """
    assert os.path.exists(audio_path), f"Datei nicht gefunden: {audio_path}"
    logger.info("Loading Sortformer diarization model '%s'", model_name)

    try:
        from nemo.collections.asr.models import SortformerEncLabelModel

        diar_model = SortformerEncLabelModel.from_pretrained(
            model_name, map_location="cpu", strict=False
        )
        diar_model.eval()
        predicted_segments = diar_model.diarize(audio=audio_path, batch_size=1)
    except Exception:  # pragma: no cover - optional dependency
        logger.warning(
            "Sortformer model not available; returning empty segments"
        )
        predicted_segments = []

    text = transcribe_audio_whisper.invoke(audio_path)
    return {"text": text, "segments": predicted_segments}


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
        logger.info("Transcribing %s", audio_path)
        text = transcribe_audio_whisper.invoke(audio_path)
        logger.info("\ud83d\udcc4 Transkribierter Text:\n%s", text)
        return text


class SortformerDiarizationWorkflow(Runnable):
    """Workflow für Transkription und Speaker-Diarization mit Sortformer.

    The workflow optionally stores each diarized segment to ChromaDB using
    :class:`SegmentSaver`. The Sortformer model can be configured via the
    ``model_name`` argument.
    """

    def invoke(
        self,
        audio_path: str,
        db_path: str = "segment_db",
        clip_dir: str = "clips",
        model_name: str = "nvidia/diar_sortformer_4spk-v1",
    ) -> str:
        logger.info("Transcribing and diarizing %s", audio_path)
        result = transcribe_diarize_sortformer.invoke(
            {"audio_path": audio_path, "model_name": model_name}
        )
        if isinstance(result, dict):
            text = result.get("text", "")
            segments = result.get("segments", [])
        else:
            text = str(result)
            segments = []

        logger.info("\ud83d\udcc4 Transkription mit Sprecherlabels:\n%s", text)

        if segments and isinstance(segments[0], dict) and "text" in segments[0]:
            logger.debug("First diarized segment: %s", segments[0])
            if SegmentSaver is None:
                raise ImportError("SegmentSaver requires optional dependencies")
            saver = SegmentSaver(db_path=db_path, output_dir=clip_dir)
            logger.info("Grouping diarized words into utterances")
            utterances = _group_utterances(
                segments,
                segments_info=result.get("segments_info"),
                merge_sentences=True,
            )
            logger.info("Saving %d utterances to %s", len(utterances), clip_dir)
            for idx, utt in enumerate(utterances, 1):
                logger.info(
                    "Saving utterance %d/%d speaker=%s", idx, len(utterances), utt.get("speaker")
                )
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
        help="Speaker-Diarization mit Sortformer verwenden",
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
        "--sortformer-model",
        default="nvidia/diar_sortformer_4spk-v1",
        help="Sortformer diarization model name or path",
    )
    args = parser.parse_args()

    if args.diarize:
        workflow = SortformerDiarizationWorkflow()
        _ = workflow.invoke(
            args.audio,
            db_path=args.db_path,
            clip_dir=args.clip_dir,
            model_name=args.sortformer_model,
        )
    else:
        workflow = TranscriptionOnlyWorkflow()
        _ = workflow.invoke(args.audio)


if __name__ == "__main__":
    main()
