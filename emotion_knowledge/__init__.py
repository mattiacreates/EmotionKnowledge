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
    absorb_interjections: bool = False,
    backchannel_max_dur: float = 0.7,
    backchannel_max_words: int = 3,
    tag_backchannels: bool = True,
    preserve_end_times: bool = True,
):
    """Merge word-level segments into full utterances.

    Parameters
    ----------
    segments : list of dict
        The word-level diarization results from WhisperX.
    max_gap : float
        Maximum allowed pause between words (in seconds) for them to be
        merged into the same utterance.
    merge_sentences : bool, optional
        When ``True`` merge consecutive utterances from the same speaker into a
        single entry. This is useful for sentence-level grouping.
    absorb_interjections : bool, optional
        When ``True`` short interjections from other speakers are merged back
        into the surrounding utterance.  When ``False`` (default) they remain
        separate utterances.
    backchannel_max_dur : float, optional
        Maximum duration (in seconds) for an utterance to qualify as a
        backchannel.
    backchannel_max_words : int, optional
        Maximum number of words for an utterance to qualify as a backchannel.
    tag_backchannels : bool, optional
        When ``True`` add ``is_backchannel=True`` to detected backchannels.
    preserve_end_times : bool, optional
        Deprecated. End timestamps are always taken from the final word in each
        utterance. This parameter is retained for backwards compatibility and
        has no effect.
    """

    if not segments:
        return []

    logger.info("Grouping %d word segments into utterances", len(segments))

    def _is_backchannel(seg):
        dur = seg["end"] - seg["start"]
        wc = len(seg["text"].strip().split())
        return dur <= backchannel_max_dur or wc <= backchannel_max_words

    def _tag_backchannel(utt):
        if tag_backchannels and _is_backchannel(utt):
            utt["is_backchannel"] = True

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
            first_word = group[0]
            last_word = group[-1]
            utt = {
                "speaker": majority_speaker,
                "start": first_word["start"],
                "end": last_word["end"],
                "text": " ".join(w["text"] for w in group),
            }
            _tag_backchannel(utt)
            grouped.append(utt)

        if merge_sentences and grouped:
            merged = [grouped[0].copy()]
            for utt in grouped[1:]:
                if (
                    utt["speaker"] == merged[-1]["speaker"]
                    and not merged[-1].get("is_backchannel")
                    and not utt.get("is_backchannel")
                ):
                    merged[-1]["text"] += " " + utt["text"]
                    merged[-1]["end"] = utt["end"]
                else:
                    merged.append(utt.copy())
            grouped = merged

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

    # WhisperX already returns the word segments in chronological order.
    # Sorting here can lead to problems when some words have missing or
    # zero timestamps (e.g. due to alignment issues).  Such words would be
    # moved to the beginning and end up in the wrong utterance.  We therefore
    # keep the original order instead of sorting by ``start`` time.

    grouped = []
    current = norm_segments[0].copy()
    prev_speaker = None
    suppress_tag = False

    def append_current(utt):
        nonlocal prev_speaker, suppress_tag
        if (
            tag_backchannels
            and not suppress_tag
            and prev_speaker is not None
            and utt["speaker"] != prev_speaker
            and _is_backchannel(utt)
        ):
            utt["is_backchannel"] = True
        grouped.append(utt)
        prev_speaker = utt["speaker"]
        suppress_tag = False

    # interjections shorter than this duration or consisting of a single word
    # are candidates to be absorbed
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
            if absorb_interjections:
                current["text"] += " " + seg["text"]
                next_seg = norm_segments[i + 1]
                current["text"] += " " + next_seg["text"]
                current["end"] = next_seg["end"]
                i += 2
                continue
            else:
                append_current(current)
                interj = seg.copy()
                if tag_backchannels and _is_backchannel(interj):
                    interj["is_backchannel"] = True
                append_current(interj)
                current = norm_segments[i + 1].copy()
                suppress_tag = True
                i += 2
                continue

        append_current(current)
        current = seg.copy()
        i += 1

    append_current(current)
    if merge_sentences and grouped:
        merged = [grouped[0].copy()]
        for utt in grouped[1:]:
            if (
                utt["speaker"] == merged[-1]["speaker"]
                and not merged[-1].get("is_backchannel")
                and not utt.get("is_backchannel")
            ):
                merged[-1]["text"] += " " + utt["text"]
                merged[-1]["end"] = utt["end"]
            else:
                merged.append(utt.copy())
        grouped = merged

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


def export_word_level_excel(
    words,
    path: str = "Single_Word_Transcript.xlsx",
    backchannel_max_dur: float = 0.7,
    backchannel_max_words: int = 3,
):
    """Export word-level segments to Excel with a CSV fallback.

    Parameters
    ----------
    words : list of dict
        Word-level diarization results.
    path : str, optional
        Output path for the Excel file.
    backchannel_max_dur : float, optional
        Duration threshold for backchannel detection in seconds.
    backchannel_max_words : int, optional
        Word-count threshold for backchannel detection.
    """

    if not words:
        return None

    import pandas as pd

    rows = []
    for idx, w in enumerate(words, 1):
        start = float(w.get("start", w.get("start_time", 0)) or 0)
        end_val = w.get("end")
        if end_val is None or float(end_val) == 0.0:
            end_val = w.get("end_time")
        if end_val is None or float(end_val) == 0.0:
            end_val = start
        end = float(end_val)
        duration = max(0.0, end - start)
        text = w.get("text", w.get("word", ""))
        word_count = len(str(text).strip().split())
        is_backchannel = (
            duration <= backchannel_max_dur or word_count <= backchannel_max_words
        )
        rows.append(
            {
                "idx": idx,
                "segment": w.get("segment"),
                "speaker": w.get("speaker"),
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
                "is_backchannel": is_backchannel,
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(["start", "idx"], inplace=True)
    try:
        df.to_excel(path, index=False)
        logger.info("Exported word-level transcript to %s", path)
        return path
    except ImportError:
        csv_path = os.path.splitext(path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        logger.info("openpyxl not installed, exported CSV to %s", csv_path)
        return csv_path


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
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info("Using compute_type '%s' on device '%s'", compute_type, device)

    logger.info("Starting WhisperX transcription using model '%s'", model_size)
    model = whisperx.load_model(
        model_size, device=device, language="de", compute_type=compute_type
    )
    transcribe_kwargs = {
        "vad_filter": False,
        "condition_on_previous_text": False,
        "no_speech_threshold": None,
        "log_prob_threshold": None,
        "compression_ratio_threshold": None,
    }
    logger.info("Transcribing with vad_filter=%s", transcribe_kwargs["vad_filter"])
    try:
        result = model.transcribe(audio_path, **transcribe_kwargs)
    except TypeError as e:  # pragma: no cover - depends on whisperx version
        logger.warning("Falling back to minimal transcribe call: %s", e)
        result = model.transcribe(audio_path, vad_filter=False)
    logger.info("Transcription complete with %d segments", len(result.get("segments", [])))

    align_model, metadata = whisperx.load_align_model(
        language_code="de", device=device
    )
    aligned_output = whisperx.align(
        result["segments"], align_model, metadata, audio_path, device=device
    )
    logger.info(
        "Alignment complete, %d word segments", len(aligned_output.get("word_segments", []))
    )
    word_segments = aligned_output["word_segments"]
    logger.debug(type(word_segments))
    logger.debug(word_segments[:2])  # Now it's safe to preview

    aligned_segments = aligned_output.get("segments", [])

    token = os.getenv("HF_TOKEN")  # set this in Colab/terminal
    diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token=token)
    diarize_segments = diarize_model(audio_path)
    logger.info("Diarization complete with %d segments", len(diarize_segments))

    result_with_speakers = whisperx.assign_word_speakers(
        diarize_segments, aligned_output
    )

    logger.info(
        "Assigned speaker labels to %d words", len(result_with_speakers.get("word_segments", []))
    )

    words = result_with_speakers["word_segments"]

    # attach segment index to each word
    seg_idx = 0
    if aligned_segments:
        seg_starts = [float(s.get("start", s.get("start_time", 0))) for s in aligned_segments]
        seg_starts.append(float("inf"))
        for w in words:
            start_val = float(w.get("start", w.get("start_time", 0)))
            while seg_idx + 1 < len(seg_starts) and start_val >= seg_starts[seg_idx + 1]:
                seg_idx += 1
            w["segment"] = seg_idx

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
    logger.info("Diarization produced %d words", len(words))
    return {"text": text, "segments": words, "segments_info": aligned_segments}


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
        preserve_backchannels: bool = True,
        preserve_end_times: bool = True,
        export_words_xlsx: bool = False,
    ) -> str:
        logger.info("Transcribing and diarizing %s", audio_path)
        result = transcribe_diarize_whisperx.invoke(
            {"audio_path": audio_path, "model_size": model_size}
        )
        if isinstance(result, dict):
            text = result.get("text", "")
            segments = result.get("segments", [])
        else:
            text = str(result)
            segments = []

        logger.info("\ud83d\udcc4 Transkription mit Sprecherlabels:\n%s", text)

        if segments:
            logger.debug("First diarized segment: %s", segments[0])
            if export_words_xlsx:
                export_word_level_excel(segments, "Single_Word_Transcript.xlsx")
            if SegmentSaver is None:
                raise ImportError("SegmentSaver requires optional dependencies")
            saver = SegmentSaver(db_path=db_path, output_dir=clip_dir)
            logger.info("Grouping diarized words into utterances")
            utterances = _group_utterances(
                segments,
                segments_info=result.get("segments_info"),
                merge_sentences=True,
                absorb_interjections=not preserve_backchannels,
                tag_backchannels=True,
                backchannel_max_dur=0.7,
                backchannel_max_words=3,
                preserve_end_times=preserve_end_times,
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
    parser.add_argument("--preserve-backchannels", action="store_true", default=True)
    parser.add_argument(
        "--no-preserve-backchannels", dest="preserve_backchannels", action="store_false"
    )
    parser.add_argument("--preserve-end-times", action="store_true", default=True)
    parser.add_argument(
        "--export-words-xlsx",
        action="store_true",
        help="Export word-level transcript to Excel (CSV fallback)",
    )
    args = parser.parse_args()

    if args.diarize:
        workflow = WhisperXDiarizationWorkflow()
        _ = workflow.invoke(
            args.audio,
            db_path=args.db_path,
            clip_dir=args.clip_dir,
            model_size=args.whisperx_model,
            preserve_backchannels=args.preserve_backchannels,
            preserve_end_times=args.preserve_end_times,
            export_words_xlsx=args.export_words_xlsx,
        )
    else:
        workflow = TranscriptionOnlyWorkflow()
        _ = workflow.invoke(args.audio)


if __name__ == "__main__":
    main()
