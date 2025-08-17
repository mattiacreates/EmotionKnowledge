import argparse
import os
import logging
from functools import lru_cache

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
    multi_spk_seg_min_words: int | None = None,
    backchannel_run_min_words: int | None = None,
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
    multi_spk_seg_min_words : int | None, optional
        Minimum total words in a multi-speaker segment required to apply the
        run-based utterance/backchannel logic. If ``None`` the standard
        grouping logic is used.
    backchannel_run_min_words : int | None, optional
        Minimum consecutive words spoken by the same speaker (a run) for that
        run to be tagged with ``is_backchannel=True``. Only used when
        ``multi_spk_seg_min_words`` is also provided.

    If both ``multi_spk_seg_min_words`` and ``backchannel_run_min_words`` are
    set (not ``None``), use run-based utterance creation and backchannel tagging
    within multi-speaker segments. Otherwise, use the standard grouping logic.
    Old duration/word-count backchannel logic is removed.
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

    use_new_run_logic = (
        multi_spk_seg_min_words is not None and backchannel_run_min_words is not None
    )

    grouped = []
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

        for group in ordered_groups:
            if use_new_run_logic:
                speakers_in_group = {w["speaker"] for w in group}
                if len(group) >= multi_spk_seg_min_words and len(speakers_in_group) >= 2:
                    runs = []
                    run_speaker = group[0]["speaker"]
                    run_words = [group[0]]
                    for w in group[1:]:
                        if w["speaker"] == run_speaker:
                            run_words.append(w)
                        else:
                            runs.append((run_speaker, run_words))
                            run_speaker = w["speaker"]
                            run_words = [w]
                    runs.append((run_speaker, run_words))

                    eligible_speakers = {
                        spk
                        for spk, run in runs
                        if len(run) >= backchannel_run_min_words
                    }
                    if len(eligible_speakers) >= 2:
                        for spk, words_run in runs:
                            utt = {
                                "speaker": spk,
                                "start": words_run[0]["start"],
                                "end": words_run[-1]["end"],
                                "text": " ".join(w["text"] for w in words_run),
                                "words": words_run,
                            }
                            if len(words_run) >= backchannel_run_min_words:
                                utt["is_backchannel"] = True
                            grouped.append(utt)
                        continue

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
                "words": group,
            }
            grouped.append(utt)

        if merge_sentences and grouped:
            merged = [grouped[0].copy()]
            merged[0]["words"] = grouped[0]["words"].copy()
            for utt in grouped[1:]:
                can_merge = (
                    utt["speaker"] == merged[-1]["speaker"]
                    and not merged[-1].get("is_backchannel")
                    and not utt.get("is_backchannel")
                )
                if can_merge:
                    merged[-1]["text"] += " " + utt["text"]
                    merged[-1]["end"] = utt["end"]
                    merged[-1]["words"].extend(utt.get("words", []))
                else:
                    new_utt = utt.copy()
                    new_utt["words"] = utt.get("words", []).copy()
                    merged.append(new_utt)
            grouped = merged
    else:
        # WhisperX already returns the word segments in chronological order.
        # Sorting here can lead to problems when some words have missing or
        # zero timestamps (e.g. due to alignment issues).  Such words would be
        # moved to the beginning and end up in the wrong utterance.  We therefore
        # keep the original order instead of sorting by ``start`` time.

        current = norm_segments[0].copy()
        current_words = [norm_segments[0].copy()]
        prev_speaker = None

        def append_current(utt, words):
            """Finalize the current utterance and store its word list."""
            nonlocal prev_speaker
            utt["words"] = words
            grouped.append(utt)
            prev_speaker = utt["speaker"]

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
                current_words.append(seg)
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
                    current_words.append(seg)
                    next_seg = norm_segments[i + 1]
                    current["text"] += " " + next_seg["text"]
                    current["end"] = next_seg["end"]
                    current_words.append(next_seg)
                    i += 2
                    continue
                else:
                    append_current(current, current_words)
                    interj = seg.copy()
                    append_current(interj, [seg])
                    current = norm_segments[i + 1].copy()
                    current_words = [norm_segments[i + 1].copy()]
                    i += 2
                    continue

            append_current(current, current_words)
            current = seg.copy()
            current_words = [seg.copy()]
            i += 1
        append_current(current, current_words)
        if merge_sentences and grouped:
            merged = [grouped[0].copy()]
            merged[0]["words"] = grouped[0]["words"].copy()
            for utt in grouped[1:]:
                can_merge = utt["speaker"] == merged[-1]["speaker"]
                if can_merge:
                    merged[-1]["text"] += " " + utt["text"]
                    merged[-1]["end"] = utt["end"]
                    merged[-1]["words"].extend(utt.get("words", []))
                else:
                    new_utt = utt.copy()
                    new_utt["words"] = utt.get("words", []).copy()
                    merged.append(new_utt)
            grouped = merged

    # Compute statistics for each utterance
    for utt in grouped:
        words = utt.get("words", [])
        utt["n_words"] = len(words)
        duration = utt["end"] - utt["start"]
        utt["duration"] = duration
        utt["words_per_sec"] = (len(words) / duration) if duration > 0 else 0.0
        gaps = [
            words[i]["start"] - words[i - 1]["end"]
            for i in range(1, len(words))
        ]
        if gaps:
            utt["mean_word_gap"] = sum(gaps) / len(gaps)
            sorted_gaps = sorted(gaps)
            k = int(round(0.95 * (len(sorted_gaps) - 1)))
            utt["p95_word_gap"] = sorted_gaps[k]
        else:
            utt["mean_word_gap"] = 0.0
            utt["p95_word_gap"] = 0.0

    # Determine whether each utterance starts during another speaker's speech
    prev_end = None
    prev_speaker = None
    for utt in grouped:
        if (
            prev_end is not None
            and utt["start"] < prev_end
            and utt["speaker"] != prev_speaker
        ):
            utt["overlaps_started"] = True
        else:
            utt["overlaps_started"] = False
        if prev_end is None or utt["end"] > prev_end:
            prev_end = utt["end"]
            prev_speaker = utt["speaker"]

    for utt in grouped:
        utt.pop("words", None)

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


def _split_multispeaker_segments(words):
    """Split long segments containing multiple speakers into subsegments.

    Existing short segments (runs of fewer than three words) remain unchanged.
    Within longer segments a new subsegment is started whenever a speaker has
    a run of three or more consecutive words. Interjections of one or two
    words are attached to the surrounding segment. Segment identifiers are
    reassigned sequentially and the original word order is preserved.

    Parameters
    ----------
    words : list of dict
        Word-level items with keys ``segment``, ``speaker``, ``start``, ``end``
        and either ``text`` or ``word``.

    Returns
    -------
    list of dict
        Recomputed segments with updated ``segment`` ids.
    """

    if not words:
        return []

    def _get_text(w):
        return w.get("text", w.get("word", ""))

    # Group by existing segment id if present
    groups = []
    if all("segment" in w for w in words):
        current_id = words[0]["segment"]
        buf = []
        for w in words:
            seg_id = w["segment"]
            if seg_id != current_id:
                if buf:
                    groups.append(buf)
                buf = [w]
                current_id = seg_id
            else:
                buf.append(w)
        if buf:
            groups.append(buf)
    else:
        groups = [words]

    new_segments = []
    new_id = 1

    for group in groups:
        if not group:
            continue
        unique_speakers = {w.get("speaker") for w in group}
        if len(group) < 3 or len(unique_speakers) == 1:
            start = group[0]["start"]
            end = group[-1]["end"]
            text = " ".join(_get_text(w) for w in group)
            speaker = group[0].get("speaker")
            for w in group:
                w["segment"] = new_id
            new_segments.append(
                {
                    "segment": new_id,
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )
            new_id += 1
            continue

        # Build runs of consecutive words by speaker
        runs = []
        run_speaker = group[0]["speaker"]
        run_words = [group[0]]
        for w in group[1:]:
            if w["speaker"] == run_speaker:
                run_words.append(w)
            else:
                runs.append([run_speaker, run_words])
                run_speaker = w["speaker"]
                run_words = [w]
        runs.append([run_speaker, run_words])

        # Merge leading short runs into the first subsequent run with >=3 words
        while (
            len(runs) > 1
            and len(runs[0][1]) < 3
            and any(len(r[1]) >= 3 for r in runs[1:])
        ):
            runs[1][1] = runs[0][1] + runs[1][1]
            runs.pop(0)

        current_speaker, current_words = runs[0]
        for spk, words_run in runs[1:]:
            if spk != current_speaker and len(words_run) >= 3:
                start = current_words[0]["start"]
                end = current_words[-1]["end"]
                text = " ".join(_get_text(w) for w in current_words)
                for w in current_words:
                    w["segment"] = new_id
                new_segments.append(
                    {
                        "segment": new_id,
                        "speaker": current_speaker,
                        "start": start,
                        "end": end,
                        "text": text,
                    }
                )
                new_id += 1
                current_speaker = spk
                current_words = words_run
            else:
                current_words.extend(words_run)

        start = current_words[0]["start"]
        end = current_words[-1]["end"]
        text = " ".join(_get_text(w) for w in current_words)
        for w in current_words:
            w["segment"] = new_id
        new_segments.append(
            {
                "segment": new_id,
                "speaker": current_speaker,
                "start": start,
                "end": end,
                "text": text,
            }
        )
        new_id += 1

    return new_segments


def export_word_level_excel(
    words,
    path: str = "Single_Word_Transcript.xlsx",
    multi_spk_seg_min_words: int | None = None,
    backchannel_run_min_words: int | None = None,
):
    """Export word-level segments to Excel with a CSV fallback.

    Parameters
    ----------
    words : list of dict
        Word-level diarization results.
    path : str, optional
        Output path for the Excel file.
    multi_spk_seg_min_words : int | None, optional
        Minimum total words in a segment to enable run-based
        utterance/backchannel logic.
    backchannel_run_min_words : int | None, optional
        Minimum consecutive words by the same speaker (run) to tag that run as a
        backchannel. Only used when ``multi_spk_seg_min_words`` is also set.
    """

    if not words:
        return None

    import pandas as pd

    utterances = _group_utterances(
        words,
        multi_spk_seg_min_words=multi_spk_seg_min_words,
        backchannel_run_min_words=backchannel_run_min_words,
    )
    utterances = sorted(utterances, key=lambda u: u.get("start", 0))

    rows = []
    utt_idx = 0
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
        while (
            utt_idx + 1 < len(utterances)
            and start >= utterances[utt_idx]["end"]
        ):
            utt_idx += 1
        is_backchannel = bool(utterances[utt_idx].get("is_backchannel", False))
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
    logger.info(
        "Starting WhisperX transcription using model '%s' with compute type '%s' on %s",
        model_size,
        compute_type,
        device,
    )
    model = whisperx.load_model(
        model_size, device=device, language="de", compute_type=compute_type
    )
    result = model.transcribe(audio_path)
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


@lru_cache(maxsize=1)
def _load_hf_whisper():
    """Load the Hugging Face Whisper pipeline lazily and cache it."""
    from transformers import pipeline

    model_id = "openai/whisper-large-v3"
    return pipeline("automatic-speech-recognition", model=model_id)


@tool
def transcribe_audio_whisper(audio_path: str) -> str:
    """Transkribiert deutsche Sprache aus einer Audiodatei mit Whisper.

    Uses the open-source ``openai/whisper-large-v3`` model hosted on
    Hugging Face.
    """
    assert os.path.exists(audio_path), f"Datei nicht gefunden: {audio_path}"

    pipe = _load_hf_whisper()
    result = pipe(audio_path, generate_kwargs={"temperature": 0.3})
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
        multi_spk_seg_min_words: int | None = None,
        backchannel_run_min_words: int | None = None,
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
                export_word_level_excel(
                    segments,
                    "Single_Word_Transcript.xlsx",
                    multi_spk_seg_min_words=multi_spk_seg_min_words,
                    backchannel_run_min_words=backchannel_run_min_words,
                )
            if SegmentSaver is None:
                raise ImportError("SegmentSaver requires optional dependencies")
            saver = SegmentSaver(db_path=db_path, output_dir=clip_dir)
            logger.info("Grouping diarized words into utterances")
            utterances = _group_utterances(
                segments,
                segments_info=result.get("segments_info"),
                merge_sentences=True,
                absorb_interjections=not preserve_backchannels,
                multi_spk_seg_min_words=multi_spk_seg_min_words,
                backchannel_run_min_words=backchannel_run_min_words,
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
    parser.add_argument(
        "--multi-spk-seg-min-words",
        type=int,
        default=None,
        help="Min total words in a segment to enable run-based utterance/backchannel logic",
    )
    parser.add_argument(
        "--backchannel-run-min-words",
        type=int,
        default=None,
        help="Min consecutive words by same speaker (run) to tag that run as backchannel",
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
            multi_spk_seg_min_words=args.multi_spk_seg_min_words,
            backchannel_run_min_words=args.backchannel_run_min_words,
        )
    else:
        workflow = TranscriptionOnlyWorkflow()
        _ = workflow.invoke(args.audio)


if __name__ == "__main__":
    main()
