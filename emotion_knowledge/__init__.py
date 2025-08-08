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
    keep_interjections: bool = True,
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
    keep_interjections : bool, optional
        When ``True`` (default) every speaker change yields a new utterance so
        short interjections from another speaker are preserved verbatim.
    """

    if not segments:
        return []

    logger.info("Grouping %d word segments into utterances", len(segments))

    norm_segments = []
    fallback_dur = 0.25
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

    # If we have segment ids we first group by segment, then split by speaker
    # changes or large gaps to avoid collapsing interjections.
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
        for seg_words in ordered_groups:
            subgroups = []
            cur = [seg_words[0]]
            for w in seg_words[1:]:
                gap = w["start"] - cur[-1]["end"]
                if w["speaker"] != cur[-1]["speaker"] or gap > max_gap:
                    subgroups.append(cur)
                    cur = [w]
                else:
                    cur.append(w)
            subgroups.append(cur)

            for g in subgroups:
                speakers = {w["speaker"] for w in g}
                if len(speakers) == 1:
                    sp = g[0]["speaker"]
                else:
                    durs = {}
                    for w in g:
                        durs[w["speaker"]] = durs.get(w["speaker"], 0.0) + max(
                            0.0, float(w["end"]) - float(w["start"])
                        )
                    sp = max(durs.items(), key=lambda x: x[1])[0]
                grouped.append(
                    {
                        "speaker": sp,
                        "start": g[0]["start"],
                        "end": g[-1]["end"],
                        "text": " ".join(w["text"] for w in g),
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
            grouped[i]["end"] = max(grouped[i]["end"], grouped[i + 1]["start"])

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

    # interjections shorter than this duration or consisting of a single word
    # may optionally be merged back into the surrounding utterance
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

        if not keep_interjections and (
            seg["speaker"] != current["speaker"]
            and (
                seg["end"] - seg["start"] <= interjection_dur
                or len(seg["text"].strip().split()) <= 1
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

        if keep_interjections and seg["speaker"] != current["speaker"]:
            logger.debug(
                "Speaker change from %s to %s at %.2f",
                current["speaker"],
                seg["speaker"],
                seg["start"],
            )

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
    # fully contains the spoken words even if WhisperX produced short end
    # timestamps.  The final utterance keeps its original end time.
    for i in range(len(grouped) - 1):
        grouped[i]["end"] = max(grouped[i]["end"], grouped[i + 1]["start"])

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
def transcribe_diarize_whisperx(
    audio_path: str,
    model_size: str = "medium",
    language: str = "de",
    compute_type: str = "int8",
    beam_size: int = 5,
    temperature: float = 0.0,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    return_embeddings: bool = False,
):
    """Transkribiert Audio auf Deutsch mit WhisperX und Speaker-Diarization.

    ``model_size`` controls which WhisperX model checkpoint is loaded
    (e.g. ``base``, ``small``, ``medium``, ``large``).  Additional parameters
    allow configuring language, compute precision and decoding behaviour.

    Returns a dict with ``text`` and ``segments`` keys so downstream code can
    further process the diarized segments.
    """
    import torch
    import whisperx
    import inspect

    assert os.path.exists(audio_path), f"Datei nicht gefunden: {audio_path}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Starting WhisperX transcription using model '%s'", model_size)
    model = whisperx.load_model(
        model_size,
        device=device,
        language=language,
        compute_type=compute_type,
    )
    transcribe_params = inspect.signature(model.transcribe).parameters
    kwargs = {
        name: val
        for name, val in {
            "beam_size": beam_size,
            "temperature": temperature,
        }.items()
        if name in transcribe_params
    }
    result = model.transcribe(audio_path, **kwargs)
    logger.info("Transcription complete with %d segments", len(result.get("segments", [])))

    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=device
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
    diarize_segments = diarize_model(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_embeddings=return_embeddings,
    )
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
        keep_interjections: bool = True,
        max_gap: float = 0.7,
        language: str = "de",
        compute_type: str = "int8",
        beam_size: int = 5,
        temperature: float = 0.0,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        return_embeddings: bool = False,
    ) -> str:
        logger.info("Transcribing and diarizing %s", audio_path)
        result = transcribe_diarize_whisperx.invoke(
            {
                "audio_path": audio_path,
                "model_size": model_size,
                "language": language,
                "compute_type": compute_type,
                "beam_size": beam_size,
                "temperature": temperature,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "return_embeddings": return_embeddings,
            }
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
            if SegmentSaver is None:
                raise ImportError("SegmentSaver requires optional dependencies")
            saver = SegmentSaver(db_path=db_path, output_dir=clip_dir)
            logger.info("Grouping diarized words into utterances")
            utterances = _group_utterances(
                segments,
                max_gap=max_gap,
                segments_info=result.get("segments_info"),
                merge_sentences=True,
                keep_interjections=keep_interjections,
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
    parser.add_argument(
        "--keep-interruptions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve short interruptions as separate utterances (default on)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.7,
        help="Maximum allowed pause between words for same utterance",
    )
    parser.add_argument(
        "--language",
        default="de",
        help="Language code for WhisperX model",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Precision for WhisperX model (e.g. float16, int8)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search width",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers for diarization",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers for diarization",
    )
    parser.add_argument(
        "--return-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return speaker embeddings from diarization pipeline",
    )
    args = parser.parse_args()

    if args.diarize:
        workflow = WhisperXDiarizationWorkflow()
        _ = workflow.invoke(
            args.audio,
            db_path=args.db_path,
            clip_dir=args.clip_dir,
            model_size=args.whisperx_model,
            keep_interjections=args.keep_interruptions,
            max_gap=args.max_gap,
            language=args.language,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            temperature=args.temperature,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            return_embeddings=args.return_embeddings,
        )
    else:
        workflow = TranscriptionOnlyWorkflow()
        _ = workflow.invoke(args.audio)


if __name__ == "__main__":
    main()
