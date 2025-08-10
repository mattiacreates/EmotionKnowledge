import os
import uuid
from typing import Any, Dict

try:
    from langchain_core.runnables import Runnable
except Exception:  # pragma: no cover - optional dependency
    class Runnable:  # minimal fallback so tests can import module
        def invoke(self, *args, **kwargs):  # pragma: no cover - not used in tests
            raise ImportError("langchain-core is required for this feature")
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    class _MissingChroma:  # minimal stub so tests can patch PersistentClient
        class PersistentClient:
            def __init__(self, *args, **kwargs):
                raise ImportError("chromadb is required for this feature")

    chromadb = _MissingChroma()
import logging

logger = logging.getLogger(__name__)


class SegmentSaver(Runnable):
    """Save diarized segments to disk and ChromaDB.

    Call :meth:`reset_db` to clear the underlying Chroma database.  This can
    also be triggered from the command line via ``python -m emotion_knowledge
    --reset-db``.
    """

    def __init__(
        self,
        collection_name: str = "segments",
        db_path: str = "segment_db",
        output_dir: str = "clips",
        end_buffer_ms: int = 50,
        trim_silence: bool = True,
    ) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.end_buffer_ms = max(0, int(end_buffer_ms))
        self.trim_silence = trim_silence

    def invoke(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Slice the audio segment and store its metadata."""
        audio_path = segment.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            logger.warning("Audio path missing or invalid for segment %s", segment)
            return {}

        # WhisperX may produce either "start"/"end" or "start_time"/"end_time".
        if "start" in segment and "end" in segment:
            start_val = segment["start"]
            end_val = segment["end"]
        elif "start_time" in segment and "end_time" in segment:
            start_val = segment["start_time"]
            end_val = segment["end_time"]
        else:
            logger.warning("Skipping segment due to missing time keys %s", list(segment.keys()))
            return {}

        try:
            start_ms = max(0, int(float(start_val) * 1000))
            end_ms = max(start_ms, int(float(end_val) * 1000))
        except Exception as exc:
            logger.error("Invalid start/end values %s %s %s", start_val, end_val, exc)
            return {}

        speaker = segment.get("speaker", "speaker").lower()
        clip_name = f"{speaker}_{uuid.uuid4().hex}.wav"
        clip_path = os.path.join(self.output_dir, clip_name)

        audio = AudioSegment.from_file(audio_path)
        clip = audio[start_ms:end_ms]

        if self.end_buffer_ms and len(clip) > self.end_buffer_ms:
            clip = clip[:-self.end_buffer_ms]

        if self.trim_silence:
            try:
                nonsilent = detect_nonsilent(
                    clip,
                    min_silence_len=150,
                    silence_thresh=clip.dBFS - 30,
                )
                if nonsilent:
                    start_trim, end_trim = nonsilent[0][0], nonsilent[-1][1]
                    clip = clip[start_trim:end_trim]
            except Exception as exc:  # pragma: no cover - silence detection optional
                logger.debug("Silence trimming failed: %s", exc)

        clip.export(clip_path, format="wav")

        duration_ms = end_ms - start_ms
        logger.info(
            "Saved clip %s (%d ms) speaker=%s text='%s'",
            clip_path,
            duration_ms,
            speaker,
            segment.get("text", ""),
        )

        doc_id = uuid.uuid4().hex
        metadata = {
            "speaker": speaker,
            "start_time": float(start_val),
            "end_time": float(end_val),
            "text": segment.get("text", ""),
            "audio_clip_path": clip_path,
        }
        for key in (
            "duration",
            "n_words",
            "words_per_sec",
            "mean_word_gap",
            "p95_word_gap",
            "overlaps_started",
        ):
            if key in segment:
                metadata[key] = segment[key]
        self.collection.add(
            documents=[metadata["text"]], metadatas=[metadata], ids=[doc_id]
        )
        return {"clip_path": clip_path, "speaker": speaker, "doc_id": doc_id}

    def reset_db(self) -> None:
        """Reset the underlying ChromaDB instance.

        Removes all collections using :meth:`chromadb.PersistentClient.reset`.
        Useful for cleaning up between runs or tests.  The same operation can be
        invoked from the command line with ``python -m emotion_knowledge
        --reset-db``.
        """

        self.client.reset()
