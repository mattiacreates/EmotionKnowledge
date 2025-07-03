import os
import uuid
from typing import Any, Dict

from langchain_core.runnables import Runnable
from pydub import AudioSegment
import chromadb
import logging

logger = logging.getLogger(__name__)


class SegmentSaver(Runnable):
    """Save diarized segments to disk and ChromaDB."""

    def __init__(self, collection_name: str = "segments", db_path: str = "segment_db", output_dir: str = "clips") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)

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
        audio[start_ms:end_ms].export(clip_path, format="wav")

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
        self.collection.add(
            documents=[metadata["text"]], metadatas=[metadata], ids=[doc_id]
        )
        return {"clip_path": clip_path, "speaker": speaker, "doc_id": doc_id}
