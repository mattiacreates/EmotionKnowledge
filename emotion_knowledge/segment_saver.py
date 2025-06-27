import os
import uuid
from typing import Any, Dict

from langchain_core.runnables import Runnable
from pydub import AudioSegment
import chromadb


class SegmentSaver(Runnable):
    """Save diarized segments to disk and ChromaDB."""

    def __init__(self, collection_name: str = "segments", db_path: str = "segment_db", output_dir: str = "clips") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def invoke(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Slice the audio segment and store its metadata."""
        audio_path = segment["audio_path"]
        start_ms = int(float(segment["start"]) * 1000)
        end_ms = int(float(segment["end"]) * 1000)
        speaker = segment.get("speaker", "speaker").lower()

        clip_name = f"{speaker}_{uuid.uuid4().hex}.wav"
        clip_path = os.path.join(self.output_dir, clip_name)

        audio = AudioSegment.from_file(audio_path)
        audio[start_ms:end_ms].export(clip_path, format="wav")

        doc_id = uuid.uuid4().hex
        metadata = {
            "speaker": speaker,
            "start_time": segment["start"],
            "end_time": segment["end"],
            "text": segment.get("text", ""),
            "audio_clip_path": clip_path,
        }
        self.collection.add(documents=[metadata["text"]], metadatas=[metadata], ids=[doc_id])
        return {"clip_path": clip_path, "speaker": speaker, "doc_id": doc_id}
