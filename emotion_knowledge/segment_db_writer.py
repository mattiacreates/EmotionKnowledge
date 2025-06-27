import json
from pathlib import Path
from typing import Dict

import numpy as np
from chromadb import PersistentClient
from pydub import AudioSegment


class _ZeroEmbedding:
    def __call__(self, input):
        return [np.zeros(1)] * len(input)

    def name(self):
        return "zero"


class SegmentDBWriter:
    """Store transcript segments in ChromaDB and export audio clips."""

    def __init__(self, db_path: str = "db", clip_dir: str = "clips"):
        self.db_path = Path(db_path)
        self.clip_dir = Path(clip_dir)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.client = PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            "segments", embedding_function=_ZeroEmbedding()
        )

    def __call__(self, transcript: Dict) -> Dict:
        audio = AudioSegment.from_file(transcript["audio_path"])
        for seg in transcript["segments"]:
            seg_id = f"{Path(transcript['audio_path']).stem}-{int(seg['start']*1000)}-{int(seg['end']*1000)}"
            existing = self.collection.get(ids=[seg_id])
            if existing.get("ids"):
                continue
            clip_path = self.clip_dir / f"{seg_id}.wav"
            if not clip_path.exists():
                clip = audio[seg["start"] * 1000 : seg["end"] * 1000]
                clip.export(clip_path, format="wav")
            metadata = {
                "speaker": seg.get("speaker"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "audio": str(clip_path),
            }
            self.collection.add(
                ids=[seg_id],
                documents=[seg.get("text", "")],
                metadatas=[metadata],
                embeddings=[np.zeros(1)],
            )
        return transcript
