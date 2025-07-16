from __future__ import annotations
import os
from typing import Dict, Any, Optional

try:
    from langchain_core.runnables import Runnable
except Exception:  # pragma: no cover - optional dependency
    class Runnable:  # minimal fallback so tests can import module
        def invoke(self, *args, **kwargs):  # pragma: no cover - not used in tests
            raise ImportError("langchain-core is required for this feature")

from .emotion_models import EmotionModel


class AudioEmotionAnnotator(Runnable):
    """Annotate utterances with the dominant emotion from audio."""

    def __init__(
        self,
        emotion_model: Optional[EmotionModel] = None,
        label_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.emotion_model = emotion_model or EmotionModel()
        # keep an optional label_map for compatibility, but default to no mapping
        self.label_map = label_map or {}

    def invoke(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Return emotion label and transcript separately."""
        audio_path = entry.get("audio_clip_path")
        text = entry.get("text", "")

        label = ""
        confidence = 0.0
        if audio_path and os.path.exists(audio_path):
            raw_label, confidence = self.emotion_model.predict(audio_path)
            if raw_label:
                label = self.label_map.get(raw_label.lower(), raw_label)

        entry["audio_emotion_label"] = label
        entry["audio_emotion_confidence"] = confidence
        entry["audio_text"] = text
        return entry


def annotate_chromadb(db_path: str, collection_name: str = "segments") -> None:
    """Add audio emotion fields to each entry in a ChromaDB collection."""
    import chromadb

    annotator = AudioEmotionAnnotator()
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)
    result = collection.get(include=["metadatas", "documents", "ids"])
    metadatas = result.get("metadatas", [])
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    new_metas = []
    for meta, doc in zip(metadatas, documents):
        entry = meta.copy()
        entry["text"] = doc
        entry = annotator.invoke(entry)
        meta["audio_emotion_label"] = entry["audio_emotion_label"]
        meta["audio_emotion_confidence"] = entry["audio_emotion_confidence"]
        meta["audio_text"] = entry["audio_text"]
        new_metas.append(meta)
    if ids:
        collection.update(ids=ids, metadatas=new_metas)
