from __future__ import annotations
import os
from typing import Dict, Any, Optional

from langchain_core.runnables import Runnable

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
        audio_path = entry.get("audio_clip_path")
        text = entry.get("text", "")
        label = "Neutral"
        scores: Dict[str, float] = {}
        if audio_path and os.path.exists(audio_path):
            scores = self.emotion_model.predict_scores(audio_path)
            if scores:
                raw_label = max(scores, key=scores.get)
                label = self.label_map.get(raw_label.lower(), raw_label)
                entry["emotion_scores"] = scores
                entry["emotion_top_label"] = raw_label

        entry["emotion_annotated_text"] = f"[{label}] {text}".strip()
        return entry


def annotate_chromadb(db_path: str, collection_name: str = "segments") -> None:
    """Add emotion_annotated_text to each entry in a ChromaDB collection."""
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
        meta["emotion_annotated_text"] = entry["emotion_annotated_text"]
        new_metas.append(meta)
    if ids:
        collection.update(ids=ids, metadatas=new_metas)
