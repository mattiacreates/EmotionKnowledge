from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from chromadb import PersistentClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F


class DBEmotionAnnotator:
    """Annotate DB segments with emotions using a HF model."""

    def __init__(
        self,
        db_path: str = "db",
        model_name: str = "oliverguhr/german-emotion-bert",
        batch_size: int = 8,
        load_in_8bit: bool = False,
    ):
        self.client = PersistentClient(path=str(Path(db_path)))
        self.collection = self.client.get_or_create_collection("segments")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"device_map": "auto"}
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def _batch(self, items: List[str], size: int) -> List[List[str]]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def __call__(self, transcript: Dict) -> Dict:
        data = self.collection.get(include=["metadatas", "documents", "ids"])
        pending = [
            (i, doc, meta)
            for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"]))
            if meta is not None and "emotion" not in meta
        ]
        texts = [doc[:200] for _, doc, _ in pending]
        ids = [data["ids"][i] for i, _, _ in pending]
        results = {}
        for batch_ids, batch_texts in zip(
            [ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)],
            self._batch(texts, self.batch_size),
        ):
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = F.softmax(logits, dim=-1)
            emotions = [self.labels[p.argmax().item()] for p in probs]
            for doc_id, emo in zip(batch_ids, emotions):
                idx = ids.index(doc_id)
                meta = data["metadatas"][idx]
                meta["emotion"] = emo
                self.collection.update(ids=[doc_id], metadatas=[meta])
                results[doc_id] = emo
            torch.cuda.empty_cache()
        return transcript
