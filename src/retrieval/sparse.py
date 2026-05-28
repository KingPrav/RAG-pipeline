from __future__ import annotations
import pickle

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.models import Document, RetrievedChunk


class SparseRetriever:
    def __init__(self) -> None:
        with open(settings.bm25_index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25: BM25Okapi = data["bm25"]
        self.corpus: list[Document] = data["corpus"]

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or settings.top_k_sparse
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:k]

        chunks = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunks.append(
                    RetrievedChunk(
                        document=self.corpus[idx],
                        score=float(scores[idx]),
                        method="sparse",
                        rank=rank + 1,
                    )
                )
        return chunks
