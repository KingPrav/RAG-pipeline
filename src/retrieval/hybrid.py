from __future__ import annotations

from src.config import settings
from src.models import RetrievalDebug, RetrievedChunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Combine two ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank)) across all lists that contain the document.
    k=60 is the standard value from the original paper (Cormack et al., 2009).
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results, 1):
        doc_id = chunk.document.id
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        chunks[doc_id] = chunk

    for rank, chunk in enumerate(sparse_results, 1):
        doc_id = chunk.document.id
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in chunks:
            chunks[doc_id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        RetrievedChunk(
            document=chunks[doc_id].document,
            score=score,
            method="hybrid",
            rank=i + 1,
        )
        for i, (doc_id, score) in enumerate(ranked)
    ]


class HybridRetriever:
    def __init__(self) -> None:
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        k = top_k or settings.top_k_dense

        dense_results = self.dense.retrieve(query, top_k=k)
        sparse_results = self.sparse.retrieve(query, top_k=k)
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        dense_ids = {c.document.id for c in dense_results}
        sparse_ids = {c.document.id for c in sparse_results}

        debug = RetrievalDebug(
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            fused_count=len(fused),
            dense_only_count=len(dense_ids - sparse_ids),
            sparse_only_count=len(sparse_ids - dense_ids),
            overlap_count=len(dense_ids & sparse_ids),
            mode="hybrid",
        )

        return fused[:k], debug
