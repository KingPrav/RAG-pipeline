from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import settings
from src.models import RetrievedChunk

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Score every (query, passage) pair with a cross-encoder and return the top-k."""
    if not chunks:
        return []

    k = top_k or settings.top_k_rerank
    model = _get_model()

    pairs = [(query, chunk.document.content) for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)

    return [
        RetrievedChunk(
            document=chunk.document,
            score=float(score),
            method=chunk.method,
            rank=i + 1,
        )
        for i, (score, chunk) in enumerate(ranked[:k])
    ]
