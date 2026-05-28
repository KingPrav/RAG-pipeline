from __future__ import annotations
import time
import uuid

from src.config import settings
from src.generation.generator import generate, generate_stream
from src.models import RAGRequest, RAGResponse, RetrievalDebug, RetrievedChunk
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import rerank


class RAGPipeline:
    """End-to-end pipeline: hybrid retrieval → cross-encoder reranking → Claude generation."""

    def __init__(self) -> None:
        if settings.enable_tracing:
            from src.observability.tracing import setup_tracing
            setup_tracing()
        self.hybrid = HybridRetriever()

    def _retrieve(self, request: RAGRequest) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        if request.retrieval_mode == "hybrid":
            return self.hybrid.retrieve(request.query, top_k=settings.top_k_dense)
        if request.retrieval_mode == "dense":
            results = self.hybrid.dense.retrieve(request.query, top_k=settings.top_k_dense)
            debug = RetrievalDebug(dense_count=len(results), mode="dense")
            return results, debug
        results = self.hybrid.sparse.retrieve(request.query, top_k=settings.top_k_sparse)
        debug = RetrievalDebug(sparse_count=len(results), mode="sparse")
        return results, debug

    def query(self, request: RAGRequest) -> RAGResponse:
        start = time.perf_counter()

        candidates, debug = self._retrieve(request)
        reranked = rerank(request.query, candidates, top_k=request.top_k)
        answer = generate(request.query, reranked)

        return RAGResponse(
            answer=answer,
            sources=reranked,
            query=request.query,
            retrieval_debug=debug,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            trace_id=str(uuid.uuid4()),
        )

    def query_stream(self, request: RAGRequest):
        """Yield (event_type, payload) tuples for streaming responses.

        Events:
          ("sources", list[RetrievedChunk])  — emitted once before generation
          ("token",   str)                    — one per generated token
        """
        candidates, _ = self._retrieve(request)
        reranked = rerank(request.query, candidates, top_k=request.top_k)

        yield "sources", reranked

        for token in generate_stream(request.query, reranked):
            yield "token", token
