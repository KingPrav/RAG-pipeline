from __future__ import annotations
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.models import RAGRequest, RAGResponse
from src.pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = RAGPipeline()
    yield
    _pipeline = None


app = FastAPI(
    title="RAG Pipeline API",
    version="0.1.0",
    description="Hybrid retrieval (BM25 + dense) + cross-encoder reranking + Claude generation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return _pipeline


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=RAGResponse)
async def query(request: RAGRequest):
    """Single-shot query — returns full answer + sources + debug info."""
    return _get_pipeline().query(request)


@app.post("/query/stream")
async def query_stream(request: RAGRequest):
    """Server-sent events stream: first a 'sources' event, then 'token' events."""
    pipeline = _get_pipeline()

    def generator():
        for event_type, payload in pipeline.query_stream(request):
            if event_type == "sources":
                data = json.dumps({"type": "sources", "data": [s.model_dump() for s in payload]})
            else:
                data = json.dumps({"type": "token", "data": payload})
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
