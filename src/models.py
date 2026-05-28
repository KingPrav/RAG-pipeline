from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    document: Document
    score: float
    method: Literal["dense", "sparse", "hybrid"]
    rank: int


class RAGRequest(BaseModel):
    query: str
    top_k: int = 5
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"


class RetrievalDebug(BaseModel):
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    dense_only_count: int = 0
    sparse_only_count: int = 0
    overlap_count: int = 0
    mode: str = "hybrid"


class RAGResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    query: str
    retrieval_debug: RetrievalDebug
    latency_ms: float
    trace_id: str = ""
