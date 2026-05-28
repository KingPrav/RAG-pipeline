from __future__ import annotations

import chromadb
from openai import OpenAI

from src.config import settings
from src.models import Document, RetrievedChunk


class DenseRetriever:
    def __init__(self) -> None:
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.chroma = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = min(top_k or settings.top_k_dense, self.collection.count())
        if k == 0:
            return []

        embedding = (
            self.openai.embeddings.create(model=settings.embedding_model, input=[query])
            .data[0]
            .embedding
        )
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i, (doc_id, doc, meta, dist) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            chunks.append(
                RetrievedChunk(
                    document=Document(id=doc_id, content=doc, metadata=meta or {}),
                    score=1.0 - dist,  # cosine distance → similarity
                    method="dense",
                    rank=i + 1,
                )
            )
        return chunks
