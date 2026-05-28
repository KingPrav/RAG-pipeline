from __future__ import annotations
import pickle
from pathlib import Path

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from rich.progress import track

from src.config import settings
from src.models import Document


class Indexer:
    def __init__(self) -> None:
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.chroma = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.openai.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [r.embedding for r in response.data]

    def index(self, documents: list[Document], batch_size: int = 100) -> None:
        # Deduplicate against existing collection
        existing_ids = set(self.collection.get()["ids"])
        new_docs = [d for d in documents if d.id not in existing_ids]

        if not new_docs:
            print("All documents already indexed.")
            return

        # Dense index (ChromaDB + OpenAI embeddings)
        for i in track(range(0, len(new_docs), batch_size), description="Embedding & indexing"):
            batch = new_docs[i : i + batch_size]
            texts = [d.content for d in batch]
            embeddings = self._embed_batch(texts)
            self.collection.add(
                ids=[d.id for d in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[d.metadata for d in batch],
            )

        # BM25 index — rebuild over full corpus (existing + new)
        all_ids = self.collection.get()["ids"]
        all_docs_raw = self.collection.get(ids=all_ids, include=["documents", "metadatas"])
        corpus = [
            Document(id=doc_id, content=content, metadata=meta or {})
            for doc_id, content, meta in zip(
                all_docs_raw["ids"],
                all_docs_raw["documents"],
                all_docs_raw["metadatas"],
            )
        ]
        tokenized = [d.content.lower().split() for d in corpus]
        bm25 = BM25Okapi(tokenized)

        bm25_path = Path(settings.bm25_index_path)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "corpus": corpus}, f)

        print(f"Indexed {len(new_docs)} new chunks ({len(corpus)} total)")
