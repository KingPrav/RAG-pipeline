#!/usr/bin/env python3
"""Ingest documents into the RAG index.

Usage:
  python scripts/ingest.py --dir data/sample
  python scripts/ingest.py --github https://github.com/tiangolo/fastapi --subdir docs/en/docs
"""
from __future__ import annotations
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

from src.config import settings
from src.ingestion.chunker import chunk_document
from src.ingestion.indexer import Indexer
from src.ingestion.loader import load_from_directory, load_from_github

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", metavar="PATH", help="Local directory to ingest")
    group.add_argument("--github", metavar="URL", help="GitHub repo URL (shallow clone)")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--subdir", default="", help="Subdirectory within the repo")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    console.rule("[bold blue]RAG Ingestion Pipeline")

    console.print("[dim]Loading documents...[/dim]")
    if args.dir:
        raw_docs = list(load_from_directory(args.dir))
        console.print(f"  Loaded [green]{len(raw_docs)}[/green] files from [cyan]{args.dir}[/cyan]")
    else:
        raw_docs = list(load_from_github(args.github, args.branch, args.subdir))
        console.print(f"  Loaded [green]{len(raw_docs)}[/green] files from [cyan]{args.github}[/cyan]")

    if not raw_docs:
        console.print("[red]No documents found — check the path and supported extensions.[/red]")
        sys.exit(1)

    console.print("[dim]Chunking...[/dim]")
    chunks = []
    for doc in raw_docs:
        doc_chunks = chunk_document(doc, args.chunk_size, args.chunk_overlap)
        chunks.extend(doc_chunks)

    # Log chunking stats
    code_chunks = sum(1 for c in chunks if c.metadata.get("chunk_type") == "code")
    text_chunks = len(chunks) - code_chunks
    console.print(
        f"  [green]{len(chunks)}[/green] chunks "
        f"([cyan]{code_chunks}[/cyan] AST code, [cyan]{text_chunks}[/cyan] text)"
    )

    console.print("[dim]Embedding and indexing...[/dim]")
    indexer = Indexer()
    indexer.index(chunks)

    console.rule("[bold green]Done")
    console.print("Run the demo:  [bold]streamlit run app/streamlit_app.py[/bold]")
    console.print("Start the API: [bold]uvicorn api.main:app --reload[/bold]")
    console.print("Run eval:      [bold]python scripts/evaluate.py[/bold]")


if __name__ == "__main__":
    main()
