from __future__ import annotations

from src.models import RetrievedChunk

SYSTEM_PROMPT = """\
You are a precise technical assistant that answers questions using retrieved documentation.

Guidelines:
- Ground every claim in the provided context passages
- Cite sources inline using [Source N] notation — e.g. "According to [Source 1]..."
- Include relevant code snippets from the context when they directly answer the question
- If the context is insufficient, say so explicitly rather than guessing
- Be concise; omit preamble like "Based on the context provided..."
"""


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.document.metadata.get("source", "unknown")
        node_name = chunk.document.metadata.get("node_name", "")
        label = f"{source} — {node_name}" if node_name else source
        context_parts.append(f"[Source {i}] {label}\n\n{chunk.document.content}")

    context = "\n\n---\n\n".join(context_parts)
    return f"<context>\n{context}\n</context>\n\nQuestion: {query}"
