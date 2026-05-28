from __future__ import annotations
import ast

from src.models import Document


def _extract_ast_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    lines: list[str],
    doc: Document,
) -> Document | None:
    start = node.lineno - 1
    end = node.end_lineno  # type: ignore[attr-defined]
    content = "\n".join(lines[start:end])
    if len(content.split()) < 10:
        return None
    return Document(
        id=f"{doc.id}_ast_{start}_{end}",
        content=content,
        metadata={
            **doc.metadata,
            "chunk_type": "code",
            "node_type": type(node).__name__,
            "node_name": getattr(node, "name", ""),
            "start_line": start + 1,
            "end_line": end,
        },
    )


def chunk_python_file(
    doc: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """AST-aware chunking: module-level and class-level definitions as atomic units."""
    try:
        tree = ast.parse(doc.content)
    except SyntaxError:
        return chunk_text(doc, chunk_size, chunk_overlap)

    lines = doc.content.splitlines()
    chunks: list[Document] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _extract_ast_node(node, lines, doc)
            if chunk:
                chunks.append(chunk)
        elif isinstance(node, ast.ClassDef):
            class_chunk = _extract_ast_node(node, lines, doc)
            if class_chunk:
                chunks.append(class_chunk)
            # Also extract individual methods for fine-grained retrieval
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = _extract_ast_node(child, lines, doc)
                    if method_chunk:
                        chunks.append(method_chunk)

    return chunks if chunks else chunk_text(doc, chunk_size, chunk_overlap)


def chunk_text(
    doc: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Sliding-window word-level chunking for prose and markdown."""
    words = doc.content.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for chunk_idx, i in enumerate(range(0, len(words), step)):
        content = " ".join(words[i : i + chunk_size])
        chunks.append(
            Document(
                id=f"{doc.id}_chunk_{chunk_idx}",
                content=content,
                metadata={**doc.metadata, "chunk_type": "text", "chunk_index": chunk_idx},
            )
        )
    return chunks


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    if doc.metadata.get("extension") == ".py":
        return chunk_python_file(doc, chunk_size, chunk_overlap)
    return chunk_text(doc, chunk_size, chunk_overlap)
