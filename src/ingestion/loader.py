from __future__ import annotations
import uuid
from pathlib import Path
from typing import Generator

import httpx

from src.models import Document

SUPPORTED_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".ipynb"}


def load_from_directory(path: str | Path) -> Generator[Document, None, None]:
    """Recursively load supported files from a local directory."""
    for filepath in Path(path).rglob("*"):
        if filepath.suffix not in SUPPORTED_EXTENSIONS or not filepath.is_file():
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                yield Document(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(filepath.resolve()))),
                    content=content,
                    metadata={
                        "source": str(filepath),
                        "filename": filepath.name,
                        "extension": filepath.suffix,
                        "type": "file",
                    },
                )
        except Exception:
            continue


def load_from_github(
    repo_url: str,
    branch: str = "main",
    subdir: str = "",
) -> Generator[Document, None, None]:
    """Clone a GitHub repo (shallow) and yield documents from it."""
    import git
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        git.Repo.clone_from(repo_url, tmpdir, branch=branch, depth=1)
        load_dir = Path(tmpdir) / subdir if subdir else Path(tmpdir)
        yield from load_from_directory(load_dir)


def load_from_urls(urls: list[str]) -> Generator[Document, None, None]:
    """Fetch raw text/markdown content from a list of URLs."""
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for url in urls:
            try:
                resp = client.get(url)
                resp.raise_for_status()
                yield Document(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, url)),
                    content=resp.text,
                    metadata={"source": url, "type": "web", "extension": ".md"},
                )
            except Exception as e:
                print(f"Skipping {url}: {e}")
