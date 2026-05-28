from __future__ import annotations
from typing import Generator

import anthropic

from src.config import settings
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.models import RetrievedChunk

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def generate(query: str, chunks: list[RetrievedChunk]) -> str:
    """Single-shot generation with Claude."""
    client = _get_client()
    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(query, chunks)}],
    )
    return message.content[0].text  # type: ignore[union-attr]


def generate_stream(query: str, chunks: list[RetrievedChunk]) -> Generator[str, None, None]:
    """Stream tokens from Claude."""
    client = _get_client()
    with client.messages.stream(
        model=settings.claude_model,
        max_tokens=settings.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(query, chunks)}],
    ) as stream:
        yield from stream.text_stream
