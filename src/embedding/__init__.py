"""Embedding utilities."""

from .embedder import (
    SciBERTEmbedder,
    load_model,
    embed_text,
    embed_text_async,
    to_binary,
    from_binary,
    translate,
)

__all__ = [
    "SciBERTEmbedder",
    "load_model",
    "embed_text",
    "embed_text_async",
    "to_binary",
    "from_binary",
    "translate",
]
