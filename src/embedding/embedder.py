"""SciBERT based embedding utilities.

This module provides a thin wrapper around the
``allenai/scibert_scivocab_uncased`` model.  It exposes synchronous and
asynchronous helpers so that embedding can run in parallel with network bound
work such as scraping.  Embeddings are returned as NumPy arrays but can be
translated to and from raw bytes for compact storage.
"""

from __future__ import annotations

import asyncio
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SciBERTEmbedder:
    """Generate text embeddings using SciBERT.

    The embedder focuses on the ``[CLS]`` token representation which provides a
    single fixed-size vector per input.  The heavy model is loaded once during
    initialisation and kept in evaluation mode.
    """

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        """Return the embedding vector for ``text``."""

        tokens = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            hidden = self.model(**tokens).last_hidden_state
        return hidden[0, 0].detach().cpu().numpy()

    async def embed_async(self, text: str) -> np.ndarray:
        """Asynchronously embed ``text`` using a thread pool."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, text)


def load_model() -> SciBERTEmbedder:
    """Return an instance of :class:`SciBERTEmbedder`."""

    return SciBERTEmbedder()


def embed_text(model: SciBERTEmbedder, text: str) -> np.ndarray:
    """Embed ``text`` synchronously using ``model``."""

    return model.embed(text)


async def embed_text_async(model: SciBERTEmbedder, text: str) -> np.ndarray:
    """Embed ``text`` asynchronously using ``model``."""

    return await model.embed_async(text)


def to_binary(embedding: Iterable[float] | np.ndarray | torch.Tensor) -> bytes:
    """Convert an embedding array to raw ``bytes``."""

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    array = np.asarray(embedding, dtype=np.float32)
    return array.tobytes()


def from_binary(blob: bytes) -> np.ndarray:
    """Translate raw ``bytes`` back into a NumPy array."""

    return np.frombuffer(blob, dtype=np.float32)


def translate(data: bytes | Iterable[float] | np.ndarray | torch.Tensor) -> bytes | np.ndarray:
    """Bidirectionally convert between binary and array forms."""

    return from_binary(data) if isinstance(data, bytes) else to_binary(data)


__all__ = [
    "SciBERTEmbedder",
    "load_model",
    "embed_text",
    "embed_text_async",
    "to_binary",
    "from_binary",
    "translate",
]

