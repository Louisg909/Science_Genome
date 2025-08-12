"""Tests for :mod:`src.embedding.embedder`."""

import asyncio

import numpy as np
import torch

from src.embedding import embedder


class DummyModel:
    """Lightweight stand-in for :class:`SciBERTEmbedder`."""

    def embed(self, text: str) -> np.ndarray:  # pragma: no cover - trivial
        return np.arange(3, dtype=np.float32)

    async def embed_async(self, text: str) -> np.ndarray:
        return self.embed(text)


def test_embed_text_sync_and_async_consistency():
    model = DummyModel()
    vec_sync = embedder.embed_text(model, "sample")
    vec_async = asyncio.run(embedder.embed_text_async(model, "sample"))

    assert vec_sync.shape == (3,)
    assert vec_async.shape == (3,)
    assert vec_sync.dtype == np.float32
    assert vec_async.dtype == np.float32
    np.testing.assert_allclose(vec_sync, vec_async)


def test_to_from_binary_numpy_roundtrip():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    blob = embedder.to_binary(vec)
    restored = embedder.from_binary(blob)

    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, vec)


def test_to_from_binary_torch_roundtrip():
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    blob = embedder.to_binary(tensor)
    restored = embedder.from_binary(blob)

    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, tensor.numpy())


def test_translate_array_bytes_roundtrip():
    vec = np.array([4.0, 5.0], dtype=np.float32)
    blob = embedder.translate(vec)
    assert isinstance(blob, bytes)

    arr = embedder.translate(blob)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, vec)

