"""Minimal embedding utilities.

The real project would load heavy language models here.  For the purposes of
the exercises and tests we only provide lightweight stubs that can be patched
in tests.
"""

from __future__ import annotations

from typing import Any


def load_model() -> Any:  # pragma: no cover - placeholder
    """Return a handle to the embedding model.

    The default implementation raises ``NotImplementedError`` to make the
    intent explicit; tests patch this function with a mock model.
    """

    raise NotImplementedError("Model loading not implemented")


def embed_text(model: Any, text: str) -> Any:  # pragma: no cover - placeholder
    """Embed ``text`` using ``model``.

    The stub raises ``NotImplementedError``; tests replace it with a simple
    lambda returning a fixed vector.
    """

    raise NotImplementedError("Embedding not implemented")


__all__ = ["load_model", "embed_text"]

