"""Embedding utilities package."""

# Submodules are exposed lazily to avoid importing heavy dependencies such as
# ``transformers`` when not required.  They can be imported directly via
# ``from src.embedding import embedder`` or ``utils``.

__all__ = ["embedder", "utils"]
