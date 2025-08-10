"""Utility helpers for text pre-processing.

This module intentionally keeps the functionality minimal – it provides only
the pieces required by the tests and the rest of the project.  Keeping the
surface area small makes it easier to maintain and reason about.
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Normalise ``text`` for embedding.

    The function collapses consecutive whitespace characters and trims leading
    and trailing spaces.  It performs no tokenisation or heavy processing,
    keeping the routine lightweight while still providing reasonably clean
    input for downstream models.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.sub(r"\s+", " ", text).strip()


__all__ = ["clean_text"]

