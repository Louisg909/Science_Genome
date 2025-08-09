"""
Tests for embedding.utils.
"""
from src.embedding import utils

def test_clean_text():
    raw = "  This is   SAMPLE text.   "
    cleaned = utils.clean_text(raw)
    assert "sample" in cleaned.lower()
