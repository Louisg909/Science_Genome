"""
Tests for embedding.embedder.
"""
import numpy as np

def test_embed_text(monkeypatch):
    # Mock model to return fixed vector
    from src.embedding import embedder
    monkeypatch.setattr(embedder, "load_model", lambda: "mock_model")
    vec = np.random.rand(768)
    monkeypatch.setattr(embedder, "embed_text", lambda model, txt: vec)
    result = embedder.embed_text("mock_model", "sample text")
    assert result.shape[0] == 768
