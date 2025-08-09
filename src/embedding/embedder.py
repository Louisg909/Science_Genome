"""
Embedder.

Purpose:
    Wrap SciBERT (or other models) to generate paper embeddings.

Goals:
    - Batch processing for throughput.
    - Pluggable model architecture.
    - Deterministic, reproducible embeddings.

Inputs:
    - Cleaned text (title + abstract).

Outputs:
    - Dense vectors (numpy arrays or torch tensors).
    - Embeddings saved to database with DOI linkage.
"""
