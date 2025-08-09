"""
Storage Helpers.

Purpose:
    High-level insertion and retrieval for papers, embeddings, citations.

Goals:
    - Hide database complexity behind clean functions.
    - Enforce schema consistency.

Inputs:
    - Metadata dicts, embedding vectors.

Outputs:
    - DB commits (insert/update).
    - Fetched records as Python dicts or objects.
"""
