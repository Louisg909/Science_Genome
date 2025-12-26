"""Tiny analysis helpers for the science embedding."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from src.models import Paper


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute a cosine similarity matrix for the embedding space."""

    return cosine_similarity(embeddings)


def nearest_neighbors(
    embeddings: np.ndarray, papers: Iterable[Paper], n_neighbors: int = 3
) -> List[Tuple[Paper, List[Paper]]]:
    """Return the closest neighbors for each paper."""

    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    neighbor_model.fit(embeddings)
    indices = neighbor_model.kneighbors(return_distance=False)

    paper_list = list(papers)
    results: List[Tuple[Paper, List[Paper]]] = []
    for idx, row in enumerate(indices):
        neighbors = [paper_list[i] for i in row if i != idx][:n_neighbors]
        results.append((paper_list[idx], neighbors))
    return results
