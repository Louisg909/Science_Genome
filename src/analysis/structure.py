"""Analysis helpers for science-space structure and diagnostics."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from src.models import Paper


def embedding_norm_diagnostics(embeddings: np.ndarray) -> dict:
    """Summarize embedding norm distribution for consistency checks."""

    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "count": int(norms.size),
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
    }


def nearest_neighbor_density(embeddings: np.ndarray, n_neighbors: int = 5) -> dict:
    """Estimate local density using mean cosine similarity to nearest neighbours."""

    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    if matrix.shape[0] < 2:
        raise ValueError("at least two embeddings are required")
    k = min(max(1, n_neighbors), matrix.shape[0] - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    model.fit(matrix)
    distances, _ = model.kneighbors(return_distance=True)
    similarities = 1.0 - distances[:, 1:]
    per_paper_density = np.mean(similarities, axis=1)
    return {
        "neighbors_used": int(k),
        "mean_density": float(np.mean(per_paper_density)),
        "std_density": float(np.std(per_paper_density)),
        "min_density": float(np.min(per_paper_density)),
        "max_density": float(np.max(per_paper_density)),
    }


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
