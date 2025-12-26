"""Lightweight dimension reduction helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:  # Optional; keeps dependencies light
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    umap = None


def reduce_dimensions(
    embeddings: np.ndarray,
    method: Literal["pca", "tsne", "umap"] = "pca",
    n_components: int = 2,
    random_state: int = 0,
) -> np.ndarray:
    """Project embeddings into a lower-dimensional space."""

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings)

    if method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, init="random")
        return reducer.fit_transform(embeddings)

    if method == "umap":
        if umap is None:
            raise ImportError("umap-learn is not installed")
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings)

    raise ValueError(f"Unknown reduction method: {method}")
