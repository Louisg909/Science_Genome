"""Minimal plotting helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt

from src.models import Paper


def scatter_embeddings(
    reduced_embeddings: np.ndarray,
    papers: Iterable[Paper],
    labels: Optional[Iterable[str]] = None,
):
    """Create a simple 2D scatter plot for reduced embeddings."""

    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys = reduced_embeddings[:, 0], reduced_embeddings[:, 1]
    ax.scatter(xs, ys, c="C0", alpha=0.75)

    for (x, y), paper in zip(zip(xs, ys), papers):
        ax.annotate(paper.title, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    if labels is not None:
        for label, (x, y) in zip(labels, zip(xs, ys)):
            ax.text(x, y, f" {label}", fontsize=8, color="gray")

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Scientific paper map")
    fig.tight_layout()
    return fig
