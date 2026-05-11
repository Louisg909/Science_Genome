"""Corpus-level inheritance fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .inheritance import Constraint, solve_inheritance


@dataclass(frozen=True)
class PaperInheritanceFit:
    """Compact serializable result for one focal paper."""

    parent_ids: tuple[str, ...]
    weights: np.ndarray
    residual_vector: np.ndarray
    residual_magnitude: float
    converged: bool
    iterations: int
    objective: float


def _paper_id(paper: Any) -> str:
    if isinstance(paper, Mapping):
        identifier = paper.get("paper_id")
    else:
        identifier = getattr(paper, "paper_id", None)
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Each paper must include a non-empty paper_id.")
    return identifier.strip()


def _parent_ids_for_focal(dag: Any, focal_paper_id: str) -> list[str]:
    if hasattr(dag, "predecessors"):
        return list(dag.predecessors(focal_paper_id))
    if isinstance(dag, Mapping):
        return list(dag.get(focal_paper_id, ()))
    raise TypeError("dag must provide either a 'predecessors' method or mapping interface.")


def fit_inheritance_for_corpus(
    papers: Sequence[Any],
    embeddings: np.ndarray,
    dag: Any,
    *,
    constraint: Constraint = "simplex",
    sparsity: int | None = None,
    l2_regularizer: float = 1e-6,
    max_iter: int = 5_000,
    learning_rate: float = 0.05,
    tolerance: float = 1e-10,
    random_state: int | None = None,
) -> Dict[str, PaperInheritanceFit]:
    """Fit inheritance decomposition for each paper in a corpus.

    Returns a dictionary keyed by focal paper_id.
    """

    embedding_matrix = np.asarray(embeddings, dtype=float)
    if embedding_matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D array with shape (num_papers, embedding_dim).")
    if embedding_matrix.shape[0] != len(papers):
        raise ValueError("embeddings row count must match number of papers.")

    paper_ids = [_paper_id(paper) for paper in papers]
    index_by_id = {pid: idx for idx, pid in enumerate(paper_ids)}

    results: Dict[str, PaperInheritanceFit] = {}
    for focal_idx, focal_paper_id in enumerate(paper_ids):
        parent_ids = [pid for pid in _parent_ids_for_focal(dag, focal_paper_id) if pid in index_by_id]
        parent_matrix = embedding_matrix[[index_by_id[pid] for pid in parent_ids]].T
        target_embedding = embedding_matrix[focal_idx]

        solved = solve_inheritance(
            target_embedding,
            parent_matrix,
            constraint=constraint,
            sparsity=sparsity,
            l2_regularizer=l2_regularizer,
            max_iter=max_iter,
            learning_rate=learning_rate,
            tolerance=tolerance,
            random_state=random_state,
        )

        results[focal_paper_id] = PaperInheritanceFit(
            parent_ids=tuple(parent_ids),
            weights=solved.weights,
            residual_vector=solved.residual,
            residual_magnitude=float(np.linalg.norm(solved.residual)),
            converged=solved.converged,
            iterations=solved.iterations,
            objective=solved.objective,
        )

    return results
