"""Corpus-level inheritance fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Sequence

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


DAGAdjacency = Mapping[str, Sequence[str]]


def citation_edges_to_parent_adjacency(edges: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """Convert citation edges into canonical child->parents adjacency mapping.

    Canonical DAG schema:
      - Mapping from `child_paper_id` -> ordered sequence of `parent_paper_id` strings.
      - Missing children imply no parents.
    """

    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError("Each DAG edge must be a (parent_id, child_id) tuple.")
        parent_id, child_id = edge
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise TypeError("Each edge parent_id must be a non-empty string.")
        if not isinstance(child_id, str) or not child_id.strip():
            raise TypeError("Each edge child_id must be a non-empty string.")
        adjacency.setdefault(child_id, []).append(parent_id)
    return adjacency


def _normalize_parent_adjacency(dag: DAGAdjacency | Any) -> dict[str, list[str]]:
    if hasattr(dag, "predecessors"):
        raise TypeError(
            "dag must be a mapping of child_id -> sequence[parent_id]. "
            "For edge lists, use citation_edges_to_parent_adjacency()."
        )
    if not isinstance(dag, Mapping):
        raise TypeError(
            "dag must be a mapping of child_id -> sequence[parent_id]. "
            "For edge lists, use citation_edges_to_parent_adjacency()."
        )

    normalized: dict[str, list[str]] = {}
    for child_id, parent_ids in dag.items():
        if not isinstance(child_id, str) or not child_id.strip():
            raise TypeError("Each dag key (child_id) must be a non-empty string.")
        if isinstance(parent_ids, (str, bytes)) or not isinstance(parent_ids, Sequence):
            raise TypeError("Each dag value must be a sequence of parent_id strings.")
        coerced: list[str] = []
        for parent_id in parent_ids:
            if not isinstance(parent_id, str) or not parent_id.strip():
                raise TypeError("Each parent_id in dag values must be a non-empty string.")
            coerced.append(parent_id)
        normalized[child_id] = coerced
    return normalized


def fit_inheritance_for_corpus(
    papers: Sequence[Any],
    embeddings: np.ndarray,
    dag: DAGAdjacency,
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

    Accepted DAG schema:
      - `dag` is a mapping: `child_paper_id -> sequence[parent_paper_id]`.
      - Both keys and parent IDs must be non-empty strings.
      - Children omitted from the mapping are treated as having zero parents.
      - To use `build_citation_dag` output (`list[tuple[parent, child]]`), first call
        `citation_edges_to_parent_adjacency`.

    Returns a dictionary keyed by focal paper_id.
    """

    embedding_matrix = np.asarray(embeddings, dtype=float)
    if embedding_matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D array with shape (num_papers, embedding_dim).")
    if embedding_matrix.shape[0] != len(papers):
        raise ValueError("embeddings row count must match number of papers.")

    paper_ids = [_paper_id(paper) for paper in papers]
    index_by_id = {pid: idx for idx, pid in enumerate(paper_ids)}

    parent_adjacency = _normalize_parent_adjacency(dag)

    results: Dict[str, PaperInheritanceFit] = {}
    for focal_idx, focal_paper_id in enumerate(paper_ids):
        parent_ids = [pid for pid in parent_adjacency.get(focal_paper_id, ()) if pid in index_by_id]
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
