"""Validation and verification orchestration for inheritance analyses."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .inheritance import InheritanceResult
from .lineage_metrics import build_lineage_analytics
from .structure import embedding_norm_diagnostics, nearest_neighbor_density


def _coerce_inheritance_result(result: Any) -> InheritanceResult:
    if isinstance(result, InheritanceResult):
        return result
    raise TypeError("inheritance_result must be an InheritanceResult instance")


def _solver_summary(inheritance_result: InheritanceResult, parent_matrix: np.ndarray) -> dict[str, Any]:
    parents = np.asarray(parent_matrix, dtype=float)
    if parents.ndim == 1:
        parents = parents.reshape(-1, 1)
    if parents.ndim != 2:
        raise ValueError("parent_matrix must be a 2D matrix")

    if parents.size == 0 or parents.shape[1] == 0:
        condition_number = float("nan")
        rank = 0
    else:
        gram = parents.T @ parents
        condition_number = float(np.linalg.cond(gram))
        rank = int(np.linalg.matrix_rank(parents))

    return {
        "converged": bool(inheritance_result.converged),
        "iterations": int(inheritance_result.iterations),
        "objective": float(inheritance_result.objective),
        "active_parent_count": int(np.sum(inheritance_result.active_parent_mask)),
        "parent_rank": rank,
        "gram_condition_number": condition_number,
    }


def _uncertainty_summary(inheritance_result: InheritanceResult) -> dict[str, Any]:
    ci_width = np.asarray(inheritance_result.weight_ci_upper) - np.asarray(inheritance_result.weight_ci_lower)
    selection_frequency = np.asarray(inheritance_result.selection_frequency)
    return {
        "ci_width": ci_width.tolist(),
        "ci_width_mean": float(np.mean(ci_width)) if ci_width.size else 0.0,
        "ci_width_max": float(np.max(ci_width)) if ci_width.size else 0.0,
        "selection_frequency": selection_frequency.tolist(),
        "selection_frequency_mean": float(np.mean(selection_frequency)) if selection_frequency.size else 0.0,
    }


def _lineage_sanity_checks(lineage_records: list[Mapping[str, Any]]) -> dict[str, Any]:
    lineage_summary = build_lineage_analytics(lineage_records)
    rows = lineage_summary.get("rows", [])
    violations = [r["paper_id"] for r in rows if r["effective_parent_count"] > r["parent_count"]]
    entropy_violations = [r["paper_id"] for r in rows if r["lineage_entropy"] < 0.0]
    return {
        "paper_count": int(len(rows)),
        "effective_parent_count_violations": violations,
        "negative_entropy_violations": entropy_violations,
        "all_checks_pass": not violations and not entropy_violations,
        "summary": lineage_summary,
    }


def generate_validation_report(
    *,
    embeddings: np.ndarray,
    inheritance_result: InheritanceResult,
    parent_matrix: np.ndarray,
    lineage_records: list[Mapping[str, Any]],
    n_neighbors: int = 5,
) -> dict[str, Any]:
    """Generate a deterministic structured V&V report for reproducible analyses."""

    resolved_result = _coerce_inheritance_result(inheritance_result)
    return {
        "embedding_diagnostics": {
            "norms": embedding_norm_diagnostics(embeddings),
            "neighbor_density": nearest_neighbor_density(embeddings, n_neighbors=n_neighbors),
        },
        "solver": _solver_summary(resolved_result, parent_matrix),
        "uncertainty": _uncertainty_summary(resolved_result),
        "lineage_sanity": _lineage_sanity_checks(lineage_records),
    }
