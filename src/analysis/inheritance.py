"""Inheritance decomposition for paper embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


Constraint = Literal["simplex", "nonnegative"]


@dataclass(frozen=True)
class InheritanceResult:
    """Output for an inheritance decomposition."""

    weights: np.ndarray
    residual: np.ndarray
    reconstruction: np.ndarray
    objective: float
    converged: bool
    iterations: int
    shapley_contributions: np.ndarray
    weight_median: np.ndarray
    weight_ci_lower: np.ndarray
    weight_ci_upper: np.ndarray
    selection_frequency: np.ndarray
    active_parent_mask: np.ndarray


def residual_metrics(residual: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute residual-derived inheritance metrics.

    Returns:
        residual_norm: L2 norm of residual.
        residual_direction: Unit vector in residual direction (zeros when norm=0).
        novelty_score: nu_i = ||r_i|| / (sum(w_i) + ||r_i||).
    """

    residual_arr = np.asarray(residual, dtype=float).reshape(-1)
    weights_arr = np.asarray(weights, dtype=float).reshape(-1)

    residual_norm = float(np.linalg.norm(residual_arr, ord=2))
    if residual_norm > 0.0:
        residual_direction = residual_arr / residual_norm
    else:
        residual_direction = np.zeros_like(residual_arr)

    weight_sum = float(weights_arr.sum())
    denominator = weight_sum + residual_norm
    novelty_score = residual_norm / denominator if denominator > 0.0 else 0.0

    return {
        "residual_norm": residual_norm,
        "residual_direction": residual_direction,
        "novelty_score": novelty_score,
    }


def attach_corpus_inheritance_metrics(
    corpus_outputs: list[dict],
) -> list[dict]:
    """Attach residual metrics to corpus-level inheritance records.

    Each record must provide `weights` and `residual` arrays.
    """

    enriched_outputs: list[dict] = []
    for record in corpus_outputs:
        metrics = residual_metrics(record["residual"], record["weights"])
        enriched_outputs.append({**record, **metrics})
    return enriched_outputs


def _project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Project vector onto the probability simplex."""

    if vector.size == 0:
        return vector
    sorted_vec = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_vec)
    rho_candidates = sorted_vec + (1.0 - cumulative) / (np.arange(vector.size) + 1)
    rho = np.nonzero(rho_candidates > 0)[0]
    if rho.size == 0:
        return np.full_like(vector, 1.0 / vector.size)
    rho_idx = rho[-1]
    theta = (cumulative[rho_idx] - 1.0) / (rho_idx + 1)
    projected = np.maximum(vector - theta, 0.0)
    return projected


def _apply_sparsity(weights: np.ndarray, sparsity: Optional[int], constraint: Constraint) -> np.ndarray:
    """Keep only top-k entries when sparsity is requested."""

    if sparsity is None or sparsity <= 0 or sparsity >= weights.size:
        return weights

    keep_idx = np.argpartition(weights, -sparsity)[-sparsity:]
    sparse_weights = np.zeros_like(weights)
    sparse_weights[keep_idx] = weights[keep_idx]

    if constraint == "simplex":
        total = sparse_weights.sum()
        if total > 0:
            sparse_weights /= total
        else:
            sparse_weights = np.full_like(weights, 1.0 / weights.size)
    return sparse_weights


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / safe_norms
    return normalized.T @ normalized


def _prune_redundant_parents(parent_matrix: np.ndarray, redundancy_threshold: Optional[float]) -> np.ndarray:
    """Return a boolean mask of active parents after greedy redundancy pruning."""

    num_parents = parent_matrix.shape[1]
    if redundancy_threshold is None:
        return np.ones(num_parents, dtype=bool)
    if not 0.0 <= redundancy_threshold <= 1.0:
        raise ValueError("redundancy_threshold must be in [0, 1]")

    cosine = _cosine_similarity_matrix(parent_matrix)
    keep = np.ones(num_parents, dtype=bool)
    for j in range(num_parents):
        if not keep[j]:
            continue
        for k in range(j + 1, num_parents):
            if keep[k] and cosine[j, k] >= redundancy_threshold:
                keep[k] = False
    return keep


def _solve_weights(
    target: np.ndarray,
    parents: np.ndarray,
    *,
    constraint: Constraint,
    sparsity: Optional[int],
    l2_regularizer: float,
    max_iter: int,
    learning_rate: float,
    tolerance: float,
    random_state: Optional[int],
) -> tuple[np.ndarray, bool, int]:
    num_parents = parents.shape[1]
    if constraint == "simplex":
        weights = np.full(num_parents, 1.0 / num_parents)
    else:
        weights = np.zeros(num_parents, dtype=float)

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        weights = weights + 1e-12 * rng.standard_normal(num_parents)
        if constraint == "simplex":
            weights = _project_to_simplex(weights)
        else:
            weights = np.maximum(weights, 0.0)

    gram = parents.T @ parents
    linear = parents.T @ target
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        gradient = gram @ weights - linear + l2_regularizer * weights
        candidate = weights - learning_rate * gradient

        if constraint == "simplex":
            candidate = _project_to_simplex(candidate)
        else:
            candidate = np.maximum(candidate, 0.0)

        candidate = _apply_sparsity(candidate, sparsity, constraint)

        delta = np.linalg.norm(candidate - weights, ord=2)
        weights = candidate
        if delta <= tolerance:
            converged = True
            break

    return weights, converged, iteration


def solve_inheritance(
    target_embedding: np.ndarray,
    parent_matrix: np.ndarray,
    *,
    constraint: Constraint = "simplex",
    sparsity: Optional[int] = None,
    l2_regularizer: float = 1e-6,
    max_iter: int = 5_000,
    learning_rate: float = 0.05,
    tolerance: float = 1e-10,
    random_state: Optional[int] = None,
    shapley_exact_threshold: int = 8,
    shapley_monte_carlo_samples: int = 256,
    compute_shapley: bool = True,
    redundancy_threshold: Optional[float] = None,
    bootstrap_samples: int = 0,
    bootstrap_ci: tuple[float, float] = (0.05, 0.95),
) -> InheritanceResult:
    """Solve e_i ≈ P_i w_i under simplex/non-negative constraints.

    Optional parent pruning removes near-duplicate parent embeddings before fit.
    Optional bootstrap resampling over parent subsets yields uncertainty summaries.
    """

    if constraint not in {"simplex", "nonnegative"}:
        raise ValueError("constraint must be either 'simplex' or 'nonnegative'")

    target = np.asarray(target_embedding, dtype=float).reshape(-1)
    parents = np.asarray(parent_matrix, dtype=float)
    if parents.ndim == 1:
        parents = parents.reshape(-1, 1)
    if parents.shape[0] != target.shape[0]:
        raise ValueError("parent_matrix rows must match target embedding size")

    num_parents = parents.shape[1]
    active_parent_mask = _prune_redundant_parents(parents, redundancy_threshold) if num_parents > 0 else np.zeros(0, dtype=bool)
    active_idx = np.flatnonzero(active_parent_mask)
    active_parents = parents[:, active_idx] if active_idx.size > 0 else np.empty((target.shape[0], 0))

    if active_parents.shape[1] == 0:
        residual = target.copy()
        zeros = np.zeros(num_parents, dtype=float)
        return InheritanceResult(
            weights=zeros,
            residual=residual,
            reconstruction=np.zeros_like(target),
            objective=0.5 * float(np.dot(residual, residual)),
            converged=True,
            iterations=0,
            shapley_contributions=np.zeros(0, dtype=float),
            weight_median=zeros,
            weight_ci_lower=zeros,
            weight_ci_upper=zeros,
            selection_frequency=zeros,
            active_parent_mask=active_parent_mask,
        )

    active_weights, converged, iteration = _solve_weights(
        target,
        active_parents,
        constraint=constraint,
        sparsity=sparsity,
        l2_regularizer=l2_regularizer,
        max_iter=max_iter,
        learning_rate=learning_rate,
        tolerance=tolerance,
        random_state=random_state,
    )

    weights = np.zeros(num_parents, dtype=float)
    weights[active_idx] = active_weights

    samples = np.zeros((max(bootstrap_samples, 1), num_parents), dtype=float)
    samples[0] = weights
    selection_counts = (weights > 0).astype(float)

    if bootstrap_samples > 0:
        low_q, high_q = bootstrap_ci
        if not 0.0 <= low_q < high_q <= 1.0:
            raise ValueError("bootstrap_ci must be valid quantiles")
        rng = np.random.default_rng(random_state)
        for b in range(bootstrap_samples):
            draw = rng.choice(active_idx, size=active_idx.size, replace=True)
            unique, first_pos = np.unique(draw, return_index=True)
            subset = unique[np.argsort(first_pos)]
            sub_parents = parents[:, subset]
            sub_weights, _, _ = _solve_weights(
                target,
                sub_parents,
                constraint=constraint,
                sparsity=sparsity,
                l2_regularizer=l2_regularizer,
                max_iter=max_iter,
                learning_rate=learning_rate,
                tolerance=tolerance,
                random_state=None if random_state is None else random_state + b + 1,
            )
            expanded = np.zeros(num_parents, dtype=float)
            expanded[subset] = sub_weights
            samples[b] = expanded
            selection_counts += (expanded > 0).astype(float)
        weight_median = np.median(samples, axis=0)
        weight_ci_lower = np.quantile(samples, low_q, axis=0)
        weight_ci_upper = np.quantile(samples, high_q, axis=0)
        selection_frequency = selection_counts / float(bootstrap_samples + 1)
    else:
        weight_median = weights.copy()
        weight_ci_lower = weights.copy()
        weight_ci_upper = weights.copy()
        selection_frequency = (weights > 0).astype(float)


    shapley_contributions = np.zeros(num_parents, dtype=float)
    if compute_shapley:
        from .shapley import estimate_shapley_contributions

        shapley_contributions = estimate_shapley_contributions(
            target_embedding=target,
            parent_matrix=parents,
            constraint=constraint,
            sparsity=sparsity,
            l2_regularizer=l2_regularizer,
            max_iter=max_iter,
            learning_rate=learning_rate,
            tolerance=tolerance,
            exact_threshold=shapley_exact_threshold,
            monte_carlo_samples=shapley_monte_carlo_samples,
            random_state=random_state,
        )

    reconstruction = parents @ weights
    residual = target - reconstruction
    objective = 0.5 * float(np.dot(residual, residual)) + 0.5 * l2_regularizer * float(np.dot(weights, weights))

    return InheritanceResult(
        weights=weights,
        residual=residual,
        reconstruction=reconstruction,
        objective=objective,
        converged=converged,
        iterations=iteration,
        shapley_contributions=shapley_contributions,
        weight_median=weight_median,
        weight_ci_lower=weight_ci_lower,
        weight_ci_upper=weight_ci_upper,
        selection_frequency=selection_frequency,
        active_parent_mask=active_parent_mask,
    )
