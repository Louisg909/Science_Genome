"""Inheritance decomposition for paper embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


Constraint = Literal["capped_simplex", "simplex", "nonnegative"]


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
    """Compute residual-derived inheritance metrics."""
    residual_arr = np.asarray(residual, dtype=float).reshape(-1)
    weights_arr = np.asarray(weights, dtype=float).reshape(-1)

    residual_norm = float(np.linalg.norm(residual_arr, ord=2))
    residual_direction = residual_arr / residual_norm if residual_norm > 0.0 else np.zeros_like(residual_arr)

    weight_sum = float(weights_arr.sum())
    denominator = weight_sum + residual_norm
    novelty_score = residual_norm / denominator if denominator > 0.0 else 0.0
    return {
        "residual_norm": residual_norm,
        "residual_direction": residual_direction,
        "novelty_score": novelty_score,
    }


def attach_corpus_inheritance_metrics(corpus_outputs: list[dict]) -> list[dict]:
    """Attach residual metrics to corpus-level inheritance records."""
    enriched_outputs: list[dict] = []
    for record in corpus_outputs:
        metrics = residual_metrics(record["residual"], record["weights"])
        enriched_outputs.append({**record, **metrics})
    return enriched_outputs


def _project_to_simplex(vector: np.ndarray) -> np.ndarray:
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
    return np.maximum(vector - theta, 0.0)


def _project_to_capped_simplex(vector: np.ndarray) -> np.ndarray:
    nonnegative = np.maximum(vector, 0.0)
    return nonnegative if nonnegative.sum() <= 1.0 else _project_to_simplex(vector)


def _apply_sparsity(weights: np.ndarray, sparsity: Optional[int], constraint: Constraint) -> np.ndarray:
    if sparsity is None or sparsity <= 0 or sparsity >= weights.size:
        return weights
    keep_idx = np.argpartition(weights, -sparsity)[-sparsity:]
    sparse_weights = np.zeros_like(weights)
    sparse_weights[keep_idx] = weights[keep_idx]
    if constraint == "simplex":
        total = sparse_weights.sum()
        return sparse_weights / total if total > 0 else np.full_like(weights, 1.0 / weights.size)
    return sparse_weights


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / safe_norms
    return normalized.T @ normalized


def _prune_redundant_parents(parent_matrix: np.ndarray, redundancy_threshold: Optional[float]) -> np.ndarray:
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


def fit_weights(
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
    """Stage 1 (Inheritance inference): solve

    w_i^* = argmin_w ||p_i - P_i w||_2^2 + lambda ||w||_2^2

    subject to w>=0 and either sum(w)=1 (simplex), sum(w)<=1 (capped_simplex),
    or unconstrained nonnegative. Matches method.tex inheritance terminology.
    """
    num_parents = parents.shape[1]
    weights = np.full(num_parents, 1.0 / num_parents) if constraint == "simplex" else np.zeros(num_parents, dtype=float)

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        weights = weights + 1e-12 * rng.standard_normal(num_parents)

    gram = parents.T @ parents
    linear = parents.T @ target
    converged = False
    for iteration in range(1, max_iter + 1):
        gradient = gram @ weights - linear + l2_regularizer * weights
        candidate = weights - learning_rate * gradient
        if constraint == "simplex":
            candidate = _project_to_simplex(candidate)
        elif constraint == "capped_simplex":
            candidate = _project_to_capped_simplex(candidate)
        else:
            candidate = np.maximum(candidate, 0.0)
        candidate = _apply_sparsity(candidate, sparsity, constraint)

        if np.linalg.norm(candidate - weights, ord=2) <= tolerance:
            converged = True
            weights = candidate
            break
        weights = candidate

    return weights, converged, iteration


def compute_residual(target: np.ndarray, parents: np.ndarray, weights: np.ndarray, l2_regularizer: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Stage 2 (Novel semantic contribution): compute residual r_i = p_i - P_i w_i^*.

    Returns (reconstruction, residual, objective) where objective is
    1/2 ||r_i||_2^2 + 1/2 lambda ||w_i^*||_2^2.
    """
    reconstruction = parents @ weights
    residual = target - reconstruction
    objective = 0.5 * float(np.dot(residual, residual)) + 0.5 * l2_regularizer * float(np.dot(weights, weights))
    return reconstruction, residual, objective


def bootstrap_weight_uncertainty(
    *,
    target: np.ndarray,
    parents: np.ndarray,
    active_idx: np.ndarray,
    point_weights: np.ndarray,
    constraint: Constraint,
    sparsity: Optional[int],
    l2_regularizer: float,
    max_iter: int,
    learning_rate: float,
    tolerance: float,
    random_state: Optional[int],
    bootstrap_samples: int,
    bootstrap_ci: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stage 3 (Handling non-orthogonal parents): bootstrap parent subsets.

    Re-estimate w_ij over bootstrap-resampled parent subsets to report median,
    CI quantiles, and selection-frequency robustness diagnostics.
    """
    num_parents = parents.shape[1]
    if bootstrap_samples <= 0:
        active = (point_weights > 0).astype(float)
        return point_weights.copy(), point_weights.copy(), point_weights.copy(), active

    low_q, high_q = bootstrap_ci
    if not 0.0 <= low_q < high_q <= 1.0:
        raise ValueError("bootstrap_ci must be valid quantiles")

    rng = np.random.default_rng(random_state)
    samples = np.zeros((bootstrap_samples + 1, num_parents), dtype=float)
    samples[0] = point_weights
    selection_counts = (point_weights > 0).astype(float)

    for b in range(bootstrap_samples):
        draw = rng.choice(active_idx, size=active_idx.size, replace=True)
        unique, first_pos = np.unique(draw, return_index=True)
        subset = unique[np.argsort(first_pos)]
        sub_weights, _, _ = fit_weights(
            target,
            parents[:, subset],
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
        samples[b + 1] = expanded
        selection_counts += (expanded > 0).astype(float)

    return (
        np.median(samples, axis=0),
        np.quantile(samples, low_q, axis=0),
        np.quantile(samples, high_q, axis=0),
        selection_counts / float(bootstrap_samples + 1),
    )


def compute_parent_attribution(
    *,
    target: np.ndarray,
    parents: np.ndarray,
    constraint: Constraint,
    sparsity: Optional[int],
    l2_regularizer: float,
    max_iter: int,
    learning_rate: float,
    tolerance: float,
    random_state: Optional[int],
    shapley_exact_threshold: int,
    shapley_monte_carlo_samples: int,
    compute_shapley: bool,
) -> np.ndarray:
    """Stage 4 (Parent contribution): compute Shapley attribution phi_ij."""
    if not compute_shapley:
        return np.zeros(parents.shape[1], dtype=float)
    from .shapley import estimate_shapley_contributions

    return estimate_shapley_contributions(
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


def solve_inheritance(target_embedding: np.ndarray, parent_matrix: np.ndarray, **kwargs) -> InheritanceResult:
    """Thin orchestrator for the four-stage inheritance pipeline."""
    constraint: Constraint = kwargs.get("constraint", "capped_simplex")
    sparsity: Optional[int] = kwargs.get("sparsity", None)
    l2_regularizer: float = kwargs.get("l2_regularizer", 1e-6)
    max_iter: int = kwargs.get("max_iter", 5_000)
    learning_rate: float = kwargs.get("learning_rate", 0.05)
    tolerance: float = kwargs.get("tolerance", 1e-10)
    random_state: Optional[int] = kwargs.get("random_state", None)
    shapley_exact_threshold: int = kwargs.get("shapley_exact_threshold", 8)
    shapley_monte_carlo_samples: int = kwargs.get("shapley_monte_carlo_samples", 256)
    compute_shapley: bool = kwargs.get("compute_shapley", True)
    redundancy_threshold: Optional[float] = kwargs.get("redundancy_threshold", None)
    bootstrap_samples: int = kwargs.get("bootstrap_samples", 0)
    bootstrap_ci: tuple[float, float] = kwargs.get("bootstrap_ci", (0.05, 0.95))

    if constraint not in {"capped_simplex", "simplex", "nonnegative"}:
        raise ValueError("constraint must be one of 'capped_simplex', 'simplex', or 'nonnegative'")

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
        return InheritanceResult(zeros, residual, np.zeros_like(target), 0.5 * float(np.dot(residual, residual)), True, 0, np.zeros(0, dtype=float), zeros, zeros, zeros, zeros, active_parent_mask)

    active_weights, converged, iteration = fit_weights(
        target, active_parents, constraint=constraint, sparsity=sparsity, l2_regularizer=l2_regularizer,
        max_iter=max_iter, learning_rate=learning_rate, tolerance=tolerance, random_state=random_state,
    )
    weights = np.zeros(num_parents, dtype=float)
    weights[active_idx] = active_weights

<<<<<< bryani/add-validation-module-and-tests
    weights = np.zeros(num_parents, dtype=float)
    weights[active_idx] = active_weights

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        weights = weights + 1e-12 * rng.standard_normal(num_parents)
        if constraint == "simplex":
            weights = _project_to_simplex(weights)
        elif constraint == "capped_simplex":
            weights = _project_to_capped_simplex(weights)
        else:
            weights = np.maximum(weights, 0.0)
======
<<<<<< bryani/fix-inheritance-logic-and-add-tests
    weights = np.zeros(num_parents, dtype=float)
    weights[active_idx] = active_weights

>>>>>> main
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
======
    weight_median, weight_ci_lower, weight_ci_upper, selection_frequency = bootstrap_weight_uncertainty(
        target=target, parents=parents, active_idx=active_idx, point_weights=weights, constraint=constraint,
        sparsity=sparsity, l2_regularizer=l2_regularizer, max_iter=max_iter, learning_rate=learning_rate,
        tolerance=tolerance, random_state=random_state, bootstrap_samples=bootstrap_samples, bootstrap_ci=bootstrap_ci,
    )
    shapley_contributions = compute_parent_attribution(
        target=target, parents=parents, constraint=constraint, sparsity=sparsity, l2_regularizer=l2_regularizer,
        max_iter=max_iter, learning_rate=learning_rate, tolerance=tolerance, random_state=random_state,
        shapley_exact_threshold=shapley_exact_threshold, shapley_monte_carlo_samples=shapley_monte_carlo_samples,
        compute_shapley=compute_shapley,
    )
    reconstruction, residual, objective = compute_residual(target, parents, weights, l2_regularizer)
>>>>> main

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
