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
) -> InheritanceResult:
    """Solve e_i ≈ P_i w_i under simplex/non-negative constraints.

    The optimizer uses projected gradient descent with deterministic updates.
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
    if num_parents == 0:
        residual = target.copy()
        return InheritanceResult(
            weights=np.zeros(0, dtype=float),
            residual=residual,
            reconstruction=np.zeros_like(target),
            objective=0.5 * float(np.dot(residual, residual)),
            converged=True,
            iterations=0,
        )

    # Deterministic initialization: seed only controls tiny perturbation in rare flat regions.
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

    reconstruction = parents @ weights
    residual = target - reconstruction
    objective = 0.5 * float(np.dot(residual, residual)) + 0.5 * l2_regularizer * float(
        np.dot(weights, weights)
    )

    return InheritanceResult(
        weights=weights,
        residual=residual,
        reconstruction=reconstruction,
        objective=objective,
        converged=converged,
        iterations=iteration,
    )
