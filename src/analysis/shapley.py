"""Shapley attribution for inheritance reconstruction utility."""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Literal, Optional

import numpy as np

from .inheritance import Constraint, solve_inheritance


def reconstruction_utility(
    target_embedding: np.ndarray,
    parent_matrix: np.ndarray,
    coalition: np.ndarray,
    *,
    constraint: Constraint = "simplex",
    sparsity: Optional[int] = None,
    l2_regularizer: float = 1e-6,
    max_iter: int = 5_000,
    learning_rate: float = 0.05,
    tolerance: float = 1e-10,
    compute_shapley: bool = False,
) -> float:
    """Compute utility V_i(S)=1-||p_i-sum_j w_ij p_j||^2 for coalition S."""

    target = np.asarray(target_embedding, dtype=float).reshape(-1)
    parents = np.asarray(parent_matrix, dtype=float)
    if parents.ndim == 1:
        parents = parents.reshape(-1, 1)

    active = np.asarray(coalition, dtype=bool)
    if active.size != parents.shape[1]:
        raise ValueError("coalition mask size must match number of parents")

    coalition_parents = parents[:, active]
    result = solve_inheritance(
        target,
        coalition_parents,
        constraint=constraint,
        sparsity=sparsity,
        l2_regularizer=l2_regularizer,
        max_iter=max_iter,
        learning_rate=learning_rate,
        tolerance=tolerance,
        compute_shapley=False,
    )
    residual_norm_sq = float(np.dot(result.residual, result.residual))
    return 1.0 - residual_norm_sq


def estimate_shapley_contributions(
    target_embedding: np.ndarray,
    parent_matrix: np.ndarray,
    *,
    constraint: Constraint = "simplex",
    sparsity: Optional[int] = None,
    l2_regularizer: float = 1e-6,
    max_iter: int = 5_000,
    learning_rate: float = 0.05,
    tolerance: float = 1e-10,
    exact_threshold: int = 8,
    monte_carlo_samples: int = 256,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Estimate Shapley contributions for cited parents.

    Uses exact subset enumeration for parent count <= ``exact_threshold`` and
    permutation Monte Carlo otherwise.
    """

    parents = np.asarray(parent_matrix, dtype=float)
    if parents.ndim == 1:
        parents = parents.reshape(-1, 1)
    n = parents.shape[1]
    if n == 0:
        return np.zeros(0, dtype=float)

    kwargs = dict(
        target_embedding=target_embedding,
        parent_matrix=parents,
        constraint=constraint,
        sparsity=sparsity,
        l2_regularizer=l2_regularizer,
        max_iter=max_iter,
        learning_rate=learning_rate,
        tolerance=tolerance,
        compute_shapley=False,
    )

    if n <= exact_threshold:
        contributions = np.zeros(n, dtype=float)
        n_fact = factorial(n)
        for i in range(n):
            others = [j for j in range(n) if j != i]
            for k in range(n):
                weight = factorial(k) * factorial(n - k - 1) / n_fact
                for subset in combinations(others, k):
                    mask_without = np.zeros(n, dtype=bool)
                    mask_without[list(subset)] = True
                    mask_with = mask_without.copy()
                    mask_with[i] = True
                    v_without = reconstruction_utility(coalition=mask_without, **kwargs)
                    v_with = reconstruction_utility(coalition=mask_with, **kwargs)
                    contributions[i] += weight * (v_with - v_without)
        return contributions

    rng = np.random.default_rng(random_state)
    contributions = np.zeros(n, dtype=float)
    samples = max(1, int(monte_carlo_samples))

    for _ in range(samples):
        perm = rng.permutation(n)
        mask = np.zeros(n, dtype=bool)
        prev_utility = reconstruction_utility(coalition=mask, **kwargs)
        for idx in perm:
            mask[idx] = True
            new_utility = reconstruction_utility(coalition=mask, **kwargs)
            contributions[idx] += new_utility - prev_utility
            prev_utility = new_utility

    return contributions / samples
