import numpy as np
import pytest

from src.analysis import solve_inheritance


def test_correlated_parents_instability_revealed_by_bootstrap():
    target = np.array([1.0, 0.95, 0.05])
    parents = np.array(
        [
            [1.0, 0.99, 0.0],
            [0.95, 0.94, 0.0],
            [0.0, 0.01, 1.0],
        ]
    )

    no_prune = solve_inheritance(
        target,
        parents,
        constraint="simplex",
        bootstrap_samples=120,
        random_state=7,
    )
    pruned = solve_inheritance(
        target,
        parents,
        constraint="simplex",
        redundancy_threshold=0.999,
        bootstrap_samples=120,
        random_state=7,
    )

    assert no_prune.active_parent_mask.sum() == 3
    assert pruned.active_parent_mask.sum() == 2
    # Correlated parents should have broad uncertainty pre-pruning.
    assert (no_prune.weight_ci_upper[0] - no_prune.weight_ci_lower[0]) > 0.05


def test_bootstrap_outputs_shape_and_consistency():
    target = np.array([0.6, 0.3, 0.1])
    parents = np.eye(3)

    result = solve_inheritance(
        target,
        parents,
        constraint="simplex",
        bootstrap_samples=50,
        random_state=42,
    )

    n = parents.shape[1]
    assert result.weight_median.shape == (n,)
    assert result.weight_ci_lower.shape == (n,)
    assert result.weight_ci_upper.shape == (n,)
    assert result.selection_frequency.shape == (n,)
    assert np.all(result.weight_ci_lower <= result.weight_median + 1e-12)
    assert np.all(result.weight_median <= result.weight_ci_upper + 1e-12)
    assert np.all((0.0 <= result.selection_frequency) & (result.selection_frequency <= 1.0))


def test_bootstrap_reproducibility_with_seed_control():
    target = np.array([1.0, 2.0, -1.0])
    parents = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.3],
            [0.2, 0.0, 1.0],
        ]
    )

    r1 = solve_inheritance(target, parents, constraint="simplex", bootstrap_samples=80, random_state=99)
    r2 = solve_inheritance(target, parents, constraint="simplex", bootstrap_samples=80, random_state=99)
    r3 = solve_inheritance(target, parents, constraint="simplex", bootstrap_samples=80, random_state=100)

    assert np.allclose(r1.weights, r2.weights)
    assert np.allclose(r1.weight_median, r2.weight_median)
    assert np.allclose(r1.weight_ci_lower, r2.weight_ci_lower)
    assert np.allclose(r1.weight_ci_upper, r2.weight_ci_upper)
    assert np.allclose(r1.selection_frequency, r2.selection_frequency)
    assert not np.allclose(r1.selection_frequency, r3.selection_frequency)


def test_no_citation_edge_case_returns_full_residual():
    target = np.array([0.2, -0.5, 1.7])
    parents = np.empty((3, 0))

    result = solve_inheritance(target, parents)

    assert result.converged
    assert result.weights.size == 0
    assert np.allclose(result.reconstruction, np.zeros_like(target))
    assert np.allclose(result.residual, target)


def test_interpretability_checks_weight_sum_and_residual_behavior():
    target = np.array([0.6, 0.4])
    parents = np.array([[1.0, 0.0], [0.0, 1.0]])

    full = solve_inheritance(target, parents, constraint="simplex")
    sparse = solve_inheritance(target, parents, constraint="simplex", sparsity=1)

    assert full.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert sparse.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.count_nonzero(sparse.weights > 1e-12) == 1
    assert np.linalg.norm(sparse.residual) >= np.linalg.norm(full.residual) - 1e-9
