import numpy as np
import pytest

from src.analysis import solve_inheritance


def test_collinear_parents_prefers_stable_simplex_solution():
    target = np.array([1.0, 0.0])
    parents = np.array([[1.0, 2.0], [0.0, 0.0]])  # perfectly collinear

    result = solve_inheritance(target, parents, constraint="simplex", random_state=123)

    assert result.converged
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(result.weights >= -1e-10)
    # Collinear parents still admit a low-residual fit in this setup.
    assert np.linalg.norm(result.residual) <= 1e-3


def test_no_citation_edge_case_returns_full_residual():
    target = np.array([0.2, -0.5, 1.7])
    parents = np.empty((3, 0))

    result = solve_inheritance(target, parents)

    assert result.converged
    assert result.weights.size == 0
    assert np.allclose(result.reconstruction, np.zeros_like(target))
    assert np.allclose(result.residual, target)


def test_stability_across_random_seeds():
    target = np.array([1.0, 2.0, -1.0])
    parents = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.3],
            [0.2, 0.0, 1.0],
        ]
    )

    results = [
        solve_inheritance(target, parents, constraint="simplex", random_state=seed)
        for seed in (1, 17, 999)
    ]

    for r in results[1:]:
        assert np.allclose(r.weights, results[0].weights, atol=1e-6)
        assert np.allclose(r.residual, results[0].residual, atol=1e-6)


def test_interpretability_checks_weight_sum_and_residual_behavior():
    target = np.array([0.6, 0.4])
    parents = np.array([[1.0, 0.0], [0.0, 1.0]])

    full = solve_inheritance(target, parents, constraint="simplex")
    sparse = solve_inheritance(target, parents, constraint="simplex", sparsity=1)

    assert full.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert sparse.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.count_nonzero(sparse.weights > 1e-12) == 1
    assert np.linalg.norm(sparse.residual) >= np.linalg.norm(full.residual) - 1e-9
