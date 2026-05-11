import numpy as np
import pytest

from src.analysis import attach_corpus_inheritance_metrics, residual_metrics, solve_inheritance


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


def test_residual_metrics_zero_residual_case():
    metrics = residual_metrics(np.zeros(3), np.array([0.4, 0.6]))

    assert metrics["residual_norm"] == pytest.approx(0.0)
    assert np.allclose(metrics["residual_direction"], np.zeros(3))
    assert metrics["novelty_score"] == pytest.approx(0.0)


def test_residual_metrics_no_parent_case():
    target = np.array([0.2, -0.5, 1.7])
    result = solve_inheritance(target, np.empty((3, 0)))
    metrics = residual_metrics(result.residual, result.weights)

    assert result.weights.size == 0
    assert metrics["residual_norm"] == pytest.approx(np.linalg.norm(target))
    assert np.allclose(metrics["residual_direction"], target / np.linalg.norm(target))
    assert metrics["novelty_score"] == pytest.approx(1.0)


def test_novelty_score_monotonic_in_residual_norm():
    weights = np.array([0.2, 0.3])
    low = residual_metrics(np.array([0.2, 0.0]), weights)
    high = residual_metrics(np.array([0.8, 0.0]), weights)

    assert low["residual_norm"] < high["residual_norm"]
    assert low["novelty_score"] < high["novelty_score"]


def test_attach_corpus_inheritance_metrics_adds_fields():
    records = [
        {"paper_id": "p1", "weights": np.array([1.0]), "residual": np.array([0.0, 0.0])},
        {"paper_id": "p2", "weights": np.array([]), "residual": np.array([1.0, 0.0])},
    ]
    enriched = attach_corpus_inheritance_metrics(records)

    assert len(enriched) == 2
    for row in enriched:
        assert "residual_norm" in row
        assert "residual_direction" in row
        assert "novelty_score" in row
