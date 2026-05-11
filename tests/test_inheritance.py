import numpy as np
import pytest

from src.analysis import attach_corpus_inheritance_metrics, residual_metrics, solve_inheritance
from src.analysis.inheritance import bootstrap_weight_uncertainty, compute_parent_attribution, compute_residual, fit_weights


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

    assert full.weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert sparse.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.count_nonzero(sparse.weights > 1e-12) == 1
    assert np.linalg.norm(sparse.residual) >= np.linalg.norm(full.residual) - 1e-9


def test_capped_simplex_default_allows_novelty_residual_mass():
    target = np.array([0.1, 0.1])
    parents = np.array([[1.0, 0.0], [0.0, 1.0]])

    capped = solve_inheritance(target, parents)

    assert capped.converged
    assert np.all(capped.weights >= -1e-10)
    assert capped.weights.sum() < 1.0
    assert capped.weights.sum() == pytest.approx(0.2, abs=1e-5)


def test_capped_simplex_matches_simplex_when_simplex_is_optimal():
    target = np.array([0.7, 0.3])
    parents = np.array([[1.0, 0.0], [0.0, 1.0]])

    capped = solve_inheritance(target, parents, constraint="capped_simplex")
    simplex = solve_inheritance(target, parents, constraint="simplex")

    assert np.all(capped.weights >= -1e-10)
    assert np.all(simplex.weights >= -1e-10)
    assert capped.weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert simplex.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.allclose(capped.weights, simplex.weights, atol=1e-6)
    assert np.allclose(capped.residual, simplex.residual, atol=1e-6)
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


def test_fit_weights_simplex_recovers_identity_target():
    target = np.array([0.8, 0.2])
    parents = np.eye(2)
    weights, converged, _ = fit_weights(
        target,
        parents,
        constraint="simplex",
        sparsity=None,
        l2_regularizer=1e-6,
        max_iter=2000,
        learning_rate=0.1,
        tolerance=1e-12,
        random_state=0,
    )
    assert converged
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.allclose(weights, target, atol=1e-3)


def test_compute_residual_matches_definition():
    target = np.array([1.0, 0.0])
    parents = np.eye(2)
    weights = np.array([0.75, 0.25])
    reconstruction, residual, objective = compute_residual(target, parents, weights, l2_regularizer=0.1)

    assert np.allclose(reconstruction, np.array([0.75, 0.25]))
    assert np.allclose(residual, np.array([0.25, -0.25]))
    expected = 0.5 * np.dot(residual, residual) + 0.5 * 0.1 * np.dot(weights, weights)
    assert objective == pytest.approx(expected)


def test_bootstrap_weight_uncertainty_returns_point_estimate_without_sampling():
    target = np.array([0.6, 0.4])
    parents = np.eye(2)
    point = np.array([0.6, 0.4])
    stats = bootstrap_weight_uncertainty(
        target=target,
        parents=parents,
        active_idx=np.array([0, 1]),
        point_weights=point,
        constraint="simplex",
        sparsity=None,
        l2_regularizer=1e-6,
        max_iter=100,
        learning_rate=0.1,
        tolerance=1e-9,
        random_state=1,
        bootstrap_samples=0,
        bootstrap_ci=(0.05, 0.95),
    )
    median, low, high, freq = stats
    assert np.allclose(median, point)
    assert np.allclose(low, point)
    assert np.allclose(high, point)
    assert np.allclose(freq, np.array([1.0, 1.0]))


def test_compute_parent_attribution_off_returns_zeros():
    target = np.array([0.6, 0.4])
    parents = np.eye(2)
    phi = compute_parent_attribution(
        target=target,
        parents=parents,
        constraint="simplex",
        sparsity=None,
        l2_regularizer=1e-6,
        max_iter=100,
        learning_rate=0.1,
        tolerance=1e-9,
        random_state=1,
        shapley_exact_threshold=8,
        shapley_monte_carlo_samples=10,
        compute_shapley=False,
    )
    assert np.allclose(phi, np.zeros(2))
