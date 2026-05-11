import numpy as np
import pytest

from src.analysis import estimate_shapley_contributions


def test_shapley_symmetry_for_interchangeable_parents():
    target = np.array([0.5, 0.5])
    parents = np.array([[1.0, 1.0], [0.0, 0.0]])

    shapley = estimate_shapley_contributions(
        target, parents, constraint="nonnegative", exact_threshold=8
    )

    assert shapley[0] == pytest.approx(shapley[1], abs=1e-8)


def test_shapley_near_zero_for_irrelevant_parent():
    target = np.array([1.0, 0.0])
    parents = np.array([[1.0, 0.0], [0.0, 1.0]])

    shapley = estimate_shapley_contributions(
        target, parents, constraint="nonnegative", exact_threshold=8
    )

    assert shapley[0] > 0.1
    assert abs(shapley[1]) < 1e-6


def test_shapley_approximation_stability_across_seeds():
    target = np.array([0.9, 0.5, -0.2])
    parents = np.array(
        [
            [1.0, 0.1, 0.0, -0.3],
            [0.0, 1.0, 0.2, 0.1],
            [0.2, -0.1, 1.0, 0.3],
        ]
    )

    s1 = estimate_shapley_contributions(
        target,
        parents,
        exact_threshold=2,
        monte_carlo_samples=400,
        random_state=7,
    )
    s2 = estimate_shapley_contributions(
        target,
        parents,
        exact_threshold=2,
        monte_carlo_samples=400,
        random_state=19,
    )

    assert np.max(np.abs(s1 - s2)) < 0.15
