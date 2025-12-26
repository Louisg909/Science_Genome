import numpy as np
import pytest

from src.dimension_reduction import reduce_dimensions


def test_reduce_dimensions_with_pca():
    data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    reduced = reduce_dimensions(data, method="pca", n_components=2, random_state=42)
    assert reduced.shape == (3, 2)


def test_reduce_dimensions_invalid_method():
    data = np.eye(3)
    with pytest.raises(ValueError):
        reduce_dimensions(data, method="unknown")
