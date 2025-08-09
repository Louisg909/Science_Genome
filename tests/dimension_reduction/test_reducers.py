"""
Tests for dimension_reduction.reducers.
"""
import numpy as np

def test_pca_shape():
    from src.dimension_reduction import reducers
    X = np.random.rand(10, 50)
    reduced = reducers.pca_reduce(X, n_components=2)
    assert reduced.shape == (10, 2)
