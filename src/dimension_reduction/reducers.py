"""Simple dimensionality reduction helpers.

Only a PCA reducer is implemented at this stage as required by the tests.  The
implementation uses ``numpy``'s singular-value decomposition which keeps the
dependency footprint low while remaining sufficiently fast for small arrays.
"""

from __future__ import annotations

import numpy as np


def pca_reduce(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Return the first ``n_components`` principal components of ``X``.

    Parameters
    ----------
    X:
        Two-dimensional array of shape ``(n_samples, n_features)``.
    n_components:
        Number of principal components to retain.
    """

    X_centered = X - X.mean(axis=0, keepdims=True)
    # Using ``full_matrices=False`` keeps the SVD economical.
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T


__all__ = ["pca_reduce"]

