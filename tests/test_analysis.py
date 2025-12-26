import numpy as np
import pytest

from src.analysis import nearest_neighbors, similarity_matrix
from src.models import Paper


def test_similarity_matrix_returns_square_matrix():
    data = np.array([[1, 0], [0, 1]])
    matrix = similarity_matrix(data)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(1.0)


def test_nearest_neighbors_returns_neighbors():
    papers = [
        Paper(title="A", abstract="alpha", references=[]),
        Paper(title="B", abstract="beta", references=[]),
        Paper(title="C", abstract="alpha beta", references=[]),
    ]
    data = np.array([[1.0, 0.0], [0.0, 5.0], [1.0, 0.1]])
    neighbors = nearest_neighbors(data, papers, n_neighbors=1)
    assert len(neighbors) == 3
    # Paper C should be closest to A given the embedding
    assert neighbors[2][1][0].title == "A"
