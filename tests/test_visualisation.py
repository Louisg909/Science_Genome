import matplotlib
import numpy as np

from src.models import Paper
from src.visualisation import scatter_embeddings

matplotlib.use("Agg")


def test_scatter_embeddings_returns_figure():
    papers = [Paper(title="A", abstract="alpha", references=[]), Paper(title="B", abstract="beta", references=[])]
    reduced = np.array([[0.0, 1.0], [1.0, 0.0]])
    fig = scatter_embeddings(reduced, papers)
    assert fig.axes[0].get_xlabel() == "Component 1"
