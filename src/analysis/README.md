# Analysis

This package hosts routines for interpreting the citation corpus and derived embeddings. Each module focuses on a different type of analysis.

## Modules

- `citation_graph.py` – build a directed acyclic graph (DAG) from `citing_doi -> cited_doi` pairs and expose traversal utilities.
- `clustering.py` – group embedding vectors with unsupervised algorithms (e.g. k-means) to reveal field structure.
- `evolution.py` – aggregate inheritance scores over time to track thematic shifts.
- `inheritance.py` – compare each paper's vector to its cited predecessors to compute semantic inheritance.
- `metrics.py` – correlate inheritance with citation counts or network measures.

## Usage

Functions are currently placeholders but are expected to be imported directly from the submodules:

```python
from src.analysis import citation_graph, clustering, inheritance

G = citation_graph.build_graph(citations)
labels = clustering.k_means(embeddings, k=10)
scores = inheritance.compute(G, embeddings)
```

## Contributing

Each file contains a module-level docstring describing the intended API.
Implement the documented routines and keep I/O formats lightweight (plain
Python data structures or NumPy arrays) so they integrate easily with other
packages.
