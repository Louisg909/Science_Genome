# Dimension Reduction

Tools for projecting high-dimensional embeddings into a smaller space for exploration and plotting.

## Modules

- `reducers.py` – implements simple algorithms such as `pca_reduce`.
- `cost_functions.py` – planned location for stress/trustworthiness metrics.

## Usage

```python
from src.dimension_reduction import reducers

coords = reducers.pca_reduce(embeddings, n_components=2)
```

## Contributing

Add new reducers (t‑SNE, UMAP, etc.) to `reducers.py` and evaluation metrics to
`cost_functions.py`. Keep functions stateless and operate on NumPy arrays to
stay compatible with the rest of the project.
