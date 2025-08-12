# Visualisation

Helpers for plotting embeddings and citation networks.

## Modules

- `interactive.py` – interactive plots built with Plotly/Bokeh.
- `network.py` – visualise citation graphs and inheritance flows.
- `plots.py` – static figures using matplotlib.

## Usage

The modules currently serve as placeholders. Future APIs may resemble:

```python
from src.visualisation import network, plots

network.draw_graph(G, out_path="graph.html")
plots.embedding_scatter(coords, labels, out_path="scatter.png")
```

## Contributing

Implement plotting functions that accept standard Python or NumPy data
structures and return either file paths or figure objects. Keep heavy plotting
dependencies optional to ease installation.
