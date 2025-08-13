# Science_Genome

## Overview
Science_Genome explores the evolution of scientific ideas by treating the literature as a pseudo-genome. Papers are embedded with transformer models and linked through a citation "family tree" to analyse how concepts change over time.

<div style="display: flex; gap: 1rem; align-items: center;">
  <img src="assets/embedding.png" alt="Embedding diagram" style="width: 48%;" />
  <img src="assets/basic_tree.png" alt="Tree diagram" style="width: 48%;" />
</div>

## Project Goals
- Map research papers into a high-dimensional embedding space.
- Build and analyse citation graphs to trace the lineage of ideas.
- Provide tooling for embedding, dimensionality reduction and visualisation.

## Installation
```bash
git clone <repository-url>
cd Science_Genome
pip install -r requirements.txt
```

## CLI Usage
The project exposes a small [Typer](https://typer.tiangolo.com/)-powered CLI.
Database maintenance commands live under the `db` subcommand. Run the
following for a list of options:

```bash
python -m src.cli --help
python -m src.cli db --help
```

Typical database workflow:

```bash
# Initialise the SQLite database
python -m src.cli db init --db science.db

# Show table counts
python -m src.cli db stats --db science.db

# Clear all data (use --yes to skip confirmation)
python -m src.cli db clear --db science.db --yes
```

## Citation DAG Example
The storage layer exposes a `citations` table that can be used to build a citation graph. The snippet below shows how to record citations and load them into a directed acyclic graph (DAG) using `networkx`:

```python
from src.data_manager import database, storage
import networkx as nx

conn = database.init_db("science.db")

storage.add_paper(conn, {"doi": "10.1/A"})
storage.add_paper(conn, {"doi": "10.1/B"})
storage.add_citation(conn, "10.1/A", "10.1/B")

G = nx.DiGraph()
for edge in storage.get_citations(conn):
    G.add_edge(edge["citing_doi"], edge["cited_doi"])
```

`G` now contains the citation relationships and can be analysed or visualised as needed.
