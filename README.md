# Science_Genome
This is the code for my transaction paper on my noval analysis of dynamic scientific space in the form of a pseudo-genome in the form of transformer embeddings, and watching the evolution of these ideas through a family tree DAG.

<div style="display: flex; gap: 1rem; align-items: center;">
  <img src="assets/embedding.png" alt="Embedding diagram" style="width: 48%;" />
  <img src="assets/basic_tree.png" alt="Tree diagram" style="width: 48%;" />
</div>


## Installation



## Citation DAG Example

The storage layer exposes a `citations` table that can be used to build a
citation graph. The snippet below shows how to record citations and load them
into a directed acyclic graph (DAG) using `networkx`:

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

`G` now contains the citation relationships and can be analysed or visualised as
needed.












