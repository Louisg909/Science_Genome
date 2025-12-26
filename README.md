# Science_Genome (lean edition)

This project provides a tiny, end-to-end pipeline for exploring scientific papers:

1. **Scrape** paper titles, abstracts, and references from the ArXiv Atom feed.
2. **Embed** the papers with SciBERT sentence representations (or inject your own model for testing).
3. **Reduce** the embedding dimensionality (PCA/TSNE/UMAP) for visual inspection.
4. **Analyse** similarities and nearest neighbours between papers.
5. **Visualise** the reduced points on a simple scatter plot.

## Scope

**Included**

- ArXiv scraping for titles, abstracts, and references.
- SciBERT-based embedding with optional model/tokenizer injection.
- Dimensionality reduction via PCA, t-SNE, and UMAP.
- Basic structure analysis (similarity matrix and nearest neighbours).
- Simple scatter plot visualisations of reduced embeddings.
- JSON-based corpus persistence and reload.

**Not included**

- Command-line interfaces, web apps, or notebooks.
- Databases, schedulers, or background job systems.
- Advanced analytics beyond neighbour similarity (e.g., clustering, graph metrics).
- Interactive or network visualisations.
- Non-Python implementations or alternative embedding models beyond SciBERT.

## Quick start

```bash
pip install -r requirements.txt
pytest
```

## Navigating the repository

- `src/` holds all runtime code, grouped by task: `scraping`, `embedding`, `dimension_reduction`, `analysis`, `visualisation`, and `storage`.
- `tests/` contains focused unit tests mirroring the `src/` layout to show expected behaviour for each step of the pipeline.
- `assets/` can be used to store generated plots or intermediate artifacts you want to keep out of source directories.
- `requirements.txt` lists the minimal dependencies needed to run and test the pipeline.

## How to use the pipeline

1. Scrape papers: `python - <<'PY'
from src.scraping.arxiv import fetch_arxiv_papers
papers = fetch_arxiv_papers(max_results=10)
PY`
2. Persist the corpus so you can avoid re-scraping: `python - <<'PY'
from src.storage import save_papers
save_papers(papers, "corpus.json")
PY`
3. Load saved papers later: `python - <<'PY'
from src.storage import load_papers
papers = load_papers("corpus.json")
PY`
4. Embed, reduce, analyse, and plot: `python - <<'PY'
from src.embedding.basic import embed_papers
from src.dimension_reduction.basic import reduce_embeddings
from src.analysis.structure import nearest_neighbours
from src.visualisation.plots import scatter_plot

embeddings = embed_papers(papers)
reduced = reduce_embeddings(embeddings, method="pca", n_components=2)
neighbours = nearest_neighbours(embeddings, k=3)
scatter_plot(reduced, papers, output_path="assets/plot.png")
PY`

## How to extend or add to the repo

- Keep additions minimal and focused on the core workflow (scrape → embed → reduce → analyse → visualise).
- Place new functionality beside related modules in `src/` and mirror coverage in `tests/`.
- Update `requirements.txt` if dependencies change, and add concise usage notes to this README when introducing new capabilities.

## Core modules
- `src/scraping/arxiv.py` – fetch and parse ArXiv feeds into `Paper` objects.
- `src/embedding/basic.py` – convert papers to SciBERT embeddings (configurable tokenizer/model).
- `src/dimension_reduction/basic.py` – reduce embedding dimensions with PCA, t-SNE, or UMAP.
- `src/analysis/structure.py` – compute similarity matrices and nearest neighbours.
- `src/visualisation/plots.py` – plot reduced embeddings with titles as labels.
- `src/storage.py` – save/load scraped papers to JSON so you can reuse a local corpus.
