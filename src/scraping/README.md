# Scraping

Modules for collecting metadata and citation information from external APIs.

## Components

- `arxiv.py` – query the arXiv API for papers.
- `crossref.py` – fetch additional metadata and citation links from Crossref.
- `parsers.py` – normalise raw API responses into the schema expected by `data_manager`.
- `scheduler.py` – orchestrate batch scraping while respecting rate limits.

## Usage

These modules currently contain design docstrings only. A future scraper might
look like:

```python
from src.scraping import arxiv, parsers

raw = arxiv.fetch("cat:cs.CL")
records = [parsers.parse_arxiv(entry) for entry in raw]
```

## Contributing

Implement the scraping logic for each provider. Aim for generator-style APIs
that yield dictionaries ready for storage and avoid hard dependencies on heavy
frameworks so the scrapers remain lightweight and testable.
