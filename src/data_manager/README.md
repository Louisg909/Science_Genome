# Data Manager

Provides the persistence layer for papers, citations and embeddings. It uses SQLite for simplicity.

## Key Files

- `database.py` – defines the `SCHEMA` and `init_db` helper for setting up a SQLite database.
- `storage.py` – high-level helpers for inserting and retrieving papers, citations and embedding vectors.
- `commands.py` – CLI-oriented database maintenance commands (init, clear, stats).

## Usage

```python
from src.data_manager import database, storage

conn = database.init_db("science.db")
storage.add_paper(conn, {"doi": "10.1/ABC", "title": "Example"})
storage.add_citation(conn, "10.1/ABC", "10.1/XYZ")
embedding = [0.1, 0.2, 0.3]
storage.add_embedding(conn, "10.1/ABC", embedding)
```

## Contributing

Extend `storage.py` with new CRUD helpers or indexes, and expose them through
`commands.py` for command line access. The tests rely on SQLite's standard
library API, so avoid adding heavy ORM dependencies.
