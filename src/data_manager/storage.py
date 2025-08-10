"""data_manager.storage
========================

High-level helpers built on top of :mod:`sqlite3`.  The functions here
abstract away SQL boilerplate and provide a tiny API for inserting and
retrieving papers and their derived data such as embeddings.

Embeddings are serialized using :func:`pickle.dumps` before being written to
the database and deserialized with :func:`pickle.loads` on retrieval.  The
approach is flexible – any picklable Python object or :mod:`numpy` array can
be stored – but note that pickle adds overhead and large vectors can quickly
bloat the database file.
"""

from __future__ import annotations

import pickle
import sqlite3
from typing import Any, Dict, Iterable, Optional


# ---------------------------------------------------------------------------
# Paper helpers
# ---------------------------------------------------------------------------
def add_paper(conn: sqlite3.Connection, paper: Dict[str, Any]) -> None:
    """Insert or update a paper record.

    Parameters
    ----------
    conn:
        Active SQLite connection.
    paper:
        Mapping containing at least the ``doi`` key.  Extra keys are ignored.
    """

    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO papers (doi, title, abstract, authors, categories, date)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                paper["doi"],
                paper.get("title"),
                paper.get("abstract"),
                paper.get("authors"),
                paper.get("categories"),
                paper.get("date"),
            ),
        )


def get_paper(conn: sqlite3.Connection, doi: str) -> Optional[Dict[str, Any]]:
    """Fetch a paper by DOI."""

    cur = conn.execute(
        "SELECT doi, title, abstract, authors, categories, date FROM papers WHERE doi = ?",
        (doi,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    keys = ["doi", "title", "abstract", "authors", "categories", "date"]
    return dict(zip(keys, row))


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def add_embedding(conn: sqlite3.Connection, doi: str, vector: Any) -> None:
    """Insert or update a picklable embedding vector."""

    blob = pickle.dumps(vector)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (doi, vector) VALUES (?, ?)",
            (doi, blob),
        )


def get_embedding(conn: sqlite3.Connection, doi: str) -> Optional[Any]:
    """Retrieve and unpickle an embedding vector by DOI."""

    cur = conn.execute(
        "SELECT vector FROM embeddings WHERE doi = ?",
        (doi,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return pickle.loads(row[0])

