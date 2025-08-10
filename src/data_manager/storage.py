"""data_manager.storage

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
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Paper helpers
# ---------------------------------------------------------------------------

def add_paper(conn: sqlite3.Connection, paper: Dict[str, Any]) -> None:
    """Insert or update a paper record.

    Parameters
    ----------
    conn:
        Active/open SQLite connection.
    paper:
        Mapping containing at least the ``doi`` key.  Extra keys are ignored.
    paper:
        Mapping containing paper metadata. Expected keys are
        ``doi``, ``title``, ``abstract``, ``authors``, ``categories`` and
        ``date``.
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


def get_paper(conn: sqlite3.Connection, doi: str) -> Optional[Dict[str, str]]:
def get_paper(conn: sqlite3.Connection, doi: str) -> Optional[Dict[str, Any]]:
    """Fetch a paper by DOI.

    Returns ``None`` if the DOI is unknown."""

    cur = conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,))
    row = cur.fetchone()
    # keys = ["doi", "title", "abstract", "authors", "categories", "date"]
    # return dict(zip(keys, row))
    return dict(row) if row else None


def add_citation(conn: sqlite3.Connection, citing_doi: str, cited_doi: str) -> None:
    """Store a citation edge between two papers.

    Duplicate edges are ignored by relying on the composite primary key of the
    ``citations`` table.
    """

    with conn:
        conn.execute(
            """INSERT OR IGNORE INTO citations (citing_doi, cited_doi)
            VALUES (?, ?)""",
            (citing_doi, cited_doi),
        )


def get_citations(
    conn: sqlite3.Connection,
    citing_doi: Optional[str] = None,
    cited_doi: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return citation edges matching the provided filters."""

    query = "SELECT citing_doi, cited_doi FROM citations"
    clauses: List[str] = []
    params: List[str] = []
    if citing_doi is not None:
        clauses.append("citing_doi = ?")
        params.append(citing_doi)
    if cited_doi is not None:
        clauses.append("cited_doi = ?")
        params.append(cited_doi)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    cur = conn.execute(query, params)
    return [dict(row) for row in cur.fetchall()]

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

__all__ = [
    "add_paper",
    "get_paper",
    "add_citation",
    "get_citations",
    "add_embedding",
    "get_embedding"
]
