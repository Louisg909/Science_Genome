"""High level helpers for interacting with the database."""

from __future__ import annotations

from typing import Dict, List, Optional

import sqlite3


def add_paper(conn: sqlite3.Connection, paper: Dict[str, str]) -> None:
    """Insert a paper record.

    Parameters
    ----------
    conn:
        Open SQLite connection.
    paper:
        Mapping containing paper metadata. Expected keys are
        ``doi``, ``title``, ``abstract``, ``authors``, ``categories`` and
        ``date``.
    """

    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO papers
                (doi, title, abstract, authors, categories, date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                paper.get("doi"),
                paper.get("title"),
                paper.get("abstract"),
                paper.get("authors"),
                paper.get("categories"),
                paper.get("date"),
            ),
        )


def get_paper(conn: sqlite3.Connection, doi: str) -> Optional[Dict[str, str]]:
    """Fetch a paper by DOI.

    Returns ``None`` if the DOI is unknown."""

    cur = conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,))
    row = cur.fetchone()
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


__all__ = [
    "add_paper",
    "get_paper",
    "add_citation",
    "get_citations",
]

