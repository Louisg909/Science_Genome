"""Database Core."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    doi TEXT PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    authors TEXT,
    categories TEXT,
    date TEXT
);

CREATE TABLE IF NOT EXISTS citations (
    citing_doi TEXT NOT NULL,
    cited_doi TEXT NOT NULL,
    PRIMARY KEY (citing_doi, cited_doi),
    FOREIGN KEY (citing_doi) REFERENCES papers(doi),
    FOREIGN KEY (cited_doi) REFERENCES papers(doi)
);

CREATE TABLE IF NOT EXISTS embeddings (
    doi TEXT PRIMARY KEY,
    vector BLOB,
    FOREIGN KEY (doi) REFERENCES papers(doi)
);
"""


def init_db(db_path: Path | str) -> sqlite3.Connection:
    """Initialise the SQLite database and return a connection.

    Parameters
    ----------
    db_path:
        Path to the database file.

    Returns
    -------
    sqlite3.Connection
        An open connection to the initialised database.
    """

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA)
    return conn


__all__ = ["init_db", "SCHEMA"]

