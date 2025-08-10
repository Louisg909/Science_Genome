"""Database Core.
data_manager.database
=================================

Minimal SQLite wrapper used by tests and examples.  The module exposes a
``SCHEMA`` dictionary containing ``CREATE TABLE`` statements and an
``init_db`` helper that applies the schema to a new or existing database
file.  The schema is intentionally small – just enough to store paper
metadata, citation edges and, for the purposes of this exercise, document
embeddings.

Embeddings are stored as **BLOBs** using Python's :mod:`pickle` to serialize
arbitrary vector objects.  Pickle is convenient and works for ``list``s or
:mod:`numpy` arrays, but the resulting binary strings can be large.  When
storing many high dimensional vectors, the database file can grow quickly;
callers may want to consider dimensionality reduction or compression before
insertion.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Each entry is a ``CREATE TABLE`` statement.  ``init_db`` iterates over this
# dictionary to ensure all tables exist.
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def init_db(db_path: Path | str) -> sqlite3.Connection:
    """Initialise the SQLite database and return an open connection.
    Path to the SQLite database file.  The file is created if it does not
        already exist.

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
  
#  Or the following, I don't know which is best or which to do
#     cur = conn.cursor()
#     for statement in SCHEMA.values():
#         cur.execute(statement)
#     conn.commit()
    return conn
  
  
__all__ = ["init_db", "SCHEMA"]
