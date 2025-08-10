"""data_manager.database
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
from typing import Dict


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Each entry is a ``CREATE TABLE`` statement.  ``init_db`` iterates over this
# dictionary to ensure all tables exist.
SCHEMA: Dict[str, str] = {
    "papers": (
        "CREATE TABLE IF NOT EXISTS papers ("
        "doi TEXT PRIMARY KEY,"
        "title TEXT,"
        "abstract TEXT,"
        "authors TEXT,"
        "categories TEXT,"
        "date TEXT"
        ")"
    ),
    "citations": (
        "CREATE TABLE IF NOT EXISTS citations ("
        "citing_doi TEXT,"
        "cited_doi TEXT,"
        "PRIMARY KEY (citing_doi, cited_doi)"
        ")"
    ),
    # New table used to persist embedding vectors serialized as pickled
    # blobs.  ``doi`` is the primary key so repeated inserts behave as
    # upserts when using ``INSERT OR REPLACE``.
    "embeddings": (
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "doi TEXT PRIMARY KEY,"
        "vector BLOB"
        ")"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def init_db(db_path: str) -> sqlite3.Connection:
    """Initialise the SQLite database and return an open connection.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  The file is created if it does not
        already exist.
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for statement in SCHEMA.values():
        cur.execute(statement)
    conn.commit()
    return conn

