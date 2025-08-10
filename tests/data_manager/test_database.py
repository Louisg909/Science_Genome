"""
Tests for data_manager.database.
"""
from src.data_manager import database

def test_init_db(temp_db_path):
    conn = database.init_db(temp_db_path)
    assert conn is not None


def test_embeddings_table_exists(temp_db_path):
    """Embeddings table is created during initialisation."""

    conn = database.init_db(temp_db_path)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    )
    assert cur.fetchone() is not None
