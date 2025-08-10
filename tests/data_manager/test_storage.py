"""
Tests for data_manager.storage.
"""
from src.data_manager import database, storage

def test_add_and_get_paper(temp_db_path, sample_paper):
    conn = database.init_db(temp_db_path)
    storage.add_paper(conn, sample_paper)
    fetched = storage.get_paper(conn, sample_paper["doi"])
    assert fetched["doi"] == sample_paper["doi"]


def test_add_and_get_embedding(temp_db_path):
    """Vectors round-trip through pickle serialization."""

    conn = database.init_db(temp_db_path)
    doi = "10.1234/example"
    vector = [0.1, 0.2, 0.3]
    storage.add_embedding(conn, doi, vector)
    fetched = storage.get_embedding(conn, doi)
    assert fetched == vector


def test_upsert_embedding(temp_db_path):
    """Second insert replaces existing vector."""

    conn = database.init_db(temp_db_path)
    doi = "10.1234/example"
    storage.add_embedding(conn, doi, [0.1])
    storage.add_embedding(conn, doi, [0.9])
    assert storage.get_embedding(conn, doi) == [0.9]
