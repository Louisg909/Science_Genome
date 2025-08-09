"""
Tests for data_manager.storage.
"""
from src.data_manager import database, storage

def test_add_and_get_paper(temp_db_path, sample_paper):
    conn = database.init_db(temp_db_path)
    storage.add_paper(conn, sample_paper)
    fetched = storage.get_paper(conn, sample_paper["doi"])
    assert fetched["doi"] == sample_paper["doi"]
