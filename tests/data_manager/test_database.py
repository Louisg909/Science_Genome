"""
Tests for data_manager.database.
"""
from src.data_manager import database

def test_init_db(temp_db_path):
    conn = database.init_db(temp_db_path)
    assert conn is not None
