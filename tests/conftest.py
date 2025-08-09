"""
Pytest Configuration.

Provides shared fixtures (e.g., temporary DB, config) for tests.
"""
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_db_path(tmp_path: Path):
    """Fixture: Temporary SQLite database path."""
    return tmp_path / "test_science_papers.db"

@pytest.fixture
def sample_paper():
    """Fixture: Example paper metadata dict."""
    return {
        "doi": "10.1234/example",
        "title": "Example Paper",
        "abstract": "This is a sample abstract.",
        "authors": "Doe, J.",
        "categories": "cs.AI",
        "date": "2023-01-01"
    }
