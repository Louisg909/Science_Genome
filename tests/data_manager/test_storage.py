"""
Tests for data_manager.storage.
"""
from src.data_manager import database, storage


def test_add_and_get_paper(temp_db_path, sample_paper):
    conn = database.init_db(temp_db_path)
    storage.add_paper(conn, sample_paper)
    fetched = storage.get_paper(conn, sample_paper["doi"])
    assert fetched["doi"] == sample_paper["doi"]


def _other_paper():
    return {
        "doi": "10.5678/other",
        "title": "Other Paper",
        "abstract": "Another abstract.",
        "authors": "Roe, R.",
        "categories": "cs.LG",
        "date": "2024-01-01",
    }


def test_add_and_get_citation(temp_db_path, sample_paper):
    conn = database.init_db(temp_db_path)
    other = _other_paper()
    storage.add_paper(conn, sample_paper)
    storage.add_paper(conn, other)
    storage.add_citation(conn, sample_paper["doi"], other["doi"])
    citations = storage.get_citations(conn, citing_doi=sample_paper["doi"])
    assert citations == [
        {"citing_doi": sample_paper["doi"], "cited_doi": other["doi"]}
    ]


def test_citation_duplicates_ignored(temp_db_path, sample_paper):
    conn = database.init_db(temp_db_path)
    other = _other_paper()
    storage.add_paper(conn, sample_paper)
    storage.add_paper(conn, other)
    storage.add_citation(conn, sample_paper["doi"], other["doi"])
    storage.add_citation(conn, sample_paper["doi"], other["doi"])
    assert len(storage.get_citations(conn)) == 1


def test_get_citations_bidirectional(temp_db_path, sample_paper):
    conn = database.init_db(temp_db_path)
    other = _other_paper()
    storage.add_paper(conn, sample_paper)
    storage.add_paper(conn, other)
    storage.add_citation(conn, sample_paper["doi"], other["doi"])
    by_citing = storage.get_citations(conn, citing_doi=sample_paper["doi"])
    by_cited = storage.get_citations(conn, cited_doi=other["doi"])
    assert by_citing == by_cited
