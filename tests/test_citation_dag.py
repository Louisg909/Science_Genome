from src.analysis.citation_dag import build_citation_dag, enforce_acyclicity_by_time
from src.models import Paper


def _paper(paper_id: str, year: int | None, references: list[str]) -> Paper:
    return Paper(paper_id=paper_id, title=paper_id, abstract="", year=year, references=references)


def test_build_citation_dag_creates_edges_from_known_references():
    papers = [
        _paper("1001.0001", 2020, []),
        _paper("1001.0002", 2021, ["arXiv:1001.0001", "unknown-ref"]),
    ]

    edges = build_citation_dag(papers)

    assert edges == [("1001.0001", "1001.0002")]


def test_build_citation_dag_ignores_unresolved_references():
    papers = [_paper("2001.0001", 2022, ["not-a-known-id"])]

    edges = build_citation_dag(papers)

    assert edges == []


def test_enforce_acyclicity_by_time_filters_temporal_violations():
    edges = [("p1", "p2"), ("p2", "p3"), ("p3", "p1")]
    year_map = {"p1": 2018, "p2": 2020, "p3": 2019}

    filtered = enforce_acyclicity_by_time(edges, year_map)

    assert ("p2", "p3") not in filtered
    assert ("p3", "p1") not in filtered
    assert ("p1", "p2") in filtered


def test_temporal_filter_produces_acyclic_graph_for_synthetic_fixture():
    papers = [
        _paper("a", 2018, ["c"]),
        _paper("b", 2019, ["a"]),
        _paper("c", 2020, ["b"]),
    ]

    raw_edges = build_citation_dag(papers)
    filtered = enforce_acyclicity_by_time(raw_edges, {p.paper_id: p.year for p in papers})

    assert raw_edges
    assert filtered == [("a", "b"), ("b", "c")]
