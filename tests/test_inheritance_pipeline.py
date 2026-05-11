import numpy as np
import pytest

from src.analysis import citation_edges_to_parent_adjacency, fit_inheritance_for_corpus


def test_parent_lookup_builds_expected_parent_order():
    papers = [{"paper_id": "A"}, {"paper_id": "B"}, {"paper_id": "C"}]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 0.75],
        ]
    )
    dag = {"C": ["A", "B"]}

    result = fit_inheritance_for_corpus(papers, embeddings, dag)

    assert result["C"].parent_ids == ("A", "B")


def test_no_parent_behavior_matches_solver_edge_case():
    papers = [{"paper_id": "solo"}]
    embeddings = np.array([[0.2, -0.5, 1.7]])
    dag = {}

    result = fit_inheritance_for_corpus(papers, embeddings, dag)

    solo = result["solo"]
    assert solo.parent_ids == ()
    assert solo.weights.size == 0
    assert np.allclose(solo.residual_vector, embeddings[0])
    assert solo.residual_magnitude == np.linalg.norm(embeddings[0])


def test_deterministic_outputs_with_fixed_inputs():
    papers = [{"paper_id": "A"}, {"paper_id": "B"}, {"paper_id": "C"}]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.3, 0.0],
        ]
    )
    dag = {"C": ["A", "B"]}

    first = fit_inheritance_for_corpus(papers, embeddings, dag, random_state=11)
    second = fit_inheritance_for_corpus(papers, embeddings, dag, random_state=11)

    assert np.allclose(first["C"].weights, second["C"].weights)
    assert np.allclose(first["C"].residual_vector, second["C"].residual_vector)
    assert first["C"].converged == second["C"].converged


def test_edge_list_adapter_builds_canonical_parent_adjacency():
    edges = [("A", "C"), ("B", "C"), ("A", "D")]

    adjacency = citation_edges_to_parent_adjacency(edges)

    assert adjacency == {"C": ["A", "B"], "D": ["A"]}


def test_fit_accepts_adapter_output_from_citation_edges():
    papers = [{"paper_id": "A"}, {"paper_id": "B"}, {"paper_id": "C"}]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.3, 0.7]])
    dag = citation_edges_to_parent_adjacency([("A", "C"), ("B", "C")])

    result = fit_inheritance_for_corpus(papers, embeddings, dag)
    assert result["C"].parent_ids == ("A", "B")


def test_fit_rejects_non_mapping_dag_with_helpful_message():
    papers = [{"paper_id": "A"}]
    embeddings = np.array([[1.0, 0.0]])

    with pytest.raises(TypeError, match="mapping of child_id -> sequence"):
        fit_inheritance_for_corpus(papers, embeddings, [("A", "B")])


def test_fit_rejects_invalid_dag_value_shape():
    papers = [{"paper_id": "A"}]
    embeddings = np.array([[1.0, 0.0]])

    with pytest.raises(TypeError, match="dag value must be a sequence"):
        fit_inheritance_for_corpus(papers, embeddings, {"A": "B"})


def test_edge_list_adapter_rejects_bad_edge_shape():
    with pytest.raises(TypeError, match=r"must be a \(parent_id, child_id\) tuple"):
        citation_edges_to_parent_adjacency([("A", "B", "C")])
