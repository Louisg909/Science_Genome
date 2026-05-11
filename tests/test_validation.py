import numpy as np

from src.analysis import generate_validation_report, solve_inheritance


def _fixture_data():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
    )
    target = embeddings[1]
    parent_matrix = embeddings[[0, 2, 3]].T
    lineage_records = [
        {"paper_id": "p1", "weights": [0.6, 0.4, 0.0], "field_label": "ml", "parents": []},
        {"paper_id": "p2", "weights": [1.0, 0.0], "field_label": "ml", "parents": []},
    ]
    return embeddings, target, parent_matrix, lineage_records


def test_validation_report_is_deterministic_with_fixed_random_state():
    embeddings, target, parent_matrix, lineage_records = _fixture_data()
    solved = solve_inheritance(
        target,
        parent_matrix,
        random_state=17,
        bootstrap_samples=10,
        compute_shapley=False,
    )

    report_one = generate_validation_report(
        embeddings=embeddings,
        inheritance_result=solved,
        parent_matrix=parent_matrix,
        lineage_records=lineage_records,
        n_neighbors=1,
    )
    report_two = generate_validation_report(
        embeddings=embeddings,
        inheritance_result=solved,
        parent_matrix=parent_matrix,
        lineage_records=lineage_records,
        n_neighbors=1,
    )

    assert report_one == report_two


def test_validation_report_schema_contains_expected_sections():
    embeddings, target, parent_matrix, lineage_records = _fixture_data()
    solved = solve_inheritance(
        target,
        parent_matrix,
        random_state=17,
        bootstrap_samples=10,
        compute_shapley=False,
    )
    report = generate_validation_report(
        embeddings=embeddings,
        inheritance_result=solved,
        parent_matrix=parent_matrix,
        lineage_records=lineage_records,
        n_neighbors=1,
    )

    assert set(report.keys()) == {"embedding_diagnostics", "solver", "uncertainty", "lineage_sanity"}
    assert set(report["embedding_diagnostics"].keys()) == {"norms", "neighbor_density"}
    assert set(report["solver"].keys()) == {
        "converged",
        "iterations",
        "objective",
        "active_parent_count",
        "parent_rank",
        "gram_condition_number",
    }
    assert set(report["uncertainty"].keys()) == {
        "ci_width",
        "ci_width_mean",
        "ci_width_max",
        "selection_frequency",
        "selection_frequency_mean",
    }
    assert report["lineage_sanity"]["all_checks_pass"] is True
