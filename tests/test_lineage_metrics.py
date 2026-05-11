import math

import pytest

from src.analysis import (
    aggregate_cross_field_transfer,
    build_lineage_analytics,
    lineage_entropy_metrics,
)


def test_lineage_entropy_uses_nonzero_parent_weights_only():
    metrics = lineage_entropy_metrics({"p1": [0.5, 0.5, 0.0], "p2": [1.0, 0.0]})

    p1 = next(row for row in metrics if row.paper_id == "p1")
    p2 = next(row for row in metrics if row.paper_id == "p2")

    assert p1.nonzero_parent_count == 2
    assert p1.entropy == pytest.approx(math.log(2.0))
    assert p1.normalized_entropy == pytest.approx(1.0)

    assert p2.nonzero_parent_count == 1
    assert p2.entropy == pytest.approx(0.0)
    assert p2.normalized_entropy == pytest.approx(0.0)


def test_lineage_entropy_supports_optional_normalization_toggle():
    metrics = lineage_entropy_metrics({"p1": [0.7, 0.3]}, normalize=False)
    assert metrics[0].normalized_entropy is None


def test_cross_field_transfer_aggregation_and_compact_table():
    records = [
        {
            "paper_id": "paper-a",
            "field_label": "ml",
            "parents": [
                {"paper_id": "pa", "field_label": "ml", "weight": 0.6},
                {"paper_id": "pb", "field_label": "physics", "weight": 0.4},
            ],
        },
        {
            "paper_id": "paper-b",
            "field_label": "physics",
            "parents": [
                {"paper_id": "pc", "field_label": "ml", "weight": 0.2},
                {"paper_id": "pd", "field_label": "physics", "weight": 0.8},
            ],
        },
    ]

    transfer = aggregate_cross_field_transfer(records)
    assert transfer["within_field_weight"] == pytest.approx(1.4)
    assert transfer["cross_field_weight"] == pytest.approx(0.6)
    assert transfer["cross_field_share"] == pytest.approx(0.3)

    analytics = build_lineage_analytics(records)
    assert set(analytics.keys()) == {
        "paper_entropy",
        "transfer_matrix",
        "within_field_weight",
        "cross_field_weight",
        "cross_field_share",
    }
    assert len(analytics["paper_entropy"]) == 2
    assert any(
        row["source_field"] == "ml" and row["target_field"] == "physics"
        for row in analytics["transfer_matrix"]
    )
