"""Lineage analytics derived from weighted parent mixtures."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class LineageMetrics:
    """Entropy metrics for a paper's weighted parent lineage."""

    paper_id: str
    nonzero_parent_count: int
    entropy: float
    normalized_entropy: float | None


def _entropy_nonzero(weights: Sequence[float]) -> tuple[float, int]:
    """Compute entropy across strictly positive weights after renormalization."""

    positive = [float(w) for w in weights if float(w) > 0.0]
    count = len(positive)
    if count == 0:
        return 0.0, 0

    total = sum(positive)
    probs = [w / total for w in positive]
    entropy = -sum(p * log(p) for p in probs)
    return entropy, count


def lineage_entropy_metrics(
    paper_parent_weights: Mapping[str, Sequence[float]],
    *,
    normalize: bool = True,
) -> list[LineageMetrics]:
    """Compute per-paper lineage entropy over nonzero parent weights.

    Args:
        paper_parent_weights: Map from paper id to iterable of parent weights.
        normalize: Whether to include entropy normalized by log(k), where k is
            the number of nonzero parents.
    """

    rows: list[LineageMetrics] = []
    for paper_id, weights in paper_parent_weights.items():
        entropy, count = _entropy_nonzero(weights)
        normalized = None
        if normalize:
            normalized = entropy / log(count) if count > 1 else 0.0

        rows.append(
            LineageMetrics(
                paper_id=paper_id,
                nonzero_parent_count=count,
                entropy=entropy,
                normalized_entropy=normalized,
            )
        )
    return rows


def aggregate_cross_field_transfer(
    paper_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate weighted parent transfer across source/target fields.

    Each paper record should include:
      - ``paper_id``: identifier for the focal paper.
      - ``field_label``: target field label for the focal paper.
      - ``parents``: iterable of mapping records with keys ``paper_id``,
        ``weight``, and ``field_label``.
    """

    transfer_matrix: dict[tuple[str, str], float] = defaultdict(float)
    within_field = 0.0
    cross_field = 0.0

    for paper in paper_records:
        target_field = str(paper["field_label"])
        for parent in paper.get("parents", []):
            source_field = str(parent["field_label"])
            weight = float(parent["weight"])
            transfer_matrix[(source_field, target_field)] += weight
            if source_field == target_field:
                within_field += weight
            else:
                cross_field += weight

    total = within_field + cross_field
    compact_matrix = [
        {"source_field": s, "target_field": t, "weight": w}
        for (s, t), w in sorted(transfer_matrix.items())
    ]

    return {
        "transfer_matrix": compact_matrix,
        "within_field_weight": within_field,
        "cross_field_weight": cross_field,
        "cross_field_share": (cross_field / total) if total > 0 else 0.0,
    }


def build_lineage_analytics(
    paper_records: Iterable[Mapping[str, Any]],
    *,
    normalize_entropy: bool = True,
) -> dict[str, Any]:
    """Build compact lineage analytics table/dict for downstream plotting."""

    records = list(paper_records)
    entropy_rows = lineage_entropy_metrics(
        {
            str(p["paper_id"]): [float(parent["weight"]) for parent in p.get("parents", [])]
            for p in records
        },
        normalize=normalize_entropy,
    )
    transfer = aggregate_cross_field_transfer(records)

    return {
        "paper_entropy": [
            {
                "paper_id": row.paper_id,
                "nonzero_parent_count": row.nonzero_parent_count,
                "entropy": row.entropy,
                "normalized_entropy": row.normalized_entropy,
            }
            for row in entropy_rows
        ],
        **transfer,
    }
