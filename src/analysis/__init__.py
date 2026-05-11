from .lineage_metrics import aggregate_cross_field_transfer, build_lineage_analytics, lineage_entropy_metrics
from .citation_dag import build_citation_dag, enforce_acyclicity_by_time
from .inheritance import InheritanceResult, solve_inheritance
from .inheritance_pipeline import PaperInheritanceFit, fit_inheritance_for_corpus
from .structure import (
    embedding_norm_diagnostics,
    nearest_neighbor_density,
    nearest_neighbors,
    similarity_matrix,
)

__all__ = [
    "InheritanceResult",
    "aggregate_cross_field_transfer",
    "build_lineage_analytics",
    "lineage_entropy_metrics",
    "PaperInheritanceFit",
    "build_citation_dag",
    "enforce_acyclicity_by_time",
    "embedding_norm_diagnostics",
    "fit_inheritance_for_corpus",
    "nearest_neighbor_density",
    "nearest_neighbors",
    "similarity_matrix",
    "solve_inheritance",
]
