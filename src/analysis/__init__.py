from .inheritance import (
    InheritanceResult,
    attach_corpus_inheritance_metrics,
    residual_metrics,
    solve_inheritance,
)
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
    "attach_corpus_inheritance_metrics",
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
    "residual_metrics",
    "similarity_matrix",
    "solve_inheritance",
]
