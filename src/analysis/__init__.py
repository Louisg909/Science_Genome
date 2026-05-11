from .citation_dag import build_citation_dag, enforce_acyclicity_by_time
from .inheritance import InheritanceResult, solve_inheritance
from .structure import (
    embedding_norm_diagnostics,
    nearest_neighbor_density,
    nearest_neighbors,
    similarity_matrix,
)

__all__ = [
    "InheritanceResult",
    "build_citation_dag",
    "enforce_acyclicity_by_time",
    "embedding_norm_diagnostics",
    "nearest_neighbor_density",
    "nearest_neighbors",
    "similarity_matrix",
    "solve_inheritance",
]
