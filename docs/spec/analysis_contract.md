# Analysis Contract: Research Objectives → Software Outputs

## Purpose
This contract translates the project’s research objectives into concrete, testable software outputs and artifacts. It defines what each objective must produce, how outputs are validated, and how claims remain traceable.

## Canonical Inputs
All analysis implementations are expected to accept and version the following input classes:

1. **arXiv metadata/text**
   - Minimum fields: paper identifier, title, authors, timestamp, abstract, categories.
2. **Citation edges**
   - Directed graph edges `citing_paper -> cited_paper` with source provenance and extraction confidence.
3. **Domain labels**
   - Single- or multi-label domain assignments (e.g., categories/subfields) used for cross-field analysis.

## Objective-to-Output Mapping

| Research objective | Required metric/function | Concrete software output(s) | Expected artifact file(s) |
|---|---|---|---|
| Idea lineage: where ideas came from | Lineage graph construction + ancestry scoring | Ranked lineage links, ancestry DAG slices, parent sets per node | `artifacts/lineage/parent_sets.parquet`, `artifacts/lineage/lineage_graph.json` |
| Inheritance and persistence of traits | Parent inheritance score function | Per-paper parent inheritance scores with trait-level breakdowns | `artifacts/metrics/parent_inheritance_scores.parquet` |
| Emergence of new traits | Residual contribution estimator | Residual novelty/contribution scores after explained parent inheritance | `artifacts/metrics/residual_contribution.parquet` |
| Divergence/convergence across domains | Cross-field transfer function | Cross-domain transfer matrix and directional transfer statistics | `artifacts/metrics/cross_field_transfer.parquet`, `artifacts/figures/cross_field_transfer_heatmap.png` |
| Branch concentration vs spread | Lineage concentration metric | Concentration indices per branch/domain/time window | `artifacts/metrics/lineage_concentration.parquet` |
| Interpretable global structure | 2D mapping/reduction function | 2D embeddings/maps with lineage overlays and domain coloring | `artifacts/maps/idea_map_2d.parquet`, `artifacts/figures/idea_map_2d.png` |

## Core Output Contract

### 1) Parent Inheritance Scores
- **Definition:** Degree to which a paper’s traits can be explained by inferred parents.
- **Required fields:** `paper_id`, `parent_id`, `inheritance_score`, `trait_group`, `time`.
- **Quality checks:** score bounds, parent set non-emptiness (when citations exist), temporal consistency.

### 2) Residual Contribution
- **Definition:** Contribution not explained by inherited traits from selected parent set.
- **Required fields:** `paper_id`, `residual_score`, `uncertainty`, `normalization_version`.
- **Quality checks:** decomposition sanity (`inherited + residual ≈ total` within tolerance).

### 3) Cross-Field Transfer
- **Definition:** Strength and direction of trait transfer across domain boundaries.
- **Required fields:** `source_domain`, `target_domain`, `transfer_score`, `time_window`.
- **Quality checks:** symmetry/asymmetry reporting, sparse-domain guardrails.

### 4) Lineage Concentration
- **Definition:** Degree to which downstream idea flow is concentrated among a few ancestors/branches.
- **Required fields:** `unit_id`, `scope` (paper/author/domain), `concentration_index`, `time_window`.
- **Quality checks:** sensitivity to branching factor, robustness across window sizes.

### 5) 2D Maps
- **Definition:** Low-dimensional representation for interpretability of semantic + lineage structure.
- **Required fields:** `paper_id`, `x`, `y`, `domain_label`, `lineage_cluster`.
- **Quality checks:** neighborhood stability, random-seed sensitivity, projection distortion diagnostics.

## V&V Output Contract

V&V is first-class and must produce explicit artifacts in every analysis run:

1. **Diagnostics**
   - Data integrity checks (missingness, graph consistency, label coverage).
   - Metric sanity checks (bounds, monotonicity expectations, decomposition checks).
   - Artifact: `artifacts/validation/diagnostics.json`.

2. **Uncertainty Summaries**
   - Confidence intervals/bootstrapped variance (or analogous uncertainty quantification).
   - Sensitivity by sampling, parameterization, and time slicing.
   - Artifact: `artifacts/validation/uncertainty_summary.parquet`.

3. **Failure-Mode Flags**
   - Explicit flags for known failure cases: sparse citation neighborhoods, domain-label ambiguity, temporal leakage, unstable map geometry, degenerate parent sets.
   - Artifact: `artifacts/validation/failure_mode_flags.parquet`.

## Traceability Table (Paper Claim → Implementation)

| Paper claim | Metric/function | Artifact file |
|---|---|---|
| “This work inherits strongly from prior lineage X.” | `compute_parent_inheritance_scores(...)` | `artifacts/metrics/parent_inheritance_scores.parquet` |
| “The paper contributes novel traits beyond its lineage.” | `compute_residual_contribution(...)` | `artifacts/metrics/residual_contribution.parquet` |
| “Ideas transfer from field A to field B over time.” | `compute_cross_field_transfer(...)` | `artifacts/metrics/cross_field_transfer.parquet` |
| “Idea flow is highly concentrated in a small ancestor set.” | `compute_lineage_concentration(...)` | `artifacts/metrics/lineage_concentration.parquet` |
| “The idea landscape shows interpretable clusters/bridges.” | `build_2d_idea_map(...)` | `artifacts/maps/idea_map_2d.parquet` + `artifacts/figures/idea_map_2d.png` |
| “Results are credible within stated limits.” | `run_validation_suite(...)` | `artifacts/validation/diagnostics.json`, `artifacts/validation/uncertainty_summary.parquet`, `artifacts/validation/failure_mode_flags.parquet` |

## Implementation Constraints (Explicit Non-Goals from `scope.md`)

The following constraints are mandatory and derived from project non-goals:

1. **Not a search engine**
   - Do not optimize for interactive retrieval/ranking UX as a primary deliverable.
   - Any retrieval components must exist only to support reproducible analysis inputs.

2. **No definitive causal claims**
   - Outputs must be framed as lineage/association/contribution evidence, not proof of causality.
   - UI/text/report templates must include caveats against causal over-interpretation.

3. **Do not replace expert judgment**
   - Artifacts must include uncertainty and failure-mode context.
   - Human interpretation remains required for substantive conclusions.

## Reproducibility and Versioning Requirements

Each produced artifact must include or be accompanied by:
- data snapshot/version identifiers,
- code version (`git` commit hash),
- parameter/config hash,
- run timestamp,
- random seed policy,
- schema version.

Recommended metadata artifact: `artifacts/run_manifest.json`.
