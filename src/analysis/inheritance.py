"""
Inheritance Factor.

Purpose:
    Compute semantic inheritance factor quantifying content derived from citations.

Goals:
    - Compare each paper’s embedding to a weighted average of its cited predecessors.
    - Produce an interpretable contribution score.

Inputs:
    - Paper embeddings from the database.
    - Citation DAG structure.

Outputs:
    - Inheritance factor per paper (float), stored or returned for plotting.
"""
