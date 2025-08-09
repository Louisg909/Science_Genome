"""
Citation Graph.

Purpose:
    Build and manage the citation DAG using NetworkX.

Goals:
    - Construct directed acyclic graph from citation pairs.
    - Provide traversal and metric utilities (ancestors, descendants, degrees).

Inputs:
    - Citation table (citing DOI → cited DOI).

Outputs:
    - NetworkX DiGraph object.
    - Graph metrics (in-degree, out-degree, topological order).
"""
