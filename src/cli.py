"""
CLI Module.

Purpose:
    Central orchestration entry point for all workflows (scraping, embedding, analysis, visualisation).

Goals:
    - Provide a single, coherent CLI using Typer/Click.
    - Expose commands: scrape, embed, analyze, visualise.
    - Offer professional usage messages, error handling, and logging.

Inputs:
    - Command-line arguments and flags (e.g., query terms, paths).

Outputs:
    - Console feedback and logs.
    - Invocation of downstream modules (DB writes, plots, etc.).
"""
