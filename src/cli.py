"""Command line interface for the project.

The CLI is intentionally lightweight – only a bare ``Typer`` application is
exposed so tests can invoke ``--help``.  Additional commands can be registered
as the project grows.
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Science Genome command line interface."""
    # The body is intentionally empty; the callback exists to provide a help
    # message when ``--help`` is invoked.
    pass


__all__ = ["app"]

