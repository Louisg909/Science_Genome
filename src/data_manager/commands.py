"""Database maintenance commands exposed via the CLI.

The commands here are intentionally small wrappers around the
:mod:`data_manager.database` helpers so they can be reused by tests and
examples.  They cover the most common administration tasks:

* ``db init``  – create the SQLite database.
* ``db clear`` – remove all rows from the tables.
* ``db stats`` – print basic table counts.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import database

app = typer.Typer(help="Database maintenance commands.")


def _db_option() -> Path:
    return typer.Option(
        Path("science.db"),
        "--db",
        "-d",
        help="Path to the SQLite database file.",
        metavar="PATH",
    )


@app.command("init")
def init_command(db: Path = _db_option()) -> None:
    """Initialise the database file and apply the schema."""

    conn = database.init_db(db)
    conn.close()
    typer.secho(f"Initialised database at {db}", fg=typer.colors.GREEN)


@app.command("clear")
def clear_command(
    db: Path = _db_option(),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm clearing the database without prompting.",
    ),
) -> None:
    """Remove all data from the database tables."""

    if not yes:
        confirm = typer.confirm(
            f"Delete all data in {db}?",
            default=False,
        )
        if not confirm:
            typer.echo("Aborted")
            raise typer.Exit(code=1)

    conn = database.init_db(db)
    with conn:
        # Delete tables in dependency order to satisfy foreign keys.
        for table in ("citations", "embeddings", "papers"):
            conn.execute(f"DELETE FROM {table}")
    conn.close()
    typer.secho(f"Cleared data in {db}", fg=typer.colors.YELLOW)


@app.command("stats")
def stats_command(db: Path = _db_option()) -> None:
    """Print simple row counts for each table."""

    conn = database.init_db(db)
    cur = conn.execute("SELECT COUNT(*) FROM papers")
    papers = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM citations")
    citations = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM embeddings")
    embeddings = cur.fetchone()[0]
    conn.close()

    typer.echo(f"papers: {papers}")
    typer.echo(f"citations: {citations}")
    typer.echo(f"embeddings: {embeddings}")


__all__ = ["app"]
