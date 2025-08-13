"""Tests for data_manager.commands."""

from pathlib import Path

from typer.testing import CliRunner

from src import cli
from src.data_manager import database, storage

runner = CliRunner()


def test_db_lifecycle(tmp_path: Path) -> None:
    db_path = tmp_path / "cli.db"

    # Initialise database
    result = runner.invoke(cli.app, ["db", "init", "--db", str(db_path)])
    assert result.exit_code == 0
    assert db_path.exists()

    # Stats should report zero rows
    result = runner.invoke(cli.app, ["db", "stats", "--db", str(db_path)])
    assert "papers: 0" in result.stdout

    # Insert sample data directly via storage helpers
    conn = database.init_db(db_path)
    storage.add_paper(conn, {"doi": "10.1/A"})
    storage.add_paper(conn, {"doi": "10.1/B"})
    storage.add_citation(conn, "10.1/A", "10.1/B")
    storage.add_embedding(conn, "10.1/A", [0.1])
    conn.close()

    result = runner.invoke(cli.app, ["db", "stats", "--db", str(db_path)])
    assert "papers: 2" in result.stdout
    assert "citations: 1" in result.stdout
    assert "embeddings: 1" in result.stdout

    # Clear database
    result = runner.invoke(cli.app, ["db", "clear", "--db", str(db_path), "--yes"])
    assert result.exit_code == 0

    result = runner.invoke(cli.app, ["db", "stats", "--db", str(db_path)])
    assert "papers: 0" in result.stdout

def test_db_stats_missing_database(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing" / "cli.db"
    result = runner.invoke(cli.app, ["db", "stats", "--db", str(missing_db)])
    assert result.exit_code != 0
    assert "unable to open database file" in str(result.exception)


def test_db_clear_missing_database(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing" / "cli.db"
    result = runner.invoke(
        cli.app, ["db", "clear", "--db", str(missing_db), "--yes"]
    )
    assert result.exit_code != 0
    assert "unable to open database file" in str(result.exception)


def test_db_init_corrupted_database(tmp_path: Path) -> None:
    corrupted_db = tmp_path / "corrupted.db"
    corrupted_db.write_text("not a database")
    result = runner.invoke(cli.app, ["db", "init", "--db", str(corrupted_db)])
    assert result.exit_code != 0
    assert "file is not a database" in str(result.exception)

