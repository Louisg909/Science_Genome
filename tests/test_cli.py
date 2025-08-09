"""
Tests for CLI commands.
"""
import pytest
from typer.testing import CliRunner
from src import cli

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
