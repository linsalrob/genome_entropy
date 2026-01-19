"""CLI smoke tests to verify commands run without crashing."""

import pytest

# Skip all CLI tests if typer is not installed
pytest.importorskip("typer")

from typer.testing import CliRunner

from orf_entropy.cli.main import app

runner = CliRunner()


def test_cli_help() -> None:
    """Test that main help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dna23di" in result.stdout.lower() or "DNA to 3Di" in result.stdout


def test_cli_version() -> None:
    """Test version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.stdout.lower() or "0.1.0" in result.stdout


def test_download_command_help() -> None:
    """Test download command help."""
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "model" in result.stdout.lower()


def test_orf_command_help() -> None:
    """Test orf command help."""
    result = runner.invoke(app, ["orf", "--help"])
    assert result.exit_code == 0
    assert "input" in result.stdout.lower()
    assert "output" in result.stdout.lower()


def test_translate_command_help() -> None:
    """Test translate command help."""
    result = runner.invoke(app, ["translate", "--help"])
    assert result.exit_code == 0
    assert "input" in result.stdout.lower()
    assert "output" in result.stdout.lower()


def test_encode3di_command_help() -> None:
    """Test encode3di command help."""
    result = runner.invoke(app, ["encode3di", "--help"])
    assert result.exit_code == 0
    assert "input" in result.stdout.lower()
    assert "output" in result.stdout.lower()


def test_entropy_command_help() -> None:
    """Test entropy command help."""
    result = runner.invoke(app, ["entropy", "--help"])
    assert result.exit_code == 0
    assert "input" in result.stdout.lower()
    assert "output" in result.stdout.lower()


def test_run_command_help() -> None:
    """Test run command help."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "input" in result.stdout.lower()
    assert "output" in result.stdout.lower()


def test_cli_no_command() -> None:
    """Test CLI with no command shows help."""
    result = runner.invoke(app, [])
    # Should show help and exit gracefully
    assert "dna23di" in result.stdout.lower() or "DNA" in result.stdout


def test_cli_invalid_command() -> None:
    """Test CLI with invalid command."""
    result = runner.invoke(app, ["invalid_command"])
    assert result.exit_code != 0


def test_estimate_tokens_command_help() -> None:
    """Test estimate-tokens command help."""
    result = runner.invoke(app, ["estimate-tokens", "--help"])
    assert result.exit_code == 0
    assert "token size" in result.stdout.lower()
    assert "model" in result.stdout.lower()
    assert "device" in result.stdout.lower()
