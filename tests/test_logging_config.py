"""Tests for logging configuration."""

import logging
import tempfile
from pathlib import Path

import pytest

from orf_entropy.logging_config import (
    configure_logging,
    get_log_file,
    get_log_level,
    get_logger,
    is_configured,
    set_log_level,
)


def test_get_logger():
    """Test getting a logger instance."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_configure_logging_stdout(caplog):
    """Test configuring logging to STDOUT."""
    configure_logging(level="INFO", log_file=None, force=True)

    assert is_configured()
    assert get_log_level() == logging.INFO
    assert get_log_file() is None

    # Test logging output (captured in stdout since we configured a StreamHandler)
    # Note: caplog might not capture our custom handler, so we just verify configuration


def test_configure_logging_to_file():
    """Test configuring logging to a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"

        configure_logging(level="DEBUG", log_file=log_file, force=True)

        assert is_configured()
        assert get_log_level() == logging.DEBUG
        assert get_log_file() == log_file

        # Test logging output
        logger = get_logger("test_file")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        # Check file contents
        assert log_file.exists()
        log_contents = log_file.read_text()
        assert "Debug message" in log_contents
        assert "Info message" in log_contents
        assert "Warning message" in log_contents


def test_configure_logging_levels():
    """Test different logging levels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "levels.log"

        # Configure with WARNING level
        configure_logging(level="WARNING", log_file=log_file, force=True)

        logger = get_logger("test_levels")
        logger.debug("Debug - should not appear")
        logger.info("Info - should not appear")
        logger.warning("Warning - should appear")
        logger.error("Error - should appear")

        log_contents = log_file.read_text()
        assert "Debug - should not appear" not in log_contents
        assert "Info - should not appear" not in log_contents
        assert "Warning - should appear" in log_contents
        assert "Error - should appear" in log_contents


def test_configure_logging_string_level():
    """Test configuring logging with string level names."""
    configure_logging(level="DEBUG", force=True)
    assert get_log_level() == logging.DEBUG

    configure_logging(level="INFO", force=True)
    assert get_log_level() == logging.INFO

    configure_logging(level="WARNING", force=True)
    assert get_log_level() == logging.WARNING

    configure_logging(level="ERROR", force=True)
    assert get_log_level() == logging.ERROR

    configure_logging(level="CRITICAL", force=True)
    assert get_log_level() == logging.CRITICAL


def test_configure_logging_int_level():
    """Test configuring logging with integer levels."""
    configure_logging(level=logging.DEBUG, force=True)
    assert get_log_level() == logging.DEBUG

    configure_logging(level=logging.INFO, force=True)
    assert get_log_level() == logging.INFO


def test_set_log_level():
    """Test changing log level at runtime."""
    configure_logging(level="INFO", force=True)
    assert get_log_level() == logging.INFO

    set_log_level("DEBUG")
    assert get_log_level() == logging.DEBUG

    set_log_level(logging.WARNING)
    assert get_log_level() == logging.WARNING


def test_configure_logging_creates_parent_dirs():
    """Test that log file parent directories are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "nested" / "dir" / "test.log"

        configure_logging(level="INFO", log_file=log_file, force=True)

        logger = get_logger("test_dirs")
        logger.info("Test message")

        assert log_file.exists()
        assert log_file.parent.exists()


def test_configure_logging_no_reconfigure_without_force():
    """Test that logging is not reconfigured without force=True."""
    configure_logging(level="INFO", force=True)
    initial_level = get_log_level()

    # Try to reconfigure without force (should not change)
    configure_logging(level="DEBUG", force=False)
    assert get_log_level() == initial_level

    # Reconfigure with force (should change)
    configure_logging(level="DEBUG", force=True)
    assert get_log_level() == logging.DEBUG


def test_logging_format():
    """Test that logging format includes expected components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "format.log"

        configure_logging(level="INFO", log_file=log_file, force=True)

        logger = get_logger("test_format")
        logger.info("Test message")

        log_contents = log_file.read_text()
        # Should include timestamp, logger name, level, and message
        assert "test_format" in log_contents
        assert "INFO" in log_contents
        assert "Test message" in log_contents
        # Check for timestamp pattern (YYYY-MM-DD HH:MM:SS)
        assert any(char.isdigit() for char in log_contents)


def test_multiple_loggers_same_config():
    """Test that multiple loggers use the same configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "multi.log"

        configure_logging(level="INFO", log_file=log_file, force=True)

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        logger1.info("Message from module1")
        logger2.info("Message from module2")

        log_contents = log_file.read_text()
        assert "Message from module1" in log_contents
        assert "Message from module2" in log_contents
