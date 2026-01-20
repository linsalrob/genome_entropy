"""Centralized logging configuration for genome_entropy.

This module provides a single source for configuring logging throughout the application.
It supports:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Output to file or STDOUT
- Consistent format across all modules
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module logger cache to avoid re-configuration
_configured = False
_log_file: Optional[Path] = None
_log_level: int = logging.INFO


def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    force: bool = False,
) -> None:
    """Configure logging for the entire application.

    This should be called once at application startup (e.g., in CLI main).

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) as int or string
        log_file: Optional path to log file. If None, logs to STDOUT
        log_format: Format string for log messages
        date_format: Format string for timestamps
        force: If True, reconfigure even if already configured

    Examples:
        >>> configure_logging(level=logging.DEBUG, log_file="app.log")
        >>> configure_logging(level="INFO")  # Log to STDOUT
        >>> configure_logging(level="DEBUG", log_file=None)  # Debug to STDOUT
    """
    global _configured, _log_file, _log_level

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Skip if already configured (unless force=True)
    if _configured and not force:
        return

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Create handler (file or stdout)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, mode="a")
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Store configuration state
    _configured = True
    _log_file = Path(log_file) if log_file else None
    _log_level = level


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    This is the preferred way to get loggers in the application.

    Args:
        name: Name of the logger (usually __name__ of the module)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


def is_configured() -> bool:
    """Check if logging has been configured.

    Returns:
        True if configure_logging() has been called
    """
    return _configured


def get_log_file() -> Optional[Path]:
    """Get the current log file path.

    Returns:
        Path to log file, or None if logging to STDOUT
    """
    return _log_file


def get_log_level() -> int:
    """Get the current logging level.

    Returns:
        Current logging level as integer
    """
    return _log_level


def set_log_level(level: Union[int, str]) -> None:
    """Change the logging level at runtime.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> set_log_level("DEBUG")
        >>> set_log_level(logging.WARNING)
    """
    global _log_level

    # Convert string to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    _log_level = level
    logging.getLogger().setLevel(level)
