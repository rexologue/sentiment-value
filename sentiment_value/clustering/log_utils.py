"""Logging helpers for clustering scripts."""
from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with Rich formatting for readable console output."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger configured via :func:`setup_logging`."""

    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
