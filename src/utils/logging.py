"""Minimal logging utilities.

The project only requires a very small abstraction over the standard library's
logging module.  ``get_logger`` configures a basic formatter on first use and
returns module-specific loggers thereafter.
"""

from __future__ import annotations

import logging
from typing import Optional


_configured = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with a simple configuration.

    The first call initialises ``logging.basicConfig`` so subsequent loggers
    share the same format and level.
    """

    global _configured
    if not _configured:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        _configured = True
    return logging.getLogger(name)


__all__ = ["get_logger"]

