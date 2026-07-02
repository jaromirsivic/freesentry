"""Centralised logging configuration for the XTX2 package.

All modules acquire loggers via :func:`get_logger`; production code never uses
``print`` (except the interactive console prompts). Logging is configured once,
idempotently, on first import.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED: bool = False
_DEFAULT_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT: str = "%H:%M:%S"


def configure_logging(*, level: int = logging.INFO) -> None:
    """Install a single stream handler on the root ``xtx2`` logger.

    Safe to call multiple times; only the first call attaches a handler.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("xtx2")
    root.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, configuring logging on first use."""

    if not _CONFIGURED:
        configure_logging()
    if not name.startswith("xtx2"):
        name = f"xtx2.{name}"
    return logging.getLogger(name)
