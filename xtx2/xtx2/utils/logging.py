"""Centralised logging configuration for the XTX2 package.

All modules acquire loggers via :func:`get_logger`; production code never uses
``print`` (except the interactive console prompts). Logging is configured once,
idempotently, on first import.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED: bool = False
_FILE_HANDLERS: dict[str, logging.FileHandler] = {}
_DEFAULT_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT: str = "%H:%M:%S"
_FILE_DATEFMT: str = "%Y-%m-%d %H:%M:%S"


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


def add_file_handler(log_path: Path | str, *, level: int = logging.INFO) -> None:
    """Mirror all ``xtx2`` log output into ``log_path`` (appending).

    Used by the trainer to keep a persistent training log next to the
    checkpoints. Idempotent per path; repeated calls with the same file are
    no-ops so resume does not stack duplicate handlers.
    """

    if not _CONFIGURED:
        configure_logging()
    path = Path(log_path).resolve()
    key = str(path)
    if key in _FILE_HANDLERS:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_FILE_DATEFMT))
    logging.getLogger("xtx2").addHandler(handler)
    _FILE_HANDLERS[key] = handler


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, configuring logging on first use."""

    if not _CONFIGURED:
        configure_logging()
    if not name.startswith("xtx2"):
        name = f"xtx2.{name}"
    return logging.getLogger(name)
