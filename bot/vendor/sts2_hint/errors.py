"""Error types and logging setup for the ``sts2_hint`` package.

Bead: ``sts2-rs-7al4`` (hint-T6).

This module owns:

* :class:`UnsupportedStateError` — raised by the adapter (T7+) whenever a
  Sts2Headless decision-point snapshot references an id, power, card or
  monster the Rust hint engine cannot model. The exception message is
  expected to carry the offending JSON path (e.g.
  ``player.relics[3]: GHIBBET_MAW``) so that the caller can show a
  precise diagnostic *and* the log line has enough context to act on.

* :func:`get_logger` — returns the singleton ``sts2_hint`` logger,
  configured on first call with a :class:`~logging.handlers.RotatingFileHandler`
  pointed at ``python/sts2_hint/logs/sts2_hint.log`` (5 MB × 3 backups).
  If the log directory is unwritable (CI sandboxes, read-only mounts,
  etc.) we fall back to a :class:`~logging.StreamHandler` on ``stderr``
  so an error is *still* visible.

The matching Rust side raises ``sts2_rs.UnsupportedStateError`` from
``crates/sts2-py/src/state_dict.rs``; the Python-side class is a
:class:`ValueError` subclass so callers can ``except ValueError`` and
match either path uniformly. The two classes intentionally do *not*
share an inheritance hierarchy: the PyO3 exception is created with
``create_exception!(... PyValueError ...)`` and lives in the
``sts2_rs`` module's namespace; this Python-side class lives in
``sts2_hint``. Either is acceptable to raise; callers should catch
``ValueError``.

Manual stderr-fallback test (no automated test for this — chmod is
host-specific and CI may run as root)::

    chmod -w python/sts2_hint/logs
    python -c "from sts2_hint.errors import get_logger; \\
               get_logger().error('forced fallback')"
    # expect: line on stderr, no file created
    chmod +w python/sts2_hint/logs
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import runtime_paths

_LOG_NAME = "sts2_hint"
# Frozen advisor exe: never write inside the install dir (Program Files on a
# default Steam install) -- logs go to %LOCALAPPDATA%\SpireMentor instead.
_LOG_DIR = (
    runtime_paths.app_data_dir() / "logs" / "sts2_hint"
    if runtime_paths.is_frozen()
    else Path(__file__).resolve().parent / "logs"
)
_LOG_FILE = _LOG_DIR / "sts2_hint.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_logger_configured = False


class UnsupportedStateError(ValueError):
    """Raised when a Sts2Headless snapshot references something the Rust
    hint engine cannot model (unknown relic, unmodelled power, mid-
    resolution state, etc.).

    Message convention: ``"<json-path>: <detail>"`` — see
    ``crates/sts2-py/src/state_dict.rs`` for the canonical paths.
    """


def _try_make_log_dir() -> bool:
    """Best-effort creation of the log directory.

    Returns ``True`` on success, ``False`` if the directory can't be
    created OR can't be written to (so the caller can fall back to
    stderr).
    """
    try:
        # ``os.makedirs(..., exist_ok=True)`` per plan §T6 — pathlib's
        # equivalent matches the same semantics.
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    # Final probe: even if the dir exists, it may be read-only.
    return os.access(_LOG_DIR, os.W_OK)


def _build_handler() -> logging.Handler:
    """Return a configured handler — rotating-file when writable, else
    stderr. The decision is made once at logger configuration time.
    """
    if _try_make_log_dir():
        try:
            handler: logging.Handler = RotatingFileHandler(
                _LOG_FILE,
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
        except OSError:
            handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def get_logger() -> logging.Logger:
    """Return the singleton ``sts2_hint`` logger, configuring it on the
    first call. Subsequent calls return the same instance without
    re-attaching handlers.
    """
    global _logger_configured
    log = logging.getLogger(_LOG_NAME)
    if not _logger_configured:
        log.setLevel(logging.INFO)
        log.addHandler(_build_handler())
        # Don't propagate to root — the parent app may have its own
        # handlers and we don't want double-logging.
        log.propagate = False
        _logger_configured = True
    return log
