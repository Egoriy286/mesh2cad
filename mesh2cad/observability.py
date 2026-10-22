"""One idempotent console/file logger for the application, CLI and worker threads."""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOCK = threading.RLock()
_SECRETS: set[str] = set()
LOGGER = logging.getLogger("mesh2cad")


def register_secret(secret: str) -> None:
    if len(secret) >= 8:
        with _LOCK:
            _SECRETS.add(secret)


def redact(text: str) -> str:
    with _LOCK:
        secrets = tuple(_SECRETS)
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{12,}", "[REDACTED]", text)
    return re.sub(r"(?i)Bearer\s+[A-Za-z0-9_.-]+", "Bearer [REDACTED]", text)


class SafeFormatter(logging.Formatter):
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(log_dir: Path | None = None) -> Path:
    default = os.environ.get("MESH2CAD_LOG_DIR")
    log_dir = (
        log_dir or (Path(default) if default else Path(__file__).resolve().parents[1] / "runs/logs")
    ).resolve()
    path = log_dir / "project.log"
    with _LOCK:
        if getattr(LOGGER, "_mesh2cad_path", None) == path:
            return path
        log_dir.mkdir(parents=True, exist_ok=True)
        for handler in tuple(LOGGER.handlers):
            LOGGER.removeHandler(handler)
            handler.close()
        formatter = SafeFormatter(
            "%(asctime)sZ %(levelname)-7s %(name)s [%(threadName)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        console = logging.StreamHandler(sys.stderr)
        file = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=3, encoding="utf-8")
        for handler in (console, file):
            handler.setFormatter(formatter)
            LOGGER.addHandler(handler)
        LOGGER.setLevel(
            getattr(logging, os.environ.get("MESH2CAD_LOG_LEVEL", "INFO").upper(), logging.INFO)
        )
        LOGGER.propagate = False
        LOGGER._mesh2cad_path = path
    return path


def log_event(event: str, *, level: int = logging.INFO, **fields) -> None:
    if not LOGGER.handlers:
        configure_logging()
    parts = [event]
    for key, value in fields.items():
        if key.lower() in {"api_key", "authorization", "payload", "response", "code", "images"}:
            continue
        parts.append(f"{key}={value!s}".replace("\n", " ")[:1500])
    LOGGER.log(level, " | ".join(parts))


@contextmanager
def timed_stage(name: str, **fields):
    started = time.monotonic()
    log_event(f"{name}.started", **fields)
    try:
        yield
    except Exception:
        LOGGER.exception("%s.failed | elapsed_seconds=%.3f", name, time.monotonic() - started)
        raise
    else:
        log_event(
            f"{name}.finished", elapsed_seconds=round(time.monotonic() - started, 3), **fields
        )


def tail_log(path: Path, *, max_bytes: int = 64_000, max_lines: int = 250) -> str:
    """Read a bounded tail, even if a run's event log has grown to gigabytes."""
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        start = max(0, size - max_bytes)
        handle.seek(start)
        if start:
            handle.readline()
        data = handle.read(max_bytes)
    return redact("\n".join(data.decode("utf-8", errors="replace").splitlines()[-max_lines:]))
