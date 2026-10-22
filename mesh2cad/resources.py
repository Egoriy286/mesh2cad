"""Monitor isolated computations without relying on macOS virtual-memory limits."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable

import psutil

from .observability import log_event

DEFAULT_MEMORY_MB = 2048


def worker_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not any(secret in name.upper() for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        and name not in {"PYTHONPATH", "PYTHONSTARTUP"}
    }
    environment.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1"
    )
    return environment


@dataclass(frozen=True)
class ProcessUsage:
    returncode: int
    reason: str | None
    peak_rss_mb: float


def tree_rss(process: psutil.Process) -> int:
    """Count resident memory of the worker AND its CAD subprocesses."""
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for child in [process, *children]:
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def kill_tree(process: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    # CAD workers create their own sessions, so killing only a process group
    # would leave them running after the orchestrator has stopped.
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()
    psutil.wait_procs(children, timeout=2)


def monitor_process(
    process: subprocess.Popen,
    *,
    memory_limit_mb: int = DEFAULT_MEMORY_MB,
    timeout: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    cancel_grace: float = 0,
    on_usage: Callable[[float, float], None] | None = None,
) -> ProcessUsage:
    """Poll RSS every 100 ms; terminate the entire tree on limit/cancel/timeout.

    This is a resident-memory watchdog, not a reservation or an address-space
    limit: a single allocation may briefly overshoot between samples.
    """
    started = time.monotonic()
    cancelled_at = None
    peak = 0.0
    reason = None
    try:
        tracked = psutil.Process(process.pid)
        while process.poll() is None:
            now = time.monotonic()
            rss = tree_rss(tracked) / (1024 * 1024)
            peak = max(peak, rss)
            if on_usage is not None:
                on_usage(rss, peak)
            if rss > memory_limit_mb:
                reason = "memory_limit"
            elif timeout is not None and now - started >= timeout:
                reason = "timeout"
            elif cancel_check is not None and cancel_check():
                cancelled_at = cancelled_at or now
                if now - cancelled_at >= cancel_grace:
                    reason = "cancelled"
            if reason:
                log_event(
                    "process.stopped",
                    pid=process.pid,
                    reason=reason,
                    rss_mb=round(rss, 1),
                    limit_mb=memory_limit_mb,
                )
                kill_tree(process)
                break
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
    except psutil.NoSuchProcess:
        process.wait()
    finally:
        if process.poll() is None:
            kill_tree(process)
    return ProcessUsage(process.returncode, reason, peak)


def watch_parent(parent_pid: int) -> None:
    """Do not leave unmonitored workers behind when the Streamlit server exits."""
    parent = psutil.Process(parent_pid)

    def watch():
        try:
            while parent.is_running() and parent.status() != psutil.STATUS_ZOMBIE:
                time.sleep(0.5)
        except psutil.NoSuchProcess:
            pass
        for child in psutil.Process().children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        os._exit(1)

    threading.Thread(target=watch, name="parent-watchdog", daemon=True).start()
