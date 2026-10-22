"""Exercise the watchdog with small real processes, including separate CAD-like sessions."""

import subprocess
import sys
import time

import psutil
import pytest

from mesh2cad.resources import monitor_process, worker_environment


def test_memory_guard_counts_and_kills_children_in_other_sessions(tmp_path) -> None:
    pids = tmp_path / "children.txt"
    script = tmp_path / "spawn.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "children = [subprocess.Popen([sys.executable, '-c', "
        "'import time; data=bytearray(32*1024*1024); time.sleep(30)'], "
        "start_new_session=True) for _ in range(2)]\n"
        f"Path({str(pids)!r}).write_text(','.join(str(p.pid) for p in children))\n"
        "time.sleep(30)\n"
    )
    process = subprocess.Popen([sys.executable, str(script)])
    try:
        usage = monitor_process(process, memory_limit_mb=64, timeout=8)
        assert usage.reason == "memory_limit"
        assert usage.peak_rss_mb > 64
        assert process.poll() is not None
        for pid in map(int, pids.read_text().split(",")):
            assert (
                not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("reason", ["timeout", "cancelled"])
def test_watchdog_stops_blocked_native_work(reason) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    started = time.monotonic()
    usage = monitor_process(
        process,
        timeout=0.2 if reason == "timeout" else 10,
        cancel_check=(lambda: True) if reason == "cancelled" else None,
    )
    assert usage.reason == reason
    assert time.monotonic() - started < 5
    assert process.poll() is not None


def test_trusted_workers_do_not_inherit_credentials_or_python_hooks(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-unit-secret")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "private-unit-token")
    monkeypatch.setenv("PYTHONPATH", "/unexpected/hooks")
    environment = worker_environment()
    assert "OPENROUTER_API_KEY" not in environment
    assert "SOME_SERVICE_TOKEN" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
