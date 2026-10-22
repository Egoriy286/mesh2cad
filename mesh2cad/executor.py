"""Run generated CadQuery code in a fresh, isolated process.

macOS uses Seatbelt. Windows and Linux use a restricted subprocess with Python
audit hooks that block network, process creation, and file access outside the
run directory. The restricted backend is defense in depth, not a kernel sandbox.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Callable

from .observability import log_event, tail_log
from .resources import DEFAULT_MEMORY_MB, monitor_process

SEATBELT = "/usr/bin/sandbox-exec"


class ExecutionCancelled(RuntimeError):
    """A caller explicitly cancelled a sandboxed proposal."""


class ExecutionMemoryLimit(RuntimeError):
    """The isolated CAD process exceeded its resident-memory allowance."""


def _backend() -> str:
    forced = os.environ.get("MESH2CAD_SANDBOX", "auto").strip().lower()
    if forced == "seatbelt":
        return "seatbelt"
    if forced == "restricted":
        return "restricted"
    if forced not in {"", "auto"}:
        return "unsupported"
    if platform.system() == "Darwin" and Path(SEATBELT).is_file():
        return "seatbelt"
    return "restricted"


def _sandbox_profile(directory: Path) -> str:
    # Read dependencies, not their enclosing repository or the user's home.
    library_paths = {
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(sys.executable).resolve().parent,
    }
    library_paths.update(
        Path(path).resolve()
        for name, path in sysconfig.get_paths().items()
        if name in {"stdlib", "platstdlib", "purelib", "platlib"} and path
    )
    # A system prefix of /usr or / must never grant broad read access.
    library_paths.difference_update(
        {Path("/"), Path("/usr"), Path("/usr/local"), Path("/opt"), Path("/opt/homebrew")}
    )
    library_paths.update(
        Path(path)
        for path in (
            "/System/Library",
            "/usr/lib",
            "/usr/share",
            "/Library/Apple/System/Library",
            "/private/var/db/dyld",
        )
    )

    def quote(path):
        return json.dumps(str(path))

    reads = "\n".join(f"    (subpath {quote(path)})" for path in sorted(library_paths))
    return f"""(version 1)
(deny default)
(allow process-exec
    (literal {quote(Path(sys.executable).absolute())})
    (literal {quote(Path(sys.executable).resolve())}))
(allow process-fork)
(allow signal (target self))
(allow sysctl-read)
(allow file-read-metadata)
; macOS dyld inspects the root directory before Python starts. This literal
; permits that directory only, without granting reads of its descendants.
(allow file-read-data (literal "/"))
(allow file-map-executable
{reads})
(allow file-read*
{reads}
    (subpath {quote(directory)})
    (literal "/dev/null")
    (literal "/dev/random")
    (literal "/dev/urandom")
    (literal "/dev/dtracehelper")
    (literal "/private/etc/localtime")
    (literal "/private/etc/hosts")
    (literal "/private/etc/resolv.conf"))
(allow file-write* (subpath {quote(directory)}) (literal "/dev/null"))
"""


def _environment(directory: Path) -> dict[str, str]:
    """Do not inherit API keys, user Python paths, or startup hooks."""
    home = directory / "home"
    temporary = directory / "tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    environment = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "TMPDIR": str(temporary),
        "TMP": str(temporary),
        "TEMP": str(temporary),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "XDG_CACHE_HOME": str(home / ".cache"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    if os.name == "nt":
        # DLL resolution on Windows needs the inherited PATH and system roots.
        # Process creation is still blocked by the worker's audit hook.
        environment["APPDATA"] = str(home / "AppData" / "Roaming")
        environment["LOCALAPPDATA"] = str(home / "AppData" / "Local")
        for name in ("PATH", "SYSTEMROOT", "SystemDrive", "WINDIR", "COMSPEC", "PATHEXT"):
            if os.environ.get(name):
                environment[name] = os.environ[name]
    else:
        environment.update(
            {
                "PATH": "/usr/bin:/bin",
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
            }
        )
    return environment


def _kill_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        taskkill = shutil.which("taskkill")
        if taskkill:
            subprocess.run(
                [taskkill, "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_process(
    command: list[str],
    directory: Path,
    timeout: float,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[int, bool]:
    """Private transport helper, also usable by worker integration tests."""
    options: dict = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    with (
        (directory / "stdout.log").open("w", encoding="utf-8") as stdout,
        (directory / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=_environment(directory),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            **options,
        )
        try:
            usage = monitor_process(
                process,
                memory_limit_mb=DEFAULT_MEMORY_MB,
                timeout=timeout,
                cancel_check=cancel_check,
            )
            if usage.reason == "cancelled":
                raise ExecutionCancelled("CAD execution was cancelled")
            if usage.reason == "memory_limit":
                raise ExecutionMemoryLimit(f"CadQuery exceeded {DEFAULT_MEMORY_MB} MiB RSS")
            return usage.returncode, usage.reason == "timeout"
        finally:
            # Reap any child processes spawned by the proposal after completion.
            _kill_tree(process)


def execute_code(
    code: str,
    work_dir: Path,
    timeout: float = 30,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """Save and execute one proposal; never fall back to unsandboxed execution.

    Results use absolute artifact paths; failed candidates expose neither STEP
    nor mesh paths.
    """
    started = time.monotonic()
    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    backend = _backend()
    log_event("cadquery.started", directory=directory, backend=backend, timeout=timeout)
    # Unlink names before writing: a previous proposal could have left symlinks.
    for name in ("code.py", "stdout.log", "stderr.log", "_worker.py", "sandbox.sb"):
        (directory / name).unlink(missing_ok=True)
    (directory / "code.py").write_text(code, encoding="utf-8")
    for name in ("stdout.log", "stderr.log"):
        (directory / name).write_text("", encoding="utf-8")
    # A retried attempt must not inherit old success artifacts or traceback.
    for name in ("worker_result.json", "candidate.stl", "candidate.step", "traceback.txt"):
        (directory / name).unlink(missing_ok=True)

    def finish(status: str, error: str | None = None) -> dict:
        for name in ("stdout.log", "stderr.log"):
            text = tail_log(directory / name, max_bytes=8000, max_lines=30)
            if text:
                log_event(f"cadquery.{name}", directory=directory, output=text)
        log_event(
            "cadquery.finished",
            status=status,
            backend=backend,
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=error,
        )
        if error:
            (directory / "traceback.txt").unlink(missing_ok=True)
            (directory / "traceback.txt").write_text(error, encoding="utf-8")
        return {
            "status": status,
            "error": error,
            "sandbox": backend,
            "mesh_path": str(directory / "candidate.stl") if status == "valid" else None,
            "step_path": str(directory / "candidate.step") if status == "valid" else None,
            "elapsed_seconds": time.monotonic() - started,
        }

    if not math.isfinite(timeout) or timeout <= 0:
        return finish("execution_error", "Execution timeout must be positive and finite")
    if cancel_check is not None and cancel_check():
        return finish("cancelled", "CAD execution was cancelled before launch")
    if backend == "unsupported":
        return finish(
            "sandbox_error",
            "MESH2CAD_SANDBOX must be one of: auto, seatbelt, restricted",
        )
    if backend == "seatbelt" and (platform.system() != "Darwin" or not Path(SEATBELT).is_file()):
        return finish(
            "sandbox_error",
            "Seatbelt is available only on macOS with /usr/bin/sandbox-exec",
        )
    # Copy the small standalone entry point, never expose the package/repository.
    worker = directory / "_worker.py"
    shutil.copyfile(Path(__file__).with_name("_worker.py"), worker)
    if backend == "seatbelt":
        profile = directory / "sandbox.sb"
        profile.write_text(_sandbox_profile(directory), encoding="utf-8")
        command = [
            SEATBELT,
            "-f",
            str(profile),
            sys.executable,
            "-I",
            "-B",
            str(worker),
        ]
    else:
        command = [sys.executable, "-I", "-B", str(worker)]
    try:
        if cancel_check is None:
            returncode, timed_out = _run_process(command, directory, timeout)
        else:
            returncode, timed_out = _run_process(
                command, directory, timeout, cancel_check=cancel_check
            )
    except ExecutionCancelled as exc:
        return finish("cancelled", str(exc))
    except ExecutionMemoryLimit as exc:
        return finish("memory_limit", str(exc))
    except OSError as exc:
        return finish("sandbox_error", f"Could not launch the worker: {exc}")
    if timed_out:
        return finish("timeout", f"CadQuery execution exceeded {timeout:g} seconds")
    result_file = directory / "worker_result.json"
    if not result_file.is_file():
        stderr_file = directory / "stderr.log"
        stderr = (
            stderr_file.read_text(encoding="utf-8", errors="replace")[-16384:]
            if not stderr_file.is_symlink()
            else "Worker replaced stderr.log with a symlink"
        )
        status = (
            "sandbox_error"
            if any(
                marker in stderr for marker in ("sandbox-exec:", "sandbox_apply:", "sandbox_init:")
            )
            else "execution_error"
        )
        return finish(
            status,
            f"Worker exited with code {returncode} without a result.\n{stderr}".rstrip(),
        )
    if result_file.is_symlink():
        return finish("execution_error", "Worker replaced its result file with a symlink")
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return finish("execution_error", f"Worker returned an unreadable result: {exc}")
    if not isinstance(payload, dict):
        return finish("execution_error", "Worker returned a result that is not a JSON object")
    if returncode == 0 and payload.get("status") == "valid":
        if all(
            not (directory / name).is_symlink()
            and (directory / name).is_file()
            and (directory / name).stat().st_size > 0
            for name in ("candidate.stl", "candidate.step")
        ):
            return finish("valid")
        return finish("execution_error", "Worker did not produce nonempty STEP and STL files")
    error = payload.get("error")
    return finish(
        "execution_error",
        error if isinstance(error, str) and error else f"Worker exited with code {returncode}",
    )
