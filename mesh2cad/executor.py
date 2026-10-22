"""Execute generated CAD in a fresh, fail-closed macOS Seatbelt sandbox."""

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


class ExecutionCancelled(RuntimeError):
    """A caller explicitly cancelled a sandboxed proposal."""


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
    """Do not inherit API keys, Codex auth, user Python paths, or startup hooks."""
    home = directory / "home"
    temporary = directory / "tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    return {
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "PATH": "/usr/bin:/bin",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(home / "matplotlib"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def _run_process(
    command: list[str],
    directory: Path,
    timeout: float,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[int, bool]:
    """Private transport helper, also usable by worker integration tests."""
    with (
        (directory / "stdout.log").open("w", encoding="utf-8") as stdout,
        (directory / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=_environment(directory),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            if cancel_check is None:
                return process.wait(timeout=timeout), False
            deadline = time.monotonic() + timeout
            while True:
                if cancel_check():
                    raise ExecutionCancelled("CAD execution was cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    return process.wait(timeout=min(0.15, remaining)), False
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return process.returncode, True
        finally:
            # Reap any child processes spawned by the proposal after completion.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)


def execute_code(
    code: str,
    work_dir: Path,
    timeout: float = 30,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """Save and execute one proposal; never fall back to unsandboxed execution.

    Only macOS with ``/usr/bin/sandbox-exec`` is supported. Results use absolute
    artifact paths; failed candidates expose neither STEP nor mesh paths.
    """
    started = time.monotonic()
    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
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
        if error:
            (directory / "traceback.txt").unlink(missing_ok=True)
            (directory / "traceback.txt").write_text(error, encoding="utf-8")
        return {
            "status": status,
            "error": error,
            "mesh_path": str(directory / "candidate.stl") if status == "valid" else None,
            "step_path": str(directory / "candidate.step") if status == "valid" else None,
            "elapsed_seconds": time.monotonic() - started,
        }

    if not math.isfinite(timeout) or timeout <= 0:
        return finish("execution_error", "Execution timeout must be positive and finite")
    if cancel_check is not None and cancel_check():
        return finish("cancelled", "CAD execution was cancelled before launch")
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        return finish(
            "sandbox_error",
            "CAD execution requires macOS sandbox-exec; unsandboxed execution is disabled",
        )
    # Copy the small standalone entry point, never expose the package/repository.
    worker = directory / "_worker.py"
    shutil.copyfile(Path(__file__).with_name("_worker.py"), worker)
    profile = directory / "sandbox.sb"
    profile.write_text(_sandbox_profile(directory), encoding="utf-8")
    command = ["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable, "-I", "-B", str(worker)]
    try:
        if cancel_check is None:
            returncode, timed_out = _run_process(command, directory, timeout)
        else:
            returncode, timed_out = _run_process(
                command, directory, timeout, cancel_check=cancel_check
            )
    except ExecutionCancelled as exc:
        return finish("cancelled", str(exc))
    except OSError as exc:
        return finish("sandbox_error", f"Could not launch macOS sandbox: {exc}")
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
            f"Sandboxed worker exited with code {returncode} without a result.\n{stderr}".rstrip(),
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
