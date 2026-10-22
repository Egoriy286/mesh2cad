"""CadQuery process entry point; invoked only by the executor's sandbox."""

from __future__ import annotations

import builtins
import json
import math
import os
import sys
import sysconfig
import traceback
from pathlib import Path

_BLOCKED_IMPORTS = frozenset(
    {
        "cffi",
        "fcntl",
        "ftplib",
        "http",
        "httpx",
        "multiprocessing",
        "pty",
        "requests",
        "resource",
        "socket",
        "smtplib",
        "subprocess",
        "telnetlib",
        "urllib",
        "webbrowser",
        "win32api",
        "win32con",
        "win32event",
        "win32file",
        "win32process",
        "winreg",
        "xmlrpc",
        "_winapi",
    }
)
_BLOCKED_EVENTS = frozenset(
    {
        "os.exec",
        "os.fork",
        "os.kill",
        "os.killpg",
        "os.posix_spawn",
        "os.spawn",
        "os.startfile",
        "os.system",
    }
)
_BLOCKED_EVENT_PREFIXES = (
    "ctypes.",
    "msvcrt.",
    "socket.",
    "subprocess.",
    "winreg.",
)
_READ_PATH_EVENTS = frozenset({"os.listdir", "os.readlink", "os.scandir"})
_PATH_EVENTS = frozenset(
    {
        "os.chmod",
        "os.chown",
        "os.link",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.symlink",
        "os.truncate",
        "os.utime",
    }
)


def _resolve_roots() -> set[Path]:
    roots = {
        Path.cwd().resolve(),
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(sys.executable).resolve().parent,
    }
    roots.update(
        Path(path).resolve()
        for name, path in sysconfig.get_paths().items()
        if name in {"stdlib", "platstdlib", "purelib", "platlib"} and path
    )
    if os.name == "nt":
        for name in ("SYSTEMROOT", "WINDIR"):
            if os.environ.get(name):
                roots.add(Path(os.environ[name]).resolve())
    else:
        roots.update(
            Path(path).resolve()
            for path in ("/System/Library", "/usr/lib", "/usr/share")
            if Path(path).exists()
        )
    return roots


def _within(path: str | bytes | os.PathLike, roots: set[Path]) -> bool:
    try:
        candidate = Path(os.fsdecode(path)).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    candidate_key = os.path.normcase(str(candidate))
    for root in roots:
        root_key = os.path.normcase(str(root))
        try:
            if os.path.commonpath([candidate_key, root_key]) == root_key:
                return True
        except ValueError:
            continue
    return False


def _install_guard(directory: Path) -> None:
    """Block the operations that would leave the per-run directory.

    This is an in-process audit hook, not a kernel boundary. It is used on
    platforms without Seatbelt (notably Windows and Linux).
    """
    read_roots = _resolve_roots()
    write_roots = {directory}

    def guard(event: str, args: tuple) -> None:
        if event == "import":
            module = args[0].split(".", 1)[0] if args and isinstance(args[0], str) else ""
            if module in _BLOCKED_IMPORTS:
                raise PermissionError(f"Import is blocked by the CAD sandbox: {module}")
            return
        if event in _BLOCKED_EVENTS or event.startswith(_BLOCKED_EVENT_PREFIXES):
            raise PermissionError(f"Operation is blocked by the CAD sandbox: {event}")
        if event == "open":
            path = args[0] if args else None
            if path is None or isinstance(path, int):
                return
            mode = args[1] if len(args) > 1 else None
            writes = isinstance(mode, str) and any(character in mode for character in "wax+")
            if not _within(path, write_roots if writes else read_roots):
                raise PermissionError(f"File access outside the run directory: {path!r}")
            return
        if event == "os.chdir":
            path = args[0] if args else None
            if path is not None and not _within(path, write_roots):
                raise PermissionError(f"chdir outside the run directory: {path!r}")
            return
        if event in _READ_PATH_EVENTS:
            for value in args:
                if isinstance(value, (str, bytes, os.PathLike)) and not _within(value, read_roots):
                    raise PermissionError(f"Read outside the run directory: {value!r}")
            return
        if event in _PATH_EVENTS:
            for value in args:
                if isinstance(value, (str, bytes, os.PathLike)) and not _within(value, write_roots):
                    raise PermissionError(f"Write outside the run directory: {value!r}")

    sys.addaudithook(guard)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Generated code may import only CadQuery and math."""
    if level or name.split(".", 1)[0] not in {"cadquery", "math"}:
        raise ImportError(f"Import is not allowed in generated CAD code: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def main() -> int:
    directory = Path.cwd()
    result_file = directory / "worker_result.json"
    try:
        print("[cadquery] importing runtime", flush=True)
        import cadquery as cq

        _install_guard(directory)
        source_path = directory / "code.py"
        source = source_path.read_text(encoding="utf-8")
        safe_builtins = dict(builtins.__dict__)
        safe_builtins["__import__"] = _safe_import
        namespace = {
            "cq": cq,
            "math": math,
            "__name__": "__mesh2cad_generated__",
            "__builtins__": safe_builtins,
        }
        print("[cadquery] executing generated program", flush=True)
        exec(compile(source, str(source_path), "exec"), namespace)
        result = namespace.get("r")
        if not isinstance(result, cq.Workplane):
            raise ValueError("Generated code must assign a cadquery.Workplane to r")
        shapes = result.vals()
        if not shapes or any(not isinstance(shape, cq.Shape) for shape in shapes):
            raise ValueError("r must contain at least one CadQuery shape")
        shape = shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)
        if shape.isNull() or not shape.isValid() or not shape.Solids():
            raise ValueError("r must contain a nonempty, valid solid")
        if not math.isfinite(shape.Volume()) or shape.Volume() <= 0:
            raise ValueError("r must have positive finite volume")
        print("[cadquery] exporting STEP and STL", flush=True)
        cq.exporters.export(shape, str(directory / "candidate.tmp.step"))
        cq.exporters.export(shape, str(directory / "candidate.tmp.stl"), tolerance=0.001)
        (directory / "candidate.tmp.step").replace(directory / "candidate.step")
        (directory / "candidate.tmp.stl").replace(directory / "candidate.stl")
        payload = {"status": "valid", "error": None}
        return_code = 0
        print("[cadquery] valid exports ready", flush=True)
    except BaseException:
        error = traceback.format_exc()
        (directory / "traceback.txt").write_text(error, encoding="utf-8")
        payload = {"status": "execution_error", "error": error}
        return_code = 1
        print("[cadquery] execution failed; see traceback.txt", flush=True)
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    return return_code


if __name__ == "__main__":
    sys.exit(main())
