"""Approximate display surfaces; never used for exports or evaluation."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import trimesh

_CACHE_SIZE = 4
_CACHE: OrderedDict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()
_LOCK = threading.Lock()


def decimated_preview(mesh: trimesh.Trimesh, limit: int = 30_000):
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 30_000:
        raise ValueError("Preview face limit must be an integer between 1 and 30000")
    if len(mesh.faces) <= limit:
        return mesh, False
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    if (
        len(faces) > 500_000
        or len(vertices) > 1_500_000
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Preview input exceeds finite triangle geometry limits")
    digest = hashlib.sha256()
    digest.update(str((vertices.shape, faces.shape)).encode("ascii"))
    digest.update(memoryview(vertices))
    digest.update(memoryview(faces))
    key = digest.hexdigest(), limit
    # Serialize expensive subprocesses and share the four most recent results.
    # Cache values are isolated arrays; callers receive copies, never originals.
    with _LOCK:
        arrays = _CACHE.get(key)
        if arrays is None:
            with tempfile.TemporaryDirectory(prefix="mesh2cad-preview-") as directory:
                folder = Path(directory)
                source, destination = folder / "input.npz", folder / "output.npz"
                np.savez(source, vertices=vertices, faces=faces)
                try:
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "api.preview_worker",
                            str(source),
                            str(destination),
                            str(limit),
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        env={
                            "PATH": "/usr/bin:/bin",
                            "HOME": str(folder),
                            "TMPDIR": str(folder),
                            "OPENBLAS_NUM_THREADS": "1",
                            "OMP_NUM_THREADS": "1",
                            "VTK_SMP_MAX_THREADS": "1",
                        },
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=18,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ValueError("Surface preview exceeded its 18 second limit") from exc
                if (
                    process.returncode
                    or not destination.is_file()
                    or destination.stat().st_size > 8 * 1024 * 1024
                ):
                    raise ValueError(
                        "Surface preview could not be generated within resource limits"
                    )
                with np.load(destination, allow_pickle=False) as output:
                    reduced_vertices = output["vertices"].copy()
                    reduced_faces = output["faces"].copy()
                if (
                    reduced_vertices.ndim != 2
                    or reduced_vertices.shape[1] != 3
                    or reduced_faces.ndim != 2
                    or reduced_faces.shape[1] != 3
                    or not 0 < len(reduced_faces) <= limit
                    or len(reduced_vertices) > 3 * limit
                    or not np.isfinite(reduced_vertices).all()
                    or reduced_faces.min() < 0
                    or reduced_faces.max() >= len(reduced_vertices)
                ):
                    raise ValueError("Surface preview returned invalid geometry")
                reduced_vertices.setflags(write=False)
                reduced_faces.setflags(write=False)
                arrays = reduced_vertices, reduced_faces
            _CACHE[key] = arrays
            if len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
        _CACHE.move_to_end(key)
        preview = trimesh.Trimesh(vertices=arrays[0].copy(), faces=arrays[1].copy(), process=False)
    preview.metadata["preview"] = {
        "method": "vtkQuadricDecimation",
        "approximate": True,
        "topology_preservation": "not_guaranteed",
        "source_faces": len(mesh.faces),
    }
    return preview, True
