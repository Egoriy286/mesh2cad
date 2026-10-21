"""Bounded geometric measurements for text conditioning in the target frame.

These are deterministic mesh measurements and heuristic primitive candidates,
not learned features or a native mesh encoder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

HYBRID_VERSION = "hybrid-v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_FACES = 500_000
MAX_VERTICES = 1_500_000
MAX_SECTION_SEGMENTS = 2048
MAX_CONTOURS = 24
MAX_CONTOUR_POINTS = 128
MAX_OUTPUT_BYTES = 32768


class HybridCancelled(ValueError):
    """Geometry preparation was cancelled by its caller."""


def _rounded(value):
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _rounded(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError("Non-finite hybrid evidence")
        return round(float(value), 8)
    return value


def _section(mesh, axis, fraction):
    coordinate = float(mesh.bounds[0, axis] + mesh.extents[axis] * fraction)
    normal = np.eye(3)[axis]
    origin = normal * coordinate
    segments = trimesh.intersections.mesh_plane(mesh, normal, origin)
    # Canonical endpoints and segment order remove face-order dependence.
    edges = set()
    for segment in segments:
        keys = [tuple(np.round(point, 7)) for point in segment]
        if keys[0] != keys[1]:
            edges.add(tuple(sorted(keys)))
    ordered = sorted(edges)
    limited = len(ordered) > MAX_SECTION_SEGMENTS
    ordered = ordered[:MAX_SECTION_SEGMENTS]
    adjacency = {}
    for a, b in ordered:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    unused = set(ordered)
    contours = []
    omitted_contours = 0
    while unused:
        # Prefer open ends; branch vertices produce explicitly open paths.
        endpoints = sorted(
            k
            for k, neighbors in adjacency.items()
            if len(neighbors) != 2 and any(tuple(sorted((k, n))) in unused for n in neighbors)
        )
        start = endpoints[0] if endpoints else min(min(edge) for edge in unused)
        chain = [start]
        current = start
        while True:
            choices = sorted(n for n in adjacency[current] if tuple(sorted((current, n))) in unused)
            if not choices:
                break
            nxt = choices[0]
            unused.remove(tuple(sorted((current, nxt))))
            chain.append(nxt)
            current = nxt
            if current == start or len(adjacency[current]) != 2:
                break
        closed = len(chain) > 2 and chain[-1] == chain[0]
        if closed:
            chain.pop()
        total = len(chain)
        reduced = total > MAX_CONTOUR_POINTS
        if reduced:
            chain = [chain[i] for i in np.linspace(0, total - 1, MAX_CONTOUR_POINTS, dtype=int)]
        if len(contours) >= MAX_CONTOURS:
            omitted_contours += 1
            continue
        contours.append(
            {
                "points": [list(p) for p in chain],
                "closed": closed,
                "open": not closed,
                "truncated": reduced,
                "source_point_count": total,
            }
        )
    return {
        "axis": "XYZ"[axis],
        "fraction": fraction,
        "coordinate": coordinate,
        "segment_count": len(edges),
        "contours": contours,
        "truncated": limited or bool(omitted_contours) or any(c["truncated"] for c in contours),
        "omitted_contour_count": omitted_contours,
        "omitted_segment_count": max(0, len(edges) - MAX_SECTION_SEGMENTS),
        "point_semantics": "ordered contour samples; truncated points do not define exact edges",
        "branch_vertices": sum(len(n) > 2 for n in adjacency.values()),
        "contour_roles": "unclassified; nested closed contours can represent holes",
    }


def _planes(mesh):
    normals = np.asarray(mesh.face_normals).copy()
    triangles = mesh.triangles
    areas = mesh.area_faces
    # Canonical unoriented normal: this measures support without assuming winding.
    for axis in range(3):
        active = (np.abs(normals[:, :axis]) < 1e-8).all(axis=1)
        flip = active & (normals[:, axis] < -1e-8)
        normals[flip] *= -1
    offsets = np.einsum("ij,ij->i", normals, triangles[:, 0])
    labels = np.round(np.column_stack((normals, offsets)), 5)
    _, inverse = np.unique(labels, axis=0, return_inverse=True)
    group_area = np.bincount(inverse, weights=areas)
    candidates = np.argsort(-group_area, kind="stable")[:24]
    result = []
    for group in candidates:
        area = float(group_area[group])
        if area < mesh.area * 0.005:
            continue
        ids = np.flatnonzero(inverse == group)
        normal = np.average(normals[ids], weights=areas[ids], axis=0)
        normal /= np.linalg.norm(normal)
        offset = float(np.average(offsets[ids], weights=areas[ids]))
        residuals = triangles[ids] @ normal - offset
        result.append(
            {
                "normal": normal.tolist(),
                "offset": offset,
                "area": area,
                "face_count": len(ids),
                "residual_max": float(np.abs(residuals).max()),
                "support": "coplanar triangles; connectivity not asserted",
            }
        )
    return result


def _cylinders(sections):
    circles = []
    for section in sections:
        axis = "XYZ".index(section["axis"])
        other = [i for i in range(3) if i != axis]
        for contour in section["contours"]:
            if not contour["closed"] or contour["truncated"] or len(contour["points"]) < 12:
                continue
            xy = np.asarray(contour["points"])[:, other]
            matrix = np.column_stack((2 * xy, np.ones(len(xy))))
            if np.linalg.matrix_rank(matrix) < 3:
                continue
            fit = np.linalg.lstsq(matrix, (xy * xy).sum(axis=1), rcond=None)[0]
            radius_squared = float(fit[2] + fit[:2] @ fit[:2])
            if radius_squared <= 1e-10:
                continue
            radius = math.sqrt(radius_squared)
            radial = np.linalg.norm(xy - fit[:2], axis=1)
            error = float(np.sqrt(np.mean((radial - radius) ** 2)))
            angles = np.sort(np.arctan2(xy[:, 1] - fit[1], xy[:, 0] - fit[0]))
            gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
            # Squares and partial arcs cannot pass the circular-support checks.
            if error / radius > 0.015 or np.max(gaps) > math.pi / 3:
                continue
            circles.append((axis, section["coordinate"], fit[:2], radius, error))
    result, used = [], set()
    for i, circle in enumerate(circles):
        if i in used:
            continue
        axis, _, center, radius, _ = circle
        group = [
            j
            for j, c in enumerate(circles)
            if c[0] == axis
            and np.linalg.norm(c[2] - center) <= max(0.002, radius * 0.02)
            and abs(c[3] - radius) <= max(0.002, radius * 0.02)
        ]
        positions = sorted({circles[j][1] for j in group})
        if len(positions) < 2:
            continue
        stations = sorted(s["coordinate"] for s in sections if s["axis"] == "XYZ"[axis])
        # Equal endpoint circles alone also occur on spheres. Require adjacent
        # measured stations and reject contradictory intermediate circle fits.
        adjacent = any(a in positions and b in positions for a, b in zip(stations, stations[1:]))
        contradictory = False
        for station in stations:
            if not positions[0] < station < positions[-1]:
                continue
            same_center = [
                c
                for c in circles
                if c[0] == axis
                and c[1] == station
                and np.linalg.norm(c[2] - center) <= max(0.002, radius * 0.02)
            ]
            if same_center and not any(
                abs(c[3] - radius) <= max(0.002, radius * 0.02) for c in same_center
            ):
                contradictory = True
        if not adjacent or contradictory:
            continue
        used.update(group)
        origin = np.zeros(3)
        origin[[k for k in range(3) if k != axis]] = center
        result.append(
            {
                "kind": "candidate",
                "axis": np.eye(3)[axis].tolist(),
                "axis_point": origin.tolist(),
                "radius": radius,
                "fit_error_rms": max(circles[j][4] for j in group),
                "support_sections": len(positions),
                "support_positions": positions,
                "warning": "Circular-section agreement only; not a confirmed CAD cylinder",
            }
        )
        if len(result) == 12:
            break
    return result


def build_evidence(mesh: trimesh.Trimesh, target_sha256: str) -> dict:
    """Measure an original mesh already in normalized-target coordinates."""
    from mesh2cad.geometry import _validate_mesh

    if (
        not isinstance(target_sha256, str)
        or len(target_sha256) != 64
        or any(c not in "0123456789abcdef" for c in target_sha256)
    ):
        raise ValueError("Expected lowercase SHA256 provenance")
    if len(mesh.faces) > MAX_FACES or len(mesh.vertices) > MAX_VERTICES:
        raise ValueError("Hybrid geometry exceeds triangle/vertex limits")
    _validate_mesh(mesh)
    sections = [
        _section(mesh, axis, fraction) for axis in range(3) for fraction in (0.25, 0.5, 0.75)
    ]
    # Stable triangle and vertex order gives sampling invariance to face permutation.
    triangles = np.asarray(mesh.triangles)
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    canonical = np.take_along_axis(triangles, order[:, :, None], axis=1)
    indices = np.lexsort(canonical.reshape(-1, 9).T[::-1])
    areas = np.asarray(mesh.area_faces)[indices]
    generator = np.random.default_rng(42)
    selected = generator.choice(len(areas), size=64, p=areas / areas.sum())
    uv = generator.random((64, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    xyz = np.einsum("ij,ijk->ik", weights, canonical[indices[selected]])
    normals = mesh.face_normals[indices[selected]]
    warnings = ["Primitive hypotheses are measurements, not a CAD reconstruction."]
    if not mesh.is_watertight:
        warnings.append("Mesh is non-watertight; open sections are possible; no volume inferred.")
    if any(s["truncated"] for s in sections):
        warnings.append("Section output limits reached; contour detail or entire contours omitted.")
    if any(s["branch_vertices"] for s in sections):
        warnings.append("Branching section intersections split into paths; topology is ambiguous.")
    result = {
        "schema_version": HYBRID_VERSION,
        "target_sha256": target_sha256,
        "coordinate_frame": "normalized-target",
        "bounds": mesh.bounds.tolist(),
        "extents": mesh.extents.tolist(),
        "sections": sections,
        "planes": _planes(mesh),
        "cylinders": _cylinders(sections),
        "sampling": {
            "method": "area-uniform barycentric",
            "seed": 42,
            "count": 64,
            "normals": "source triangle winding; orientation not repaired",
        },
        "point_samples": [{"xyz": p.tolist(), "normal": n.tolist()} for p, n in zip(xyz, normals)],
        "warnings": warnings,
    }
    result = _rounded(result)
    _limit_evidence(result)
    return result


def _compact_size(result):
    return len(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _limit_evidence(result):
    """Bound prompt bytes, preserving observed closure and explicit omissions.

    Sparse contour points are samples along the original path; they are not
    replacement polygon edges, and no topology is inferred from simplification.
    """
    if _compact_size(result) <= MAX_OUTPUT_BYTES:
        return
    result["warnings"].append(
        "32 KiB evidence budget reached; contour samples reduced or contours omitted; "
        "section topology and exact contour edges cannot be reconstructed from sparse samples."
    )
    while _compact_size(result) > MAX_OUTPUT_BYTES:
        choices = [
            (len(c["points"]), si, ci)
            for si, section in enumerate(result["sections"])
            for ci, c in enumerate(section["contours"])
            if len(c["points"]) > 8
        ]
        if choices:
            _, si, ci = max(choices)
            section = result["sections"][si]
            contour = section["contours"][ci]
            points = contour["points"]
            count = max(8, len(points) // 2)
            contour["points"] = [
                points[i] for i in np.linspace(0, len(points) - 1, count, dtype=int)
            ]
            contour["truncated"] = True
            section["truncated"] = True
            continue
        choices = [
            (len(section["contours"]), si)
            for si, section in enumerate(result["sections"])
            if section["contours"]
        ]
        if choices:
            _, si = max(choices)
            section = result["sections"][si]
            section["contours"].pop()
            section["omitted_contour_count"] += 1
            section["truncated"] = True
            continue
        # With fixed caps for samples, planes and cylinders this cannot normally
        # be reached. Refuse evidence rather than silently strip other fields.
        raise ValueError("Hybrid evidence metadata exceeds prompt byte limit")


def _read_valid(path, digest):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_OUTPUT_BYTES:
        return None
    try:
        result = json.loads(
            path.read_text(),
            parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Non-finite JSON")),
        )
        if (
            isinstance(result, dict)
            and result.get("schema_version") == HYBRID_VERSION
            and result.get("target_sha256") == digest
            and result.get("coordinate_frame") == "normalized-target"
            and all(
                k in result
                for k in (
                    "bounds",
                    "extents",
                    "sections",
                    "planes",
                    "cylinders",
                    "sampling",
                    "warnings",
                )
            )
        ):
            if _compact_size(result) <= MAX_OUTPUT_BYTES:
                return result
    except (ValueError, OSError, TypeError):
        pass
    return None


def prepare_evidence(
    target_path: Path,
    output_path: Path,
    cancel_check: Callable[[], bool] | None = None,
    timeout: float = 25,
) -> dict:
    """Extract in a resource-limited process; atomically cache exact-file evidence."""
    target_path, output_path = Path(target_path), Path(output_path)
    if (
        target_path.is_symlink()
        or output_path.is_symlink()
        or target_path.parent.is_symlink()
        or output_path.parent.is_symlink()
    ):
        raise ValueError("Hybrid target/cache paths must not be symlinks")
    target_path, output_path = target_path.resolve(), output_path.resolve()
    if target_path == output_path:
        raise ValueError("Hybrid output must differ from target")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Hybrid timeout must be positive")

    def check_cancel():
        if cancel_check and cancel_check():
            raise HybridCancelled("Hybrid geometry preparation cancelled")

    check_cancel()
    if target_path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Hybrid input exceeds file limit")
    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    cached = _read_valid(output_path, digest)
    if cached is not None:
        check_cancel()
        return cached
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hybrid-", dir=output_path.parent) as directory:
        temporary = Path(directory) / "evidence.json"
        process = subprocess.Popen(
            [sys.executable, "-m", "api.hybrid_worker", str(target_path), str(temporary)],
            cwd=Path(__file__).resolve().parents[1],
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": directory,
                "TMPDIR": directory,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                check_cancel()
                if time.monotonic() - started > timeout:
                    raise ValueError("Hybrid geometry preparation timed out")
                time.sleep(0.05)
            check_cancel()
            if process.returncode:
                raise ValueError("Hybrid geometry preparation failed within resource limits")
            result = _read_valid(temporary, digest)
            if result is None:
                raise ValueError("Hybrid worker returned invalid or stale evidence")
            # Detect a target changed during extraction, including a cache race.
            if hashlib.sha256(target_path.read_bytes()).hexdigest() != digest:
                raise ValueError("Hybrid target changed during extraction")
            os.replace(temporary, output_path)
            return result
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
