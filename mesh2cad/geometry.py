"""Small, explicit CPU-only helpers used by the Streamlit pipeline.

Everything here is deterministic and observable: meshes are loaded as-is,
normalization returns its matrix, and the LLM image has a fixed 2x2 layout.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import trimesh

VIEW_SIZE = 512
VIEWS = (
    ("Front", 0, -90),
    ("Side", 0, 0),
    ("Top", 90, -90),
    ("Isometric", 25, -55),
)
_FRAME = (-0.6, 0.6)


def validate_mesh(mesh: trimesh.Trimesh) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Ожидается треугольная сетка")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("В сетке нет треугольной поверхности")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Сетка содержит неконечные координаты")
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ValueError("Сетка содержит индексы граней вне диапазона")
    if not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Площадь поверхности не является конечной и положительной")


def _as_mesh(loaded: Any) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Scene):
        parts = [part for part in loaded.dump() if isinstance(part, trimesh.Trimesh)]
        if not parts:
            raise ValueError("Сцена не содержит треугольных сеток")
        loaded = trimesh.util.concatenate(parts)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("Файл не содержит треугольной сетки")
    return loaded


def load_mesh(source: str | Path | bytes | bytearray | BinaryIO) -> trimesh.Trimesh:
    """Load an STL from a path, raw bytes, or an already-open binary stream.

    Duplicate STL vertices are merged; no geometry is otherwise repaired.
    """
    if isinstance(source, (bytes, bytearray)):
        loaded = trimesh.load(io.BytesIO(bytes(source)), file_type="stl", process=False)
    elif hasattr(source, "read"):
        loaded = trimesh.load(source, file_type="stl", process=False)
    else:
        loaded = trimesh.load(str(source), process=False)
    mesh = _as_mesh(loaded).copy()
    validate_mesh(mesh)
    mesh.merge_vertices()
    validate_mesh(mesh)
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Center the bounding box at zero and make its longest side exactly one.

    The returned matrices map source -> normalized and normalized -> source.
    The source mesh is not modified.
    """
    validate_mesh(mesh)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extent = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Bounding box не имеет конечного положительного размера")
    transform = np.eye(4)
    transform[:3, :3] /= extent
    transform[:3, 3] = -center / extent
    inverse = np.eye(4)
    inverse[:3, :3] *= extent
    inverse[:3, 3] = center
    normalized = mesh.copy()
    normalized.apply_transform(transform)
    return normalized, {
        "transform": transform.tolist(),
        "inverse_transform": inverse.tolist(),
        "source_bounds": bounds.tolist(),
        "normalized_bounds": np.asarray(normalized.bounds, dtype=np.float64).tolist(),
    }


def mesh_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": bounds.tolist(),
        "extents": (bounds[1] - bounds[0]).tolist(),
        "area": float(mesh.area),
        "volume": float(mesh.volume) if mesh.is_volume else None,
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
    }


def evaluate_iou(target: trimesh.Trimesh, prediction: trimesh.Trimesh) -> dict[str, Any]:
    """Exact volume IoU through manifold booleans, with explicit failure reasons.

    IoU is ``|target ∩ prediction| / |target ∪ prediction|``. Geometry is used
    as-is: the prediction is never moved, rescaled, or repaired to fit.
    """
    validate_mesh(target)
    validate_mesh(prediction)
    result: dict[str, Any] = {
        "iou": None,
        "status": "ok",
        "target_volume": float(target.volume) if target.is_volume else None,
        "prediction_volume": float(prediction.volume) if prediction.is_volume else None,
        "intersection_volume": None,
        "union_volume": None,
        "target_watertight": bool(target.is_watertight),
        "prediction_watertight": bool(prediction.is_watertight),
    }
    target_closed, prediction_closed = target.is_watertight, prediction.is_watertight
    if not target_closed or not prediction_closed:
        reason = (
            "both"
            if not target_closed and not prediction_closed
            else ("target" if not target_closed else "prediction")
        )
        result["status"] = f"non_watertight_{reason}"
        return result
    if not target.is_volume or not prediction.is_volume:
        reason = (
            "both"
            if not target.is_volume and not prediction.is_volume
            else ("target" if not target.is_volume else "prediction")
        )
        result["status"] = f"invalid_volume_{reason}"
        return result
    if importlib.util.find_spec("manifold3d") is None:
        result["status"] = "manifold3d_unavailable"
        return result
    try:
        intersection = trimesh.boolean.intersection([target, prediction], engine="manifold")
        union = trimesh.boolean.union([target, prediction], engine="manifold")
        intersection_volume = float(intersection.volume)
        union_volume = float(union.volume)
        if not np.isfinite(union_volume) or union_volume <= 0:
            result["status"] = "invalid_union_volume"
            return result
        if not np.isfinite(intersection_volume) or intersection_volume < 0:
            result["status"] = "invalid_intersection_volume"
            return result
        ratio = intersection_volume / union_volume
        if ratio < -1e-6 or ratio > 1 + 1e-6:
            result["status"] = "invalid_iou_range"
            return result
        result.update(
            iou=float(np.clip(ratio, 0, 1)),
            intersection_volume=intersection_volume,
            union_volume=union_volume,
        )
        return result
    except Exception as exc:
        result["status"] = f"boolean_error:{type(exc).__name__}"
        return result


def _faces_and_colors(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    light = np.asarray([0.25, -0.45, 0.86], dtype=np.float64)
    light /= np.linalg.norm(light)
    brightness = 0.35 + 0.65 * np.abs(np.asarray(mesh.face_normals) @ light)
    colors = np.column_stack((0.30 * brightness, 0.61 * brightness, 0.84 * brightness))
    return np.asarray(mesh.triangles), colors


def _render_png(mesh: trimesh.Trimesh, size: int) -> bytes:
    if not isinstance(size, int) or isinstance(size, bool) or size < 64:
        raise ValueError("Размер изображения должен быть целым числом не меньше 64")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    validate_mesh(mesh)
    triangles, colors = _faces_and_colors(mesh)
    figure = Figure(figsize=(size / 100.0, size / 100.0), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    try:
        positions = (
            (0.00, 0.50, 0.50, 0.50),
            (0.50, 0.50, 0.50, 0.50),
            (0.00, 0.00, 0.50, 0.50),
            (0.50, 0.00, 0.50, 0.50),
        )
        for (label, elevation, azimuth), position in zip(VIEWS, positions, strict=True):
            axis = figure.add_axes(position, projection="3d")
            surface = Poly3DCollection(
                triangles,
                facecolors=colors,
                edgecolors="none",
                linewidths=0,
                rasterized=True,
            )
            axis.add_collection3d(surface)
            axis.set(xlim=_FRAME, ylim=_FRAME, zlim=_FRAME)
            axis.set_box_aspect((1, 1, 1))
            axis.set_proj_type("ortho")
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_axis_off()
            axis.text2D(
                0.03,
                0.98,
                label,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#333333",
            )
        width, height = canvas.get_width_height()
        if (width, height) != (size, size):
            raise RuntimeError(f"Canvas has {width}x{height}, expected {size}x{size}")
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        return buffer.getvalue()
    finally:
        figure.clear()


def render_views(mesh: trimesh.Trimesh, output_path: str | Path, size: int = VIEW_SIZE) -> None:
    """Write an exact ``size`` x ``size`` PNG with front/side/top/isometric views."""
    data = _render_png(mesh, size)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)


def render_views_bytes(mesh: trimesh.Trimesh, size: int = VIEW_SIZE) -> bytes:
    """Return the same PNG as :func:`render_views` without touching the disk."""
    return _render_png(mesh, size)
