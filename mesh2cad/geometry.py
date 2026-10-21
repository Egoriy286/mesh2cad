"""CPU-only mesh preparation, shared-camera images, and geometric evaluation.

Only the input mesh is normalized. Predictions are measured in the coordinates
requested in the generation prompt; moving or rescaling them would hide errors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

_VIEWS = (
    ("Front (X/Z)", 0, -90),
    ("Side (Y/Z)", 0, 0),
    ("Top (X/Y)", 90, -90),
    ("Isometric", 25, -55),
)
_FRAME = (-0.6, 0.6)


def _validate_mesh(mesh: trimesh.Trimesh) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected a triangular mesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no triangular surface")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh contains non-finite vertices")
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ValueError("Mesh contains out-of-range face indices")
    if not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Mesh has no finite positive surface area")


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load triangles, retain scene transforms, and merge STL duplicate vertices.

    Merging identical vertices makes STL watertightness meaningful. No holes,
    face winding, position, or scale are repaired here.
    """
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        # dump applies each scene instance's world transform before concatenation.
        parts = [part for part in loaded.dump() if isinstance(part, trimesh.Trimesh)]
        if not parts:
            raise ValueError("Scene contains no triangular meshes")
        loaded = trimesh.util.concatenate(parts)
    _validate_mesh(loaded)
    loaded = loaded.copy()
    loaded.merge_vertices()
    _validate_mesh(loaded)
    return loaded


def normalize_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Center the bounding box at zero and make its longest side one unit.

    The returned homogeneous matrices map source -> normalized coordinates and
    normalized -> source coordinates respectively. The source mesh is untouched.
    """
    _validate_mesh(mesh)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extent = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Mesh bounding box has no finite positive extent")
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
    }


def _render(meshes: list[tuple[str, trimesh.Trimesh]], output_path: Path) -> None:
    # FigureCanvasAgg avoids pyplot, a display server, and GPU/OpenGL dependencies.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for _, mesh in meshes:
        _validate_mesh(mesh)
    figure = Figure(figsize=(12, 3.2 * len(meshes)), dpi=120, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    for row, (label, mesh) in enumerate(meshes):
        # Fixed face lighting preserves shape cues without random materials.
        light = np.asarray([0.25, -0.45, 0.86])
        light /= np.linalg.norm(light)
        brightness = 0.35 + 0.65 * np.abs(mesh.face_normals @ light)
        colors = np.column_stack((0.30 * brightness, 0.61 * brightness, 0.84 * brightness))
        for column, (name, elevation, azimuth) in enumerate(_VIEWS):
            axis = figure.add_subplot(len(meshes), 4, row * 4 + column + 1, projection="3d")
            surface = Poly3DCollection(
                mesh.triangles,
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
            axis.set_title(f"{label} — {name}" if label else name, fontsize=10, pad=0)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92, wspace=0, hspace=0.12)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        canvas.print_png(str(output_path))
    finally:
        figure.clear()


def render_views(mesh: trimesh.Trimesh, output_path: Path) -> None:
    """Write a PNG containing front, side, top, and isometric orthographic views."""
    _render([("", mesh)], output_path)


def render_comparison(
    target: trimesh.Trimesh, prediction: trimesh.Trimesh, output_path: Path
) -> None:
    """Write target/prediction rows using identical views and fixed unit framing."""
    _render([("Target", target), ("Prediction", prediction)], output_path)


def _sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    """Sample triangles by area and points uniformly by barycentric coordinates."""
    generator = np.random.default_rng(seed)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    faces = generator.choice(len(areas), size=count, p=areas / areas.sum())
    triangles = mesh.triangles[faces]
    uv = generator.random((count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    return np.einsum("ij,ijk->ik", weights, triangles)


def _volume_iou(target: trimesh.Trimesh, prediction: trimesh.Trimesh) -> tuple[float | None, str]:
    target_closed, prediction_closed = target.is_watertight, prediction.is_watertight
    if not target_closed or not prediction_closed:
        reason = (
            "both"
            if not target_closed and not prediction_closed
            else ("target" if not target_closed else "prediction")
        )
        return None, f"non_watertight_{reason}"
    if not target.is_volume or not prediction.is_volume:
        reason = (
            "both"
            if not target.is_volume and not prediction.is_volume
            else ("target" if not target.is_volume else "prediction")
        )
        return None, f"invalid_volume_{reason}"
    if importlib.util.find_spec("manifold3d") is None:
        return None, "manifold3d_unavailable"
    try:
        intersection = trimesh.boolean.intersection([target, prediction], engine="manifold")
        union = trimesh.boolean.union([target, prediction], engine="manifold")
        intersection_volume = float(intersection.volume)
        union_volume = float(union.volume)
        if not np.isfinite(union_volume) or union_volume <= 0:
            return None, "invalid_union_volume"
        if not np.isfinite(intersection_volume) or intersection_volume < 0:
            return None, "invalid_intersection_volume"
        ratio = intersection_volume / union_volume
        if ratio < -1e-6 or ratio > 1 + 1e-6:
            return None, "invalid_iou_range"
        # Manifold uses floating-point vertices; tolerate rounding at the bounds.
        return float(np.clip(ratio, 0, 1)), "ok"
    except Exception as exc:
        return None, f"boolean_error:{type(exc).__name__}"


def evaluate_meshes(
    target: trimesh.Trimesh,
    prediction: trimesh.Trimesh,
    n_points: int = 8192,
    seed: int = 42,
) -> dict[str, Any]:
    """Return sampled symmetric squared Chamfer and exact boolean volume IoU.

    Chamfer is mean(d(target, prediction)^2) + mean(d(prediction, target)^2),
    with no multiplier or independent alignment. A local RNG makes repeated
    evaluations reproducible without changing global NumPy state. IoU is null
    when surfaces do not bound valid closed solids or the boolean engine fails.
    """
    _validate_mesh(target)
    _validate_mesh(prediction)
    if not isinstance(n_points, (int, np.integer)) or isinstance(n_points, bool) or n_points <= 0:
        raise ValueError("n_points must be a positive integer")
    target_points = _sample_surface(target, int(n_points), seed)
    prediction_points = _sample_surface(prediction, int(n_points), seed)
    target_distances = cKDTree(prediction_points).query(target_points, k=1)[0]
    prediction_distances = cKDTree(target_points).query(prediction_points, k=1)[0]
    chamfer = float(np.mean(target_distances**2) + np.mean(prediction_distances**2))
    if not np.isfinite(chamfer):
        raise ValueError("Chamfer distance is non-finite")
    iou, iou_status = _volume_iou(target, prediction)
    return {
        "chamfer": chamfer,
        "iou": iou,
        "iou_status": iou_status,
        "target_watertight": bool(target.is_watertight),
        "prediction_watertight": bool(prediction.is_watertight),
    }
