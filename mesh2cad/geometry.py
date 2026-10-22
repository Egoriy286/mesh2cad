"""CPU-only mesh helpers and deterministic 512x512 RGB/depth renders.

The mesh is loaded as-is; normalization returns its matrix. The LLM receives
one 3x2 contact sheet of top/left/right, isometric, bottom, and front views;
each tile blends the RGB render with its depth map on a black background.
"""

from __future__ import annotations

import importlib.util
import io
import math
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import trimesh
from PIL import Image

from .observability import timed_stage


class GeometryCancelled(RuntimeError):
    """A bounded geometry batch observed cancellation or an expired deadline."""


VIEW_SIZE = 512
VIEWS = (
    ("top", 90.0, -90.0),
    ("left", 0.0, 180.0),
    ("right", 0.0, 0.0),
    ("isometric", -25.0, -55.0),
    ("bottom", -90.0, -90.0),
    ("front", 0.0, -90.0),
)
VIEW_NAMES = tuple(name for name, _elevation, _azimuth in VIEWS)
_FRAME_SCALE = 1.15
_SUPERSAMPLE = 2


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


def mesh_difference(
    target: trimesh.Trimesh,
    prediction: trimesh.Trimesh,
    *,
    threshold: float = 0.015,
    max_edge: float = 0.03,
    cancel_check=None,
) -> dict[str, Any]:
    """Split the target surface into covered and missing regions.

    Missing means a subdivided target face whose center is farther than
    ``threshold`` from the prediction surface. The returned arrays can be
    rendered as a gray target and an opaque red "missing" overlay.
    """
    validate_mesh(target)
    validate_mesh(prediction)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if not math.isfinite(max_edge) or max_edge <= 0:
        raise ValueError("max_edge must be positive and finite")
    detailed = target
    # Bound subdivision *before* allocating a potentially enormous mesh.
    for _ in range(8):
        triangles = detailed.triangles
        edges = triangles - np.roll(triangles, 1, axis=1)
        long = np.linalg.norm(edges, axis=2).max(axis=1) > max_edge
        if not long.any() or len(detailed.faces) + 3 * int(long.sum()) > 30_000:
            break
        vertices, faces = trimesh.remesh.subdivide(
            detailed.vertices, detailed.faces, face_index=np.flatnonzero(long)
        )
        detailed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if cancel_check is not None and cancel_check():
            raise GeometryCancelled("Surface comparison cancelled")
    centers = np.asarray(detailed.triangles_center, dtype=np.float64)
    try:
        # Only triangles within the threshold cube can make a face covered.
        # This gives the same missing/covered classification without expensive
        # nearest-surface searches for points far away from the candidate.
        tree = prediction.triangles_tree
        triangles = prediction.triangles
        distances = np.full(len(centers), np.inf)
        for start in range(0, len(centers), 128):
            if cancel_check is not None and cancel_check():
                raise GeometryCancelled("Surface comparison cancelled")
            point_ids, triangle_ids, count = [], [], 0

            def flush():
                if not point_ids:
                    return
                pi, ti = np.concatenate(point_ids), np.concatenate(triangle_ids)
                nearest = trimesh.triangles.closest_point(triangles[ti], centers[pi])
                np.minimum.at(distances, pi, np.linalg.norm(nearest - centers[pi], axis=1))
                point_ids.clear()
                triangle_ids.clear()

            for index in range(start, min(start + 128, len(centers))):
                point = centers[index]
                candidates = np.fromiter(
                    tree.intersection(np.r_[point - threshold, point + threshold]), dtype=np.int64
                )
                for offset in range(0, len(candidates), 50_000):
                    ids = candidates[offset : offset + 50_000]
                    triangle_ids.append(ids)
                    point_ids.append(np.full(len(ids), index, dtype=np.int64))
                    count += len(ids)
                    if count >= 50_000:
                        flush()
                        count = 0
            flush()
    except ModuleNotFoundError as exc:
        raise RuntimeError("Для контрастного наложения нужен rtree. Запустите `uv sync`.") from exc
    missing_mask = np.asarray(distances) > threshold
    covered_mask = ~missing_mask
    return {
        "threshold": float(threshold),
        "missing_faces": int(missing_mask.sum()),
        "covered_faces": int(covered_mask.sum()),
        "total_faces": int(len(missing_mask)),
        "missing_ratio": float(missing_mask.mean()),
        "max_distance": None if missing_mask.any() else float(distances.max()),
        "distance_status": "threshold_classification",
        "covered_vertices": detailed.vertices,
        "covered_triangles": detailed.faces[covered_mask],
        "missing_vertices": detailed.vertices,
        "missing_triangles": detailed.faces[missing_mask],
    }


def _camera_basis(elevation: float, azimuth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (forward, right, up) for an orthographic camera."""
    elevation_radians = math.radians(elevation)
    azimuth_radians = math.radians(azimuth)
    camera = np.asarray(
        [
            math.cos(elevation_radians) * math.cos(azimuth_radians),
            math.cos(elevation_radians) * math.sin(azimuth_radians),
            math.sin(elevation_radians),
        ],
        dtype=np.float64,
    )
    forward = -camera
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(forward @ world_up)) > 0.999:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return forward, right, up


def _project(vertices: np.ndarray, right: np.ndarray, up: np.ndarray, size: int):
    u = (vertices @ right) / _FRAME_SCALE + 0.5
    v = (vertices @ up) / _FRAME_SCALE + 0.5
    return u * (size - 1), (1.0 - v) * (size - 1)


def _raster_buffers(parts, elevation, azimuth, size, *, cancel_check=None):
    """Rasterize bounded NumPy batches instead of allocating a grid per triangle."""
    render_size = size * _SUPERSAMPLE
    forward, right, up = _camera_basis(elevation, azimuth)
    light = np.asarray([0.25, -0.45, 0.86], dtype=np.float64)
    light /= np.linalg.norm(light)
    projected, colors, validity = [], [], []
    depth_min, depth_max = np.inf, -np.inf
    for mesh, base_color, bias in parts:
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        x, y = _project(vertices, right, up, render_size)
        depth = vertices @ forward + bias
        depth_min, depth_max = min(depth_min, depth.min()), max(depth_max, depth.max())
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals[valid] /= lengths[valid, None]
        brightness = 0.55 + 0.45 * np.abs(normals @ light)
        colors.append((np.asarray(base_color)[None, :] * brightness[:, None]).astype(np.float32))
        projected.append(np.stack((x[faces], y[faces], depth[faces]), axis=1))
        validity.append(valid)
    projected = np.concatenate(projected)
    face_colors = np.concatenate(colors)
    valid = np.concatenate(validity)
    x, y, z = projected[:, 0], projected[:, 1], projected[:, 2]
    xmin = np.maximum(np.floor(x.min(axis=1)).astype(int), 0)
    xmax = np.minimum(np.ceil(x.max(axis=1)).astype(int) + 1, render_size)
    ymin = np.maximum(np.floor(y.min(axis=1)).astype(int), 0)
    ymax = np.minimum(np.ceil(y.max(axis=1)).astype(int) + 1, render_size)
    denominator = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (
        y[:, 0] - y[:, 2]
    )
    valid &= (xmax > xmin) & (ymax > ymin) & (np.abs(denominator) >= 1e-12)
    ids = np.flatnonzero(valid)
    widths, heights = xmax - xmin, ymax - ymin
    buckets = np.ceil(np.log2(widths[ids])).astype(int) * 32 + np.ceil(
        np.log2(heights[ids])
    ).astype(int)
    zbuffer = np.full(render_size * render_size, np.inf)
    empty = np.iinfo(np.int64).max
    winners = np.full(render_size * render_size, empty, dtype=np.int64)
    for bucket in np.unique(buckets):
        group = ids[buckets == bucket]
        width, height = int(widths[group].max()), int(heights[group].max())
        batch_size = max(1, 200_000 // (width * height))
        for offset in range(0, len(group), batch_size):
            if cancel_check is not None and cancel_check():
                raise GeometryCancelled("Rendering cancelled")
            batch = group[offset : offset + batch_size]
            gx = xmin[batch, None, None] + np.arange(width)[None, None, :] + 0.5
            gy = ymin[batch, None, None] + np.arange(height)[None, :, None] + 0.5

            def column(values):
                return values[batch, None, None]

            w0 = (
                column(y[:, 1] - y[:, 2]) * (gx - column(x[:, 2]))
                + column(x[:, 2] - x[:, 1]) * (gy - column(y[:, 2]))
            ) / column(denominator)
            w1 = (
                column(y[:, 2] - y[:, 0]) * (gx - column(x[:, 2]))
                + column(x[:, 0] - x[:, 2]) * (gy - column(y[:, 2]))
            ) / column(denominator)
            w2 = 1.0 - w0 - w1
            mask = (
                (w0 >= 0)
                & (w1 >= 0)
                & (w2 >= 0)
                & (gx < xmax[batch, None, None])
                & (gy < ymax[batch, None, None])
            )
            rows, py, px = np.nonzero(mask)
            if not len(rows):
                continue
            face_ids = batch[rows]
            flat = (ymin[face_ids] + py) * render_size + xmin[face_ids] + px
            depths = (
                w0[mask] * z[face_ids, 0] + w1[mask] * z[face_ids, 1] + w2[mask] * z[face_ids, 2]
            )
            changed = np.unique(flat[depths < zbuffer[flat]])
            np.minimum.at(zbuffer, flat, depths)
            winners[changed] = empty
            nearest = depths == zbuffer[flat]
            np.minimum.at(winners, flat[nearest], face_ids[nearest])
    color = np.zeros((render_size * render_size, 3), dtype=np.float32)
    covered = winners != empty
    color[covered] = face_colors[winners[covered]]
    image = Image.fromarray(
        np.clip(color.reshape(render_size, render_size, 3) * 255, 0, 255).astype(np.uint8)
    )
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image, zbuffer.reshape(render_size, render_size), float(depth_min), float(depth_max)


def _rasterize(mesh, elevation, azimuth, size, *, cancel_check=None):
    rgb, depth, low, high = _raster_buffers(
        [(mesh, (0.30, 0.68, 1.00), 0.0)], elevation, azimuth, size, cancel_check=cancel_check
    )
    covered = np.isfinite(depth)
    normalized = np.clip((depth - low) / (high - low), 0, 1) if high > low else np.zeros_like(depth)
    shade = np.zeros_like(normalized)
    shade[covered] = 4.0 + 140.0 * (1.0 - normalized[covered]) ** 1.6
    depth_image = Image.fromarray(np.clip(shade, 0, 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.LANCZOS
    )
    return rgb, depth_image.convert("RGB")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_llm_images(
    mesh: trimesh.Trimesh, size: int = VIEW_SIZE, *, cancel_check=None
) -> dict[str, bytes]:
    """Return 12 PNGs: RGB plus depth for each of the six views."""
    validate_mesh(mesh)
    if not isinstance(size, int) or isinstance(size, bool) or size < 64:
        raise ValueError("Размер изображения должен быть целым числом не меньше 64")
    images: dict[str, bytes] = {}
    for name, elevation, azimuth in VIEWS:
        with timed_stage("render.view", view=name, faces=len(mesh.faces), size=size):
            rgb, depth = _rasterize(mesh, elevation, azimuth, size, cancel_check=cancel_check)
        images[f"{name}-rgb"] = _png_bytes(rgb)
        images[f"{name}-depth"] = _png_bytes(depth)
    return images


def _combine_rgb_depth(rgb: bytes, depth: bytes) -> Image.Image:
    """Blend one RGB view and its depth map into a single equal-weight tile."""
    with Image.open(io.BytesIO(rgb)) as image:
        color = np.asarray(image.convert("RGB"), dtype=np.float32)
    with Image.open(io.BytesIO(depth)) as image:
        depth_values = np.asarray(image.convert("L"), dtype=np.float32)
    combined = 0.5 * color + 0.5 * depth_values[..., None]
    return Image.fromarray(np.clip(combined, 0, 255).astype(np.uint8))


def _rasterize_parts(parts, elevation, azimuth, size, *, cancel_check=None):
    return _raster_buffers(parts, elevation, azimuth, size, cancel_check=cancel_check)[0]


def render_overlay_sheet(
    difference: dict[str, Any],
    candidate: trimesh.Trimesh,
    size: int = VIEW_SIZE,
    *,
    cancel_check=None,
) -> bytes:
    """Render the 3x2 contrast scheme sent to the improvement request.

    Gray is the covered target surface, blue is the current CadQuery model,
    and red is the target region missing from CadQuery.
    """
    validate_mesh(candidate)
    if not isinstance(size, int) or isinstance(size, bool) or size < 64:
        raise ValueError("Размер изображения должен быть целым числом не меньше 64")
    parts: list[tuple[trimesh.Trimesh, tuple[float, float, float], float]] = [
        (candidate, (0.20, 0.55, 1.00), 0.0005),
    ]
    covered = trimesh.Trimesh(
        vertices=difference["covered_vertices"],
        faces=difference["covered_triangles"],
        process=False,
    )
    if len(covered.faces):
        parts.append((covered, (0.55, 0.55, 0.55), -0.0005))
    missing = trimesh.Trimesh(
        vertices=difference["missing_vertices"],
        faces=difference["missing_triangles"],
        process=False,
    )
    if len(missing.faces):
        parts.append((missing, (1.00, 0.05, 0.05), 0.001))
    columns = 3
    rows = math.ceil(len(VIEWS) / columns)
    sheet = Image.new("RGB", (size * columns, size * rows), "black")
    for index, (_name, elevation, azimuth) in enumerate(VIEWS):
        row, column = divmod(index, columns)
        with timed_stage(
            "render.overlay_view", view=_name, faces=sum(len(p[0].faces) for p in parts)
        ):
            tile = _rasterize_parts(parts, elevation, azimuth, size, cancel_check=cancel_check)
        sheet.paste(tile, (column * size, row * size))
    return _png_bytes(sheet)


def compose_views(images: dict[str, bytes], size: int = VIEW_SIZE) -> bytes:
    """Combine all views into the single image sent to the LLM.

    Layout: three columns, two rows (top/left/right, then isometric/bottom/front).
    Each tile is one ``size``-pixel blend of RGB and its depth map; the
    background is black. No labels or frames are added.
    """
    columns = 3
    rows = math.ceil(len(VIEWS) / columns)
    sheet = Image.new("RGB", (size * columns, size * rows), "black")
    for index, (name, _elevation, _azimuth) in enumerate(VIEWS):
        row, column = divmod(index, columns)
        combined = _combine_rgb_depth(images[f"{name}-rgb"], images[f"{name}-depth"])
        sheet.paste(combined, (column * size, row * size))
    return _png_bytes(sheet)


def render_views(mesh: trimesh.Trimesh, output_path: str | Path, size: int = VIEW_SIZE) -> None:
    """Write a 3x2 preview sheet of combined RGB+depth tiles, each ``size`` pixels."""
    data = render_views_bytes(mesh, size)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)


def render_views_bytes(mesh: trimesh.Trimesh, size: int = VIEW_SIZE) -> bytes:
    """Return the preview sheet without touching the disk."""
    return compose_views(render_llm_images(mesh, size), size)
