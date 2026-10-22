"""CPU-only mesh helpers and deterministic 512x512 RGB/depth renders.

The mesh is loaded as-is; normalization returns its matrix. The LLM receives
one contact sheet: top/left/right columns, RGB row above its depth row, all on
a black background.
"""

from __future__ import annotations

import importlib.util
import io
import math
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

VIEW_SIZE = 512
VIEWS = (
    ("top", 90.0, -90.0),
    ("left", 0.0, 180.0),
    ("right", 0.0, 0.0),
)
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


def _rasterize(
    mesh: trimesh.Trimesh,
    elevation: float,
    azimuth: float,
    size: int,
) -> tuple[Image.Image, Image.Image]:
    """Render one orthographic view as a paired RGB image and depth map."""
    render_size = size * _SUPERSAMPLE
    forward, right, up = _camera_basis(elevation, azimuth)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    x, y = _project(vertices, right, up, render_size)
    depth = vertices @ forward
    depth_min, depth_max = float(depth.min()), float(depth.max())

    zbuffer = np.full(render_size * render_size, np.inf, dtype=np.float64)
    color = np.zeros((render_size * render_size, 3), dtype=np.float32)

    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    light = np.asarray([0.25, -0.45, 0.86], dtype=np.float64)
    light /= np.linalg.norm(light)
    brightness = 0.55 + 0.45 * np.abs(normals @ light)
    face_colors = np.column_stack((0.30 * brightness, 0.68 * brightness, 1.00 * brightness)).astype(
        np.float32
    )

    for index, face in enumerate(faces):
        if not valid[index]:
            continue
        x0, x1, x2 = x[face]
        y0, y1, y2 = y[face]
        z0, z1, z2 = depth[face]
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))) + 1, render_size)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))) + 1, render_size)
        if xmin >= xmax or ymin >= ymax:
            continue
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-12:
            continue
        pixel_x = np.arange(xmin, xmax) + 0.5
        pixel_y = np.arange(ymin, ymax) + 0.5
        grid_x, grid_y = np.meshgrid(pixel_x, pixel_y)
        weight0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
        weight1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
        weight2 = 1.0 - weight0 - weight1
        mask = (weight0 >= 0) & (weight1 >= 0) & (weight2 >= 0)
        if not mask.any():
            continue
        pixel_depth = weight0 * z0 + weight1 * z1 + weight2 * z2
        rows, columns = np.nonzero(mask)
        flat = (rows + ymin) * render_size + (columns + xmin)
        values = pixel_depth[mask]
        closer = values < zbuffer[flat]
        if not closer.any():
            continue
        flat = flat[closer]
        zbuffer[flat] = values[closer]
        color[flat] = face_colors[index]

    rgb = color.reshape(render_size, render_size, 3)
    rgb_image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
    rgb_image = rgb_image.resize((size, size), Image.Resampling.LANCZOS)

    depth_buffer = zbuffer.reshape(render_size, render_size)
    covered = np.isfinite(depth_buffer)
    if depth_max > depth_min:
        normalized = np.clip(
            (depth_buffer - depth_min) / (depth_max - depth_min),
            0.0,
            1.0,
        )
    else:
        normalized = np.zeros_like(depth_buffer)
    shade = np.zeros_like(normalized)
    # Near is lighter, far is darker; the map stays deliberately dark and the
    # background remains pure black.
    shade[covered] = 4.0 + 140.0 * (1.0 - normalized[covered]) ** 1.6
    depth_image = Image.fromarray(np.clip(shade, 0, 255).astype(np.uint8))
    depth_image = depth_image.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")
    return rgb_image, depth_image


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_llm_images(mesh: trimesh.Trimesh, size: int = VIEW_SIZE) -> dict[str, bytes]:
    """Return six PNGs: top/left/right RGB plus a depth map for each view."""
    validate_mesh(mesh)
    if not isinstance(size, int) or isinstance(size, bool) or size < 64:
        raise ValueError("Размер изображения должен быть целым числом не меньше 64")
    images: dict[str, bytes] = {}
    for name, elevation, azimuth in VIEWS:
        rgb, depth = _rasterize(mesh, elevation, azimuth, size)
        images[f"{name}-rgb"] = _png_bytes(rgb)
        images[f"{name}-depth"] = _png_bytes(depth)
    return images


def compose_views(images: dict[str, bytes], size: int = VIEW_SIZE) -> bytes:
    """Combine the six PNGs into the single image sent to the LLM.

    Layout: three columns (top, left, right), RGB in the upper row and the
    matching depth map directly below it. Each tile is exactly ``size`` pixels.
    """
    sheet = Image.new("RGB", (size * len(VIEWS), size * 2), "black")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=max(12, size // 24))
    except TypeError:
        font = ImageFont.load_default()
    for column, (name, _elevation, _azimuth) in enumerate(VIEWS):
        with Image.open(io.BytesIO(images[f"{name}-rgb"])) as image:
            sheet.paste(image.convert("RGB"), (column * size, 0))
        with Image.open(io.BytesIO(images[f"{name}-depth"])) as image:
            sheet.paste(image.convert("RGB"), (column * size, size))
        draw.text((column * size + 12, 12), f"{name} RGB", fill="white", font=font)
        draw.text((column * size + 12, size + 12), f"{name} depth", fill="white", font=font)
    return _png_bytes(sheet)


def render_views(mesh: trimesh.Trimesh, output_path: str | Path, size: int = VIEW_SIZE) -> None:
    """Write a preview sheet: RGB row over depth row, each tile ``size`` pixels."""
    data = render_views_bytes(mesh, size)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)


def render_views_bytes(mesh: trimesh.Trimesh, size: int = VIEW_SIZE) -> bytes:
    """Return the preview sheet without touching the disk."""
    return compose_views(render_llm_images(mesh, size), size)
