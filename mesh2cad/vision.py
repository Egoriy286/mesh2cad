"""Calibrated linear depth sheets and numeric comparison metrics."""

from __future__ import annotations

import io
import itertools
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import VIEW_SIZE, VIEWS, _camera_basis, _raster_buffers
from .observability import timed_stage
from .storage import digest, read_json, write_json

VISION_VERSION = 3
FRAME_WIDTH = 1.15


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _sheet(tiles: list[Image.Image], size: int) -> Image.Image:
    sheet = Image.new("L", (size * 3, size * 2))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * size, (index // 3) * size))
    return sheet


def render_observations(
    mesh, reference_extents, *, size: int = VIEW_SIZE, cancel_check=None
) -> dict:
    """Use the SAME camera, frame and depth range for reference and prediction.

    Raw depth is the visible surface's dot product with camera.forward in the
    normalized world frame. Black is empty. Finite depth maps linearly to 1..255,
    with nearer surfaces brighter. Candidate coordinates are never changed.
    """
    extents = np.asarray(reference_extents, dtype=float)
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * extents / 2
    # A cube's isometric projection exceeds its longest axis. Fit the target bbox
    # once for ALL sheets, then retain that exact frame for every candidate.
    frame_width = max(
        FRAME_WIDTH,
        1.08
        * max(
            float(np.abs(axis) @ extents)
            for _, elevation, azimuth in VIEWS
            for axis in _camera_basis(elevation, azimuth)[1:]
        ),
    )
    depth_tiles, calibration, raw = [], [], {}
    for name, elevation, azimuth in VIEWS:
        with timed_stage("vision.render", view=name, faces=len(mesh.faces)):
            _, high_res, _, _ = _raster_buffers(
                [(mesh, (0.30, 0.68, 1.00), 0.0)],
                elevation,
                azimuth,
                size,
                cancel_check=cancel_check,
                frame_width=frame_width,
                render_color=False,
            )
            factor = high_res.shape[0] // size
            depth = high_res.reshape(size, factor, size, factor).min(axis=(1, 3)).astype(np.float32)
            forward, right, up = _camera_basis(elevation, azimuth)
            projected = corners @ forward
            near, far = float(projected.min() - 0.05), float(projected.max() + 0.05)
            mask = np.isfinite(depth)
            pixels = np.zeros((size, size), dtype=np.uint8)
            pixels[mask] = np.rint(
                1 + 254 * np.clip((far - depth[mask]) / (far - near), 0, 1)
            ).astype(np.uint8)
            depth_tiles.append(Image.fromarray(pixels))
            raw[name] = depth
            values = depth[mask]
            levels = []
            if len(values):
                # Direct measurements, not fitted parameters: planar surfaces
                # form repeated visible depths. Quantization is 0.001 world units.
                unique, counts = np.unique(np.round(values, 3), return_counts=True)
                for index in np.argsort(-counts)[:8]:
                    fraction = float(counts[index] / len(values))
                    if fraction >= 0.01:
                        levels.append(
                            {
                                "depth": round(float(unique[index]), 3),
                                "fraction": round(fraction, 4),
                            }
                        )
            calibration.append(
                {
                    "view": name,
                    "camera_direction": (-forward).round(8).tolist(),
                    "image_right": right.round(8).tolist(),
                    "image_up": up.round(8).tolist(),
                    "forward": forward.round(8).tolist(),
                    "near": near,
                    "far": far,
                    "occupied_pixels": int(mask.sum()),
                    "visible_depth_range": [float(values.min()), float(values.max())]
                    if len(values)
                    else None,
                    "dominant_depth_levels": levels,
                }
            )
    return {
        "depth": _png(_sheet(depth_tiles, size)),
        "raw_depth": raw,
        "calibration": {
            "version": VISION_VERSION,
            "size": size,
            "frame_width": frame_width,
            "reference_extents": extents.tolist(),
            "tile_order": [v[0] for v in VIEWS],
            "encoding": "0=empty; 1..255 linear; near=255, far=1; raw=world_dot_forward",
            "views": calibration,
        },
    }


def save_observations(bundle: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "depth.png").write_bytes(bundle["depth"])
    np.savez_compressed(folder / "depth.npz", **bundle["raw_depth"])
    write_json(folder / "calibration.json", bundle["calibration"])


def load_observations(folder: Path) -> dict:
    with np.load(folder / "depth.npz", allow_pickle=False) as depths:
        raw = {name: depths[name] for name, _, _ in VIEWS}
    return {
        "depth": (folder / "depth.png").read_bytes(),
        "raw_depth": raw,
        "calibration": read_json(folder / "calibration.json"),
    }


def observation_fingerprint(mesh, reference_extents) -> str:
    # File bytes/checksums already validated by the harness; this guards renderer and frame changes.
    return digest(
        {
            "version": VISION_VERSION,
            "extents": list(reference_extents),
            "bounds": mesh.bounds.tolist(),
            "faces": len(mesh.faces),
            "vertices": len(mesh.vertices),
        }
    )


def compare_observations(reference: dict, prediction: dict, *, threshold: float = 0.01) -> dict:
    if any(
        reference["calibration"][key] != prediction["calibration"][key]
        for key in ("version", "size", "frame_width", "reference_extents", "tile_order", "encoding")
    ):
        raise ValueError("Observation frames differ")
    for a, b in zip(
        reference["calibration"]["views"], prediction["calibration"]["views"], strict=True
    ):
        if any(a[key] != b[key] for key in ("forward", "near", "far", "image_right", "image_up")):
            raise ValueError("Depth calibrations differ")
    metrics = []
    for name, _, _ in VIEWS:
        a, b = reference["raw_depth"][name], prediction["raw_depth"][name]
        am, bm = np.isfinite(a), np.isfinite(b)
        shared, union = am & bm, am | bm
        difference = np.zeros_like(a)
        np.subtract(b, a, out=difference, where=shared)
        missing = (am & ~bm) | (shared & (difference > threshold))
        excess = (bm & ~am) | (shared & (difference < -threshold))
        silhouette_iou = float(shared.sum() / union.sum()) if union.any() else None
        mae = float(np.abs(difference[shared]).mean()) if shared.any() else None
        metrics.append(
            {
                "view": name,
                "silhouette_iou": silhouette_iou,
                "depth_mae": mae,
                "target_visible_pixels": int(am.sum()),
                "prediction_visible_pixels": int(bm.sum()),
                "missing_or_farther_pixels": int(missing.sum()),
                "excess_or_nearer_pixels": int(excess.sum()),
            }
        )
    return {
        "metrics": {
            "threshold": threshold,
            "views": metrics,
            "description": "Depth and silhouette metrics of visible surfaces; not volume IoU",
        },
    }
