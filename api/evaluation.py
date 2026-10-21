"""Comparable, deterministic evaluations in the target coordinate frame."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mesh2cad.geometry import _sample_surface, evaluate_meshes, load_mesh
from mesh2cad.storage import file_digest

METRIC_VERSION = "workbench-v1"
MAX_METRIC_FILE_BYTES = 128 * 1024 * 1024


def evaluate_pair(
    target_path: str | Path,
    prediction_path: str | Path,
    n_points: int = 8192,
    seed: int = 42,
    threshold: float = 0.01,
) -> dict:
    """Evaluate without centering, scaling, or aligning either input.

    Precision samples the prediction; recall samples the target. H95 is the
    maximum of the two directional 95th percentiles, not exact Hausdorff.
    Distances/threshold are in the input coordinates (normalized target units
    for the workbench). Volume errors remain unavailable on non-volume meshes.
    """
    if not isinstance(n_points, int) or isinstance(n_points, bool) or not 1 <= n_points <= 131072:
        raise ValueError("n_points must be an integer between 1 and 131072")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    target_path, prediction_path = Path(target_path), Path(prediction_path)
    for path in (target_path, prediction_path):
        if path.stat().st_size > MAX_METRIC_FILE_BYTES:
            raise ValueError("Evaluation input exceeds the 128 MiB geometry limit")
    target, prediction = load_mesh(target_path), load_mesh(prediction_path)
    result = evaluate_meshes(target, prediction, n_points=n_points, seed=seed)
    target_points = _sample_surface(target, n_points, seed)
    prediction_points = _sample_surface(prediction, n_points, seed)
    to_prediction = cKDTree(prediction_points).query(target_points, k=1)[0]
    to_target = cKDTree(target_points).query(prediction_points, k=1)[0]
    precision = float(np.mean(to_target <= threshold))
    recall = float(np.mean(to_prediction <= threshold))
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    valid_volumes = bool(target.is_volume and prediction.is_volume)
    target_volume, prediction_volume = float(target.volume), float(prediction.volume)
    volume_error = (
        abs(prediction_volume - target_volume) / target_volume
        if valid_volumes and target_volume > 0
        else None
    )
    result.update(
        precision=precision,
        recall=recall,
        f_score=f_score,
        h95=float(max(np.quantile(to_prediction, 0.95), np.quantile(to_target, 0.95))),
        volume_relative_error=volume_error,
        volume_error_status="ok" if volume_error is not None else "requires_valid_closed_volumes",
        bbox_extents_error=np.abs(prediction.extents - target.extents).tolist(),
        bbox_center_distance=float(np.linalg.norm(prediction.bounds.mean(0) - target.bounds.mean(0))),
        n_points=n_points,
        seed=seed,
        threshold=float(threshold),
        metric_version=METRIC_VERSION,
        coordinate_frame="normalized_target_unaligned",
        target_sha256=file_digest(target_path),
        prediction_sha256=file_digest(prediction_path),
        target_stats=_stats(target),
        prediction_stats=_stats(prediction),
    )
    return result


def _stats(mesh) -> dict:
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "components": int(mesh.body_count),
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "area": float(mesh.area),
        "volume": float(mesh.volume) if mesh.is_volume else None,
        "extents": mesh.extents.tolist(),
        "bounds": mesh.bounds.tolist(),
    }
