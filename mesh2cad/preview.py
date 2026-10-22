"""Bounded display meshes and full-resolution comparison of saved run artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .geometry import load_mesh, normalize_mesh
from .observability import log_event
from .resources import DEFAULT_MEMORY_MB, monitor_process, worker_environment
from .storage import read_json


def preview_mesh(path: Path, *, normalize: bool = False, max_faces: int = 40_000) -> dict:
    """Thin only the display triangles; keep coordinates and evaluation inputs intact."""
    if max_faces < 1:
        raise ValueError("max_faces must be positive")
    mesh = load_mesh(path)
    log_event("preview.loaded", path=path, faces=len(mesh.faces))
    if normalize:
        mesh, _ = normalize_mesh(mesh)
    count = len(mesh.faces)
    faces = np.asarray(mesh.faces)
    if count > max_faces:
        faces = faces[np.linspace(0, count - 1, max_faces, dtype=int)]
    indices, inverse = np.unique(faces, return_inverse=True)
    return {
        "vertices": np.asarray(mesh.vertices)[indices],
        "faces": inverse.reshape((-1, 3)),
        "total_faces": count,
        "bounds": np.asarray(mesh.bounds),
    }


def comparison_figure(
    target: dict,
    candidate: dict | None,
    *,
    mode: str = "overlay",
    target_opacity: float = 0.35,
    candidate_opacity: float = 0.85,
    revision: str = "preview",
) -> go.Figure:
    if mode not in {"overlay", "side_by_side", "target", "candidate"}:
        raise ValueError("Unknown preview mode")
    side = mode == "side_by_side"
    figure = (
        make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=["Исходник", "CadQuery"],
        )
        if side
        else go.Figure()
    )
    for data, name, color, opacity, column in (
        (target, "Исходник", "#94a3b8", target_opacity, 1),
        (candidate, "CadQuery", "#3b82f6", candidate_opacity, 2),
    ):
        if (
            data is None
            or (mode == "target" and column == 2)
            or (mode == "candidate" and column == 1)
        ):
            continue
        vertices, faces = data["vertices"], data["faces"]
        trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=color,
            opacity=opacity if mode == "overlay" and candidate is not None else 1,
            flatshading=True,
            showlegend=True,
            lighting=dict(ambient=0.55, diffuse=0.75, specular=0.15),
        )
        if side:
            figure.add_trace(trace, row=1, col=column)
        else:
            figure.add_trace(trace)
    bounds = np.array([target["bounds"]] + ([candidate["bounds"]] if candidate is not None else []))
    lower, upper = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    padding = max(float((upper - lower).max()), 0.01) * 0.07
    scene = dict(aspectmode="data", bgcolor="rgba(0,0,0,0)")
    for axis, low, high in zip("xyz", lower, upper, strict=True):
        scene[f"{axis}axis"] = dict(visible=False, range=[low - padding, high + padding])
    figure.update_scenes(**scene)
    figure.update_layout(
        height=430,
        margin=dict(l=0, r=0, t=35 if side else 0, b=0),
        uirevision=revision,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=0, x=0),
    )
    return figure


def evaluate_saved_pair(target: Path, candidate: Path, *, normalize_target: bool = False) -> dict:
    """Use complete STL meshes; never align or rescale the candidate for a preview score."""
    with TemporaryDirectory(prefix="mesh2cad-iou-") as temporary:
        output = Path(temporary) / "iou.json"
        command = [
            sys.executable,
            "-m",
            "mesh2cad.geometry_worker",
            str(target.resolve()),
            str(candidate.resolve()),
            str(output),
            str(os.getpid()),
        ]
        if normalize_target:
            command.append("--normalize-target")
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=worker_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
        )
        usage = monitor_process(process, timeout=120)
        log_event(
            "preview.iou_finished",
            target=target,
            candidate=candidate,
            peak_rss_mb=round(usage.peak_rss_mb, 1),
            reason=usage.reason,
        )
        if usage.reason == "memory_limit":
            raise RuntimeError(f"IoU exceeded {DEFAULT_MEMORY_MB} MiB RSS; worker stopped")
        if usage.reason == "timeout":
            raise RuntimeError("IoU exceeded 120 seconds; worker stopped")
        if usage.returncode or not output.is_file():
            raise RuntimeError(f"IoU worker exited with code {usage.returncode}")
        return read_json(output)
