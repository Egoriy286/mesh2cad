"""Evaluate a saved STL pair outside the GUI's address space."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .geometry import (
    OVERLAY_DEPTH_LEGEND,
    OVERLAY_VERSION,
    evaluate_iou,
    load_mesh,
    normalize_mesh,
    overlay_depth_calibration,
    render_overlay_sheet,
    solid_difference_parts,
)
from .resources import watch_parent
from .storage import file_digest, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("parent_pid", type=int)
    parser.add_argument("--normalize-target", action="store_true")
    parser.add_argument("--overlay-folder", type=Path)
    args = parser.parse_args()
    watch_parent(args.parent_pid)
    target = load_mesh(args.target)
    if args.normalize_target:
        target, _ = normalize_mesh(target)
    candidate = load_mesh(args.candidate)
    iou = evaluate_iou(target, candidate)
    write_json(args.output, iou)
    if args.overlay_folder:
        folder = args.overlay_folder
        folder.mkdir(parents=True, exist_ok=True)
        parts = solid_difference_parts(target, candidate)
        pixels = render_overlay_sheet(target, candidate, parts=parts)
        image = folder / "overlay-solid.png"
        temporary = folder / f"overlay-solid-{os.getpid()}.tmp"
        temporary.write_bytes(pixels)
        temporary.replace(image)
        arrays = {}
        for name, mesh in parts.items():
            faces = mesh.faces
            if len(faces) > 40_000:
                faces = faces[np.linspace(0, len(faces) - 1, 40_000, dtype=int)]
            ids, inverse = np.unique(faces, return_inverse=True)
            arrays[name + "_vertices"] = mesh.vertices[ids]
            arrays[name + "_faces"] = inverse.reshape(-1, 3)
        temporary_arrays = folder / f"overlay-solid-{os.getpid()}.npz"
        np.savez_compressed(temporary_arrays, **arrays)
        temporary_arrays.replace(folder / "overlay-solid.npz")
        write_json(
            folder / "overlay-solid.json",
            {
                "version": OVERLAY_VERSION,
                "legend": OVERLAY_DEPTH_LEGEND,
                "depth_calibration": overlay_depth_calibration(target),
                "target_sha256": file_digest(args.target),
                "candidate_sha256": file_digest(args.candidate),
                "normalize_target": args.normalize_target,
                "volumes": {
                    name: float(mesh.volume) if len(mesh.faces) else 0.0
                    for name, mesh in parts.items()
                },
            },
        )


if __name__ == "__main__":
    main()
