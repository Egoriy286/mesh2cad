"""Evaluate a saved STL pair outside the GUI's address space."""

from __future__ import annotations

import argparse
from pathlib import Path

from .geometry import evaluate_iou, load_mesh, normalize_mesh
from .resources import watch_parent
from .storage import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("parent_pid", type=int)
    parser.add_argument("--normalize-target", action="store_true")
    args = parser.parse_args()
    watch_parent(args.parent_pid)
    target = load_mesh(args.target)
    if args.normalize_target:
        target, _ = normalize_mesh(target)
    write_json(args.output, evaluate_iou(target, load_mesh(args.candidate)))


if __name__ == "__main__":
    main()
