"""The prepare/run/report command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Mesh2CAD: CadQuery generation and bounded repair")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare a pinned STL subset and cache views")
    prepare.add_argument("--output", type=Path, default=Path("data/deepcad"))
    prepare.add_argument(
        "--source-dir", type=Path, help="Use local STL files instead of Hugging Face"
    )
    prepare.add_argument("--repo", default="maksimko123/deepcad_test_mesh")
    prepare.add_argument(
        "--revision", help="Hugging Face commit SHA (default: resolve current revision)"
    )
    prepare.add_argument("--dev-count", type=int, default=10)
    prepare.add_argument("--test-count", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)

    run = commands.add_parser("run", help="Run reconstruction; --resume keeps completed work")
    run.add_argument("--manifest", type=Path, default=Path("data/deepcad/manifest.json"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--split", choices=("dev", "test"), default="test")
    run.add_argument(
        "--limit", type=int, help="Default: 10 for dev, 50 for test; may increase on resume"
    )
    run.add_argument("--model", default="gpt-6-luna")
    run.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default="medium",
    )
    run.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=3)
    run.add_argument("--generation-timeout", type=float, default=180)
    run.add_argument("--execution-timeout", type=float, default=30)
    run.add_argument("--n-points", type=int, default=8192)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--smoke-only",
        action="store_true",
        help="One model request and sandbox/export check; no dataset required",
    )

    report = commands.add_parser("report", help="Rebuild CSV/Markdown reports without model calls")
    report.add_argument("run_dir", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            from .dataset import prepare_dataset
            from .storage import read_json

            manifest = prepare_dataset(
                args.output,
                source_dir=args.source_dir,
                repo=args.repo,
                revision=args.revision,
                dev_count=args.dev_count,
                test_count=args.test_count,
                seed=args.seed,
            )
            items = read_json(manifest)["items"]
            failures = sum(item["status"] != "ready" for item in items)
            print(f"Manifest: {manifest} ({len(items)} selected, {failures} preparation errors)")
            return 1 if failures else 0
        if args.command == "run":
            from .backend import CodexBackend
            from .harness import run_experiment, run_smoke

            backend = CodexBackend(args.model, args.reasoning_effort, args.generation_timeout)
            if args.smoke_only:
                if args.resume or any(args.output.glob("*")):
                    raise ValueError("Smoke-only requires a fresh output directory and no --resume")
                state = run_smoke(backend, args.output, args.execution_timeout)
            else:
                from .report import build_report

                state = run_experiment(
                    args.manifest,
                    args.output,
                    split=args.split,
                    limit=args.limit
                    if args.limit is not None
                    else (10 if args.split == "dev" else 50),
                    max_attempts=args.max_attempts,
                    execution_timeout=args.execution_timeout,
                    n_points=args.n_points,
                    seed=args.seed,
                    resume=args.resume,
                    backend=backend,
                )
                build_report(args.output)
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return (
                0
                if state["status"] in {"ok", "complete"}
                else (2 if state["status"] == "quota_exhausted" else 1)
            )
        if args.command == "report":
            from .report import build_report

            summary = build_report(args.run_dir)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
    except KeyboardInterrupt:
        if args.command == "run" and (args.output / "selection.json").exists():
            from .report import build_report

            build_report(args.output)
        print("Interrupted; saved checkpoints can be continued with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1
