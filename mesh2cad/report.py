"""Honest, denominator-preserving reports from checkpointed experiment records."""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.parse import quote

from .storage import read_json, resolve_artifact, write_json

_UNFINISHED = {"pending", "running", "generating", "interrupted", "quota_exhausted"}
_TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens")


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _average(values: list[float]) -> float | None:
    return mean(values) if values else None


def _stage(attempt: dict | None, item_status: str = "pending") -> str:
    if attempt is None:
        return item_status if item_status in _UNFINISHED else "not_generated"
    status = attempt.get("status", "pending")
    if status in _UNFINISHED:
        return status
    if status in {"ok", "generated"} and attempt.get("execution_status") is None:
        return "pending"
    if attempt.get("execution_status") == "valid":
        return (
            "scored"
            if _number((attempt.get("metrics") or {}).get("chamfer")) is not None
            else "evaluation_error"
        )
    if attempt.get("execution_status") is not None:
        return "execution_error"
    return "generation_error"


def _completed(attempt: dict) -> bool:
    if "finished" in attempt:
        return bool(attempt["finished"]) and attempt.get("status") != "quota_exhausted"
    return attempt.get("status") not in _UNFINISHED | {"ok", "generated"}


def _cost(attempts: list[dict]) -> dict:
    records = [generation for attempt in attempts for generation in attempt.get("generations", [])]
    launched = [record for record in records if record.get("launches", 0) > 0]
    cost: dict[str, Any] = {
        "generation_records": len(records),
        "launches": sum(record.get("launches", 0) for record in records),
        "generation_seconds": sum(
            _number(record.get("elapsed_seconds")) or 0 for record in records
        ),
        "execution_seconds": sum(
            _number(attempt.get("execution_seconds")) or 0 for attempt in attempts
        ),
        "unknown_usage_records": sum(
            any(
                _number((record.get("usage") or {}).get(key)) is None
                for key in ("input_tokens", "output_tokens")
            )
            for record in launched
        ),
    }
    for key in _TOKEN_KEYS:
        values = [_number((record.get("usage") or {}).get(key)) for record in launched]
        known = sum(value for value in values if value is not None)
        cost[f"known_{key}"] = int(known)
        cost[key] = int(known) if all(value is not None for value in values) else None
    cost["total_tokens"] = (
        cost["input_tokens"] + cost["output_tokens"]
        if cost["input_tokens"] is not None and cost["output_tokens"] is not None
        else None
    )
    cost["elapsed_seconds"] = cost["generation_seconds"] + cost["execution_seconds"]
    return cost


def _preflight_attempts(run_dir: Path) -> list[dict]:
    attempts = []
    for invocation in sorted((run_dir / "preflight").glob("invocation-*")):
        if not invocation.is_dir():
            continue
        smoke, generation = {}, None
        smoke_path = invocation / "smoke.json"
        if smoke_path.is_file():
            try:
                smoke = read_json(smoke_path)
                generation = smoke.get("generation")
            except (ValueError, OSError):
                # An interrupted non-atomic older writer may leave a partial log.
                smoke = {}
        generation_path = invocation / "generation" / "generation.json"
        if not isinstance(generation, dict) and generation_path.is_file():
            try:
                generation = read_json(generation_path)
            except (ValueError, OSError):
                generation = None
        if not isinstance(generation, dict):
            if not (invocation / "generation" / "command.json").is_file():
                continue
            # There is evidence of a call but no final usage checkpoint. Preserve
            # the launch as unknown rather than silently losing its possible cost.
            generation = {"launches": 1, "usage": {}, "status": "interrupted"}
        attempts.append(
            {
                "generations": [generation],
                "execution_seconds": (smoke.get("execution") or {}).get("elapsed_seconds", 0),
            }
        )
    return attempts


def _mode(attempt: dict | None, target_closed: bool | None, item_status: str) -> dict:
    metrics = (attempt or {}).get("metrics") or {}
    stage = _stage(attempt, item_status)
    valid = (attempt or {}).get("execution_status") == "valid"
    chamfer = _number(metrics.get("chamfer")) if valid else None
    iou, iou_status = None, "target_watertight_unknown"
    if target_closed is False:
        iou_status = "target_not_watertight"
    elif target_closed is True:
        if stage in {"generation_error", "execution_error"}:
            iou, iou_status = 0.0, f"{stage}_zero"
        elif stage == "scored":
            iou = _number(metrics.get("iou"))
            iou_status = metrics.get("iou_status", "unavailable")
            if iou is None and iou_status == "ok":
                iou_status = "invalid_metric"
        else:
            iou_status = stage
    return {
        "index": (attempt or {}).get("index"),
        "status": (attempt or {}).get("status", item_status),
        "stage": stage,
        "execution_status": (attempt or {}).get("execution_status"),
        "valid": valid,
        "chamfer": chamfer,
        "iou": iou,
        "iou_status": iou_status,
        "artifacts": (attempt or {}).get("artifacts") or {},
    }


def _aggregate(rows: list[dict], mode: str) -> dict:
    values = [row[mode] for row in rows]
    chamfers = [value["chamfer"] for value in values if value["chamfer"] is not None]
    ious = [value["iou"] for value in values if value["iou"] is not None]
    eligible = sum(row["target_watertight"] is True for row in rows)
    valid_count = sum(value["valid"] for value in values)
    return {
        "cad_valid_count": valid_count,
        "cad_valid_rate": valid_count / len(rows) if rows else None,
        "scored_count": len(chamfers),
        "mean_chamfer": _average(chamfers),
        "stage_counts": dict(sorted(Counter(value["stage"] for value in values).items())),
        "iou_eligible_count": eligible,
        "iou_known_count": len(ious),
        "iou_unavailable_count": eligible - len(ious),
        "iou_open_target_count": sum(row["target_watertight"] is False for row in rows),
        "iou_unknown_target_count": sum(row["target_watertight"] is None for row in rows),
        "mean_iou_known": _average(ious),
        "mean_iou": _average(ious) if len(ious) == eligible else None,
        "iou_status_counts": dict(sorted(Counter(value["iou_status"] for value in values).items())),
    }


def _display(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"


def _links(run_dir: Path, row: dict) -> str:
    links = []
    target = f"items/{row['id']}/target.png"
    if resolve_artifact(run_dir, target).is_file():
        links.append(f"[target]({quote(target, safe='/')})")
    for mode in ("baseline", "harness", "last"):
        candidate = row[mode]
        if mode == "last" and candidate["index"] == row["harness"]["index"]:
            continue
        for kind in ("comparison", "code", "step", "mesh"):
            path = candidate["artifacts"].get(kind)
            if path:
                # Check containment even for a missing artifact; never emit escaping links.
                resolve_artifact(run_dir, path)
                links.append(f"[{mode} {kind}]({quote(path, safe='/')})")
    return " · ".join(links) or "Artifacts unavailable."


def _markdown(run_dir: Path, summary: dict, rows: list[dict], config: dict) -> str:
    lines = [
        "# Mesh2CAD reconstruction and repair experiment",
        "",
        f"Selected items: **{summary['selected_count']}**. Incomplete results: **{summary['pending_count']}**.",
        f"Model: `{config.get('model', 'unknown')}`. Item statuses: `{summary['status_counts']}`.",
        "",
        "The baseline reuses attempt 0 from the same repair trajectory. Harness reports the saved best valid candidate by Chamfer; last reports the latest completed attempt. This is not an independent best-of-three experiment.",
        "",
        "## Geometry",
        "",
        "| Mode | Valid CAD / all selected | Scored CD count | Mean squared Chamfer | Known IoU mean | Known / eligible IoU | Unavailable IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "harness", "last"):
        block = summary[mode]
        lines.append(
            f"| {mode} | {block['cad_valid_count']}/{summary['selected_count']} "
            f"({_display(100 * block['cad_valid_rate'] if block['cad_valid_rate'] is not None else None)}%) | "
            f"{block['scored_count']} | {_display(block['mean_chamfer'])} | "
            f"{_display(block['mean_iou_known'])} | {block['iou_known_count']}/{block['iou_eligible_count']} | "
            f"{block['iou_unavailable_count']} |"
        )
    paired = summary["paired_chamfer"]
    lines.extend(
        [
            "",
            f"Chamfer is the sum of the two mean squared nearest-neighbor distances on {config.get('n_points', 8192)} sampled surface points, without independent prediction alignment or a scale multiplier. Means use scored candidates only; the scored count is shown above.",
            "",
            f"Paired baseline/harness comparison: **n={paired['count']}**; baseline mean {_display(paired['baseline_mean'])}; harness mean {_display(paired['harness_mean'])}; mean improvement (baseline − harness) **{_display(paired['mean_improvement'])}**. Improved: {paired['improved_count']}; unchanged: {paired['unchanged_count']}; worse: {paired['worsened_count']}.",
            f"Recovered valid CAD: {summary['recovered_valid_count']}. Latest scored repair worse than saved best: {summary['last_worse_than_best_count']}.",
            "",
            "IoU is eligible only for known watertight targets. Generation/execution failures contribute zero on those targets. Pending work, scoring/boolean failures, open predictions, and unknown target eligibility remain unavailable. Known-only IoU means explicitly exclude unavailable cases; `summary.json` also provides `mean_iou`, which is null unless every eligible case has a known value.",
            "",
            "## Failure and availability accounting",
            "",
        ]
    )
    for mode in ("baseline", "harness", "last"):
        lines.append(
            f"- {mode}: stages `{summary[mode]['stage_counts']}`; IoU `{summary[mode]['iou_status_counts']}`."
        )
    lines.extend(
        [
            "",
            "## Recorded cost",
            "",
            "| Mode ledger | CLI launches | Known input / cached / output tokens | Missing usage records | Generation seconds | Execution seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in ("baseline", "harness", "preflight", "overall"):
        cost = summary[f"{label}_cost"]
        lines.append(
            f"| {label} | {cost['launches']} | {cost['known_input_tokens']} / "
            f"{cost['known_cached_input_tokens']} / {cost['known_output_tokens']} | "
            f"{cost['unknown_usage_records']} | {cost['generation_seconds']:.3f} | {cost['execution_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The baseline ledger includes every generation record for attempt 0, including quota retries. The harness ledger includes all attempts and retries, so it contains the baseline cost. Preflight counts each invocation once, recovering generation logs or counting a command with missing final logs as one launch with unknown usage; the latest-smoke alias is not counted twice. Overall includes harness plus preflight. Cached input is a subset of input tokens and is not added again. Missing usage stays unknown in totals; the table shows only known subtotals. Timing sums recorded generation/execution durations and excludes missing timings, rendering, and other orchestration overhead. Tokens and Plus CLI launches are not API dollar costs.",
            "",
            "## Examples",
            "",
        ]
    )
    improved = sorted(
        [
            row
            for row in rows
            if row["chamfer_improvement"] is not None and row["chamfer_improvement"] > 0
        ],
        key=lambda row: row["chamfer_improvement"],
        reverse=True,
    )[:3]
    failures = [
        row
        for row in rows
        if row["harness"]["stage"] in {"generation_error", "execution_error", "evaluation_error"}
        or row["status"] == "input_error"
        or row["last_worse_than_best"]
    ][:3]
    recoveries = [row for row in rows if not row["baseline"]["valid"] and row["harness"]["valid"]][
        :3
    ]
    for label, examples in (
        ("Improvement", improved),
        ("Recovery", recoveries),
        ("Failure or worsening repair", failures),
    ):
        for row in examples:
            lines.append(
                f"- {label} `{row['id']}`: baseline CD {_display(row['baseline']['chamfer'])}; "
                f"best CD {_display(row['harness']['chamfer'])}; last CD {_display(row['last']['chamfer'])}. "
                + _links(run_dir, row)
            )
    if not improved and not failures and not recoveries:
        lines.append("No improvements, recoveries, or failed/worsening repairs were recorded.")
    lines.extend(
        [
            "",
            "Results apply to this selected dataset and frozen configuration. They do not establish generalization to furniture/household objects, a 5–10× API saving, or a quality improvement when the measured deltas do not show one.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(run_dir: Path) -> dict:
    """Write one CSV row per selected ID plus Markdown and machine-readable stats."""
    run_dir = Path(run_dir).resolve()
    selection = read_json(run_dir / "selection.json")
    config = read_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else {}
    selected = selection["items"]
    ids = [item["id"] for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Selection contains duplicate IDs")
    rows, all_attempts, baseline_attempts = [], [], []
    for item in selected:
        path = resolve_artifact(run_dir, f"items/{item['id']}/result.json")
        result = read_json(path) if path.is_file() else {"status": "pending", "attempts": []}
        attempts = result.get("attempts", [])
        by_index = {attempt["index"]: attempt for attempt in attempts}
        if len(by_index) != len(attempts):
            raise ValueError(f"Duplicate attempt indices for {item['id']}")
        baseline = by_index.get(0)
        completed = [attempt for attempt in attempts if _completed(attempt)]
        last = max(completed, key=lambda attempt: attempt["index"], default=None)
        best_index = result.get("best_index")
        if best_index is not None and best_index not in by_index:
            raise ValueError(f"Missing saved best attempt for {item['id']}")
        best = by_index[best_index] if best_index is not None else last
        if best is None and attempts:
            best = max(attempts, key=lambda attempt: attempt["index"])
        target_closed = item.get("target_watertight")
        if not isinstance(target_closed, bool):
            target_closed = next(
                (
                    attempt["metrics"]["target_watertight"]
                    for attempt in attempts
                    if isinstance((attempt.get("metrics") or {}).get("target_watertight"), bool)
                ),
                None,
            )
        status = result.get("status", "pending")
        row = {
            "id": item["id"],
            "split": item.get("split", result.get("split", "")),
            "status": status,
            "target_watertight": target_closed,
        }
        for mode, attempt in (("baseline", baseline), ("harness", best), ("last", last)):
            row[mode] = _mode(attempt, target_closed, status)
        baseline_cd, best_cd, last_cd = (
            row[mode]["chamfer"] for mode in ("baseline", "harness", "last")
        )
        row["chamfer_improvement"] = (
            baseline_cd - best_cd if baseline_cd is not None and best_cd is not None else None
        )
        row["last_worse_than_best"] = (
            last_cd is not None and best_cd is not None and last_cd > best_cd
        )
        row["baseline_cost"] = _cost([baseline] if baseline is not None else [])
        row["harness_cost"] = _cost(attempts)
        rows.append(row)
        all_attempts.extend(attempts)
        if baseline is not None:
            baseline_attempts.append(baseline)
    preflight_attempts = _preflight_attempts(run_dir)
    paired = [row for row in rows if row["chamfer_improvement"] is not None]
    deltas = [row["chamfer_improvement"] for row in paired]
    summary = {
        "schema_version": 1,
        "selected_count": len(rows),
        "pending_count": sum(row["status"] not in {"complete", "input_error"} for row in rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        **{mode: _aggregate(rows, mode) for mode in ("baseline", "harness", "last")},
        "paired_chamfer": {
            "count": len(paired),
            "baseline_mean": _average([row["baseline"]["chamfer"] for row in paired]),
            "harness_mean": _average([row["harness"]["chamfer"] for row in paired]),
            "mean_improvement": _average(deltas),
            "median_improvement": median(deltas) if deltas else None,
            "improved_count": sum(value > 0 for value in deltas),
            "unchanged_count": sum(value == 0 for value in deltas),
            "worsened_count": sum(value < 0 for value in deltas),
        },
        "recovered_valid_count": sum(
            not row["baseline"]["valid"] and row["harness"]["valid"] for row in rows
        ),
        "last_worse_than_best_count": sum(row["last_worse_than_best"] for row in rows),
        "baseline_cost": _cost(baseline_attempts),
        "harness_cost": _cost(all_attempts),
        "preflight_cost": _cost(preflight_attempts),
        "overall_cost": _cost(all_attempts + preflight_attempts),
    }
    flat_rows = []
    for row in rows:
        flat = {
            key: row[key]
            for key in (
                "id",
                "split",
                "status",
                "target_watertight",
                "chamfer_improvement",
                "last_worse_than_best",
            )
        }
        for mode in ("baseline", "harness", "last"):
            flat.update(
                {f"{mode}_{key}": value for key, value in row[mode].items() if key != "artifacts"}
            )
            flat.update(
                {
                    f"{mode}_{key}": row[mode]["artifacts"].get(key)
                    for key in ("code", "mesh", "step", "comparison")
                }
            )
        for mode in ("baseline", "harness"):
            flat.update({f"{mode}_{key}": value for key, value in row[f"{mode}_cost"].items()})
        flat_rows.append(flat)
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(flat_rows[0]) if flat_rows else ["id", "split", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    (run_dir / "report.md").write_text(_markdown(run_dir, summary, rows, config), encoding="utf-8")
    write_json(run_dir / "summary.json", summary)
    return summary
