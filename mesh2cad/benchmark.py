"""Paired, resumable three-condition benchmark using the production harness.

Run with ``python -m mesh2cad.benchmark --output output/benchmarks/20261023``.
Paid calls happen only with --run. First cheap generations are shared with the
harness arm, so the comparison measures the effect of continuing the same run.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .agent import OpenRouterAgent
from .harness import HarnessConfig, Orchestrator, RunLock, create_run, utc_now
from .pipeline import PipelineError, api_key_from_environment
from .resources import monitor_process, watch_parent, worker_environment
from .storage import file_digest, read_json, write_json

PROJECT = Path(__file__).resolve().parents[1]
CHEAP = "openai/gpt-6-luna"
EXPENSIVE = "openai/gpt-6-sol"
CONDITIONS = ("cheap_one_prompt", "cheap_harness", "expensive_one_prompt")


@contextmanager
def budget_lock(folder):
    for _ in range(300):
        try:
            lease = RunLock(folder, ".budget.lock")
            break
        except RuntimeError:
            time.sleep(0.01)
    else:
        raise RuntimeError("Cannot acquire benchmark budget ledger")
    try:
        yield
    finally:
        lease.close()


def ordered_cohorts(protocol):
    """Interleave source strata so resource-limited prefixes retain both sources."""
    items = protocol["cohorts"]["interior300"]
    indoor = [i for i in items if i["source"] == "indoorcad"]
    poly = [i for i in items if i["source"] == "polyhaven"]
    interior = []
    for index, item in enumerate(poly):
        interior.extend(indoor[index * 4 : index * 4 + 4])
        interior.append(item)
    interior.extend(indoor[len(poly) * 4 :])
    return {"deepcad": protocol["cohorts"]["deepcad"], "interior300": interior}


class BudgetAgent(OpenRouterAgent):
    def __init__(self, *args, ledger: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = ledger

    def complete(self, payload, api_key, timeout):
        # Reserve the full output allowance, plus 20k prompt tokens. Current
        # contact sheets use ~2k image/text tokens; reject larger text contexts.
        text_chars = sum(
            len(m["content"])
            if isinstance(m.get("content"), str)
            else sum(len(p.get("text", "")) for p in m.get("content", []))
            for m in payload["messages"]
        )
        if text_chars > 60000:
            raise PipelineError("Benchmark prompt exceeds reserved input allowance")
        prices = read_json(self.ledger)["prices"][self.model]
        reserve = 20000 * prices["prompt"] + payload["max_tokens"] * prices["completion"]
        deadline = time.monotonic() + timeout
        while True:
            with budget_lock(self.ledger.parent):
                state = read_json(self.ledger)
                if state["spent"] + reserve > state["cap"]:
                    raise PipelineError(
                        "Benchmark dollar budget: cannot reserve next full response"
                    )
                if state["spent"] + sum(state["reserved"].values()) + reserve <= state["cap"]:
                    ticket = str(os.getpid())
                    state["reserved"][ticket] = reserve
                    write_json(self.ledger, state)
                    break
            # Other workers may have reserved long answers but used few tokens.
            # Wait for their actual usage instead of reporting a false failure.
            if time.monotonic() >= deadline:
                raise PipelineError("Timed out waiting for benchmark budget reservation")
            time.sleep(0.1)
        response = None
        try:
            response = super().complete(payload, api_key, timeout)
            return response
        finally:
            # A failed transport may still have been charged. Retain its full
            # reservation as uncertain spend rather than silently assuming $0.
            with budget_lock(self.ledger.parent):
                state = read_json(self.ledger)
                held = state["reserved"].pop(ticket)
                cost = (response or {}).get("usage", {}).get("cost")
                state["spent"] += cost if isinstance(cost, (int, float)) else held
                state["unknown_cost_calls"] += not isinstance(cost, (int, float))
                write_json(self.ledger, state)


def worker(root: Path, ledger: Path, parent: int):
    watch_parent(parent)
    key = sys.stdin.readline().strip()
    cfg = HarnessConfig.from_dict(read_json(root / "run.json")["config"])
    agent = BudgetAgent(
        cfg.model,
        ledger=ledger,
        system_prompt=cfg.system_prompt,
        request=cfg.request,
        reasoning_effort=cfg.reasoning_effort,
    )
    result = Orchestrator(root, key, agent=agent).run()
    raise SystemExit(0 if result["status"] == "completed" else 1)


def record_execution_memory_failure(root: Path) -> bool:
    """Make a killed generated program a durable failed attempt, not a dropped input."""
    state = read_json(root / "run.json")
    memory_file = root / "benchmark-memory.json"
    if not memory_file.exists() or read_json(memory_file).get("reason") != "memory_limit":
        return False
    active_id = state.get("current", {}).get("item_id")
    item = (
        next((i for i in state["items"] if i.get("id") == active_id), None) if active_id else None
    )
    if item is None:
        if len(state["items"]) != 1:
            return False
        item = state["items"][0]
    if not item["attempts"] or state.get("current", {}).get("stage") != "cadquery":
        return False
    attempt = item["attempts"][-1]
    folder = root / attempt["path"]
    if attempt["status"] != "executing" or not (folder / "response.json").exists():
        return False
    usage = read_json(memory_file)
    seconds = max(
        0,
        memory_file.stat().st_mtime
        - datetime.fromisoformat(state["current"]["started_at"]).timestamp(),
    )
    error = (
        f"Generated CAD exceeded the fixed 2048 MiB worker+children memory limit "
        f"(observed {usage['peak_rss_mb']:.0f} MiB). Simplify the construction; "
        "avoid large simultaneous Boolean operations and excessive feature counts."
    )
    execution = {"status": "memory_limit", "error": error, "elapsed_seconds": seconds}
    attempt.update(
        status="execution_error",
        error=error,
        execution=execution,
        elapsed_seconds=sum(c.get("elapsed_seconds", 0) for c in attempt.get("calls", []))
        + seconds,
    )
    attempt.setdefault("executions", []).append(execution)
    item["stale_iterations"] = item.get("stale_iterations", 0) + 1
    if item["stale_iterations"] >= state["config"]["patience"]:
        item.update(status="stalled", stop_reason="no_improvement")
    state.update(status="pending", stop_reason="execution_memory_failure_recorded")
    state["active_seconds"] += seconds
    stages = state.setdefault("stage_seconds", {})
    stages["cadquery"] = stages.get("cadquery", 0) + seconds
    state.setdefault("benchmark_resource_failures", []).append(
        {"iteration": attempt["iteration"], "stage": "cadquery", **usage}
    )
    write_json(folder / "execution.json", execution)
    write_json(folder / "attempt.json", attempt)
    write_json(folder / "memory-failure.json", usage)
    write_json(root / "run.json", state)
    return True


def execute(root: Path, ledger: Path, key: str):
    record_execution_memory_failure(root)
    if read_json(root / "run.json")["status"] == "completed":
        return
    for _ in range(6):
        with (root / "worker.log").open("a") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mesh2cad.benchmark",
                    "--worker",
                    str(root),
                    "--ledger",
                    str(ledger),
                    "--parent",
                    str(os.getpid()),
                ],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=output,
                text=True,
                cwd=PROJECT,
                env=worker_environment(),
                start_new_session=True,
            )
            process.stdin.write(key + "\n")
            process.stdin.close()
            usage = monitor_process(process, memory_limit_mb=2048, timeout=650)
            write_json(root / "benchmark-memory.json", asdict(usage))
            if usage.reason == "memory_limit" and record_execution_memory_failure(root):
                continue
            if usage.reason:
                raise RuntimeError(f"Benchmark worker stopped: {usage.reason}: {root}")
            return
    raise RuntimeError("Execution resource recovery exceeded maximum iterations")


def item_row(domain, condition, root, state):
    item = state["items"][0]
    attempts = item["attempts"]
    selected = attempts[:1] if condition == "cheap_one_prompt" else attempts
    scores = [(a.get("iou") or {}).get("iou") for a in selected]
    scores = [s for s in scores if s is not None]
    calls = [c for a in selected for c in a.get("calls", [])]
    received = [c for c in calls if c["status"] == "received"]
    costs = [c.get("usage", {}).get("cost") for c in received]
    return {
        "domain": domain,
        "condition": condition,
        "id": item["id"],
        "source": item.get("source"),
        "category": item.get("category"),
        "iou": max(scores) if scores else None,
        "iou_failure_zero": max(scores) if scores else 0.0,
        "cad_valid": any(a.get("execution", {}).get("status") == "valid" for a in selected),
        "scorable": bool(scores),
        "success_0_9": bool(scores and max(scores) >= 0.9),
        "calls": len(calls),
        "attempts": len(selected),
        "memory_failures": sum(
            a.get("execution", {}).get("status") == "memory_limit" for a in selected
        ),
        "cost_usd": sum(c for c in costs if isinstance(c, (int, float))),
        "unknown_cost_calls": sum(not isinstance(c, (int, float)) for c in costs),
        "tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in received),
        "seconds": sum(a.get("elapsed_seconds", 0) for a in selected)
        + sum(state.get("stage_seconds", {}).get(k, 0) for k in ("normalize", "render_target")),
        "run": str(root.relative_to(PROJECT)),
        "input_sha256": item["input_sha256"],
    }


def aggregate(rows):
    result = {}
    for domain in ("deepcad", "interior300"):
        result[domain] = {}
        for condition in CONDITIONS:
            group = [r for r in rows if r["domain"] == domain and r["condition"] == condition]
            if not group:
                continue
            n = len(group)
            values = np.array([r["iou_failure_zero"] for r in group])
            scored = [r["iou"] for r in group if r["scorable"]]
            result[domain][condition] = {
                "n": n,
                "mean_iou": float(values.mean()),
                "median_iou": float(np.median(values)),
                "mean_iou_scored": float(np.mean(scored)) if scored else None,
                "median_iou_scored": float(np.median(scored)) if scored else None,
                "cad_valid": sum(r["cad_valid"] for r in group),
                "scorable": sum(r["scorable"] for r in group),
                "success_0_9": sum(r["success_0_9"] for r in group),
                **{
                    key: sum(r.get(key, 0) for r in group)
                    for key in (
                        "calls",
                        "cost_usd",
                        "unknown_cost_calls",
                        "tokens",
                        "memory_failures",
                    )
                },
                "mean_seconds": sum(r["seconds"] for r in group) / n,
            }
        if all(c in result[domain] for c in CONDITIONS):
            by_id = {
                c: {r["id"]: r for r in rows if r["domain"] == domain and r["condition"] == c}
                for c in CONDITIONS
            }
            for baseline in ("cheap_one_prompt", "expensive_one_prompt"):
                gains = np.array(
                    [
                        by_id["cheap_harness"][i]["iou_failure_zero"]
                        - by_id[baseline][i]["iou_failure_zero"]
                        for i in sorted(by_id[baseline])
                    ]
                )
                boots = np.random.default_rng(42).choice(gains, (20000, len(gains))).mean(axis=1)
                result[domain]["harness_minus_" + baseline] = {
                    "mean": float(gains.mean()),
                    "ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
                    "improved": int((gains > 1e-6).sum()),
                    "n": len(gains),
                }
    return result


def report(output):
    index = read_json(output / "index.json")
    rows = []
    incomplete = []
    for domain, items in index["runs"].items():
        for item_id, arms in items.items():
            states = {arm: read_json(Path(root) / "run.json") for arm, root in arms.items()}
            if len(states) != 2 or any(s["status"] != "completed" for s in states.values()):
                incomplete.append(
                    {
                        "domain": domain,
                        "id": item_id,
                        "statuses": {k: s["status"] for k, s in states.items()},
                    }
                )
                continue
            for condition in CONDITIONS:
                arm = "cheap_harness" if condition.startswith("cheap") else "expensive_one_prompt"
                rows.append(item_row(domain, condition, Path(arms[arm]), states[arm]))
    summary = {
        "updated_at": utc_now(),
        "groups": aggregate(rows),
        "incomplete": incomplete,
        "budget": read_json(output / "budget.json"),
        "protocol": "protocol.json",
    }
    write_json(output / "summary.json", summary)
    if rows:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (output / "items.csv").write_text(stream.getvalue())
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "output/benchmarks/20261023")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--budget", type=float, default=1.35)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.ledger, args.parent)
        return
    output = args.output.resolve()
    if not args.run:
        print(json.dumps(report(output), indent=2))
        return
    key = api_key_from_environment(PROJECT)
    if not key:
        parser.error("OpenRouter key required")
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "budget.json"
    if not ledger.exists():
        models = read_json(PROJECT / ".cache/benchmark-models.json")
        prices = {
            m["id"]: {k: float(m["pricing"][k]) for k in ("prompt", "completion")}
            for m in models
            if m["id"] in (CHEAP, EXPENSIVE)
        }
        write_json(
            ledger,
            {
                "cap": args.budget,
                "spent": 0,
                "reserved": {},
                "unknown_cost_calls": 0,
                "prices": prices,
            },
        )
    if not (output / "protocol.json").exists():
        protocol = read_json(PROJECT / "output/latex/mesh2cad-research/experiment-protocol.json")
        protocol.update(
            status="running",
            started_at=utc_now(),
            model_pair={"cheap": CHEAP, "expensive": EXPENSIVE},
            maximum_new_spend_usd=args.budget,
            resource_ladder="matched prefixes in steps of 5",
            shared_first_answer=True,
            eligible_prefix_order=ordered_cohorts(protocol),
        )
        protocol["source_sha256"] = {
            str(p.relative_to(PROJECT)): file_digest(p) for p in (PROJECT / "mesh2cad").glob("*.py")
        }
        write_json(output / "protocol.json", protocol)
        write_json(output / "index.json", {"runs": {"deepcad": {}, "interior300": {}}})
    protocol = read_json(output / "protocol.json")
    index = read_json(output / "index.json")
    cohorts = protocol["eligible_prefix_order"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, min(args.limit, 50), 5):
            end = min(start + 5, args.limit, 50)
            completed_ranks = sum(
                all(
                    (arms := index["runs"][domain].get(cohorts[domain][rank]["id"], {}))
                    and len(arms) == 2
                    and all(
                        read_json(Path(root) / "run.json")["status"] == "completed"
                        for root in arms.values()
                    )
                    for domain in cohorts
                )
                for rank in range(min(args.limit, 50))
            )
            if completed_ranks >= end:
                continue
            budget = read_json(ledger)
            new_ranks = end - max(start, completed_ranks)
            # Admit complete matched batches using observed spend only, never
            # quality. Individual calls additionally reserve their full output.
            estimated = budget["spent"] / max(1, completed_ranks) * new_ranks
            existing_batch = all(
                cohorts[domain][rank]["id"] in index["runs"][domain]
                for domain in cohorts
                for rank in range(start, end)
            )
            if not existing_batch and budget["cap"] - budget["spent"] < max(0.42, estimated + 0.20):
                protocol["stop_reason"] = "insufficient_budget_for_next_matched_batch"
                break
            tasks = []
            for rank in range(start, end):
                for domain in ("deepcad", "interior300"):
                    item = cohorts[domain][rank]
                    arms = index["runs"][domain].setdefault(item["id"], {})
                    for arm, model, iterations in (
                        ("cheap_harness", CHEAP, 5),
                        ("expensive_one_prompt", EXPENSIVE, 1),
                    ):
                        if arm not in arms:
                            cfg = HarnessConfig(
                                model=model,
                                max_iterations=iterations,
                                max_tokens=16000,
                                total_token_budget=100000,
                                max_wall_seconds=600,
                                request_timeout=180,
                                max_retries=1,
                            )
                            root = create_run(
                                PROJECT / "data" / domain / "manifest.json",
                                output / "runs" / domain / item["id"] / arm,
                                cfg,
                                split="test" if domain == "deepcad" else "all",
                                item_ids=[item["id"]],
                            )
                            arms[arm] = str(root)
                        tasks.append(Path(arms[arm]))
            write_json(output / "index.json", index)
            for future in [pool.submit(execute, root, ledger, key) for root in tasks]:
                future.result()
            summary = report(output)
            print(
                json.dumps(
                    {
                        "rank": end,
                        "spent": summary["budget"]["spent"],
                        "incomplete": summary["incomplete"],
                        "time": utc_now(),
                    }
                ),
                flush=True,
            )
            if summary["incomplete"]:
                break
    protocol.update(status="finished", finished_at=utc_now())
    write_json(output / "protocol.json", protocol)
    report(output)


if __name__ == "__main__":
    main()
