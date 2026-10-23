"""Frozen hard-DeepCAD/interior cohorts, 15-STL runs and resumable parallel execution.

Prepare: python -m mesh2cad.batch_benchmark --prepare
Execute: python -m mesh2cad.batch_benchmark --run
Report:  python -m mesh2cad.batch_benchmark
Only --run sends paid model requests. Preparation reads the key's remaining limit.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path

import httpx
import numpy as np

from .agent import OpenRouterAgent
from .benchmark import budget_lock, record_execution_memory_failure
from .geometry import load_mesh
from .harness import (
    TERMINAL_ITEMS,
    HarnessConfig,
    Orchestrator,
    RunLock,
    create_run,
    summarize,
    utc_now,
)
from .pipeline import PipelineError, api_key_from_environment
from .prompts import DEFAULT_REQUEST, SYSTEM_PROMPT
from .resources import monitor_process, watch_parent, worker_environment
from .storage import file_digest, read_json, write_json

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "output/benchmarks/20261023-hard50-interior50"
MODEL = "openai/gpt-6-luna"
INTERIOR_REQUEST = (
    "Reconstruct the complete furniture or interior object from the contact sheet. "
    "Preserve all visible parts, supports, openings and their proportions."
)


def chunks(items, size=15):
    return [items[i : i + size] for i in range(0, len(items), size)]


def geometry_complexity(mesh):
    """Input-only proxy: supporting planes, disconnected bodies and concavity."""
    offsets = np.einsum("ij,ij->i", mesh.face_normals, mesh.triangles_center)
    planes = len(np.unique(np.round(np.column_stack((mesh.face_normals, offsets)), 3), axis=0))
    bodies = int(mesh.body_count)
    concavity = float(np.clip(1 - mesh.volume / mesh.convex_hull.volume, 0, 1))
    return {
        "faces": len(mesh.faces),
        "supporting_planes": planes,
        "bodies": bodies,
        "concavity": concavity,
        "complexity_score": float(math.log1p(planes) + 2 * math.log(bodies) + 3 * concavity),
    }


def stratified_interior(items, seed=42):
    """40 IndoorCAD + 10 Poly Haven, round-robin categories, interleaved 4:1."""
    rng = random.Random(seed)
    selected = {}
    for source, count in (("indoorcad", 40), ("polyhaven", 10)):
        pools = defaultdict(list)
        for item in sorted(items, key=lambda i: i["id"]):
            if item["source"] == source:
                pools[item["category"]].append(item)
        categories = sorted(pools)
        rng.shuffle(categories)
        for pool in pools.values():
            rng.shuffle(pool)
        chosen = []
        while len(chosen) < count:
            progress = False
            for category in categories:
                if pools[category] and len(chosen) < count:
                    chosen.append(pools[category].pop())
                    progress = True
            if not progress:
                raise ValueError(f"Not enough verified volume STL inputs from {source}")
        selected[source] = chosen
    result = []
    for i, item in enumerate(selected["polyhaven"]):
        result.extend(selected["indoorcad"][i * 4 : i * 4 + 4])
        result.append(item)
    return result


def provider_snapshot(key, model_id=MODEL):
    with httpx.Client(timeout=30) as client:
        response = client.get(
            "https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key}
        )
        response.raise_for_status()
        data = response.json()["data"]
        response = client.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        model = next(m for m in response.json()["data"] if m["id"] == model_id)
    return {
        "checked_at": utc_now(),
        "limit": data.get("limit"),
        "limit_remaining": data.get("limit_remaining"),
        "usage": data.get("usage"),
        "model": model_id,
        "pricing": model["pricing"],
    }


def prepare(output, key, workers=2):
    if (output / "protocol.json").exists():
        return read_json(output / "protocol.json")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = provider_snapshot(key)
    if snapshot["limit_remaining"] is None:
        raise ValueError("A finite available key limit is required for the dollar guard")
    deepcad = []
    rejected = []
    for item in read_json(PROJECT / "data/deepcad/manifest.json")["items"]:
        if item["split"] != "test" or item["status"] != "ready":
            continue
        path = PROJECT / "data/deepcad" / item["mesh_path"]
        if file_digest(path) != item["mesh_sha256"]:
            raise ValueError(f"Changed input {item['id']}")
        mesh = load_mesh(path)
        if not mesh.is_volume:
            rejected.append({"domain": "deepcad", "id": item["id"], "reason": "not_volume"})
            continue
        deepcad.append({**item, **geometry_complexity(mesh)})
    deepcad.sort(key=lambda i: (-i["complexity_score"], i["id"]))
    if len(deepcad) < 50:
        raise ValueError("Need 50 eligible DeepCAD test inputs")
    interior = []
    for item in read_json(PROJECT / "data/interior300/manifest.json")["items"]:
        if not item.get("is_volume"):
            continue
        path = PROJECT / "data/interior300" / item["mesh_path"]
        if file_digest(path) != item["sha256"]:
            raise ValueError(f"Changed input {item['id']}")
        mesh = load_mesh(path)
        if not mesh.is_volume:
            rejected.append({"domain": "interior300", "id": item["id"], "reason": "not_volume"})
            continue
        interior.append(item)
    cohorts = {"deepcad": deepcad[:50], "interior300": stratified_interior(interior)}
    protocol = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "prepared",
        "model": MODEL,
        "orchestrator": "classic",
        "max_output_tokens": 16000,
        "total_token_budget_per_run": 1000000,
        "batch_sizes_per_domain": [15, 15, 15, 5],
        "parallel_workers": workers,
        "memory_limit_mb_per_worker_tree": 2048,
        "selection": {
            "deepcad": "Top 50 of verified local test pool by input geometry complexity",
            "complexity_formula": "ln(1+planes)+2*ln(bodies)+3*(1-volume/convex_hull_volume)",
            "planes": "unique rounded (normal_xyz, offset) tuples, 3 decimals, normalized STL",
            "caveat": "Geometry proxy, not an official difficulty annotation; no output-based selection",
            "interior300": "40 IndoorCAD + 10 Poly Haven; category round-robin, seed 42; 4:1 order",
        },
        "eligible_deepcad": len(deepcad),
        "deepcad_ranking": deepcad,
        "rejected_inputs": rejected,
        "cohorts": cohorts,
        "provider_before": snapshot,
        "source_sha256": {
            str(p.relative_to(PROJECT)): file_digest(p) for p in (PROJECT / "mesh2cad").glob("*.py")
        },
        "batches": [],
    }
    write_json(
        output / "budget.json",
        {
            "cap": max(0, snapshot["limit_remaining"] - 0.01),
            "spent": 0,
            "reserved": {},
            "unknown_cost_calls": 0,
            "prices": snapshot["pricing"],
        },
    )
    # Each batch has its own engine-lock directory, enabling actual parallelism.
    for domain, items in cohorts.items():
        for ordinal, batch in enumerate(chunks(items), 1):
            interior_domain = domain == "interior300"
            cfg = HarnessConfig(
                model=MODEL,
                max_iterations=5,
                max_tokens=16000,
                total_token_budget=1000000,
                max_wall_seconds=14400,
                execution_timeout=60,
                request_timeout=180,
                max_retries=1,
                memory_limit_mb=2048,
                system_prompt=SYSTEM_PROMPT.replace(
                    "a mechanical part", "a piece of furniture or an interior object"
                )
                if interior_domain
                else SYSTEM_PROMPT,
                request=INTERIOR_REQUEST if interior_domain else DEFAULT_REQUEST,
            )
            ids = [i["id"] for i in batch]
            folder = create_run(
                PROJECT / "data" / domain / "manifest.json",
                output / "runs" / domain / f"batch-{ordinal:02d}",
                cfg,
                split="test" if domain == "deepcad" else "all",
                item_ids=ids,
            )
            state = read_json(folder / "run.json")
            state["items"].sort(key=lambda i: ids.index(i["id"]))
            state["experiment"] = {
                "domain": domain,
                "batch": ordinal,
                "protocol": str(output / "protocol.json"),
            }
            write_json(folder / "run.json", state)
            protocol["batches"].append(
                {"domain": domain, "batch": ordinal, "ids": ids, "run": str(folder)}
            )
    write_json(output / "protocol.json", protocol)
    return protocol


class GuardedAgent(OpenRouterAgent):
    def __init__(self, *args, ledger, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = ledger

    def complete(self, payload, api_key, timeout):
        # UTF-8 byte count is a conservative text-token upper bound. Reserve
        # generous image headroom; never count base64 bytes as text tokens.
        text_bytes = 0
        images = 0
        for message in payload["messages"]:
            content = message.get("content", "")
            if isinstance(content, str):
                text_bytes += len(content.encode())
            else:
                text_bytes += sum(len(p.get("text", "").encode()) for p in content)
                images += sum(p.get("type") == "image_url" for p in content)
        input_bound = text_bytes + images * 12000
        prices = read_json(self.ledger)["prices"]
        prompt, completion = float(prices["prompt"]), float(prices["completion"])
        for override in prices.get("overrides", []):
            if input_bound >= override["min_prompt_tokens"]:
                prompt, completion = float(override["prompt"]), float(override["completion"])
        reserve = input_bound * prompt + payload["max_tokens"] * completion
        deadline = time.monotonic() + timeout
        ticket = str(os.getpid())
        while True:
            with budget_lock(self.ledger.parent):
                state = read_json(self.ledger)
                if state["spent"] + reserve > state["cap"]:
                    raise PipelineError("Experiment dollar budget exhausted before next request")
                if state["spent"] + sum(state["reserved"].values()) + reserve <= state["cap"]:
                    state["reserved"][ticket] = reserve
                    write_json(self.ledger, state)
                    break
            if time.monotonic() >= deadline:
                raise PipelineError("Timed out awaiting experiment dollar reservation")
            time.sleep(0.1)
        response = None
        try:
            response = super().complete(payload, api_key, timeout)
            return response
        finally:
            with budget_lock(self.ledger.parent):
                state = read_json(self.ledger)
                held = state["reserved"].pop(ticket)
                cost = (response or {}).get("usage", {}).get("cost")
                known = isinstance(cost, (int, float))
                state["spent"] += cost if known else held
                state["unknown_cost_calls"] += not known
                write_json(self.ledger, state)


def worker(root, ledger, parent):
    watch_parent(parent)
    key = sys.stdin.readline().strip()
    cfg = HarnessConfig.from_dict(read_json(root / "run.json")["config"])
    agent = GuardedAgent(
        cfg.model,
        ledger=ledger,
        system_prompt=cfg.system_prompt,
        request=cfg.request,
        reasoning_effort=cfg.reasoning_effort,
    )
    result = ParallelOrchestrator(root, key, agent=agent, geometry_pool=ledger.parent).run()
    raise SystemExit(0 if result["status"] == "completed" else 1)


@contextmanager
def geometry_slot(folder, slots=2, check=lambda: None):
    """Bound heavy geometry across processes while model requests overlap."""
    lease = None
    while lease is None:
        check()
        for index in range(slots):
            try:
                lease = RunLock(folder, f".geometry-{index}.lock")
                break
            except RuntimeError:
                continue
        if lease is None:
            time.sleep(0.05)
    try:
        yield
    finally:
        lease.close()


class ParallelOrchestrator(Orchestrator):
    """Production algorithm unchanged; only heavy-stage scheduling is bounded."""

    def __init__(self, *args, geometry_pool, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_pool = geometry_pool

    @contextmanager
    def _stage(self, name, item_id, iteration=None):
        heavy = {"normalize", "render_target", "cadquery", "iou", "surface_distance", "overlay"}
        if name in heavy:
            with geometry_slot(self.geometry_pool, check=self._check):
                with super()._stage(name, item_id, iteration):
                    yield
        else:
            with super()._stage(name, item_id, iteration):
                yield


def execute_batch(root, ledger, key):
    record_execution_memory_failure(root)
    for _ in range(80):
        state = read_json(root / "run.json")
        if state["status"] == "completed":
            return
        with (root / "worker.log").open("a") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mesh2cad.batch_benchmark",
                    "--worker",
                    str(root),
                    "--ledger",
                    str(ledger),
                    "--parent",
                    str(os.getpid()),
                ],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=log,
                text=True,
                cwd=PROJECT,
                env=worker_environment(),
                start_new_session=True,
            )
            process.stdin.write(key + "\n")
            process.stdin.close()
            usage = monitor_process(
                process, memory_limit_mb=2048, timeout=state["config"]["max_wall_seconds"] + 60
            )
        write_json(root / "benchmark-memory.json", asdict(usage))
        if usage.reason == "memory_limit" and record_execution_memory_failure(root):
            continue
        if usage.reason:
            state = read_json(root / "run.json")
            state.update(status="failed", stop_reason=usage.reason)
            write_json(root / "run.json", state)
        return
    raise RuntimeError("Too many memory recovery attempts")


def item_result(domain, batch, root, item, preparation_seconds=0):
    attempts = item["attempts"]
    calls = [c for a in attempts for c in a.get("calls", []) if c["status"] == "received"]
    scores = [(a.get("iou") or {}).get("iou") for a in attempts]
    scored = [s for s in scores if s is not None]
    return {
        "domain": domain,
        "batch": batch,
        "id": item["id"],
        "name": item.get("name"),
        "category": item.get("category"),
        "source": item.get("source"),
        "status": item["status"],
        "finished": item["status"] in TERMINAL_ITEMS,
        "started": bool(attempts),
        "first_iou": scores[0] if scores else None,
        "best_iou": max(scored) if scored else None,
        "iou_failure_zero": max(scored) if scored else 0,
        "cad_valid": any(a.get("execution", {}).get("status") == "valid" for a in attempts),
        "scorable": bool(scored),
        "iou_ge_0_9": bool(scored and max(scored) >= 0.9),
        "attempts": len(attempts),
        "calls": len(calls),
        "tokens": sum((c.get("usage") or {}).get("total_tokens", 0) for c in calls),
        "cost_usd": sum((c.get("usage") or {}).get("cost", 0) or 0 for c in calls),
        "seconds": preparation_seconds + sum(a.get("elapsed_seconds", 0) for a in attempts),
        "memory_failures": sum(
            a.get("execution", {}).get("status") == "memory_limit" for a in attempts
        ),
        "stop_reason": item.get("stop_reason"),
        "input_sha256": item["input_sha256"],
        "run": str(root),
    }


def group_result(rows):
    complete = [r for r in rows if r["finished"]]
    n = len(complete)
    scores = [r["iou_failure_zero"] for r in complete]
    return {
        "selected": len(rows),
        "started": sum(r["started"] for r in rows),
        "finished": n,
        "mean_first_iou": float(np.mean([r["first_iou"] or 0 for r in complete])) if n else None,
        "mean_best_iou": float(np.mean(scores)) if n else None,
        "median_best_iou": float(np.median(scores)) if n else None,
        "cad_valid": sum(r["cad_valid"] for r in complete),
        "scorable": sum(r["scorable"] for r in complete),
        "iou_ge_0_9": sum(r["iou_ge_0_9"] for r in complete),
        "improved": sum((r["best_iou"] or 0) > (r["first_iou"] or 0) + 1e-6 for r in complete),
        **{
            k: sum(r[k] for r in rows)
            for k in ("calls", "tokens", "cost_usd", "seconds", "memory_failures")
        },
    }


def report(output):
    protocol = read_json(output / "protocol.json")
    rows, batches = [], []
    for batch in protocol["batches"]:
        root = Path(batch["run"])
        state = read_json(root / "run.json")
        preparation = defaultdict(float)
        if (root / "events.jsonl").exists():
            for line in (root / "events.jsonl").read_text().splitlines():
                try:
                    e = json.loads(line)
                except ValueError:  # A concurrently appended last line may be incomplete.
                    continue
                if e["event"] == "stage_finished" and e.get("stage") in {
                    "normalize",
                    "render_target",
                }:
                    preparation[e["item_id"]] += e["elapsed_seconds"]
        rows.extend(
            item_result(batch["domain"], batch["batch"], root, i, preparation[i["id"]])
            for i in state["items"]
        )
        batches.append(
            {
                **batch,
                "status": state["status"],
                "stop_reason": state.get("stop_reason"),
                "error": state.get("error"),
                "summary": summarize(state),
            }
        )
    groups = {
        d: group_result([r for r in rows if r["domain"] == d]) for d in ("deepcad", "interior300")
    }
    groups["all"] = group_result(rows)
    result = {
        "updated_at": utc_now(),
        "groups": groups,
        "batches": batches,
        "budget": read_json(output / "budget.json"),
    }
    write_json(output / "summary.json", result)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (output / "items.csv").write_text(stream.getvalue())
    table = [
        "domain,selected,finished,mean_first_iou,mean_best_iou,median_best_iou,cad_valid,scorable,iou_ge_0_9,improved,calls,tokens,cost_usd,seconds,memory_failures"
    ]
    table.extend(
        ",".join(str(d if k == "domain" else g[k]) for k in table[0].split(","))
        for d, g in groups.items()
    )
    (output / "table.csv").write_text("\n".join(table) + "\n")
    lines = [
        "# Classic harness: 50 сложных DeepCAD + 50 интерьерных STL",
        "",
        f"Модель: `{protocol['model']}`. Группы на домен: 15 + 15 + 15 + 5.",
        "Выход: до 16 000 токенов на ответ. Общий лимит каждой группы: 1 000 000 токенов.",
        "",
        "| Домен | Готово / выбрано | IoU первого ответа | IoU harness | Медиана | Валидный CAD | IoU ≥ 0.9 | Улучшены | Вызовы | Токены | USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(x):
        return "—" if x is None else f"{x:.4f}"

    for domain, g in groups.items():
        lines.append(
            f"| {domain} | {g['finished']}/{g['selected']} | {fmt(g['mean_first_iou'])} | {fmt(g['mean_best_iou'])} | {fmt(g['median_best_iou'])} | {g['cad_valid']}/{g['finished']} | {g['iou_ge_0_9']}/{g['finished']} | {g['improved']} | {g['calls']} | {g['tokens']} | {g['cost_usd']:.6f} |"
        )
    lines += [
        "",
        "Средние включают завершённые неудачные реконструкции как 0. Незавершённые и не начатые STL в средние не входят.",
        "Первый ответ уже входит в цену harness, это один физический запуск. IoU объёмный, без подгонки геометрии.",
        "Сложность DeepCAD: геометрический показатель входной сетки, а не официальная разметка сложности. Отбор зафиксирован до генерации.",
        "STL, запросы, ответы, код, STEP, метрики и checkpoints лежат в `runs/`. Полный выбор и SHA-256: `protocol.json`.",
        "",
        "## Запуски",
        "",
        "| Домен | Группа | STL | Статус | Завершено | Причина |",
        "|---|---:|---:|---|---:|---|",
    ]
    for b in batches:
        lines.append(
            f"| {b['domain']} | {b['batch']} | {len(b['ids'])} | {b['status']} | {b['summary']['finished']} | {b['stop_reason'] or ''} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.ledger, args.parent)
        return
    output = args.output.resolve()
    key = api_key_from_environment(PROJECT) if args.run or args.prepare else None
    if args.prepare:
        prepare(output, key, args.workers)
    if args.run:
        if not key:
            parser.error("OpenRouter key required")
        with closing(RunLock(output, ".coordinator.lock")):
            protocol = read_json(output / "protocol.json")
            protocol.update(
                status="running",
                started_at=protocol.get("started_at") or utc_now(),
                parallel_workers=args.workers,
                heavy_geometry_slots=2,
            )
            protocol["source_sha256"]["mesh2cad/batch_benchmark.py"] = file_digest(Path(__file__))
            write_json(output / "protocol.json", protocol)
            # Batch API calls overlap; two heavy geometry slots bound memory.
            jobs = sorted(protocol["batches"], key=lambda b: (b["batch"], b["domain"]))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(execute_batch, Path(b["run"]), output / "budget.json", key): b
                    for b in jobs
                }
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    result = report(output)
                    print(
                        json.dumps(
                            {
                                "time": utc_now(),
                                "batches_finished": [
                                    futures[f]["domain"] + str(futures[f]["batch"]) for f in done
                                ],
                                "groups": result["groups"],
                            }
                        ),
                        flush=True,
                    )
            result = report(output)
            protocol.update(
                status="completed" if result["groups"]["all"]["finished"] == 100 else "incomplete",
                finished_at=utc_now(),
            )
            write_json(output / "protocol.json", protocol)
            write_json(output / "provider-after.json", provider_snapshot(key, protocol["model"]))
    result = report(output)
    print(json.dumps(result["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
