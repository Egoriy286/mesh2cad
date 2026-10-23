import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest
import trimesh

from mesh2cad.batch_benchmark import (
    GuardedAgent,
    chunks,
    geometry_complexity,
    geometry_slot,
    group_result,
    stratified_interior,
)
from mesh2cad.benchmark import record_execution_memory_failure
from mesh2cad.harness import utc_now
from mesh2cad.pipeline import PipelineError
from mesh2cad.storage import read_json, write_json


def test_hundred_inputs_are_split_without_drops_or_duplicates():
    for domain in range(2):
        ids = [f"{domain}-{i}" for i in range(50)]
        groups = chunks(ids)
        assert [len(g) for g in groups] == [15, 15, 15, 5]
        assert sum(groups, []) == ids


def test_input_complexity_distinguishes_multiple_bodies_from_box():
    box = trimesh.creation.box()
    other = box.copy()
    other.apply_translation([2, 0, 0])
    complex_mesh = trimesh.util.concatenate([box, other])
    assert (
        geometry_complexity(complex_mesh)["complexity_score"]
        > geometry_complexity(box)["complexity_score"]
    )


def test_interior_selection_is_reproducible_and_preserves_categories_and_sources():
    items = [
        {"id": f"{s}-{i}", "source": s, "category": str(i % 4)}
        for s in ("indoorcad", "polyhaven")
        for i in range(50)
    ]
    selected = stratified_interior(items)
    assert selected == stratified_interior(list(reversed(items)))
    assert len({i["id"] for i in selected}) == 50
    assert all(
        [i["source"] for i in g] == ["indoorcad"] * 4 + ["polyhaven"] for g in chunks(selected, 5)
    )
    assert {i["category"] for i in selected} == {"0", "1", "2", "3"}


def test_report_excludes_pending_but_includes_finished_failure_zero():
    base = {
        "started": True,
        "finished": True,
        "first_iou": None,
        "best_iou": None,
        "iou_failure_zero": 0.0,
        "cad_valid": False,
        "scorable": False,
        "iou_ge_0_9": False,
        "calls": 1,
        "tokens": 10,
        "cost_usd": 0.01,
        "seconds": 1.0,
        "memory_failures": 0,
    }
    result = group_result(
        [
            base,
            {
                **base,
                "first_iou": 0.4,
                "best_iou": 0.8,
                "iou_failure_zero": 0.8,
                "scorable": True,
                "cad_valid": True,
            },
            {
                **base,
                "finished": False,
                "started": False,
                "calls": 0,
                "tokens": 0,
                "cost_usd": 0,
                "seconds": 0,
            },
        ]
    )
    assert result["selected"] == 3 and result["finished"] == 2
    assert result["mean_best_iou"] == 0.4 and result["mean_first_iou"] == 0.2
    assert result["cost_usd"] == 0.02 and result["improved"] == 1


def test_parallel_guard_never_overbooks_dollar_cap(tmp_path, monkeypatch):
    ledger = tmp_path / "budget.json"
    write_json(
        ledger,
        {
            "cap": 0.02,
            "spent": 0,
            "reserved": {},
            "unknown_cost_calls": 0,
            "prices": {"prompt": "0", "completion": ".01"},
        },
    )
    monkeypatch.setattr("mesh2cad.batch_benchmark.os.getpid", threading.get_ident)
    monkeypatch.setattr("mesh2cad.agent.call_openrouter", lambda *a, **k: {"usage": {"cost": 0.01}})

    def call():
        try:
            GuardedAgent("test/model", ledger=ledger).complete(
                {"messages": [], "max_tokens": 1}, "fake", 2
            )
            return True
        except PipelineError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(lambda _: call(), range(4)))
    assert sum(accepted) == 2
    assert read_json(ledger)["spent"] == 0.02 and not read_json(ledger)["reserved"]


def test_unknown_charge_retains_reservation(tmp_path, monkeypatch):
    ledger = tmp_path / "budget.json"
    write_json(
        ledger,
        {
            "cap": 0.1,
            "spent": 0,
            "reserved": {},
            "unknown_cost_calls": 0,
            "prices": {"prompt": "0", "completion": ".01"},
        },
    )

    def broken(*args, **kwargs):
        raise PipelineError("connection lost")

    monkeypatch.setattr("mesh2cad.agent.call_openrouter", broken)
    with pytest.raises(PipelineError):
        GuardedAgent("test/model", ledger=ledger).complete(
            {"messages": [], "max_tokens": 1}, "fake", 1
        )
    state = read_json(ledger)
    assert state["spent"] == 0.01 and state["unknown_cost_calls"] == 1


def test_geometry_slots_bound_concurrent_heavy_stages(tmp_path):
    mutex = threading.Lock()
    active = peak = 0

    def work():
        nonlocal active, peak
        with geometry_slot(tmp_path):
            with mutex:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with mutex:
                active -= 1
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(lambda _: work(), range(8)))
    assert peak == 2 and active == 0


def test_coordinator_runs_batches_and_writes_shared_report(tmp_path, monkeypatch):
    from mesh2cad import batch_benchmark as module
    from mesh2cad.harness import HarnessConfig

    batches = []
    for domain in ("deepcad", "interior300"):
        root = tmp_path / domain
        write_json(
            root / "run.json",
            {
                "status": "pending",
                "config": asdict(HarnessConfig()),
                "active_seconds": 0,
                "items": [
                    {
                        "id": domain,
                        "status": "pending",
                        "attempts": [],
                        "best_attempt": None,
                        "input_sha256": "hash",
                    }
                ],
            },
        )
        batches.append({"domain": domain, "batch": 1, "ids": [domain], "run": str(root)})
    write_json(tmp_path / "protocol.json", {"batches": batches, "source_sha256": {}, "model": "openai/gpt-6-sol"})
    write_json(tmp_path / "budget.json", {"spent": 0})

    def fake_execute(root, ledger, key):
        state = read_json(root / "run.json")
        state["status"] = "completed"
        state["items"][0]["status"] = "failed"
        write_json(root / "run.json", state)

    monkeypatch.setattr(module, "execute_batch", fake_execute)
    monkeypatch.setattr(module, "api_key_from_environment", lambda _: "fake")
    monkeypatch.setattr(module, "provider_snapshot", lambda *args: {"usage": 0})
    monkeypatch.setattr("sys.argv", ["batch_benchmark", "--run", "--output", str(tmp_path)])
    module.main()
    assert read_json(tmp_path / "summary.json")["groups"]["all"]["finished"] == 2
    assert (tmp_path / "items.csv").is_file() and (tmp_path / "table.csv").is_file()


def test_batch_memory_recovery_uses_active_item_and_keeps_previous_items(tmp_path):
    attempt = {
        "iteration": 1,
        "path": "items/second/attempt-001",
        "status": "executing",
        "calls": [{"status": "received", "elapsed_seconds": 2}],
    }
    write_json(tmp_path / attempt["path"] / "response.json", {"usage": {"cost": 0.01}})
    first = {"id": "first", "status": "success", "attempts": []}
    write_json(
        tmp_path / "run.json",
        {
            "config": {"patience": 3},
            "active_seconds": 3,
            "status": "running",
            "items": [first, {"id": "second", "status": "running", "attempts": [attempt]}],
            "current": {"stage": "cadquery", "item_id": "second", "started_at": utc_now()},
        },
    )
    write_json(tmp_path / "benchmark-memory.json", {"reason": "memory_limit", "peak_rss_mb": 2200})
    assert record_execution_memory_failure(tmp_path)
    state = read_json(tmp_path / "run.json")
    assert state["items"][0] == first
    assert state["items"][1]["attempts"][0]["execution"]["status"] == "memory_limit"
    assert not record_execution_memory_failure(tmp_path)
