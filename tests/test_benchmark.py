"""Accounting and paired comparisons must not hide failed paid generations."""

import pytest

from mesh2cad.benchmark import BudgetAgent, aggregate, item_row, ordered_cohorts
from mesh2cad.pipeline import PipelineError
from mesh2cad.storage import read_json, write_json


def test_first_response_is_frozen_and_full_harness_cost_is_counted(tmp_path):
    attempts = [
        {
            "iou": None,
            "execution": {"status": "error"},
            "elapsed_seconds": 1,
            "calls": [{"status": "received", "usage": {"cost": 0.01, "total_tokens": 12}}],
        },
        {
            "iou": {"iou": 0.92},
            "execution": {"status": "valid"},
            "elapsed_seconds": 2,
            "calls": [{"status": "received", "usage": {"cost": 0.02, "total_tokens": 24}}],
        },
    ]
    state = {"items": [{"id": "a", "attempts": attempts, "input_sha256": "hash"}]}
    from mesh2cad.benchmark import PROJECT

    first = item_row("deepcad", "cheap_one_prompt", PROJECT / "runs/example", state)
    best = item_row("deepcad", "cheap_harness", PROJECT / "runs/example", state)
    assert first["iou"] is None and first["iou_failure_zero"] == 0
    assert first["cost_usd"] == 0.01 and first["calls"] == 1
    assert best["iou"] == 0.92 and best["cost_usd"] == pytest.approx(0.03)
    assert best["tokens"] == 36 and best["seconds"] == 3


def test_aggregate_includes_failure_zero_and_pairs_by_id():
    rows = []
    for condition, values in (
        ("cheap_one_prompt", [0, 0.6]),
        ("cheap_harness", [0.9, 0.8]),
        ("expensive_one_prompt", [0.8, 0.7]),
    ):
        for item_id, score in zip(("b", "a"), values):
            rows.append(
                {
                    "domain": "deepcad",
                    "condition": condition,
                    "id": item_id,
                    "iou": score if score else None,
                    "iou_failure_zero": score,
                    "cad_valid": score > 0,
                    "scorable": score > 0,
                    "success_0_9": score >= 0.9,
                    "calls": 1,
                    "cost_usd": 0.01,
                    "unknown_cost_calls": 0,
                    "tokens": 10,
                    "seconds": 2,
                }
            )
    result = aggregate(list(reversed(rows)))["deepcad"]
    assert result["cheap_one_prompt"]["mean_iou"] == 0.3
    assert result["cheap_one_prompt"]["scorable"] == 1
    assert result["harness_minus_cheap_one_prompt"]["mean"] == pytest.approx(0.55)
    assert result["harness_minus_expensive_one_prompt"]["mean"] == pytest.approx(0.1)


def test_resource_limited_prefix_keeps_source_strata():
    indoor = [{"id": str(i), "source": "indoorcad"} for i in range(8)]
    poly = [{"id": str(i), "source": "polyhaven"} for i in range(2)]
    result = ordered_cohorts({"cohorts": {"deepcad": [], "interior300": indoor + poly}})
    assert [i["source"] for i in result["interior300"][:5]].count("polyhaven") == 1


def test_budget_rejection_happens_before_api_call(tmp_path, monkeypatch):
    ledger = tmp_path / "budget.json"
    state = {
        "cap": 0.05,
        "spent": 0.04,
        "reserved": {},
        "unknown_cost_calls": 0,
        "prices": {"test/model": {"prompt": 0.000002, "completion": 0.00001}},
    }
    write_json(ledger, state)
    agent = BudgetAgent("test/model", ledger=ledger)
    monkeypatch.setattr("mesh2cad.agent.call_openrouter", lambda *a, **k: pytest.fail("paid call"))
    with pytest.raises(PipelineError, match="budget"):
        agent.complete({"messages": [], "max_tokens": 16000}, "fake", 1)
    assert read_json(ledger) == state


def test_concurrent_paid_calls_share_one_budget(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    ledger = tmp_path / "budget.json"
    write_json(
        ledger,
        {
            "cap": 0.02,
            "spent": 0,
            "reserved": {},
            "unknown_cost_calls": 0,
            "prices": {"test/model": {"prompt": 0, "completion": 0.01}},
        },
    )
    monkeypatch.setattr("mesh2cad.agent.call_openrouter", lambda *a, **k: {"usage": {"cost": 0.01}})
    # Actual workers have distinct process IDs; simulate those per-thread IDs.
    import threading

    monkeypatch.setattr("mesh2cad.benchmark.os.getpid", threading.get_ident)

    def call():
        agent = BudgetAgent("test/model", ledger=ledger)
        try:
            agent.complete({"messages": [], "max_tokens": 1}, "fake", 1)
            return True
        except PipelineError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(lambda _: call(), range(4)))
    assert sum(accepted) == 2
    result = read_json(ledger)
    assert result["spent"] == 0.02 and result["reserved"] == {}


def test_memory_failure_keeps_paid_response_and_becomes_tool_feedback(tmp_path):
    from mesh2cad.benchmark import record_execution_memory_failure
    from mesh2cad.harness import utc_now

    attempt = {
        "iteration": 1,
        "path": "items/a/attempt-001",
        "status": "executing",
        "calls": [{"status": "received", "usage": {"cost": 0.01}, "elapsed_seconds": 2}],
    }
    response = {"usage": {"cost": 0.01}, "choices": [{"message": {"content": "paid code"}}]}
    write_json(tmp_path / attempt["path"] / "response.json", response)
    write_json(
        tmp_path / "run.json",
        {
            "config": {"patience": 3},
            "active_seconds": 3,
            "status": "running",
            "items": [{"status": "running", "attempts": [attempt]}],
            "current": {"stage": "cadquery", "started_at": utc_now()},
        },
    )
    write_json(
        tmp_path / "benchmark-memory.json",
        {"reason": "memory_limit", "peak_rss_mb": 2060, "returncode": -9},
    )
    assert record_execution_memory_failure(tmp_path)
    state = read_json(tmp_path / "run.json")
    assert state["status"] == "pending"
    first = state["items"][0]["attempts"][0]
    assert first["status"] == "execution_error"
    assert first["execution"]["status"] == "memory_limit"
    assert first["calls"][0]["usage"]["cost"] == 0.01
    assert read_json(tmp_path / attempt["path"] / "response.json") == response
    assert not record_execution_memory_failure(tmp_path)  # Recovery is idempotent.
