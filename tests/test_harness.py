"""Exercise orchestration with deterministic agents and real CAD/IoU tools."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import trimesh

from mesh2cad.agent import AgentContext, OpenRouterAgent
from mesh2cad.harness import (
    HarnessConfig,
    Orchestrator,
    RunLock,
    cancel_run,
    create_run,
    is_active,
    launch_run,
    results_csv,
    run_isolated,
)
from mesh2cad.pipeline import PipelineError
from mesh2cad.storage import read_json, write_json

BOX = 'import cadquery as cq\nr = cq.Workplane("XY").box(1, 1, 1)'
SMALL_BOX = 'import cadquery as cq\nr = cq.Workplane("XY").box(0.5, 0.5, 0.5)'


def response(code: str, *, cost=0.002) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps({"code": code})}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": cost},
    }


class ScriptedAgent(OpenRouterAgent):
    def __init__(self, outputs) -> None:
        super().__init__("test/model")
        self.outputs = iter(outputs)
        self.requests: list[dict] = []

    def complete(self, payload, api_key, timeout):
        self.requests.append(payload)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


@pytest.fixture
def manifest(tmp_path) -> Path:
    root = tmp_path / "dataset"
    (root / "meshes").mkdir(parents=True)
    items = []
    for index in range(2):
        data = trimesh.creation.box().export(file_type="stl")
        relative = f"meshes/box{index}.stl"
        (root / relative).write_bytes(data)
        items.append(
            {
                "id": f"box{index}",
                "source_file": f"original{index}.stl",
                "split": "dev",
                "status": "ready",
                "mesh_path": relative,
                "mesh_sha256": hashlib.sha256(data).hexdigest(),
                "image_path": f"views/box{index}.png",
                "image_sha256": "absent",
            }
        )
    path = root / "manifest.json"
    write_json(path, {"schema_version": 1, "source": {"kind": "local"}, "items": items})
    return path


def new_run(manifest, tmp_path, **overrides) -> Path:
    config = HarnessConfig(
        model="test/model",
        max_iterations=4,
        patience=4,
        request_timeout=10,
        retry_backoff=0,
        **overrides,
    )
    return create_run(manifest, tmp_path / "runs", config, limit=1)


def test_isolated_worker_recovers_saved_response_without_network(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path)
    state = read_json(root / "run.json")
    relative = "items/box0/attempt-001"
    (root / relative).mkdir(parents=True)
    write_json(root / relative / "response.json", response(BOX))
    state["items"][0]["attempts"] = [
        {
            "iteration": 1,
            "path": relative,
            "status": "requesting",
            "calls": [{"status": "requesting"}],
        }
    ]
    write_json(root / "run.json", state)
    state = run_isolated(root, "isolated-unit-secret")
    assert state["status"] == "completed"
    assert state["items"][0]["status"] == "success"
    assert state["summary"]["llm_calls"] == 1
    assert state["summary"]["mean_iou"] == pytest.approx(1)
    assert not is_active(root)
    memory = read_json(root / "memory.json")
    assert memory["peak_rss_mb"] > 0
    assert memory["rss_mb"] == 0
    assert memory["limit_mb"] == 2048
    for path in root.rglob("*.json"):
        assert "isolated-unit-secret" not in path.read_text()


def test_memory_limit_preserves_checkpoint_and_paid_artifacts(
    manifest, tmp_path, monkeypatch
) -> None:
    import mesh2cad.harness as harness

    root = new_run(manifest, tmp_path, memory_limit_mb=256)
    paid = root / "saved-response.json"
    write_json(paid, response(BOX))
    real_popen = harness.subprocess.Popen
    commands = []

    def hungry_worker(command, **kwargs):
        commands.append(command)
        return real_popen(
            [command[0], "-c", "import time; data=bytearray(320*1024*1024); time.sleep(30)"],
            **kwargs,
        )

    monkeypatch.setattr(harness.subprocess, "Popen", hungry_worker)
    state = run_isolated(root, "never-put-this-key-in-argv")
    assert state["status"] == "failed"
    assert state["stop_reason"] == "memory_limit"
    assert "256 MiB" in state["error"]
    assert read_json(paid) == response(BOX)
    assert read_json(root / "memory.json")["peak_rss_mb"] > 256
    assert not is_active(root)
    assert "never-put-this-key-in-argv" not in str(commands)


def test_agent_repairs_execution_then_geometry_using_real_feedback(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path)
    agent = ScriptedAgent(
        [response("raise ValueError('broken sketch')"), response(SMALL_BOX), response(BOX)]
    )
    state = Orchestrator(root, "unit-test-key", agent=agent).run()
    item = state["items"][0]
    assert state["status"] == "completed"
    assert item["status"] == "success"
    assert [a["status"] for a in item["attempts"]] == ["execution_error", "evaluated", "evaluated"]
    assert item["best_attempt"] == 2
    assert item["attempts"][1]["iou"]["iou"] == pytest.approx(0.125)
    assert item["attempts"][2]["iou"]["iou"] == pytest.approx(1)
    assert "broken sketch" in json.dumps(agent.requests[1])
    assert "best-contrast-overlay" in json.dumps(agent.requests[2])
    summary = state["summary"]
    assert summary["total_tokens"] == 360
    assert summary["cost_usd"] == pytest.approx(0.006)
    assert summary["valid_attempts"] == 2
    assert summary["success_rate"] == 1
    best = root / item["attempts"][2]["path"]
    assert (best / "cadquery/candidate.step").is_file()
    assert (best / "overlay.png").is_file()
    assert read_json(best / "request.json") == agent.requests[2]
    assert "unit-test-key" not in (root / "run.json").read_text()
    assert "unit-test-key" not in (root / "events.jsonl").read_text()
    assert "best_iou" in results_csv(state)


def test_best_program_survives_a_regression(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path)
    state = read_json(root / "run.json")
    state["config"].update(max_iterations=2, target_iou=1)
    write_json(root / "run.json", state)
    bigger = 'r = cq.Workplane("XY").box(0.8, 0.8, 0.8)'
    state = Orchestrator(
        root, "test", agent=ScriptedAgent([response(bigger), response(SMALL_BOX)])
    ).run()
    item = state["items"][0]
    assert item["status"] == "exhausted"
    assert item["best_attempt"] == 0
    assert state["summary"]["mean_iou"] == pytest.approx(0.512, abs=1e-6)


def test_transient_error_retries_but_auth_error_stops_batch(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path)
    agent = ScriptedAgent([PipelineError("rate limit", retryable=True), response(BOX)])
    state = Orchestrator(root, "test", agent=agent).run()
    assert state["items"][0]["status"] == "success"
    assert state["summary"]["llm_calls"] == 2
    assert state["summary"]["total_tokens"] == 120

    root2 = create_run(manifest, tmp_path / "runs", HarnessConfig(model="test/model"))
    broken = ScriptedAgent([PipelineError("key unit-secret was rejected")])
    state = Orchestrator(root2, "unit-secret", agent=broken).run()
    assert state["status"] == "failed"
    assert len(broken.requests) == 1
    assert state["items"][1]["status"] == "pending"
    assert "unit-secret" not in (root2 / "run.json").read_text()
    assert "unit-secret" not in (root2 / "events.jsonl").read_text()
    resumed = Orchestrator(
        root2, "fixed-key", agent=ScriptedAgent([response(BOX), response(BOX)])
    ).run()
    assert resumed["summary"]["successes"] == 2
    assert len(resumed["items"][0]["attempts"]) == 1


def test_cancel_cad_then_resume_reuses_paid_response(manifest, tmp_path, monkeypatch) -> None:
    import mesh2cad.harness as harness

    root = new_run(manifest, tmp_path)
    real_execute = harness.execute_code

    def cancel_during_execution(*args, **kwargs):
        timer = threading.Timer(0.2, cancel_run, args=(root,))
        timer.start()
        try:
            return real_execute(*args, **kwargs)
        finally:
            timer.cancel()

    monkeypatch.setattr(harness, "execute_code", cancel_during_execution)
    agent = ScriptedAgent([response(BOX)])
    state = Orchestrator(root, "test", agent=agent).run()
    assert state["status"] == "cancelled"
    assert state["items"][0]["attempts"][0]["execution"]["status"] == "cancelled"
    assert not is_active(root)
    monkeypatch.setattr(harness, "execute_code", real_execute)
    # No further responses exist. Any duplicated API call would fail this run.
    state = Orchestrator(root, "test", agent=agent).run()
    assert state["items"][0]["status"] == "success"
    assert len(agent.requests) == 1
    assert state["summary"]["total_tokens"] == 120
    assert len(state["items"][0]["attempts"]) == 1
    attempt = state["items"][0]["attempts"][0]
    assert len(attempt["executions"]) == 2
    assert state["summary"]["execution_seconds"] > attempt["execution"]["elapsed_seconds"]


def test_checkpoint_recovery_accounts_response_persisted_before_state(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path)
    state = read_json(root / "run.json")
    relative = "items/box0/attempt-001"
    (root / relative).mkdir(parents=True)
    write_json(root / relative / "response.json", response(BOX))
    state["items"][0]["attempts"] = [
        {
            "iteration": 1,
            "path": relative,
            "status": "requesting",
            "calls": [{"status": "requesting"}],
        }
    ]
    write_json(root / "run.json", state)
    result = Orchestrator(root, "test", agent=ScriptedAgent([])).run()
    assert result["summary"]["total_tokens"] == 120
    assert result["items"][0]["status"] == "success"


def test_token_budget_prevents_next_call_and_unknown_cost_is_not_zero(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path, total_token_budget=120)
    output = response(SMALL_BOX)
    output["usage"].pop("cost")
    agent = ScriptedAgent([output])
    state = Orchestrator(root, "test", agent=agent).run()
    assert state["status"] == "budget_exhausted"
    assert state["stop_reason"] == "token_budget"
    assert state["summary"]["cost_usd"] is None
    assert state["summary"]["cost_missing"] == 1
    assert len(agent.requests) == 1


def test_wall_budget_and_single_worker_lock(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path, max_wall_seconds=0.000001)
    lease = RunLock(root)
    try:
        assert is_active(root)
        with pytest.raises(RuntimeError, match="active worker"):
            Orchestrator(root, "test").run()
    finally:
        lease.close()
    assert not is_active(root)
    agent = ScriptedAgent([])
    state = Orchestrator(root, "test", agent=agent).run()
    assert state["status"] == "budget_exhausted"
    assert state["stop_reason"] == "wall_time_budget"
    assert not agent.requests


def test_invalid_model_response_becomes_feedback_and_can_be_repaired(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path)
    bad = {"choices": [{"message": {"content": "not JSON"}}]}
    agent = ScriptedAgent([bad, response(BOX)])
    state = Orchestrator(root, "test", agent=agent).run()
    assert state["items"][0]["status"] == "success"
    assert "invalid_response" in json.dumps(agent.requests[1])
    assert state["summary"]["usage_missing"] == 1
    assert (root / state["items"][0]["attempts"][0]["path"] / "response.json").is_file()


def test_stagnation_stops_automatically(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path)
    state = read_json(root / "run.json")
    state["config"]["patience"] = 1
    state["config"]["min_improvement"] = 0
    write_json(root / "run.json", state)
    agent = ScriptedAgent([response(SMALL_BOX), response(SMALL_BOX)])
    result = Orchestrator(root, "test", agent=agent).run()
    assert result["items"][0]["status"] == "stalled"
    assert len(agent.requests) == 2


def test_unscorable_target_does_not_call_agent(manifest, tmp_path) -> None:
    data = read_json(manifest)
    mesh = trimesh.creation.box()
    mesh.update_faces(range(len(mesh.faces) - 1))
    payload = mesh.export(file_type="stl")
    (manifest.parent / data["items"][0]["mesh_path"]).write_bytes(payload)
    data["items"][0]["mesh_sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(manifest, data)
    root = new_run(manifest, tmp_path)
    result = Orchestrator(root, "test", agent=ScriptedAgent([])).run()
    assert result["items"][0]["status"] == "unscorable"
    assert result["summary"]["mean_iou"] is None
    assert result["summary"]["mean_iou_all"] == 0


def test_dataset_selection_snapshots_and_hash_validation(manifest, tmp_path) -> None:
    root = create_run(manifest, tmp_path / "runs", HarnessConfig(), item_ids=["box1"])
    state = read_json(root / "run.json")
    assert [i["id"] for i in state["items"]] == ["box1"]
    assert (root / "inputs/box1.stl").is_file()
    with pytest.raises(ValueError, match="selected IDs"):
        create_run(manifest, tmp_path / "runs", HarnessConfig(), item_ids=["nonexistent"])
    data = read_json(manifest)
    (manifest.parent / data["items"][0]["mesh_path"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed input"):
        create_run(manifest, tmp_path / "runs", HarnessConfig())


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_iou": float("nan")},
        {"max_wall_seconds": float("inf")},
        {"max_iterations": 0},
        {"patience": True},
        {"target_iou": 1.1},
        {"model": " "},
        {"memory_limit_mb": 0},
        {"memory_limit_mb": True},
        {"memory_limit_mb": 80_000},
    ],
)
def test_invalid_budgets_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        replace(HarnessConfig(), **overrides)


def test_agent_keeps_original_image_and_limits_conversation() -> None:
    attempts = [
        {"iteration": i, "status": "invalid_response", "error": "bad JSON"} for i in range(6)
    ]
    context = AgentContext(
        extents=[1, 1, 1], image=b"\x89PNG\r\n\x1a\nimage", attempts=attempts, best_attempt=None
    )
    payload = OpenRouterAgent("test/model").build_request(context)
    assert len(payload["messages"]) == 8  # system, target, last 3 proposal/result pairs
    assert payload["messages"][1]["content"][2]["type"] == "image_url"
    assert payload["messages"][2]["content"] == '{"code": ""}'


def test_duplicate_launch_reuses_existing_worker(manifest, tmp_path, monkeypatch) -> None:
    leases = []
    starts = []

    def start(root, key):
        starts.append(root)
        leases.append(RunLock(root))

    monkeypatch.setattr("mesh2cad.harness.start_run", start)
    args = (manifest, tmp_path / "runs", HarnessConfig(), "test")
    try:
        first, reused = launch_run(*args, split="dev", item_ids=["box0"], limit=1)
        second, reused_again = launch_run(*args, split="dev", item_ids=["box0"], limit=1)
        assert not reused and reused_again
        assert first == second
        assert starts == [first]
        assert len(list((tmp_path / "runs").glob("*/run.json"))) == 1
    finally:
        for lease in leases:
            lease.close()


def test_busy_engine_queues_and_cancels_without_calling_agent(manifest, tmp_path) -> None:
    root = new_run(manifest, tmp_path)
    engine = RunLock(root.parent, ".engine.lock")
    agent = ScriptedAgent([])
    worker = threading.Thread(target=Orchestrator(root, "test", agent=agent).run, daemon=True)
    try:
        worker.start()
        deadline = time.monotonic() + 4
        while read_json(root / "run.json")["status"] != "queued":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert is_active(root)
        cancel_run(root)
        worker.join(timeout=3)
        assert not worker.is_alive()
        state = read_json(root / "run.json")
        assert state["status"] == "cancelled"
        assert state["active_seconds"] < 1
        assert not agent.requests
    finally:
        cancel_run(root)
        engine.close()
        worker.join(timeout=3)


def test_truncated_output_stops_instead_of_spending_tokens_on_same_request(
    manifest, tmp_path
) -> None:
    root = new_run(manifest, tmp_path)
    agent = ScriptedAgent([{"choices": [{"finish_reason": "length", "message": {"content": ""}}]}])
    state = Orchestrator(root, "test", agent=agent).run()
    assert state["items"][0]["status"] == "failed"
    assert state["items"][0]["stop_reason"] == "output_token_limit"
    assert len(agent.requests) == 1
