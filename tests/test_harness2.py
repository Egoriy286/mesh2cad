"""Exercise depth-only analysis → CAD → depth comparison → repair with real CAD."""

import json

import pytest

from mesh2cad.agent import DepthAwareAgent
from mesh2cad.harness import HarnessConfig, cancel_run, create_run, make_orchestrator, run_isolated
from mesh2cad.harness2 import Orchestrator2, parse_analysis
from mesh2cad.storage import read_json, write_json

BOX = 'import cadquery as cq\nr = cq.Workplane("XY").box(1, 1, 1)'
SMALL = 'import cadquery as cq\nr = cq.Workplane("XY").box(.5, .5, .5)'


def response(content):
    return {
        "choices": [{"message": {"content": json.dumps(content)}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost": 0.001,
        },
    }


PLAN = response(
    {
        "components": [
            {"name": "body", "bounds": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], "primitive": "box"}
        ],
        "openings": [],
        "construction_plan": ["Make a centered unit box"],
        "uncertainties": [],
    }
)


class ScriptedVisionAgent(DepthAwareAgent):
    def __init__(self, outputs, after_call=None):
        super().__init__(
            "test/model", system_prompt=HarnessConfig(orchestrator="vision2").system_prompt
        )
        self.outputs = iter(outputs)
        self.requests = []
        self.after_call = after_call

    def complete(self, payload, api_key, timeout):
        self.requests.append(payload)
        result = next(self.outputs)
        if self.after_call:
            self.after_call()
        return result


def new_run(manifest, tmp_path, **overrides):
    return create_run(
        manifest,
        tmp_path / "runs",
        HarnessConfig(
            orchestrator="vision2",
            model="test/model",
            max_iterations=2,
            retry_backoff=0,
            **overrides,
        ),
        limit=1,
    )


def image_count(payload):
    return sum(
        p.get("type") == "image_url"
        for m in payload["messages"]
        for p in (m["content"] if isinstance(m["content"], list) else [])
    )


def test_v2_analyzes_depth_then_repairs_geometry_without_any_fitting(interior_manifest, tmp_path):
    root = new_run(interior_manifest, tmp_path)
    agent = ScriptedVisionAgent([PLAN, response({"code": SMALL}), response({"code": BOX})])
    worker = make_orchestrator(root, "unit-key", agent=agent)
    assert isinstance(worker, Orchestrator2)
    state = worker.run()
    assert state["vision_prompt_version"] == 2
    item = state["items"][0]
    assert item["status"] == "success"
    assert len(agent.requests) == 3
    assert item["attempts"][0]["iou"]["iou"] == pytest.approx(0.125)
    assert item["attempts"][1]["iou"]["iou"] == pytest.approx(1)
    assert [image_count(p) for p in agent.requests] == [1, 1, 2]
    assert (
        'Return exactly JSON with a string field "code"'
        not in agent.requests[0]["messages"][0]["content"]
    )
    assert "target-linear-depth" in json.dumps(agent.requests[1])
    assert "best-cad-linear-depth" in json.dumps(agent.requests[2])
    for payload in agent.requests:
        serialized = json.dumps(payload).lower()
        assert "rgb" not in serialized and "blueprint" not in serialized
        assert "difference-scheme" not in serialized
    assert "COMPLETE executable CadQuery Python" in json.dumps(agent.requests[2]["messages"][-1])
    assert item["analysis"]["status"] == "ready"
    assert state["summary"]["llm_calls"] == 3
    assert state["summary"]["total_tokens"] == 360
    assert state["summary"]["cost_usd"] == pytest.approx(0.003)
    for attempt in item["attempts"]:
        assert "optimization" not in attempt and "raw_iou" not in attempt
        folder = root / attempt["path"]
        assert (folder / "vision/depth.png").is_file()
        assert not (folder / "vision/rgb.png").exists()
        assert not (folder / "vision/blueprint.png").exists()
        assert not (folder / "difference-scheme.png").exists()
        assert not (folder / "overlay.png").exists()
    prepared = root / "items" / item["id"]
    assert (prepared / "target.png").read_bytes() == (prepared / "vision/depth.png").read_bytes()
    assert all(v["depth_mae"] == 0 for v in item["attempts"][1]["visual_metrics"]["views"])


def test_analysis_is_durable_and_counts_towards_budget(interior_manifest, tmp_path):
    root = new_run(interior_manifest, tmp_path, total_token_budget=120)
    agent = ScriptedVisionAgent([PLAN])
    state = make_orchestrator(root, "key", agent=agent).run()
    assert state["status"] == "budget_exhausted"
    assert len(agent.requests) == 1
    assert state["summary"]["total_tokens"] == 120
    assert not list(root.glob("items/*/attempt-*/cadquery/candidate.stl"))


def test_cancel_after_analysis_resumes_without_repaying_analysis(interior_manifest, tmp_path):
    root = new_run(interior_manifest, tmp_path)
    first = ScriptedVisionAgent([PLAN], after_call=lambda: cancel_run(root))
    stopped = make_orchestrator(root, "key", agent=first).run()
    assert stopped["status"] == "cancelled"
    second = ScriptedVisionAgent([response({"code": BOX})])
    state = make_orchestrator(root, "key", agent=second).run()
    assert state["items"][0]["status"] == "success"
    assert len(first.requests) == len(second.requests) == 1
    assert state["summary"]["llm_calls"] == 2
    assert state["summary"]["total_tokens"] == 240


def test_subprocess_factory_uses_v2_with_saved_response(interior_manifest, tmp_path):
    root = new_run(interior_manifest, tmp_path)
    state = read_json(root / "run.json")
    item = state["items"][0]
    relative = f"items/{item['id']}/attempt-001"
    (root / relative).mkdir(parents=True)
    write_json(root / relative / "response.json", response({"code": BOX}))
    item["attempts"] = [
        {
            "iteration": 1,
            "path": relative,
            "status": "requesting",
            "calls": [{"status": "requesting", "phase": "generation"}],
        }
    ]
    write_json(root / "run.json", state)
    result = run_isolated(root, "fake-key-no-network")
    assert result["items"][0]["status"] == "success"
    assert "visual_metrics" in result["items"][0]["attempts"][0]
    assert result["summary"]["llm_calls"] == 1
    assert read_json(root / "memory.json")["peak_rss_mb"] < 2048


def test_analysis_bounds_and_legacy_options_are_validated():
    bad = response({"components": [{"bounds": [[0, 0, 0], [0, 1, 1]]}]})
    with pytest.raises(ValueError):
        parse_analysis(bad, [1, 1, 1])
    config = HarnessConfig.from_dict({"numerical_refinement": True, "optimization_seconds": 90})
    assert config.orchestrator == "classic"
    assert not hasattr(config, "numerical_refinement")
    from mesh2cad.prompts import LEGACY_VISION_SYSTEM_PROMPT, VISION_SYSTEM_PROMPT

    migrated = HarnessConfig.from_dict(
        {"orchestrator": "vision2", "system_prompt": LEGACY_VISION_SYSTEM_PROMPT}
    )
    assert migrated.system_prompt == VISION_SYSTEM_PROMPT


def test_depth_only_launch_does_not_reuse_old_multimodal_job(
    interior_manifest, tmp_path, monkeypatch
):
    from mesh2cad.harness import launch_run
    from mesh2cad.prompts import LEGACY_VISION_SYSTEM_PROMPT

    config = HarnessConfig(orchestrator="vision2")
    old = create_run(interior_manifest, tmp_path / "runs", config, limit=1)
    state = read_json(old / "run.json")
    state["vision_prompt_version"] = 1
    state["config"]["system_prompt"] = LEGACY_VISION_SYSTEM_PROMPT
    write_json(old / "run.json", state)
    starts = []
    monkeypatch.setattr("mesh2cad.harness.start_run", lambda root, key: starts.append(root))
    fresh, reused = launch_run(
        interior_manifest,
        tmp_path / "runs",
        config,
        "test-key",
        split="all",
        item_ids=[state["items"][0]["id"]],
        limit=1,
    )
    assert not reused and fresh != old
    assert starts == [fresh]
    assert read_json(fresh / "run.json")["vision_prompt_version"] == 2
