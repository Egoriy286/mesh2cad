import pytest
import trimesh

from mesh2cad import dataset, harness
from mesh2cad.storage import read_json


class FakeBackend:
    model = "fake"
    reasoning_effort = "medium"
    timeout = 10

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0

    def generate(self, prompt, image_path, output_dir):
        self.calls += 1
        code = next(self.replies)
        return {
            "status": "quota_exhausted" if code == "QUOTA" else "ok",
            "code": None if code == "QUOTA" else code,
            "usage": {"input_tokens": 20, "output_tokens": 5},
            "error": "quota" if code == "QUOTA" else None,
            "elapsed_seconds": 0.1,
            "launches": 1,
        }


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        trimesh.creation.box().export(source / f"{index}.stl")

    def render(mesh, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"image")

    monkeypatch.setattr(dataset, "render_views", render)
    monkeypatch.setattr(harness, "render_comparison", lambda a, b, path: render(a, path))

    def execute(code, work_dir, timeout):
        work_dir.mkdir(parents=True, exist_ok=True)
        if code == "BAD":
            return {
                "status": "execution_error",
                "error": "bad code",
                "mesh_path": None,
                "step_path": None,
                "elapsed_seconds": 0.1,
            }
        mesh = trimesh.creation.box(extents=[float(code)] * 3)
        mesh_path, step_path = work_dir / "candidate.stl", work_dir / "candidate.step"
        mesh.export(mesh_path)
        step_path.write_text("test STEP")
        return {
            "status": "valid",
            "error": None,
            "mesh_path": str(mesh_path),
            "step_path": str(step_path),
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(harness, "execute_code", execute)
    manifest = dataset.prepare_dataset(
        tmp_path / "data", source_dir=source, dev_count=1, test_count=2
    )
    return manifest, tmp_path / "run"


def test_best_so_far_preserves_first_and_resume_extends_without_repeating(experiment):
    manifest, output = experiment
    backend = FakeBackend(["1", "0.6", "0.8"])
    state = harness.run_experiment(
        manifest, output, backend=backend, limit=1, n_points=100, preflight=False
    )
    assert state["status"] == "complete"
    selected = read_json(output / "selection.json")["items"]
    result = read_json(output / "items" / selected[0]["id"] / "result.json")
    assert result["best_index"] == 0
    assert len(result["attempts"]) == 3
    assert backend.calls == 3
    backend = FakeBackend(["BAD", "1", "0.7"])
    state = harness.run_experiment(
        manifest, output, backend=backend, limit=2, n_points=100, preflight=False, resume=True
    )
    assert state["completed"] == 2
    assert backend.calls == 3
    second = read_json(output / "selection.json")["items"][1]
    result = read_json(output / "items" / second["id"] / "result.json")
    assert result["best_index"] == 1
    assert result["attempts"][0]["execution_status"] == "execution_error"
    assert (output / "items" / second["id"] / "best.step").exists()


def test_quota_retry_keeps_usage_and_finished_attempts(experiment):
    manifest, output = experiment
    backend = FakeBackend(["0.8", "QUOTA"])
    state = harness.run_experiment(
        manifest, output, backend=backend, limit=1, n_points=100, preflight=False
    )
    assert state["status"] == "quota_exhausted"
    backend = FakeBackend(["1", "0.7"])
    harness.run_experiment(
        manifest, output, backend=backend, limit=1, n_points=100, preflight=False, resume=True
    )
    assert backend.calls == 2
    selected = read_json(output / "selection.json")["items"][0]
    result = read_json(output / "items" / selected["id"] / "result.json")
    assert len(result["attempts"][1]["generations"]) == 2
    assert result["attempts"][0]["generations"][0]["code"] == "0.8"


def test_resume_rejects_model_and_budget_changes(experiment):
    manifest, output = experiment
    harness.run_experiment(
        manifest, output, backend=FakeBackend(["1"]), limit=1, max_attempts=1, preflight=False
    )
    with pytest.raises(ValueError, match="changed"):
        harness.run_experiment(
            manifest,
            output,
            backend=FakeBackend([]),
            limit=1,
            max_attempts=2,
            preflight=False,
            resume=True,
        )


def test_missing_inputs_keep_denominator(experiment):
    manifest, output = experiment
    data = read_json(manifest)
    test_item = next(i for i in data["items"] if i["split"] == "test")
    test_item.update(status="prepare_error", error="broken STL")
    from mesh2cad.storage import write_json

    write_json(manifest, data)
    backend = FakeBackend([])
    state = harness.run_experiment(manifest, output, backend=backend, limit=1, preflight=False)
    assert state["completed"] == 1
    assert backend.calls == 0
    assert read_json(output / "items" / test_item["id"] / "result.json")["status"] == "input_error"


def test_choose_best_rejects_unscored_and_nonfinite_metric_improvements():
    assert harness.choose_best([]) is None
    attempts = [
        {"index": 0, "execution_status": "valid", "metrics": None},
        {"index": 1, "execution_status": "valid", "metrics": {"chamfer": 0.1}},
        {"index": 2, "execution_status": "valid", "metrics": {"chamfer": float("nan")}},
    ]
    assert harness.choose_best(attempts) == 1


def test_resume_reuses_paid_generation_when_code_checkpoint_is_missing(experiment):
    manifest, output = experiment
    harness.run_experiment(
        manifest,
        output,
        backend=FakeBackend(["1"]),
        limit=1,
        max_attempts=1,
        n_points=100,
        preflight=False,
    )
    selected = read_json(output / "selection.json")["items"][0]
    path = output / "items" / selected["id"] / "result.json"
    result = read_json(path)
    result["status"] = "interrupted"
    attempt = result["attempts"][0]
    attempt["finished"] = False
    (output / attempt["artifacts"]["code"]).unlink()
    from mesh2cad.storage import write_json

    write_json(path, result)
    backend = FakeBackend([])
    harness.run_experiment(
        manifest,
        output,
        backend=backend,
        limit=1,
        max_attempts=1,
        n_points=100,
        preflight=False,
        resume=True,
    )
    assert backend.calls == 0
    assert (output / attempt["artifacts"]["code"]).read_text() == "1"


def test_preflight_recovers_successful_generation_after_execution_interruption(
    experiment, monkeypatch
):
    _, output = experiment
    backend = FakeBackend(["1"])
    original_execute = harness.execute_code
    monkeypatch.setattr(harness, "render_views", lambda mesh, path: path.write_bytes(b"image"))

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(harness, "execute_code", interrupted)
    with pytest.raises(KeyboardInterrupt):
        harness.run_smoke(backend, output)
    assert backend.calls == 1
    monkeypatch.setattr(harness, "execute_code", original_execute)
    assert harness.run_smoke(backend, output)["status"] == "ok"
    assert backend.calls == 1
