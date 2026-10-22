"""The GUI starts workers once and keeps server-side credentials out of widgets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import trimesh
from streamlit.testing.v1 import AppTest

from mesh2cad import harness_ui
from mesh2cad.harness import HarnessConfig, RunLock, create_run, summarize
from mesh2cad.preview import evaluate_saved_pair
from mesh2cad.storage import read_json, write_json


def test_gui_starts_once_survives_rerun_and_requests_cancellation(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "data/deepcad"
    (dataset / "meshes").mkdir(parents=True)
    data = trimesh.creation.box().export(file_type="stl")
    (dataset / "meshes/box.stl").write_bytes(data)
    write_json(
        dataset / "manifest.json",
        {
            "schema_version": 1,
            "items": [
                {
                    "id": "box",
                    "source_file": "box.stl",
                    "split": "dev",
                    "status": "ready",
                    "mesh_path": "meshes/box.stl",
                    "mesh_sha256": hashlib.sha256(data).hexdigest(),
                }
            ],
        },
    )
    original = harness_ui.render_harness
    monkeypatch.setattr(harness_ui, "render_harness", lambda _: original(tmp_path))
    monkeypatch.setattr(harness_ui, "api_key_from_environment", lambda _: "server-only-secret")
    starts = []
    leases = []

    def start(root, key):
        assert key == "server-only-secret"
        starts.append(root)
        leases.append(RunLock(root))
        state = read_json(root / "run.json")
        state["status"] = "running"
        write_json(root / "run.json", state)

    monkeypatch.setattr(harness_ui, "start_run", start)
    monkeypatch.setattr("mesh2cad.harness.start_run", start)
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app)).run(timeout=30)
    try:
        assert not at.exception
        password = next(v for v in at.text_input if "API key" in v.label)
        assert password.value == ""
        next(v for v in at.button if v.label == "Запустить оркестратор").click().run(timeout=30)
        assert not at.exception
        assert len(starts) == 1
        at.run(timeout=30)
        assert not at.exception
        assert len(starts) == 1
        assert next(v for v in at.button if v.label == "Продолжить").disabled
        next(v for v in at.button if v.label == "Остановить").click().run(timeout=30)
        assert (starts[0] / "cancel.requested").is_file()
        assert "server-only-secret" not in (starts[0] / "run.json").read_text()
    finally:
        for lease in leases:
            lease.close()


def test_manual_pipeline_remains_available(monkeypatch) -> None:
    monkeypatch.setattr(harness_ui, "api_key_from_environment", lambda _: "")
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app)).run(timeout=30)
    next(v for v in at.radio if v.label == "Режим").set_value("Ручной pipeline").run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "STL → LLM"
    assert any(v.label == "Выберите STL" for v in at.get("file_uploader"))


def test_manual_pipeline_evaluates_byte_stl_in_bounded_worker(tmp_path, monkeypatch) -> None:
    mesh_dir = tmp_path / "data/deepcad/meshes"
    mesh_dir.mkdir(parents=True)
    trimesh.creation.box().export(mesh_dir / "box.stl")
    monkeypatch.setattr(
        "mesh2cad.pipeline.call_openrouter",
        lambda *args, **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"code": 'import cadquery as cq\nr = cq.Workplane("XY").box(1, 1, 1)'}
                        )
                    }
                }
            ]
        },
    )
    app = Path(__file__).resolve().parents[1] / "app.py"
    source = app.read_text().replace(
        "ROOT = Path(__file__).resolve().parent", f"ROOT = Path({str(tmp_path)!r})"
    )
    at = AppTest.from_string(source)
    at.session_state["app_mode"] = "Ручной pipeline"
    at.session_state["source_mode"] = "Датасет DeepCAD"
    at.session_state["api_key"] = "manual-test-secret"
    at.run(timeout=30)
    assert not at.exception
    next(v for v in at.button if v.label == "Отправить в OpenRouter").click().run(timeout=30)
    assert not at.exception
    next(v for v in at.button if v.label == "Выполнить в CadQuery и посчитать IoU").click().run(
        timeout=30
    )
    assert not at.exception
    cad = at.session_state["cad"]
    assert "metric_error" not in cad
    assert "difference_error" not in cad
    assert cad["iou"]["iou"] == pytest.approx(1)
    assert cad["overlay_sheet"].startswith(b"\x89PNG")


def test_project_console_does_not_render_hidden_meshes(interior_manifest, tmp_path, monkeypatch):
    original = harness_ui.render_harness
    monkeypatch.setattr(harness_ui, "render_harness", lambda _: original(tmp_path))
    monkeypatch.setattr(harness_ui, "api_key_from_environment", lambda _: "")

    def hidden_work(*args, **kwargs):
        raise AssertionError("Hidden tabs must not build previews or dashboards")

    monkeypatch.setattr(harness_ui, "_comparison", hidden_work)
    monkeypatch.setattr(harness_ui, "_dashboard", hidden_work)
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app))
    at.session_state["harness_workspace"] = "Консоль проекта"
    at.run(timeout=30)
    assert not at.exception
    assert any(v.label == "Скачать project.log" for v in at.get("download_button"))
    assert not any(v.label == "Запустить оркестратор" for v in at.button)
    assert len(at.get("plotly_chart")) == 0


def test_interior_dataset_filters_and_preview(interior_manifest, tmp_path, monkeypatch) -> None:
    original = harness_ui.render_harness
    monkeypatch.setattr(harness_ui, "render_harness", lambda _: original(tmp_path))
    monkeypatch.setattr(harness_ui, "api_key_from_environment", lambda _: "")
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app)).run(timeout=30)
    next(v for v in at.selectbox if v.label == "Датасет").set_value("Interior300").run(timeout=30)
    assert not at.exception
    assert len(next(v for v in at.selectbox if v.label == "Preview модели").options) == 1
    assert next(v for v in at.radio if v.label == "Split").options == ["all"]
    next(v for v in at.checkbox if v.label == "Только модели для объёмного IoU").uncheck().run()
    assert len(next(v for v in at.selectbox if v.label == "Preview модели").options) == 2
    next(v for v in at.multiselect if v.label == "Категории").set_value(["furniture"]).run()
    assert len(next(v for v in at.selectbox if v.label == "Preview модели").options) == 1
    next(v for v in at.checkbox if v.label == "Только модели для объёмного IoU").check().run()
    assert not at.exception
    assert next(v for v in at.button if v.label == "Запустить оркестратор").disabled


def test_saved_run_preview_follows_selected_iteration(
    interior_manifest, tmp_path, monkeypatch
) -> None:
    root = create_run(interior_manifest, tmp_path / "runs/harness", HarnessConfig(), limit=1)
    state = read_json(root / "run.json")
    item = state["items"][0]
    target = root / "items" / item["id"] / "target.stl"
    target.parent.mkdir(parents=True)
    trimesh.creation.box().export(target)
    attempts = []
    for number, size in enumerate((0.8, 0.5), 1):
        relative = f"items/{item['id']}/attempt-{number:03d}"
        candidate = root / relative / "cadquery/candidate.stl"
        candidate.parent.mkdir(parents=True)
        trimesh.creation.box(extents=[size] * 3).export(candidate)
        attempts.append(
            {
                "iteration": number,
                "status": "evaluated",
                "path": relative,
                "calls": [],
                "iou": evaluate_saved_pair(target, candidate),
                "execution": {"status": "valid", "mesh_path": str(candidate), "elapsed_seconds": 1},
            }
        )
    item.update(attempts=attempts, best_attempt=0, status="exhausted", stop_reason="max_iterations")
    state.update(status="completed", summary=summarize(state))
    write_json(root / "run.json", state)
    original = harness_ui.render_harness
    monkeypatch.setattr(harness_ui, "render_harness", lambda _: original(tmp_path))
    monkeypatch.setattr(harness_ui, "api_key_from_environment", lambda _: "")
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app)).run(timeout=30)
    assert not at.exception

    def current_score():
        return float(next(v for v in at.metric if v.label == "IoU выбранной итерации").value)

    assert current_score() == pytest.approx(0.512)
    next(v for v in at.selectbox if v.label == "Просмотр итерации").set_value("latest").run()
    assert not at.exception
    assert current_score() == pytest.approx(0.125)
    next(v for v in at.radio if v.label == "Режим preview").set_value("Рядом").run()
    assert not at.exception
    next(v for v in at.button if v.label == "Пересчитать IoU по STL").click().run()
    assert not at.exception
    assert current_score() == pytest.approx(0.125)
    assert read_json(root / "run.json")["items"][0]["best_attempt"] == 0
