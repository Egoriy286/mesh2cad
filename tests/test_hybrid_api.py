"""Hybrid input contracts without credentials, external requests, or paid inference."""

import hashlib
import json
import time

import pytest
import trimesh
from fastapi.testclient import TestClient

from api import backend
from api.app import create_app
from api.service import Service


def evidence_for(target):
    return {
        "schema_version": "hybrid-v1",
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "coordinate_frame": "normalized-target",
        "sections": [{"axis": "z", "offset": 0, "area": 1}],
        "planes": [],
        "cylinders": [],
        "point_samples": [[0, 0, 0]],
        "warnings": [],
    }


def add_asset(service):
    folder = service.path("assets", "box", "")
    folder.mkdir(parents=True)
    (folder / "target.stl").write_bytes(trimesh.creation.box().export(file_type="stl"))
    (folder / "image.png").write_bytes(b"PNG fixture")
    asset = {"id": "box", "stats": {"normalized_extents": [1, 1, 1]}}
    with service.store.transaction():
        service.store.put("assets", asset)
    return folder


def wait(client, job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{job['id']}").json()
        if result["status"] in {"complete", "failed", "cancelled"}:
            return result
        time.sleep(0.02)
    raise AssertionError(result)


@pytest.mark.parametrize("mode", ["images", "hybrid"])
def test_transport_evidence_is_actually_sent_only_in_hybrid(tmp_path, monkeypatch, mode):
    captured = {}

    async def request(key, payload, cancel_check):
        captured.update(payload)
        return {"usage": {"cost": 0}}

    monkeypatch.setattr(backend, "_request", request)
    image = tmp_path / "image.png"
    image.write_bytes(b"PNG")
    evidence = {"schema_version": "hybrid-v1", "point_samples": [[0.123, 0.456, 0.789]]}
    backend.invoke(
        "never-log-secret",
        {
            "kind": "predict",
            "model": backend.SUPPORTED_MODEL,
            "reasoning_effort": "medium",
            "input_mode": mode,
            "_hybrid_evidence": evidence,
        },
        {"stats": {"normalized_extents": [1, 1, 1]}},
        None,
        image,
        lambda: False,
    )
    encoded = json.dumps(captured)
    assert "never-log-secret" not in encoded
    assert ("0.123" in encoded) == (mode == "hybrid")
    assert ("uncertain fits" in encoded) == (mode == "hybrid")
    assert captured["messages"][1]["content"][1]["type"] == "image_url"


def test_default_snapshot_and_legacy_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    service = Service(tmp_path, start_worker=False)
    add_asset(service)
    try:
        assert service.settings()["input_mode"] == "hybrid"
        assert service.settings()["supported_input_modes"] == ["images", "hybrid"]
        job = service.submit({"kind": "predict", "asset_id": "box"})
        service.update_settings({"input_mode": "images"})
        assert service.store.get("jobs", job["id"])["input_mode"] == "hybrid"
        assert service.submit({"kind": "predict", "asset_id": "box"})["input_mode"] == "images"
        with service.store.transaction():
            settings = service.store.get("settings", "global")
            settings.pop("input_mode")
            service.store.put("settings", settings)
        assert service.settings()["input_mode"] == "hybrid"
    finally:
        service.close()


def test_hybrid_cache_provenance_report_and_no_bulky_state(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    captured = []

    def prepare(target, output, **kwargs):
        evidence = evidence_for(target)
        output.write_text(json.dumps(evidence))
        return evidence

    def invoke(key, job, asset, parent, image, cancel_check):
        captured.append(dict(job))
        return {"choices": [{"message": {"content": '{"code":"r = invalid"}'}}]}, 0, {}

    monkeypatch.setattr("api.hybrid.prepare_evidence", prepare)
    monkeypatch.setattr("api.service.invoke", invoke)
    monkeypatch.setattr(
        "api.service.execute_code",
        lambda *args, **kwargs: {
            "status": "execution_error",
            "error": "fixture invalid code",
        },
    )
    with TestClient(create_app(tmp_path)) as client:
        service = client.app.state.service
        folder = add_asset(service)
        assert client.get("/api/assets/box/evidence").json()["status"] == "unavailable"
        client.post(
            "/api/settings",
            json={"provider": "openrouter", "api_key": "secret", "budget_limit_usd": 1},
        )
        result = wait(
            client, client.post("/api/jobs", json={"kind": "predict", "asset_id": "box"}).json()
        )
        assert result["status"] == "complete", result
        assert captured[0]["_hybrid_evidence"]["sections"]
        state = client.get("/api/state").json()
        assert "_hybrid_evidence" not in json.dumps(state)
        assert "secret" not in json.dumps(state)
        candidate = state["candidates"][0]
        assert candidate["input_mode"] == "hybrid"
        assert candidate["model_input_sent"] is True
        summary = candidate["input_evidence"]
        assert summary["sections"] == 1
        assert state["assets"][0]["input_evidence"] == summary
        assert client.get("/api/assets/box/evidence").json()["summary"] == summary
        report = client.get("/api/assets/box/report").json()
        assert report["candidates"][0]["input_evidence"] == summary
        assert "evidence_sha256" in client.get("/api/assets/box/report?format=csv").text
        # Provenance verification rejects stale/tampered target caches without regeneration.
        (folder / "target.stl").write_bytes(b"changed")
        assert client.get("/api/assets/box/evidence").json()["status"] == "unavailable"


@pytest.mark.parametrize("failure", ["preparation", "oversized"])
def test_preprocessing_failure_never_starts_paid_call(tmp_path, monkeypatch, failure):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def prepare(target, output, **kwargs):
        if failure == "preparation":
            raise ValueError("bounded worker timeout")
        return {**evidence_for(target), "warnings": ["x" * backend.MAX_EVIDENCE_BYTES]}

    def forbidden(*args, **kwargs):
        raise AssertionError("Paid transport must never run")

    monkeypatch.setattr("api.hybrid.prepare_evidence", prepare)
    monkeypatch.setattr("api.service.invoke", forbidden)
    with TestClient(create_app(tmp_path)) as client:
        service = client.app.state.service
        add_asset(service)
        client.post(
            "/api/settings",
            json={"provider": "openrouter", "api_key": "secret", "budget_limit_usd": 1},
        )
        job = client.post(
            "/api/jobs", json={"kind": "predict", "asset_id": "box", "max_attempts": 2}
        ).json()
        result = wait(client, job)
        assert result["status"] == "failed"
        assert "Hybrid evidence preparation failed" in result["error"]
        assert result["model_input_sent"] is False
        for call_id in result["reservations"]:
            call = service.store.get("calls", call_id)
            assert call["status"] == "unused"
            assert call["reserved_usd"] == 0


def test_legacy_job_executes_image_baseline(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    service = Service(tmp_path, start_worker=False)
    add_asset(service)
    service.update_settings({"provider": "openrouter", "api_key": "secret", "budget_limit_usd": 1})
    job = service.submit({"kind": "predict", "asset_id": "box"})
    job.pop("input_mode")
    captured = []
    monkeypatch.setattr(
        "api.hybrid.prepare_evidence",
        lambda *args, **kwargs: pytest.fail("legacy job prepared hybrid"),
    )
    monkeypatch.setattr(
        "api.service.invoke",
        lambda key, snapshot, *args: (
            captured.append(snapshot["input_mode"])
            or {"choices": [{"message": {"content": '{"code":"invalid"}'}}]},
            0,
            {},
        ),
    )
    monkeypatch.setattr(
        "api.service.execute_code",
        lambda *args, **kwargs: {"status": "execution_error", "error": "fixture"},
    )
    try:
        service.run(job)
        assert captured == ["images"]
        assert service.store.all("candidates")[0]["input_mode"] == "images"
    finally:
        service.close()


def test_prepare_input_is_local_and_creates_no_candidate(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def prepare(target, output, **kwargs):
        evidence = evidence_for(target)
        output.write_text(json.dumps(evidence))
        return evidence

    monkeypatch.setattr("api.hybrid.prepare_evidence", prepare)
    with TestClient(create_app(tmp_path)) as client:
        add_asset(client.app.state.service)
        job = client.post(
            "/api/jobs", json={"kind": "prepare_input", "asset_id": "box", "input_mode": "images"}
        ).json()
        result = wait(client, job)
        assert result["status"] == "complete", result
        assert result["input_evidence"]["sections"] == 1
        assert result["reservations"] == []
        assert result["model_input_sent"] is False
        assert client.get("/api/state").json()["candidates"] == []


def test_real_local_prepare_worker_and_cache_route(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        add_asset(client.app.state.service)
        job = client.post("/api/jobs", json={"kind": "prepare_input", "asset_id": "box"}).json()
        result = wait(client, job)
        assert result["status"] == "complete", result
        assert result["input_evidence"]["planes"] == 6
        assert result["input_evidence"]["sections"] == 9
        assert result["input_evidence"]["point_samples"] == 64
        assert result["model_input_sent"] is False
        cached = client.get("/api/assets/box/evidence").json()
        assert cached["status"] == "ready"
        assert cached["summary"] == result["input_evidence"]
        assert client.get("/api/state").json()["candidates"] == []


def test_invalid_input_mode_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/settings", json={"input_mode": "mesh"}).status_code == 422
        assert (
            client.post(
                "/api/jobs", json={"kind": "predict", "asset_id": "box", "input_mode": "mesh"}
            ).status_code
            == 422
        )


def test_csv_preserves_unknown_historical_input_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(create_app(tmp_path, start_worker=False)) as client:
        service = client.app.state.service
        add_asset(service)
        with service.store.transaction():
            service.store.put(
                "candidates",
                {
                    "id": "historical",
                    "asset_id": "box",
                    "parent_id": None,
                    "label": "Offline demo",
                    "status": "execution_error",
                    "error": None,
                    "metrics": None,
                    "source": "deterministic_offline_demo",
                },
            )
        import csv
        import io

        rows = list(
            csv.DictReader(io.StringIO(client.get("/api/assets/box/report?format=csv").text))
        )
        assert rows[0]["input_mode"] == "unknown"
        assert rows[0]["model_input_sent"] == ""
