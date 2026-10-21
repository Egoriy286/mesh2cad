"""Offline contract/integration checks; no API credentials or paid calls."""

import json
import time

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from api.app import create_app
from api.backend import CALL_RESERVE
from api.service import Service


def wait_job(client, job_id, timeout=35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"complete", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.1)
    raise AssertionError(f"Job did not finish: {job}")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        yield client


def test_real_demo_execute_explain_edit_and_exports(client):
    asset = client.post("/api/examples/block").json()
    state = client.get("/api/state").json()
    job = wait_job(client, state["jobs"][0]["id"])
    assert job["status"] == "complete", job
    state = client.get("/api/state").json()
    first = state["candidates"][0]
    assert first["status"] == "valid", first
    assert first["source"] == "deterministic_offline_demo"
    assert first["metrics"]["iou"] == pytest.approx(1, abs=1e-6)
    assert first["metrics"]["n_points"] == 8192
    for field in ("mesh_url", "step_url", "comparison_url"):
        assert client.get(first[field]).status_code == 200
    explain = client.post(
        "/api/jobs",
        json={
            "kind": "explain",
            "asset_id": asset["id"],
            "parent_id": first["id"],
            "prompt": "why?",
        },
    ).json()
    assert wait_job(client, explain["id"])["status"] == "complete"
    assert len(client.get("/api/state").json()["candidates"]) == 1
    edit = client.post(
        "/api/jobs",
        json={
            "kind": "edit",
            "asset_id": asset["id"],
            "parent_id": first["id"],
            "prompt": "round the edges",
        },
    ).json()
    assert wait_job(client, edit["id"])["status"] == "complete"
    state = client.get("/api/state").json()
    assert state["candidates"][0] == first
    second = state["candidates"][1]
    assert second["parent_id"] == first["id"]
    assert second["status"] == "valid", second
    geometry = client.get(
        "/api/geometry", params={"asset_id": asset["id"], "candidate_id": second["id"]}
    ).json()
    assert geometry["target"]["faces"] and geometry["candidate"]["faces"]
    assert geometry["preview"]["evaluation_uses_original"]
    assert (
        client.get(f"/api/assets/{asset['id']}/report").json()["candidates"][1]["id"]
        == second["id"]
    )
    assert "candidate_id" in client.get(f"/api/assets/{asset['id']}/report?format=csv").text
    assert state["messages"][-1]["related_candidate_id"] == second["id"]


@pytest.mark.parametrize("fmt", ["stl", "obj", "ply", "off", "glb"])
def test_surface_formats_and_normalization(client, fmt):
    mesh = trimesh.creation.box(extents=[2, 4, 1])
    mesh.apply_translation([10, 20, 30])
    raw = mesh.export(file_type=fmt)
    if isinstance(raw, str):
        raw = raw.encode()
    response = client.post("/api/assets", files={"file": (f"shape.{fmt}", raw)})
    assert response.status_code == 201, response.text
    asset = response.json()
    assert asset["stats"]["normalized_extents"] == pytest.approx([0.5, 1, 0.25])
    forward = np.array(asset["normalization"]["transform"])
    inverse = np.array(asset["normalization"]["inverse_transform"])
    assert np.allclose(forward @ inverse, np.eye(4))


def test_glb_scene_transforms_preserved(client):
    scene = trimesh.Scene()
    for index in range(2):
        transform = np.eye(4)
        transform[0, 3] = index * 5
        scene.add_geometry(trimesh.creation.box(), transform=transform)
    response = client.post(
        "/api/assets", files={"file": ("instances.glb", scene.export(file_type="glb"))}
    )
    assert response.status_code == 201, response.text
    assert response.json()["stats"]["normalized_extents"] == pytest.approx([1, 1 / 6, 1 / 6])


def test_point_only_and_external_resource_rejection(client):
    raw = trimesh.points.PointCloud([[0, 0, 0], [1, 1, 1]]).export(file_type="ply")
    response = client.post("/api/assets", files={"file": ("points.ply", raw)})
    assert response.status_code == 400
    assert "triangular surface" in response.json()["detail"]
    raw = b"mtllib /etc/passwd\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    response = client.post("/api/assets", files={"file": ("external.obj", raw)})
    assert response.status_code == 400
    assert "External OBJ" in response.json()["detail"]


def test_manual_failure_is_not_zero_metrics(client):
    asset = client.post(
        "/api/assets", files={"file": ("box.stl", trimesh.creation.box().export(file_type="stl"))}
    ).json()
    response = client.post(
        "/api/jobs", json={"kind": "execute", "asset_id": asset["id"], "code": "r = missing_name"}
    )
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "complete"
    candidate = client.get("/api/state").json()["candidates"][0]
    assert candidate["status"] == "execution_error"
    assert candidate["metrics"] is None
    assert candidate["mesh_url"] is None
    assert client.get(f"/api/candidates/{candidate['id']}/mesh").status_code == 404


def test_idempotency_budget_cancel_and_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(create_app(tmp_path, start_worker=False)) as client:
        asset = client.post(
            "/api/assets",
            files={"file": ("box.stl", trimesh.creation.box().export(file_type="stl"))},
        ).json()
        secret = "test-secret-never-persist"
        settings = client.post(
            "/api/settings",
            json={"api_key": secret, "provider": "openrouter", "budget_limit_usd": 0.5},
        ).json()
        assert secret not in json.dumps(settings)
        request = {"kind": "predict", "asset_id": asset["id"]}
        first = client.post("/api/jobs", json=request, headers={"Idempotency-Key": "same"}).json()
        duplicate = client.post(
            "/api/jobs", json=request, headers={"Idempotency-Key": "same"}
        ).json()
        assert first["id"] == duplicate["id"]
        budget = client.get("/api/settings").json()["budget"]
        assert budget["reserved_usd"] >= CALL_RESERVE
        assert (
            client.post(
                "/api/jobs",
                json={**request, "prompt": "different"},
                headers={"Idempotency-Key": "same"},
            ).status_code
            == 400
        )
        assert client.post(f"/api/jobs/{first['id']}/cancel").json()["status"] == "cancelled"
        assert secret not in json.dumps(client.get("/api/state").json())
    assert secret.encode() not in (tmp_path / "workbench.sqlite3").read_bytes()
    with TestClient(create_app(tmp_path, start_worker=False)) as client:
        assert not client.get("/api/settings").json()["settings"]["api_key_configured"]
        assert (
            client.post("/api/jobs", json=request, headers={"Idempotency-Key": "same"}).json()["id"]
            == first["id"]
        )
        assert client.post("/api/jobs", json=request).status_code == 400


def test_running_cancel_is_prompt(client):
    asset = client.post(
        "/api/assets", files={"file": ("box.stl", trimesh.creation.box().export(file_type="stl"))}
    ).json()
    job = client.post(
        "/api/jobs", json={"kind": "execute", "asset_id": asset["id"], "code": "while True: pass"}
    ).json()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = client.get(f"/api/jobs/{job['id']}").json()
        if current["stage"] == "sandbox execution":
            break
        time.sleep(0.05)
    started = time.monotonic()
    cancelled = client.post(f"/api/jobs/{job['id']}/cancel")
    assert time.monotonic() - started < 1
    assert cancelled.status_code == 200
    assert wait_job(client, job["id"], timeout=5)["status"] == "cancelled"


def test_protocol_bounds_and_origin(client):
    for values in ({"budget_limit_usd": -1}, {"budget_limit_usd": "NaN"}, {"max_attempts": 100}):
        assert client.post("/api/settings", json=values).status_code == 422
    assert client.post("/api/settings", json={"provider": "codex"}).status_code == 400
    assert (
        client.post(
            "/api/jobs", json={"kind": "execute", "asset_id": "x", "code": "x" * 60001}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/settings", json={}, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/settings", json={}, headers={"Origin": "http://127.0.0.1:8501"}
        ).status_code
        == 200
    )


def test_recovery_retains_unknown_call_but_releases_unstarted(tmp_path):
    service = Service(tmp_path, start_worker=False)
    with service.store.transaction():
        for call_id, status in (("sent", "in_flight"), ("unused", "not_started")):
            service.store.put(
                "calls",
                {"id": call_id, "reserved_usd": CALL_RESERVE, "known_usd": 0, "status": status},
            )
        service.store.put(
            "jobs", {"id": "crashed", "status": "running", "reservations": ["sent", "unused"]}
        )
    service.close()
    restarted = Service(tmp_path, start_worker=False)
    assert restarted.store.get("jobs", "crashed")["status"] == "interrupted"
    assert restarted.store.get("calls", "sent")["reserved_usd"] == CALL_RESERVE
    assert restarted.store.get("calls", "unused")["reserved_usd"] == 0
    restarted.close()


def test_failed_sandbox_symlink_never_becomes_geometry(client, tmp_path):
    fixture = tmp_path / "external.stl"
    fixture.write_bytes(trimesh.creation.box(extents=[9, 8, 7]).export(file_type="stl"))
    asset = client.post(
        "/api/assets", files={"file": ("box.stl", trimesh.creation.box().export(file_type="stl"))}
    ).json()
    code = f"import os\nos.symlink({str(fixture)!r}, 'candidate.stl')\nraise RuntimeError('deliberate failure')\n"
    job = client.post(
        "/api/jobs", json={"kind": "execute", "asset_id": asset["id"], "code": code}
    ).json()
    assert wait_job(client, job["id"])["status"] == "complete"
    candidate = client.get("/api/state").json()["candidates"][0]
    assert candidate["status"] == "execution_error"
    assert candidate["mesh_url"] is None
    geometry = client.get(
        "/api/geometry", params={"asset_id": asset["id"], "candidate_id": candidate["id"]}
    ).json()
    assert geometry["candidate"] is None
    assert client.get(f"/api/candidates/{candidate['id']}/mesh").status_code == 404


def test_terminal_cancel_and_updates_are_monotonic(tmp_path):
    service = Service(tmp_path, start_worker=False)
    with service.store.transaction():
        service.store.put(
            "jobs",
            {"id": "finished", "status": "complete", "stage": "complete", "reservations": []},
        )
    assert service.cancel("finished")["status"] == "complete"
    assert service.update_job("finished", status="cancelling")["status"] == "complete"
    service.close()


def test_openrouter_refinement_budget_and_context(tmp_path, monkeypatch):
    calls = []

    def fake_invoke(key, job, asset, parent, image, cancel_check):
        calls.append({"parent": parent, "image": image, "kind": job["kind"]})
        if job["kind"] == "explain":
            return (
                {"choices": [{"message": {"content": "A plain explanation"}}]},
                0.001,
                {"cost": 0.001},
            )
        code = 'import cadquery as cq\nr = cq.Workplane("XY").box(1,1,1)\n'
        return (
            {"choices": [{"message": {"content": json.dumps({"code": code})}}]},
            0.002,
            {"cost": 0.002},
        )

    monkeypatch.setattr("api.service.invoke", fake_invoke)
    with TestClient(create_app(tmp_path)) as client:
        client.post("/api/settings", json={"api_key": "offline-test-key", "provider": "openrouter"})
        asset = client.post(
            "/api/assets",
            files={"file": ("box.stl", trimesh.creation.box().export(file_type="stl"))},
        ).json()
        budget_before = client.get("/api/settings").json()["budget"]
        job = client.post(
            "/api/jobs", json={"kind": "predict", "asset_id": asset["id"], "max_attempts": 2}
        ).json()
        result = wait_job(client, job["id"])
        assert result["status"] == "complete", result
        candidates = client.get("/api/state").json()["candidates"]
        assert len(candidates) == 2
        assert candidates[1]["parent_id"] == candidates[0]["id"]
        assert calls[1]["parent"]["metrics"]["iou"] == pytest.approx(1)
        assert calls[1]["image"].name == "comparison.png"
        explain = client.post(
            "/api/jobs",
            json={"kind": "explain", "asset_id": asset["id"], "parent_id": candidates[0]["id"]},
        ).json()
        assert wait_job(client, explain["id"])["status"] == "complete"
        assert calls[-1]["parent"]["metrics"]
        assert calls[-1]["image"].name == "comparison.png"
        budget = client.get("/api/settings").json()["budget"]
        assert budget["known_usd"] == pytest.approx(budget_before["known_usd"] + 0.005)
        assert budget["reserved_usd"] == pytest.approx(budget_before["reserved_usd"])


def test_openrouter_protocol_has_price_guard_and_geometry(tmp_path, monkeypatch):
    from api import backend

    captured = {}

    async def fake_request(key, payload, cancel_check):
        captured.update(payload)
        return {"choices": [{"message": {"content": "Explanation"}}], "usage": {"cost": 0.003}}

    monkeypatch.setattr(backend, "_request", fake_request)
    image = tmp_path / "comparison.png"
    image.write_bytes(b"PNG placeholder for transport test")
    parent = {"code": "r = ...", "metrics": {"iou": 0.5}}
    _, cost, _ = backend.invoke(
        "secret",
        {"kind": "explain", "model": backend.SUPPORTED_MODEL, "reasoning_effort": "medium"},
        {"stats": {"normalized_extents": [1, 1, 1]}},
        parent,
        image,
        lambda: False,
    )
    assert cost == 0.003
    assert captured["provider"] == {
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 0.1, "completion": 0.5},
    }
    assert "0.5" in captured["messages"][1]["content"][0]["text"]
    assert captured["messages"][1]["content"][1]["type"] == "image_url"
    assert "secret" not in json.dumps(captured)


def test_queued_paid_restart_without_key_never_sends(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Transport must not run without credentials")

    monkeypatch.setattr("api.service.invoke", forbidden)
    with TestClient(create_app(tmp_path, start_worker=False)) as client:
        client.post("/api/settings", json={"api_key": "memory-only", "provider": "openrouter"})
        asset = client.post(
            "/api/assets",
            files={"file": ("box.stl", trimesh.creation.box().export(file_type="stl"))},
        ).json()
        budget_before = client.get("/api/settings").json()["budget"]
        job = client.post("/api/jobs", json={"kind": "predict", "asset_id": asset["id"]}).json()
    with TestClient(create_app(tmp_path)) as client:
        finished = wait_job(client, job["id"])
        assert finished["status"] == "failed"
        assert "Configure the server API key" in finished["error"]
        assert client.get("/api/settings").json()["budget"]["reserved_usd"] == pytest.approx(
            budget_before["reserved_usd"]
        )
