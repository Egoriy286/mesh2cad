"""Authorization failures never consume a generation reservation or expose keys."""

import asyncio
import json

import httpx
import pytest
import trimesh
from fastapi.testclient import TestClient

from api import backend
from api.app import create_app
from api.backend import AUTH_ERROR, CALL_RESERVE, OpenRouterHTTPError
from api.service import Service
from tests.test_workbench_api import wait_job


@pytest.mark.parametrize("status", [401, 503])
def test_transport_status_is_structured_and_response_is_not_exposed(monkeypatch, status):
    original = httpx.AsyncClient

    def response(request):
        assert request.headers["Authorization"] == "Bearer private-test-key"
        return httpx.Response(status, json={"error": {"message": "private-test-key"}})

    monkeypatch.setattr(
        backend.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(response), **kwargs),
    )
    with pytest.raises(OpenRouterHTTPError) as error:
        asyncio.run(backend._request("private-test-key", {}, lambda: False))
    assert error.value.status_code == status
    assert "private-test-key" not in str(error.value)
    assert ("освобождён" in str(error.value)) == (status == 401)


@pytest.mark.parametrize("status,expected", [(200, "valid"), (401, "invalid"), (503, "unverified")])
def test_key_check_only_reads_auth_metadata(monkeypatch, status, expected):
    original = httpx.Client
    requests = []

    def response(request):
        requests.append(request)
        return httpx.Response(status, json={"data": {"label": "private-test-key"}})

    monkeypatch.setattr(
        backend.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(response), **kwargs),
    )
    result = backend.verify_key("private-test-key")
    assert result["status"] == expected
    assert "private-test-key" not in json.dumps(result)
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == "https://openrouter.ai/api/v1/key"


@pytest.mark.parametrize("status", [401, 503])
def test_auth_failure_releases_only_known_rejection(tmp_path, monkeypatch, status):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    sent = []

    def reject(*args, **kwargs):
        sent.append(True)
        raise OpenRouterHTTPError(status)

    monkeypatch.setattr("api.service.invoke", reject)
    with TestClient(create_app(tmp_path)) as client:
        client.post(
            "/api/settings", json={"api_key": "  `private-test-key`  ", "provider": "openrouter"}
        )
        asset = client.post(
            "/api/assets",
            files={"file": ("box.stl", trimesh.creation.box().export(file_type="stl"))},
        ).json()
        before = client.get("/api/settings").json()["budget"]
        body = {"kind": "predict", "asset_id": asset["id"], "max_attempts": 2}
        submitted = client.post("/api/jobs", json=body).json()
        job = wait_job(client, submitted["id"])
        assert job["status"] == "failed"
        settings = client.get("/api/settings").json()
        expected_reserve = before["reserved_usd"] + (CALL_RESERVE if status != 401 else 0)
        assert settings["budget"]["reserved_usd"] == pytest.approx(expected_reserve)
        assert settings["budget"]["known_usd"] == before["known_usd"]
        assert len(sent) == 1
        if status == 401:
            assert job["error"] == AUTH_ERROR
            assert settings["settings"]["api_key_status"] == "invalid"
            assert client.post("/api/jobs", json=body).status_code == 400
            assert len(sent) == 1
            monkeypatch.setattr(
                "api.service.verify_key", lambda key: {"status": "valid", "message": "OK"}
            )
            assert client.post("/api/auth/check").json()["status"] == "valid"
            assert client.get("/api/settings").json()["settings"]["api_key_status"] == "valid"
        assert b"private-test-key" not in (tmp_path / "workbench.sqlite3").read_bytes()


def test_legacy_401_recovery_does_not_release_unknown_costs(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    service = Service(tmp_path, start_worker=False)
    with service.store.transaction():
        for status in (401, 503):
            service.store.put(
                "calls",
                {
                    "id": str(status),
                    "status": "in_flight",
                    "reserved_usd": CALL_RESERVE,
                    "known_usd": 0,
                },
            )
            service.store.put(
                "jobs",
                {
                    "id": str(status),
                    "status": "failed",
                    "reservations": [str(status)],
                    "error": f"OpenRouter HTTP {status}; reservation retained pending reconciliation",
                },
            )
    before = service.budget()
    service.close()
    for _ in range(2):
        service = Service(tmp_path, start_worker=False)
        assert service.store.get("calls", "401")["status"] == "rejected"
        assert service.store.get("calls", "401")["reserved_usd"] == 0
        assert service.store.get("calls", "503")["reserved_usd"] == CALL_RESERVE
        assert service.budget()["reserved_usd"] == pytest.approx(
            before["reserved_usd"] - CALL_RESERVE
        )
        service.close()


def test_pasted_key_cleanup_and_invalid_whitespace_are_atomic(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    service = Service(tmp_path, start_worker=False)
    service.update_settings({"api_key": "  `Bearer private-test-key` \n"})
    assert service.key == "private-test-key"
    with pytest.raises(ValueError, match="пробелы"):
        service.update_settings({"api_key": "bad\nkey", "provider": "openrouter"})
    assert service.key == "private-test-key"
    assert service.settings()["provider"] == "demo"
    service.close()
