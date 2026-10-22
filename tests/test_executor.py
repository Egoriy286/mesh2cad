"""Sandbox and CadQuery execution tests."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest
import trimesh

from mesh2cad import executor

SANDBOX = platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").is_file()


def test_environment_excludes_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "/private/credentials")
    environment = executor._environment(tmp_path)
    assert "OPENROUTER_API_KEY" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["HOME"] == str(tmp_path / "home")


def test_profile_denies_network_and_repository_reads(tmp_path) -> None:
    profile = executor._sandbox_profile(tmp_path)
    assert "(deny default)" in profile
    assert "(allow network" not in profile
    assert f'(subpath "{Path.cwd()}")' not in profile


def test_missing_sandbox_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(executor.platform, "system", lambda: "Linux")
    result = executor.execute_code('r = cq.Workplane("XY").box(1, 1, 1)', tmp_path)
    assert result["status"] == "sandbox_error"
    assert result["mesh_path"] is None
    assert result["step_path"] is None


@pytest.mark.skipif(not SANDBOX, reason="macOS Seatbelt is required")
def test_execute_code_exports_real_step_and_mesh(tmp_path) -> None:
    pytest.importorskip("cadquery")
    result = executor.execute_code(
        'r = cq.Workplane("XY").box(1, 2, 3)',
        tmp_path,
        timeout=60,
    )
    assert result["status"] == "valid", result.get("error")
    assert "ISO-10303-21" in Path(result["step_path"]).read_text(encoding="utf-8")
    mesh = trimesh.load_mesh(result["mesh_path"])
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(6.0, rel=1e-4)


@pytest.mark.skipif(not SANDBOX, reason="macOS Seatbelt is required")
def test_execute_code_reports_traceback(tmp_path) -> None:
    pytest.importorskip("cadquery")
    result = executor.execute_code(
        'raise RuntimeError("broken CAD")',
        tmp_path,
        timeout=60,
    )
    assert result["status"] == "execution_error"
    assert "broken CAD" in result["error"]
    assert not (tmp_path / "candidate.stl").exists()
