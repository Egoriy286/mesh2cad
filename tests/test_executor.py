"""Cross-platform executor and CadQuery sandbox tests."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest
import trimesh

from mesh2cad import executor

SEATBELT = platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").is_file()


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


def test_auto_backend_is_supported() -> None:
    assert executor._backend() in {"seatbelt", "restricted"}


@pytest.mark.skipif(SEATBELT, reason="Seatbelt is the native backend on this Mac")
def test_seatbelt_is_rejected_off_macos(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MESH2CAD_SANDBOX", "seatbelt")
    result = executor.execute_code('r = cq.Workplane("XY").box(1, 1, 1)', tmp_path)
    assert result["status"] == "sandbox_error"
    assert result["mesh_path"] is None
    assert result["step_path"] is None


def test_execute_code_exports_real_step_and_mesh(tmp_path) -> None:
    pytest.importorskip("cadquery")
    result = executor.execute_code(
        'r = cq.Workplane("XY").box(1, 2, 3)',
        tmp_path,
        timeout=60,
    )
    assert result["status"] == "valid", result.get("error")
    assert result["sandbox"] in {"seatbelt", "restricted"}
    assert "ISO-10303-21" in Path(result["step_path"]).read_text(encoding="utf-8")
    mesh = trimesh.load_mesh(result["mesh_path"])
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(6.0, rel=1e-4)


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


def test_restricted_backend_exports_real_step(tmp_path, monkeypatch) -> None:
    pytest.importorskip("cadquery")
    monkeypatch.setenv("MESH2CAD_SANDBOX", "restricted")
    result = executor.execute_code(
        'r = cq.Workplane("XY").box(1, 1, 1)',
        tmp_path,
        timeout=60,
    )
    assert result["status"] == "valid", result.get("error")
    assert result["sandbox"] == "restricted"


def test_restricted_blocks_dangerous_imports(tmp_path, monkeypatch) -> None:
    pytest.importorskip("cadquery")
    monkeypatch.setenv("MESH2CAD_SANDBOX", "restricted")
    result = executor.execute_code(
        'import socket\nr = cq.Workplane("XY").box(1, 1, 1)',
        tmp_path,
        timeout=60,
    )
    assert result["status"] == "execution_error"
    assert "not allowed" in result["error"] or "blocked" in result["error"]


def test_restricted_blocks_reads_outside_run_directory(tmp_path, monkeypatch) -> None:
    pytest.importorskip("cadquery")
    monkeypatch.setenv("MESH2CAD_SANDBOX", "restricted")
    outside = (Path.cwd() / "README.md").resolve()
    code = f'r = cq.Workplane("XY").box(1, 1, 1)\nopen({str(outside)!r}).read()'
    result = executor.execute_code(code, tmp_path, timeout=60)
    assert result["status"] == "execution_error"
    assert "outside the run directory" in result["error"]


def test_restricted_allows_writes_inside_run_directory(tmp_path, monkeypatch) -> None:
    pytest.importorskip("cadquery")
    monkeypatch.setenv("MESH2CAD_SANDBOX", "restricted")
    result = executor.execute_code(
        'open("inside.txt", "w").write("ok")\nr = cq.Workplane("XY").box(1, 1, 1)',
        tmp_path,
        timeout=60,
    )
    assert result["status"] == "valid", result.get("error")
    assert (tmp_path / "inside.txt").read_text(encoding="utf-8") == "ok"
