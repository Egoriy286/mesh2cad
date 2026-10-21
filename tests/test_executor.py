"""Real CadQuery worker tests; outer CI sandboxes need not permit Seatbelt."""

import shutil
import sys
import time
from pathlib import Path

import pytest

from mesh2cad import executor


def run_worker(tmp_path: Path, code: str, timeout: float = 30):
    pytest.importorskip("cadquery")
    (tmp_path / "code.py").write_text(code, encoding="utf-8")
    worker = tmp_path / "_worker.py"
    shutil.copyfile(Path(executor.__file__).with_name("_worker.py"), worker)
    return executor._run_process([sys.executable, "-I", "-B", str(worker)], tmp_path, timeout)


def test_cube_exports_real_step_and_mesh(tmp_path):
    returncode, timed_out = run_worker(tmp_path, 'r = cq.Workplane("XY").box(1, 2, 3)')
    assert (returncode, timed_out) == (0, False)
    assert "ISO-10303-21" in (tmp_path / "candidate.step").read_text()
    import trimesh

    mesh = trimesh.load_mesh(tmp_path / "candidate.stl")
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(6)


@pytest.mark.parametrize(
    "code, message",
    [
        ('raise ValueError("broken CAD")', "broken CAD"),
        ('r = cq.Workplane("XY")', "at least one CadQuery shape"),
        ("r = None", "cadquery.Workplane"),
        ('r = cq.Workplane("XY").circle(1)', "nonempty, valid solid"),
    ],
)
def test_worker_reports_errors(tmp_path, code, message):
    returncode, timed_out = run_worker(tmp_path, code)
    assert (returncode, timed_out) == (1, False)
    assert message in (tmp_path / "traceback.txt").read_text()
    assert not (tmp_path / "candidate.step").exists()


def test_worker_timeout(tmp_path):
    returncode, timed_out = run_worker(tmp_path, "while True: pass", timeout=0.1)
    assert timed_out
    assert returncode != 0


def test_missing_sandbox_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(executor.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        executor, "_run_process", lambda *args: pytest.fail("Must not execute without sandbox")
    )
    result = executor.execute_code('r = cq.Workplane("XY").box(1, 1, 1)', tmp_path)
    assert result["status"] == "sandbox_error"
    assert result["mesh_path"] is None
    assert result["step_path"] is None
    assert (tmp_path / "code.py").exists()


def test_environment_excludes_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "/private/credentials")
    env = executor._environment(tmp_path)
    assert "OPENAI_API_KEY" not in env
    assert "PYTHONPATH" not in env
    assert env["HOME"] == str(tmp_path / "home")


def test_profile_does_not_allow_repository(tmp_path):
    profile = executor._sandbox_profile(tmp_path)
    assert "(deny default)" in profile
    assert "(allow network" not in profile
    assert '(subpath "/")' not in profile
    assert f'(subpath "{Path.cwd()}")' not in profile


def test_profile_allows_lexical_and_resolved_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(executor.sys, "executable", str(tmp_path / "venv" / "bin" / "python"))
    interpreter = tmp_path / "real-python"
    interpreter.touch()
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    Path(executor.sys.executable).symlink_to(interpreter)
    profile = executor._sandbox_profile(tmp_path)
    assert f'(literal "{executor.sys.executable}")' in profile
    assert f'(literal "{interpreter}")' in profile


def test_cancellation_stops_an_active_worker_without_waiting_for_timeout(tmp_path):
    started = time.monotonic()
    with pytest.raises(executor.ExecutionCancelled):
        executor._run_process(
            [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
            tmp_path,
            timeout=30,
            cancel_check=lambda: time.monotonic() - started > .25,
        )
    assert time.monotonic() - started < 3


def test_already_cancelled_proposal_does_not_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "_run_process", lambda *a, **kw: pytest.fail("Must not launch"))
    result = executor.execute_code("r = cq.Workplane('XY').box(1, 1, 1)", tmp_path,
                                   cancel_check=lambda: True)
    assert result["status"] == "cancelled"
    assert result["mesh_path"] is None
