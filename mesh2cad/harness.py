"""A bounded repair loop with atomic checkpoints and an immutable run contract."""

from __future__ import annotations

import importlib.metadata
import math
import platform
import shutil
import subprocess
import time
from pathlib import Path

import trimesh

from . import __version__
from .backend import CodexBackend
from .dataset import load_manifest
from .executor import execute_code
from .geometry import evaluate_meshes, load_mesh, render_comparison, render_views
from .prompts import BASE_PROMPT, PROMPT_VERSION, initial_prompt, repair_prompt
from .storage import digest, file_digest, read_json, resolve_artifact, write_json


def implementation_digest() -> str:
    return digest(
        {path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    )


def _relative(root: Path, path: str | Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _valid(attempt: dict) -> bool:
    return attempt.get("execution_status") == "valid"


def choose_best(attempts: list[dict]) -> int | None:
    """A valid unscored CAD is better than failure; scored CADs rank by CD."""
    best, best_distance = None, math.inf
    for attempt in attempts:
        if not _valid(attempt):
            continue
        cd = (attempt.get("metrics") or {}).get("chamfer")
        if cd is not None and (not math.isfinite(cd) or cd < 0):
            cd = None
        if best is None or (cd is not None and cd < best_distance):
            best = attempt["index"]
            best_distance = cd if cd is not None else math.inf
    return best


def _export_selected(run_dir: Path, result: dict) -> None:
    item_dir = run_dir / "items" / result["id"]
    selections = {"baseline": 0, "best": result.get("best_index")}
    for name, index in selections.items():
        attempt = next((a for a in result["attempts"] if a["index"] == index), None)
        if attempt is None:
            continue
        for key, extension in (("code", "py"), ("mesh", "stl"), ("step", "step")):
            relative = attempt.get("artifacts", {}).get(key)
            if relative:
                source = resolve_artifact(run_dir, relative)
                if source.is_file():
                    shutil.copyfile(source, item_dir / f"{name}.{extension}")


def _save_result(run_dir: Path, result: dict) -> None:
    result["best_index"] = choose_best(result["attempts"])
    write_json(run_dir / "items" / result["id"] / "result.json", result)
    _export_selected(run_dir, result)


def _run_contract(
    manifest: dict,
    split: str,
    backend: CodexBackend,
    max_attempts: int,
    execution_timeout: float,
    n_points: int,
    seed: int,
) -> dict:
    dependencies = {}
    for name in (
        "cadquery",
        "cadquery-ocp",
        "trimesh",
        "numpy",
        "scipy",
        "manifold3d",
        "matplotlib",
    ):
        dependencies[name] = importlib.metadata.version(name)
    codex_version = "injected backend"
    if hasattr(backend, "executable"):
        version = subprocess.run(
            [backend.executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if version.returncode != 0:
            raise ValueError("Could not determine installed Codex CLI version")
        codex_version = version.stdout.strip()
    lockfile = Path(__file__).resolve().parent.parent / "uv.lock"
    return {
        "schema_version": 1,
        "package_version": __version__,
        "implementation_digest": implementation_digest(),
        "manifest_digest": digest(manifest),
        "split": split,
        "model": backend.model,
        "reasoning_effort": backend.reasoning_effort,
        "generation_timeout": backend.timeout,
        "execution_timeout": execution_timeout,
        "max_attempts": max_attempts,
        "n_points": n_points,
        "metric_seed": seed,
        "prompt_version": PROMPT_VERSION,
        "prompt_digest": digest(BASE_PROMPT),
        "dependencies": dependencies,
        "codex_version": codex_version,
        "lockfile_digest": file_digest(lockfile) if lockfile.is_file() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def run_smoke(backend: CodexBackend, output: Path, execution_timeout: float = 30) -> dict:
    """One live request validates model access, protocol and the sandbox/export path."""
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "smoke.json"
    smoke = read_json(checkpoint) if checkpoint.exists() else {"status": "running"}
    if smoke["status"] == "ok":
        return smoke
    target = trimesh.creation.box(extents=(1.0, 0.6, 0.4))
    image = output / "target.png"
    if not image.exists():
        render_views(target, image)
    try:
        generation_path = output / "generation" / "generation.json"
        generation = (
            read_json(generation_path) if generation_path.exists() else smoke.get("generation")
        )
        if generation is None or generation["status"] != "ok":
            smoke["generation"] = {
                "status": "interrupted",
                "usage": {},
                "launches": 1,
                "elapsed_seconds": 0,
                "error": "Preflight invocation did not finish checkpointing",
            }
            write_json(checkpoint, smoke)
            generation = backend.generate(initial_prompt(), image, output / "generation")
        smoke.update(
            status="generated" if generation["status"] == "ok" else generation["status"],
            generation=generation,
        )
        write_json(checkpoint, smoke)
        if generation["status"] == "ok":
            execution = smoke.get("execution")
            if execution is None or execution["status"] == "sandbox_error":
                execution = execute_code(
                    generation["code"], output / "execution", execution_timeout
                )
                smoke["execution"] = execution
            smoke["status"] = execution["status"]
            write_json(checkpoint, smoke)
            if execution["status"] == "valid":
                prediction = load_mesh(execution["mesh_path"])
                smoke["metrics"] = evaluate_meshes(target, prediction)
                render_comparison(target, prediction, output / "comparison.png")
                smoke["status"] = "ok"
    except KeyboardInterrupt:
        smoke["status"] = "interrupted"
        write_json(checkpoint, smoke)
        raise
    write_json(checkpoint, smoke)
    return smoke


def run_experiment(
    manifest_path: Path,
    run_dir: Path,
    *,
    split: str = "test",
    limit: int = 50,
    max_attempts: int = 3,
    execution_timeout: float = 30,
    n_points: int = 8192,
    seed: int = 42,
    resume: bool = False,
    backend: CodexBackend | None = None,
    preflight: bool = True,
    progress=print,
) -> dict:
    if split not in {"dev", "test"} or limit < 1 or not 1 <= max_attempts <= 3:
        raise ValueError("Use dev/test split, positive limit, and max-attempts between 1 and 3")
    if execution_timeout <= 0 or n_points < 1:
        raise ValueError("Timeout and n-points must be positive")
    backend = backend or CodexBackend()
    manifest_path, run_dir = manifest_path.resolve(), run_dir.resolve()
    manifest = load_manifest(manifest_path)
    available = [item for item in manifest["items"] if item["split"] == split]
    if limit > len(available):
        raise ValueError(f"Requested {limit} {split} items, manifest has {len(available)}")
    selected = available[:limit]
    contract = _run_contract(
        manifest, split, backend, max_attempts, execution_timeout, n_points, seed
    )
    config_path = run_dir / "config.json"
    if config_path.exists():
        if not resume:
            raise ValueError("Run already exists; use --resume or choose another --output")
        if read_json(config_path) != contract:
            raise ValueError(
                "Run configuration, prompts, implementation, inputs or dependencies changed; use a new output"
            )
        previous = read_json(run_dir / "selection.json")["items"]
        if len(selected) < len(previous) or selected[: len(previous)] != previous:
            raise ValueError("Resume may only keep or extend the original selection prefix")
    else:
        if any(run_dir.glob("*")):
            raise ValueError("Output directory is nonempty and has no run configuration")
        write_json(config_path, contract)
        write_json(run_dir / "dataset.json", manifest)
        (run_dir / "prompt.txt").write_text(BASE_PROMPT, encoding="utf-8")
    write_json(run_dir / "selection.json", {"items": selected})
    state = {"status": "running", "selected": len(selected), "completed": 0}
    write_json(run_dir / "state.json", state)

    if preflight:
        smoke_path = run_dir / "preflight" / "smoke.json"
        smoke = read_json(smoke_path) if smoke_path.exists() else None
        if smoke is None or smoke["status"] != "ok":
            # Preserve failed preflight invocations instead of overwriting their usage.
            smoke_root = run_dir / "preflight"
            invocations = sorted(smoke_root.glob("invocation-*"))
            latest = invocations[-1] if invocations else None
            latest_state = (
                read_json(latest / "smoke.json")
                if latest and (latest / "smoke.json").exists()
                else {}
            )
            recoverable = latest is not None and latest_state.get("status") in {
                None,
                "interrupted",
                "running",
                "valid",
                "sandbox_error",
                "generated",
            }
            destination = (
                latest if recoverable else smoke_root / f"invocation-{len(invocations):03d}"
            )
            smoke = run_smoke(backend, destination, execution_timeout)
            write_json(smoke_path, smoke)
        if smoke["status"] != "ok":
            state.update(status=smoke["status"], error="Preflight failed; see preflight/smoke.json")
            write_json(run_dir / "state.json", state)
            return state

    for item in selected:
        item_dir = run_dir / "items" / item["id"]
        result_path = item_dir / "result.json"
        result = (
            read_json(result_path)
            if result_path.exists()
            else {
                "id": item["id"],
                "split": split,
                "status": "pending",
                "attempts": [],
                "best_index": None,
            }
        )
        if result["status"] in {"complete", "input_error"}:
            state["completed"] += 1
            continue
        if item["status"] != "ready":
            result.update(status="input_error", error=item.get("error", "Input was not prepared"))
            _save_result(run_dir, result)
            state["completed"] += 1
            continue
        item_dir.mkdir(parents=True, exist_ok=True)
        target = load_mesh(resolve_artifact(manifest_path.parent, item["mesh_path"]))
        target_image = item_dir / "target.png"
        shutil.copyfile(resolve_artifact(manifest_path.parent, item["image_path"]), target_image)
        write_json(item_dir / "input.json", item)
        try:
            for index in range(max_attempts):
                existing = next((a for a in result["attempts"] if a["index"] == index), None)
                if existing and existing.get("finished"):
                    continue
                attempt = existing or {
                    "index": index,
                    "status": "pending",
                    "finished": False,
                    "generations": [],
                    "artifacts": {},
                    "metrics": None,
                    "execution_status": None,
                }
                if existing is None:
                    result["attempts"].append(attempt)
                attempt_dir = item_dir / "attempts" / f"{index:03d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                code_path = attempt_dir / "code.py"
                if not code_path.exists():
                    if index == 0:
                        prompt, image = initial_prompt(), target_image
                    else:
                        previous = result["attempts"][index - 1]
                        best = choose_best(result["attempts"][:index])
                        feedback = previous
                        if previous.get("execution_status") == "valid" and best is not None:
                            feedback = next(a for a in result["attempts"] if a["index"] == best)
                        previous_code = ""
                        if feedback.get("artifacts", {}).get("code"):
                            previous_code = resolve_artifact(
                                run_dir, feedback["artifacts"]["code"]
                            ).read_text()
                        prompt = repair_prompt(
                            previous_code,
                            error=feedback.get("error")
                            or (
                                None
                                if feedback.get("metrics")
                                else "No valid CAD program was produced."
                            ),
                            metrics=feedback.get("metrics"),
                        )
                        comparison = feedback.get("artifacts", {}).get("comparison")
                        image = (
                            resolve_artifact(run_dir, comparison) if comparison else target_image
                        )
                    generation = attempt["generations"][-1] if attempt["generations"] else None
                    if generation and generation["status"] == "interrupted":
                        saved = (
                            attempt_dir
                            / f"generation-{len(attempt['generations']) - 1:03d}"
                            / "generation.json"
                        )
                        if saved.exists():
                            generation = read_json(saved)
                            attempt["generations"][-1] = generation
                    if generation is None or generation["status"] != "ok":
                        invocation = len(attempt["generations"])
                        attempt["generations"].append(
                            {
                                "status": "interrupted",
                                "code": None,
                                "usage": {},
                                "launches": 1,
                                "elapsed_seconds": 0,
                                "error": "Invocation did not finish checkpointing",
                            }
                        )
                        _save_result(run_dir, result)
                        generation = backend.generate(
                            prompt, image, attempt_dir / f"generation-{invocation:03d}"
                        )
                        attempt["generations"][-1] = generation
                    attempt.update(status=generation["status"], error=generation.get("error"))
                    _save_result(run_dir, result)
                    if generation["status"] != "ok":
                        # Infrastructure/auth/protocol failures halt the run, avoiding a
                        # whole dataset of failures caused by one unavailable backend.
                        result["status"] = generation["status"]
                        _save_result(run_dir, result)
                        state.update(
                            status=generation["status"],
                            current_id=item["id"],
                            error=generation.get("error"),
                        )
                        write_json(run_dir / "state.json", state)
                        return state
                    code_path.write_text(generation["code"], encoding="utf-8")
                    attempt["artifacts"]["code"] = _relative(run_dir, code_path)
                    attempt["status"] = "generated"
                    _save_result(run_dir, result)
                code = code_path.read_text(encoding="utf-8")
                attempt["artifacts"]["code"] = _relative(run_dir, code_path)
                if attempt.get("execution", {}).get("status") == "sandbox_error":
                    # A sandbox startup failure is retryable after permissions are
                    # restored, without paying for the already generated code.
                    del attempt["execution"]
                if "execution" not in attempt:
                    execution = execute_code(code, attempt_dir / "execution", execution_timeout)
                    attempt["execution"] = execution
                    attempt["execution_status"] = execution["status"]
                    attempt["execution_seconds"] = execution["elapsed_seconds"]
                    attempt["error"] = execution.get("error")
                    _save_result(run_dir, result)
                execution = attempt["execution"]
                attempt["status"] = execution["status"]
                if execution["status"] == "sandbox_error":
                    result["status"] = "sandbox_error"
                    _save_result(run_dir, result)
                    state.update(
                        status="sandbox_error", current_id=item["id"], error=execution.get("error")
                    )
                    write_json(run_dir / "state.json", state)
                    return state
                if execution["status"] == "valid":
                    attempt["artifacts"].update(
                        mesh=_relative(run_dir, execution["mesh_path"]),
                        step=_relative(run_dir, execution["step_path"]),
                    )
                    try:
                        prediction = load_mesh(execution["mesh_path"])
                        attempt["metrics"] = evaluate_meshes(target, prediction, n_points, seed)
                        comparison = attempt_dir / "comparison.png"
                        render_comparison(target, prediction, comparison)
                        attempt["artifacts"]["comparison"] = _relative(run_dir, comparison)
                    except Exception as exc:
                        attempt.update(status="metric_error", error=f"{type(exc).__name__}: {exc}")
                attempt["finished"] = True
                _save_result(run_dir, result)
                progress(f"{item['id']} attempt {index + 1}/{max_attempts}: {attempt['status']}")
            result["status"] = "complete"
            _save_result(run_dir, result)
            state["completed"] += 1
            write_json(run_dir / "state.json", state)
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            _save_result(run_dir, result)
            state.update(status="interrupted", current_id=item["id"])
            write_json(run_dir / "state.json", state)
            raise
    state["status"] = "complete"
    state["finished_at_unix"] = time.time()
    write_json(run_dir / "state.json", state)
    return state
