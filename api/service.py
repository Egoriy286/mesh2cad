"""Durable job queue with a single independent worker and immutable candidates."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mesh2cad.executor import execute_code
from mesh2cad.geometry import load_mesh, render_comparison, render_views

from .backend import (
    AUTH_ERROR,
    CALL_RESERVE,
    SUPPORTED_MODEL,
    Cancelled,
    OpenRouterHTTPError,
    invoke,
    normalize_key,
    response_content,
    verify_key,
)
from .demo import demo_code
from .importer import import_mesh
from .storage import Store

TERMINAL = {"cancelled", "complete", "failed", "interrupted"}


def now():
    return datetime.now(timezone.utc).isoformat()


def ident():
    return uuid.uuid4().hex


class Service:
    def __init__(self, root, start_worker=True):
        self.store = Store(root)
        self.worker_lock = None
        if start_worker:
            self.worker_lock = (self.store.root / "worker.lock").open("a+")
            try:
                fcntl.flock(self.worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self.worker_lock.close()
                raise RuntimeError(
                    "Another workbench worker already owns this data directory"
                ) from exc
        self.key = normalize_key(os.environ.get("OPENROUTER_API_KEY", ""))
        self.key_status = "unverified" if self.key else "missing"
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = None
        self.heartbeat_thread = None
        self.active_job_id = None
        self.import_lock = threading.Lock()
        with self.store.transaction():
            if not self.store.get("settings", "global"):
                self.store.put(
                    "settings",
                    {
                        "id": "global",
                        "provider": "demo",
                        "model": SUPPORTED_MODEL,
                        "reasoning_effort": "medium",
                        "budget_limit_usd": 0.5,
                        "max_attempts": 1,
                    },
                )
                # Import existing ledger once. Never alter historical research data.
                ledger = (
                    Path(__file__).resolve().parents[1]
                    / "runs/openrouter-complex-20261020-01/budget.json"
                )
                if ledger.is_file():
                    data = json.loads(ledger.read_text())
                    known = data.get("known_complex_cost_usd", 0) + data.get(
                        "prior_mini_test_usd", 0
                    )
                    reserve = max(
                        0,
                        data.get("conservative_complex_spend_usd", 0)
                        - data.get("known_complex_cost_usd", 0),
                    )
                    self.store.put(
                        "calls",
                        {
                            "id": "historical-import",
                            "known_usd": known,
                            "reserved_usd": reserve,
                            "status": "historical_unknown" if reserve else "settled",
                        },
                    )
            for job in self.store.all("jobs"):
                # Migrate only the exact legacy pre-generation 401 failure.
                # Unknown network/provider charges and historical reserves stay intact.
                if job.get("status") == "failed" and job.get("error") == (
                    "OpenRouter HTTP 401; reservation retained pending reconciliation"
                ):
                    for call_id in job.get("reservations", []):
                        call = self.store.get("calls", call_id)
                        if (
                            call
                            and call.get("status") in {"in_flight", "unknown"}
                            and not call.get("known_usd", 0)
                        ):
                            call.update(status="rejected", reserved_usd=0, http_status=401)
                            self.store.put("calls", call)
                    job.update(error=AUTH_ERROR, updated_at=now())
                    self.store.put("jobs", job)
                if job["status"] in {"running", "cancelling"}:
                    job.update(
                        status="interrupted",
                        stage="interrupted",
                        error="Worker restarted. Request was not retried; any uncertain charge remains reserved.",
                        updated_at=now(),
                    )
                    self.store.put("jobs", job)
                    for call_id in job.get("reservations", []):
                        call = self.store.get("calls", call_id)
                        if call and call["status"] == "not_started":
                            call.update(status="unused", reserved_usd=0)
                            self.store.put("calls", call)
        if start_worker:
            self.thread = threading.Thread(target=self.worker, name="mesh2cad-worker", daemon=True)
            self.thread.start()
            self.heartbeat_thread = threading.Thread(
                target=self.heartbeat, name="mesh2cad-heartbeat", daemon=True
            )
            self.heartbeat_thread.start()

    def heartbeat(self):
        while not self.stop.wait(1):
            job_id = self.active_job_id
            if job_id:
                self.update_job(job_id, heartbeat_at=now())

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=5)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2)
        if self.worker_lock and (not self.thread or not self.thread.is_alive()):
            fcntl.flock(self.worker_lock, fcntl.LOCK_UN)
            self.worker_lock.close()

    def settings(self):
        result = self.store.get("settings", "global")
        result.pop("id", None)
        return {
            **result,
            "api_key_configured": bool(self.key),
            "api_key_status": self.key_status,
            "supported_providers": ["demo", "openrouter"],
            "supported_models": [SUPPORTED_MODEL],
            "call_reserve_usd": CALL_RESERVE,
        }

    def budget(self):
        calls = self.store.all("calls")
        known = sum(call.get("known_usd", 0) for call in calls)
        reserved = sum(call.get("reserved_usd", 0) for call in calls)
        limit = self.store.get("settings", "global")["budget_limit_usd"]
        return {
            "limit_usd": limit,
            "known_usd": known,
            "reserved_usd": reserved,
            "available_usd": max(0, limit - known - reserved),
            "unknown_calls": sum(
                call.get("reserved_usd", 0) > 0
                and call.get("status") in {"in_flight", "unknown", "historical_unknown"}
                for call in calls
            ),
        }

    def update_settings(self, values):
        values = dict(values)
        key = values.pop("api_key", None)
        if key is not None:
            key = normalize_key(key)
        if values.get("provider") == "codex":
            raise ValueError(
                "Codex is available through the existing audited CLI only; choose demo or OpenRouter"
            )
        with self.store.transaction():
            settings = self.store.get("settings", "global")
            settings.update(values)
            self.store.put("settings", settings)
            if key is not None and key != self.key:
                self.key, self.key_status = key, "unverified" if key else "missing"
        return {"settings": self.settings(), "budget": self.budget()}

    def check_key(self):
        key = self.key
        result = verify_key(key)
        with self.store.transaction():
            if self.key == key and not (
                self.key_status == "invalid" and result["status"] == "unverified"
            ):
                self.key_status = result["status"]
        return result

    def import_asset(self, data, name, example_id=None):
        if not self.import_lock.acquire(blocking=False):
            raise ValueError("Another mesh is importing; retry after it finishes")
        try:
            return self._import_asset(data, name, example_id)
        finally:
            self.import_lock.release()

    def _import_asset(self, data, name, example_id=None):
        asset_id = ident()
        folder = self.store.root / "assets" / asset_id
        folder.mkdir(parents=True)
        fmt = Path(name).suffix.lower().lstrip(".")
        source = folder / f"source.{fmt}"
        source.write_bytes(data)
        metadata = import_mesh(source, fmt, folder)
        mesh = load_mesh(folder / "target.stl")
        # Preview rendering is bounded separately from metric geometry.
        render_views(self.preview(mesh)[0], folder / "image.png")
        asset = {
            "id": asset_id,
            "name": Path(name).name[:200],
            "format": fmt,
            "created_at": now(),
            **metadata,
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "mesh_url": f"/api/assets/{asset_id}/mesh",
            "image_url": f"/api/assets/{asset_id}/image",
            "example_id": example_id,
        }
        with self.store.transaction():
            self.store.put("assets", asset)
        return asset

    def path(self, kind, item_id, filename):
        return self.store.root / kind / item_id / filename

    def artifact_path(self, kind, item_id, filename):
        path = self.path(kind, item_id, filename)
        expected = self.store.root / kind / item_id
        if (
            expected.is_symlink()
            or path.is_symlink()
            or not path.is_file()
            or path.resolve().parent != expected.resolve()
        ):
            raise ValueError("Artifact unavailable or unsafe")
        return path

    @staticmethod
    def preview(mesh, limit=30000):
        from .preview import decimated_preview

        return decimated_preview(mesh, limit)

    def submit(self, request, key=None):
        if key and len(key) > 200:
            raise ValueError("Idempotency key exceeds 200 characters")
        fingerprint = json.dumps(request, sort_keys=True)
        with self.store.transaction():
            if key:
                row = self.store.db.execute(
                    "SELECT request,job_id FROM idempotency WHERE key=?", (key,)
                ).fetchone()
                if row:
                    if row[0] != fingerprint:
                        raise ValueError("Idempotency key already belongs to a different request")
                    return self.store.get("jobs", row[1])
            asset = self.store.get("assets", request["asset_id"])
            if not asset:
                raise ValueError("Asset does not exist")
            parent = self.store.get("candidates", request.get("parent_id"))
            if request.get("parent_id") and (not parent or parent["asset_id"] != asset["id"]):
                raise ValueError("Selected parent does not belong to this asset")
            if request["kind"] in {"edit", "explain", "evaluate"} and not parent:
                raise ValueError("Select a candidate first")
            if request["kind"] == "evaluate" and not parent.get("mesh_url"):
                raise ValueError("Selected candidate has no valid geometry to evaluate")
            settings = self.settings()
            job = {**settings, **{k: v for k, v in request.items() if v is not None}}
            job.pop("api_key_configured", None)
            if job["provider"] == "codex":
                raise ValueError("Codex workbench jobs are unsupported; use the audited CLI")
            if job["provider"] == "openrouter" and job["model"] != SUPPORTED_MODEL:
                raise ValueError(f"Budget pricing currently supports only {SUPPORTED_MODEL}")
            paid = job["provider"] == "openrouter" and job["kind"] in {"predict", "edit", "explain"}
            if paid and not self.key:
                raise ValueError("Set an OpenRouter API key in server settings")
            if paid and self.key_status == "invalid":
                raise ValueError(AUTH_ERROR)
            if paid and self.budget()["available_usd"] + 1e-12 < CALL_RESERVE * job["max_attempts"]:
                raise ValueError(
                    "Insufficient budget for conservative reservations across all requested attempts"
                )
            job.update(
                id=ident(),
                status="queued",
                stage="queued",
                created_at=now(),
                updated_at=now(),
                elapsed_seconds=0,
                error=None,
                result_candidate_id=None,
                parent_id=request.get("parent_id"),
                reservations=[],
            )
            if paid:
                for attempt in range(job["max_attempts"]):
                    call_id = ident()
                    self.store.put(
                        "calls",
                        {
                            "id": call_id,
                            "job_id": job["id"],
                            "attempt": attempt,
                            "known_usd": 0,
                            "reserved_usd": CALL_RESERVE,
                            "status": "not_started",
                        },
                    )
                    job["reservations"].append(call_id)
            self.store.put("jobs", job)
            if key:
                self.store.db.execute(
                    "INSERT INTO idempotency VALUES(?,?,?)", (key, fingerprint, job["id"])
                )
            if job["kind"] in {"predict", "edit", "explain"}:
                self.store.put(
                    "messages",
                    {
                        "id": ident(),
                        "asset_id": asset["id"],
                        "candidate_id": job["parent_id"],
                        "role": "user",
                        "content": job.get("prompt") or job["kind"],
                        "created_at": now(),
                        "job_id": job["id"],
                        "related_candidate_id": None,
                    },
                )
        self.wake.set()
        return job

    def cancelled(self, job_id):
        job = self.store.get("jobs", job_id)
        return self.stop.is_set() or job["status"] in {"cancelling", "cancelled"}

    def update_job(self, job_id, **values):
        with self.store.transaction():
            job = self.store.get("jobs", job_id)
            if job["status"] in TERMINAL:
                return job
            if job["status"] == "cancelling":
                if values.get("status") in {"complete", "failed", "cancelled"}:
                    values.update(status="cancelled", stage="cancelled")
                else:
                    values["stage"] = "cancelling"
            job.update(values, updated_at=now())
            job["elapsed_seconds"] = max(
                0,
                (
                    datetime.now(timezone.utc) - datetime.fromisoformat(job["created_at"])
                ).total_seconds(),
            )
            self.store.put("jobs", job)
        return job

    def release_unused(self, job):
        with self.store.transaction():
            for call_id in job["reservations"]:
                call = self.store.get("calls", call_id)
                if call["status"] == "not_started":
                    call.update(reserved_usd=0, status="unused")
                    self.store.put("calls", call)

    def cancel(self, job_id):
        with self.store.transaction():
            # Re-read and transition in one transaction. A completed job must
            # never be put back into a nonterminal cancelling state.
            job = self.store.get("jobs", job_id)
            if not job:
                raise ValueError("Job not found")
            if job["status"] not in TERMINAL:
                queued = job["status"] == "queued"
                job.update(
                    status="cancelled" if queued else "cancelling",
                    stage="cancelled" if queued else "cancelling",
                    updated_at=now(),
                )
                self.store.put("jobs", job)
                for call_id in job["reservations"]:
                    call = self.store.get("calls", call_id)
                    if call["status"] == "not_started":
                        call.update(status="unused", reserved_usd=0)
                        self.store.put("calls", call)
        return job

    def worker(self):
        while not self.stop.is_set():
            job = None
            with self.store.transaction():
                for queued in self.store.all("jobs"):
                    if queued["status"] == "queued":
                        queued.update(status="running", stage="starting", updated_at=now())
                        self.store.put("jobs", queued)
                        job = queued
                        break
            if job:
                self.active_job_id = job["id"]
                try:
                    self.run(job)
                except Cancelled:
                    self.update_job(job["id"], status="cancelled", stage="cancelled")
                except Exception as exc:
                    # Never expose transport exceptions that could contain headers/keys.
                    error = str(exc)
                    if self.key:
                        error = error.replace(self.key, "[redacted]")
                    self.update_job(job["id"], status="failed", stage="failed", error=error[:4000])
                finally:
                    self.release_unused(job)
                    self.active_job_id = None
            else:
                self.wake.wait(0.25)
                self.wake.clear()

    def evaluate(self, job, target, prediction, output):
        command = [
            sys.executable,
            "-m",
            "api.evaluation_worker",
            str(target),
            str(prediction),
            str(output),
            str(job.get("n_points", 8192)),
            str(job.get("seed", 42)),
            str(job.get("threshold", 0.01)),
        ]
        with (output.parent / "evaluation.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=log,
                cwd=Path(__file__).resolve().parents[1],
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": str(output.parent),
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
            )
            deadline = time.monotonic() + 90
            try:
                while process.poll() is None:
                    if self.cancelled(job["id"]):
                        raise Cancelled()
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Evaluation exceeded 90 second limit")
                    time.sleep(0.05)
                if process.returncode:
                    raise RuntimeError("Evaluation subprocess failed; inspect local evaluation.log")
                return json.loads(output.read_text())
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()

    def run(self, job):
        job_id = job["id"]
        asset = self.store.get("assets", job["asset_id"])
        parent = self.store.get("candidates", job["parent_id"])
        target_path = self.path("assets", asset["id"], "target.stl")

        def check():
            if self.cancelled(job_id):
                raise Cancelled()

        attempts = job["max_attempts"] if job["kind"] in {"predict", "edit"} else 1
        candidate = None
        for attempt in range(attempts):
            check()
            if job["kind"] in {"predict", "edit", "explain"}:
                self.update_job(
                    job_id,
                    stage=f"{'offline demo' if job['provider'] == 'demo' else 'model response'} · attempt {attempt + 1}",
                )
                if job["provider"] == "demo":
                    if job["kind"] == "explain":
                        text = (
                            "Deterministic offline explanation (no model call). "
                            f"Selected candidate status: {parent['status']}. "
                            f"Measured geometry: {json.dumps(parent.get('metrics'))}. "
                            "Coordinates use the normalized target frame. Null metrics are unavailable. "
                            "This demo does not interpret the question; inspect the selected code and comparison."
                        )
                    else:
                        code = demo_code(asset, parent, job["kind"] == "edit")
                else:
                    api_key = self.key
                    if not api_key:
                        raise RuntimeError(
                            "OpenRouter credentials are unavailable after restart. Configure the server API key and submit a new job; no request was sent."
                        )
                    if self.key_status == "invalid":
                        raise ValueError(AUTH_ERROR)
                    call_id = job["reservations"][attempt]
                    with self.store.transaction():
                        # Mark before sending: recovery must never repeat an uncertain call.
                        call = self.store.get("calls", call_id)
                        if call["status"] != "not_started":
                            raise RuntimeError(
                                "Paid call was already started; automatic replay is disabled"
                            )
                        check()
                        call.update(status="in_flight", started_at=now())
                        self.store.put("calls", call)
                    image = (
                        self.artifact_path("candidates", parent["id"], "comparison.png")
                        if parent and parent.get("comparison_url")
                        else self.artifact_path("assets", asset["id"], "image.png")
                    )
                    try:
                        data, cost, usage = invoke(
                            api_key, job, asset, parent, image, lambda: self.cancelled(job_id)
                        )
                    except OpenRouterHTTPError as exc:
                        if exc.status_code == 401:
                            with self.store.transaction():
                                call.update(status="rejected", reserved_usd=0, http_status=401)
                                self.store.put("calls", call)
                                if self.key == api_key:
                                    self.key_status = "invalid"
                        raise
                    with self.store.transaction():
                        call.update(
                            status="settled" if cost is not None else "unknown",
                            known_usd=cost or 0,
                            reserved_usd=0 if cost is not None else CALL_RESERVE,
                            usage=usage,
                        )
                        self.store.put("calls", call)
                    output = response_content(data, job["kind"] == "explain")
                    if job["kind"] == "explain":
                        text = output
                    else:
                        code = output
                if job["kind"] == "explain":
                    check()
                    with self.store.transaction():
                        self.store.put(
                            "messages",
                            {
                                "id": ident(),
                                "asset_id": asset["id"],
                                "candidate_id": parent["id"],
                                "role": "assistant",
                                "content": text,
                                "created_at": now(),
                                "job_id": job_id,
                                "related_candidate_id": None,
                            },
                        )
                    self.update_job(job_id, status="complete", stage="complete")
                    return
            elif job["kind"] == "execute":
                code = job["code"]
            else:
                code = parent["code"]
            check()
            candidate_id = ident()
            folder = self.path("candidates", candidate_id, "")
            folder.mkdir(parents=True)
            (folder / "code.py").write_text(code)
            if job["kind"] == "evaluate":
                import shutil

                shutil.copyfile(
                    self.artifact_path("candidates", parent["id"], "candidate.stl"),
                    folder / "candidate.stl",
                )
                parent_step = self.path("candidates", parent["id"], "candidate.step")
                if parent_step.is_file():
                    shutil.copyfile(
                        self.artifact_path("candidates", parent["id"], "candidate.step"),
                        folder / "candidate.step",
                    )
                execution = {"status": "valid", "error": None}
            else:
                self.update_job(job_id, stage="sandbox execution")
                # Generated code can only write inside this child directory. Copy
                # validated exports into the trusted artifact directory afterwards.
                sandbox = folder / "sandbox"
                execution = execute_code(
                    code, sandbox, timeout=40, cancel_check=lambda: self.cancelled(job_id)
                )
                if execution["status"] == "valid":
                    import shutil

                    try:
                        for filename in ("candidate.stl", "candidate.step"):
                            source = sandbox / filename
                            if (
                                source.is_symlink()
                                or source.resolve().parent != sandbox.resolve()
                                or source.stat().st_size > 25 * 1024 * 1024
                            ):
                                raise ValueError("Sandbox export is unsafe or exceeds 25 MiB")
                        exported = load_mesh(sandbox / "candidate.stl")
                        if len(exported.faces) > 500000 or len(exported.vertices) > 1500000:
                            raise ValueError("Sandbox export exceeds mesh element limit")
                        for filename in ("candidate.stl", "candidate.step"):
                            shutil.copyfile(sandbox / filename, folder / filename)
                    except (OSError, ValueError) as exc:
                        execution = {"status": "execution_error", "error": str(exc)}
            check()
            metrics, error = None, execution.get("error")
            status = "valid" if execution["status"] == "valid" else "execution_error"
            if status == "valid":
                self.update_job(job_id, stage="evaluation")
                try:
                    metrics = self.evaluate(
                        job, target_path, folder / "candidate.stl", folder / "metrics.json"
                    )
                except Cancelled:
                    raise
                except Exception as exc:
                    status, error = (
                        "metric_error",
                        f"Evaluation failed: {type(exc).__name__}: {exc}",
                    )
                check()
                self.update_job(job_id, stage="comparison preview")
                render_comparison(
                    self.preview(load_mesh(target_path))[0],
                    self.preview(load_mesh(folder / "candidate.stl"))[0],
                    folder / "comparison.png",
                )
            check()
            valid_mesh = (
                execution["status"] == "valid"
                and not (folder / "candidate.stl").is_symlink()
                and (folder / "candidate.stl").is_file()
            )
            prefix = f"/api/candidates/{candidate_id}"
            candidate = {
                "id": candidate_id,
                "asset_id": asset["id"],
                "parent_id": parent["id"] if parent else None,
                "label": f"{'Offline demo' if job['provider'] == 'demo' and job['kind'] in {'predict', 'edit'} else job['kind'].capitalize()} · {attempt + 1}",
                "status": status,
                "created_at": now(),
                "code": code,
                "metrics": metrics,
                "error": error,
                "mesh_url": prefix + "/mesh" if valid_mesh else None,
                "step_url": prefix + "/step"
                if valid_mesh
                and not (folder / "candidate.step").is_symlink()
                and (folder / "candidate.step").is_file()
                else None,
                "comparison_url": prefix + "/comparison"
                if (folder / "comparison.png").is_file()
                else None,
                "job_id": job_id,
                "source": "deterministic_offline_demo"
                if job["provider"] == "demo" and job["kind"] in {"predict", "edit"}
                else job["kind"],
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
            with self.store.transaction():
                self.store.put("candidates", candidate)
            # Every configured attempt is an immutable geometric-feedback step.
            # The captured job parent stays fixed, while candidate lineage records
            # the preceding attempt used as context for this refinement.
            parent = candidate
            if status == "execution_error":
                job["repair_error"] = error or "Invalid CAD"
            else:
                job.pop("repair_error", None)
        check()
        if job["kind"] in {"predict", "edit"}:
            with self.store.transaction():
                self.store.put(
                    "messages",
                    {
                        "id": ident(),
                        "asset_id": asset["id"],
                        "candidate_id": job["parent_id"],
                        "role": "assistant",
                        "content": "Created a deterministic offline demo candidate; prompt interpretation is disabled."
                        if job["provider"] == "demo"
                        else "Created and evaluated a new candidate.",
                        "created_at": now(),
                        "job_id": job_id,
                        "related_candidate_id": candidate["id"],
                    },
                )
        self.update_job(
            job_id, status="complete", stage="complete", result_candidate_id=candidate["id"]
        )
