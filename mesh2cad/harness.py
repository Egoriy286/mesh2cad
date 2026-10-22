"""Durable, bounded agent orchestration over real STL/CadQuery evaluations.

The worker has no Streamlit dependency. JSON checkpoints are the source of
truth; an OS file lock prevents GUI/CLI processes from evaluating the same run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .agent import AgentContext, CadAgent, OpenRouterAgent
from .dataset import load_manifest, read_catalogue
from .executor import execute_code
from .geometry import (
    compose_views,
    evaluate_iou,
    load_mesh,
    mesh_difference,
    mesh_stats,
    normalize_mesh,
    render_llm_images,
    render_overlay_sheet,
)
from .observability import configure_logging, log_event, register_secret
from .pipeline import (
    DEFAULT_MODEL,
    REASONING_EFFORTS,
    PipelineError,
    api_key_from_environment,
    extract_code,
    normalize_key,
    payload_digest,
)
from .prompts import DEFAULT_REQUEST, PROMPT_VERSION, SYSTEM_PROMPT
from .resources import DEFAULT_MEMORY_MB, kill_tree, monitor_process, worker_environment
from .storage import file_digest, read_json, resolve_artifact, write_json

TERMINAL_ITEMS = {"success", "exhausted", "stalled", "failed", "unscorable"}
FINISHED_ATTEMPTS = {
    "evaluated",
    "execution_error",
    "invalid_response",
    "api_error",
    "metric_error",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class HarnessConfig:
    model: str = DEFAULT_MODEL
    reasoning_effort: str = "medium"
    system_prompt: str = SYSTEM_PROMPT
    request: str = DEFAULT_REQUEST
    max_iterations: int = 5
    target_iou: float = 0.9
    patience: int = 3
    min_improvement: float = 0.001
    max_tokens: int = 4096
    total_token_budget: int = 100_000
    max_wall_seconds: float = 1800
    execution_timeout: float = 60
    request_timeout: float = 180
    max_retries: int = 2
    retry_backoff: float = 1
    memory_limit_mb: int = DEFAULT_MEMORY_MB

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.system_prompt.strip():
            raise ValueError("Model and system prompt must not be empty")
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("Unsupported reasoning effort")
        for name in ("max_iterations", "patience", "max_tokens", "total_token_budget"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool):
            raise ValueError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        for name in ("max_wall_seconds", "execution_timeout", "request_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("target_iou", "min_improvement", "retry_backoff"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.target_iou > 1 or self.min_improvement > 1:
            raise ValueError("IoU and minimum improvement must be at most 1")
        if (
            not isinstance(self.memory_limit_mb, int)
            or isinstance(self.memory_limit_mb, bool)
            or not 256 <= self.memory_limit_mb <= 8192
        ):
            raise ValueError("memory_limit_mb must be an integer between 256 and 8192")


class RunLock:
    """A process-safe lock automatically released if the worker dies."""

    def __init__(self, root: Path, filename: str = ".worker.lock") -> None:
        self.handle = (root / filename).open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                if not self.handle.read(1):
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError("This run already has an active worker") from exc

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def is_active(root: Path) -> bool:
    try:
        lease = RunLock(root)
    except RuntimeError:
        return True
    lease.close()
    return False


def _number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return max(0, value)
    return None


def summarize(state: dict) -> dict:
    items = state["items"]
    attempts = [attempt for item in items for attempt in item["attempts"]]
    calls = [call for attempt in attempts for call in attempt.get("calls", [])]
    usages = [call["usage"] for call in calls if isinstance(call.get("usage"), dict)]
    costs = [value for usage in usages if (value := _number(usage.get("cost"))) is not None]
    scores = [
        item["attempts"][item["best_attempt"]]["iou"]["iou"]
        for item in items
        if item["best_attempt"] is not None
    ]
    prompt_tokens = sum(_number(u.get("prompt_tokens")) or 0 for u in usages)
    completion_tokens = sum(_number(u.get("completion_tokens")) or 0 for u in usages)
    total_tokens = sum(
        _number(u.get("total_tokens"))
        or ((_number(u.get("prompt_tokens")) or 0) + (_number(u.get("completion_tokens")) or 0))
        for u in usages
    )
    return {
        "total": len(items),
        "finished": sum(item["status"] in TERMINAL_ITEMS for item in items),
        "successes": sum(item["status"] == "success" for item in items),
        "success_rate": sum(item["status"] == "success" for item in items) / len(items),
        "evaluated_items": len(scores),
        "mean_iou": sum(scores) / len(scores) if scores else None,
        # This second mean includes unscored/failed inputs as 0, avoiding
        # survivor bias while retaining None for unavailable individual scores.
        "mean_iou_all": sum(scores) / len(items),
        "attempts": len(attempts),
        "valid_attempts": sum(a.get("execution", {}).get("status") == "valid" for a in attempts),
        "llm_calls": len(calls),
        "responses": sum(call.get("status") == "received" for call in calls),
        "usage_missing": sum(
            call.get("status") == "received" and not isinstance(call.get("usage"), dict)
            for call in calls
        ),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "cost_usd": sum(costs) if costs else None,
        "cost_missing": sum(
            call.get("status") == "received"
            and (
                not isinstance(call.get("usage"), dict)
                or _number(call["usage"].get("cost")) is None
            )
            for call in calls
        ),
        "llm_seconds": sum(call.get("elapsed_seconds", 0) for call in calls),
        "execution_seconds": sum(
            execution.get("elapsed_seconds", 0)
            for attempt in attempts
            for execution in attempt.get("executions", [attempt.get("execution", {})])
        ),
    }


def results_csv(state: dict) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id",
            "split",
            "source_file",
            "name",
            "category",
            "source",
            "status",
            "attempts",
            "best_iou",
            "reason",
        ],
    )
    writer.writeheader()
    for item in state["items"]:
        best = item["best_attempt"]
        writer.writerow(
            {
                "id": item["id"],
                "split": item["split"],
                "source_file": item["source_file"],
                "name": item.get("name", item["source_file"]),
                "category": item.get("category"),
                "source": item.get("source"),
                "status": item["status"],
                "attempts": len(item["attempts"]),
                "best_iou": item["attempts"][best]["iou"]["iou"] if best is not None else "",
                "reason": item.get("stop_reason", ""),
            }
        )
    return output.getvalue()


def create_run(
    manifest_path: Path,
    runs_dir: Path,
    config: HarnessConfig,
    *,
    split: str | None = None,
    item_ids: list[str] | None = None,
    limit: int | None = None,
) -> Path:
    if split not in {None, "dev", "test", "all"}:
        raise ValueError("Split must be dev, test or all")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("Limit must be a positive integer")
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path, verify_images=False)
    if split is None:
        split = "dev" if any(i["split"] == "dev" for i in manifest["items"]) else "all"
    selected = [
        item
        for item in manifest["items"]
        if item["status"] == "ready"
        and (split == "all" or item["split"] == split)
        and (item_ids is None or item["id"] in item_ids)
    ]
    if item_ids is not None and set(item_ids) != {item["id"] for item in selected}:
        raise ValueError("Some selected IDs are missing, unprepared or belong to another split")
    selected = selected[:limit]
    if not selected:
        raise ValueError("No ready STL inputs selected")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = runs_dir.resolve() / f"{stamp}-{uuid.uuid4().hex[:8]}"
    (root / "inputs").mkdir(parents=True)
    items = []
    for item in selected:
        data = resolve_artifact(manifest_path.parent, item["mesh_path"]).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != item["mesh_sha256"]:
            raise ValueError(f"Input changed while creating run: {item['id']}")
        relative = f"inputs/{item['id']}.stl"
        (root / relative).write_bytes(data)
        items.append(
            {
                "id": item["id"],
                "split": item["split"],
                "source_file": item["source_file"],
                "input_path": relative,
                "name": item.get("name", item["source_file"]),
                "category": item.get("category"),
                "source": item.get("source"),
                "input_sha256": sha,
                "status": "pending",
                "attempts": [],
                "best_attempt": None,
                "stop_reason": None,
            }
        )
    write_json(root / "manifest.json", manifest)
    state = {
        "schema_version": 1,
        "id": root.name,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "pending",
        "stop_reason": None,
        "config": asdict(config),
        "prompt_version": PROMPT_VERSION,
        "dataset": {
            "name": manifest["name"],
            "format": manifest["format"],
            "manifest": str(manifest_path),
            "sha256": file_digest(manifest_path),
            "source": manifest.get("source"),
            "split": split,
        },
        "active_seconds": 0.0,
        "items": items,
    }
    state["summary"] = summarize(state)
    write_json(root / "run.json", state)
    return root


class _Stopped(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class InfrastructureError(RuntimeError):
    """A broken execution environment should stop the batch immediately."""


class Orchestrator:
    def __init__(self, root: Path, api_key: str, *, agent: CadAgent | None = None) -> None:
        self.root = root.resolve()
        self.state = read_json(self.root / "run.json")
        if self.state.get("schema_version") != 1:
            raise ValueError("Unsupported run schema")
        self.config = HarnessConfig(**self.state["config"])
        self.key = normalize_key(api_key)
        configure_logging()
        register_secret(self.key)
        if not self.key:
            raise ValueError("OpenRouter API key is required")
        self.agent = agent or OpenRouterAgent(
            self.config.model,
            system_prompt=self.config.system_prompt,
            request=self.config.request,
            reasoning_effort=self.config.reasoning_effort,
        )
        self.started = time.monotonic()
        self.previous_seconds = self.state["active_seconds"]

    def _safe_error(self, exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}".replace(self.key, "[REDACTED]")[-16000:]

    def _save(self) -> None:
        self.state["updated_at"] = utc_now()
        self.state["active_seconds"] = self.previous_seconds + time.monotonic() - self.started
        self.state["summary"] = summarize(self.state)
        write_json(self.root / "run.json", self.state)

    def _event(self, event: str, *, level: str = "INFO", item_id=None, **details) -> None:
        log_event(
            event,
            level=getattr(logging, level, logging.INFO),
            run=self.root.name,
            item_id=item_id,
            **details,
        )
        entry = {"time": utc_now(), "level": level, "event": event, "item_id": item_id, **details}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

    @contextmanager
    def _stage(self, name: str, item_id: str, iteration=None):
        started = time.monotonic()
        self.state["current"] = {
            "stage": name,
            "item_id": item_id,
            "iteration": iteration,
            "started_at": utc_now(),
        }
        self._save()
        self._event("stage_started", item_id=item_id, stage=name, iteration=iteration)
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            durations = self.state.setdefault("stage_seconds", {})
            durations[name] = durations.get(name, 0) + elapsed
            self._event(
                "stage_finished",
                item_id=item_id,
                stage=name,
                iteration=iteration,
                elapsed_seconds=round(elapsed, 3),
            )
            self._save()

    def _remaining(self) -> float:
        return (
            self.config.max_wall_seconds - self.previous_seconds - time.monotonic() + self.started
        )

    def _stopped(self) -> str | None:
        if (self.root / "cancel.requested").exists():
            return "cancelled"
        if self._remaining() <= 0:
            return "wall_time_budget"
        return None

    def _check(self, *, before_call: bool = False) -> None:
        if reason := self._stopped():
            raise _Stopped(reason)
        if before_call and summarize(self.state)["total_tokens"] >= self.config.total_token_budget:
            raise _Stopped("token_budget")

    def _request(self, attempt: dict, folder: Path, payload: dict, item_id: str) -> dict:
        for retry in range(self.config.max_retries + 1):
            self._check(before_call=True)
            call = {"started_at": utc_now(), "status": "requesting", "retry": retry}
            attempt.setdefault("calls", []).append(call)
            self._save()
            self._event("llm_request", item_id=item_id, iteration=attempt["iteration"], retry=retry)
            started = time.monotonic()
            try:
                with self._stage("llm", item_id, attempt["iteration"]):
                    response = self.agent.complete(
                        payload,
                        self.key,
                        timeout=min(self.config.request_timeout, self._remaining()),
                    )
                # Exact provider output, including malformed code, is durable
                # before parsing. A resumed evaluation can reuse this response.
                write_json(folder / "response.json", response)
                call.update(
                    status="received",
                    usage=response.get("usage"),
                    elapsed_seconds=time.monotonic() - started,
                )
                attempt["status"] = "response_received"
                self._save()
                self._event(
                    "llm_response",
                    item_id=item_id,
                    iteration=attempt["iteration"],
                    usage=response.get("usage"),
                    elapsed_seconds=call["elapsed_seconds"],
                )
                return response
            except PipelineError as exc:
                call.update(
                    status="error",
                    error=self._safe_error(exc),
                    elapsed_seconds=time.monotonic() - started,
                )
                self._save()
                self._event(
                    "llm_error", level="WARNING", item_id=item_id, error=call["error"], retry=retry
                )
                if not exc.retryable or retry == self.config.max_retries:
                    raise
                until = time.monotonic() + self.config.retry_backoff * (2**retry)
                while time.monotonic() < until:
                    self._check()
                    time.sleep(max(0, min(0.1, until - time.monotonic())))
        raise AssertionError("Unreachable retry state")

    def _evaluate(self, item: dict, target, sheet: bytes, attempt: dict) -> None:
        cfg = self.config
        folder = self.root / attempt["path"]
        folder.mkdir(parents=True, exist_ok=True)
        best = item["attempts"][item["best_attempt"]] if item["best_attempt"] is not None else None
        overlay_path = self.root / best["path"] / "overlay.png" if best else None
        context = AgentContext(
            extents=mesh_stats(target)["extents"],
            image=sheet,
            attempts=[a for a in item["attempts"] if a["status"] in FINISHED_ATTEMPTS],
            best_attempt=best,
            overlay=overlay_path.read_bytes() if overlay_path and overlay_path.is_file() else None,
            max_tokens=min(
                cfg.max_tokens,
                max(1, cfg.total_token_budget - summarize(self.state)["total_tokens"]),
            ),
        )
        if (folder / "response.json").is_file():
            response = read_json(folder / "response.json")
            if attempt.get("calls") and attempt["calls"][-1]["status"] == "requesting":
                # Recover accounting when the process died between persisting
                # the provider response and committing the run checkpoint.
                attempt["calls"][-1].update(status="received", usage=response.get("usage"))
            self._event("response_reused", item_id=item["id"], iteration=attempt["iteration"])
        else:
            payload = self.agent.build_request(context)
            write_json(folder / "request.json", payload)
            attempt["request_sha256"] = payload_digest(payload)
            attempt["status"] = "requesting"
            self._save()
            try:
                response = self._request(attempt, folder, payload, item["id"])
            except PipelineError as exc:
                attempt.update(status="api_error", error=self._safe_error(exc))
                # Auth/configuration/provider exhaustion should stop the batch,
                # rather than repeat the same broken API call on every STL.
                item.update(status="pending", stop_reason="api_error")
                self._save()
                raise
        self._check()
        try:
            attempt["code"] = extract_code(response)
        except PipelineError as exc:
            attempt.update(status="invalid_response", error=self._safe_error(exc))
            choices = response.get("choices") or [{}]
            if choices[0].get("finish_reason") == "length":
                attempt["error"] = (
                    "Output token limit reached; lower reasoning effort or raise max_tokens."
                )
                item.update(status="failed", stop_reason="output_token_limit")
            return
        (folder / "code.py").write_text(attempt["code"] + "\n", encoding="utf-8")
        execution_path = folder / "execution.json"
        execution = read_json(execution_path) if execution_path.is_file() else {}
        if execution.get("status") != "valid" or not all(
            Path(execution.get(key) or "").is_file() for key in ("mesh_path", "step_path")
        ):
            attempt["status"] = "executing"
            self._save()
            self._event("execution_started", item_id=item["id"], iteration=attempt["iteration"])
            with self._stage("cadquery", item["id"], attempt["iteration"]):
                execution = execute_code(
                    attempt["code"],
                    folder / "cadquery",
                    timeout=min(cfg.execution_timeout, max(0.01, self._remaining())),
                    cancel_check=lambda: self._stopped() is not None,
                )
            write_json(execution_path, execution)
        if execution != attempt.get("execution"):
            attempt.setdefault("executions", []).append(execution)
        attempt["execution"] = execution
        self._save()
        self._event(
            "execution_finished",
            item_id=item["id"],
            iteration=attempt["iteration"],
            status=execution["status"],
            elapsed_seconds=execution["elapsed_seconds"],
        )
        if execution["status"] == "cancelled":
            attempt["status"] = "response_received"
            self._check()
        if execution["status"] == "sandbox_error":
            raise InfrastructureError(execution.get("error", "Sandbox unavailable"))
        if execution["status"] != "valid":
            attempt.update(status="execution_error", error=execution.get("error"))
            return
        self._check()
        attempt["status"] = "evaluating"
        with self._stage("iou", item["id"], attempt["iteration"]):
            candidate = load_mesh(execution["mesh_path"])
            iou = evaluate_iou(target, candidate)
        attempt.update(status="evaluated" if iou["iou"] is not None else "metric_error", iou=iou)
        write_json(folder / "iou.json", iou)
        self._save()
        if iou["iou"] is None:
            attempt["error"] = f"IoU unavailable: {iou['status']}"
            if iou["status"] in {"non_watertight_prediction", "invalid_volume_prediction"}:
                attempt["status"] = "execution_error"
            return
        try:
            attempt["status"] = "rendering_overlay"
            with self._stage("surface_distance", item["id"], attempt["iteration"]):
                difference = mesh_difference(
                    target, candidate, cancel_check=lambda: self._stopped() is not None
                )
            summary = {
                key: value
                for key, value in difference.items()
                if key
                not in {
                    "covered_vertices",
                    "covered_triangles",
                    "missing_vertices",
                    "missing_triangles",
                }
            }
            attempt["difference"] = summary
            write_json(folder / "difference.json", summary)
            with self._stage("overlay", item["id"], attempt["iteration"]):
                (folder / "overlay.png").write_bytes(
                    render_overlay_sheet(
                        difference, candidate, cancel_check=lambda: self._stopped() is not None
                    )
                )
        except Exception as exc:
            self._check()
            attempt["overlay_error"] = self._safe_error(exc)
            self._event(
                "overlay_error", level="WARNING", item_id=item["id"], error=attempt["overlay_error"]
            )
        attempt["status"] = "evaluated"

    def _item(self, item: dict) -> None:
        self._check()
        item["status"] = "running"
        self._save()
        self._event("item_started", item_id=item["id"])
        source = resolve_artifact(self.root, item["input_path"])
        if file_digest(source) != item["input_sha256"]:
            raise ValueError("STL snapshot checksum mismatch")
        with self._stage("normalize", item["id"]):
            target, normalization = normalize_mesh(load_mesh(source))
        item["target"] = mesh_stats(target)
        if not target.is_volume:
            item.update(status="unscorable", stop_reason="target_is_not_a_closed_volume")
            return
        prepared = self.root / "items" / item["id"]
        prepared.mkdir(parents=True, exist_ok=True)
        write_json(prepared / "normalization.json", normalization)
        target.export(prepared / "target.stl")
        with self._stage("render_target", item["id"]):
            if (prepared / "target.png").is_file():
                sheet = (prepared / "target.png").read_bytes()
                self._event("target_render_reused", item_id=item["id"])
            else:
                sheet = compose_views(
                    render_llm_images(target, cancel_check=lambda: self._stopped() is not None)
                )
                (prepared / "target.png").write_bytes(sheet)
        self._check()
        while True:
            last = item["attempts"][-1] if item["attempts"] else None
            if last and last["status"] == "api_error":
                # HTTP failures are infrastructure failures, not failed CAD
                # proposals. Resuming with a fixed key retries this same slot.
                last["status"] = "pending"
                last.pop("error", None)
            if last is None or last["status"] in FINISHED_ATTEMPTS:
                if len(item["attempts"]) >= self.config.max_iterations:
                    item.update(status="exhausted", stop_reason="max_iterations")
                    break
                self._check(before_call=True)
                iteration = len(item["attempts"]) + 1
                last = {
                    "iteration": iteration,
                    "status": "pending",
                    "calls": [],
                    "path": f"items/{item['id']}/attempt-{iteration:03d}",
                }
                item["attempts"].append(last)
                self._save()
            self._check()
            started = time.monotonic()
            self._evaluate(item, target, sheet, last)
            last["elapsed_seconds"] = last.get("elapsed_seconds", 0) + time.monotonic() - started
            self._update_best(item, last)
            self._save()
            write_json(self.root / last["path"] / "attempt.json", last)
            self._event(
                "attempt_finished",
                item_id=item["id"],
                iteration=last["iteration"],
                status=last["status"],
                iou=last.get("iou"),
                error=last.get("error"),
            )
            if item["status"] in TERMINAL_ITEMS:
                break
        self._event(
            "item_finished", item_id=item["id"], status=item["status"], reason=item["stop_reason"]
        )

    def _update_best(self, item: dict, attempt: dict) -> None:
        if item["status"] == "failed":
            return
        score = (attempt.get("iou") or {}).get("iou")
        best_index = item["best_attempt"]
        previous = item["attempts"][best_index]["iou"]["iou"] if best_index is not None else None
        if score is not None and (previous is None or score > previous):
            item["best_attempt"] = attempt["iteration"] - 1
        if score is not None and score >= self.config.target_iou:
            item.update(status="success", stop_reason="target_iou")
        elif attempt["status"] == "metric_error":
            item.update(status="failed", stop_reason="metric_unavailable")
        else:
            significant = score is not None and (
                previous is None
                or (score > previous and score - previous >= self.config.min_improvement)
            )
            item["stale_iterations"] = 0 if significant else item.get("stale_iterations", 0) + 1
            if item["stale_iterations"] >= self.config.patience:
                item.update(status="stalled", stop_reason="no_improvement")

    def run(self, *, _lease: RunLock | None = None) -> dict:
        lease = _lease or RunLock(self.root)
        engine_lease = None
        try:
            # Read again after acquiring ownership: another process may have
            # completed the run since this Orchestrator object was constructed.
            self.state = read_json(self.root / "run.json")
            self.previous_seconds = self.state["active_seconds"]
            self.started = time.monotonic()
            if self.state["status"] == "completed":
                return self.state
            if _lease is None:
                (self.root / "cancel.requested").unlink(missing_ok=True)
            while engine_lease is None:
                try:
                    engine_lease = RunLock(self.root.parent, ".engine.lock")
                except RuntimeError:
                    if (self.root / "cancel.requested").exists():
                        self.started = time.monotonic()
                        raise _Stopped("cancelled")
                    if self.state["status"] != "queued":
                        self.state.update(
                            status="queued", current={"stage": "queued", "started_at": utc_now()}
                        )
                        write_json(self.root / "run.json", self.state)
                        self._event("run_queued")
                    time.sleep(0.2)
            self.started = time.monotonic()
            self.state.update(status="running", stop_reason=None)
            self.state.pop("error", None)
            self._save()
            self._event("run_started", model=self.config.model)
            for item in self.state["items"]:
                if item["status"] in TERMINAL_ITEMS:
                    continue
                self._check()
                try:
                    self._item(item)
                except (PipelineError, InfrastructureError, _Stopped):
                    raise
                except Exception as exc:
                    item.update(
                        status="failed", stop_reason="item_error", error=self._safe_error(exc)
                    )
                    self._event(
                        "item_error", level="ERROR", item_id=item["id"], error=item["error"]
                    )
                self._save()
            self.state.update(status="completed", stop_reason="queue_finished")
        except _Stopped as exc:
            self.state.update(
                status="cancelled" if exc.reason == "cancelled" else "budget_exhausted",
                stop_reason=exc.reason,
            )
        except Exception as exc:
            self.state.update(
                status="failed", stop_reason="worker_error", error=self._safe_error(exc)
            )
            self._event("run_error", level="ERROR", error=self.state["error"])
        finally:
            try:
                self._save()
                (self.root / "results.csv").write_text(results_csv(self.state), encoding="utf-8")
                self._event(
                    "run_finished", status=self.state["status"], reason=self.state["stop_reason"]
                )
            finally:
                if engine_lease is not None:
                    engine_lease.close()
                lease.close()
        return self.state


def run_isolated(root: Path, api_key: str, *, _lease: RunLock | None = None) -> dict:
    """Run outside the GUI, with a resident-memory limit for the whole process tree."""
    worker = Orchestrator(root, api_key)  # Validate configuration/key before spawning.
    lease = _lease or RunLock(root)
    root = worker.root
    if _lease is None:
        (root / "cancel.requested").unlink(missing_ok=True)
    last_report = 0.0
    usage_path = root / "memory.json"
    usage = {"limit_mb": worker.config.memory_limit_mb, "peak_rss_mb": 0, "status": "starting"}
    process = None

    def report(rss, peak):
        nonlocal last_report
        usage.update(rss_mb=round(rss, 1), peak_rss_mb=round(peak, 1), status="running")
        if time.monotonic() - last_report >= 2:
            usage["updated_at"] = utc_now()
            write_json(usage_path, usage)
            log_event("worker.memory", run=root.name, **usage)
            last_report = time.monotonic()

    def stopped(reason: str, error: str) -> None:
        state = read_json(root / "run.json")
        state.update(
            status="cancelled" if reason == "cancelled" else "failed",
            stop_reason=reason,
            error=error,
            updated_at=utc_now(),
        )
        state["summary"] = summarize(state)
        write_json(root / "run.json", state)
        (root / "results.csv").write_text(results_csv(state), encoding="utf-8")
        log_event("run_stopped", level=logging.ERROR, run=root.name, reason=reason, error=error)
        with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"time": utc_now(), "event": "run_stopped", "reason": reason, "error": error},
                    ensure_ascii=False,
                )
                + "\n"
            )

    try:
        # The child imports trusted project code. Never inherit API credentials;
        # deliver only this run's key over its private stdin pipe.
        process = subprocess.Popen(
            [sys.executable, "-m", "mesh2cad.harness_worker", str(root), str(os.getpid())],
            cwd=Path(__file__).resolve().parents[1],
            env=worker_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            start_new_session=os.name != "nt",
        )
        usage["pid"] = process.pid
        try:
            process.stdin.write(worker.key + "\n")
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # The monitor below records a failed startup as worker_exit.
        log_event(
            "worker.started",
            run=root.name,
            pid=process.pid,
            memory_limit_mb=worker.config.memory_limit_mb,
        )
        result = monitor_process(
            process,
            memory_limit_mb=worker.config.memory_limit_mb,
            cancel_check=lambda: (root / "cancel.requested").exists(),
            cancel_grace=2,
            on_usage=report,
        )
        usage.update(rss_mb=0, peak_rss_mb=round(result.peak_rss_mb, 1), status="finished")
        if result.reason == "memory_limit":
            stopped(
                "memory_limit",
                f"Worker exceeded {worker.config.memory_limit_mb} MiB RSS "
                f"including CadQuery (observed {result.peak_rss_mb:.0f} MiB). "
                "Saved responses and CAD artifacts can be resumed.",
            )
        elif result.reason == "cancelled":
            stopped("cancelled", "Worker stopped by user; saved artifacts retained.")
        elif read_json(root / "run.json")["status"] in {"pending", "queued", "running"}:
            stopped("worker_exit", f"Worker exited unexpectedly with code {result.returncode}")
    except KeyboardInterrupt:
        stopped("cancelled", "Worker interrupted; saved artifacts retained.")
        raise
    except Exception as exc:
        stopped("worker_error", worker._safe_error(exc))
    finally:
        try:
            if process is not None and process.poll() is None:
                kill_tree(process)
            usage.update(status="finished", rss_mb=0, updated_at=utc_now())
            write_json(usage_path, usage)
        finally:
            lease.close()
    return read_json(root / "run.json")


def start_run(root: Path, api_key: str) -> None:
    """One monitor thread; the actual work runs in a disposable bounded process."""
    Orchestrator(root, api_key)
    lease = RunLock(root)
    (root / "cancel.requested").unlink(missing_ok=True)
    thread = threading.Thread(
        target=run_isolated,
        args=(root, api_key),
        kwargs={"_lease": lease},
        name=f"harness-monitor-{root.name}",
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        lease.close()
        raise


def cancel_run(root: Path) -> None:
    (root / "cancel.requested").touch()
    log_event("cancel.requested", run=root.name)


def launch_run(
    manifest_path: Path,
    runs_dir: Path,
    config: HarnessConfig,
    api_key: str,
    *,
    split: str,
    item_ids: list[str],
    limit: int,
) -> tuple[Path, bool]:
    """Atomically reuse an equivalent pending/active job instead of paying twice."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    catalogue = read_catalogue(manifest_path)
    selected = [
        i["id"]
        for i in catalogue["items"]
        if i["id"] in item_ids and (split == "all" or i["split"] == split)
    ][:limit]
    lease = RunLock(runs_dir, ".launch.lock")
    try:
        for path in sorted(runs_dir.glob("*/run.json"), reverse=True):
            state = read_json(path)
            if (
                state["status"] in {"pending", "queued", "running"}
                and not (path.parent / "cancel.requested").exists()
                and asdict(HarnessConfig(**state["config"])) == asdict(config)
                and Path(state["dataset"]["manifest"]).resolve() == manifest_path.resolve()
                and [i["id"] for i in state["items"]] == selected
            ):
                if not is_active(path.parent):
                    start_run(path.parent, api_key)
                log_event("run.reused", run=path.parent.name)
                return path.parent, True
        root = create_run(
            manifest_path, runs_dir, config, split=split, item_ids=item_ids, limit=limit
        )
        start_run(root, api_key)
        return root, False
    finally:
        lease.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the STL → agent → CadQuery harness")
    parser.add_argument("--manifest", type=Path, default=Path("data/deepcad/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/harness"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--split", choices=["dev", "test", "all"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--target-iou", type=float, default=0.9)
    parser.add_argument("--token-budget", type=int, default=100_000)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    parser.add_argument("--memory-limit-mb", type=int, default=DEFAULT_MEMORY_MB)
    args = parser.parse_args()
    key = api_key_from_environment(Path.cwd())
    if not key:
        parser.error("Set OPENROUTER_API_KEY in the environment or .env")
    config = HarnessConfig(
        model=args.model,
        max_iterations=args.max_iterations,
        target_iou=args.target_iou,
        total_token_budget=args.token_budget,
        max_wall_seconds=args.wall_seconds,
        memory_limit_mb=args.memory_limit_mb,
    )
    root = args.resume or create_run(
        args.manifest, args.output, config, split=args.split, limit=args.limit
    )
    print(f"Run: {root}", flush=True)
    try:
        state = run_isolated(root, key)
    except KeyboardInterrupt:
        cancel_run(root)
        raise SystemExit(130) from None
    print(json.dumps(state["summary"], ensure_ascii=False, indent=2))
    raise SystemExit(0 if state["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
