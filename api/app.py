"""FastAPI HTTP contract for the local Mesh2CAD workbench."""

from __future__ import annotations

import csv
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .backend import SUPPORTED_MODEL
from .demo import EXAMPLES, example_mesh
from .importer import MAX_BYTES
from .service import Service


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["predict", "edit", "explain", "execute", "evaluate"]
    asset_id: str = Field(min_length=1, max_length=64)
    parent_id: str | None = Field(default=None, max_length=64)
    prompt: str | None = Field(default=None, max_length=12000)
    code: str | None = Field(default=None, max_length=60000)
    provider: Literal["demo", "openrouter", "codex"] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=150)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    max_attempts: int | None = Field(default=None, ge=1, le=3, strict=True)
    n_points: int = Field(default=8192, ge=128, le=65536, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    threshold: float = Field(default=0.01, gt=0, le=1)

    @model_validator(mode="after")
    def code_required(self):
        if self.kind == "execute" and (not self.code or not self.code.strip()):
            raise ValueError("Manual execution requires nonempty CadQuery code")
        return self


class SettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    provider: Literal["demo", "openrouter", "codex"] | None = None
    model: Literal[SUPPORTED_MODEL] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    api_key: str | None = Field(default=None, max_length=500)
    budget_limit_usd: float | None = Field(default=None, ge=0, le=100)
    max_attempts: int | None = Field(default=None, ge=1, le=3, strict=True)


def create_app(data_dir=None, start_worker=True):
    root = Path(data_dir or os.environ.get("MESH2CAD_DATA_DIR", ".cache/workbench"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = Service(root, start_worker=start_worker)
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(title="Mesh2CAD local workbench", lifespan=lifespan)

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin:
                parsed = urlparse(origin)
                if (
                    parsed.scheme not in {"http", "https"}
                    or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                    or parsed.port not in {8501, 8765}
                ):
                    return Response(
                        '{"detail":"Cross-origin writes are disabled"}',
                        status_code=403,
                        media_type="application/json",
                    )
            size = request.headers.get("content-length")
            try:
                if size and int(size) > MAX_BYTES + 65536:
                    return Response(
                        '{"detail":"Request exceeds 25 MiB limit"}',
                        status_code=413,
                        media_type="application/json",
                    )
            except ValueError:
                return Response(status_code=400)
            # Bound chunked uploads as well as requests declaring their length.
            original_receive = request._receive
            total = 0

            async def bounded_receive():
                nonlocal total
                message = await original_receive()
                total += len(message.get("body", b""))
                if total > MAX_BYTES + 65536:
                    raise HTTPException(413, "Request exceeds 25 MiB limit")
                return message

            request._receive = bounded_receive
        return await call_next(request)

    def service():
        return app.state.service

    def get(kind, item_id):
        item = service().store.get(kind, item_id)
        if not item:
            raise HTTPException(404, f"{kind.rstrip('s').capitalize()} not found")
        return item

    def action(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def artifact(kind, item_id, name, media):
        get(kind, item_id)
        try:
            path = service().artifact_path(kind, item_id, name)
        except ValueError as exc:
            raise HTTPException(404, "Artifact unavailable for this item") from exc
        return FileResponse(path, media_type=media, filename=f"{item_id}-{name}")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "worker": bool(service().thread and service().thread.is_alive()),
            "mode": "local",
            "formats": ["STL", "OBJ", "PLY", "OFF", "GLB"],
        }

    @app.get("/api/state")
    def state(asset_id: str | None = None):
        assets = service().store.all("assets")
        if asset_id is not None:
            get("assets", asset_id)
        else:
            asset_id = assets[-1]["id"] if assets else None
        return {
            "assets": assets,
            "candidates": [
                item for item in service().store.all("candidates") if item["asset_id"] == asset_id
            ],
            "jobs": [item for item in service().store.all("jobs") if item["asset_id"] == asset_id],
            "messages": [
                item for item in service().store.all("messages") if item["asset_id"] == asset_id
            ],
            "settings": service().settings(),
            "budget": service().budget(),
        }

    @app.get("/api/examples")
    def examples():
        return {"examples": EXAMPLES}

    @app.post("/api/examples/{example_id}")
    def example(example_id: str):
        if example_id not in {example["id"] for example in EXAMPLES}:
            raise HTTPException(404, "Example not found")
        mesh = example_mesh(example_id)
        asset = action(
            service().import_asset,
            mesh.export(file_type="stl"),
            f"Offline demo - {example_id}.stl",
            example_id=example_id,
        )
        service().submit(
            {
                "kind": "predict",
                "asset_id": asset["id"],
                "provider": "demo",
                "max_attempts": 1,
                "prompt": "Deterministic offline example; no model call",
            }
        )
        return asset

    @app.post("/api/assets", status_code=201)
    async def upload(file: UploadFile = File(...)):
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "File exceeds 25 MiB limit")
        # Blocking parser/image work belongs to the threadpool, not the event loop.
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(
            action, service().import_asset, data, file.filename or "upload"
        )

    @app.get("/api/assets/{asset_id}/mesh")
    def asset_mesh(asset_id: str):
        return artifact("assets", asset_id, "target.stl", "model/stl")

    @app.get("/api/assets/{asset_id}/image")
    def asset_image(asset_id: str):
        return artifact("assets", asset_id, "image.png", "image/png")

    @app.get("/api/geometry")
    def geometry(asset_id: str, candidate_id: str | None = None):
        from mesh2cad.geometry import load_mesh

        asset = get("assets", asset_id)
        target, target_limited = action(
            service().preview,
            load_mesh(action(service().artifact_path, "assets", asset_id, "target.stl"))
        )

        def geometry_data(mesh):
            return {"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}

        candidate, limited = None, False
        if candidate_id:
            selected = get("candidates", candidate_id)
            if selected["asset_id"] != asset_id:
                raise HTTPException(400, "Candidate does not belong to selected asset")
            if selected["mesh_url"]:
                mesh, limited = action(
                    service().preview,
                    load_mesh(
                        action(service().artifact_path, "candidates", candidate_id, "candidate.stl")
                    )
                )
                candidate = geometry_data(mesh)
        return {
            "frame_id": f"normalized-target:{asset['id']}",
            "target": geometry_data(target),
            "candidate": candidate,
            "preview": {
                "face_limit": 30000,
                "target_limited": target_limited,
                "candidate_limited": limited,
                "evaluation_uses_original": True,
            },
        }

    @app.post("/api/jobs", status_code=202)
    def submit(request: JobRequest, idempotency_key: str | None = Header(default=None)):
        return action(service().submit, request.model_dump(exclude_none=True), idempotency_key)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        get("jobs", job_id)
        return action(service().cancel, job_id)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return get("jobs", job_id)

    @app.get("/api/candidates/{candidate_id}/{artifact_name}")
    def candidate_artifact(candidate_id: str, artifact_name: str):
        files = {
            "mesh": ("candidate.stl", "model/stl"),
            "step": ("candidate.step", "application/step"),
            "code": ("code.py", "text/plain"),
            "comparison": ("comparison.png", "image/png"),
        }
        if artifact_name not in files:
            raise HTTPException(404, "Unsupported artifact")
        return artifact("candidates", candidate_id, *files[artifact_name])

    @app.get("/api/assets/{asset_id}/report")
    def report(asset_id: str, format: Literal["json", "csv"] = "json"):
        asset = get("assets", asset_id)
        candidates = [
            item for item in service().store.all("candidates") if item["asset_id"] == asset_id
        ]
        if format == "json":
            return {
                "asset": asset,
                "candidates": candidates,
                "budget": service().budget(),
                "coordinate_frame": "normalized-target",
                "selection_policy": "manual; recommend maximum available IoU, Chamfer tie-break; fallback minimum Chamfer",
            }
        rows = [
            {
                "candidate_id": item["id"],
                "parent_id": item["parent_id"],
                "label": item["label"],
                "status": item["status"],
                "error": item["error"],
                **(item["metrics"] or {}),
            }
            for item in candidates
        ]
        fields = list(dict.fromkeys(key for row in rows for key in row)) or [
            "candidate_id",
            "status",
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            # Spreadsheet formula injection must not turn user filenames/code into actions.
            writer.writerow(
                {
                    k: (
                        "'" + str(v)
                        if isinstance(v, str) and v.startswith(("=", "+", "-", "@"))
                        else v
                    )
                    for k, v in row.items()
                }
            )
        return Response(
            output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{asset_id}-report.csv"'},
        )

    @app.get("/api/settings")
    def settings():
        return {"settings": service().settings(), "budget": service().budget()}

    @app.post("/api/settings")
    def set_settings(request: SettingsRequest):
        return action(service().update_settings, request.model_dump(exclude_none=True))

    @app.post("/api/auth/check")
    def check_key():
        return action(service().check_key)

    return app
