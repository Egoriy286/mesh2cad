"""Pinned STL subsets and cached, normalized inputs for reconstruction."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Callable

from .geometry import load_mesh, normalize_mesh, render_views
from .storage import digest, file_digest, read_json, resolve_artifact, write_json

DEFAULT_REPO = "maksimko123/deepcad_test_mesh"


def prepare_dataset(
    output: Path,
    *,
    source_dir: Path | None = None,
    repo: str = DEFAULT_REPO,
    revision: str | None = None,
    dev_count: int = 10,
    test_count: int = 100,
    seed: int = 42,
    progress: Callable[[str], None] = print,
) -> Path:
    if dev_count < 0 or test_count < 1:
        raise ValueError("dev-count must be nonnegative and test-count must be positive")
    output = output.resolve()
    manifest_path = output / "manifest.json"
    requested_source = (
        {"kind": "local", "path": str(source_dir.resolve())}
        if source_dir is not None
        else {"kind": "huggingface", "repo": repo}
    )
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            manifest["requested_source"] != requested_source
            or manifest["seed"] != seed
            or manifest["dev_count"] != dev_count
            or manifest["test_count"] != test_count
            or (revision is not None and revision != manifest["source"].get("revision"))
        ):
            raise ValueError("Existing manifest has different settings; choose a new --output")
    else:
        if source_dir is not None:
            source_dir = source_dir.resolve()
            if not source_dir.is_dir():
                raise ValueError(f"STL source directory does not exist: {source_dir}")
            files = sorted(
                p.relative_to(source_dir).as_posix()
                for p in source_dir.rglob("*")
                if p.is_file() and p.suffix.lower() == ".stl"
            )
            source = requested_source.copy()
        else:
            from huggingface_hub import HfApi

            api = HfApi()
            commit = api.repo_info(repo, repo_type="dataset", revision=revision).sha
            if not commit:
                raise ValueError("Hugging Face did not return a pinned dataset revision")
            source = {**requested_source, "revision": commit}
            files = sorted(
                name
                for name in api.list_repo_files(repo, repo_type="dataset", revision=commit)
                if name.lower().endswith(".stl")
            )
        count = dev_count + test_count
        if len(files) < count:
            raise ValueError(f"Need {count} STL files, found {len(files)}")
        random.Random(seed).shuffle(files)
        manifest = {
            "schema_version": 1,
            "requested_source": requested_source,
            "source": source,
            "seed": seed,
            "dev_count": dev_count,
            "test_count": test_count,
            "items": [
                {
                    "id": digest(name)[:16],
                    "source_file": name,
                    "split": "dev" if index < dev_count else "test",
                    "status": "pending",
                }
                for index, name in enumerate(files[:count])
            ],
        }
        write_json(manifest_path, manifest)

    if manifest["source"]["kind"] == "huggingface":
        pending_files = [
            item["source_file"] for item in manifest["items"] if item["status"] != "ready"
        ]
        if pending_files:
            from huggingface_hub import snapshot_download

            # Download only the chosen files, concurrently; rendering and model
            # calls remain bounded and sequential on the laptop.
            snapshot_download(
                manifest["source"]["repo"],
                repo_type="dataset",
                revision=manifest["source"]["revision"],
                allow_patterns=pending_files,
                cache_dir=output / ".hf-cache",
                max_workers=4,
            )

    for item in manifest["items"]:
        if item["status"] == "ready":
            # Do not silently reuse modified or partially deleted caches.
            if all(
                resolve_artifact(output, item[key]).is_file()
                and file_digest(resolve_artifact(output, item[key])) == item[hash_key]
                for key, hash_key in (
                    ("source_path", "source_sha256"),
                    ("mesh_path", "mesh_sha256"),
                    ("image_path", "image_sha256"),
                )
            ):
                continue
            raise ValueError(f"Prepared artifacts changed for {item['id']}; use a new output")
        try:
            original = output / "sources" / f"{item['id']}.stl"
            original.parent.mkdir(parents=True, exist_ok=True)
            if manifest["source"]["kind"] == "local":
                source = resolve_artifact(Path(manifest["source"]["path"]), item["source_file"])
            else:
                from huggingface_hub import hf_hub_download

                source = Path(
                    hf_hub_download(
                        manifest["source"]["repo"],
                        item["source_file"],
                        repo_type="dataset",
                        revision=manifest["source"]["revision"],
                        cache_dir=output / ".hf-cache",
                    )
                )
            shutil.copyfile(source, original)
            mesh, normalization = normalize_mesh(load_mesh(original))
            mesh_path = output / "meshes" / f"{item['id']}.stl"
            image_path = output / "views" / f"{item['id']}.png"
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(mesh_path)
            render_views(mesh, image_path)
            item.update(
                status="ready",
                error=None,
                normalization=normalization,
                target_watertight=bool(mesh.is_watertight and mesh.is_volume),
                source_path=original.relative_to(output).as_posix(),
                source_sha256=file_digest(original),
                mesh_path=mesh_path.relative_to(output).as_posix(),
                mesh_sha256=file_digest(mesh_path),
                image_path=image_path.relative_to(output).as_posix(),
                image_sha256=file_digest(image_path),
            )
        except KeyboardInterrupt:
            write_json(manifest_path, manifest)
            raise
        except Exception as exc:
            item.update(status="prepare_error", error=f"{type(exc).__name__}: {exc}")
        write_json(manifest_path, manifest)
        progress(f"prepare {item['split']} {item['id']}: {item['status']}")
    return manifest_path


def load_manifest(path: Path) -> dict:
    manifest = read_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest version")
    items = manifest.get("items", [])
    ids = [item["id"] for item in items]
    if not items or len(set(ids)) != len(ids):
        raise ValueError("Manifest must contain unique, nonempty item IDs")
    for item in items:
        if not isinstance(item["id"], str) or not item["id"].isalnum():
            raise ValueError("Manifest IDs must be alphanumeric")
        if item["split"] not in {"dev", "test"}:
            raise ValueError("Manifest split must be dev or test")
        if item["status"] == "ready":
            for key in ("mesh_path", "image_path"):
                artifact = resolve_artifact(path.parent, item[key])
                if (
                    not artifact.is_file()
                    or file_digest(artifact) != item[key.replace("path", "sha256")]
                ):
                    raise ValueError(f"Missing or changed input artifact: {artifact}")
    return manifest
