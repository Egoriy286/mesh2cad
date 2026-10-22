"""Small attributed interior inputs shared by dataset and GUI tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import trimesh
from PIL import Image

from mesh2cad.storage import write_json


@pytest.fixture(autouse=True)
def isolated_project_logs(tmp_path, monkeypatch):
    # Unit-test runs must not be mixed with the user's actual console history.
    monkeypatch.setenv("MESH2CAD_LOG_DIR", str(tmp_path / "logs"))


@pytest.fixture
def interior_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "data/interior300"
    (root / "meshes").mkdir(parents=True)
    (root / "previews").mkdir()
    items = []
    for name, category, source, closed in (
        ("indoorcad_box-model_01", "cabinet", "indoorcad", True),
        ("polyhaven_open_shelf", "furniture", "polyhaven", False),
    ):
        mesh = trimesh.creation.box()
        if not closed:
            mesh.update_faces(range(len(mesh.faces) - 1))
        payload = mesh.export(file_type="stl")
        relative = f"meshes/{name}.stl"
        (root / relative).write_bytes(payload)
        Image.new("RGB", (24, 24), "#6ea8fe").save(root / "previews" / f"{name}_aligned.jpg")
        items.append(
            {
                "id": name,
                "name": name.replace("_", " "),
                "source": source,
                "category": category,
                "mesh_path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "watertight": closed,
                "is_volume": closed,
                "faces": len(mesh.faces),
                "license": "fixture",
                "revision": "fixture-revision",
            }
        )
    path = root / "manifest.json"
    write_json(path, {"schema_version": 1, "seed": 42, "summary": {"count": 2}, "items": items})
    return path
