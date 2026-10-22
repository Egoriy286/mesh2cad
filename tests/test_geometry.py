"""Geometry tests: loading, normalization, and the exact 512x512 LLM image."""

from __future__ import annotations

import io

import trimesh
from PIL import Image

from mesh2cad.geometry import (
    VIEW_SIZE,
    load_mesh,
    mesh_stats,
    normalize_mesh,
    render_views,
    render_views_bytes,
)


def box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(2.0, 1.0, 0.5))


def test_load_mesh_from_bytes_merges_stl_vertices() -> None:
    payload = box().export(file_type="stl")
    mesh = load_mesh(payload)
    assert mesh.is_watertight
    assert len(mesh.vertices) == 8
    assert len(mesh.faces) == 12


def test_normalization_is_explicit_and_reversible() -> None:
    normalized, values = normalize_mesh(box())
    stats = mesh_stats(normalized)
    assert max(stats["extents"]) == 1.0
    assert all(abs(value) < 1e-9 for value in (normalized.bounds[0] + normalized.bounds[1]))
    assert len(values["transform"]) == 4
    assert len(values["inverse_transform"]) == 4


def test_render_views_is_exactly_512() -> None:
    normalized, _ = normalize_mesh(box())
    payload = render_views_bytes(normalized)
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (VIEW_SIZE, VIEW_SIZE)
        assert image.mode in {"RGB", "RGBA"}


def test_render_views_can_write_a_file(tmp_path) -> None:
    normalized, _ = normalize_mesh(box())
    output = tmp_path / "views.png"
    render_views(normalized, output, size=128)
    with Image.open(output) as image:
        assert image.size == (128, 128)
