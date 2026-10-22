"""Geometry tests: loading, normalization, RGB/depth renders, and IoU."""

from __future__ import annotations

import io
import warnings

import pytest
import trimesh
from PIL import Image

from mesh2cad.geometry import (
    VIEW_NAMES,
    VIEW_SIZE,
    evaluate_iou,
    load_mesh,
    mesh_stats,
    normalize_mesh,
    render_llm_images,
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


def test_llm_images_are_black_512_pngs_for_every_view() -> None:
    normalized, _ = normalize_mesh(box())
    images = render_llm_images(normalized)
    expected = {f"{view}-{kind}" for view in VIEW_NAMES for kind in ("rgb", "depth")}
    assert set(images) == expected
    for label, payload in images.items():
        assert payload.startswith(b"\x89PNG\r\n\x1a\n"), label
        with Image.open(io.BytesIO(payload)) as image:
            assert image.size == (VIEW_SIZE, VIEW_SIZE), label
            assert image.convert("RGB").getpixel((0, 0)) == (0, 0, 0), label
    assert images["top-rgb"] != images["top-depth"]
    assert len({images[f"{view}-rgb"] for view in VIEW_NAMES}) >= 4


def test_render_views_builds_combined_sheet() -> None:
    normalized, _ = normalize_mesh(box())
    payload = render_views_bytes(normalized)
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (VIEW_SIZE * 3, VIEW_SIZE * 2)
        assert image.mode == "RGB"
        assert image.convert("RGB").getpixel((VIEW_SIZE - 1, VIEW_SIZE - 1)) == (0, 0, 0)


def test_render_views_can_write_a_file(tmp_path) -> None:
    normalized, _ = normalize_mesh(box())
    output = tmp_path / "views.png"
    render_views(normalized, output, size=128)
    with Image.open(output) as image:
        assert image.size == (128 * 3, 256)


def test_iou_of_identical_box_is_one() -> None:
    pytest.importorskip("manifold3d")
    target = trimesh.creation.box()
    result = evaluate_iou(target, target.copy())
    assert result["status"] == "ok"
    assert result["iou"] == pytest.approx(1.0)
    assert result["intersection_volume"] == pytest.approx(1.0)
    assert result["union_volume"] == pytest.approx(1.0)


def test_iou_uses_true_boolean_union() -> None:
    pytest.importorskip("manifold3d")
    target = trimesh.creation.box()
    prediction = target.copy()
    prediction.apply_translation((0.5, 0, 0))
    result = evaluate_iou(target, prediction)
    assert result["status"] == "ok"
    assert result["iou"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["intersection_volume"] == pytest.approx(0.5)
    assert result["union_volume"] == pytest.approx(1.5)


def test_iou_is_zero_for_disjoint_boxes() -> None:
    pytest.importorskip("manifold3d")
    target = trimesh.creation.box()
    prediction = target.copy()
    prediction.apply_translation((3, 0, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = evaluate_iou(target, prediction)
    assert result["status"] == "ok"
    assert result["iou"] == pytest.approx(0, abs=1e-9)


def test_iou_is_unavailable_for_open_surface() -> None:
    target = trimesh.creation.box()
    prediction = trimesh.Trimesh(vertices=target.vertices, faces=target.faces[:-1], process=False)
    result = evaluate_iou(target, prediction)
    assert result["iou"] is None
    assert result["status"] == "non_watertight_prediction"
    assert result["prediction_watertight"] is False


def test_iou_boolean_failure_is_visible(monkeypatch) -> None:
    pytest.importorskip("manifold3d")

    def broken_boolean(*_args, **_kwargs):
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(trimesh.boolean, "intersection", broken_boolean)
    target = trimesh.creation.box()
    result = evaluate_iou(target, target)
    assert result["iou"] is None
    assert result["status"] == "boolean_error:RuntimeError"
