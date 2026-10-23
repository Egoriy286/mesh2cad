"""Geometry tests: loading, normalization, RGB/depth renders, and IoU."""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
import trimesh
from PIL import Image

from mesh2cad.geometry import (
    VIEW_NAMES,
    VIEW_SIZE,
    GeometryCancelled,
    _camera_basis,
    _project,
    _raster_buffers,
    evaluate_iou,
    load_mesh,
    mesh_difference,
    mesh_stats,
    normalize_mesh,
    overlay_depth_calibration,
    render_llm_images,
    render_overlay_sheet,
    render_views,
    render_views_bytes,
    section_evidence,
    solid_difference_parts,
)


def box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(2.0, 1.0, 0.5))


def test_zero_area_export_faces_are_removed_without_filling_real_holes():
    mesh = trimesh.creation.box()
    degenerate = np.array([[0, 0, 1]])
    with_zero = trimesh.Trimesh(
        vertices=mesh.vertices, faces=np.vstack([mesh.faces, degenerate]), process=False
    )
    cleaned = load_mesh(with_zero.export(file_type="stl"))
    assert cleaned.is_volume
    assert len(cleaned.faces) == len(mesh.faces)
    assert cleaned.metadata["removed_zero_area_faces"] == 1
    assert evaluate_iou(mesh, cleaned)["iou"] == pytest.approx(1)
    open_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=np.vstack([mesh.faces[1:], degenerate]), process=False
    )
    still_open = load_mesh(open_mesh.export(file_type="stl"))
    assert not still_open.is_watertight
    assert evaluate_iou(mesh, still_open)["iou"] is None


def test_section_measurements_expose_holes_and_original_coordinates():
    mesh = trimesh.boolean.difference(
        [trimesh.creation.box(), trimesh.creation.box(extents=[0.4, 0.4, 2])], engine="manifold"
    )
    mesh.apply_translation([0.1, 0.2, 0.3])
    evidence = section_evidence(mesh)
    assert len(evidence["sections"]) == 9
    middle = next(s for s in evidence["sections"] if s["axis"] == "z" and s["at"] == 0.3)
    assert middle["closed_loops"] == 2
    assert middle["in_plane_axes"] == "xy"
    assert evidence["bounds"] == mesh.bounds.tolist()


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


def test_difference_of_identical_box_is_empty() -> None:
    pytest.importorskip("rtree")
    target = trimesh.creation.box()
    result = mesh_difference(target, target.copy())
    assert result["missing_faces"] == 0
    assert result["max_distance"] == pytest.approx(0, abs=1e-9)
    assert len(result["missing_triangles"]) == 0


def test_difference_marks_missing_volume_in_red() -> None:
    pytest.importorskip("rtree")
    target = trimesh.creation.box()
    prediction = target.copy()
    prediction.apply_translation((0.5, 0, 0))
    result = mesh_difference(target, prediction)
    assert result["missing_faces"] > 0
    assert result["missing_ratio"] > 0.3
    assert len(result["missing_triangles"]) == result["missing_faces"]
    assert result["covered_faces"] + result["missing_faces"] == result["total_faces"]


def test_difference_marks_target_surface_missing_for_small_prediction() -> None:
    pytest.importorskip("rtree")
    target = trimesh.creation.box()
    prediction = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    result = mesh_difference(target, prediction)
    assert result["missing_ratio"] > 0.9
    assert result["missing_faces"] == result["total_faces"]


def test_overlay_sheet_marks_missing_region_red() -> None:
    target = trimesh.creation.box()
    prediction = target.copy()
    prediction.apply_translation((0.5, 0, 0))
    payload = render_overlay_sheet(target, prediction, size=128)
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (128 * 3, 128 * 2)
        pixels = np.asarray(image.convert("RGB"))
    red = (pixels[..., 0] > 180) & (pixels[..., 1] < 80) & (pixels[..., 2] < 80)
    assert red.any()


def test_overlay_sheet_has_no_red_for_identical_boxes() -> None:
    target = trimesh.creation.box()
    payload = render_overlay_sheet(target, target, size=128)
    with Image.open(io.BytesIO(payload)) as image:
        pixels = np.asarray(image.convert("RGB"))
    red = (pixels[..., 0] > 180) & (pixels[..., 1] < 80) & (pixels[..., 2] < 80)
    assert not red.any()
    green = (pixels[..., 1] > 100) & (pixels[..., 0] < 80) & (pixels[..., 2] < 80)
    blue = (pixels[..., 2] > 100) & (pixels[..., 0] < 100) & (pixels[..., 1] < pixels[..., 2] * 0.8)
    assert green.any() and not blue.any()


def test_solid_colors_distinguish_contained_missing_and_outside_geometry():
    target = trimesh.creation.box()
    small = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    nested = solid_difference_parts(target, small)
    assert nested["excess"].is_empty
    assert nested["covered"].volume == pytest.approx(0.125)
    assert nested["missing"].volume == pytest.approx(0.875)
    extended = target.copy()
    extended.apply_translation([0.5, 0, 0])
    parts = solid_difference_parts(target, extended)
    assert all(part.volume == pytest.approx(0.5) for part in parts.values())
    with Image.open(io.BytesIO(render_overlay_sheet(target, extended, size=128))) as image:
        p = np.array(image)
    assert ((p[..., 1] > 100) & (p[..., 0] < 80) & (p[..., 2] < 80)).any()
    assert ((p[..., 2] > 100) & (p[..., 0] < 80) & (p[..., 1] < p[..., 2] * 0.8)).any()


def test_overlay_blue_includes_cad_filling_a_target_hole_not_just_bbox_excess():
    cube = trimesh.creation.box()
    target = trimesh.boolean.difference(
        [cube, trimesh.creation.box(extents=[0.4, 0.4, 2])], engine="manifold"
    )
    parts = solid_difference_parts(target, cube)
    assert parts["missing"].is_empty
    assert parts["excess"].volume == pytest.approx(0.16)
    with Image.open(io.BytesIO(render_overlay_sheet(target, cube, size=128))) as image:
        p = np.array(image)
    assert ((p[..., 2] > 100) & (p[..., 0] < 80) & (p[..., 1] < p[..., 2] * 0.8)).any()


def test_overlay_depth_changes_green_shade_between_parallel_surfaces():
    base = trimesh.creation.box(extents=[1, 1, 0.4])
    base.apply_translation([0, 0, -0.3])
    step = trimesh.creation.box(extents=[0.4, 0.6, 0.7])
    step.apply_translation([-0.25, 0, 0.15])
    target = trimesh.boolean.union([base, step], engine="manifold")
    original = target.vertices.copy()
    size = 128
    with Image.open(io.BytesIO(render_overlay_sheet(target, target, size=size))) as image:
        top = np.array(image)[:size, :size]
    _, right, up = _camera_basis(90, -90)
    points = np.array([[-0.25, 0, 0.5], [0.25, 0, -0.1]])
    x, y = _project(points, right, up, size)
    near, far = top[np.rint(y).astype(int), np.rint(x).astype(int)].astype(float)
    assert near[1] > far[1] + 70  # Same hue/normals; nearer green must be brighter.
    np.testing.assert_allclose(near / near[1], far / far[1], atol=0.015)
    planes = overlay_depth_calibration(target)["views"][0]
    for color, point in zip((near, far), points, strict=True):
        depth = point @ _camera_basis(90, -90)[0]
        expected = 0.35 + 0.65 * (planes["far"] - depth) / (planes["far"] - planes["near"])
        assert color[1] == pytest.approx(255 * 0.95 * expected, abs=2)
    np.testing.assert_array_equal(target.vertices, original)


def test_colored_depth_uses_fixed_planes_when_other_geometry_moves():
    near = trimesh.creation.box(extents=[0.3, 0.3, 0.2])
    near.apply_translation([-0.25, 0, 0.2])
    far = near.copy()
    far.apply_translation([0.5, 0, -0.4])
    options = {"color_depth_range": (-0.55, 0.55)}
    parts = [(near, (0.05, 0.95, 0.2), 0), (far, (0.05, 0.95, 0.2), 0)]
    before = np.array(_raster_buffers(parts, 90, -90, 128, **options)[0])
    outlier = trimesh.creation.box(extents=[0.1] * 3)
    outlier.apply_translation([0, 0.4, 4])
    after = np.array(
        _raster_buffers(parts + [(outlier, (1, 0.05, 0.05), 0)], 90, -90, 128, **options)[0]
    )
    _, right, up = _camera_basis(90, -90)
    x, y = _project(np.array([[-0.25, 0, 0.3], [0.25, 0, -0.1]]), right, up, 128)
    indices = (np.rint(y).astype(int), np.rint(x).astype(int))
    np.testing.assert_array_equal(before[indices], after[indices])
    assert np.all(after[0, 0] == 0)


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


def test_threshold_surface_classification_matches_full_nearest_query() -> None:
    target = trimesh.creation.box()
    prediction = trimesh.creation.icosphere(subdivisions=1, radius=0.55)
    prediction.apply_translation([0.1, 0, 0])
    result = mesh_difference(target, prediction, max_edge=0.12)
    covered = result["covered_triangles"]
    missing = result["missing_triangles"]
    vertices = result["covered_vertices"]
    faces = np.vstack((covered, missing))
    centers = vertices[faces].mean(axis=1)
    _, distances, _ = trimesh.proximity.closest_point(prediction, centers)
    expected = distances > result["threshold"]
    assert not expected[: len(covered)].any()
    assert expected[len(covered) :].all()
    assert result["max_distance"] is None


def test_render_and_surface_query_observe_cancellation() -> None:
    mesh = trimesh.creation.box()
    with pytest.raises(GeometryCancelled):
        render_llm_images(mesh, size=64, cancel_check=lambda: True)
    with pytest.raises(GeometryCancelled):
        mesh_difference(mesh, mesh.copy(), cancel_check=lambda: True)
