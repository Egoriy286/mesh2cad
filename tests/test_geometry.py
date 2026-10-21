from pathlib import Path

import numpy as np
import pytest
import trimesh
from matplotlib import image as matplotlib_image

from mesh2cad.geometry import (
    evaluate_meshes,
    load_mesh,
    normalize_mesh,
    render_comparison,
    render_views,
)


def test_normalization_is_reversible_and_does_not_move_source():
    mesh = trimesh.creation.box(extents=(2, 4, 8))
    mesh.apply_translation((5, -3, 11))
    original = mesh.vertices.copy()
    normalized, transform = normalize_mesh(mesh)
    np.testing.assert_allclose(normalized.bounds.mean(axis=0), 0, atol=1e-14)
    assert max(normalized.extents) == pytest.approx(1)
    restored = trimesh.transform_points(normalized.vertices, transform["inverse_transform"])
    np.testing.assert_allclose(restored, original)
    np.testing.assert_array_equal(mesh.vertices, original)
    np.testing.assert_allclose(
        np.asarray(transform["inverse_transform"]) @ transform["transform"], np.eye(4)
    )


def test_stl_load_merges_duplicates_for_watertightness(tmp_path: Path):
    path = tmp_path / "box.stl"
    trimesh.creation.box().export(path)
    loaded = load_mesh(path)
    assert loaded.is_watertight
    assert loaded.volume == pytest.approx(1)


def test_scene_load_preserves_instance_transforms(tmp_path: Path):
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(), transform=trimesh.transformations.translation_matrix((4, 0, 0))
    )
    scene.add_geometry(
        trimesh.creation.box(), transform=trimesh.transformations.translation_matrix((-4, 0, 0))
    )
    path = tmp_path / "scene.glb"
    path.write_bytes(scene.export(file_type="glb"))
    mesh = load_mesh(path)
    np.testing.assert_allclose(mesh.bounds, ((-4.5, -0.5, -0.5), (4.5, 0.5, 0.5)))
    assert mesh.volume == pytest.approx(2)


@pytest.mark.parametrize("kind", ["empty", "nonfinite", "degenerate"])
def test_invalid_mesh_rejected(kind):
    if kind == "empty":
        mesh = trimesh.Trimesh()
    elif kind == "nonfinite":
        mesh = trimesh.creation.box()
        mesh.vertices[0, 0] = float("nan")
    else:
        mesh = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=[[0, 1, 2]], process=False)
    with pytest.raises(ValueError):
        normalize_mesh(mesh)


def test_identical_mesh_has_zero_chamfer_and_unit_iou():
    target = trimesh.creation.box()
    metrics = evaluate_meshes(target, target.copy())
    assert metrics["chamfer"] == 0
    assert metrics["iou"] == pytest.approx(1)
    assert metrics["iou_status"] == "ok"
    assert metrics["target_watertight"] and metrics["prediction_watertight"]


def test_moved_box_uses_true_union_and_retains_position_error():
    target = trimesh.creation.box()
    prediction = target.copy()
    prediction.apply_translation((0.5, 0, 0))
    metrics = evaluate_meshes(target, prediction)
    # Intersection = 0.5, union = 1.5, rather than union's bounding-box volume.
    assert metrics["iou"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["chamfer"] > 0.05
    assert evaluate_meshes(target, prediction) == metrics


def test_diagonal_offset_uses_boolean_union_instead_of_bounding_box():
    target = trimesh.creation.box()
    prediction = target.copy()
    prediction.apply_translation((0.5, 0.5, 0))
    metrics = evaluate_meshes(target, prediction, n_points=512)
    # Intersection = 0.25, union = 1.75; the enclosing box has volume 2.25.
    assert metrics["iou"] == pytest.approx(1 / 7, abs=1e-6)


def test_disjoint_mesh_has_zero_iou():
    target = trimesh.creation.box()
    prediction = target.copy()
    prediction.apply_translation((3, 0, 0))
    metrics = evaluate_meshes(target, prediction, n_points=512)
    assert metrics["iou"] == 0
    assert metrics["chamfer"] >= 8


def test_scale_error_is_not_normalized_away():
    target = trimesh.creation.box()
    prediction = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    metrics = evaluate_meshes(target, prediction)
    assert metrics["iou"] == pytest.approx(0.125)
    assert metrics["chamfer"] > 0.1
    np.testing.assert_allclose(prediction.extents, 0.5)


def test_open_surface_has_chamfer_but_explicitly_unavailable_iou():
    target = trimesh.creation.box()
    prediction = trimesh.Trimesh(vertices=target.vertices, faces=target.faces[:-1], process=False)
    metrics = evaluate_meshes(target, prediction, n_points=512)
    assert metrics["chamfer"] >= 0
    assert metrics["iou"] is None
    assert metrics["iou_status"] == "non_watertight_prediction"
    assert metrics["target_watertight"]
    assert not metrics["prediction_watertight"]


def test_boolean_failure_is_visible(monkeypatch):
    def broken_boolean(*args, **kwargs):
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(trimesh.boolean, "intersection", broken_boolean)
    target = trimesh.creation.box()
    metrics = evaluate_meshes(target, target, n_points=64)
    assert metrics["chamfer"] == 0
    assert metrics["iou"] is None
    assert metrics["iou_status"] == "boolean_error:RuntimeError"


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_sampling_count_rejected(count):
    box = trimesh.creation.box()
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_meshes(box, box, n_points=count)


def test_cpu_render_writes_nonblank_views_and_comparison(tmp_path: Path):
    mesh = trimesh.creation.box(extents=(1, 0.6, 0.3))
    views, comparison = tmp_path / "views.png", tmp_path / "comparison.png"
    render_views(mesh, views)
    render_comparison(mesh, mesh, comparison)
    view_image = matplotlib_image.imread(views)
    comparison_image = matplotlib_image.imread(comparison)
    assert view_image.shape[:2] == (384, 1440)
    assert comparison_image.shape[:2] == (768, 1440)
    assert (view_image[:, :, :3] < 0.7).any()
    assert (comparison_image[:, :, :3] < 0.7).any()
