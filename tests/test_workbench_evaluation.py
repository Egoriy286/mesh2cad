import pytest
import trimesh

from api.evaluation import evaluate_pair


def test_identical_mesh_and_translated_prediction_are_not_aligned(tmp_path):
    target = trimesh.creation.box()
    target_path, same_path, moved_path = (tmp_path / name for name in ("target.stl", "same.stl", "moved.stl"))
    target.export(target_path)
    target.export(same_path)
    target.apply_translation([3, 0, 0])
    target.export(moved_path)
    same = evaluate_pair(target_path, same_path, n_points=512)
    assert same["iou"] == pytest.approx(1)
    assert same["chamfer"] == pytest.approx(0)
    assert same["f_score"] == pytest.approx(1)
    assert same["h95"] == pytest.approx(0)
    assert same["volume_relative_error"] == pytest.approx(0)
    moved = evaluate_pair(target_path, moved_path, n_points=512)
    assert moved["iou"] == pytest.approx(0)
    assert moved["f_score"] == 0
    assert moved["chamfer"] > 1
    assert moved["bbox_center_distance"] == pytest.approx(3)
    assert moved["target_sha256"] != moved["prediction_sha256"]


def test_open_surface_metrics_keep_iou_and_volume_unavailable(tmp_path):
    target = trimesh.creation.box()
    open_mesh = target.copy()
    open_mesh.update_faces(list(range(len(open_mesh.faces) - 1)))
    target_path, prediction_path = tmp_path / "target.stl", tmp_path / "open.stl"
    target.export(target_path)
    open_mesh.export(prediction_path)
    result = evaluate_pair(target_path, prediction_path, n_points=512)
    assert result["iou"] is None
    assert result["iou_status"] == "non_watertight_prediction"
    assert result["volume_relative_error"] is None
    assert result["chamfer"] >= 0
    assert result["prediction_stats"]["volume"] is None


def test_evaluation_is_reproducible_and_records_threshold(tmp_path):
    target, candidate = tmp_path / "target.stl", tmp_path / "candidate.stl"
    trimesh.creation.box().export(target)
    trimesh.creation.icosphere(subdivisions=1, radius=0.5).export(candidate)
    first = evaluate_pair(target, candidate, n_points=256, seed=7, threshold=.02)
    assert first == evaluate_pair(target, candidate, n_points=256, seed=7, threshold=.02)
    assert first["seed"] == 7 and first["threshold"] == .02
    with pytest.raises(ValueError):
        evaluate_pair(target, candidate, threshold=float("nan"))
