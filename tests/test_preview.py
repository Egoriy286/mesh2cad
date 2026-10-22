"""The display may be thinned; IoU always uses untouched full meshes."""

import numpy as np
import pytest
import trimesh

from mesh2cad.preview import comparison_figure, evaluate_saved_pair, preview_mesh
from mesh2cad.storage import file_digest


def test_preview_thins_triangles_without_changing_coordinates_or_source(tmp_path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.apply_translation([5, 2, -1])
    path = tmp_path / "target.stl"
    mesh.export(path)
    original_hash = file_digest(path)
    preview = preview_mesh(path, max_faces=80)
    assert len(preview["faces"]) == 80
    assert preview["total_faces"] == len(mesh.faces)
    assert preview["vertices"][:, 0].min() > 3.9
    assert preview["faces"].max() < len(preview["vertices"])
    assert file_digest(path) == original_hash
    score = evaluate_saved_pair(path, path)
    assert score["iou"] == pytest.approx(1)


def test_saved_pair_iou_does_not_align_the_prediction(tmp_path) -> None:
    target = trimesh.creation.box()
    candidate = target.copy()
    candidate.apply_translation([0.5, 0, 0])
    a, b = tmp_path / "a.stl", tmp_path / "b.stl"
    target.export(a)
    candidate.export(b)
    score = evaluate_saved_pair(a, b)
    assert score["iou"] == pytest.approx(1 / 3)
    previews = preview_mesh(a), preview_mesh(b)
    for mode in ("overlay", "side_by_side"):
        figure = comparison_figure(*previews, mode=mode)
        assert len(figure.data) == 2
        assert np.mean(figure.data[0].x) == pytest.approx(0)
        assert np.mean(figure.data[1].x) == pytest.approx(0.5)
        assert figure.layout.scene.xaxis.range is not None
        if mode == "side_by_side":
            assert figure.layout.scene.xaxis.range == figure.layout.scene2.xaxis.range


def test_pending_run_normalizes_only_raw_target_snapshot(tmp_path) -> None:
    target = trimesh.creation.box(extents=[2, 2, 2])
    target.apply_translation([7, -3, 10])
    a, b = tmp_path / "source.stl", tmp_path / "candidate.stl"
    target.export(a)
    trimesh.creation.box().export(b)
    assert evaluate_saved_pair(a, b, normalize_target=True)["iou"] == pytest.approx(1)
    assert evaluate_saved_pair(a, b)["iou"] == pytest.approx(0)


def test_open_interior_surface_has_no_fabricated_volume_iou(interior_manifest) -> None:
    from mesh2cad.dataset import read_catalogue

    item = read_catalogue(interior_manifest)["items"][1]
    path = interior_manifest.parent / item["mesh_path"]
    score = evaluate_saved_pair(path, path)
    assert score["iou"] is None
    assert score["status"] == "non_watertight_both"
