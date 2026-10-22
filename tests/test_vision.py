"""A shared camera/depth scale must reveal geometric errors without changing the STL."""

import io

import numpy as np
import pytest
import trimesh
from PIL import Image

from mesh2cad.geometry import GeometryCancelled, _raster_buffers
from mesh2cad.vision import (
    compare_observations,
    load_observations,
    render_observations,
    save_observations,
)


def test_linear_depth_uses_target_planes_and_detects_wrong_thickness(tmp_path):
    target = trimesh.creation.box(extents=[1, 0.8, 0.6])
    thin = trimesh.creation.box(extents=[1, 0.8, 0.4])
    original = thin.vertices.copy()
    a = render_observations(target, target.extents, size=64)
    b = render_observations(thin, target.extents, size=64)
    assert a["raw_depth"]["top"][32, 32] == pytest.approx(-0.3)
    assert b["raw_depth"]["top"][32, 32] == pytest.approx(-0.2)
    with Image.open(io.BytesIO(a["depth"])) as image:
        g = image.getpixel((32, 32))
        assert image.getpixel((0, 0)) == 0
    planes = a["calibration"]["views"][0]
    assert planes["dominant_depth_levels"][0]["depth"] == pytest.approx(-0.3)
    assert planes["dominant_depth_levels"][0]["fraction"] == pytest.approx(1)
    decoded = planes["far"] - (g - 1) / 254 * (planes["far"] - planes["near"])
    assert decoded == pytest.approx(-0.3, abs=0.003)
    metrics = compare_observations(a, b)["metrics"]["views"]
    top = metrics[0]
    assert top["silhouette_iou"] == pytest.approx(1)
    assert top["depth_mae"] == pytest.approx(0.1)
    assert top["missing_or_farther_pixels"] > 0
    np.testing.assert_array_equal(thin.vertices, original)
    save_observations(a, tmp_path / "vision")
    restored = load_observations(tmp_path / "vision")
    np.testing.assert_array_equal(restored["raw_depth"]["top"], a["raw_depth"]["top"])
    assert restored["calibration"] == a["calibration"]


def test_registered_sheets_and_matching_geometry_have_zero_visual_error():
    mesh = trimesh.creation.box()
    a = render_observations(mesh, mesh.extents, size=64)
    assert set(a) == {"depth", "raw_depth", "calibration"}
    with Image.open(io.BytesIO(a["depth"])) as image:
        assert image.size == (192, 128)
        assert image.mode == "L"
    metrics = compare_observations(a, a)["metrics"]["views"]
    assert all(v["silhouette_iou"] == 1 and v["depth_mae"] == 0 for v in metrics)
    assert len(a["calibration"]["views"]) == 6
    assert a["calibration"]["views"][0]["camera_direction"] == [0.0, 0.0, 1.0]
    # The full isometric silhouette must fit, including the unit cube's corners.
    assert a["calibration"]["frame_width"] > 1.4
    for depth in a["raw_depth"].values():
        assert not np.isfinite(depth[[0, -1], :]).any()
        assert not np.isfinite(depth[:, [0, -1]]).any()


def test_vision_batches_honor_cancellation():
    mesh = trimesh.creation.box()
    with pytest.raises(GeometryCancelled):
        render_observations(mesh, mesh.extents, size=64, cancel_check=lambda: True)


def test_depth_only_raster_matches_color_path_without_creating_image():
    mesh = trimesh.creation.box(extents=[1, 0.8, 0.6])
    parts = [(mesh, (0.3, 0.68, 1), 0)]
    color, reference, low, high = _raster_buffers(parts, -25, -55, 64)
    image, actual, actual_low, actual_high = _raster_buffers(
        parts, -25, -55, 64, render_color=False
    )
    assert color is not None and image is None
    np.testing.assert_array_equal(actual, reference)
    assert (actual_low, actual_high) == (low, high)
