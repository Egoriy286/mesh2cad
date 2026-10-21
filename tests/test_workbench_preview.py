"""Real bounded preview decimation, without altering evaluated geometry."""

import numpy as np
import trimesh

from api.service import Service


def test_dense_closed_preview_preserves_surface_frame_originals_and_cache(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=7, radius=0.5)
    mesh.apply_scale([1, 0.7, 0.4])
    mesh.apply_translation([0.13, -0.07, 0.02])
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    preview, limited = Service.preview(mesh)
    assert limited
    assert len(mesh.faces) == 327680
    assert 0 < len(preview.faces) <= 30000
    assert preview.is_watertight
    assert len(preview.split()) == 1
    assert np.allclose(preview.bounds, mesh.bounds, atol=2e-4, rtol=0)
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.faces, faces)
    assert preview.metadata["preview"]["approximate"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Repeated geometry should use the bounded cache")

    monkeypatch.setattr("api.preview.subprocess.run", forbidden)
    cached, _ = Service.preview(mesh)
    cached.vertices[0] += 123
    fresh, _ = Service.preview(mesh)
    assert np.allclose(fresh.bounds, preview.bounds)
    assert np.array_equal(mesh.vertices, vertices)


def test_open_preview_remains_valid_without_filling_its_boundary():
    mesh = trimesh.creation.icosphere(subdivisions=6, radius=0.5)
    mesh.update_faces(np.all(mesh.vertices[mesh.faces, 2] >= 0, axis=1))
    mesh.remove_unreferenced_vertices()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    assert not mesh.is_watertight
    preview, limited = Service.preview(mesh, limit=10000)
    assert limited
    assert 0 < len(preview.faces) <= 10000
    assert np.isfinite(preview.vertices).all()
    assert not preview.is_watertight
    assert np.allclose(preview.bounds, mesh.bounds, atol=0.005, rtol=0)
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.faces, faces)
