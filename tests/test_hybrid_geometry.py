"""Geometric and subprocess contracts for bounded hybrid evidence."""

import hashlib
import json

import numpy as np
import pytest
import trimesh

from api.hybrid import HYBRID_VERSION, HybridCancelled, build_evidence, prepare_evidence

SHA = "a" * 64


def test_box_measurements_and_no_false_cylinder():
    evidence = build_evidence(trimesh.creation.box(extents=[1, 0.5, 0.25]), SHA)
    assert evidence["schema_version"] == HYBRID_VERSION
    assert evidence["coordinate_frame"] == "normalized-target"
    assert evidence["extents"] == [1, 0.5, 0.25]
    assert len(evidence["planes"]) == 6
    assert sorted(p["area"] for p in evidence["planes"]) == [0.125, 0.125, 0.25, 0.25, 0.5, 0.5]
    assert max(p["residual_max"] for p in evidence["planes"]) == 0
    assert len(evidence["sections"]) == 9
    assert all(
        len(s["contours"]) == 1 and s["contours"][0]["closed"] and not s["truncated"]
        for s in evidence["sections"]
    )
    assert evidence["cylinders"] == []
    assert len(evidence["point_samples"]) == 64
    json.dumps(evidence, allow_nan=False)


def test_cylinder_agreement_and_annulus_hole_contours():
    cylinder = trimesh.creation.cylinder(radius=0.3, height=1, sections=48)
    candidates = build_evidence(cylinder, SHA)["cylinders"]
    assert len(candidates) == 1
    assert candidates[0]["axis"] == [0, 0, 1]
    assert candidates[0]["radius"] == pytest.approx(0.3, abs=0.001)
    assert candidates[0]["support_sections"] == 3
    annulus = trimesh.creation.annulus(r_min=0.15, r_max=0.4, height=1, sections=48)
    evidence = build_evidence(annulus, SHA)
    z_sections = [s for s in evidence["sections"] if s["axis"] == "Z"]
    assert all(len(s["contours"]) == 2 for s in z_sections)
    assert all(c["closed"] for s in z_sections for c in s["contours"])
    assert all(s["omitted_contour_count"] == 0 for s in z_sections)
    assert sorted(c["radius"] for c in evidence["cylinders"]) == pytest.approx(
        [0.15, 0.4], abs=0.001
    )


def test_deterministic_and_face_permutation_invariant_samples():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    first = build_evidence(mesh, SHA)
    assert first == build_evidence(mesh, SHA)
    permuted = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[::-1], process=False)
    assert first["point_samples"] == build_evidence(permuted, SHA)["point_samples"]
    samples = np.array([p["xyz"] for p in first["point_samples"]])
    assert np.isfinite(samples).all()
    assert np.linalg.norm(samples, axis=1).min() > 0.97


def test_non_watertight_warns_and_does_not_invent_volume():
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    evidence = build_evidence(mesh, SHA)
    assert any("non-watertight" in warning for warning in evidence["warnings"])
    assert "volume" not in evidence
    json.dumps(evidence, allow_nan=False)


def test_high_resolution_contour_is_explicitly_truncated():
    mesh = trimesh.creation.cylinder(radius=0.3, height=1, sections=300)
    evidence = build_evidence(mesh, SHA)
    assert any(s["truncated"] for s in evidence["sections"])
    assert all(len(c["points"]) <= 128 for s in evidence["sections"] for c in s["contours"])
    assert any("limits reached" in warning for warning in evidence["warnings"])
    assert len(json.dumps(evidence)) < 1024 * 1024


def test_prepare_worker_cache_and_stale_provenance(tmp_path, monkeypatch):
    source, output = tmp_path / "target.stl", tmp_path / "evidence.json"
    trimesh.creation.box().export(source)
    evidence = prepare_evidence(source, output)
    assert evidence["target_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    import api.hybrid as hybrid

    original = hybrid.subprocess.Popen
    monkeypatch.setattr(hybrid.subprocess, "Popen", lambda *a, **k: pytest.fail("Cache missed"))
    assert prepare_evidence(source, output) == evidence
    monkeypatch.setattr(hybrid.subprocess, "Popen", original)
    trimesh.creation.box(extents=[1, 0.5, 0.25]).export(source)
    updated = prepare_evidence(source, output)
    assert updated["target_sha256"] != evidence["target_sha256"]
    assert updated["extents"] == [1, 0.5, 0.25]
    output.write_text('{"schema_version":"hybrid-v1","target_sha256":"wrong"}')
    assert prepare_evidence(source, output) == updated


def test_cancellation_and_timeout_do_not_accept_stale_output(tmp_path):
    source, output = tmp_path / "target.stl", tmp_path / "evidence.json"
    trimesh.creation.box().export(source)
    output.write_text("stale")
    with pytest.raises(HybridCancelled):
        prepare_evidence(source, output, cancel_check=lambda: True)
    calls = 0

    def cancel_running():
        nonlocal calls
        calls += 1
        return calls > 2

    with pytest.raises(HybridCancelled):
        prepare_evidence(source, output, cancel_check=cancel_running)
    with pytest.raises(ValueError, match="timed out"):
        prepare_evidence(source, output, timeout=0.001)
    assert output.read_text() == "stale"
    assert not list(tmp_path.glob("hybrid-*"))


def test_rejects_invalid_provenance_and_nonfinite_mesh():
    with pytest.raises(ValueError, match="SHA256"):
        build_evidence(trimesh.creation.box(), "wrong")
    mesh = trimesh.creation.box()
    mesh.vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_evidence(mesh, SHA)


def test_sphere_does_not_become_a_cylinder_from_equal_endpoint_radii():
    evidence = build_evidence(trimesh.creation.icosphere(subdivisions=3, radius=0.5), SHA)
    assert evidence["cylinders"] == []


def test_dense_multiple_loops_obey_prompt_budget_and_report_omissions():
    parts = []
    for x in range(5):
        for y in range(5):
            part = trimesh.creation.annulus(r_min=0.025, r_max=0.07, height=1, sections=48)
            part.apply_translation([x * 0.2 - 0.4, y * 0.2 - 0.4, 0])
            parts.append(part)
    mesh = trimesh.util.concatenate(parts)
    first = build_evidence(mesh, SHA)
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert len(encoded) <= 32768
    assert any("budget reached" in warning for warning in first["warnings"])
    assert any(s["omitted_contour_count"] > 0 for s in first["sections"])
    assert all(s["truncated"] for s in first["sections"] if s["omitted_contour_count"])
    for section in first["sections"]:
        for contour in section["contours"]:
            if contour["source_point_count"] > len(contour["points"]):
                assert contour["truncated"]
    assert first == build_evidence(mesh, SHA)


def test_rejects_symlink_paths(tmp_path):
    source, output = tmp_path / "target.stl", tmp_path / "evidence.json"
    trimesh.creation.box().export(source)
    linked = tmp_path / "linked.stl"
    linked.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        prepare_evidence(linked, output)
    output.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        prepare_evidence(source, output)
    output.unlink()
    folder_link = tmp_path / "linked-folder"
    folder_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_evidence(source, folder_link / "evidence.json")
