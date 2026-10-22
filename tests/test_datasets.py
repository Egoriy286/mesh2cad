"""Accept native Interior300 without rewriting data or inventing split labels."""

import pytest

from mesh2cad.dataset import load_manifest, read_catalogue
from mesh2cad.harness import HarnessConfig, create_run, results_csv
from mesh2cad.storage import file_digest, read_json, write_json


def test_interior_native_adapter_preserves_metadata_and_original_manifest(
    interior_manifest,
) -> None:
    before = interior_manifest.read_bytes()
    catalogue = load_manifest(interior_manifest)
    assert catalogue["name"] == "Interior300"
    assert catalogue["format"] == "interior300"
    assert {i["split"] for i in catalogue["items"]} == {"all"}
    assert all(i["status"] == "ready" for i in catalogue["items"])
    assert all(i["mesh_sha256"] == i["sha256"] for i in catalogue["items"])
    assert catalogue["items"][0]["license"] == "fixture"
    assert catalogue["items"][0]["revision"] == "fixture-revision"
    assert catalogue["source"]["sources"] == ["indoorcad", "polyhaven"]
    assert interior_manifest.read_bytes() == before


def test_interior_snapshot_is_usable_without_image_cache(interior_manifest, tmp_path) -> None:
    root = create_run(interior_manifest, tmp_path / "runs", HarnessConfig(), limit=1)
    state = read_json(root / "run.json")
    item = state["items"][0]
    assert state["dataset"]["name"] == "Interior300"
    assert state["dataset"]["split"] == "all"
    assert item["category"] == "cabinet"
    assert item["source"] == "indoorcad"
    assert file_digest(root / item["input_path"]) == item["input_sha256"]
    assert "indoorcad_box-model_01" in results_csv(state)
    assert "category" in results_csv(state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "../escape"),
        ("mesh_path", "../../outside.stl"),
        ("sha256", "bad-hash"),
    ],
)
def test_native_manifest_checks_paths_ids_and_hashes(interior_manifest, field, value) -> None:
    manifest = read_json(interior_manifest)
    manifest["items"][0][field] = value
    write_json(interior_manifest, manifest)
    with pytest.raises(ValueError):
        read_catalogue(interior_manifest)


def test_native_manifest_detects_modified_stl(interior_manifest) -> None:
    catalogue = read_catalogue(interior_manifest)
    (interior_manifest.parent / catalogue["items"][0]["mesh_path"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed input artifact"):
        load_manifest(interior_manifest)


def test_normalized_interior_snapshot_retains_its_format(interior_manifest, tmp_path) -> None:
    normalized = read_catalogue(interior_manifest)
    path = tmp_path / "manifest.json"
    write_json(path, normalized)
    assert read_catalogue(path)["format"] == "interior300"
