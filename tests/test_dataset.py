import pytest
import trimesh

from mesh2cad import dataset
from mesh2cad.storage import read_json


@pytest.fixture
def local_stls(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(5):
        mesh = trimesh.creation.box(extents=(index + 1, 0.5, 0.25))
        mesh.apply_translation([2, 3, 4])
        mesh.export(source / f"{index}.stl")

    def cheap_render(mesh, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"cached-image")

    monkeypatch.setattr(dataset, "render_views", cheap_render)
    return source


def test_prepare_disjoint_repeatable_and_cache_integrity(tmp_path, local_stls):
    first = dataset.prepare_dataset(
        tmp_path / "one", source_dir=local_stls, dev_count=2, test_count=3
    )
    second = dataset.prepare_dataset(
        tmp_path / "two", source_dir=local_stls, dev_count=2, test_count=3
    )
    a, b = read_json(first), read_json(second)
    assert [(item["id"], item["split"]) for item in a["items"]] == [
        (item["id"], item["split"]) for item in b["items"]
    ]
    assert len({item["id"] for item in a["items"]}) == 5
    assert sum(item["split"] == "dev" for item in a["items"]) == 2
    assert all(item["status"] == "ready" for item in a["items"])
    mesh = trimesh.load_mesh(first.parent / a["items"][0]["mesh_path"])
    assert mesh.extents.max() == pytest.approx(1)
    assert mesh.bounds.mean(axis=0) == pytest.approx([0, 0, 0])
    assert (
        dataset.prepare_dataset(first.parent, source_dir=local_stls, dev_count=2, test_count=3)
        == first
    )
    (first.parent / a["items"][0]["mesh_path"]).write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        dataset.load_manifest(first)


def test_prepare_error_retains_selected_id(tmp_path, local_stls):
    (local_stls / "0.stl").write_text("not an STL")
    path = dataset.prepare_dataset(
        tmp_path / "data", source_dir=local_stls, dev_count=2, test_count=3
    )
    manifest = dataset.load_manifest(path)
    assert len(manifest["items"]) == 5
    assert sum(item["status"] == "prepare_error" for item in manifest["items"]) == 1


def test_prepare_rejects_changed_settings(tmp_path, local_stls):
    dataset.prepare_dataset(tmp_path / "data", source_dir=local_stls, dev_count=2, test_count=3)
    with pytest.raises(ValueError, match="different settings"):
        dataset.prepare_dataset(tmp_path / "data", source_dir=local_stls, dev_count=1, test_count=3)


def test_manifest_rejects_path_traversal(tmp_path):
    from mesh2cad.storage import resolve_artifact

    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact(tmp_path, "../outside.stl")
