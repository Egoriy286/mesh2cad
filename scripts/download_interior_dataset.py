"""Download a bounded, attributed interior mesh collection (no API keys needed)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx
import numpy as np
import trimesh

REPO = "anon-neurips-2026/IndoorCAD"
REVISION = "42ec9cea625996a0f9ff1b0b9bf18207b993ba1c"
CATEGORIES = {
    "Bed": "bed",
    "Cabinet(storage_furniture)": "cabinet",
    "Chair": "chair",
    "Desk": "table",
    "Desktop_object": "desktop_object",
    "Lamp": "lamp",
    "Sofa": "sofa",
    "other_furniture": "other_furniture",
}
CLIENT = httpx.Client(
    follow_redirects=True,
    timeout=90,
    headers={"User-Agent": "msu-ai-school-interior-dataset/1.0"},
    limits=httpx.Limits(max_connections=8),
)


def get(url: str) -> httpx.Response:
    for attempt in range(4):
        try:
            response = CLIENT.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(folder: str) -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{quote(folder, safe='/')}?limit=1000"
    result = []
    while url:
        response = get(url)
        result.extend(response.json())
        url = response.links.get("next", {}).get("url")
    return result


def download(url: str, path: Path, *, md5: str | None = None, size: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size == size):
        if md5 is None or hashlib.md5(path.read_bytes()).hexdigest() == md5:
            return
    data = get(url).content
    if size is not None and len(data) != size:
        raise ValueError(f"Size mismatch for {url}")
    if md5 is not None and hashlib.md5(data).hexdigest() != md5:
        raise ValueError(f"MD5 mismatch for {url}")
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def inspect_mesh(path: Path) -> dict:
    mesh = trimesh.load_mesh(path, process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Empty mesh")
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.extents <= 0):
        raise ValueError("Non-finite or flat mesh")
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "extents": mesh.extents.tolist(),
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def indoor_candidates(root: Path) -> list[dict]:
    cache = root / "metadata" / "indoorcad_candidates.json"
    if cache.exists():
        return json.loads(cache.read_text())
    candidates = []
    for source_category, category in CATEGORIES.items():
        folders = [x for x in tree(source_category) if x["type"] == "directory"]
        random.Random(42).shuffle(folders)
        # Exclude obvious isolated components; upstream labels are still approximate.
        rejected = re.compile(r"headboard|footboard|pillow|cushion|wheel|handle|hinge|screw|bolt|door|полка|ножк", re.I)
        folders = [x for x in folders if not rejected.search(Path(x["path"]).name)]
        candidates.extend(
            {"source": "indoorcad", "category": category, "folder": x["path"]}
            for x in folders
        )
        print(f"catalog {category}: {len(folders)} candidates", flush=True)
    save_json(cache, candidates)
    return candidates


def collect_indoor(root: Path, candidate: dict) -> dict:
    folder = candidate["folder"]
    item_id = "indoorcad_" + hashlib.sha256(folder.encode()).hexdigest()[:14]
    metadata_path = root / "metadata" / f"{item_id}.json"
    if metadata_path.exists():
        item = json.loads(metadata_path.read_text())
        if (root / item["mesh_path"]).exists() and sha256(root / item["mesh_path"]) == item["sha256"]:
            return item
    files = tree(folder)
    stls = [x for x in files if x["path"].lower().endswith(".stl")]
    if len(stls) != 1:
        raise ValueError(f"Expected one STL in {folder}, found {len(stls)}")
    source = stls[0]
    if source["size"] > 10_000_000:
        raise ValueError("Source exceeds 10 MB per-object limit")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(source['path'], safe='/')}"
    path = root / "meshes" / candidate["category"] / f"{item_id}.stl"
    download(url, path, size=source["size"])
    stats = inspect_mesh(path)
    if "lfs" in source and source["lfs"]["oid"] != stats["sha256"]:
        raise ValueError("Upstream SHA-256 mismatch")
    sidecars = {}
    for entry in files:
        name = Path(entry["path"]).name
        if name in {"description.txt", "alignment_meta.json"}:
            sidecar = root / "metadata" / item_id / name
            source_url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(entry['path'], safe='/')}"
            download(source_url, sidecar, size=entry["size"])
            sidecars[name] = sidecar.relative_to(root).as_posix()
    item = {
        **candidate, "id": item_id, "name": Path(folder).name,
        "source_page": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}/{quote(folder, safe='/')}",
        "download_url": url, "revision": REVISION,
        "license": "MIT (declared in upstream dataset card)",
        "license_url": f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
        "mesh_path": path.relative_to(root).as_posix(),
        "changes": "None; original STL bytes preserved", "units": "unspecified by source STL",
        "sidecars": sidecars, **stats,
    }
    save_json(metadata_path, item)
    return item


def poly_candidates(root: Path) -> list[dict]:
    cache = root / "metadata" / "polyhaven_catalog.json"
    if cache.exists():
        assets = json.loads(cache.read_text())
    else:
        assets = get("https://api.polyhaven.com/assets?type=models").json()
        save_json(cache, assets)
    groups = {"Furniture": [], "Lighting": [], "Decor & Art": [], "Food & Kitchen": []}
    for slug, data in sorted(assets.items()):
        source_category = data.get("category", "")
        group = source_category.split("/")[0]
        if group == "Lighting" and any(x in source_category for x in ("Flashlights", "Street & Outdoor")):
            continue
        if group == "Food & Kitchen" and any(x in source_category for x in ("Produce", "Prepared Food", "Packaged & Pantry")):
            continue
        if group in groups:
            groups[group].append({"source": "polyhaven", "slug": slug, "asset": data})
    for values in groups.values():
        random.Random(42).shuffle(values)
    # Round-robin creates diversity even if only a subset is requested.
    return [values[i] for i in range(max(map(len, groups.values()))) for values in groups.values() if i < len(values)]


def collect_poly(root: Path, candidate: dict) -> dict:
    slug = candidate["slug"]
    item_id = "polyhaven_" + slug
    metadata_path = root / "metadata" / f"{item_id}.json"
    if metadata_path.exists():
        item = json.loads(metadata_path.read_text())
        if (root / item["mesh_path"]).exists() and sha256(root / item["mesh_path"]) == item["sha256"]:
            return item
    file_cache = root / "metadata" / f"{item_id}_files.json"
    if file_cache.exists():
        files = json.loads(file_cache.read_text())
    else:
        files = get(f"https://api.polyhaven.com/files/{slug}").json()
        save_json(file_cache, files)
    resolutions = files.get("gltf", {})
    resolution = "1k" if "1k" in resolutions else sorted(resolutions)[0]
    file = resolutions[resolution]["gltf"]
    source_dir = root / "sources" / item_id
    path = source_dir / Path(urlparse(file["url"]).path).name
    download(file["url"], path, md5=file["md5"], size=file["size"])
    geometry_files = []
    # STL has no textures. Download only glTF geometry buffers, never image files.
    gltf = json.loads(path.read_text())
    for buffer in gltf.get("buffers", []):
        uri = buffer["uri"]
        if uri.startswith("data:"):
            continue
        target = (source_dir / uri).resolve()
        if not target.is_relative_to(source_dir.resolve()):
            raise ValueError("Unsafe glTF buffer path")
        dependency = file["include"][uri]
        if dependency["size"] > 40_000_000:
            raise ValueError("Buffer exceeds 40 MB per-object limit")
        download(dependency["url"], target, md5=dependency["md5"], size=dependency["size"])
        geometry_files.append({"path": target.relative_to(root).as_posix(), "url": dependency["url"], "sha256": sha256(target)})
    scene = trimesh.load_scene(path, skip_materials=True)
    mesh = scene.to_mesh()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    group = candidate["asset"]["category"].split("/")[0]
    category = {"Furniture": "furniture", "Lighting": "lamp", "Decor & Art": "decor", "Food & Kitchen": "kitchen"}[group]
    mesh_path = root / "meshes" / category / f"{item_id}.stl"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)
    item = {
        "id": item_id, "source": "polyhaven", "name": candidate["asset"]["name"],
        "category": category, "source_category": candidate["asset"]["category"],
        "authors": candidate["asset"].get("authors", {}),
        "source_page": f"https://polyhaven.com/a/{slug}",
        "download_url": file["url"], "license": "CC0-1.0",
        "license_url": "https://polyhaven.com/license", "units": "metres",
        "mesh_path": mesh_path.relative_to(root).as_posix(),
        "source_path": path.relative_to(root).as_posix(), "source_sha256": sha256(path),
        "geometry_files": geometry_files,
        "changes": "glTF scene transforms applied; Y-up rotated to Z-up; exported binary STL; textures omitted",
        **inspect_mesh(mesh_path),
    }
    save_json(metadata_path, item)
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/interior300"))
    parser.add_argument("--indoor-per-category", type=int, default=30)
    parser.add_argument("--poly-count", type=int, default=60)
    args = parser.parse_args()
    if args.indoor_per_category < 0 or args.poly_count < 0:
        parser.error("Counts must be nonnegative")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    card = root / "metadata" / "IndoorCAD_README.md"
    download(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md", card)
    items, failures, seen = [], [], set()

    def accept(candidate, future):
        try:
            item = future.result()
            if item["sha256"] in seen:
                raise ValueError("Duplicate STL SHA-256")
            seen.add(item["sha256"])
            items.append(item)
            print(f"ready {len(items)}: {item['source']} / {item['category']} / {item['name']}", flush=True)
            save_json(root / "manifest.json", {"items": items, "failures": failures})
            return True
        except Exception as exc:
            failures.append({"candidate": {k: v for k, v in candidate.items() if k != "asset"}, "error": f"{type(exc).__name__}: {exc}"})
            print(f"skip: {exc}", flush=True)
            return False

    def collect_pool(pool, target, collector):
        count, position = 0, 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            while count < target and position < len(pool):
                batch = pool[position:position + min(4, target - count)]
                position += len(batch)
                futures = [executor.submit(collector, root, candidate) for candidate in batch]
                for candidate, future in zip(batch, futures):
                    count += accept(candidate, future)
        return count

    # Four workers; selected file count is fixed even after skips or duplicates.
    candidates = indoor_candidates(root) if args.indoor_per_category else []
    quotas = {
        category: min(args.indoor_per_category, sum(x["category"] == category for x in candidates))
        for category in CATEGORIES.values()
    }
    quotas["cabinet"] += args.indoor_per_category * len(CATEGORIES) - sum(quotas.values())
    for category in CATEGORIES.values():
        pool = [x for x in candidates if x["category"] == category]
        # Leave a small reserve for the tiny lamp category.
        target = quotas[category]
        if category == "lamp" and target > 20:
            quotas["other_furniture"] += target - 20
            target = 20
        count = collect_pool(pool, target, collect_indoor)
        if count != target:
            raise RuntimeError(f"Only {count} usable models in {category}")
    count = collect_pool(poly_candidates(root) if args.poly_count else [], args.poly_count, collect_poly)
    if count != args.poly_count:
        raise RuntimeError(f"Only {count} usable Poly Haven models")
    summary = {
        "count": len(items), "by_source": dict(Counter(x["source"] for x in items)),
        "by_category": dict(Counter(x["category"] for x in items)),
        "watertight": sum(x["watertight"] for x in items),
        "volumes": sum(x["is_volume"] for x in items),
        "stl_bytes": sum(x["bytes"] for x in items),
    }
    save_json(root / "manifest.json", {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": 42, "summary": summary, "items": items, "failures": failures,
    })
    save_json(root / "summary.json", summary)
    with (root / "manifest.csv").open("w", newline="") as stream:
        columns = ["id", "source", "category", "name", "mesh_path", "license", "source_page", "faces", "watertight", "is_volume", "bytes", "sha256"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    # Remove rejected mesh files from the final collection, retaining failure metadata.
    selected = {x["mesh_path"] for x in items}
    for path in (root / "meshes").rglob("*.stl"):
        if path.relative_to(root).as_posix() not in selected:
            path.unlink()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
