"""Render a lightweight visual index of a downloaded interior STL collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


def thumbnail(path: Path, size: int = 240, alignment: dict | None = None) -> Image.Image:
    mesh = trimesh.load_mesh(path)
    if alignment and alignment.get("status") == "verified":
        angles = np.radians(alignment.get("align_rotation", [0, 0, 0]))
        mesh.apply_transform(trimesh.transformations.euler_matrix(*angles))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(alignment.get("extra_rz", 0)), [0, 0, 1]
        ))
    vertices = mesh.vertices - mesh.bounds.mean(axis=0)
    vertices /= max(mesh.extents)
    forward = np.array([1.2, -1.7, 1.1])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    xy = np.column_stack([vertices @ right, -(vertices @ up)])
    xy -= (xy.max(axis=0) + xy.min(axis=0)) / 2
    xy = xy * ((size - 28) / np.ptp(xy, axis=0).max()) + size / 2
    triangles = xy[mesh.faces]
    order = np.argsort((vertices @ forward)[mesh.faces].mean(axis=1))
    normals = mesh.face_normals
    light = np.array([0.3, -0.4, 0.85])
    brightness = 0.4 + 0.6 * np.abs(normals @ light)
    image = Image.new("RGB", (size, size), "#edf1f5")
    draw = ImageDraw.Draw(image)
    for index in order:
        value = float(brightness[index])
        color = (int(115 * value), int(160 * value), int(205 * value))
        draw.polygon([tuple(v) for v in triangles[index]], fill=color)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/interior300"))
    args = parser.parse_args()
    root = args.dataset.resolve()
    items = json.loads((root / "manifest.json").read_text())["items"]
    previews = root / "previews"
    previews.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for index, item in enumerate(items):
        target = previews / (item["id"] + "_aligned.jpg")
        if not target.exists():
            alignment_path = item.get("sidecars", {}).get("alignment_meta.json")
            alignment = json.loads((root / alignment_path).read_text()) if alignment_path else None
            thumbnail(root / item["mesh_path"], alignment=alignment).save(target, quality=85)
        if index % 25 == 0:
            print(f"previews {index + 1}/{len(items)}", flush=True)
    # Index pages include all items in manifest order, 25 per page.
    for start in range(0, len(items), 25):
        sheet = Image.new("RGB", (1200, 1450), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, item in enumerate(items[start:start + 25]):
            x, y = offset % 5 * 240, offset // 5 * 290
            sheet.paste(Image.open(previews / (item["id"] + "_aligned.jpg")), (x, y))
            name = item["name"]
            label = f"{start + offset + 1:03d}  {item['category']}\n{name[:30]}\n{item['source']}"
            draw.multiline_text((x + 6, y + 242), label, font=font, fill="#202c3c")
        sheet.save(previews / f"index_{start // 25 + 1:02d}.jpg", quality=90)
    print(f"Saved {len(items)} previews and {(len(items) + 24) // 25} index pages", flush=True)


if __name__ == "__main__":
    main()
