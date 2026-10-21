"""Bounded offline surface importer. The parser runs outside the API process."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import trimesh

from mesh2cad.geometry import _validate_mesh, normalize_mesh

MAX_BYTES = 25 * 1024 * 1024
MAX_FACES = 500_000
MAX_VERTICES = 1_500_000
FORMATS = {"stl", "obj", "ply", "off", "glb"}


def parse(path, fmt, output):
    data = Path(path).read_bytes()
    if len(data) > MAX_BYTES:
        raise ValueError("File exceeds 25 MiB import limit")
    # Refuse unreasonable declared allocations before the parser can allocate
    # based on an untrusted count. Binary STL has fixed 50-byte face records.
    if fmt == "stl" and len(data) >= 84:
        count = int.from_bytes(data[80:84], "little")
        if len(data) == 84 + count * 50 and count > MAX_FACES:
            raise ValueError("STL exceeds triangle limit")
    if fmt == "ply":
        end = data.find(b"end_header")
        if end < 0 or end > 65536:
            raise ValueError("Invalid or oversized PLY header")
        for line in data[:end].decode("ascii").splitlines():
            fields = line.split()
            if (
                len(fields) == 3
                and fields[:2] == ["element", "vertex"]
                and int(fields[2]) > MAX_VERTICES
            ):
                raise ValueError("PLY exceeds vertex limit")
            if (
                len(fields) == 3
                and fields[:2] == ["element", "face"]
                and int(fields[2]) > MAX_FACES
            ):
                raise ValueError("PLY exceeds triangle limit")
    if fmt == "off":
        lines = [
            line.strip()
            for line in data.decode("utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        counts = (lines[0][3:].strip() or lines[1]).split()
        if int(counts[0]) > MAX_VERTICES or int(counts[1]) > MAX_FACES:
            raise ValueError("OFF exceeds mesh element limit")
    if fmt == "obj":
        # Materials are irrelevant to triangle geometry. Explicitly prohibit all
        # resource references instead of allowing a resolver to touch the host.
        text = data.decode("utf-8")
        lines = text.splitlines()
        if (
            sum(line.lstrip().startswith("v ") for line in lines) > MAX_VERTICES
            or sum(line.lstrip().startswith("f ") for line in lines) > MAX_FACES
        ):
            raise ValueError("OBJ exceeds mesh element limit")
        if any(line.split(maxsplit=1)[0] in {"mtllib", "call"} for line in lines if line.split()):
            raise ValueError("External OBJ materials/resources are disabled; remove mtllib")
    if fmt == "glb":
        if data[:4] != b"glTF" or len(data) < 20:
            raise ValueError("Invalid GLB header")
        length = int.from_bytes(data[12:16], "little")
        doc = json.loads(data[20 : 20 + length])
        nodes = doc.get("nodes", [])
        if len(nodes) > 10000 or len(doc.get("accessors", [])) > 10000:
            raise ValueError("GLB exceeds scene element limit")
        accessors = doc.get("accessors", [])
        parents = set()
        for node in nodes:
            for child in node.get("children", []):
                if (
                    not isinstance(child, int)
                    or child < 0
                    or child >= len(nodes)
                    or child in parents
                ):
                    raise ValueError("GLB must have a valid tree of scene nodes")
                parents.add(child)
        visited = set()

        def visit(node_index, chain):
            if node_index in chain or len(chain) > 64:
                raise ValueError("GLB scene has a cycle or exceeds depth limit")
            if node_index in visited:
                return
            for child in nodes[node_index].get("children", []):
                visit(child, chain | {node_index})
            visited.add(node_index)

        for node_index in range(len(nodes)):
            visit(node_index, set())
        if sum(accessor.get("count", 0) for accessor in accessors) > MAX_VERTICES * 12:
            raise ValueError("GLB total accessor allocation exceeds limit")
        for accessor in doc.get("accessors", []):
            if accessor.get("count", 0) > MAX_VERTICES * 3:
                raise ValueError("GLB accessor exceeds mesh element limit")
        meshes = doc.get("meshes", [])
        total_vertices, total_faces = 0, 0
        for node in nodes:
            mesh_index = node.get("mesh")
            if mesh_index is None:
                continue
            if not isinstance(mesh_index, int) or mesh_index < 0 or mesh_index >= len(meshes):
                raise ValueError("Invalid GLB mesh index")
            for primitive in meshes[mesh_index].get("primitives", []):
                positions = primitive.get("attributes", {}).get("POSITION")
                if positions is None or positions < 0 or positions >= len(accessors):
                    raise ValueError("Invalid GLB position accessor")
                vertices = accessors[positions].get("count", 0)
                indices = primitive.get("indices")
                index_count = (
                    accessors[indices].get("count", 0) if indices is not None else vertices
                )
                total_vertices += vertices
                total_faces += index_count // 3
        if total_vertices > MAX_VERTICES or total_faces > MAX_FACES:
            raise ValueError("GLB scene instance expansion exceeds mesh element limit")
        primitive_count = sum(len(mesh.get("primitives", [])) for mesh in meshes)
        if primitive_count * max(1, len(nodes)) > 100000:
            raise ValueError("GLB scene instance expansion exceeds limit")
        for buffer in doc.get("buffers", []):
            if buffer.get("byteLength", 0) > MAX_BYTES:
                raise ValueError("GLB buffer exceeds byte limit")
        for item in doc.get("buffers", []) + doc.get("images", []):
            if "uri" in item and not item["uri"].startswith("data:"):
                raise ValueError("GLB external resources are disabled")
    loaded = trimesh.load(
        io.BytesIO(data), file_type=fmt, process=False, resolver={}, skip_materials=True
    )
    if isinstance(loaded, trimesh.Scene):
        parts = [part for part in loaded.dump() if isinstance(part, trimesh.Trimesh)]
        if not parts:
            raise ValueError("Point-only data has no triangular surface")
        loaded = trimesh.util.concatenate(parts)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("Point-only data has no triangular surface")
    if len(loaded.faces) > MAX_FACES or len(loaded.vertices) > MAX_VERTICES:
        raise ValueError("Mesh exceeds 500,000 triangles / 1,500,000 vertices")
    _validate_mesh(loaded)
    loaded.remove_unreferenced_vertices()
    loaded.merge_vertices()
    normalized, transform = normalize_mesh(loaded)
    out = Path(output)
    normalized.export(out / "target.stl")
    metadata = {
        "normalization": {**transform, "units": "unknown", "frame_id": "normalized-target"},
        "stats": {
            "vertices": len(loaded.vertices),
            "faces": len(loaded.faces),
            "watertight": bool(loaded.is_watertight),
            "source_bounds": loaded.bounds.tolist(),
            "normalized_extents": normalized.extents.tolist(),
            "normalized_bounds": normalized.bounds.tolist(),
            "units": "unknown",
        },
    }
    (out / "import.json").write_text(json.dumps(metadata))


def import_mesh(path, fmt, output):
    if fmt not in FORMATS:
        raise ValueError("Supported formats: STL, OBJ, PLY, OFF, GLB")
    command = [sys.executable, "-m", "api.importer", str(path), fmt, str(output)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=25,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(output),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            cwd=Path(__file__).resolve().parents[1],
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Mesh import exceeded 25 second limit") from exc
    if result.returncode:
        reason = result.stderr.decode(errors="replace").strip().splitlines()
        raise ValueError(reason[-1] if reason else "Mesh parser failed within resource limits")
    return json.loads((Path(output) / "import.json").read_text())


if __name__ == "__main__":
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    # macOS RLIMIT_AS is unsupported; the byte/element limits and subprocess
    # timeout are enforced on every platform. Linux also caps virtual memory.
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    parse(*sys.argv[1:])
