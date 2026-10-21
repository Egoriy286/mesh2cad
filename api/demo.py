"""Deterministic offline examples; these are not model predictions."""

import trimesh

EXAMPLES = [
    {
        "id": "block",
        "name": "Offline demo · rectangular block",
        "description": "Deterministic box, real CadQuery export and measured evaluation.",
    },
    {
        "id": "cylinder",
        "name": "Offline demo · cylinder",
        "description": "Deterministic cylinder; no network or model calls.",
    },
]


def example_mesh(name):
    if name == "block":
        return trimesh.creation.box(extents=(1, 0.7, 0.4))
    if name == "cylinder":
        return trimesh.creation.cylinder(radius=0.4, height=1, sections=64)
    raise ValueError("Unknown example")


def demo_code(asset, parent=None, edit=False):
    if edit and parent:
        # A deterministic visible operation; never interpret natural language as code.
        return parent["code"] + '\nr = r.edges("|Z").fillet(0.025)\n'
    if asset.get("example_id") == "cylinder":
        return 'import cadquery as cq\nr = cq.Workplane("XY").circle(0.4).extrude(1).translate((0, 0, -0.5))\n'
    ext = asset["stats"]["normalized_extents"]
    return (
        f'import cadquery as cq\nr = cq.Workplane("XY").box({ext[0]!r}, {ext[1]!r}, {ext[2]!r})\n'
    )
