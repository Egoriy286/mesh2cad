"""Versioned prompts: changing these invalidates experiment resume."""

PROMPT_VERSION = 1

BASE_PROMPT = """Reconstruct the mechanical part in the attached four-view image as a compact
parametric CadQuery program. Return only the requested JSON object with one field, code.
The views are front (camera at -Y), side (camera at +X), top (camera at +Z), and isometric;
use the view labels in the image. The target is centered at its bounding-box
center and its longest dimension is exactly 1. Coordinates are in normalized units.
Do not rescale or recenter your result after construction.
Use import cadquery as cq and optionally import math; no other imports, file reads,
network, processes, mesh loading, or external assets. The final object must be a
nonempty cq.Workplane named r, containing valid solids. Prefer editable sketches,
extrusions, revolutions, cuts and unions. Do not export files; the host exports r.
Do not use any tools, shell, browser, other agents or filesystem access.
"""


def initial_prompt() -> str:
    return BASE_PROMPT


def repair_prompt(code: str, *, error: str | None, metrics: dict | None) -> str:
    if error:
        feedback = "The previous program failed. Fix the following error:\n" + error[-6000:]
    else:
        feedback = (
            "The attached comparison shows target and current reconstruction with the same "
            "cameras and coordinates. Correct missing/excess geometry and dimensions. "
            f"Measured symmetric squared Chamfer distance (lower is better): {metrics['chamfer']:.8g}. "
            f"Volumetric IoU: {metrics.get('iou')} ({metrics.get('iou_status')})."
        )
    return (
        BASE_PROMPT
        + "\n"
        + feedback
        + "\nReturn the complete corrected program.\nPrevious code:\n```python\n"
        + code
        + "\n```\n"
    )
