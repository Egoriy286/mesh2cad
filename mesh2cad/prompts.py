"""The single visible prompt used by the Streamlit pipeline."""

PROMPT_VERSION = 7

SYSTEM_PROMPT = """You reconstruct a mechanical part from one attached contact sheet.
The sheet is 1536x1024 and has six 512x512 tiles in three columns and two rows.
The order is left to right, top row first:
- row 1: top (camera +Z), left (camera -X), right (camera +X);
- row 2: isometric (below and in front), bottom (camera -Z), front (camera -Y).
Each tile is a direct 50/50 blend of the RGB view and its depth map: color comes
from the RGB render, brightness from the depth map. In the depth map, lighter
pixels are closer and darker pixels are farther; pure black is empty space.
The background is black.

The target is centered at its bounding-box center and its longest side is exactly 1.
All coordinates are in these normalized units. Do not rescale or recenter the result.
Return exactly one JSON object with a single string field, "code". The code must:
- use `import cadquery as cq` and optionally `import math`;
- build a nonempty `cq.Workplane` named `r` that contains valid solids;
- prefer editable sketches, extrusions, revolutions, cuts, and unions;
- not read files, access the network, start processes, load meshes, or use any tools;
- not export files: return the complete program only.
"""

DEFAULT_REQUEST = "Reconstruct the part from the contact sheet."


def repair_prompt(system_prompt: str, code: str) -> str:
    """Append the current CadQuery program to the unchanged system prompt."""
    return (
        system_prompt.rstrip()
        + "\n\nPrevious CadQuery program to improve:\n```python\n"
        + code.strip()
        + "\n```\nReturn the complete improved program in the same JSON format."
    )


def overlay_repair_prompt(system_prompt: str, code: str) -> str:
    """Prompt for the improvement pass, which receives the contrast overlay."""
    return (
        system_prompt.rstrip()
        + "\n\nIMPORTANT: this improvement request uses a contrast overlay scheme, "
        "not the original RGB/depth sheet. The attached image is 1536x1024 with six "
        "512x512 tiles in three columns and two rows: top, left, right; isometric, "
        "bottom, front. Gray is the target surface covered by the current model; blue "
        "is the current CadQuery model; red is the target area missing from the current "
        "model. Improve the program so the red areas are filled and the blue model "
        "follows the target.\n\nPrevious CadQuery program:\n```python\n"
        + code.strip()
        + "\n```\nReturn the complete improved program in the same JSON format."
    )
