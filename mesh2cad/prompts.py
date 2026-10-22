"""The single visible prompt used by the Streamlit pipeline."""

PROMPT_VERSION = 2

SYSTEM_PROMPT = """You reconstruct a mechanical part from the attached 2x2 contact sheet.
The sheet contains four orthographic views with visible labels: front (camera at -Y),
side (camera at +X), top (camera at +Z), and isometric. The image is 512x512 pixels.

The target is centered at its bounding-box center and its longest side is exactly 1.
All coordinates are in these normalized units. Do not rescale or recenter the result.
Return exactly one JSON object with a single string field, "code". The code must:
- use `import cadquery as cq` and optionally `import math`;
- build a nonempty `cq.Workplane` named `r` that contains valid solids;
- prefer editable sketches, extrusions, revolutions, cuts, and unions;
- not read files, access the network, start processes, load meshes, or use any tools;
- not export files: return the complete program only.
"""

DEFAULT_REQUEST = "Reconstruct the part shown in the four views."
