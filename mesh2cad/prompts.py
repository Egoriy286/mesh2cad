"""Explicit prompt contracts for classic reconstruction and calibrated vision."""

PROMPT_VERSION = 7
VISION_PROMPT_VERSION = 2

LEGACY_VISION_SYSTEM_PROMPT = """You reconstruct editable solid CAD from separate, registered RGB,
linear depth and engineering blueprint sheets. Never infer shape from image color alone.
All sheets use the SAME six orthographic cameras, canvas and normalized world frame.
The tile order is: top, left, right / isometric, bottom, front. Camera metadata gives
image_right, image_up and forward in WORLD XYZ: use these signs, not object labels.
Top is +Z, left -X, right +X, bottom -Z, front -Y. The isometric camera looks
from below/front. Do not confuse an object's semantic front with world -Y.

RGB shows appearance and boundaries. Depth is separate GRAYSCALE, LINEAR, not RGB
shading: 0 means EMPTY, 1..255 means visible surface, brighter is nearer. For a
foreground pixel g, raw_depth = far - (g-1)/254 * (far-near); this equals
dot(world_point, forward). Each view has its own documented near/far planes;
these planes are fixed between target and prediction. Compare the SAME view only.
Calibration also lists measured visible depth ranges and dominant surface levels
(rounded to 0.001 world units); use these to constrain planar walls and step heights.
The blueprint shows silhouettes and depth discontinuities on a grid with 0.1-unit
spacing, projected span dimensions, and WORLD XYZ arrows. It is evidence, not a
suggested solid: preserve visible openings and empty space.

The target bounding box is centered at zero, longest extent exactly 1. The supplied
bounds and x/y/z extents are hard dimensional constraints. Do not rotate/recenter
the result globally. A 0..height extrusion must be translated to -height/2..+height/2.
Use measured target sections to distinguish panels, rods, annuli and holes. Their
closed contours may include interior holes; a filled outer contour is not enough.

First reconstruct large masses and empty regions, then supports and openings,
then decoration. Cross-check every dimension against two orthographic views and
depth. A nearly correct silhouette can still have wrong thickness and low volume IoU.
Prefer a compact, editable construction with explicit named dimensions. Touching
parts intended to fuse must have positive overlap; avoid tangent-only contacts.
Return exactly JSON with a string field "code": complete Python using import
cadquery as cq and optionally import math, producing valid solids in cq.Workplane r.
No file reads, mesh imports, network, subprocesses, tools or exports in the program.
No numerical fitting is performed after your answer: your code must directly create
the correct geometry in the target frame.
"""

VISION_SYSTEM_PROMPT = """You reconstruct editable solid CAD from calibrated linear depth maps only.
The six views use the SAME cameras, canvas and normalized world frame as the candidate.
The tile order is: top, left, right / isometric, bottom, front. Camera metadata gives
image_right, image_up and forward in WORLD XYZ: use these signs, not object labels.
Top is +Z, left -X, right +X, bottom -Z, front -Y. The isometric camera looks
from below/front. Do not confuse an object's semantic front with world -Y.

Depth is GRAYSCALE and LINEAR: 0 means EMPTY, 1..255 means visible surface, brighter is nearer. For a
foreground pixel g, raw_depth = far - (g-1)/254 * (far-near); this equals
dot(world_point, forward). Each view has its own documented near/far planes;
these planes are fixed between target and prediction. Compare the SAME view only.
Calibration also lists measured visible depth ranges and dominant surface levels
(rounded to 0.001 world units); use these to constrain planar walls and step heights.
Infer silhouettes and depth discontinuities from the depth maps; preserve visible
openings and empty space. The calibration supplies world axes and dimensions.

The target bounding box is centered at zero, longest extent exactly 1. The supplied
bounds and x/y/z extents are hard dimensional constraints. Do not rotate/recenter
the result globally. A 0..height extrusion must be translated to -height/2..+height/2.
Use measured target sections to distinguish panels, rods, annuli and holes. Their
closed contours may include interior holes; a filled outer contour is not enough.

First reconstruct large masses and empty regions, then supports and openings,
then decoration. Cross-check every dimension against two orthographic views and
depth. A nearly correct silhouette can still have wrong thickness and low volume IoU.
Prefer a compact, editable construction with explicit named dimensions. Touching
parts intended to fuse must have positive overlap; avoid tangent-only contacts.
Return exactly JSON with a string field "code": complete Python using import
cadquery as cq and optionally import math, producing valid solids in cq.Workplane r.
No file reads, mesh imports, network, subprocesses, tools or exports in the program.
No numerical fitting is performed after your answer: your code must directly create
the correct geometry in the target frame.
"""

VISION_ANALYSIS_PROMPT = """Analyze the calibrated target linear depth maps
before CAD construction. Use the camera/depth calibration and measured sections.
Return one compact JSON object with keys:
coordinate_checks: object explaining the sign/direction of X,Y,Z in the views;
components: array (at most 20) of named main solids, each with bounds [[xmin,ymin,zmin],
[xmax,ymax,zmax]], primitive/operation and dimensions in normalized units;
openings: array of holes/empty regions and their axes/bounds;
construction_plan: ordered list of CadQuery operations;
uncertainties: list of ambiguous dimensions and the views that constrain them.
Use depth to verify thickness and side placement. Account
for the supplied volume and section contours. Omit code and unnecessary prose.
"""

VISION_ANALYSIS_SYSTEM_PROMPT = (
    VISION_SYSTEM_PROMPT.split("Return exactly JSON")[0] + "\n" + VISION_ANALYSIS_PROMPT
)

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
        "bottom, front. Green is the volume intersection of target and CAD; blue "
        "is excess CAD outside the target mesh (CAD minus target); red is the "
        "missing CAD volume (target minus CAD). Restore red geometry and remove "
        "blue geometry.\n\nPrevious CadQuery program:\n```python\n"
        + code.strip()
        + "\n```\nReturn the complete improved program in the same JSON format."
    )
