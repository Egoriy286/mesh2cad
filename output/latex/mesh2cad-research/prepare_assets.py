"""Copy actual experiment images and derive slide numbers from durable checkpoints.

Run with a Python environment containing Pillow. No LLM calls or geometry fitting.
The TeX project already contains its generated assets and compiles independently.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[2]
RUNS = REPO / "runs/harness"
CLASSIC = "20261022T184742Z-b4c867e2"
DEEP = "20261022T184705Z-7cef923e"
INTERIOR = "20261022T185141Z-59366e3d"
DEEP_ID = "274a034e5c4ac827"
INTERIOR_ID = "indoorcad_d8441ba72f8641"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item(state, identifier):
    return next(v for v in state["items"] if v["id"] == identifier)


def calls(obj):
    return [c for attempt in obj["attempts"] for c in attempt.get("calls", [])]


def best(obj):
    return obj["attempts"][obj["best_attempt"]]


def tile(source, destination, index):
    with Image.open(source) as image:
        width, height = image.size
        assert width % 3 == 0 and height % 2 == 0
        size = width // 3
        assert height == size * 2
        col, row = index % 3, index // 3
        image.crop((col * size, row * size, (col + 1) * size, (row + 1) * size)).save(
            PROJECT / "images" / destination
        )
    crops.append(
        {
            "output": "images/" + destination,
            "source": str(source.relative_to(REPO)),
            "source_sha256": sha(source),
            "tile_index_zero_based": index,
            "box_pixels": [col * size, row * size, (col + 1) * size, (row + 1) * size],
        }
    )


classic, deep, interior = [read(RUNS / run / "run.json") for run in (CLASSIC, DEEP, INTERIOR)]
c, d, i = item(classic, DEEP_ID), item(deep, DEEP_ID), item(interior, INTERIOR_ID)
assert all(s["status"] == "completed" for s in (classic, deep, interior))
assert deep["config"]["orchestrator"] == interior["config"]["orchestrator"] == "vision2"
(PROJECT / "images").mkdir(exist_ok=True)
crops = []
interior_target = RUNS / INTERIOR / "items" / INTERIOR_ID / "vision"
deep_target = RUNS / DEEP / "items" / DEEP_ID / "vision"
for modality in ("rgb", "depth", "blueprint"):
    tile(interior_target / f"{modality}.png", f"interior-target-{modality}.png", 3)
tile(RUNS / INTERIOR / best(i)["path"] / "difference-scheme.png", "interior-difference.png", 3)
tile(deep_target / "rgb.png", "deepcad-target.png", 3)
tile(RUNS / DEEP / best(d)["path"] / "vision/rgb.png", "deepcad-v2.png", 3)
tile(interior_target / "rgb.png", "interior-target-right.png", 2)
for number, name in ((0, "first"), (1, "second")):
    tile(
        RUNS / INTERIOR / i["attempts"][number]["path"] / "vision/rgb.png",
        f"interior-{name}-right.png",
        2,
    )

values = {
    "ClassicIoU": f"{best(c)['iou']['iou']:.4f}",
    "DeepIoU": f"{best(d)['iou']['iou']:.4f}",
    "ClassicCost": f"{sum(v['usage']['cost'] for v in calls(c)):.5f}",
    "DeepCost": f"{deep['summary']['cost_usd']:.5f}",
    "DeepSeconds": f"{deep['active_seconds']:.1f}",
    "DeepMemory": f"{read(RUNS / DEEP / 'memory.json')['peak_rss_mb']:.0f}",
    "InteriorFirst": f"{i['attempts'][0]['iou']['iou']:.4f}",
    "InteriorBest": f"{best(i)['iou']['iou']:.4f}",
    "InteriorGain": f"+{best(i)['iou']['iou'] - i['attempts'][0]['iou']['iou']:.4f}",
    "InteriorCost": f"{interior['summary']['cost_usd']:.5f}",
    "InteriorSeconds": f"{interior['active_seconds']:.1f}",
    "InteriorMemory": f"{read(RUNS / INTERIOR / 'memory.json')['peak_rss_mb']:.0f}",
}
(PROJECT / "results.tex").write_text(
    "% Generated from saved run.json checkpoints by prepare_assets.py.\n"
    + "\n".join("\\newcommand{\\" + name + "}{" + value + "}" for name, value in values.items())
    + "\n"
)

records = []
for run, state, obj in ((CLASSIC, classic, c), (DEEP, deep, d), (INTERIOR, interior, i)):
    records.append(
        {
            "run": run,
            "checkpoint_sha256": sha(RUNS / run / "run.json"),
            "item_id": obj["id"],
            "input_sha256": obj["input_sha256"],
            "model": state["config"]["model"],
            "reasoning_effort": state["config"]["reasoning_effort"],
            "max_iterations": state["config"]["max_iterations"],
            "max_output_tokens": state["config"]["max_tokens"],
            "total_token_budget": state["config"]["total_token_budget"],
            "orchestrator": state["config"].get("orchestrator", "classic"),
            "prompt_version": state["prompt_version"],
            "vision_prompt_version": state.get("vision_prompt_version"),
            "target_calibration": read(RUNS / run / "items" / obj["id"] / "vision/calibration.json")
            if (RUNS / run / "items" / obj["id"] / "vision/calibration.json").is_file()
            else None,
            "best_iou": best(obj)["iou"]["iou"],
            "item_calls": calls(obj),
            "item_cost_usd": sum(v["usage"]["cost"] for v in calls(obj)),
            "run_active_seconds": state["active_seconds"],
            "run_item_count": len(state["items"]),
            "run_peak_rss_mb": read(RUNS / run / "memory.json").get("peak_rss_mb"),
            "attempts": [
                {
                    "iteration": a["iteration"],
                    "status": a["status"],
                    "iou": a.get("iou"),
                    "code_sha256": hashlib.sha256(a.get("code", "").encode()).hexdigest(),
                    "visual_metrics": a.get("visual_metrics"),
                }
                for a in obj["attempts"]
            ],
        }
    )

source_files = [
    "mesh2cad/harness.py", "mesh2cad/harness2.py", "mesh2cad/agent.py",
    "mesh2cad/vision.py", "mesh2cad/geometry.py", "mesh2cad/executor.py",
    "mesh2cad/resources.py", "mesh2cad/harness_ui.py", "mesh2cad/observability.py",
    "mesh2cad/dataset.py", "mesh2cad/prompts.py", "README.md",
    "data/deepcad/manifest.json", "data/interior300/manifest.json",
]
evidence = {
    "as_of": "2026-10-22",
    "repository": {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "working_tree_has_uncommitted_implementation": True,
        "file_sha256": {name: sha(REPO / name) for name in source_files},
    },
    "interpretation": {
        "unit_of_analysis": "selected object; not dataset mean",
        "controlled_ablation": False,
        "classic_cost_scope": "selected DeepCAD item only, all five calls",
        "v2_cost_scope": "all calls including geometry analysis and failed generation",
        "numeric_parameter_optimization": False,
        "target_normalization": "bbox centered; longest extent 1",
        "metric_alignment": "no candidate pose or scale fitting",
        "v2_protocol_changed_after_deepcad": "repair prompt explicitly requires executable code",
        "v2_renderer_changed_between_pilots": "DeepCAD calibration v1; Interior calibration v2 adds bbox-fitting frame and dominant depth levels",
        "figures": "unmodified crops of saved renders, no regenerated geometry",
    },
    "runs": records,
    "figure_sources": crops,
    "slide_macros": values,
}
(PROJECT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
print("Prepared", len(crops), "actual render crops and", len(values), "verified slide values.")
