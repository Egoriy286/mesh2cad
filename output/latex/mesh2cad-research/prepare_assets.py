"""Derive auditable presentation figures/tables from saved local experiments.

No network, provider calls, CAD generation, fitting, or checkpoint writes. Run with
PROJECT_ROOT/.venv/bin/python. The PDF itself needs only the generated artifacts.
The audit cache is optional after first generation: evidence.json embeds its inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[2]
sys.path.insert(0, str(REPO))

from mesh2cad.geometry import VIEWS, _raster_buffers, load_mesh, normalize_mesh  # noqa: E402
from mesh2cad.vision import render_observations  # noqa: E402

DEEP_RUN = "20261022T202330Z-cdd721bd"
INTERIOR_RUN = "20261022T195322Z-764b4929"
EXAMPLE_ID = "39e6d36142177f5b"
TERMINAL = {"success", "exhausted", "stalled", "failed", "unscorable"}
RUNS = REPO / "runs/harness"
IMAGES = PROJECT / "images"
FIGURES: list[dict] = []


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.resolve().relative_to(REPO))


def audit_inputs():
    cache = REPO / ".cache/presentation-audit/audit.json"
    if cache.is_file():
        audit = read(cache)
        with (cache.parent / "items.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        return audit, rows, {"path": relative(cache), "sha256": sha(cache)}
    saved = read(PROJECT / "evidence.json")["audit_snapshot"]
    return saved["audit"], saved["item_rows"], saved["source"]


def record_image(name, sources, interpretation, **extra):
    path = IMAGES / name
    with Image.open(path) as image:
        dimensions = list(image.size)
    FIGURES.append(
        {
            "output": relative(path),
            "sha256": sha(path),
            "dimensions": dimensions,
            "sources": sources,
            "interpretation": interpretation,
            **extra,
        }
    )


def source(path):
    return {"path": relative(path), "sha256": sha(path)}


def copy_figure(path, name, interpretation, **extra):
    destination = IMAGES / name
    if path.resolve() != destination.resolve():
        shutil.copyfile(path, destination)
    record_image(name, [source(path)], interpretation, **extra)


def render_single(mesh, name, mesh_path, *, size=640, normalization=None):
    _, elevation, azimuth = VIEWS[3]
    image, *_ = _raster_buffers(
        [(mesh, (0.30, 0.68, 1.0), 0.0)], elevation, azimuth, size, frame_width=1.8
    )
    image.save(IMAGES / name)
    record_image(
        name,
        [source(mesh_path)],
        "Current deterministic re-render of saved geometry, not an image sent in a historical call; no candidate fitting.",
        camera={
            "view": "isometric",
            "elevation": elevation,
            "azimuth": azimuth,
            "frame_width": 1.8,
        },
        normalization=normalization,
    )


def contact_strip(paths, name):
    sheet = Image.new("RGB", (1536, 512), "black")
    sources = []
    _, elevation, azimuth = VIEWS[3]
    for index, path in enumerate(paths):
        mesh, matrix = normalize_mesh(load_mesh(path))
        image, *_ = _raster_buffers(
            [(mesh, (0.30, 0.68, 1.0), 0.0)], elevation, azimuth, 512, frame_width=1.8
        )
        sheet.paste(image, (512 * index, 0))
        sources.append({**source(path), "normalization": matrix})
    sheet.save(IMAGES / name)
    record_image(
        name,
        sources,
        "Illustrative domain examples; each target independently bbox-normalized once. Same camera, full saved meshes; not scored predictions.",
        camera={
            "view": "isometric",
            "elevation": elevation,
            "azimuth": azimuth,
            "frame_width": 1.8,
        },
    )


def make_macros(deep, interior, example, deep_state):
    values = {}
    for prefix, aggregate in [("Deep", deep), ("Interior", interior)]:
        values[prefix + "Count"] = str(aggregate["n"])
        for suffix, key in [("First", "first"), ("Best", "best")]:
            phase = aggregate[key]
            values[prefix + suffix + "IoU"] = f"{phase['mean_iou_failure_zero']:.4f}"
            values[prefix + suffix + "Median"] = f"{phase['median_iou_failure_zero']:.4f}"
            values[prefix + suffix + "Cost"] = f"{phase['reported_cost_usd']:.6f}"
            values[prefix + suffix + "CostPerItem"] = (
                f"{phase['reported_cost_usd'] / aggregate['n']:.6f}"
            )
            values[prefix + suffix + "Calls"] = str(phase["provider_calls"])
            values[prefix + suffix + "Valid"] = str(phase["n_cad_execution_valid"])
            values[prefix + suffix + "Scored"] = str(phase["n_iou_available"])
            values[prefix + suffix + "Success"] = str(phase["iou_at_least_0_9"])
            values[prefix + suffix + "Seconds"] = f"{phase['mean_attempt_elapsed_seconds']:.1f}"
        values[prefix + "Gain"] = f"+{aggregate['mean_paired_iou_gain']:.4f}"
        values[prefix + "Improved"] = str(aggregate["improved_items"])
    values.update(
        DeepPlanned=str(len(deep_state["items"])),
        DeepRemainingPlanned=str(len(deep_state["items"]) - deep["n"]),
        DeepPartial="1",
        DeepPending="18",
        ExampleFirstIoU=f"{example['attempts'][1]['iou']['iou']:.4f}",
        ExampleBestIoU=f"{example['attempts'][4]['iou']['iou']:.4f}",
        ExampleFirstIteration="2",
        ExampleBestIteration="5",
        InteriorPlanned="6",
        InteriorUnscorable="1",
        FutureDeepCount="50",
        FutureInteriorCount="50",
    )
    (PROJECT / "results.tex").write_text(
        "% Derived by prepare_assets.py from the frozen local audit.\n"
        "% First = first CAD attempt; v2 first includes analysis. Unavailable first IoU = 0 only in mean.\n"
        + "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in values.items())
        + "\n"
    )
    return values


def paired_summary(state):
    """Independent verification from saved attempts, not cached aggregate means."""
    pairs = []
    for rank, item in enumerate(state["items"], 1):
        if item["status"] not in TERMINAL:
            continue
        first = item["attempts"][0]
        raw = (first.get("iou") or {}).get("iou")
        saved = item["attempts"][item["best_attempt"]]["iou"]["iou"]
        actual = max(
            (a.get("iou") or {})["iou"]
            for a in item["attempts"]
            if (a.get("iou") or {}).get("iou") is not None
        )
        assert saved == actual
        calls = [c for a in item["attempts"] for c in a.get("calls", [])]
        first_calls = first.get("calls", [])
        assert len(first_calls) == 1 and first_calls[0]["phase"] == "generation"
        assert all(
            c["status"] == "received" and isinstance(c.get("usage", {}).get("cost"), (float, int))
            for c in calls
        )
        pairs.append(
            {
                "object_rank": rank,
                "item_id": item["id"],
                "first_iou_raw": raw,
                "first_iou": raw if raw is not None else 0.0,
                "best_iou": saved,
                "gain": saved - (raw if raw is not None else 0.0),
                "first_cost_usd": sum(c["usage"]["cost"] for c in first_calls),
                "total_cost_usd": sum(c["usage"]["cost"] for c in calls),
                "first_calls": len(first_calls),
                "total_calls": len(calls),
            }
        )
    assert len(pairs) == 31
    first = np.array([p["first_iou"] for p in pairs])
    best = np.array([p["best_iou"] for p in pairs])
    gain = best - first
    boots = (
        np.random.default_rng(42).choice(gain, size=(20000, len(gain)), replace=True).mean(axis=1)
    )
    ci = np.quantile(boots, [0.025, 0.975], method="linear").tolist()
    summary = {
        "source_run": DEEP_RUN,
        "n": len(pairs),
        "score_rule": "First CAD attempt, unavailable first IoU contributes0; best saved full-volume IoU. Original run order.",
        "first_iou": {"mean": float(first.mean()), "median": float(np.median(first))},
        "best_iou": {"mean": float(best.mean()), "median": float(np.median(best))},
        "paired_gain": {
            "mean": float(gain.mean()),
            "median": float(np.median(gain)),
            "improved": int((gain > 1e-6).sum()),
            "bootstrap_95_percent_ci": ci,
            "bootstrap_method": "20000 paired percentile resamples; numpy.default_rng42; quantile linear",
        },
        "first_cost_usd": sum(p["first_cost_usd"] for p in pairs),
        "best_cost_usd": sum(p["total_cost_usd"] for p in pairs),
        "caveat": "Descriptive statistics conditional on31 completed prefix, not whole-dataset confidence or controlled feedback ablation.",
        "pairs": pairs,
    }
    cache = REPO / ".cache/presentation-audit/paired-summary.json"
    if cache.is_file():
        other = read(cache)
        assert np.allclose(ci, other["paired_gain"]["bootstrap_95_percent_ci"], rtol=0, atol=1e-14)
        assert np.isclose(summary["first_iou"]["mean"], other["first_iou"]["mean"])
        assert np.isclose(summary["best_cost_usd"], other["harness_call_accounting"]["cost_usd"])
        summary["independent_audit_agrees"] = source(cache)
    (PROJECT / "paired-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (IMAGES / "new-deepcad-pairs.dat").open("w") as stream:
        stream.write("rank first best first_cost total_cost gain\n")
        for pair in pairs:
            stream.write(
                f"{pair['object_rank']} {pair['first_iou']:.12g} {pair['best_iou']:.12g} {pair['first_cost_usd']:.12g} {pair['total_cost_usd']:.12g} {pair['gain']:.12g}\n"
            )
    return summary


def protocol(deep_state, manifests):
    """Freeze future same-ID comparisons; does not execute any paid job."""
    deep_manifest, interior_manifest = manifests
    deep_eligible = []
    for item in deep_manifest["items"]:
        if item["split"] == "test":
            path = REPO / "data/deepcad" / item["mesh_path"]
            if load_mesh(path).is_volume:
                deep_eligible.append(item)
    eligible = {x["id"]: x for x in deep_eligible}
    ids = [x["id"] for x in deep_state["items"] if x["id"] in eligible]
    remaining = sorted(set(eligible) - set(ids))
    random.Random(42).shuffle(remaining)
    ids = (ids + remaining)[:50]
    assert len(ids) == 50 and len(set(ids)) == 50
    deep_cohort = [
        {
            "id": key,
            "mesh_path": "data/deepcad/" + eligible[key]["mesh_path"],
            "sha256": eligible[key]["mesh_sha256"],
            "split": "test",
        }
        for key in ids
    ]
    interior_cohort = []
    rejected = []
    for upstream, count in [("indoorcad", 40), ("polyhaven", 10)]:
        candidates = sorted(
            [
                x
                for x in interior_manifest["items"]
                if x.get("is_volume") and x["source"] == upstream
            ],
            key=lambda x: x["id"],
        )
        random.Random(42).shuffle(candidates)
        accepted = 0
        for item in candidates:
            if accepted >= count:
                break
            path = REPO / "data/interior300" / item["mesh_path"]
            assert sha(path) == item["sha256"]
            mesh = load_mesh(path)
            if not mesh.is_volume:
                rejected.append(
                    {
                        "id": item["id"],
                        "reason": "not_volume_in_current_full_STL_loader_despite_manifest_flag",
                    }
                )
                del mesh
                continue
            del mesh
            accepted += 1
            interior_cohort.append(
                {
                    "id": item["id"],
                    "mesh_path": relative(path),
                    "sha256": item["sha256"],
                    "source": upstream,
                    "category": item["category"],
                    "split": "all",
                }
            )
    assert len(interior_cohort) == 50
    return {
        "schema_version": 1,
        "status": "planned_not_executed",
        "seed": 42,
        "cohort_fixed_on": "2026-10-23 Europe/Moscow",
        "selection": {
            "deepcad": "Reuse volume-eligible IDs from historical planned 50 test items in their saved order; fill any ineligible gap from other volume-eligible test IDs sorted then shuffled seed42. Never select on achieved quality.",
            "interior": "Source-stratified random sample of manifest-marked volumes:40 IndoorCAD from95 marked,10 PolyHaven from23 marked. Within each stratum IDs sorted then shuffled seed42; verify each full STL volume and hash, skip rejected inputs and continue in the same pre-shuffled order until quota. Manifest flags alone do not establish eligibility.",
            "limitation": "Protocol frozen after exploratory experiments, not a claim of untouched holdout. Original local DeepCAD dev/test split preserved; Interior300 has no official split.",
        },
        "cohorts": {"deepcad": deep_cohort, "interior300": interior_cohort},
        "input_eligibility_rejections": rejected,
        "deepcad_verified_volume_test_pool": len(deep_eligible),
        "required_conditions": [
            {
                "name": "cheap_one_prompt",
                "model": "openai/gpt-6-luna",
                "cad_generations": 1,
                "analysis_calls": 0,
                "feedback": False,
            },
            {
                "name": "cheap_harness",
                "model": "openai/gpt-6-luna",
                "orchestrator": "classic",
                "cad_generations_max": 5,
                "analysis_calls": 0,
                "feedback": True,
            },
            {
                "name": "expensive_one_prompt",
                "model": None,
                "model_status": "must_fix_available_expensive_model_before_run",
                "cad_generations": 1,
                "analysis_calls": 0,
                "feedback": False,
            },
        ],
        "optional_conditions": [
            {"name": "expensive_harness", "enabled": False, "reason": "only if budget remains"}
        ],
        "shared_settings": {
            "input": "same six-view target sheet, normalized extents, request and base system prompt for all three conditions; classic repair adds measured feedback",
            "reasoning_effort": "medium",
            "max_output_tokens": 16000,
            "cad_execution_timeout_seconds": 60,
            "request_timeout_seconds": 180,
            "memory_limit_mib": 2048,
            "target_iou": 0.9,
            "patience": 3,
            "min_improvement": 0.001,
            "numerical_optimization": False,
            "independent_alignment_or_scale_fit": False,
        },
        "before_paid_run": {
            "fixed_expensive_model": None,
            "maximum_total_usd": None,
            "harness_time_budget_per_object_seconds": None,
            "harness_token_budget_per_object": None,
            "renderer_and_prompt_sha256": "capture exact current source revision at launch",
            "status": "No paid benchmark was launched; unresolved budget/model settings must be fixed before execution.",
        },
        "metrics": {
            "primary": "mean best full-volume IoU over all50 eligible targets per domain; exhausted/failed prediction unavailable treated0 in primary mean, also report availability",
            "secondary": [
                "valid CAD rate",
                "mean/median IoU on scored outputs",
                "fraction IoU>=0.9",
                "provider calls",
                "all provider-reported USD including failed generations and analysis if any",
                "unknown cost count",
                "mean elapsed seconds per item",
            ],
            "paired": "Report per-ID first versus best difference and bootstrapped paired95%CI; first historical attempt is only a diagnostic proxy, not a separately randomized baseline",
            "failures": "Input eligibility fixed before generation; do not treat unprocessed or partial jobs as measured failures. Report incomplete coverage separately.",
        },
    }


def main():
    IMAGES.mkdir(exist_ok=True)
    audit, rows, audit_source = audit_inputs()
    summaries = {x["run_id"]: x for x in audit["runs"]}
    states = {key: read(RUNS / key / "run.json") for key in (DEEP_RUN, INTERIOR_RUN)}
    deep = summaries[DEEP_RUN]["terminal_scorable_targets"]
    interior = summaries[INTERIOR_RUN]["terminal_scorable_targets"]
    assert deep["n"] == 31 and interior["n"] == 5
    memberships = {
        DEEP_RUN: {x["id"] for x in states[DEEP_RUN]["items"] if x["status"] in TERMINAL},
        INTERIOR_RUN: {
            x["id"]
            for x in states[INTERIOR_RUN]["items"]
            if x["status"] in TERMINAL and x["status"] != "unscorable"
        },
    }
    selected_rows = [
        {
            **x,
            "cohort": "deepcad31_completed"
            if x["run_id"] == DEEP_RUN
            else "interior5_legacy_multimodal",
            "first_is_independent_one_shot_baseline": "False",
        }
        for x in rows
        if x["run_id"] in memberships and x["item_id"] in memberships[x["run_id"]]
    ]
    assert len(selected_rows) == 36
    with (PROJECT / "benchmark-items.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    example = next(x for x in states[DEEP_RUN]["items"] if x["id"] == EXAMPLE_ID)
    values = make_macros(deep, interior, example, states[DEEP_RUN])
    pairs = paired_summary(states[DEEP_RUN])
    for key in [
        "DeepCount",
        "DeepFirstIoU",
        "DeepBestIoU",
        "DeepGain",
        "DeepFirstCost",
        "DeepBestCost",
        "InteriorCount",
        "InteriorFirstIoU",
        "InteriorBestIoU",
    ]:
        print(key, values[key], flush=True)
    root = RUNS / DEEP_RUN / "items" / EXAMPLE_ID
    copy_figure(
        root / "target.png",
        "new-input-sheet.png",
        "Exact saved six-view RGB/depth target image from the historical classic job.",
        run_id=DEEP_RUN,
        item_id=EXAMPLE_ID,
    )
    feedback = REPO / ".cache/overlay/green-blue-red-depth.png"
    if not feedback.is_file():
        feedback = IMAGES / "new-feedback-sheet.png"
    copy_figure(
        feedback,
        "new-feedback-sheet.png",
        "Current-method illustration re-rendered after the historical job: true volume regions with depth brightness. NOT evidence that historical model calls saw these pixels.",
        run_id=DEEP_RUN,
        item_id=EXAMPLE_ID,
        candidate_iteration=2,
        historical_request_image=False,
        renderer_overlay_version=3,
    )
    for name, path in [
        ("new-example-target.png", root / "target.stl"),
        ("new-example-first.png", Path(example["attempts"][1]["execution"]["mesh_path"])),
        ("new-example-best.png", Path(example["attempts"][4]["execution"]["mesh_path"])),
    ]:
        render_single(load_mesh(path), name, path)
    deep_paths = [
        RUNS / DEEP_RUN / "items" / identifier / "target.stl"
        for identifier in ["3376e6b0a71b3a98", EXAMPLE_ID, "fe06896782f584a6"]
    ]
    contact_strip(deep_paths, "new-domains-deepcad.png")
    im_manifest = read(REPO / "data/interior300/manifest.json")
    selected = [
        "polyhaven_plastic_monobloc_chair_01",
        "polyhaven_dining_table",
        "indoorcad_6d254132d4cff1",
    ]
    interior_paths = [
        REPO
        / "data/interior300"
        / next(x["mesh_path"] for x in im_manifest["items"] if x["id"] == identifier)
        for identifier in selected
    ]
    contact_strip(interior_paths, "new-domains-interior.png")
    depth_source = interior_paths[0]
    mesh, matrix = normalize_mesh(load_mesh(depth_source))
    (IMAGES / "new-interior-depth.png").write_bytes(
        render_observations(mesh, mesh.extents)["depth"]
    )
    record_image(
        "new-interior-depth.png",
        [source(depth_source)],
        "Current depth-only rendering of known original interior target; illustration, no model call and not the legacy Interior5 input.",
        normalization=matrix,
        historical_request_image=False,
    )
    manifests = [read(REPO / "data/deepcad/manifest.json"), im_manifest]
    experiment = protocol(states[DEEP_RUN], manifests)
    (PROJECT / "experiment-protocol.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2) + "\n"
    )
    source_files = [
        "mesh2cad/" + name + ".py"
        for name in [
            "harness",
            "harness2",
            "agent",
            "geometry",
            "vision",
            "prompts",
            "executor",
            "resources",
            "dataset",
        ]
    ]
    source_files += [
        "data/deepcad/manifest.json",
        "data/interior300/manifest.json",
        "output/latex/mesh2cad-research/prepare_assets.py",
    ]
    brief = subprocess.check_output(["git", "show", "d63d25e:README.md"], cwd=REPO)
    evidence = {
        "schema_version": 2,
        "as_of": "2026-10-23 Europe/Moscow",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_project_brief": {
            "git_object": "d63d25e:README.md",
            "sha256": hashlib.sha256(brief).hexdigest().lower(),
            "text": brief.decode(),
            "team_names": "not found; pending user supplied names",
        },
        "repository": {
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=REPO, text=True
            ).strip(),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "file_sha256": {name: sha(REPO / name) for name in source_files},
        },
        "cohorts": [
            {
                "label": label,
                "run_id": run_id,
                "checkpoint_sha256": sha(RUNS / run_id / "run.json"),
                "included_ids": sorted(memberships[run_id]),
                "aggregate": aggregate,
                "run_summary": summaries[run_id],
                "exclusions": [
                    {
                        "id": x["id"],
                        "status": x["status"],
                        "reason": "input_unscorable"
                        if x["status"] == "unscorable"
                        else "partial_or_unprocessed_not_a_failure",
                    }
                    for x in states[run_id]["items"]
                    if x["id"] not in memberships[run_id]
                ],
            }
            for label, run_id, aggregate in [
                ("deepcad31_completed", DEEP_RUN, deep),
                ("interior5_legacy_multimodal", INTERIOR_RUN, interior),
            ]
        ],
        "interpretation": {
            "independent_one_shot_baseline_available": False,
            "expensive_geometry_baseline_available": False,
            "historic_deep_feedback": "Earlier classic feedback version; may differ from current green/blue/red depth-shaded illustration. Inspect archived request image bytes for historical context.",
            "historic_interior_feedback": "Legacy vision_prompt_version1: RGB,depth,blueprint; analysis included in first-attempt provider calls; not current depth-only v2.",
            "observed_gain_is_dataset_generalization_proof": False,
            "cost": "All provider-reported usage.cost for included IDs; excludes extra partial Deep32. Unknown charges not imputed.",
            "partial_32nd_deep_item": "Excluded from31-terminal cohort; not counted as failure.18 other planned items never processed.",
            "normalization": "Targets bbox-centered, max extent1; generated candidates never independently fitted.",
        },
        "paired_statistics": pairs,
        "example": {
            "run_id": DEEP_RUN,
            "item_id": EXAMPLE_ID,
            "iteration1": "execution_error, no scored CAD",
            "displayed_first_iteration": 2,
            "displayed_best_iteration": 5,
            "attempts": [
                {key: a.get(key) for key in ["iteration", "status", "iou", "execution"]}
                for a in example["attempts"]
            ],
        },
        "figure_sources": FIGURES,
        "slide_macros": values,
        "audit_snapshot": {"source": audit_source, "audit": audit, "item_rows": rows},
        "future_protocol": {
            "path": "experiment-protocol.json",
            "sha256": sha(PROJECT / "experiment-protocol.json"),
            "status": "planned_not_executed",
        },
        "benchmark_items": {
            "path": "benchmark-items.csv",
            "sha256": sha(PROJECT / "benchmark-items.csv"),
            "rows": len(selected_rows),
        },
    }
    (PROJECT / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Prepared",
        len(FIGURES),
        "figures,",
        len(values),
        "macros and",
        len(selected_rows),
        "per-object rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()
