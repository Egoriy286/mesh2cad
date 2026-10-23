"""Prepare Sol on the exact frozen hard50/interior50 cohort; no generation calls."""

import copy
import json
from dataclasses import replace
from pathlib import Path

from mesh2cad.batch_benchmark import PROJECT, provider_snapshot
from mesh2cad.harness import HarnessConfig, create_run, utc_now
from mesh2cad.pipeline import api_key_from_environment
from mesh2cad.storage import file_digest, read_json, write_json

SOURCE = PROJECT / "output/benchmarks/20261023-hard50-interior50"
OUTPUT = PROJECT / "output/benchmarks/20261023-sol-hard50-interior50"
MODEL = "openai/gpt-6-sol"


def prepare():
    if (OUTPUT / "protocol.json").exists():
        return read_json(OUTPUT / "protocol.json")
    old = read_json(SOURCE / "protocol.json")
    snapshot = provider_snapshot(api_key_from_environment(PROJECT), MODEL)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = copy.deepcopy(old)
    for name in (
        "started_at",
        "initial_started_at",
        "finished_at",
        "resumes",
        "last_resume_started_at",
        "wall_seconds_including_pause",
    ):
        protocol.pop(name, None)
    protocol.update(
        created_at=utc_now(),
        status="prepared",
        model=MODEL,
        max_iterations=2,
        parallel_workers=8,
        provider_before=snapshot,
        source_protocol=str(SOURCE / "protocol.json"),
        source_protocol_sha256=file_digest(SOURCE / "protocol.json"),
        comparison="First response (one prompt) versus best of at most two total iterations; shared first response",
        source_sha256={
            str(p.relative_to(PROJECT)): file_digest(p) for p in (PROJECT / "mesh2cad").glob("*.py")
        },
        batches=[],
    )
    write_json(
        OUTPUT / "budget.json",
        dict(
            cap=max(0, snapshot["limit_remaining"] - 0.01),
            spent=0,
            reserved={},
            unknown_cost_calls=0,
            prices=snapshot["pricing"],
        ),
    )
    for batch in old["batches"]:
        previous = read_json(Path(batch["run"]) / "run.json")
        cfg = replace(HarnessConfig.from_dict(previous["config"]), model=MODEL, max_iterations=2)
        domain, ordinal, ids = batch["domain"], batch["batch"], batch["ids"]
        folder = create_run(
            PROJECT / "data" / domain / "manifest.json",
            OUTPUT / "runs" / domain / f"batch-{ordinal:02d}",
            cfg,
            split="test" if domain == "deepcad" else "all",
            item_ids=ids,
        )
        state = read_json(folder / "run.json")
        state["items"].sort(key=lambda i: ids.index(i["id"]))
        expected = {i["id"]: i["input_sha256"] for i in previous["items"]}
        assert all(i["input_sha256"] == expected[i["id"]] for i in state["items"])
        state["experiment"] = dict(
            domain=domain, batch=ordinal, protocol=str(OUTPUT / "protocol.json")
        )
        write_json(folder / "run.json", state)
        protocol["batches"].append(dict(domain=domain, batch=ordinal, ids=ids, run=str(folder)))
        link = PROJECT / "runs/harness" / folder.name
        if not link.exists():
            link.symlink_to(folder, target_is_directory=True)
    write_json(OUTPUT / "protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    p = prepare()
    print(
        json.dumps(
            dict(
                output=str(OUTPUT),
                model=p["model"],
                batches=len(p["batches"]),
                provider=p["provider_before"],
            ),
            indent=2,
        )
    )
