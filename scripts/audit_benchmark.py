"""Independently verify paired inputs, saved responses, scores and USD totals."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "output/benchmarks/20261023"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = list(csv.DictReader((BENCH / "items.csv").open()))
    groups = read(BENCH / "summary.json")["groups"]
    checked = []
    physical_spend = 0
    for domain in groups:
        identities = {r["id"] for r in rows if r["domain"] == domain}
        for identity in sorted(identities):
            group = {
                r["condition"]: r for r in rows if r["domain"] == domain and r["id"] == identity
            }
            cheap = ROOT / group["cheap_harness"]["run"]
            expensive = ROOT / group["expensive_one_prompt"]["run"]
            states = [read(p / "run.json") for p in (cheap, expensive)]
            items = [s["items"][0] for s in states]
            assert items[0]["input_sha256"] == items[1]["input_sha256"]
            requests = []
            proof = {
                "domain": domain,
                "id": identity,
                "same_initial_request_except_model": True,
                "input_sha256": items[0]["input_sha256"],
                "run_sha256": [],
                "responses": [],
            }
            for folder, state, item in zip((cheap, expensive), states, items):
                assert state["status"] == "completed"
                assert digest(folder / item["input_path"]) == item["input_sha256"]
                assert item["status"] != "unscorable"
                first = read(folder / item["attempts"][0]["path"] / "request.json")
                first.pop("model")
                requests.append(first)
                proof["run_sha256"].append(
                    {
                        "path": str((folder / "run.json").relative_to(ROOT)),
                        "sha256": digest(folder / "run.json"),
                    }
                )
                scores = []
                costs = []
                for attempt in item["attempts"]:
                    path = folder / attempt["path"]
                    response = read(path / "response.json")
                    usage = response["usage"]
                    received = [c for c in attempt["calls"] if c["status"] == "received"]
                    assert len(received) == 1 and received[0]["usage"] == usage
                    costs.append(usage["cost"])
                    proof["responses"].append(
                        {
                            "path": str((path / "response.json").relative_to(ROOT)),
                            "sha256": digest(path / "response.json"),
                            "provider_id": response.get("id"),
                            "usage": usage,
                        }
                    )
                    score = (attempt.get("iou") or {}).get("iou")
                    if score is not None:
                        assert read(path / "iou.json")["iou"] == score
                        assert attempt["execution"]["status"] == "valid"
                        scores.append(score)
                condition = "cheap_harness" if folder == cheap else "expensive_one_prompt"
                assert (
                    abs(float(group[condition]["iou_failure_zero"]) - max(scores, default=0))
                    < 1e-12
                )
                assert abs(float(group[condition]["cost_usd"]) - sum(costs)) < 1e-12
                physical_spend += sum(costs)
                if folder == cheap:
                    raw = (item["attempts"][0].get("iou") or {}).get("iou")
                    assert (
                        abs(float(group["cheap_one_prompt"]["iou_failure_zero"]) - (raw or 0))
                        < 1e-12
                    )
                    assert abs(float(group["cheap_one_prompt"]["cost_usd"]) - costs[0]) < 1e-12
            assert requests[0] == requests[1]
            checked.append(proof)
        for condition in ("cheap_one_prompt", "cheap_harness", "expensive_one_prompt"):
            values = [
                float(r["iou_failure_zero"])
                for r in rows
                if r["domain"] == domain and r["condition"] == condition
            ]
            assert abs(sum(values) / len(values) - groups[domain][condition]["mean_iou"]) < 1e-12
    budget = read(BENCH / "budget.json")
    assert abs(physical_spend - budget["spent"]) < 1e-9
    assert not budget["reserved"]
    write = {
        "status": "passed",
        "n_paired_inputs": len(checked),
        "physical_spend_usd": physical_spend,
        "items": checked,
    }
    (BENCH / "audit.json").write_text(json.dumps(write, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in write.items() if k != "items"}))


if __name__ == "__main__":
    main()
