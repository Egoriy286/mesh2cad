"""Independent audit of the completed 100-STL batch experiment. No API calls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "output/benchmarks/20261023-hard50-interior50"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    folder = args.output
    protocol = read(folder / "protocol.json")
    assert protocol["status"] == "completed", "Wait for all eight batches"
    rows = list(csv.DictReader((folder / "items.csv").open()))
    assert len(rows) == 100 and len({r["id"] for r in rows}) == 100
    by_id = {r["id"]: r for r in rows}
    pinned = {
        i["id"]: i.get("mesh_sha256", i.get("sha256"))
        for group in protocol["cohorts"].values()
        for i in group
    }
    responses, total_tokens, cost = [], 0, 0.0
    transport_errors = []
    batch_checks = []
    for domain in ("deepcad", "interior300"):
        selected = protocol["cohorts"][domain]
        assert len(selected) == 50
        assert {r["id"] for r in rows if r["domain"] == domain} == {i["id"] for i in selected}
        assert [len(b["ids"]) for b in protocol["batches"] if b["domain"] == domain] == [
            15,
            15,
            15,
            5,
        ]
    for batch in protocol["batches"]:
        root = Path(batch["run"])
        state = read(root / "run.json")
        cfg = state["config"]
        assert state["status"] == "completed"
        assert cfg["max_tokens"] == 16000 and cfg["total_token_budget"] == 1000000
        assert cfg["model"] == protocol["model"] and cfg["orchestrator"] == "classic"
        assert {i["id"] for i in state["items"]} == set(batch["ids"])
        batch_tokens = 0
        unknown_token_bound = 0
        for item in state["items"]:
            row = by_id[item["id"]]
            assert row["finished"] == "True"
            assert digest(root / item["input_path"]) == item["input_sha256"] == row["input_sha256"]
            assert item["input_sha256"] == pinned[item["id"]]
            target = root / "items" / item["id"] / "target.stl"
            assert target.is_file()
            scores, item_cost, calls = [], 0.0, 0
            for attempt in item["attempts"]:
                path = root / attempt["path"]
                request = read(path / "request.json")
                assert request["max_tokens"] == 16000 and request["model"] == protocol["model"]
                errors = [c for c in attempt["calls"] if c["status"] == "error"]
                text_bytes, images = 0, 0
                for message in request["messages"]:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        text_bytes += len(content.encode())
                    else:
                        text_bytes += sum(len(p.get("text", "").encode()) for p in content)
                        images += sum(p.get("type") == "image_url" for p in content)
                unknown_token_bound += len(errors) * (text_bytes + images * 12000 + 16000)
                transport_errors.extend(
                    {
                        "domain": batch["domain"],
                        "id": item["id"],
                        "iteration": attempt["iteration"],
                        "time": c["started_at"],
                        "error": c.get("error"),
                    }
                    for c in errors
                )
                response = read(path / "response.json")
                received = [c for c in attempt["calls"] if c["status"] == "received"]
                assert len(received) == 1 and received[0]["usage"] == response["usage"]
                usage = response["usage"]
                assert usage["completion_tokens"] <= 16000
                batch_tokens += usage["total_tokens"]
                total_tokens += usage["total_tokens"]
                item_cost += usage["cost"]
                calls += 1
                responses.append(
                    {
                        "id": response["id"],
                        "path": str(path / "response.json"),
                        "sha256": digest(path / "response.json"),
                        "usage": usage,
                    }
                )
                score = (attempt.get("iou") or {}).get("iou")
                if score is not None:
                    assert read(path / "iou.json")["iou"] == score
                    assert attempt["execution"]["status"] == "valid"
                    scores.append(score)
            assert abs(float(row["cost_usd"]) - item_cost) < 1e-10
            assert int(row["calls"]) == calls
            assert abs(float(row["iou_failure_zero"]) - max(scores, default=0)) < 1e-12
            if scores:
                assert item["best_attempt"] is not None
                assert state["items"][state["items"].index(item)]["attempts"][item["best_attempt"]][
                    "iou"
                ]["iou"] == max(scores)
            cost += item_cost
        assert batch_tokens <= cfg["total_token_budget"]
        assert batch_tokens + unknown_token_bound <= cfg["total_token_budget"]
        batch_checks.append(
            {
                "domain": batch["domain"],
                "batch": batch["batch"],
                "items": len(state["items"]),
                "tokens": batch_tokens,
                "tokens_conservative_bound_with_transport_errors": batch_tokens
                + unknown_token_bound,
                "config_sha256": digest(root / "run.json"),
            }
        )
    assert len({r["id"] for r in responses}) == len(responses), "Response billed more than once"
    summary = read(folder / "summary.json")
    for domain in ("deepcad", "interior300", "all"):
        group = [r for r in rows if domain == "all" or r["domain"] == domain]
        assert len(group) == summary["groups"][domain]["finished"]
        assert (
            abs(
                np.mean([float(r["iou_failure_zero"]) for r in group])
                - summary["groups"][domain]["mean_best_iou"]
            )
            < 1e-12
        )
    ledger = read(folder / "budget.json")
    assert not ledger["reserved"]
    assert ledger["unknown_cost_calls"] == len(transport_errors)
    assert cost <= ledger["spent"] + 1e-9
    after = read(folder / "provider-after.json")
    billed_delta = after["usage"] - protocol["provider_before"]["usage"]
    unallocated_billing = billed_delta - cost
    assert -1e-8 <= unallocated_billing <= ledger["spent"] - cost + 1e-8
    paired = {}
    for domain in ("deepcad", "interior300"):
        group = [r for r in rows if r["domain"] == domain]
        gains = np.array([float(r["iou_failure_zero"]) - float(r["first_iou"] or 0) for r in group])
        means = np.random.default_rng(42).choice(gains, (20000, len(gains))).mean(axis=1)
        paired[domain] = {
            "mean_gain": float(gains.mean()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "improved": int((gains > 1e-6).sum()),
            "n": len(gains),
        }
    result = {
        "status": "passed_with_unattributed_transport_billing"
        if unallocated_billing > 1e-8
        else "passed",
        "selected": 100,
        "finished": 100,
        "unique_responses": len(responses),
        "physical_cost_usd": billed_delta,
        "saved_response_cost_usd": cost,
        "unattributed_billed_usd": max(0, unallocated_billing),
        "ledger_conservative_upper_bound_usd": ledger["spent"],
        "tokens_in_saved_responses": total_tokens,
        "token_usage_complete": not transport_errors,
        "transport_error_calls": transport_errors,
        "http_requests_total": len(responses) + len(transport_errors),
        "provider_usage_delta_usd": billed_delta,
        "provider_billing_matches": abs(billed_delta - cost) < 1e-8,
        "batches": batch_checks,
        "paired_bootstrap": paired,
        "responses": responses,
    }
    (folder / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("responses", "batches")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
