import csv
from pathlib import Path

import pytest

from mesh2cad.report import build_report
from mesh2cad.storage import read_json, write_json


def _generation(*, usage=True, status="ok", launches=1):
    return {
        "status": status,
        "launches": launches,
        "elapsed_seconds": 2.5,
        "usage": {"input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 5}
        if usage
        else {},
    }


def _attempt(
    index,
    *,
    cd=None,
    iou=None,
    iou_status="ok",
    execution_status="valid",
    status="complete",
    generations=None,
):
    return {
        "index": index,
        "status": status,
        "execution_status": execution_status,
        "metrics": {"chamfer": cd, "iou": iou, "iou_status": iou_status, "target_watertight": True}
        if cd is not None
        else None,
        "artifacts": {
            "code": f"items/a/attempt-{index}/candidate.py",
            "step": f"items/a/attempt-{index}/candidate.step",
            "comparison": f"items/a/attempt-{index}/comparison.png",
        },
        "generations": [_generation()] if generations is None else generations,
        "execution_seconds": 0.5 if execution_status else 0,
    }


def _run(tmp_path, items, results):
    write_json(tmp_path / "selection.json", {"items": items})
    write_json(tmp_path / "config.json", {"model": "test-model"})
    for item_id, result in results.items():
        write_json(tmp_path / "items" / item_id / "result.json", result)
    return build_report(tmp_path)


def _result(attempts, best_index=None, status="complete"):
    return {"status": status, "attempts": attempts, "best_index": best_index}


def test_denominator_includes_missing_and_invalid_first_recovers(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            {"id": "a", "split": "test", "target_watertight": True},
            {"id": "b", "split": "test", "target_watertight": True},
            {"id": "c", "split": "test", "target_watertight": True},
        ],
        {
            "a": _result([_attempt(0, cd=0.1, iou=0.8)], 0),
            "b": _result(
                [_attempt(0, execution_status="execution_error"), _attempt(1, cd=0.2, iou=0.6)], 1
            ),
        },
    )
    assert summary["selected_count"] == 3
    assert summary["pending_count"] == 1
    assert summary["baseline"]["cad_valid_count"] == 1
    assert summary["harness"]["cad_valid_rate"] == pytest.approx(2 / 3)
    assert summary["recovered_valid_count"] == 1
    assert summary["paired_chamfer"]["count"] == 1
    assert summary["baseline"]["mean_iou_known"] == pytest.approx(0.4)
    assert summary["baseline"]["mean_iou"] is None
    assert summary["baseline"]["iou_unavailable_count"] == 1
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert [row["id"] for row in rows] == ["a", "b", "c"]
    assert rows[2]["status"] == "pending"
    assert rows[2]["harness_iou"] == ""
    assert read_json(tmp_path / "summary.json") == summary


def test_worsening_repair_does_not_replace_saved_candidate_or_its_iou(tmp_path):
    (tmp_path / "items" / "a").mkdir(parents=True)
    (tmp_path / "items" / "a" / "target.png").write_bytes(b"target")
    summary = _run(
        tmp_path,
        [{"id": "a", "target_watertight": True}],
        {
            "a": _result(
                [
                    _attempt(0, cd=0.3, iou=0.4),
                    _attempt(1, cd=0.1, iou=0.6),
                    _attempt(2, cd=0.2, iou=0.99),
                ],
                1,
            ),
        },
    )
    assert summary["harness"]["mean_chamfer"] == 0.1
    assert summary["harness"]["mean_iou"] == 0.6
    assert summary["last"]["mean_chamfer"] == 0.2
    assert summary["last_worse_than_best_count"] == 1
    assert summary["paired_chamfer"]["mean_improvement"] == pytest.approx(0.2)
    markdown = (tmp_path / "report.md").read_text()
    assert "[target](items/a/target.png)" in markdown
    assert "[harness step](items/a/attempt-1/candidate.step)" in markdown
    assert "not an independent best-of-three" in markdown


def test_unavailable_boolean_and_unknown_target_do_not_become_zero(tmp_path):
    summary = _run(
        tmp_path,
        [
            {"id": "a", "target_watertight": True},
            {"id": "b", "target_watertight": False},
            {"id": "c"},
        ],
        {
            "a": _result([_attempt(0, cd=0.1, iou_status="boolean_error:RuntimeError")], 0),
            "b": _result([_attempt(0, cd=0.2, iou_status="non_watertight_target")], 0),
            "c": _result([_attempt(0, execution_status=None, status="generation_error")]),
        },
    )
    block = summary["harness"]
    assert block["iou_eligible_count"] == 1
    assert block["iou_unavailable_count"] == 1
    assert block["iou_unknown_target_count"] == 1
    assert block["iou_open_target_count"] == 1
    assert block["mean_iou_known"] is None
    assert block["iou_status_counts"]["boolean_error:RuntimeError"] == 1


def test_failure_zero_and_scoring_failure_unavailable_are_distinct(tmp_path):
    summary = _run(
        tmp_path,
        [
            {"id": "a", "target_watertight": True},
            {"id": "b", "target_watertight": True},
        ],
        {
            "a": _result([_attempt(0, execution_status=None, status="generation_error")]),
            "b": _result([_attempt(0, execution_status="valid", status="evaluation_error")]),
        },
    )
    assert summary["baseline"]["cad_valid_count"] == 1
    assert summary["baseline"]["iou_known_count"] == 1
    assert summary["baseline"]["mean_iou_known"] == 0
    assert summary["baseline"]["iou_unavailable_count"] == 1
    assert summary["baseline"]["stage_counts"] == {"evaluation_error": 1, "generation_error": 1}


def test_quota_retry_ledgers_include_launches_and_preserve_unknown_usage(tmp_path):
    summary = _run(
        tmp_path,
        [{"id": "a", "target_watertight": True}],
        {
            "a": _result(
                [
                    _attempt(
                        0,
                        cd=0.2,
                        iou=0.5,
                        generations=[
                            _generation(usage=False, status="quota_exhausted"),
                            _generation(),
                        ],
                    ),
                    _attempt(1, cd=0.1, iou=0.7),
                    _attempt(
                        2,
                        execution_status=None,
                        status="quota_exhausted",
                        generations=[_generation(usage=False, status="quota_exhausted")],
                    ),
                ],
                1,
                status="quota_exhausted",
            ),
        },
    )
    baseline, harness = summary["baseline_cost"], summary["harness_cost"]
    assert baseline["launches"] == 2
    assert harness["launches"] == 4
    assert harness["unknown_usage_records"] == 2
    assert harness["input_tokens"] is None
    assert harness["total_tokens"] is None
    assert harness["known_input_tokens"] == 20
    assert harness["known_cached_input_tokens"] == 6
    assert harness["known_output_tokens"] == 10
    assert harness["generation_seconds"] == 10
    assert summary["last"]["mean_chamfer"] == 0.1
    assert summary["harness"]["mean_chamfer"] == 0.1


def test_empty_selection_writes_valid_empty_report(tmp_path):
    summary = _run(tmp_path, [], {})
    assert summary["selected_count"] == 0
    assert summary["baseline"]["cad_valid_rate"] is None
    assert (tmp_path / "results.csv").read_text().strip() == "id,split,status"


def test_each_preflight_invocation_is_counted_once_and_alias_is_ignored(tmp_path):
    failed = {"status": "quota_exhausted", "generation": _generation(usage=False)}
    valid = {"status": "ok", "generation": _generation(), "execution": {"elapsed_seconds": 1.5}}
    write_json(tmp_path / "preflight" / "invocation-000" / "smoke.json", failed)
    write_json(tmp_path / "preflight" / "invocation-001" / "smoke.json", valid)
    write_json(tmp_path / "preflight" / "smoke.json", valid)
    summary = _run(
        tmp_path,
        [{"id": "a", "target_watertight": True}],
        {
            "a": _result([_attempt(0, cd=0.1, iou=0.8)], 0),
        },
    )
    assert summary["baseline_cost"]["launches"] == 1
    assert summary["harness_cost"]["launches"] == 1
    assert summary["preflight_cost"]["launches"] == 2
    assert summary["preflight_cost"]["execution_seconds"] == 1.5
    assert summary["overall_cost"]["launches"] == 3
    assert summary["overall_cost"]["known_input_tokens"] == 20
    assert summary["overall_cost"]["input_tokens"] is None


def test_preflight_recovers_partial_invocations_and_counts_incomplete_statuses(tmp_path):
    generation = _generation()
    write_json(
        tmp_path / "preflight" / "invocation-000" / "generation" / "generation.json", generation
    )
    write_json(tmp_path / "preflight" / "invocation-001" / "generation" / "command.json", {})
    write_json(tmp_path / "preflight" / "invocation-002" / "smoke.json", {"generation": generation})
    write_json(
        tmp_path / "preflight" / "invocation-002" / "generation" / "generation.json", generation
    )
    summary = _run(
        tmp_path,
        [{"id": value} for value in "abcde"],
        {
            "a": _result([], status="quota_exhausted"),
            "b": _result([], status="interrupted"),
            "c": _result([], status="protocol_violation"),
            "d": _result([], status="input_error"),
        },
    )
    assert summary["pending_count"] == 4
    assert summary["preflight_cost"]["launches"] == 3
    assert summary["preflight_cost"]["known_input_tokens"] == 20
    assert summary["preflight_cost"]["unknown_usage_records"] == 1
    assert summary["preflight_cost"]["input_tokens"] is None
