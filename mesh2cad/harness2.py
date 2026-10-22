"""Orchestrator 2: analyze calibrated observations, generate CAD, inspect depth errors."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from .agent import DepthAwareAgent
from .geometry import load_mesh, section_evidence
from .harness import Orchestrator, summarize
from .pipeline import payload_digest
from .storage import read_json, write_json
from .vision import (
    compare_observations,
    load_observations,
    observation_fingerprint,
    render_observations,
    save_observations,
)


def parse_analysis(response: dict, extents) -> dict:
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    analysis = json.loads(content)
    if not isinstance(analysis, dict) or not isinstance(analysis.get("components"), list):
        raise ValueError("Geometry analysis must contain a components array")
    if not analysis["components"] or len(analysis["components"]) > 20:
        raise ValueError("Geometry analysis needs 1..20 main components")
    issues = []
    for component in analysis["components"]:
        if not isinstance(component, dict) or not isinstance(component.get("bounds"), list):
            raise ValueError("Every component must contain numeric bounds")
        bounds = component["bounds"]
        if len(bounds) != 2 or any(not isinstance(p, list) or len(p) != 3 for p in bounds):
            raise ValueError("Invalid component bounds shape")
        for axis in range(3):
            low, high = bounds[0][axis], bounds[1][axis]
            if (
                any(type(v) not in (int, float) or not math.isfinite(v) for v in (low, high))
                or low >= high
            ):
                raise ValueError("Component bounds must be finite and increasing")
            if low < -extents[axis] / 2 - 1e-4 or high > extents[axis] / 2 + 1e-4:
                issues.append(
                    f"{component.get('name', '?')}: proposed bounds exceed target on {'XYZ'[axis]}"
                )
    analysis["dimension_checks"] = issues or [
        "All component bounds lie within the measured target frame"
    ]
    return analysis


class Orchestrator2(Orchestrator):
    def _render_overlay(self, target, candidate, folder: Path, item: dict, attempt: dict) -> None:
        # Depth maps are the only visual input in this mode.
        return

    def __init__(self, root: Path, api_key: str, *, agent=None):
        super().__init__(root, api_key, agent=agent)
        if self.config.orchestrator != "vision2":
            raise ValueError("Orchestrator 2 requires orchestrator=vision2")
        if agent is None:
            self.agent = DepthAwareAgent(
                self.config.model,
                system_prompt=self.config.system_prompt,
                request=self.config.request,
                reasoning_effort=self.config.reasoning_effort,
            )

    def _render_target(self, target, prepared: Path, item: dict) -> bytes:
        folder = prepared / "vision"
        signature = observation_fingerprint(target, target.extents)
        cached = folder / "fingerprint.json"
        if (
            cached.is_file()
            and read_json(cached).get("signature") == signature
            and all(
                (folder / name).is_file() for name in ("depth.png", "depth.npz", "calibration.json")
            )
        ):
            bundle = load_observations(folder)
            self._event("target_observations_reused", item_id=item["id"])
        else:
            bundle = render_observations(
                target, target.extents, cancel_check=lambda: self._stopped() is not None
            )
            save_observations(bundle, folder)
            write_json(cached, {"signature": signature})
        if "target_evidence" not in item:
            try:
                item["target_evidence"] = section_evidence(target)
            except Exception as exc:
                self._event(
                    "section_evidence_unavailable",
                    level="WARNING",
                    item_id=item["id"],
                    error=self._safe_error(exc),
                )
                item["target_evidence"] = {
                    "bounds": target.bounds.tolist(),
                    "volume": float(target.volume),
                }
        (prepared / "target.png").write_bytes(bundle["depth"])
        return bundle["depth"]

    def _context(self, item: dict, target, sheet: bytes, attempt: dict):
        context = super()._context(item, target, sheet, attempt)
        prepared = self.root / "items" / item["id"]
        bundle = load_observations(prepared / "vision")
        # Raw arrays stay on disk/in local comparison; never embed them into an LLM request.
        visuals = {key: bundle[key] for key in ("depth", "calibration")}
        plan = None
        analysis_dir = prepared / "analysis"
        response_path = analysis_dir / "response.json"
        if (
            not (self.root / attempt["path"] / "response.json").is_file()
            and not response_path.is_file()
        ):
            self._check(before_call=True)
            analysis_dir.mkdir(parents=True, exist_ok=True)
            analysis_context = replace(
                context,
                visual_inputs=visuals,
                target_evidence=item.get("target_evidence"),
                max_tokens=min(
                    self.config.analysis_max_tokens,
                    max(1, self.config.total_token_budget - summarize(self.state)["total_tokens"]),
                ),
            )
            payload = DepthAwareAgent(
                self.config.model,
                system_prompt=self.config.system_prompt,
                request=self.config.request,
            ).build_analysis_request(analysis_context)
            write_json(analysis_dir / "request.json", payload)
            item["analysis"] = {"status": "requesting", "request_sha256": payload_digest(payload)}
            self._request(attempt, analysis_dir, payload, item["id"], phase="geometry_analysis")
        if response_path.is_file():
            response = read_json(response_path)
            # Recover billing/accounting if the worker stopped just after the response was written.
            for previous in item["attempts"]:
                for call in previous.get("calls", []):
                    if call.get("phase") == "geometry_analysis" and call["status"] == "requesting":
                        call.update(status="received", usage=response.get("usage"))
            try:
                plan = parse_analysis(response, context.extents)
                item["analysis"] = {"status": "ready", "plan": plan}
                write_json(analysis_dir / "geometry.json", plan)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                item["analysis"] = {"status": "invalid", "error": self._safe_error(exc)}
                self._event(
                    "analysis_invalid",
                    level="WARNING",
                    item_id=item["id"],
                    error=item["analysis"]["error"],
                )
        self._save()
        # Recalculate output-token allowance after paying for the analysis pass.
        context = super()._context(item, target, sheet, attempt)
        feedback = None
        if context.best_attempt:
            folder = self.root / context.best_attempt["path"]
            if (folder / "vision/depth.png").is_file():
                feedback = {
                    "depth": (folder / "vision/depth.png").read_bytes(),
                }
        return replace(
            context,
            visual_inputs=visuals,
            target_evidence=item.get("target_evidence"),
            analysis=plan,
            visual_feedback=feedback,
        )

    def _evaluate(self, item: dict, target, sheet: bytes, attempt: dict) -> None:
        super()._evaluate(item, target, sheet, attempt)
        execution = attempt.get("execution") or {}
        if execution.get("status") != "valid":
            return
        try:
            with self._stage("depth_comparison", item["id"], attempt["iteration"]):
                self._check()
                folder = self.root / attempt["path"]
                prediction = render_observations(
                    load_mesh(execution["mesh_path"]),
                    target.extents,
                    cancel_check=lambda: self._stopped() is not None,
                )
                save_observations(prediction, folder / "vision")
                reference = load_observations(self.root / "items" / item["id"] / "vision")
                comparison = compare_observations(reference, prediction)
                write_json(folder / "visual-metrics.json", comparison["metrics"])
                attempt["visual_metrics"] = comparison["metrics"]
                self._save()
        except Exception as exc:
            self._check()
            attempt["visual_error"] = self._safe_error(exc)
            self._event(
                "depth_comparison_error",
                level="WARNING",
                item_id=item["id"],
                error=attempt["visual_error"],
            )
