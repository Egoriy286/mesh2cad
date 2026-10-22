"""A reconstruction agent; execution and scoring belong to the harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .pipeline import build_payload, call_openrouter
from .prompts import DEFAULT_REQUEST, SYSTEM_PROMPT


@dataclass(frozen=True)
class AgentContext:
    extents: list[float]
    image: bytes
    attempts: list[dict[str, Any]]
    best_attempt: dict[str, Any] | None
    overlay: bytes | None = None
    max_tokens: int = 4096
    visual_inputs: dict | None = None
    analysis: dict | None = None
    target_evidence: dict | None = None
    visual_feedback: dict | None = None


class CadAgent(Protocol):
    def build_request(self, context: AgentContext) -> dict: ...

    def complete(self, payload: dict, api_key: str, timeout: float) -> dict: ...


class OpenRouterAgent:
    """Keep visual context and a bounded conversation of actual tool feedback."""

    def __init__(
        self,
        model: str,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        request: str = DEFAULT_REQUEST,
        reasoning_effort: str = "medium",
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.request = request
        self.reasoning_effort = reasoning_effort

    def build_request(self, context: AgentContext) -> dict:
        payload = build_payload(
            model=self.model,
            system_prompt=self.system_prompt,
            request=self.request,
            images=[("target-rgb-depth", context.image)],
            extents=context.extents,
            reasoning_effort=self.reasoning_effort,
            max_tokens=context.max_tokens,
        )
        for attempt in context.attempts[-3:]:
            payload["messages"].append(
                {
                    "role": "assistant",
                    "content": json.dumps({"code": attempt.get("code", "")}),
                }
            )
            feedback = {
                "iteration": attempt["iteration"],
                "status": attempt["status"],
                "error": (attempt.get("error") or "")[-12000:],
                "iou": attempt.get("iou"),
                "difference": attempt.get("difference"),
            }
            payload["messages"].append(
                {
                    "role": "user",
                    "content": "Harness tool results:\n"
                    + json.dumps(feedback, ensure_ascii=False)
                    + "\nFix execution errors or improve geometry. Return the complete program.",
                }
            )
        if context.best_attempt and context.attempts:
            best = context.best_attempt
            payload["messages"].append(
                {
                    "role": "user",
                    "content": f"Best evaluated program so far (IoU={best['iou']['iou']}):\n"
                    + best["code"],
                }
            )
        if context.overlay:
            overlay_payload = build_payload(
                model=self.model,
                system_prompt=self.system_prompt,
                request=(
                    "Contrast overlay of the BEST program above: green = volume intersection, "
                    "blue = excess CAD outside the target mesh (candidate minus target), "
                    "red = missing CAD (target minus candidate). Tile order matches the original. "
                    "Use the original target image for dimensions. Remove blue and restore red geometry."
                ),
                images=[("best-contrast-overlay", context.overlay)],
                extents=context.extents,
                reasoning_effort=self.reasoning_effort,
            )
            payload["messages"].append(overlay_payload["messages"][1])
        return payload

    def complete(self, payload: dict, api_key: str, timeout: float) -> dict:
        return call_openrouter(payload, api_key, timeout=timeout)


class DepthAwareAgent(OpenRouterAgent):
    """A geometry analysis pass followed by grounded CAD generation/repair."""

    def _visual_request(self, context: AgentContext, *, analysis=False) -> dict:
        from .prompts import VISION_ANALYSIS_PROMPT, VISION_ANALYSIS_SYSTEM_PROMPT

        bundle = context.visual_inputs
        if bundle is None:
            raise ValueError("Orchestrator 2 requires calibrated linear depth")
        request = (
            self.request
            + "\n"
            + (
                VISION_ANALYSIS_PROMPT
                if analysis
                else "Build the complete CAD program from the target depth sheet and verified dimensions."
            )
        )
        payload = build_payload(
            model=self.model,
            system_prompt=VISION_ANALYSIS_SYSTEM_PROMPT if analysis else self.system_prompt,
            request=request,
            images=[("target-linear-depth", bundle["depth"])],
            extents=context.extents,
            reasoning_effort="low" if analysis else self.reasoning_effort,
            max_tokens=context.max_tokens,
        )
        payload["messages"][1]["content"].append(
            {
                "type": "text",
                "text": "Camera and linear depth calibration:\n"
                + json.dumps(bundle["calibration"])
                + "\nMeasured target sections, volume and centroid:\n"
                + json.dumps(context.target_evidence),
            }
        )
        return payload

    def build_analysis_request(self, context: AgentContext) -> dict:
        return self._visual_request(context, analysis=True)

    def build_request(self, context: AgentContext) -> dict:
        payload = self._visual_request(context)
        if context.analysis:
            payload["messages"][1]["content"].append(
                {
                    "type": "text",
                    "text": "Prior geometric analysis (cross-check against the images and measured bounds):\n"
                    + json.dumps(context.analysis),
                }
            )
        for attempt in context.attempts[-2:]:
            payload["messages"].extend(
                [
                    {"role": "assistant", "content": json.dumps({"code": attempt.get("code", "")})},
                    {
                        "role": "user",
                        "content": "Actual tool results:\n"
                        + json.dumps(
                            {
                                "iteration": attempt["iteration"],
                                "status": attempt["status"],
                                "error": (attempt.get("error") or "")[-8000:],
                                "iou": attempt.get("iou"),
                                "surface_difference": attempt.get("difference"),
                                "view_metrics": attempt.get("visual_metrics"),
                            }
                        )
                        + "\nCorrect the underlying construction. Preserve successful dimensions; "
                        "repair holes, walls and side placement. Return the complete program.",
                    },
                ]
            )
        if context.best_attempt:
            payload["messages"].append(
                {
                    "role": "user",
                    "content": f"BEST actual CAD (volume IoU={context.best_attempt['iou']['iou']}):\n"
                    + context.best_attempt["code"],
                }
            )
        if context.visual_feedback:
            feedback = context.visual_feedback
            extra = build_payload(
                model=self.model,
                system_prompt=self.system_prompt,
                request="This depth sheet belongs to the BEST CAD above, in the target camera frame. "
                "Compare target and prediction linear depth in the same view and depth scale. "
                "Depth/silhouette errors are view evidence; the reported volume IoU is the score. "
                "Use the largest construction error to guide the repair. Return JSON with "
                "a single string field code containing the COMPLETE executable CadQuery Python "
                "program. Include any diagnosis only as Python comments inside that program. "
                "An explanation without a complete CAD program is not an acceptable answer.",
                images=[
                    ("best-cad-linear-depth", feedback["depth"]),
                ],
                extents=context.extents,
            )
            payload["messages"].append(extra["messages"][1])
        return payload
