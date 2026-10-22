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
                    "Contrast overlay of the BEST program above: gray = covered target, "
                    "blue = prediction, red = missing target. Tile order matches the original. "
                    "Use the original target image for dimensions and remove excess blue geometry."
                ),
                images=[("best-contrast-overlay", context.overlay)],
                extents=context.extents,
                reasoning_effort=self.reasoning_effort,
            )
            payload["messages"].append(overlay_payload["messages"][1])
        return payload

    def complete(self, payload: dict, api_key: str, timeout: float) -> dict:
        return call_openrouter(payload, api_key, timeout=timeout)
