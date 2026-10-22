"""Transparent LLM input/output handling. No hidden prompt rewriting."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-6-luna"
DEFAULT_MAX_TOKENS = 4096
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


class PipelineError(RuntimeError):
    """A user-facing failure with no credentials in the message."""


def normalize_key(key: str) -> str:
    key = (key or "").strip().strip("`\"'").strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if any(character.isspace() for character in key):
        raise PipelineError("API-ключ содержит пробелы или перенос строки")
    return key


def api_key_from_environment(project_root: str | Path) -> str:
    """Read OPENROUTER_API_KEY from the environment, then from a local .env file."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return normalize_key(key)
    env_path = Path(project_root) / ".env"
    if not env_path.is_file():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == "OPENROUTER_API_KEY":
            return normalize_key(value)
    return ""


def build_user_text(extents: list[float] | tuple[float, ...], request: str) -> str:
    if len(extents) != 3:
        raise ValueError("extents must contain exactly three numbers")
    x, y, z = (float(value) for value in extents)
    if not request.strip():
        request = "Reconstruct the part shown in the four views."
    return (
        f"Target extents (normalized units, x/y/z): [{x:.8g}, {y:.8g}, {z:.8g}]\nRequest: {request}"
    )


def build_payload(
    *,
    model: str,
    system_prompt: str,
    request: str,
    image_png: bytes,
    extents: list[float] | tuple[float, ...],
    reasoning_effort: str = "medium",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    if not image_png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("LLM input must be a PNG image")
    if not model.strip():
        raise ValueError("Model name is empty")
    if not system_prompt.strip():
        raise ValueError("System prompt is empty")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"Unknown reasoning effort: {reasoning_effort}")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    data_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
    return {
        "model": model.strip(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_user_text(extents, request)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "reasoning": {"effort": reasoning_effort},
        "response_format": {"type": "json_object"},
        "usage": {"include": True},
    }


def preview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the exact request with the base64 image replaced for on-screen display."""
    preview = json.loads(json.dumps(payload))
    for message in preview.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    part["image_url"]["url"] = (
                        f"data:image/png;base64,<{len(url)} chars omitted in preview>"
                    )
    return preview


def payload_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def call_openrouter(
    payload: dict[str, Any], api_key: str, timeout: float = 180.0
) -> dict[str, Any]:
    key = normalize_key(api_key)
    if not key:
        raise PipelineError("Введите API-ключ OpenRouter")
    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.HTTPError as exc:
        # Deliberately do not include request headers: they contain the secret.
        raise PipelineError(f"Сетевая ошибка OpenRouter: {type(exc).__name__}") from exc
    if response.status_code != 200:
        detail = response.text.strip().replace("\n", " ")[:1000]
        raise PipelineError(f"OpenRouter HTTP {response.status_code}: {detail}")
    try:
        data = response.json()
    except ValueError as exc:
        raise PipelineError("OpenRouter вернул не JSON") from exc
    if not isinstance(data, dict):
        raise PipelineError("Ответ OpenRouter должен быть JSON-объектом")
    data["_elapsed_seconds"] = round(time.monotonic() - started, 3)
    return data


def _strip_code_fence(value: str) -> str:
    value = value.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return value
    return "\n".join(lines[1:-1]).strip()


def extract_code(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PipelineError("В ответе нет choices[0].message.content") from exc
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        )
    if not isinstance(content, str) or not content.strip():
        raise PipelineError("Модель вернула пустой ответ")
    text = _strip_code_fence(content)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Ответ модели не является JSON: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("code"), str):
        raise PipelineError('Ответ должен быть объектом с полем "code"')
    code = _strip_code_fence(value["code"])
    if not code:
        raise PipelineError("Поле code пустое")
    if len(code) > 60_000:
        raise PipelineError("Поле code длиннее 60000 символов")
    return code


def save_run(
    root: str | Path,
    *,
    source_name: str,
    source_bytes: bytes,
    image_png: bytes,
    payload: dict[str, Any],
    response: dict[str, Any],
    code: str,
) -> Path:
    """Persist the exact request, response, and inputs next to each other."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(root) / f"{stamp}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "source.stl").write_bytes(source_bytes)
    (folder / "llm-input-512.png").write_bytes(image_png)
    (folder / "request.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (folder / "response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (folder / "code.py").write_text(code + "\n", encoding="utf-8")
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "image_sha256": hashlib.sha256(image_png).hexdigest(),
        "request_sha256": payload_digest(payload),
        "model": payload.get("model"),
        "usage": response.get("usage"),
        "elapsed_seconds": response.get("_elapsed_seconds"),
    }
    (folder / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return folder
