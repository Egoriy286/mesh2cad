"""Tool-free OpenRouter transport; credentials never enter artifacts or SQLite."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time

import httpx

SUPPORTED_MODEL = "openai/gpt-6-luna"
CALL_RESERVE = 0.107048
AUTH_ERROR = (
    "OpenRouter отклонил API-ключ (401). Откройте Настройки, вставьте действующий "
    "ключ и нажмите «Проверить и применить ключ». Резерв этого запроса освобождён."
)


class Cancelled(Exception):
    pass


class OpenRouterHTTPError(RuntimeError):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(
            AUTH_ERROR
            if status_code == 401
            else f"OpenRouter HTTP {status_code}; reservation retained pending reconciliation"
        )


def normalize_key(key):
    key = key.strip().strip("`\"'").strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if any(character.isspace() for character in key):
        raise ValueError("API-ключ содержит пробелы или перенос строки. Вставьте только ключ.")
    return key


def verify_key(key):
    """Read account-key metadata only; never submit a model/inference request."""
    if not key:
        return {"status": "missing", "message": "Введите API-ключ OpenRouter."}
    try:
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            with client.stream(
                "GET",
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {key}"},
            ) as response:
                if response.status_code == 200:
                    return {
                        "status": "valid",
                        "message": "Ключ подтверждён OpenRouter. Модель не вызывалась.",
                    }
                if response.status_code == 401:
                    return {
                        "status": "invalid",
                        "message": "OpenRouter отклонил ключ (401). Проверьте его или создайте новый ключ.",
                    }
                return {
                    "status": "unverified",
                    "message": f"Проверка недоступна: OpenRouter HTTP {response.status_code}. Модель не вызывалась.",
                }
    except httpx.HTTPError:
        # Do not expose an exception that could include request headers.
        return {
            "status": "unverified",
            "message": "Не удалось связаться с OpenRouter для проверки ключа. Модель не вызывалась.",
        }


async def _request(key, payload, cancel_check):
    async with httpx.AsyncClient(timeout=180, follow_redirects=False) as client:

        async def receive():
            async with client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            ) as response:
                if response.status_code != 200:
                    raise OpenRouterHTTPError(response.status_code)
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > 1_000_000:
                        raise RuntimeError("OpenRouter response exceeded 1 MB protocol limit")
                return json.loads(data)

        task = asyncio.create_task(receive())
        deadline = time.monotonic() + 180
        try:
            while not task.done():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "OpenRouter request exceeded 180 second deadline; reservation retained"
                    )
                if cancel_check():
                    raise Cancelled()
                await asyncio.sleep(0.05)
            return await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


def invoke(key, job, asset, parent, image, cancel_check):
    explain = job["kind"] == "explain"
    system = (
        "You are a CAD assistant. Explain the selected CadQuery program and its limitations in plain text. Do not emit executable code."
        if explain
        else 'Return exactly one JSON object with a nonempty "code" string. Create a valid CadQuery solid assigned to r. Use import cadquery as cq. Coordinates are target-normalized: bounding-box center zero, longest side one. Do not align or rescale predictions. No files, network, or tools.'
    )
    text = f"Target extents: {asset['stats']['normalized_extents']}\nRequest: {job.get('prompt', '')}\n"
    if parent:
        text += (
            "Selected immutable parent code:\n"
            + parent["code"]
            + "\nMeasured geometry feedback (null means unavailable):\n"
            + json.dumps(parent.get("metrics"))
        )
    if job.get("repair_error"):
        text += "\nPrevious execution error:\n" + job["repair_error"][-4000:]
    content = [{"type": "text", "text": text}]
    if image.is_file() and not image.is_symlink():
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
                },
            }
        )
    payload = {
        "model": job["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "max_tokens": 4096,
        "reasoning": {"effort": job["reasoning_effort"]},
        "usage": {"include": True},
        "provider": {
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {"prompt": 0.1, "completion": 0.5},
        },
    }
    if not explain:
        payload["response_format"] = {"type": "json_object"}
    data = asyncio.run(_request(key, payload, cancel_check))
    if not isinstance(data, dict):
        raise RuntimeError("OpenRouter response must be a JSON object")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    cost = usage.get("cost")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost < 0
    ):
        cost = None
    # Commit cost before validating response content: malformed answers can cost money.
    clean_usage = {
        name: value
        for name, value in usage.items()
        if name in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    }
    clean_usage["cost"] = cost
    return data, cost, clean_usage


def response_content(data, explain=False):
    try:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip() or len(content) > 100_000:
            raise ValueError()
        if explain:
            return content
        obj = json.loads(content)
        if (
            set(obj) != {"code"}
            or not isinstance(obj["code"], str)
            or not obj["code"].strip()
            or len(obj["code"]) > 60_000
        ):
            raise ValueError()
        return obj["code"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("Model response violated the plain-text/code protocol") from exc
