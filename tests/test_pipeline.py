"""Pipeline tests: exact payload, code extraction, and durable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh2cad.pipeline import (
    PipelineError,
    api_key_from_environment,
    build_payload,
    call_openrouter,
    extract_code,
    normalize_key,
    payload_digest,
    preview_payload,
    save_run,
)
from mesh2cad.prompts import overlay_repair_prompt

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-for-payload-tests"
IMAGES = [
    (label, PNG + label.encode("ascii"))
    for label in (
        "top-rgb",
        "top-depth",
        "left-rgb",
        "left-depth",
        "right-rgb",
        "right-depth",
    )
]


def make_payload() -> dict:
    return build_payload(
        model="openai/test-model",
        system_prompt="Return JSON.",
        request="Reconstruct this part.",
        images=IMAGES,
        extents=[1.0, 0.5, 0.25],
        reasoning_effort="low",
    )


def test_payload_has_exact_structure() -> None:
    payload = make_payload()
    assert payload["model"] == "openai/test-model"
    assert payload["max_tokens"] == 4096
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][1]["content"][0]["text"].startswith(
        "Target extents (normalized units, x/y/z): [1, 0.5, 0.25]"
    )
    content = payload["messages"][1]["content"]
    assert len(content) == 1 + 2 * len(IMAGES)
    assert content[1] == {"type": "text", "text": "Image: top-rgb"}
    image_urls = [part for part in content if part["type"] == "image_url"]
    assert len(image_urls) == len(IMAGES)
    assert all(part["image_url"]["url"].startswith("data:image/png;base64,") for part in image_urls)


def test_single_sheet_payload() -> None:
    payload = build_payload(
        model="openai/test-model",
        system_prompt="Return JSON.",
        request="Reconstruct this part.",
        images=[("views-rgb-depth", PNG)],
        extents=[1.0, 0.5, 0.25],
    )
    content = payload["messages"][1]["content"]
    assert len(content) == 3
    assert content[1] == {"type": "text", "text": "Image: views-rgb-depth"}
    assert content[2]["type"] == "image_url"


def test_preview_hides_only_the_base64_images() -> None:
    payload = make_payload()
    preview = preview_payload(payload)
    content = preview["messages"][1]["content"]
    assert content[0] == payload["messages"][1]["content"][0]
    for part in content:
        if part["type"] == "image_url":
            assert len(part["image_url"]["url"]) < 100


def test_user_text_is_not_rewritten() -> None:
    payload = build_payload(
        model="openai/test-model",
        system_prompt="  keep surrounding spaces  ",
        request="  exact request  ",
        images=IMAGES,
        extents=[1.0, 0.5, 0.25],
    )
    assert payload["messages"][0]["content"] == "  keep surrounding spaces  "
    assert payload["messages"][1]["content"][0]["text"].endswith("Request:   exact request  ")


def test_payload_digest_is_stable() -> None:
    assert payload_digest(make_payload()) == payload_digest(make_payload())


def test_overlay_repair_prompt_adds_the_scheme_and_code() -> None:
    prompt = overlay_repair_prompt("BASE PROMPT", 'r = cq.Workplane("XY").box(1, 1, 1)')
    assert prompt.startswith("BASE PROMPT")
    assert "contrast overlay" in prompt
    assert "red" in prompt
    assert "Previous CadQuery program" in prompt
    assert 'r = cq.Workplane("XY").box(1, 1, 1)' in prompt


def test_extract_code() -> None:
    response = {"choices": [{"message": {"content": '{"code": "r = 1"}'}}]}
    assert extract_code(response) == "r = 1"


def test_extract_code_accepts_json_and_python_fences() -> None:
    content = '```json\n{"code": "```python\\nr = 2\\n```"}\n```'
    response = {"choices": [{"message": {"content": content}}]}
    assert extract_code(response) == "r = 2"


def test_extract_code_rejects_non_json() -> None:
    with pytest.raises(PipelineError):
        extract_code({"choices": [{"message": {"content": "r = 1"}}]})


def test_normalize_key() -> None:
    assert normalize_key("  Bearer secret  ") == "secret"
    with pytest.raises(PipelineError):
        normalize_key("two words")


def test_api_key_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-file\n", encoding="utf-8")
    assert api_key_from_environment(tmp_path) == "from-file"


def test_save_run_writes_exact_inputs(tmp_path: Path) -> None:
    payload = make_payload()
    response = {
        "choices": [{"message": {"content": '{"code": "r = 3"}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    folder = save_run(
        tmp_path,
        source_name="part.stl",
        source_bytes=b"solid part\nendsolid part\n",
        images=dict(IMAGES),
        payload=payload,
        response=response,
        code="r = 3",
        extra_images={"top-rgb": PNG},
    )
    assert (folder / "request.json").is_file()
    assert (folder / "response.json").is_file()
    assert (folder / "code.py").read_text(encoding="utf-8") == "r = 3\n"
    assert (folder / "source.stl").is_file()
    for label in dict(IMAGES):
        assert (folder / "llm-images" / f"{label}.png").is_file()
    assert (folder / "renders" / "top-rgb.png").is_file()
    saved = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    assert saved == payload
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["images"]) == set(dict(IMAGES))
    assert set(meta["renders"]) == {"top-rgb"}
    assert "api_key" not in (folder / "meta.json").read_text(encoding="utf-8").lower()


def test_call_openrouter_uses_authorization_without_saving_it(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"code": "r = 4"}'}}]}

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            return None

        def post(self, url, headers, json):
            captured.update(url=url, headers=headers, json=json)
            return FakeResponse()

    monkeypatch.setattr("mesh2cad.pipeline.httpx.Client", FakeClient)
    payload = make_payload()
    result = call_openrouter(payload, "super-secret")
    assert captured["headers"]["Authorization"] == "Bearer super-secret"
    assert captured["json"] == payload
    assert result["_elapsed_seconds"] >= 0


def test_call_openrouter_does_not_leak_key_in_error(monkeypatch) -> None:
    class FakeResponse:
        status_code = 401
        text = "unauthorized"

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            return None

        def post(self, *_args, **_kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("mesh2cad.pipeline.httpx.Client", FakeClient)
    with pytest.raises(PipelineError) as error:
        call_openrouter(make_payload(), "super-secret")
    assert "super-secret" not in str(error.value)
    assert "401" in str(error.value)
