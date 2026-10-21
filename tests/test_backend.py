"""Credential-free contract tests for the real subprocess/JSONL boundary."""

import json
import sys
from pathlib import Path

import pytest

from mesh2cad.backend import CodexBackend


def _events(payload=None):
    if payload is None:
        payload = {"code": "import cadquery as cq\nr = cq.Workplane('XY').box(1, 1, 1)"}
    return [
        {"type": "thread.started", "thread_id": "fake-thread"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "r", "type": "reasoning", "text": "Plan"}},
        {
            "type": "item.completed",
            "item": {"id": "a", "type": "agent_message", "text": json.dumps(payload)},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 11, "cached_input_tokens": 2, "output_tokens": 7},
        },
    ]


def _backend(tmp_path, events=None, *, extra="", timeout=5):
    executable = tmp_path / "fake-codex"
    capture = tmp_path / "capture.json"
    source = (
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        f"with open({str(capture)!r}, 'w') as f:\n"
        "    json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'prompt': sys.stdin.read()}, f)\n"
        f"for event in {events or []!r}:\n"
        "    print(json.dumps(event), flush=True)\n" + extra
    )
    executable.write_text(source, encoding="utf-8")
    executable.chmod(0o755)
    image = tmp_path / "views.png"
    image.write_bytes(b"fake image; CLI stub does not decode it")
    return CodexBackend(executable=str(executable), timeout=timeout), image, capture


def test_success_persists_artifacts_and_uses_fresh_tool_free_session(tmp_path):
    backend, image, capture_path = _backend(tmp_path, _events())
    output = tmp_path / "attempt"
    result = backend.generate("Propose a cube", image, output)
    assert result["status"] == "ok", result
    assert "r = " in result["code"]
    assert result["usage"] == {"input_tokens": 11, "cached_input_tokens": 2, "output_tokens": 7}
    assert result["launches"] == 1
    capture = json.loads(capture_path.read_text())
    argv = capture["argv"]
    for flag in (
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
    ):
        assert flag in argv
    assert argv[argv.index("--model") + 1] == "gpt-6-luna"
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[-3:] == ["--image", str(image.resolve()), "-"]
    assert 'web_search="disabled"' in argv
    assert "suppress_unstable_features_warning=true" in argv
    for feature in (
        "shell_tool",
        "apps",
        "plugins",
        "multi_agent",
        "hooks",
        "memories",
        "view_image",
        "browser_use",
        "computer_use",
    ):
        assert ["--disable", feature] in [argv[i : i + 2] for i in range(len(argv) - 1)]
    assert capture["prompt"] == "Propose a cube"
    assert Path(capture["cwd"]) != tmp_path
    assert not Path(capture["cwd"]).exists()
    for name in (
        "events.jsonl",
        "stderr.txt",
        "prompt.txt",
        "response_schema.json",
        "final_response.txt",
        "command.json",
        "generation.json",
    ):
        assert (output / name).is_file()
    metadata = json.loads((output / "command.json").read_text())
    assert metadata["terminal_event"] == "turn.completed"
    assert metadata["exit_code"] == 0
    backend.generate("Second", image, tmp_path / "second")
    assert json.loads(capture_path.read_text())["cwd"] != capture["cwd"]


@pytest.mark.parametrize(
    "message",
    [
        "You've hit your usage limit. Try again at 8:30 PM.",
        "Your plan limit resets at 18:00 UTC.",
        "Rate limit exceeded for this model",
        "Insufficient quota",
        "Too many requests",
    ],
)
def test_quota_is_distinct_and_trace_is_preserved(tmp_path, message):
    events = _events()[:2] + [{"type": "turn.failed", "error": {"message": message}}]
    backend, image, _ = _backend(tmp_path, events)
    output = tmp_path / "attempt"
    result = backend.generate("cube", image, output)
    assert result["status"] == "quota_exhausted"
    assert result["code"] is None
    assert message in result["error"]
    assert '"turn.failed"' in (output / "events.jsonl").read_text()
    assert json.loads((output / "command.json").read_text())["terminal_event"] == "turn.failed"


def test_quota_error_code_and_stderr_are_recognized(tmp_path):
    events = [
        {"type": "error", "error": {"code": "rate_limit_exceeded", "message": "Reset at 09:30"}}
    ]
    backend, image, _ = _backend(tmp_path, events)
    assert backend.generate("cube", image, tmp_path / "code")["status"] == "quota_exhausted"
    backend, image, _ = _backend(
        tmp_path, extra="print('usage limit exceeded', file=sys.stderr)\nsys.exit(1)\n"
    )
    assert backend.generate("cube", image, tmp_path / "stderr")["status"] == "quota_exhausted"


@pytest.mark.parametrize(
    "item_type",
    [
        "command_execution",
        "mcp_tool_call",
        "file_change",
        "web_search",
        "collab_tool_call",
        "tool_call",
        "unknown_tool",
    ],
)
def test_tool_and_unknown_items_abort_immediately(tmp_path, item_type):
    events = _events()[:2] + [{"type": "item.started", "item": {"id": "x", "type": item_type}}]
    backend, image, _ = _backend(tmp_path, events, extra="time.sleep(60)\n")
    result = backend.generate("cube", image, tmp_path / "attempt")
    assert result["status"] == "protocol_violation"
    assert item_type in result["error"]
    assert result["elapsed_seconds"] < 3
    assert result["code"] is None


@pytest.mark.parametrize(
    "payload", [{"wrong": "code"}, {"code": "x", "extra": 1}, {"code": ""}, {"code": 3}, ["code"]]
)
def test_invalid_structured_payload_fails_closed(tmp_path, payload):
    backend, image, _ = _backend(tmp_path, _events(payload))
    result = backend.generate("cube", image, tmp_path / "attempt")
    assert result["status"] == "protocol_violation"
    assert result["code"] is None


def test_malformed_jsonl_is_a_protocol_violation(tmp_path):
    backend, image, _ = _backend(tmp_path, extra="print('not JSON', flush=True)\n")
    result = backend.generate("cube", image, tmp_path / "attempt")
    assert result["status"] == "protocol_violation"
    assert "Malformed JSONL" in result["error"]


def test_unknown_event_fails_closed_even_after_completed_turn(tmp_path):
    backend, image, _ = _backend(tmp_path, _events() + [{"type": "new.tool.event"}])
    assert backend.generate("cube", image, tmp_path / "attempt")["status"] == "protocol_violation"


def test_missing_terminal_event_is_not_a_success(tmp_path):
    backend, image, _ = _backend(tmp_path, _events()[:-1])
    result = backend.generate("cube", image, tmp_path / "attempt")
    assert result["status"] == "error"
    assert "without turn.completed" in result["error"]


def test_timeout_preserves_partial_trace_and_stderr(tmp_path):
    backend, image, _ = _backend(
        tmp_path,
        _events()[:2],
        timeout=0.75,
        extra="print('waiting', file=sys.stderr, flush=True)\ntime.sleep(60)\n",
    )
    output = tmp_path / "attempt"
    result = backend.generate("cube", image, output)
    assert result["status"] == "error"
    assert "timed out" in result["error"]
    assert result["elapsed_seconds"] < 3
    assert '"turn.started"' in (output / "events.jsonl").read_text()
    assert "waiting" in (output / "stderr.txt").read_text()


def test_process_error_does_not_use_a_stale_final_response(tmp_path):
    backend, image, _ = _backend(
        tmp_path, extra="print('unsupported model', file=sys.stderr)\nsys.exit(2)\n"
    )
    output = tmp_path / "attempt"
    output.mkdir()
    (output / "final_response.txt").write_text('{"code":"stale"}')
    result = backend.generate("cube", image, output)
    assert result["status"] == "error"
    assert result["code"] is None
    assert "unsupported model" in result["error"]
    assert (output / "final_response.txt").read_text() == ""


def test_failed_checkpoint_commit_preserves_trace_without_a_partial_result(tmp_path, monkeypatch):
    from mesh2cad import storage

    backend, image, _ = _backend(tmp_path, _events())
    output = tmp_path / "attempt"

    def fail_flush(_fd):
        raise OSError("simulated checkpoint flush failure")

    monkeypatch.setattr(storage.os, "fsync", fail_flush)
    with pytest.raises(OSError, match="checkpoint flush failure"):
        backend.generate("cube", image, output)
    assert not (output / "generation.json").exists()
    assert '"turn.completed"' in (output / "events.jsonl").read_text()
    assert json.loads((output / "final_response.txt").read_text())["code"]


@pytest.mark.parametrize("position", [1, 2])
def test_nonfatal_cli_diagnostics_before_or_within_turn_allow_success(tmp_path, position):
    messages = [
        "Under-development features enabled: skip_host_skill_discovery. "
        "Under-development features are incomplete and may behave unpredictably. "
        "To suppress this warning, set `suppress_unstable_features_warning = true` "
        "in /Users/egoriy/.codex/config.toml.",
        "Code Mode is unavailable because code-mode host is disabled. "
        "Code mode will fail closed; enable `features.code_mode_host` "
        "and install `codex-code-mode-host`.",
    ]
    events = _events()
    diagnostics = [
        {"type": "item.completed", "item": {"id": f"d{i}", "type": "error", "message": message}}
        for i, message in enumerate(messages)
    ]
    events[position:position] = diagnostics
    backend, image, _ = _backend(tmp_path, events)
    output = tmp_path / "attempt"
    result = backend.generate("cube", image, output)
    assert result["status"] == "ok", result
    assert result["diagnostics"] == messages
    assert json.loads((output / "generation.json").read_text())["diagnostics"] == messages


def test_diagnostic_quota_item_before_turn_stops_invocation(tmp_path):
    events = _events()[:1] + [
        {
            "type": "item.completed",
            "item": {
                "id": "d",
                "type": "error",
                "message": "Your usage limit resets at 18:00 UTC.",
            },
        }
    ]
    backend, image, _ = _backend(tmp_path, events, extra="time.sleep(60)\n")
    result = backend.generate("cube", image, tmp_path / "attempt")
    assert result["status"] == "quota_exhausted"
    assert result["code"] is None
    assert result["elapsed_seconds"] < 3


@pytest.mark.parametrize("message", [None, 7, {"text": "warning"}])
def test_diagnostic_item_requires_string_message(tmp_path, message):
    events = _events()[:1] + [
        {
            "type": "item.completed",
            "item": {"id": "d", "type": "error", "message": message},
        }
    ]
    backend, image, _ = _backend(tmp_path, events)
    result = backend.generate("cube", image, tmp_path / "attempt")
    assert result["status"] == "protocol_violation"
