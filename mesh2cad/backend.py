"""A deliberately tool-free, audited adapter for non-interactive Codex.

The installed Codex CLI retains its normal login, but each invocation starts in
an empty temporary directory and ignores user configuration and policy rules.
The JSONL audit is an additional guard: an unexpected tool or event aborts the
invocation rather than silently contaminating an experiment.
"""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .storage import write_json

_DISABLED_FEATURES = (
    "shell_tool",
    "shell_snapshot",
    "shell_snapshot_v2",
    "unified_exec",
    "unified_exec_tty",
    "apps",
    "plugins",
    "hooks",
    "memories",
    "multi_agent",
    "multi_agent_v2",
    "view_image",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "in_app_browser",
    "in_app_local_automation",
    "image_generation",
    "code_mode",
    "code_mode_host",
    "skill_search",
    "skill_mcp_dependency_install",
    "workspace_dependencies",
    "sleep_tool",
    "request_permissions_tool",
)
_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string", "minLength": 1}},
    "required": ["code"],
    "additionalProperties": False,
}
_QUOTA_CODES = {
    "rate_limit_exceeded",
    "rate_limit_error",
    "rate_limit",
    "usage_limit_reached",
    "usage_limit_exceeded",
    "insufficient_quota",
    "quota_exceeded",
    "over_quota",
    "plan_limit_exceeded",
}
_QUOTA_PATTERNS = re.compile(
    r"quota.{0,30}(?:exceed|exhaust|insufficient)|insufficient.{0,20}quota|"
    r"(?:rate|usage|plan|subscription)[ _-]+limit|too many requests|"
    r"(?:you(?:'|’)?ve|you have).{0,15}hit.{0,30}limit|"
    r"(?:limit|quota).{0,25}reset(?:s|ting)?\b",
    re.IGNORECASE,
)


def _is_quota_error(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"code", "type", "error_code"} and str(item).lower() in _QUOTA_CODES:
                return True
            if key in {"status", "status_code", "http_status"} and str(item) == "429":
                return True
            if _is_quota_error(item):
                return True
        return False
    if isinstance(value, list):
        return any(_is_quota_error(item) for item in value)
    return isinstance(value, str) and bool(_QUOTA_PATTERNS.search(value))


def _error_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("message"), str):
            return value["message"]
        if "error" in value:
            return _error_text(value["error"])
    return json.dumps(value, ensure_ascii=False)


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Stop the whole invocation, including any accidentally spawned child."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    # A child can outlive the group leader (and can ignore SIGTERM). Always
    # clean up the remaining group, even if waiting for the leader succeeded.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=1)


class _Audit:
    """Only ordinary assistant text/reasoning and one terminal turn are allowed."""

    def __init__(self) -> None:
        self.thread_started = False
        self.turn_started = False
        self.terminal: str | None = None
        self.final_text: str | None = None
        self.usage: dict[str, Any] = {}
        self.error: str | None = None
        self.quota = False
        self.diagnostics: list[str] = []

    def feed(self, line: bytes) -> None:
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed JSONL event: {exc}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("Event must be an object with a string type")
        kind = event["type"]
        if self.terminal is not None:
            raise ValueError(f"Unexpected event after {self.terminal}: {kind}")
        if kind == "thread.started":
            if self.thread_started or self.turn_started:
                raise ValueError("Duplicate or out-of-order thread.started event")
            self.thread_started = True
        elif kind == "turn.started":
            if self.turn_started or not self.thread_started:
                raise ValueError("Duplicate or out-of-order turn.started event")
            self.turn_started = True
        elif kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item")
            if kind == "item.completed" and isinstance(item, dict) and item.get("type") == "error":
                # CLI startup diagnostics use completed error items, including
                # before turn.started. They are text, never tool executions.
                message = item.get("message")
                if not isinstance(message, str):
                    raise ValueError("Diagnostic error item has no string message")
                self.diagnostics.append(message)
                if _is_quota_error(item):
                    self.quota, self.error = True, message
                return
            if not isinstance(item, dict) or item.get("type") not in {"reasoning", "agent_message"}:
                item_type = item.get("type") if isinstance(item, dict) else None
                raise ValueError(f"Forbidden or unknown item type: {item_type!r}")
            if not self.turn_started:
                raise ValueError("Item outside an active turn")
            if kind == "item.completed" and item["type"] == "agent_message":
                if not isinstance(item.get("text"), str):
                    raise ValueError("Completed agent_message has no string text")
                self.final_text = item["text"]
        elif kind == "turn.completed":
            if not self.turn_started:
                raise ValueError("turn.completed without turn.started")
            usage = event.get("usage")
            if not isinstance(usage, dict) or any(
                not isinstance(number, int) or isinstance(number, bool) or number < 0
                for number in usage.values()
            ):
                raise ValueError("turn.completed usage must contain nonnegative integer counters")
            self.usage = usage
            self.terminal = kind
        elif kind in {"turn.failed", "error"}:
            details = event.get("error", event)
            self.error = _error_text(details)
            self.quota = _is_quota_error(details)
            if kind == "turn.failed":
                self.terminal = kind
        else:
            raise ValueError(f"Forbidden or unknown event type: {kind!r}")


class CodexBackend:
    """Generate one CadQuery proposal using a fresh, audited Codex session."""

    def __init__(
        self,
        model: str = "gpt-6-luna",
        reasoning_effort: str = "medium",
        timeout: float = 180,
        *,
        executable: str = "codex",
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if reasoning_effort not in {"minimal", "low", "medium", "high", "xhigh", "none"}:
            raise ValueError("Unsupported reasoning effort")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.executable = executable

    def _command(self, image_path: Path, output_dir: Path) -> list[str]:
        command = [
            self.executable,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--json",
            "--color",
            "never",
            "--model",
            self.model,
            "--output-schema",
            str(output_dir / "response_schema.json"),
            "--output-last-message",
            str(output_dir / "final_response.txt"),
        ]
        for feature in _DISABLED_FEATURES:
            command.extend(["--disable", feature])
        for setting in (
            'web_search="disabled"',
            'approval_policy="never"',
            "mcp_servers={}",
            "features.skip_host_skill_discovery=true",
            "suppress_unstable_features_warning=true",
            f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
        ):
            command.extend(["--config", setting])
        command.extend(["--image", str(image_path), "-"])
        return command

    def generate(self, prompt: str, image_path: Path, output_dir: Path) -> dict[str, Any]:
        """Persist the complete invocation and return its audited outcome.

        A quota failure is distinct from a bad proposal, so the runner can stop
        without consuming the remaining repair attempts. No retry or model
        fallback is performed here.
        """
        started = time.monotonic()
        image_path = Path(image_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (output_dir / "response_schema.json").write_text(
            json.dumps(_SCHEMA, indent=2) + "\n", encoding="utf-8"
        )
        audit = _Audit()
        status, error = "error", None
        process: subprocess.Popen[bytes] | None = None
        command = self._command(image_path, output_dir)
        metadata: dict[str, Any] = {
            "command": command,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "timeout_seconds": self.timeout,
            "audit_version": 1,
        }
        final_path = output_dir / "final_response.txt"
        # Never mistake a previous invocation's final file for a new response.
        final_path.write_text("", encoding="utf-8")
        stdout_tail = b""
        stderr_bytes = bytearray()
        with tempfile.TemporaryDirectory(prefix="mesh2cad-codex-") as neutral_cwd:
            metadata["cwd"] = neutral_cwd
            (output_dir / "command.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            with (
                (output_dir / "prompt.txt").open("rb") as prompt_file,
                (output_dir / "events.jsonl").open("wb") as trace_file,
                (output_dir / "stderr.txt").open("wb") as stderr_file,
                selectors.DefaultSelector() as selector,
            ):
                try:
                    if not image_path.is_file():
                        raise FileNotFoundError(f"Input image does not exist: {image_path}")
                    process = subprocess.Popen(
                        command,
                        cwd=neutral_cwd,
                        stdin=prompt_file,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    )
                    assert process.stdout is not None and process.stderr is not None
                    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                    deadline = started + self.timeout
                    aborted = False
                    drain_deadline: float | None = None
                    while selector.get_map() or process.poll() is None:
                        now = time.monotonic()
                        if not aborted and now >= deadline:
                            status, error = (
                                "error",
                                f"Codex timed out after {self.timeout:g} seconds",
                            )
                            _terminate(process)
                            aborted, drain_deadline = True, time.monotonic() + 1
                        if drain_deadline is not None and now >= drain_deadline:
                            break
                        for key, _ in selector.select(timeout=0.05):
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            if key.data == "stderr":
                                stderr_file.write(chunk)
                                stderr_file.flush()
                                stderr_bytes.extend(chunk)
                                if not aborted and _is_quota_error(
                                    stderr_bytes.decode("utf-8", errors="replace")
                                ):
                                    status = "quota_exhausted"
                                    error = stderr_bytes.decode("utf-8", errors="replace").strip()
                                    _terminate(process)
                                    aborted, drain_deadline = True, time.monotonic() + 1
                                continue
                            trace_file.write(chunk)
                            trace_file.flush()
                            if aborted:
                                continue
                            stdout_tail += chunk
                            while b"\n" in stdout_tail:
                                line, stdout_tail = stdout_tail.split(b"\n", 1)
                                try:
                                    audit.feed(line)
                                except ValueError as exc:
                                    status, error = "protocol_violation", str(exc)
                                    _terminate(process)
                                    aborted, drain_deadline = True, time.monotonic() + 1
                                    break
                                if audit.quota:
                                    status, error = "quota_exhausted", audit.error
                                    _terminate(process)
                                    aborted, drain_deadline = True, time.monotonic() + 1
                                    break
                    if not aborted and stdout_tail:
                        try:
                            audit.feed(stdout_tail)
                        except ValueError as exc:
                            status, error = "protocol_violation", str(exc)
                            aborted = True
                    if not aborted:
                        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                        if audit.quota or _is_quota_error(stderr_text):
                            status, error = "quota_exhausted", audit.error or stderr_text.strip()
                        elif process.returncode != 0:
                            status, error = (
                                "error",
                                audit.error
                                or stderr_text.strip()
                                or (f"Codex exited with status {process.returncode}"),
                            )
                        elif audit.terminal != "turn.completed":
                            status, error = (
                                "error",
                                audit.error or "Codex stream ended without turn.completed",
                            )
                        elif audit.final_text is None:
                            status, error = (
                                "protocol_violation",
                                "Completed turn has no agent response",
                            )
                        else:
                            status = "ok"
                except (OSError, subprocess.SubprocessError) as exc:
                    status, error = "error", str(exc)
                finally:
                    if process is not None:
                        if process.poll() is None:
                            _terminate(process)
                        if process.stdout is not None:
                            process.stdout.close()
                        if process.stderr is not None:
                            process.stderr.close()
        code: str | None = None
        if audit.final_text is not None:
            try:
                file_text = final_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                file_text = ""
                status, error = "protocol_violation", "Final response file is not valid UTF-8"
            if (
                status == "ok"
                and file_text.strip()
                and file_text.strip() != audit.final_text.strip()
            ):
                status, error = (
                    "protocol_violation",
                    "Final response file differs from audited stream",
                )
            final_path.write_text(audit.final_text, encoding="utf-8")
        if status == "ok":
            try:
                payload = json.loads(audit.final_text or "")
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"code"}
                    or not isinstance(payload["code"], str)
                    or not payload["code"].strip()
                ):
                    raise ValueError("Expected exactly one nonempty string field: code")
                code = payload["code"]
            except (ValueError, json.JSONDecodeError) as exc:
                status, error = "protocol_violation", f"Invalid structured response: {exc}"
        result = {
            "status": status,
            "code": code,
            "usage": audit.usage,
            "diagnostics": audit.diagnostics,
            "error": error,
            "elapsed_seconds": time.monotonic() - started,
            "launches": 1,
        }
        metadata.update(
            exit_code=process.returncode if process is not None else None,
            terminal_event=audit.terminal,
            status=status,
            elapsed_seconds=result["elapsed_seconds"],
        )
        (output_dir / "command.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        # The harness can recover this paid response after an interruption. A
        # partial write must never look like a committed generation checkpoint.
        write_json(output_dir / "generation.json", result)
        return result
