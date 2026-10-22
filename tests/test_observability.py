"""Console/file logging is bounded, idempotent and redacts credentials."""

import logging

import pytest

from mesh2cad.observability import (
    LOGGER,
    SafeFormatter,
    configure_logging,
    log_event,
    register_secret,
    tail_log,
    timed_stage,
)


def test_console_and_rotating_file_receive_redacted_events(tmp_path, capfd) -> None:
    secret = "test-secret-for-logging-123"
    register_secret(secret)
    path = configure_logging(tmp_path)
    configure_logging(tmp_path)
    assert len(LOGGER.handlers) == 2
    log_event("component.test", error=f"Bearer {secret}", api_key=secret)
    with pytest.raises(ValueError):
        with timed_stage("test.operation"):
            raise ValueError(secret)
    output = path.read_text()
    console = capfd.readouterr().err
    assert "component.test" in output and "component.test" in console
    assert "test.operation.failed" in output
    assert "[REDACTED]" in output
    assert secret not in output and secret not in console
    assert output.count("component.test") == 1


def test_formatter_redacts_provider_keys_without_registration() -> None:
    key = "sk-or-v1-" + "a" * 32
    record = logging.LogRecord("test", logging.ERROR, "test.py", 1, "bad key %s", (key,), None)
    assert key not in SafeFormatter().format(record)


def test_log_tail_reads_only_a_bounded_suffix(tmp_path) -> None:
    path = tmp_path / "large.log"
    path.write_text("ignored line\n" * 100_000 + "last line\n")
    result = tail_log(path, max_bytes=4096, max_lines=4)
    assert len(result.splitlines()) == 4
    assert result.endswith("last line")
    assert tail_log(tmp_path / "absent.log") == ""
