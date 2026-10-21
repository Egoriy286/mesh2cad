"""Launch the local API and Streamlit workbench with one command."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--data-dir", type=Path, default=Path(".cache/workbench"))
    args = parser.parse_args()
    if args.api_port == args.ui_port or not all(
        1 <= port <= 65535 and _available(port) for port in (args.api_port, args.ui_port)
    ):
        parser.error("Choose two different available local ports (--api-port / --ui-port)")
    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    api_url = f"http://127.0.0.1:{args.api_port}"
    environment.update(
        MESH2CAD_API_URL=api_url,
        MESH2CAD_DATA_DIR=str(args.data_dir.resolve()),
        PYTHONUNBUFFERED="1",
    )
    children = []
    try:
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.app:create_app", "--factory",
             "--host", "127.0.0.1", "--port", str(args.api_port), "--no-access-log"],
            cwd=root, env=environment,
        )
        children.append(api)
        deadline = time.monotonic() + 40
        while True:
            if api.poll() is not None:
                raise RuntimeError("API exited during startup; see its diagnostic above")
            try:
                with urllib.request.urlopen(api_url + "/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("API did not become ready within 40 seconds")
            time.sleep(.2)
        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(root / "frontend" / "app.py"),
             "--server.address", "127.0.0.1", "--server.port", str(args.ui_port),
             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
            cwd=root, env=environment,
        )
        children.append(ui)
        print(f"\nMesh2CAD Workbench: http://127.0.0.1:{args.ui_port}", flush=True)
        print(f"Local API: {api_url}\nCtrl-C stops both services.\n", flush=True)
        while api.poll() is None and ui.poll() is None:
            time.sleep(.3)
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Workbench could not start: {exc}", file=sys.stderr)
        return 1
    finally:
        for process in reversed(children):
            _stop(process)


if __name__ == "__main__":
    raise SystemExit(main())
