"""Private subprocess entry point. Credentials arrive on stdin, never in argv/files."""

from __future__ import annotations

import sys
from pathlib import Path

from .harness import Orchestrator, RunLock
from .resources import watch_parent


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    watch_parent(int(sys.argv[2]))
    key = sys.stdin.readline().strip()
    sys.stdin.close()
    # The monitor owns .worker.lock for the full lifetime of this process.
    # This independent lease also guards against a second private worker.
    lease = RunLock(root, ".process.lock")
    state = Orchestrator(root, key).run(_lease=lease)
    raise SystemExit(0 if state["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
