"""Cancellable evaluation subprocess; never executes generated Python."""

import json
import sys
from pathlib import Path

from .evaluation import evaluate_pair

if __name__ == "__main__":
    result = evaluate_pair(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        n_points=int(sys.argv[4]),
        seed=int(sys.argv[5]),
        threshold=float(sys.argv[6]),
    )
    Path(sys.argv[3]).write_text(json.dumps(result, allow_nan=False))
