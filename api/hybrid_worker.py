"""CLI: python -m api.hybrid_worker NORMALIZED_TARGET_MESH OUTPUT_JSON."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
from pathlib import Path

from .hybrid import MAX_FILE_BYTES, MAX_OUTPUT_BYTES, build_evidence


def main(source: Path, destination: Path) -> None:
    from mesh2cad.geometry import load_mesh

    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Hybrid input exceeds file limit")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = build_evidence(load_mesh(source), digest)
    encoded = json.dumps(evidence, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode()) > MAX_OUTPUT_BYTES:
        raise ValueError("Hybrid evidence exceeds output limit")
    destination.write_text(encoded)


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    main(Path(sys.argv[1]), Path(sys.argv[2]))
