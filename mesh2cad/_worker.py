"""CadQuery process entry point; invoked only inside the executor's sandbox."""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path


def main() -> int:
    directory = Path.cwd()
    result_file = directory / "worker_result.json"
    try:
        import cadquery as cq

        source_path = directory / "code.py"
        source = source_path.read_text(encoding="utf-8")
        namespace = {"cq": cq, "math": math, "__name__": "__mesh2cad_generated__"}
        exec(compile(source, str(source_path), "exec"), namespace)
        result = namespace.get("r")
        if not isinstance(result, cq.Workplane):
            raise ValueError("Generated code must assign a cadquery.Workplane to r")
        shapes = result.vals()
        if not shapes or any(not isinstance(shape, cq.Shape) for shape in shapes):
            raise ValueError("r must contain at least one CadQuery shape")
        shape = shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)
        if shape.isNull() or not shape.isValid() or not shape.Solids():
            raise ValueError("r must contain a nonempty, valid solid")
        if not math.isfinite(shape.Volume()) or shape.Volume() <= 0:
            raise ValueError("r must have positive finite volume")
        cq.exporters.export(shape, str(directory / "candidate.tmp.step"))
        cq.exporters.export(shape, str(directory / "candidate.tmp.stl"), tolerance=0.001)
        (directory / "candidate.tmp.step").replace(directory / "candidate.step")
        (directory / "candidate.tmp.stl").replace(directory / "candidate.stl")
        payload = {"status": "valid", "error": None}
        return_code = 0
    except BaseException:
        error = traceback.format_exc()
        (directory / "traceback.txt").write_text(error, encoding="utf-8")
        payload = {"status": "execution_error", "error": error}
        return_code = 1
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    return return_code


if __name__ == "__main__":
    sys.exit(main())
