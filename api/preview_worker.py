"""Bounded VTK preview generation; inputs and outputs contain arrays only."""

from __future__ import annotations

import resource
import sys
from pathlib import Path

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_FACES = 500_000
MAX_VERTICES = 1_500_000
MAX_PREVIEW_FACES = 30_000


def main(source: Path, destination: Path, limit: int) -> None:
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

    if source.stat().st_size > MAX_FILE_BYTES or not 1 <= limit <= MAX_PREVIEW_FACES:
        raise ValueError("Preview input exceeds file/face limits")
    with np.load(source, allow_pickle=False) as arrays:
        vertices, faces = arrays["vertices"], arrays["faces"]
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(vertices) > MAX_VERTICES
        or not 0 < len(faces) <= MAX_FACES
        or not np.isfinite(vertices).all()
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Preview requires finite bounded triangle geometry")
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    polygons = vtk.vtkCellArray()
    polygons.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces, dtype=np.int64).ravel(), deep=True),
    )
    geometry = vtk.vtkPolyData()
    geometry.SetPoints(points)
    geometry.SetPolys(polygons)
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(geometry)
    # A tiny margin accounts for the two-face granularity of closed surfaces.
    decimator.SetTargetReduction(1 - max(1, limit - 2) / len(faces))
    decimator.Update()
    result = decimator.GetOutput()
    if result.GetPoints() is None or result.GetNumberOfPolys() == 0:
        raise ValueError("Decimation produced no preview surface")
    coordinates = vtk_to_numpy(result.GetPoints().GetData()).copy()
    offsets = vtk_to_numpy(result.GetPolys().GetOffsetsArray())
    if not np.all(np.diff(offsets) == 3):
        raise ValueError("Decimation returned non-triangle polygons")
    triangles = vtk_to_numpy(result.GetPolys().GetConnectivityArray()).reshape(-1, 3).copy()
    if (
        len(triangles) > limit
        or len(coordinates) > MAX_VERTICES
        or not np.isfinite(coordinates).all()
        or triangles.min() < 0
        or triangles.max() >= len(coordinates)
    ):
        raise ValueError("Decimation returned invalid preview geometry")
    np.savez(destination, vertices=coordinates, faces=triangles)


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
