import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from phydrax._meshcore import load_meshcore, meshcore_available


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
@pytest.mark.parametrize("cell_kind", ("triangle", "tetrahedron"))
def test_audit_evaluation_matches_without_meshcore(
    cell_kind: str, tmp_path: Path
) -> None:
    script = """
import dataclasses
import sys

import numpy as np
import phydrax as phx
from phydrax._meshcore import meshcore_available

assert meshcore_available() == (sys.argv[3] == "present")
kind = sys.argv[1]
vertices = np.asarray(
    phx.discretization.reference_cell_topology(kind).vertices, dtype=np.float64
)
mesh = phx.discretization.CellMesh(
    vertices,
    (phx.discretization.CellBlock(
        "cells", kind, np.arange(len(vertices), dtype=np.int32)[None, :]
    ),),
)
geometry = phx.discretization.CellGeometrySpec.affine(mesh)
evaluation = phx.meshing.audit_cell_mesh(mesh, geometry).quality.evaluation
arrays = {
    field.name: np.asarray(value)
    for field in dataclasses.fields(evaluation)
    if hasattr(value := getattr(evaluation, field.name), "dtype")
}
assert arrays
np.savez(sys.argv[2], **arrays)
"""
    outputs = []
    for state, library in (
        ("absent", tmp_path / "missing-meshcore"),
        ("present", load_meshcore().path),
    ):
        output = tmp_path / f"{state}.npz"
        environment = os.environ.copy()
        environment["PHYDRAX_MESHCORE_LIBRARY"] = str(library)
        subprocess.run(
            [sys.executable, "-c", script, cell_kind, str(output), state],
            env=environment,
            check=True,
        )
        outputs.append(output)

    with np.load(outputs[0]) as absent, np.load(outputs[1]) as present:
        assert set(absent.files) == set(present.files)
        for name in absent.files:
            left, right = absent[name], present[name]
            assert left.shape == right.shape, name
            assert left.dtype == right.dtype, name
            assert left.tobytes() == right.tobytes(), name
