import numpy as np
import pytest

from phydrax._meshcore import angle_arctangent2
from phydrax.discretization import CellBlock, CellMesh
from phydrax.meshing import evaluate_cell_quality


@pytest.mark.meshcore
def test_triangle_minimum_angle_matches_native_bits() -> None:
    # At the origin, the incident edges are (1, 0) and (x, y).
    x = float.fromhex("0x1.76878f544fe21p-1")
    y = float.fromhex("0x1.8cd9fb2ad7b1ap-2")
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [x, y]], dtype=np.float64)
    cells = np.asarray([[0, 1, 2]], dtype=np.int32)
    mesh = CellMesh(vertices, (CellBlock("cells", "triangle", cells),))

    native = angle_arctangent2(
        np.asarray([y], dtype=np.float64), np.asarray([x], dtype=np.float64)
    )
    expected = np.asarray([float.fromhex("0x1.f2e5313a53903p-2")], dtype=np.float64)
    actual = np.asarray(evaluate_cell_quality(mesh).minimum_angle, dtype=np.float64)
    assert np.array_equal(native.view(np.uint64), expected.view(np.uint64))
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
