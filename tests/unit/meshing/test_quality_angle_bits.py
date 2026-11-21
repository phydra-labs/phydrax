import numpy as np
import pytest

from phydrax._meshcore import angle_arctangent2, meshcore_available
from phydrax.discretization import CellBlock, CellMesh
from phydrax.meshing import evaluate_cell_quality


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
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


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
@pytest.mark.parametrize(
    ("kind", "corners"),
    (("tetrahedron", 4), ("hexahedron", 8), ("prism", 6), ("pyramid", 5)),
)
def test_native_solid_quality_kernels_match_host_bits(
    monkeypatch: pytest.MonkeyPatch, kind: str, corners: int
) -> None:
    from phydrax.meshing import _quality

    rng = np.random.default_rng(7)
    points = rng.standard_normal((257, corners, 3))
    points[:3] = 0.0
    points[3:6, 1] = points[3:6, 0]
    points[6, 0, 0] = np.nan
    points[7] *= 1e-300
    points[8] *= 1e300
    points[9, 2] = np.inf
    table = _quality._corner_table(kind, corners)
    samples = _quality._dihedral_table(kind).samples
    with np.errstate(all="ignore"):
        native = (
            *_quality._host_solid_corner_measures(points, table),
            _quality._host_dihedral_angles(points, samples),
        )
        monkeypatch.setattr(_quality, "meshcore_available", lambda: False)
        host = (
            *_quality._host_solid_corner_measures(points, table),
            _quality._host_dihedral_angles(points, samples),
        )
    for native_values, host_values in zip(native, host, strict=True):
        assert np.array_equal(
            np.ascontiguousarray(native_values, dtype=np.float64).view(np.uint64),
            np.ascontiguousarray(host_values, dtype=np.float64).view(np.uint64),
        )
