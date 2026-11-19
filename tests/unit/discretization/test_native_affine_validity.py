"""The route-selected native validity proof of affine tetrahedra.

The reference is the rational per-cell proof of the same certificate type and
exact rational determinants of the stored binary64 coordinates.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

from phydrax._meshcore import MeshcoreUnavailableError
from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)


def _mesh() -> CellMesh:
    # A Delaunay ball with a flipped cell, a sliver under the determinant floor
    # and a cell with a coordinate below the native exact domain (2**-60).
    generator = np.random.default_rng(7)
    points = generator.uniform(-1.0, 1.0, size=(60, 3))
    cells = np.asarray(Delaunay(points).simplices, dtype=np.int64)
    a, b, c, d = (points[cells[:, k]] for k in range(4))
    negative = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0.0
    cells[negative] = cells[negative][:, (1, 0, 2, 3)]
    extra = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [3.5, 2.0**-40, 0.5],
            [4.0, 2.0**-70, 0.2],
        ]
    )
    base = points.shape[0]
    added = np.asarray(
        [
            [base + 1, base, base + 2, base + 3],  # flipped
            [base + 1, base, base + 3, base + 4],  # sliver
            [base, base + 2, base + 3, base + 5],  # outside the native domain
        ]
    )
    return CellMesh.from_simplices(
        np.concatenate((points, extra)), np.concatenate((cells, added)), dimension=3
    )


def _exact_determinant(points: np.ndarray, cell: np.ndarray) -> Fraction:
    origin = [Fraction(float(value)) for value in points[cell[0]]]
    e = [
        [Fraction(float(points[cell[k], axis])) - origin[axis] for axis in range(3)]
        for k in range(1, 4)
    ]
    return (
        e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
        - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
        + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])
    )


def test_native_proof_matches_the_rational_proof_and_encloses_exact_determinants() -> (
    None
):
    mesh = _mesh()
    geometry = CellGeometrySpec.affine(mesh)
    policy = CellValidityPolicy(relative_determinant_floor=2.0**-10)
    rational = certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy)
    native = certify_cell_geometry_validity(
        geometry, mesh=mesh, policy=policy, native_affine_tetrahedra=True
    )
    status = np.asarray(native.status)
    assert native.native_affine_tetrahedra and not rational.native_affine_tetrahedra
    np.testing.assert_array_equal(status, np.asarray(rational.status))
    assert status[-3] == CellValidityStatus.INVALID
    assert status[-2] == CellValidityStatus.INVALID
    assert status[-1] == CellValidityStatus.CERTIFIED_VALID
    assert native.certificate_id != rational.certificate_id
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    lower = np.asarray(native.determinant_lower)
    upper = np.asarray(native.determinant_upper)
    for row in range(cells.shape[0]):
        determinant = _exact_determinant(points, cells[row])
        assert Fraction(float(lower[row])) <= determinant <= Fraction(float(upper[row]))
    # Only the rational fallback row records a reusable exact map.
    digests = np.frombuffer(native.cell_map_digests, dtype=np.uint8).reshape(-1, 2, 16)
    assert np.flatnonzero(np.any(digests != 0, axis=(1, 2))).tolist() == [
        cells.shape[0] - 1
    ]


def test_native_proof_is_reproduced_by_revalidation_and_stored_field_restore() -> None:
    mesh = _mesh()
    geometry = CellGeometrySpec.affine(mesh)
    policy = CellValidityPolicy()
    native = certify_cell_geometry_validity(
        geometry, mesh=mesh, policy=policy, native_affine_tetrahedra=True
    )
    native.revalidate(geometry, mesh=mesh)
    restored = CellValidityCertificate.from_stored_fields(
        native, mesh, geometry, policy=policy
    )
    assert restored.certificate_id == native.certificate_id
    assert restored.native_affine_tetrahedra


def test_native_proof_requires_the_affine_map_of_a_tetrahedral_mesh() -> None:
    mesh = CellMesh.from_simplices(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[0, 1, 2]]),
        dimension=2,
    )
    with pytest.raises(ValueError, match="affine vertex map"):
        certify_cell_geometry_validity(mesh, native_affine_tetrahedra=True)
    with pytest.raises(TypeError, match="must be a bool"):
        certify_cell_geometry_validity(
            mesh,
            native_affine_tetrahedra=1,  # ty: ignore[invalid-argument-type]
        )


def test_the_rational_proof_never_loads_the_native_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mesh = _mesh()
    monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", str(tmp_path / "missing-meshcore"))
    certificate = certify_cell_geometry_validity(mesh)
    assert certificate.invalid_count == 2
    assert not certificate.native_affine_tetrahedra
    with pytest.raises(MeshcoreUnavailableError):
        certify_cell_geometry_validity(mesh, native_affine_tetrahedra=True)
