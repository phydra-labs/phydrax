from __future__ import annotations

from itertools import product

import numpy as np

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import (
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.geometry._global_embedding import certify_global_embedding


def _skew_torus(*, folded: bool = False) -> tuple[CellMesh, CellGeometrySpec]:
    lattice = np.asarray(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
    indices = tuple(product(range(3), repeat=2))
    index = {point: row for row, point in enumerate(indices)}
    coordinates = []
    representatives = []
    shifts = []
    for point in indices:
        representative = tuple(value % 2 for value in point)
        image = (
            0.5 * representative[0] * lattice[0] + 0.5 * representative[1] * lattice[1]
        )
        if folded and representative == (1, 1):
            image = image + np.asarray((0.5, -1.0), dtype=np.float64)
        shift = tuple(value // 2 for value in point)
        coordinates.append(image + shift[0] * lattice[0] + shift[1] * lattice[1])
        representatives.append(index[representative])
        shifts.append(shift)
    triangles = []
    for i, j in product(range(2), repeat=2):
        lower_left = index[(i, j)]
        lower_right = index[(i + 1, j)]
        upper_right = index[(i + 1, j + 1)]
        upper_left = index[(i, j + 1)]
        triangles.extend(
            (
                (lower_left, lower_right, upper_right),
                (lower_left, upper_right, upper_left),
            )
        )
    block = CellBlock("triangles", "triangle", np.asarray(triangles, dtype=np.int32))
    lifted = CellMesh(np.asarray(coordinates), (block,))
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicCell(lattice),
        np.asarray(representatives, dtype=np.int32),
        np.asarray(shifts, dtype=np.int32),
    )
    mesh = CellMesh(np.asarray(coordinates), (block,), periodic_topology=topology)
    return mesh, CellGeometrySpec.affine(mesh)


def test_skew_periodic_triangle_quotient_has_exact_degree_one() -> None:
    mesh, geometry = _skew_torus()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    independent_area = sum(
        abs(
            float(
                np.linalg.det(
                    np.stack(
                        (
                            coordinates[int(cell[1])] - coordinates[int(cell[0])],
                            coordinates[int(cell[2])] - coordinates[int(cell[0])],
                        )
                    )
                )
            )
        )
        / 2.0
        for cell in np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    )
    periodic_topology = mesh.periodic_topology
    assert periodic_topology is not None
    period = periodic_topology.cell
    assert isinstance(period, PeriodicCell)
    lattice_area = abs(float(np.linalg.det(np.asarray(period.vectors))))

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert certificate.periodic_affine_degree is not None
    assert certificate.periodic_affine_degree.covering_degree == 1
    assert certificate.periodic_affine_degree.pullback_area == (1, 1)
    assert certificate.periodic_affine_degree.lattice_area == (1, 1)
    assert independent_area == lattice_area
    assert {
        "periodic_affine_torus_manifold",
        "periodic_affine_torus_trace",
        "periodic_affine_torus_orientation",
        "periodic_affine_torus_degree_one",
    } <= set(certificate.evaluated_checks)


def test_folded_skew_periodic_map_is_invalid_and_not_admitted() -> None:
    mesh, geometry = _skew_torus(folded=True)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert not validity.all_certified
    assert np.any(np.asarray(validity.status) == CellValidityStatus.INVALID)
    assert certificate.status == "violated"
    assert certificate.periodic_affine_degree is None
    assert any(
        finding.check == "cell_validity" and finding.status == "violated"
        for finding in certificate.findings
    )
