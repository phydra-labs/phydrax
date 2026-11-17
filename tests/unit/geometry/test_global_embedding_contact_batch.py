from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.geometry import _global_embedding as embedding


def _curved_contacts(*, overlap: bool) -> tuple[CellMesh, CellGeometrySpec]:
    element = lagrange_element("tetrahedron", 2)
    vertices = np.asarray(
        reference_cell_topology("tetrahedron").vertices, dtype=np.float64
    )
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    points: list[tuple[float, float, float]] = []
    connectivity: list[tuple[int, ...]] = []
    dofs: list[tuple[float, float, float]] = []
    routes: list[tuple[int, ...]] = []

    def image(point: np.ndarray, group: int, second: bool) -> tuple[float, float, float]:
        x, y, z = point
        if second:
            if group == 0:
                x, y, z = -x, -z, -y  # shared vertex
            elif group == 1:
                x, y = -x, -y  # shared edge
            elif group == 2:
                x, y, z = y, x, -z  # shared face
            elif not overlap:
                x, y, z = -x, -z, -y
        return float(x + x * y / 8 + 4 * group), float(y), float(z)

    for group in range(4):
        vertex_rows: dict[tuple[float, float, float], int] = {}
        for second in (False, True):
            row = []
            for point in vertices:
                physical = image(point, group, second)
                # In the overlap case only the origin has a declared identity;
                # other coincident corners are deliberately distinct vertices.
                if group == 3 and second and overlap and physical != (12.0, 0.0, 0.0):
                    index = len(points)
                    points.append(physical)
                else:
                    index = vertex_rows.get(physical, -1)
                    if index == -1:
                        index = len(points)
                        vertex_rows[physical] = index
                        points.append(physical)
                row.append(index)
            connectivity.append(tuple(row))
            routes.append(tuple(range(len(dofs), len(dofs) + len(nodes))))
            dofs.extend(image(point, group, second) for point in nodes)
    mesh = CellMesh(
        np.asarray(points, dtype=np.float64),
        (CellBlock("curved", "tetrahedron", np.asarray(connectivity, dtype=np.int32)),),
    )
    geometry = CellGeometrySpec(
        {"curved": element},
        {"curved": np.asarray(routes, dtype=np.int32)},
        np.asarray(dofs, dtype=np.float64),
    )
    return mesh, geometry


@pytest.mark.parametrize("overlap", (False, True))
def test_curved_contact_certificates_with_exact_contact_oracles(
    overlap: bool,
) -> None:
    mesh, geometry = _curved_contacts(overlap=overlap)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified

    # The common map F(x,y,z)=(x(1+y/8),y,z) is injective on
    # these tetrahedra because 1+y/8 >= 7/8.  Their reference images
    # meet respectively at a vertex, an edge, and a face.
    def image(
        x: Fraction, y: Fraction, z: Fraction, group: int
    ) -> tuple[Fraction, Fraction, Fraction]:
        return x * (1 + y / 8) + 4 * group, y, z

    zero, quarter = Fraction(0), Fraction(1, 4)
    assert image(zero, zero, zero, 0) == (0, 0, 0)
    assert image(zero, zero, quarter, 1) == (4, 0, quarter)
    assert image(quarter, quarter, zero, 2) == (
        Fraction(8) + Fraction(33, 128),
        quarter,
        zero,
    )
    nodes = lagrange_element("tetrahedron", 2).reference_nodes
    routes = np.asarray(geometry.geometry_dofs[0])
    coordinates = np.asarray(geometry.coordinates)
    for group in range(4):
        for node_index, node in enumerate(nodes):
            x, y, z = (Fraction(float(value)) for value in node)
            first = image(x, y, z, group)
            if group == 0 or (group == 3 and not overlap):
                second = image(-x, -z, -y, group)
            elif group == 1:
                second = image(-x, -y, z, group)
            elif group == 2:
                second = image(y, x, -z, group)
            else:
                second = first
            first_dof = coordinates[routes[2 * group, node_index]]
            second_dof = coordinates[routes[2 * group + 1, node_index]]
            assert tuple(Fraction(float(v)) for v in first_dof) == first
            assert tuple(Fraction(float(v)) for v in second_dof) == second
    # In the overlap case both fourth-group cells map the same strictly
    # interior barycentric point; otherwise their reference x signs differ.
    interior = image(quarter, quarter, quarter, 3)
    assert interior == (Fraction(12) + Fraction(33, 128), quarter, quarter)
    certificate = embedding.certify_global_embedding(mesh, geometry, validity)
    if overlap:
        assert certificate.status != "certified"
        assert any(
            finding.check == "mapped_cell_contact" for finding in certificate.findings
        )
    else:
        assert certificate.status == "certified", certificate.findings
