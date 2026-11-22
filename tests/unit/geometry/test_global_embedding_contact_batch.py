from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
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


def _shared_edge_certificate(
    selected: tuple[int, ...],
) -> tuple[str, tuple[str, ...]]:
    vertices = np.asarray(
        (
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 0),
            (0, 1, 0),
            (-1, 1, 0),
            (-1, -1, 0),
            (1, -1, 0),
            (2, -1, 0),
        ),
        dtype=np.float64,
    )
    cells = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 1, 6, 7))
    element = lagrange_element("tetrahedron", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)

    def image(points: np.ndarray) -> np.ndarray:
        x, y, z = points.T
        return np.column_stack((x + y * z / 8, y, z))

    identifiers = sorted({vertex for index in selected for vertex in cells[index]})
    local = {identifier: row for row, identifier in enumerate(identifiers)}
    connectivity = np.asarray(
        [[local[vertex] for vertex in cells[index]] for index in selected], dtype=np.int32
    )
    mesh = CellMesh(
        image(vertices[identifiers]),
        (
            CellBlock(
                "curved",
                "tetrahedron",
                connectivity,
                global_ids=np.asarray([10 + index for index in selected], dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(identifiers, dtype=np.int64),
    )
    coordinates = []
    routes = []
    for index in selected:
        corners = vertices[np.asarray(cells[index], dtype=np.int64)]
        reference = corners[0] + nodes @ (corners[1:] - corners[0])
        start = len(coordinates)
        coordinates.extend(image(reference))
        routes.append(tuple(range(start, len(coordinates))))
    geometry = CellGeometrySpec(
        {"curved": element},
        {"curved": np.asarray(routes, dtype=np.int32)},
        np.asarray(coordinates, dtype=np.float64),
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = embedding.certify_global_embedding(mesh, geometry, validity)
    assert validity.all_certified and certificate.status == "certified"
    evidence = tuple(
        value
        for witness in certificate.witnesses
        for value in witness.exact_values
        if value.startswith("cell-pair:")
    )
    return certificate.certificate_id, evidence


def test_shared_edge_certificate_matches_independent_pair_proofs() -> None:
    _, values = _shared_edge_certificate((0, 1, 2))
    _, first_values = _shared_edge_certificate((0, 1))
    _, second_values = _shared_edge_certificate((0, 2))
    assert first_values == tuple(value for value in values if "cell-pair:10:11:" in value)
    assert second_values == tuple(
        value for value in values if "cell-pair:10:12:" in value
    )


# The native vertex-cone screen proves these straight shared edges by their
# cross-residual cones; the rational route proves them by subdivision, so the
# certificate identities below are those of the native route.
@pytest.mark.skipif(
    not meshcore_available(), reason="the native meshcore library is absent"
)
def test_shared_edge_certificate_identities_are_stable() -> None:
    assert tuple(
        _shared_edge_certificate(selected)[0] for selected in ((0, 1, 2), (0, 1), (0, 2))
    ) == (
        "e7bdde70079a6c4d0cdad06908c25250f694a80063cdb4f99c31ba3e29f0f170",
        "9422c999aa1c227826527065f2d115dc31a008121e561bd0399a861b458d2d9f",
        "e39cd5b18a98ed30838891a6cd501026f74df6a0b69199465046a7d04716bc3f",
    )
