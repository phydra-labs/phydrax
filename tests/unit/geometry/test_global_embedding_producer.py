from __future__ import annotations

import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np

from phydrax._bvh import (
    bvh_overlap_pair_blocks,
    bvh_overlap_pairs_host,
    prepare_bvh,
)
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization._cell_geometry import (
    _coordinate_coefficient_identity,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import (
    add,
    axes,
    constant,
    evaluate_expression,
    Expression,
    multiply,
    polynomial_bounds,
    scale,
    subtract,
)
from phydrax.discretization._exact_algebraic_geometry import (
    ExactScalar,
    RadicalElement,
)
from phydrax.discretization._nested_reference_types import (
    ExactReferenceChartExpression,
    ReferenceCellMapWitnesses,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem._reference import (
    lagrange_element,
    lagrange_reference_node_data,
)
from phydrax.geometry._boundary_degree_embedding import (
    _face_trace_image_evidence,
    build_oriented_boundary,
)
from phydrax.geometry._embedding_cells import EmbeddingCell, prepare_embedding_cells
from phydrax.geometry._exact_polyhedral_geometry import Point3
from phydrax.geometry._global_embedding import (
    _nonperiodic_contacts,
    _shared_curved_facet_separation,
    _shared_edge_cone_separation,
    _shared_edge_separation,
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from phydrax.geometry._mapped_embedding import prove_mapped_cell_injective
from phydrax.geometry._restricted_embedding import (
    reference_chart_composition_equal,
    restriction_trace_equal,
)
from phydrax.meshing._curving import _exact_coordinate_bank_source_map


def _mapped_hexahedron(
    global_id: int,
    vertex_ids: tuple[int, ...],
    coordinates: tuple[Expression, ...],
    /,
) -> EmbeddingCell:
    topology = reference_cell_topology("hexahedron")
    references = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    corners = tuple(
        tuple(evaluate_expression(expression, point) for expression in coordinates)
        for point in references
    )
    bounds = tuple(polynomial_bounds(expression, "box", 3) for expression in coordinates)
    return EmbeddingCell(
        global_id,
        "mapped",
        0,
        "hexahedron",
        3,
        "box",
        tuple(range(8)),
        vertex_ids,
        references,
        coordinates,
        corners,
        None,
        None,
        tuple(lower for lower, _ in bounds),
        tuple(upper for _, upper in bounds),
        tuple(tuple(face) for face in topology.entities[2]),
    )


def _mapped_tetrahedron(
    global_id: int,
    vertex_ids: tuple[int, ...],
    coordinates: tuple[Expression, ...],
    /,
) -> EmbeddingCell:
    topology = reference_cell_topology("tetrahedron")
    references = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    corners = tuple(
        tuple(evaluate_expression(expression, point) for expression in coordinates)
        for point in references
    )
    bounds = tuple(
        polynomial_bounds(expression, "simplex", 3) for expression in coordinates
    )
    return EmbeddingCell(
        global_id,
        "mapped",
        0,
        "tetrahedron",
        3,
        "simplex",
        tuple(range(4)),
        vertex_ids,
        references,
        coordinates,
        corners,
        None,
        None,
        tuple(lower for lower, _ in bounds),
        tuple(upper for _, upper in bounds),
        tuple(tuple(face) for face in topology.entities[2]),
    )


def _radical_linear_tetra_geometry(
    exact_coordinates: tuple[tuple[ExactScalar, ...], ...],
    cells: tuple[tuple[int, int, int, int], ...],
    cell_ids: tuple[int, ...],
    vertex_ids: tuple[int, ...],
    block_name: str,
    /,
) -> tuple[CellMesh, CellGeometrySpec, CellValidityCertificate]:
    execution = np.asarray(
        tuple(tuple(float(value) for value in point) for point in exact_coordinates),
        dtype=np.float64,
    )
    cell_connectivity = np.asarray(cells, dtype=np.int32)
    mesh = CellMesh(
        execution,
        (
            CellBlock(
                block_name,
                "tetrahedron",
                cell_connectivity,
                global_ids=np.asarray(cell_ids, dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(vertex_ids, dtype=np.int64),
    )
    element = lagrange_element("tetrahedron", 1)
    source_map = _exact_coordinate_bank_source_map(
        mesh,
        f"{block_name}-source",
        "radical-exact-r1",
        (block_name,),
        exact_coordinates,
        execution,
    )
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: cell_connectivity},
        execution,
        source_map=source_map,
        source_mesh=mesh,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    return mesh, geometry, validity


def _radical_curved_edge_geometry() -> tuple[
    CellMesh, CellGeometrySpec, CellValidityCertificate
]:
    root5 = RadicalElement.sqrt(Fraction(5))
    root3 = RadicalElement.sqrt(Fraction(3))
    epsilon = Fraction(1, 1024)
    topology = reference_cell_topology("tetrahedron")
    vertex_ids = (101, 103, 107, 109, 113, 127)
    cell_vertices = ((0, 1, 2, 3), (0, 1, 4, 5))
    exact_vertices: tuple[tuple[ExactScalar, ...], ...] = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (root5, Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), -root3),
        (Fraction(0), Fraction(1) - root5, Fraction(4) - root3),
        (Fraction(0), Fraction(-1), root3),
        (Fraction(0), Fraction(-1) + root5, Fraction(-4) + root3),
    )
    execution_vertices = np.asarray(
        tuple(tuple(float(value) for value in point) for point in exact_vertices),
        dtype=np.float64,
    )
    cell_connectivity = np.asarray(cell_vertices, dtype=np.int32)
    block_name = "radical-curved-edge"
    mesh = CellMesh(
        execution_vertices,
        (
            CellBlock(
                block_name,
                "tetrahedron",
                cell_connectivity,
                global_ids=np.asarray((137, 139), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(vertex_ids, dtype=np.int64),
    )
    element = lagrange_element("tetrahedron", 2)
    nodes = lagrange_reference_node_data(element)
    x, y, z = axes(3)
    edge_bubble = scale(multiply(y, z), epsilon)
    cell_coordinates: tuple[tuple[Expression, ...], ...] = (
        (
            add(scale(x, root5), edge_bubble),
            add(y, scale(z, Fraction(1) - root5)),
            add(scale(y, -root3), scale(z, Fraction(4) - root3)),
        ),
        (
            add(scale(x, root5), edge_bubble),
            add(scale(y, Fraction(-1)), scale(z, root5 - Fraction(1))),
            add(scale(y, root3), scale(z, -Fraction(4) + root3)),
        ),
    )
    row_by_entity: dict[tuple[int, ...], int] = {}
    coordinate_rows: list[tuple[ExactScalar, ...]] = []
    routes = np.empty((len(cell_vertices), element.local_dof_count), dtype=np.int32)
    for cell_row, local_vertices in enumerate(cell_vertices):
        dof_entity: dict[int, tuple[int, ...]] = {}
        for vertex, dofs in enumerate(element.entity_dofs[0]):
            entity = (vertex_ids[local_vertices[vertex]],)
            for dof in dofs:
                dof_entity[dof] = entity
        for edge, dofs in enumerate(element.entity_dofs[1]):
            entity = tuple(
                sorted(
                    vertex_ids[local_vertices[vertex]]
                    for vertex in topology.entities[1][edge]
                )
            )
            for dof in dofs:
                dof_entity[dof] = entity
        if len(dof_entity) != element.local_dof_count:
            raise AssertionError("The P2 tetrahedron has an unkeyed geometry DOF.")
        for node in nodes:
            if node.exact_point is None:
                raise AssertionError("The radical contact map requires exact P2 nodes.")
            values = tuple(
                evaluate_expression(expression, node.exact_point)
                for expression in cell_coordinates[cell_row]
            )
            entity = dof_entity[node.local_dof]
            geometry_row = row_by_entity.get(entity)
            if geometry_row is None:
                geometry_row = len(coordinate_rows)
                row_by_entity[entity] = geometry_row
                coordinate_rows.append(values)
            elif coordinate_rows[geometry_row] != values:
                raise AssertionError("The P2 source maps disagree on their shared edge.")
            routes[cell_row, node.local_dof] = geometry_row
    exact_coordinates = tuple(coordinate_rows)
    execution_coordinates = np.asarray(
        tuple(tuple(float(value) for value in point) for point in exact_coordinates),
        dtype=np.float64,
    )
    source_map = _exact_coordinate_bank_source_map(
        mesh,
        f"{block_name}-source",
        "radical-exact-r1",
        (block_name,),
        exact_coordinates,
        execution_coordinates,
    )
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: routes},
        execution_coordinates,
        source_map=source_map,
        source_mesh=mesh,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    return mesh, geometry, validity


def _radical_boundary_contact_identity() -> str:
    root5 = RadicalElement.sqrt(Fraction(5))
    degree_vertices: tuple[tuple[ExactScalar, ...], ...] = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (root5, Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    degree_mesh, degree_geometry, degree_validity = _radical_linear_tetra_geometry(
        degree_vertices,
        ((0, 1, 2, 3),),
        (131,),
        (101, 103, 107, 109),
        "radical-degree",
    )
    degree_certificate = certify_global_embedding(
        degree_mesh, degree_geometry, degree_validity
    )
    boundary = build_oriented_boundary(
        degree_mesh, prepare_embedding_cells(degree_mesh, degree_geometry)
    )
    if (
        not degree_validity.all_certified
        or degree_certificate.status != "certified"
        or boundary.status != "certified"
        or not boundary.boundary_degree_one
        or boundary.shell_signed_measures != (root5 / 6,)
    ):
        raise AssertionError(
            "The radical affine tetrahedron lost its degree-one shell proof."
        )
    degree_witnesses = tuple(
        (witness.theorem, witness.witness_id, witness.exact_values)
        for witness in degree_certificate.witnesses
    )
    if not any(
        theorem == "global_degree_one_injectivity" for theorem, _, _ in degree_witnesses
    ):
        raise AssertionError("The radical shell case omitted its global degree witness.")

    contact_mesh, contact_geometry, contact_validity = _radical_curved_edge_geometry()
    if not contact_validity.all_certified:
        raise AssertionError("The radical shared-edge cells lost local validity.")
    contact_cells = prepare_embedding_cells(contact_mesh, contact_geometry)
    edge_witness = _shared_edge_separation(contact_cells[0], contact_cells[1], (101, 103))
    if (
        edge_witness is None
        or edge_witness[1] != -edge_witness[2]
        or not edge_witness[3]
        or not edge_witness[4]
        or not any(isinstance(value, RadicalElement) for value in edge_witness[0])
    ):
        raise AssertionError(
            "The radical shared edge did not retain its exact normal proof."
        )
    contact_values: list[str] = []
    contact_findings = _nonperiodic_contacts(
        contact_mesh,
        contact_geometry,
        contact_cells,
        contact_values,
    )
    if contact_findings or not contact_values:
        normal_identity = (
            None
            if edge_witness is None
            else tuple(
                _coordinate_coefficient_identity(value) for value in edge_witness[0]
            )
        )
        trace_equal = restriction_trace_equal(
            contact_cells[0], contact_cells[1], (101, 103)
        )
        finding_identity = tuple(
            (finding.check, finding.status, finding.entity_ids, finding.detail)
            for finding in contact_findings
        )
        cell_identity = tuple(
            (
                cell.global_id,
                cell.vertex_ids,
                cell.image_lower,
                cell.image_upper,
                cell.affine,
            )
            for cell in contact_cells
        )
        contact_tree = prepare_bvh(
            np.asarray([cell.image_lower for cell in contact_cells], dtype=np.float64),
            np.asarray([cell.image_upper for cell in contact_cells], dtype=np.float64),
        )
        first_items, second_items = bvh_overlap_pairs_host(
            contact_tree, contact_tree, include_touching=True
        )
        candidate_pairs = tuple(
            (int(first), int(second))
            for first, second in zip(first_items, second_items, strict=True)
            if first != second
        )
        pair_blocks = tuple(
            tuple(
                (int(first), int(second))
                for first, second in zip(first_block, second_block, strict=True)
            )
            for first_block, second_block in bvh_overlap_pair_blocks(
                contact_tree, contact_tree, include_touching=True
            )
        )
        raise AssertionError(
            "Radical shared-edge contact evidence was incomplete: "
            f"trace={trace_equal!r}, findings={finding_identity!r}, "
            f"proof_values={tuple(contact_values)!r}, normal={normal_identity!r}, "
            f"cells={cell_identity!r}, bvh_pairs={candidate_pairs!r}, "
            f"pair_blocks={pair_blocks!r}."
        )
    return canonical_fingerprint(
        {
            "degree_certificate": degree_certificate.certificate_id,
            "degree_witnesses": degree_witnesses,
            "shell_measures": tuple(
                _coordinate_coefficient_identity(value)
                for value in boundary.shell_signed_measures
            ),
            "shared_edge_normal": tuple(
                _coordinate_coefficient_identity(value) for value in edge_witness[0]
            ),
            "shared_edge_sides": edge_witness[1:],
            "contact_proofs": tuple(contact_values),
        }
    )


def _radical_global_embedding_identity() -> tuple[
    str,
    str,
    tuple[tuple[str, str, tuple[str, ...]], ...],
    str,
]:
    topology = reference_cell_topology("tetrahedron")
    mesh = CellMesh(
        np.asarray(topology.vertices, dtype=np.float64),
        (
            CellBlock(
                "radical-tetrahedra",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((73,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((101, 103, 107, 109), dtype=np.int64),
    )
    element = lagrange_element("tetrahedron", 2)
    nodes = lagrange_reference_node_data(element)
    x, y, z = axes(3)
    radical = RadicalElement.sqrt(Fraction(5))
    coordinates = (
        add(x, scale(multiply(x, y), radical)),
        y,
        z,
    )
    exact_coordinate_rows: list[tuple[ExactScalar, ...]] = []
    for node in nodes:
        if node.exact_point is None:
            raise AssertionError("The regression element must retain exact nodal points.")
        exact_coordinate_rows.append(
            tuple(
                evaluate_expression(expression, node.exact_point)
                for expression in coordinates
            )
        )
    exact_coordinates = tuple(exact_coordinate_rows)
    execution_coordinates = np.asarray(
        tuple(tuple(float(value) for value in row) for row in exact_coordinates),
        dtype=np.float64,
    )
    source_map = _exact_coordinate_bank_source_map(
        mesh,
        "radical-global-witness-source",
        "radical-global-witness-r1",
        ("deterministic-radical-global-witness",),
        exact_coordinates,
        execution_coordinates,
    )
    geometry = CellGeometrySpec(
        {"radical-tetrahedra": element},
        {"radical-tetrahedra": np.arange(len(nodes), dtype=np.int32)[None, :]},
        execution_coordinates,
        source_map=source_map,
        source_mesh=mesh,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)
    if not validity.all_certified or certificate.status != "certified":
        raise AssertionError(
            f"Exact radical geometry did not certify: {certificate.findings}"
        )
    witnesses = tuple(
        (witness.theorem, witness.witness_id, witness.exact_values)
        for witness in certificate.witnesses
    )
    if not any(theorem == "mapped_cell_injective" for theorem, _, _ in witnesses):
        raise AssertionError(
            "The radical fixture did not retain its injectivity witness."
        )
    return (
        str(certificate.status),
        certificate.certificate_id,
        witnesses,
        _radical_boundary_contact_identity(),
    )


def _curved_p2_hexahedral_pair() -> tuple[CellMesh, CellGeometrySpec]:
    element = lagrange_element("hexahedron", 2)
    source_references = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    topology = reference_cell_topology("hexahedron")
    reference_vertices = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    vertex_ids = (
        (0, 1, 2, 3, 100, 101, 102, 103),
        (101, 102, 103, 100, 104, 105, 106, 107),
    )

    def image(
        point: tuple[Fraction, ...], cell_index: int, /
    ) -> tuple[Fraction, Fraction, Fraction]:
        first, second, third = point
        if cell_index == 0:
            return (
                first,
                second,
                third * (1 + first * second / 4),
            )
        return (
            1 - second,
            first,
            1 + third + first * (1 - second) / 4,
        )

    physical_vertices: dict[int, tuple[Fraction, Fraction, Fraction]] = {}
    for cell_index, ids in enumerate(vertex_ids):
        for point, identifier in zip(reference_vertices, ids, strict=True):
            physical = image(point, cell_index)
            previous = physical_vertices.get(identifier)
            if previous is not None and previous != physical:
                raise ValueError("Shared hexahedral corners must have exact images.")
            physical_vertices[identifier] = physical
    ordered_vertex_ids = tuple(sorted(physical_vertices))
    vertex_rows = {identifier: row for row, identifier in enumerate(ordered_vertex_ids)}
    mesh = CellMesh(
        np.asarray(
            tuple(
                tuple(float(value) for value in physical_vertices[identifier])
                for identifier in ordered_vertex_ids
            ),
            dtype=np.float64,
        ),
        (
            CellBlock(
                "curved",
                "hexahedron",
                np.asarray(
                    tuple(
                        tuple(vertex_rows[identifier] for identifier in ids)
                        for ids in vertex_ids
                    ),
                    dtype=np.int32,
                ),
                global_ids=np.asarray((701, 702), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(ordered_vertex_ids, dtype=np.int64),
    )
    coordinate_rows: list[tuple[int, ...]] = []
    coordinate_indices: dict[tuple[Fraction, Fraction, Fraction], int] = {}
    coordinate_values: list[tuple[Fraction, Fraction, Fraction]] = []
    for cell_index in range(2):
        route = []
        for point in source_references:
            physical = image(point, cell_index)
            index = coordinate_indices.get(physical)
            if index is None:
                index = len(coordinate_values)
                coordinate_indices[physical] = index
                coordinate_values.append(physical)
            route.append(index)
        coordinate_rows.append(tuple(route))
    geometry = CellGeometrySpec(
        {"curved": element},
        {"curved": np.asarray(coordinate_rows, dtype=np.int32)},
        np.asarray(
            tuple(tuple(float(value) for value in point) for point in coordinate_values),
            dtype=np.float64,
        ),
    )
    return mesh, geometry


def test_exact_center_jacobian_preconditioning_proves_mapped_injectivity() -> None:
    element = lagrange_element("quadrilateral", 2)
    source_references = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )

    def image(point: tuple[Fraction, ...], /) -> tuple[float, float]:
        x, y = point
        return float(x + 10 * y), float(y + x * x / 1024)

    topology = reference_cell_topology("quadrilateral")
    corner_references = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    mesh = CellMesh(
        np.asarray(tuple(image(point) for point in corner_references)),
        (
            CellBlock(
                "curved",
                "quadrilateral",
                np.arange(4, dtype=np.int32)[None, :],
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"curved": element},
        {"curved": np.arange(len(source_references), dtype=np.int32)[None, :]},
        np.asarray(tuple(image(point) for point in source_references)),
    )

    (cell,) = prepare_embedding_cells(mesh, geometry)
    witness = prove_mapped_cell_injective(cell)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert witness is not None
    assert witness.projection_rows == (0, 1)
    assert witness.projection_matrix == (
        (Fraction(512, 507), Fraction(-5120, 507)),
        (Fraction(-1, 1014), Fraction(512, 507)),
    )
    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert any(
        witness.theorem == "mapped_cell_injective" for witness in certificate.witnesses
    )


def test_forward_reference_chart_composition_checks_exact_nonaffine_maps() -> None:
    amplitude = Fraction(1, 64)
    bubble = {
        (1, 1, 1): Fraction(1),
        (2, 1, 1): Fraction(-1),
        (1, 2, 1): Fraction(-1),
        (1, 1, 2): Fraction(-1),
        (2, 2, 1): Fraction(1),
        (2, 1, 2): Fraction(1),
        (1, 2, 2): Fraction(1),
        (2, 2, 2): Fraction(-1),
    }
    chart_x: dict[tuple[int, ...], ExactScalar] = {
        exponent: amplitude * coefficient for exponent, coefficient in bubble.items()
    }
    chart_x[(1, 0, 0)] = Fraction(1)
    chart_y: dict[tuple[int, ...], ExactScalar] = {(0, 1, 0): Fraction(1)}
    chart_z: dict[tuple[int, ...], ExactScalar] = {(0, 0, 1): Fraction(1)}
    chart_arguments: tuple[Expression, ...] = (chart_x, chart_y, chart_z)
    parent = _mapped_hexahedron(
        801,
        (0, 1, 2, 3, 4, 5, 6, 7),
        (
            {(1, 0, 0): Fraction(1)},
            {(0, 1, 0): Fraction(1)},
            {(0, 0, 1): Fraction(1)},
        ),
    )
    target = _mapped_hexahedron(
        802,
        (100, 101, 102, 103, 104, 105, 106, 107),
        chart_arguments,
    )
    mismatch_x: dict[tuple[int, ...], ExactScalar] = dict(chart_x)
    mismatch_x[(0, 0, 0)] = Fraction(1, 128)
    mismatched_target = _mapped_hexahedron(
        803,
        (200, 201, 202, 203, 204, 205, 206, 207),
        (mismatch_x, chart_y, chart_z),
    )

    assert reference_chart_composition_equal(parent, target, chart_arguments) is True
    assert (
        reference_chart_composition_equal(parent, mismatched_target, chart_arguments)
        is False
    )


def test_curved_shared_edge_uses_two_factor_interior_signs() -> None:
    first_coordinates: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1), (0, 0, 1): Fraction(1)},
        {
            (0, 0, 1): Fraction(1),
            (2, 0, 0): Fraction(1, 4),
        },
    )
    second_coordinates: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(-1), (0, 0, 1): Fraction(-1)},
        {
            (0, 0, 1): Fraction(-1),
            (2, 0, 0): Fraction(1, 4),
        },
    )
    first = _mapped_hexahedron(
        501, (100, 101, 102, 103, 104, 105, 106, 107), first_coordinates
    )
    second = _mapped_hexahedron(
        502, (100, 101, 202, 203, 204, 205, 206, 207), second_coordinates
    )

    assert restriction_trace_equal(first, second, (100, 101)) is True
    witness = _shared_edge_separation(first, second, (100, 101))

    assert witness is not None
    assert witness[1] == -witness[2]
    assert witness[3] and witness[4]


def test_two_factor_edge_proof_rejects_an_extra_shared_facet() -> None:
    first_coordinates: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    second_coordinates: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(-1)},
        {(0, 0, 1): Fraction(1)},
    )
    first = _mapped_hexahedron(
        511, (100, 101, 102, 103, 104, 105, 106, 107), first_coordinates
    )
    second = _mapped_hexahedron(
        512, (100, 101, 212, 213, 214, 215, 216, 217), second_coordinates
    )

    assert restriction_trace_equal(first, second, (100, 101)) is True
    assert _shared_edge_separation(first, second, (100, 101)) is None


def test_curved_shared_edge_cone_separation_handles_two_factor_sign_changes() -> None:
    first_coordinates: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1), (0, 0, 1): Fraction(-1)},
        {
            (0, 0, 1): Fraction(1),
            (2, 0, 0): Fraction(1, 4),
            (0, 1, 0): Fraction(1),
        },
    )
    second_coordinates: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(-1), (0, 0, 1): Fraction(1)},
        {
            (0, 0, 1): Fraction(-1),
            (2, 0, 0): Fraction(1, 4),
            (0, 1, 0): Fraction(-1),
        },
    )
    first = _mapped_hexahedron(
        521, (100, 101, 102, 103, 104, 105, 106, 107), first_coordinates
    )
    second = _mapped_hexahedron(
        522, (100, 101, 222, 223, 224, 225, 226, 227), second_coordinates
    )

    assert restriction_trace_equal(first, second, (100, 101)) is True
    assert _shared_edge_separation(first, second, (100, 101)) is None
    witness = _shared_edge_cone_separation(first, second, (100, 101))

    assert witness is not None
    assert any(value.startswith("separator:") for value in witness)


def test_rotated_quad_trace_proves_exact_implicit_facet_separation() -> None:
    zero = (0, 0, 0)
    first_coordinates: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {
            (0, 0, 1): Fraction(1),
            (1, 1, 1): Fraction(1, 4),
        },
    )
    second_coordinates: tuple[Expression, ...] = (
        {zero: Fraction(1), (0, 1, 0): Fraction(-1)},
        {(1, 0, 0): Fraction(1)},
        {
            zero: Fraction(1),
            (0, 0, 1): Fraction(1),
            (1, 0, 0): Fraction(1, 4),
            (1, 1, 0): Fraction(-1, 4),
        },
    )
    first = _mapped_hexahedron(601, (0, 1, 2, 3, 100, 101, 102, 103), first_coordinates)
    second = _mapped_hexahedron(
        602, (101, 102, 103, 100, 104, 105, 106, 107), second_coordinates
    )
    shared = (100, 101, 102, 103)

    assert restriction_trace_equal(first, second, shared) is True
    assert restriction_trace_equal(second, first, shared) is True
    witness = _shared_curved_facet_separation(first, second, shared)

    assert witness is not None
    assert witness[1] == -witness[2]


def test_exact_p2_curved_facet_contacts_cold_recompute_without_root_source() -> None:
    mesh, geometry = _curved_p2_hexahedral_pair()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    certificate = certify_global_embedding(mesh, geometry, validity)
    recomputed = certify_global_embedding(mesh, geometry, validity)

    assert geometry.restriction_source is None
    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert recomputed.status == "certified", recomputed.findings
    assert certificate.certificate_id == recomputed.certificate_id
    assert tuple(
        (witness.theorem, witness.entity_ids, witness.exact_values)
        for witness in certificate.witnesses
    ) == tuple(
        (witness.theorem, witness.entity_ids, witness.exact_values)
        for witness in recomputed.witnesses
    )
    exact_values = tuple(
        value for witness in certificate.witnesses for value in witness.exact_values
    )
    assert any(
        "exact-opposite-side-curved-shared-facet" in value for value in exact_values
    )
    assert not any("same-root:" in value for value in exact_values)


def test_global_degree_refuses_mismatched_exact_interior_trace() -> None:
    mesh, source_geometry = _curved_p2_hexahedral_pair()
    block_name = source_geometry.block_names[0]
    element = lagrange_element("hexahedron", 2)
    reference_nodes = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    lower = tuple(min(point[axis] for point in reference_nodes) for axis in range(3))
    upper = tuple(max(point[axis] for point in reference_nodes) for axis in range(3))
    shared_face_center = (
        (lower[0] + upper[0]) / 2,
        (lower[1] + upper[1]) / 2,
        lower[2],
    )
    node_index = reference_nodes.index(shared_face_center)
    coordinates = np.asarray(source_geometry.coordinates).copy()
    routes = np.asarray(source_geometry.geometry_dofs[0], dtype=np.int32).copy()
    changed_point = coordinates[routes[1, node_index]].copy()
    changed_point[2] += 1 / 4096
    routes[1, node_index] = len(coordinates)
    coordinates = np.concatenate((coordinates, changed_point[None, :]), axis=0)
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: routes},
        coordinates,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(
        finding.check in ("mapped_shared_trace", "mismatched_interior_trace")
        and finding.status == "violated"
        for finding in certificate.findings
    )
    assert not any(
        witness.theorem == "global_degree_one_injectivity"
        for witness in certificate.witnesses
    )


def _embedded_triangles(
    coordinates: np.ndarray, cells: tuple[tuple[int, int, int], ...]
) -> tuple[CellMesh, CellGeometrySpec, CellValidityCertificate]:
    mesh = CellMesh(
        coordinates,
        (CellBlock("triangles", "triangle", np.asarray(cells, dtype=np.int32)),),
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    return mesh, geometry, validity


def _embedded_tetrahedra(
    coordinates: np.ndarray, cells: tuple[tuple[int, int, int, int], ...]
) -> tuple[CellMesh, CellGeometrySpec, CellValidityCertificate]:
    mesh = CellMesh(
        coordinates,
        (CellBlock("tetrahedra", "tetrahedron", np.asarray(cells, dtype=np.int32)),),
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    return mesh, geometry, validity


def _curved_restricted_quadrilaterals(
    charts: tuple[tuple[Fraction, Fraction], ...],
    /,
    *,
    retained_root: bool = True,
    preserve_vertex_identity: bool = True,
) -> tuple[CellMesh, CellGeometrySpec, CellValidityCertificate]:
    topology = reference_cell_topology("quadrilateral")
    target_references = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    source_element = lagrange_element("quadrilateral", 2)
    source_references = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(source_element.reference_nodes)
    )

    def image(point: tuple[Fraction, ...], /) -> tuple[float, float]:
        x, y = point
        return float(x), float(y + Fraction(1, 4) * x * (1 - x))

    source_coordinates = np.asarray(
        tuple(image(point) for point in source_references), dtype=np.float64
    )
    root_vertices = np.asarray(
        tuple(image(point) for point in target_references), dtype=np.float64
    )
    root_vertex_ids = np.asarray((101, 102, 103, 104), dtype=np.int64)
    root_mesh = CellMesh(
        root_vertices,
        (
            CellBlock(
                "root",
                "quadrilateral",
                np.arange(4, dtype=np.int32)[None, :],
                global_ids=np.asarray((71,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=root_vertex_ids,
    )
    root_geometry = CellGeometrySpec(
        {"root": source_element},
        {"root": np.arange(len(source_references), dtype=np.int32)[None, :]},
        source_coordinates,
    )

    zero, axis_x, axis_y = (0, 0), (1, 0), (0, 1)
    chart_arguments: tuple[tuple[ExactReferenceChartExpression, ...], ...] = tuple(
        (
            {zero: origin, axis_x: width},
            {axis_y: Fraction(1)},
        )
        for origin, width in charts
    )
    exact_source_corners = tuple(
        tuple(
            (
                origin + width * reference[0],
                reference[1],
            )
            for reference in target_references
        )
        for origin, width in charts
    )
    target_ids = np.arange(901, 901 + len(charts), dtype=np.int64)
    reference_witness = ReferenceCellMapWitnesses(
        np.arange(len(charts), dtype=np.int64),
        np.full((len(charts),), 71, dtype=np.int64),
        "quadrilateral",
        "quadrilateral",
        np.asarray(
            [
                [[float(value) for value in point] for point in corners]
                for corners in exact_source_corners
            ],
            dtype=np.float64,
        ),
        exact_source_corners,
        chart_arguments,
    )

    target_dof_coordinates = np.asarray(
        tuple(
            image((origin + width * reference[0], reference[1]))
            for origin, width in charts
            for reference in source_references
        ),
        dtype=np.float64,
    )
    target_routes = np.arange(
        len(charts) * len(source_references), dtype=np.int32
    ).reshape((len(charts), len(source_references)))
    vertex_rows: list[tuple[int, ...]] = []
    vertex_points: list[tuple[float, float]] = []
    vertex_lookup: dict[tuple[Fraction, Fraction], int] = {}
    for origin, width in charts:
        row = []
        for reference in target_references:
            source_point = (origin + width * reference[0], reference[1])
            index = vertex_lookup.get(source_point) if preserve_vertex_identity else None
            if index is None:
                index = len(vertex_points)
                vertex_points.append(image(source_point))
                if preserve_vertex_identity:
                    vertex_lookup[source_point] = index
            row.append(index)
        vertex_rows.append(tuple(row))
    target_mesh = CellMesh(
        np.asarray(vertex_points, dtype=np.float64),
        (
            CellBlock(
                "curved",
                "quadrilateral",
                np.asarray(vertex_rows, dtype=np.int32),
                global_ids=target_ids,
            ),
        ),
        vertex_global_ids=np.arange(2001, 2001 + len(vertex_points), dtype=np.int64),
    )
    restriction_source = None
    if retained_root:
        restriction_source = CellGeometryRestrictionSource(
            cell_geometry_id(root_geometry),
            root_mesh.topology_id,
            {"curved": np.full((len(charts),), 71, dtype=np.int64)},
            {"curved": np.tile(root_vertex_ids, (len(charts), 1))},
            block_parent_reference_maps={"curved": reference_witness},
        )
    geometry = CellGeometrySpec(
        {"curved": source_element},
        {"curved": target_routes},
        target_dof_coordinates,
        restriction_source=restriction_source,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=target_mesh)
    return target_mesh, geometry, validity


def test_affine_planar_cells_certify_only_their_declared_shared_edge() -> None:
    mesh, geometry, validity = _embedded_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ((0, 1, 2), (0, 2, 3)),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.cell_count == certificate.source_bound_count == 2
    assert {witness.theorem for witness in certificate.witnesses} >= {
        "positive_local_validity",
        "complete_global_cell_contact_theorem",
    }
    restored = GlobalEmbeddingCertificate.from_stored_fields(certificate)
    assert restored.certificate_id == certificate.certificate_id
    restored.require_current(mesh, geometry)


def test_positive_local_jacobians_do_not_hide_overlapping_planar_cells() -> None:
    mesh, geometry, validity = _embedded_triangles(
        np.asarray(
            (
                (0.0, 0.0),
                (2.0, 0.0),
                (0.0, 2.0),
                (0.25, 0.25),
                (2.25, 0.25),
                (0.25, 2.25),
            )
        ),
        ((0, 1, 2), (3, 4, 5)),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(finding.check == "cell_contact" for finding in certificate.findings)


def test_affine_volume_neighbors_share_only_their_exact_face() -> None:
    mesh, geometry, validity = _embedded_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        ),
        ((0, 1, 2, 3), (0, 2, 1, 4)),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.boundary_facet_count > 0
    assert certificate.shell_count == 1
    assert "exterior_degree" in certificate.evaluated_checks
    assert "global_degree_one_injectivity" in certificate.evaluated_checks
    assert any(
        witness.theorem == "global_degree_one_injectivity"
        for witness in certificate.witnesses
    )


def test_planar_nonlinear_boundary_face_requires_exact_image_proof() -> None:
    x, y, z = axes(3)
    bubble = multiply(
        multiply(x, y),
        subtract(subtract(subtract(constant(1, 3), x), y), z),
    )
    curved_x = add(x, scale(bubble, Fraction(1, 4)))
    cell = _mapped_tetrahedron(1, (0, 1, 2, 3), (curved_x, y, z))
    facet = next(
        route
        for route in cell.facet_routes
        if set(cell.vertex_ids[index] for index in route) == {0, 1, 2}
    )
    vertex_ids = tuple(cell.vertex_ids[index] for index in facet)
    points = cast(
        tuple[Point3, ...],
        tuple(cell.corner_images[index] for index in facet),
    )
    mesh = CellMesh(
        np.asarray(
            tuple(tuple(float(value) for value in point) for point in cell.corner_images),
            dtype=np.float64,
        ),
        (
            CellBlock(
                "mapped-tetrahedra",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((1,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((0, 1, 2, 3), dtype=np.int64),
    )
    boundary = build_oriented_boundary(mesh, (cell,))
    assert boundary.status == "certified"
    assert boundary.boundary_degree_one
    assert (
        sum(
            face.map_evidence[0] == "exact-planar-injective-face-image"
            for face in boundary.faces
        )
        == 1
    )

    evidence = _face_trace_image_evidence(cell, vertex_ids, points)
    assert evidence is not None
    assert evidence[0] == "exact-planar-injective-face-image"
    assert any(item.startswith("edge:") for item in evidence)
    folded_x = add(x, scale(bubble, Fraction(8)))
    folded = _mapped_tetrahedron(3, (0, 1, 2, 3), (folded_x, y, z))
    folded_facet = next(
        route
        for route in folded.facet_routes
        if set(folded.vertex_ids[index] for index in route) == {0, 1, 2}
    )
    folded_ids = tuple(folded.vertex_ids[index] for index in folded_facet)
    folded_points = cast(
        tuple[Point3, ...],
        tuple(folded.corner_images[index] for index in folded_facet),
    )
    first_image = tuple(
        evaluate_expression(expression, (Fraction(1, 4), Fraction(1, 2), Fraction(0)))
        for expression in folded.coordinates
    )
    second_image = tuple(
        evaluate_expression(expression, (Fraction(1, 2), Fraction(1, 2), Fraction(0)))
        for expression in folded.coordinates
    )
    assert first_image == second_image
    assert _face_trace_image_evidence(folded, folded_ids, folded_points) is None

    curved_z = add(z, scale(bubble, Fraction(1, 4)))
    nonplanar = _mapped_tetrahedron(2, (0, 1, 2, 3), (x, y, curved_z))
    nonplanar_facet = next(
        route
        for route in nonplanar.facet_routes
        if set(nonplanar.vertex_ids[index] for index in route) == {0, 1, 2}
    )
    nonplanar_ids = tuple(nonplanar.vertex_ids[index] for index in nonplanar_facet)
    nonplanar_points = cast(
        tuple[Point3, ...],
        tuple(nonplanar.corner_images[index] for index in nonplanar_facet),
    )
    assert _face_trace_image_evidence(nonplanar, nonplanar_ids, nonplanar_points) is None


def test_global_degree_one_certifies_nonlinear_map_with_affine_exterior() -> None:
    topology = reference_cell_topology("hexahedron")
    element = lagrange_element("hexahedron", 2)
    references = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    vertices = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    lower = tuple(min(point[axis] for point in vertices) for axis in range(3))
    upper = tuple(max(point[axis] for point in vertices) for axis in range(3))
    amplitude = Fraction(1, 64)

    def image(point: tuple[Fraction, ...], /) -> tuple[Fraction, ...]:
        x, y, z = point
        bubble = (
            (x - lower[0])
            * (upper[0] - x)
            * (y - lower[1])
            * (upper[1] - y)
            * (z - lower[2])
            * (upper[2] - z)
        )
        return x + amplitude * bubble, y, z

    geometry_points = tuple(image(point) for point in references)
    mesh_points = tuple(image(point) for point in vertices)
    mesh = CellMesh(
        np.asarray(
            tuple(tuple(float(value) for value in point) for point in mesh_points)
        ),
        (
            CellBlock(
                "curved",
                "hexahedron",
                np.arange(len(vertices), dtype=np.int32)[None, :],
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"curved": element},
        {"curved": np.arange(len(references), dtype=np.int32)[None, :]},
        np.asarray(
            tuple(tuple(float(value) for value in point) for point in geometry_points)
        ),
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    degree_witness = next(
        witness
        for witness in certificate.witnesses
        if witness.theorem == "global_degree_one_injectivity"
    )
    assert (
        "global_degree_argument:embedded-oriented-sphere-has-winding-one-inside"
        in degree_witness.exact_values
    )


def test_global_embedding_radical_witness_identity_is_cold_process_stable() -> None:
    first = _radical_global_embedding_identity()
    assert first[0] == "certified"
    assert first == _radical_global_embedding_identity()

    repository = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "true"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository), environment.get("PYTHONPATH", ""))
    )
    child_code = (
        "import json; "
        "from tests.unit.geometry.test_global_embedding_producer "
        "import _radical_global_embedding_identity; "
        "print(json.dumps(_radical_global_embedding_identity()))"
    )
    process = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert json.loads(process.stdout) == json.loads(json.dumps(first))


def test_global_degree_refuses_crossing_exterior_triangles() -> None:
    first = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    second = first + np.asarray((0.5, 0.0, 0.0))
    mesh, geometry, validity = _embedded_tetrahedra(
        np.concatenate((first, second), axis=0),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(
        finding.check == "exterior_boundary_self_contact" and finding.status == "violated"
        for finding in certificate.findings
    )
    assert not any(
        witness.theorem == "global_degree_one_injectivity"
        for witness in certificate.witnesses
    )


def test_global_degree_never_masks_a_nonpositive_cell_jacobian() -> None:
    mesh, geometry, validity = _embedded_tetrahedra(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        ((0, 2, 1, 3),),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert not validity.all_certified
    assert certificate.status == "violated"
    assert any(
        finding.check == "cell_validity" and finding.status == "violated"
        for finding in certificate.findings
    )
    assert not any(
        witness.theorem == "global_degree_one_injectivity"
        for witness in certificate.witnesses
    )


def test_affine_volume_cells_sharing_only_an_edge_still_certify() -> None:
    mesh, geometry, validity = _embedded_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
            )
        ),
        ((0, 2, 3, 1), (0, 4, 5, 1)),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert "exact_cell_contacts" in certificate.evaluated_checks


def test_affine_volume_cells_sharing_only_a_vertex_still_certify() -> None:
    mesh, geometry, validity = _embedded_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0),
            )
        ),
        ((0, 1, 2, 3), (0, 4, 6, 5)),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert "exact_cell_contacts" in certificate.evaluated_checks


def test_positive_tetrahedral_validity_does_not_hide_contained_overlap() -> None:
    large = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    small = 0.2 * large + 0.1
    mesh, geometry, validity = _embedded_tetrahedra(
        np.concatenate((large, small), axis=0),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(finding.check == "cell_contact" for finding in certificate.findings)


def test_affine_prism_uses_exact_cell_decomposition() -> None:
    prism_coordinates = np.asarray(
        reference_cell_topology("prism").vertices, dtype=np.float64
    )
    prism = CellMesh(
        prism_coordinates,
        (CellBlock("prisms", "prism", np.arange(6, dtype=np.int32)[None, :]),),
    )
    geometry = CellGeometrySpec.affine(prism)
    validity = certify_cell_geometry_validity(geometry, mesh=prism)

    certificate = certify_global_embedding(prism, geometry, validity)

    assert validity.all_certified and certificate.status == "certified"


def _convex_cube_polyhedron() -> tuple[
    CellMesh, CellGeometrySpec, CellValidityCertificate
]:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    mesh = CellMesh.from_polyhedra(
        coordinates,
        [[np.asarray(face, dtype=np.int32) for face in faces]],
    )
    geometry = CellGeometrySpec.affine(mesh)
    return mesh, geometry, certify_cell_geometry_validity(geometry, mesh=mesh)


def test_convex_polyhedral_cell_uses_exact_face_decomposition() -> None:
    mesh, geometry, validity = _convex_cube_polyhedron()

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified and certificate.status == "certified"


def test_single_curved_q2_cell_certifies_by_exact_local_and_global_proofs() -> None:
    mesh, geometry, validity = _curved_restricted_quadrilaterals(
        ((Fraction(0), Fraction(1)),)
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert any(
        witness.theorem == "mapped_cell_injective" for witness in certificate.witnesses
    )


def test_adjacent_curved_q2_cells_certify_exact_same_root_trace() -> None:
    mesh, geometry, validity = _curved_restricted_quadrilaterals(
        ((Fraction(0), Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 2)))
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert any(
        "same-root:" in value and "injective-chart-contact" in value
        for witness in certificate.witnesses
        for value in witness.exact_values
    )
    restored = GlobalEmbeddingCertificate.from_stored_fields(certificate)
    assert restored.certificate_id == certificate.certificate_id


def test_adjacent_curved_cells_prove_the_shared_facet_without_root_ancestry() -> None:
    mesh, geometry, validity = _curved_restricted_quadrilaterals(
        ((Fraction(0), Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 2))),
        retained_root=False,
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert any(
        "exact-opposite-side-shared-facet" in value
        for witness in certificate.witnesses
        for value in witness.exact_values
    )


def test_disjoint_curved_cells_with_overlapping_aabbs_use_exact_projection_bounds() -> (
    None
):
    element = lagrange_element("quadrilateral", 2)
    nodes = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    corners = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology("quadrilateral").vertices
    )
    translations = ((Fraction(0), Fraction(0)), (Fraction(3, 2), Fraction(-3, 2)))

    def image(
        point: tuple[Fraction, ...],
        offset: tuple[Fraction, ...],
        /,
    ) -> tuple[float, float]:
        u, v = point
        return (
            float(u + v - 1 + Fraction(1, 8) * u * (1 - u) + offset[0]),
            float(-u + v + offset[1]),
        )

    mesh_points = tuple(
        image(point, offset) for offset in translations for point in corners
    )
    mesh = CellMesh(
        np.asarray(mesh_points, dtype=np.float64),
        (
            CellBlock(
                "curved",
                "quadrilateral",
                np.asarray((tuple(range(4)), tuple(range(4, 8))), dtype=np.int32),
            ),
        ),
        vertex_global_ids=np.arange(3001, 3009, dtype=np.int64),
    )
    coordinate_values = np.asarray(
        tuple(image(point, offset) for offset in translations for point in nodes),
        dtype=np.float64,
    )
    routes = np.arange(2 * len(nodes), dtype=np.int32).reshape((2, len(nodes)))
    geometry = CellGeometrySpec(
        {"curved": element},
        {"curved": routes},
        coordinate_values,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert any(
        "outward-bernstein-projection-separation" in value
        for witness in certificate.witnesses
        for value in witness.exact_values
    )


def test_overlapping_same_root_curved_cells_report_exact_chart_violation() -> None:
    mesh, geometry, validity = _curved_restricted_quadrilaterals(
        ((Fraction(0), Fraction(3, 4)), (Fraction(1, 4), Fraction(3, 4)))
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(
        finding.check == "mapped_cell_contact"
        and finding.status == "violated"
        and "same-root injective-map" in finding.detail
        for finding in certificate.findings
    )


def test_unrelated_curved_overlap_is_not_assumed_to_share_a_root() -> None:
    mesh, geometry, validity = _curved_restricted_quadrilaterals(
        ((Fraction(0), Fraction(1)), (Fraction(1, 4), Fraction(1))),
        retained_root=False,
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "unresolved"
    assert any(
        finding.check == "mapped_cell_contact"
        and finding.status == "unresolved"
        and "no authoritative common-root" in finding.detail
        for finding in certificate.findings
    )


def test_unrelated_curved_duplicate_corners_prove_contact_violation() -> None:
    mesh, geometry, validity = _curved_restricted_quadrilaterals(
        ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1))),
        retained_root=False,
        preserve_vertex_identity=False,
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(
        finding.check == "mapped_cell_contact"
        and finding.status == "violated"
        and "Distinct mapped corner identities" in finding.detail
        for finding in certificate.findings
    )
