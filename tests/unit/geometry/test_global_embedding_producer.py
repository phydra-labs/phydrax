from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from random import Random
from typing import cast

import equinox as eqx
import numpy as np
import pytest

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
    _BlockCertificate,
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
    Polynomial,
    polynomial_bounds,
    scale,
    subtract,
)
from phydrax.discretization._exact_algebraic_geometry import (
    ExactScalar,
    RadicalElement,
    rational_coefficient,
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
from phydrax.geometry._exact_polyhedral_geometry import (
    outward_spatial_triangle_enclosures,
    Point3,
    point_on_segment2,
    triangle_contact_points2,
)
from phydrax.geometry._global_embedding import (
    _affine_chart_cell,
    _cartesian_chart_contact,
    _full_column_rank_modulo_prime,
    _nonperiodic_contacts,
    _planar_faced_chart_cell,
    _rational_markowitz_nullspace,
    _shared_curved_facet_separation,
    _shared_edge_cone_separation,
    _shared_edge_separation,
    _SpatialTriangleFilter,
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from phydrax.geometry._mapped_embedding import (
    MappedInjectivityWitness,
    prove_mapped_cell_injective,
)
from phydrax.geometry._planar_primitives import TriangleEnclosures
from phydrax.geometry._restricted_embedding import (
    rational_simplex_trace_values,
    reference_chart_composition_equal,
    restriction_trace_equal,
)
from phydrax.linalg._exact import factor_exact_sparse
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


def test_planar_faced_physical_map_requires_full_injectivity_and_face_images() -> None:
    u, v, w = axes(3)
    bubble = multiply(
        multiply(u, subtract(constant(1, 3), u)),
        multiply(
            multiply(v, subtract(constant(1, 3), v)),
            multiply(w, subtract(constant(1, 3), w)),
        ),
    )
    cell = _mapped_hexahedron(
        1, tuple(range(8)), (add(u, scale(bubble, Fraction(1, 10))), v, w)
    )
    assert _planar_faced_chart_cell(cell, cell.coordinates) is not None
    # Same corners, but a bowed face: corner planarity is not sufficient.
    face_bump = multiply(v, subtract(constant(1, 3), v))
    bowed = _mapped_hexahedron(
        2, tuple(range(8)), (add(u, scale(face_bump, Fraction(1, 10))), v, w)
    )
    assert _planar_faced_chart_cell(bowed, bowed.coordinates) is None
    # A folded map retaining the same boundary is not an injective solid.
    folded = _mapped_hexahedron(3, tuple(range(8)), (add(u, scale(bubble, 100)), v, w))
    assert _planar_faced_chart_cell(folded, folded.coordinates) is None


def test_planar_contact_reuses_only_the_proved_physical_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.geometry._global_embedding as embedding

    u, v, w = axes(3)
    bubble = multiply(
        multiply(u, subtract(constant(1, 3), u)),
        multiply(
            multiply(v, subtract(constant(1, 3), v)),
            multiply(w, subtract(constant(1, 3), w)),
        ),
    )
    cell = _mapped_hexahedron(
        1, tuple(range(8)), (add(u, scale(bubble, Fraction(1, 10))), v, w)
    )
    assert embedding.prove_mapped_cell_injective(cell) is not None
    original = embedding.prove_mapped_cell_injective
    calls = 0

    def count_proofs(candidate: EmbeddingCell, /) -> MappedInjectivityWitness | None:
        nonlocal calls
        calls += 1
        return original(candidate)

    monkeypatch.setattr(embedding, "prove_mapped_cell_injective", count_proofs)
    assert (
        embedding._planar_faced_chart_cell(
            cell, cell.coordinates, physical_map_injective=True
        )
        is not None
    )
    assert calls == 0
    changed = (add(u, scale(bubble, 100)), v, w)
    assert (
        embedding._planar_faced_chart_cell(cell, changed, physical_map_injective=True)
        is None
    )
    assert calls == 1


@pytest.mark.parametrize(
    "shift,expected", [(Fraction(1), "clear"), (Fraction(1, 2), "overlap")]
)
def test_planar_faced_mapped_cells_preserve_exact_contact(
    shift: Fraction,
    expected: str,
) -> None:
    u, v, w = axes(3)
    bubble = multiply(
        multiply(u, subtract(constant(1, 3), u)),
        multiply(
            multiply(v, subtract(constant(1, 3), v)),
            multiply(w, subtract(constant(1, 3), w)),
        ),
    )
    base = add(u, scale(bubble, Fraction(1, 16)))
    maps = ((base, v, w), (add(base, constant(shift, 3)), v, w))
    topology = reference_cell_topology("hexahedron")
    reference_vertices = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    element = lagrange_element("hexahedron", 2)
    reference_nodes = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in element.reference_nodes
    )
    vertices: list[tuple[ExactScalar, ...]] = []
    vertex_rows: dict[tuple[ExactScalar, ...], int] = {}
    connectivity: list[tuple[int, ...]] = []
    dofs: list[tuple[ExactScalar, ...]] = []
    routes: list[tuple[int, ...]] = []
    for expressions in maps:
        corners = tuple(
            tuple(evaluate_expression(expression, point) for expression in expressions)
            for point in reference_vertices
        )
        row = []
        for corner in corners:
            if corner not in vertex_rows:
                vertex_rows[corner] = len(vertices)
                vertices.append(corner)
            row.append(vertex_rows[corner])
        connectivity.append(tuple(row))
        start = len(dofs)
        dofs.extend(
            tuple(evaluate_expression(expression, point) for expression in expressions)
            for point in reference_nodes
        )
        routes.append(tuple(range(start, len(dofs))))
    mesh = CellMesh(
        np.asarray(vertices, dtype=np.float64),
        (CellBlock("hex", "hexahedron", np.asarray(connectivity)),),
    )
    geometry = CellGeometrySpec(
        {"hex": element}, {"hex": np.asarray(routes)}, np.asarray(dofs, dtype=np.float64)
    )
    findings = _nonperiodic_contacts(
        mesh, geometry, prepare_embedding_cells(mesh, geometry)
    )
    if expected == "clear":
        assert not findings
    else:
        assert any(
            finding.check == "cell_contact" and finding.status == "violated"
            for finding in findings
        )


def test_cartesian_root_chart_contact_checks_complete_declared_stratum() -> None:
    variables = axes(3)
    reference = reference_cell_topology("hexahedron").vertices
    first_ids = tuple(range(8))
    by_corner = {tuple(point): first_ids[row] for row, point in enumerate(reference)}
    touching_ids = tuple(
        by_corner[(1.0, point[1], point[2])] if point[0] == 0 else 100 + row
        for row, point in enumerate(reference)
    )
    first = _affine_chart_cell(_mapped_hexahedron(1, first_ids, variables), variables)
    shifted = (add(variables[0], constant(1, 3)), *variables[1:])
    touching = _affine_chart_cell(_mapped_hexahedron(2, touching_ids, variables), shifted)
    assert first is not None and touching is not None
    assert _cartesian_chart_contact(first, touching) == "declared"
    overlapping = _affine_chart_cell(
        _mapped_hexahedron(3, tuple(range(200, 208)), variables),
        (add(variables[0], constant(Fraction(1, 2), 3)), *variables[1:]),
    )
    assert overlapping is not None
    assert _cartesian_chart_contact(first, overlapping) == "outside"
    separated = _affine_chart_cell(
        _mapped_hexahedron(4, tuple(range(300, 308)), variables),
        (add(variables[0], constant(2, 3)), *variables[1:]),
    )
    assert separated is not None
    assert _cartesian_chart_contact(first, separated) == "declared"


def test_cartesian_corners_do_not_prove_an_affine_chart() -> None:
    variables = axes(3)
    curved: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(2), (2, 0, 0): Fraction(-1)},
        variables[1],
        variables[2],
    )
    cell = _mapped_hexahedron(5, tuple(range(8)), curved)
    assert set(cell.corner_images) == set(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology("hexahedron").vertices
    )
    assert _affine_chart_cell(cell, curved) is None
    assert _cartesian_chart_contact(cell, cell) is None


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


def test_simplex_image_bounds_fall_back_when_interval_is_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.geometry._global_embedding as embedding

    u, v = axes(2)
    coordinates = (add(u, v), subtract(u, v))

    def inconclusive(
        expression: Expression, corners: tuple[tuple[Fraction, ...], ...], /
    ) -> tuple[float, float]:
        return -float("inf"), float("inf")

    monkeypatch.setattr(embedding, "restricted_polynomial_bounds", inconclusive)
    root = embedding._root_reference_cell("triangle", coordinates, 1)
    assert root is not None
    chart = embedding._affine_chart_cell(root, coordinates)
    assert chart is not None
    expected = tuple(
        polynomial_bounds(expression, "simplex", 2) for expression in coordinates
    )
    assert tuple(zip(root.image_lower, root.image_upper, strict=True)) == expected
    assert tuple(zip(chart.image_lower, chart.image_upper, strict=True)) == expected


def test_strict_simplex_separation_falls_back_for_overlapping_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.geometry._global_embedding as embedding

    u, v = axes(2)
    first = embedding._root_reference_cell("triangle", (u, v), 1)
    separated = embedding._root_reference_cell("triangle", (add(u, constant(2, 2)), v), 2)
    overlapping = embedding._root_reference_cell(
        "triangle", (add(u, constant(Fraction(1, 2), 2)), v), 3
    )
    assert first is not None and separated is not None and overlapping is not None

    def wide_bounds(
        expression: Expression, corners: tuple[tuple[Fraction, ...], ...], /
    ) -> tuple[float, float]:
        return -100.0, 100.0

    def candidate_planes(
        first: EmbeddingCell, second: EmbeddingCell, shadows: dict[int, object], /
    ) -> tuple[tuple[Fraction, ...], ...]:
        return ((Fraction(1), Fraction(0)),)

    monkeypatch.setattr(embedding, "restricted_polynomial_bounds", wide_bounds)
    monkeypatch.setattr(embedding, "strict_separation_planes", candidate_planes)
    assert embedding._strict_image_separation(first, separated, {})
    assert not embedding._strict_image_separation(first, overlapping, {})


def test_strict_simplex_separation_accepts_certified_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.geometry._global_embedding as embedding

    u, v = axes(2)
    first = embedding._root_reference_cell("triangle", (u, v), 1)
    second = embedding._root_reference_cell("triangle", (add(u, constant(2, 2)), v), 2)
    assert first is not None and second is not None

    def candidate_planes(
        first: EmbeddingCell, second: EmbeddingCell, shadows: dict[int, object], /
    ) -> tuple[tuple[Fraction, ...], ...]:
        return ((Fraction(1), Fraction(0)),)

    monkeypatch.setattr(embedding, "strict_separation_planes", candidate_planes)
    assert embedding._strict_image_separation(first, second, {})


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
    edge_witness = _shared_edge_separation(
        contact_cells[0], contact_cells[1], (101, 103), {}
    )
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
    if not all(
        "control_digests:" in expression and "sha256:" in expression
        for theorem, _, values in witnesses
        if theorem == "mapped_cell_injective"
        for expression in values
    ):
        raise AssertionError("Mapped injectivity evidence must retain control digests.")
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
    cache: dict[tuple[int, tuple[int, ...]], tuple[Expression, ...] | None] = {}
    assert restriction_trace_equal(first, second, (100, 101), cache=cache) is True
    assert len(cache) == 2
    assert restriction_trace_equal(second, first, (100, 101), cache=cache) is True
    assert len(cache) == 2
    witness = _shared_edge_separation(first, second, (100, 101), {})

    assert witness is not None
    assert witness[1] == -witness[2]
    assert witness[3] and witness[4]


def test_nonplanar_shared_edge_cannot_have_a_separating_plane() -> None:
    first_coordinates: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1), (2, 0, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1), (3, 0, 0): Fraction(1)},
    )
    second_coordinates: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(-1), (2, 0, 0): Fraction(1)},
        {(0, 0, 1): Fraction(-1), (3, 0, 0): Fraction(1)},
    )
    first = _mapped_hexahedron(
        503, (100, 101, 102, 103, 104, 105, 106, 107), first_coordinates
    )
    second = _mapped_hexahedron(
        504, (100, 101, 202, 203, 204, 205, 206, 207), second_coordinates
    )
    assert restriction_trace_equal(first, second, (100, 101)) is True
    assert _shared_edge_separation(first, second, (100, 101), {}) is None


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
    assert _shared_edge_separation(first, second, (100, 101), {}) is None


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
    assert _shared_edge_separation(first, second, (100, 101), {}) is None
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
        "exact-opposite-side-curved-shared-facet" in value
        or "glued-facet-monotone-map" in value
        for value in exact_values
    )
    assert not any("same-root:" in value for value in exact_values)


@pytest.mark.parametrize("overlap", (False, True), ids=("separated", "overlapping"))
def test_quadratic_parameter_on_shared_line_preserves_contact_contract(
    overlap: bool,
) -> None:
    from phydrax.meshing._curving import _straight_geometry

    second_side = (
        ((1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))
        if overlap
        else ((-2.0, -1.0, 0.0), (2.0, -1.0, 0.0))
    )
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
            *second_side,
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "cells",
                "simplex:3",
                np.asarray(((0, 1, 2, 3), (0, 1, 4, 5)), dtype=np.int32),
            ),
        ),
    )
    straight = _straight_geometry(mesh, 2)
    x, y, z = np.asarray(straight.coordinates).T
    height = z + z * z / 16.0
    # Height is strictly increasing on [0, 1]; the transverse map has
    # determinant 1 + 4 z². The final affine shear preserves injectivity.
    # Thus image overlap is exactly that of the authored straight tetrahedra.
    coordinates = np.column_stack(
        (
            x - 2.0 * y * z + height / 4.0 + 1.0 / 8.0,
            y + 2.0 * x * z - height / 8.0 - 1.0 / 4.0,
            height + 3.0 / 8.0,
        )
    )
    geometry = CellGeometrySpec(
        dict(zip(straight.block_names, straight.elements, strict=True)),
        dict(zip(straight.block_names, straight.geometry_dofs, strict=True)),
        coordinates,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified

    certificate = certify_global_embedding(mesh, geometry, validity)
    if overlap:
        assert certificate.status != "certified"
        assert any(
            finding.entity_kind == "cell" and finding.entity_ids == (0, 1)
            for finding in certificate.findings
        )
    else:
        assert certificate.status == "certified", certificate.findings


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


def _tetrahedra_sharing_a_curved_face(
    first_bump: Fraction, second_bump: Fraction, second_cubic: Fraction, /
) -> tuple[EmbeddingCell, EmbeddingCell]:
    """Mapped tetrahedra on either side of the reference face ``z = 0``.

    Both share vertices 10, 11 and 12. Their traces on that face are
    ``(x, y, bump * x * y)``, the second with ``cubic * x * y * (1 - x - y)``
    added, which vanishes at every corner and edge midpoint of the face.
    """
    x, y, z = axes(3)
    face_bubble = multiply(multiply(x, y), subtract(subtract(constant(1, 3), x), y))
    first = _mapped_tetrahedron(
        1, (10, 11, 12, 13), (x, y, add(z, scale(multiply(x, y), first_bump)))
    )
    second = _mapped_tetrahedron(
        2,
        (10, 11, 12, 14),
        (
            x,
            y,
            add(
                add(scale(z, Fraction(-1)), scale(multiply(x, y), second_bump)),
                scale(face_bubble, second_cubic),
            ),
        ),
    )
    return first, second


@pytest.mark.parametrize("order", (2, 3, 4, 5))
def test_random_rational_simplex_trace_matches_fraction_reference(order: int) -> None:
    rng = Random(817 + order)
    polynomials: tuple[Polynomial, ...] = tuple(
        {
            (x, y, z): Fraction(rng.randint(-9, 9), rng.randint(1, 11))
            for x in range(order + 1)
            for y in range(order + 1 - x)
            for z in range(order + 1 - x - y)
        }
        for _ in range(3)
    )
    polynomials[0][(1, 1, 0)] = Fraction(5, 7)
    first = _mapped_tetrahedron(1, (10, 11, 12, 13), polynomials)
    same_face = tuple(dict(polynomial) for polynomial in polynomials)
    same_face[0][(0, 0, 1)] += Fraction(7, 13)
    second = _mapped_tetrahedron(2, (10, 11, 12, 14), same_face)
    different_face = tuple(dict(polynomial) for polynomial in same_face)
    different_face[0][(1, 1, 0)] += Fraction(1, 17)
    third = _mapped_tetrahedron(3, (10, 11, 12, 15), different_face)

    for cell, expected in ((second, True), (third, False)):
        assert restriction_trace_equal(first, cell, (10, 11, 12)) is expected
    lattice = rational_simplex_trace_values(first, (10, 11, 12))
    assert lattice is not None
    actual_order, rows = lattice
    assert actual_order == order
    references = first.reference_vertices[:3]
    caught_dropped_term = False
    caught_wrong_scale = False
    for weights, values in rows:
        point = tuple(
            sum(
                (weight * reference[axis] for weight, reference in zip(weights, references, strict=True)),
                Fraction(0),
            )
            / order
            for axis in range(3)
        )
        expected_values = tuple(
            sum(
                (
                    coefficient
                    * point[0] ** exponent[0]
                    * point[1] ** exponent[1]
                    * point[2] ** exponent[2]
                    for exponent, coefficient in polynomial.items()
                ),
                Fraction(0),
            )
            for polynomial in polynomials
        )
        assert values == expected_values
        dropped = expected_values[0] - Fraction(5, 7) * point[0] * point[1]
        caught_dropped_term |= values[0] != dropped
        caught_wrong_scale |= values[0] != expected_values[0] / 2

    assert caught_dropped_term
    assert caught_wrong_scale


def test_simplex_traces_agree_exactly_on_their_shared_curved_face() -> None:
    first, second = _tetrahedra_sharing_a_curved_face(
        Fraction(1, 8), Fraction(1, 8), Fraction(0)
    )

    assert restriction_trace_equal(first, second, (10, 11, 12)) is True
    assert restriction_trace_equal(second, first, (11, 12)) is True
    values: dict[tuple[int, tuple[Fraction, ...]], tuple[ExactScalar, ...]] = {}
    assert (
        restriction_trace_equal(first, second, (10, 11, 12), point_values=values) is True
    )
    assert restriction_trace_equal(first, second, (10, 12), point_values=values) is True


def test_simplex_trace_mismatch_at_one_face_midpoint_is_refused() -> None:
    first, second = _tetrahedra_sharing_a_curved_face(
        Fraction(1, 8), Fraction(1, 8) + Fraction(1, 4096), Fraction(0)
    )

    # The maps differ only where x * y is nonzero: on the face and on the edge
    # opposite vertex 10, but not on the two edges through vertex 10.
    assert restriction_trace_equal(first, second, (10, 11, 12)) is False
    assert restriction_trace_equal(first, second, (11, 12)) is False
    assert restriction_trace_equal(first, second, (10, 11)) is True
    assert restriction_trace_equal(first, second, (10, 12)) is True


def test_simplex_trace_mismatch_invisible_to_quadratic_nodes_is_refused() -> None:
    # A cubic face bubble vanishes at all six quadratic nodes of the face, so
    # only a lattice of the maps' actual degree separates the traces.
    first, second = _tetrahedra_sharing_a_curved_face(
        Fraction(1, 8), Fraction(1, 8), Fraction(1, 64)
    )

    assert restriction_trace_equal(first, second, (10, 11, 12)) is False
    assert restriction_trace_equal(first, second, (10, 11)) is True
    assert restriction_trace_equal(first, second, (11, 12)) is True


def test_simplex_face_image_evidence_separates_affine_curved_and_planar_faces() -> None:
    x, y, z = axes(3)

    def evidence(
        cell: EmbeddingCell, face_ids: tuple[int, int, int], /
    ) -> tuple[str, ...] | None:
        images = tuple(
            cell.corner_images[cell.vertex_ids.index(identifier)]
            for identifier in face_ids
        )
        points: tuple[Point3, ...] = tuple(
            (image[0], image[1], image[2]) for image in images
        )
        return _face_trace_image_evidence(cell, face_ids, points)

    # The face z = 0 bends out of its corner plane; the face x = 0 stays affine.
    bent = _mapped_tetrahedron(
        1, (10, 11, 12, 13), (x, y, add(z, scale(multiply(x, y), Fraction(1, 8))))
    )
    assert evidence(bent, (10, 11, 12)) is None
    assert evidence(bent, (10, 12, 13)) == ("exact-affine-face-image",)
    # An in-plane cubic bubble keeps the face z = 0 in its plane but not affine:
    # only the complete planar image proof may accept it, never the affine one.
    bubble = multiply(multiply(x, y), subtract(subtract(constant(1, 3), x), y))
    sheared = _mapped_tetrahedron(
        2, (10, 11, 12, 13), (add(x, scale(bubble, Fraction(1, 8))), y, z)
    )
    planar = evidence(sheared, (10, 11, 12))
    assert planar is not None
    assert planar != ("exact-affine-face-image",)


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


@pytest.mark.parametrize("dimension", (2, 3))
def test_affine_simplex_preparation_retains_exact_corners_without_polynomials(
    dimension: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.geometry import _embedding_cells

    points = np.vstack((np.zeros((1, dimension)), np.eye(dimension)))
    kind = "triangle" if dimension == 2 else "tetrahedron"
    mesh = CellMesh(
        points,
        (CellBlock("cells", kind, np.arange(dimension + 1)[None, :]),),
    )
    geometry = CellGeometrySpec.affine(mesh)

    def unexpected_polynomial(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Affine simplex preparation must use its exact corner frame."
        )

    with monkeypatch.context() as compact:
        compact.setattr(_embedding_cells, "_cell_expressions", unexpected_polynomial)
        compact.setattr(_embedding_cells, "_affine_expressions", unexpected_polynomial)
        (cell,) = prepare_embedding_cells(mesh, geometry)
    assert cell.coordinate_expressions is None
    assert cell.corner_images == tuple(geometry.source_coordinates())
    assert (
        tuple(
            tuple(
                evaluate_expression(expression, point) for expression in cell.coordinates
            )
            for point in cell.reference_vertices
        )
        == cell.corner_images
    )


@pytest.mark.parametrize("fallback", (False, True))
def test_curved_cell_image_boxes_retain_interior_contact(
    fallback: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.geometry import _embedding_cells

    element = lagrange_element("triangle", 2)
    nodes = lagrange_reference_node_data(element)
    vertices = reference_cell_topology("triangle").vertices

    def image(cell: int, point: tuple[Fraction, ...], /) -> tuple[Fraction, Fraction]:
        u, v = point
        if cell == 0:
            return u, v - u * (1 - u)
        if cell == 2:
            return u + 3, v - u * (1 - u)
        return Fraction(3, 8) + u / 4, -Fraction(1, 8) - v / 8 - u * (1 - u) / 8

    coordinates = np.empty((3 * element.local_dof_count, 2), dtype=np.float64)
    for cell in range(3):
        for node in nodes:
            if node.exact_point is None:
                raise AssertionError("Quadratic triangle nodes must be exact.")
            point = tuple(
                rational_coefficient(value, role="triangle node")
                for value in node.exact_point
            )
            coordinates[cell * element.local_dof_count + node.local_dof] = tuple(
                float(value) for value in image(cell, point)
            )
    points = np.asarray(
        [
            tuple(
                float(value)
                for value in image(cell, tuple(Fraction(float(x)) for x in point))
            )
            for cell in range(3)
            for point in vertices
        ],
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (CellBlock("cells", "triangle", np.arange(9, dtype=np.int32).reshape(3, 3)),),
    )
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(len(coordinates), dtype=np.int32).reshape(3, -1)},
        coordinates,
    )
    if fallback:
        monkeypatch.setattr(
            _embedding_cells.PreparedIntervalBasis, "bounds", lambda self, local: None
        )
    cells = prepare_embedding_cells(mesh, geometry)
    first_point = (Fraction(1, 2), Fraction(3, 32))
    second_point = (Fraction(1, 2), Fraction(0))
    contact = image(0, first_point)
    assert contact == image(1, second_point)
    for cell, point in zip(cells[:2], (first_point, second_point), strict=True):
        assert (
            tuple(evaluate_expression(expr, point) for expr in cell.coordinates)
            == contact
        )
        assert all(
            Fraction(lower) <= value <= Fraction(upper)
            for lower, value, upper in zip(
                cell.image_lower, contact, cell.image_upper, strict=True
            )
        )
    distant = image(2, first_point)
    assert all(
        Fraction(lower) <= value <= Fraction(upper)
        for lower, value, upper in zip(
            cells[2].image_lower, distant, cells[2].image_upper, strict=True
        )
    )
    tree = prepare_bvh(
        np.asarray([cell.image_lower for cell in cells], dtype=np.float64),
        np.asarray([cell.image_upper for cell in cells], dtype=np.float64),
    )
    left, right = bvh_overlap_pairs_host(tree, tree, include_touching=True)
    assert any({int(a), int(b)} == {0, 1} for a, b in zip(left, right, strict=True))


def test_restored_embedding_refuses_a_forged_stored_validity_proof() -> None:
    mesh, geometry, validity = _embedded_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ((0, 1, 2), (0, 2, 3)),
    )
    certificate = certify_global_embedding(mesh, geometry, validity)
    altered_claim = eqx.tree_at(lambda value: value.status, certificate, "violated")
    assert altered_claim.certificate_id == certificate.certificate_id
    with pytest.raises(ValueError, match="do not reproduce"):
        GlobalEmbeddingCertificate.from_stored_fields(altered_claim)
    lower = np.asarray(validity.determinant_lower) * 0.5
    # Simulate stored field injection, not a public certificate construction.
    forged = CellValidityCertificate._from_computed(
        validity.execution_status,
        lower,
        validity.determinant_upper,
        validity.depth,
        source_proof=_BlockCertificate(
            np.asarray(validity.source_status),
            lower,
            np.asarray(validity.source_determinant_upper),
            np.asarray(validity.source_depth),
        ),
        maximum_coefficient_rounding_error=float(
            validity.maximum_coefficient_rounding_error
        ),
        maximum_element_coefficient_rounding_error=float(
            validity.maximum_element_coefficient_rounding_error
        ),
        apex_derivative_unique=np.asarray(validity.apex_derivative_unique),
        geometry=geometry,
        mesh=mesh,
        policy=validity.policy,
        block_names=validity.block_names,
        block_offsets=validity.block_offsets,
        unsupported_block_names=validity.unsupported_block_names,
        geometry_id=validity.geometry_id,
        geometry_layout_id=validity.geometry_layout_id,
        policy_id=validity.policy_id,
    )
    # The forged proof is bound to the same operands, so only a renewed proof
    # can refuse it when the embedding is restored from stored fields.
    forged.require_current(geometry, mesh=mesh)
    transformed_validity = eqx.tree_at(
        lambda value: value.determinant_lower, forged, forged.determinant_lower.copy()
    )
    with pytest.raises(ValueError, match="does not reproduce the current mathematical"):
        certify_global_embedding(mesh, geometry, transformed_validity)
    stored = eqx.tree_at(lambda value: value.validity, certificate, forged)
    with pytest.raises(ValueError, match="does not reproduce the current mathematical"):
        GlobalEmbeddingCertificate.from_stored_fields(stored)


def test_computed_validity_host_proof_avoids_duplicate_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._cell_geometry_validity import (
        CellValidityPolicy,
        ComputedCellValidity,
    )

    mesh, geometry, _ = _embedded_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ((0, 1, 2), (0, 2, 3)),
    )
    calls = [0]
    original = CellValidityCertificate.revalidate

    def counted(
        value: CellValidityCertificate,
        bound_geometry: CellGeometrySpec,
        /,
        *,
        mesh: CellMesh | None = None,
    ) -> None:
        calls[0] += 1
        original(value, bound_geometry, mesh=mesh)

    monkeypatch.setattr(CellValidityCertificate, "revalidate", counted)
    computed = ComputedCellValidity(geometry, mesh, CellValidityPolicy(), None)
    result = certify_global_embedding(
        mesh, geometry, computed.certificate, validated_validity=computed
    )
    assert result.status == "certified"
    assert calls == [0]
    altered = eqx.tree_at(
        lambda value: value.determinant_lower,
        computed.certificate,
        computed.certificate.determinant_lower * 0.5,
    )
    with pytest.raises(ValueError, match="different exact operands"):
        certify_global_embedding(mesh, geometry, altered, validated_validity=computed)
    with pytest.raises(ValueError, match="does not reproduce"):
        certify_global_embedding(mesh, geometry, altered)


def test_planar_segments_sharing_one_vertex_certify_their_exact_corner() -> None:
    # Planar segment contacts are lifted to R3 with z = 0; the shared corner
    # must still match its planar vertex.
    mesh = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))),
        (CellBlock("segments", "interval", np.asarray(((0, 1), (1, 2)), np.int32)),),
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert "exact_cell_contacts" in certificate.evaluated_checks


@pytest.mark.parametrize(
    ("second", "expected"),
    (
        (((0, 0, 0), (1, 1, -1), (1, 1, 1)), False),
        (((0, 0, 0), (2, 0, 0), (1, 1, 0)), False),
        (((0, 0, 0), (1, 0, 0), (2, 0, 0)), False),
        (((0, 0, 0), (2, 0, 0), (0, 0, 1)), True),
        (((0, 0, 0), (1, 0, 1), (0, 1, 1)), True),
        (((3, 0, 0), (4, 0, 0), (3, 1, 0)), True),
        (((3, 0, 0), (4, 0, 0), (5, 0, 0)), False),
    ),
)
def test_spatial_triangle_enclosure_contact_decisions(
    second: tuple[Point3, ...], expected: bool
) -> None:
    first: tuple[Point3, ...] = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(2), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(2), Fraction(0)),
    )
    enclosure = outward_spatial_triangle_enclosures((first, second))
    result = enclosure.common_hull_contacts(
        np.asarray([0], dtype=np.intp), np.asarray([1], dtype=np.intp)
    )
    assert bool(result[0]) is expected
    reverse = enclosure.common_hull_contacts(
        np.asarray([1], dtype=np.intp), np.asarray([0], dtype=np.intp)
    )
    assert bool(reverse[0]) is expected


@pytest.mark.parametrize(
    "second",
    (
        ((2, 0, 2), (0, 0, 0), (0, -2, -2)),
        ((0, 0, 0), (-2, 1, -1), (-1, 2, 1)),
    ),
)
def test_spatial_projection_certifies_oblique_coplanar_neighbors(
    second: tuple[Point3, ...],
) -> None:
    first = tuple(
        (Fraction(point[0]), Fraction(point[1]), Fraction(point[2]))
        for point in ((0, 0, 0), (2, 0, 2), (0, 2, 2))
    )
    enclosure = outward_spatial_triangle_enclosures((first, second))
    left, right = np.asarray([0], dtype=np.intp), np.asarray([1], dtype=np.intp)
    assert enclosure.common_hull_contacts(left, right)[0]


def test_spatial_projection_refuses_overlap_degeneracy_and_false_identities() -> None:
    triangles = tuple(
        tuple((Fraction(p[0]), Fraction(p[1]), Fraction(p[2])) for p in triangle)
        for triangle in (
            ((0, 0, 0), (2, 0, 0), (0, 2, 0)),
            ((0, 0, 0), (2, 0, 0), (1, 1, 0)),
            ((0, 0, 1), (2, 0, 1), (0, 2, 1)),
            ((0, 0, 1), (-1, 0, 1), (0, -1, 1)),
        )
    )
    enclosure = outward_spatial_triangle_enclosures(triangles)
    left = np.asarray([0], dtype=np.intp)
    fold = np.asarray([1], dtype=np.intp)
    overlap = np.asarray([2], dtype=np.intp)
    projected_corner = np.asarray([3], dtype=np.intp)
    assert all(
        not bank.common_hull_contacts(left, fold)[0] for bank in enclosure.projections
    )
    assert not enclosure.projections[2].common_hull_contacts(left, overlap)[0]
    assert not enclosure.projections[0].common_hull_contacts(left, overlap)[0]
    assert not enclosure.projections[2].common_hull_contacts(left, projected_corner)[0]
    assert enclosure.common_hull_contacts(left, overlap)[0]


def test_spatial_filter_refuses_duplicate_points_with_distinct_vertex_ids() -> None:
    mesh = CellMesh(
        np.asarray(
            ((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 0), (0, 0, 1), (2, 0, 0)),
            dtype=np.float64,
        ),
        (
            CellBlock(
                "triangles",
                "triangle",
                np.asarray(((0, 1, 2), (3, 5, 4)), dtype=np.int32),
            ),
        ),
    )
    geometry = CellGeometrySpec.affine(mesh)
    cells = prepare_embedding_cells(mesh, geometry)
    screen = _SpatialTriangleFilter.prepare(mesh, cells)
    assert screen is not None
    assert not screen.certify(
        np.asarray([0], dtype=np.intp), np.asarray([1], dtype=np.intp)
    )[0]


def test_bent_spatial_triangles_meet_only_on_their_shared_edge() -> None:
    mesh, geometry, validity = _embedded_triangles(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        ((0, 1, 2), (0, 3, 1)),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert certificate.status == "certified", certificate.findings


def test_spatial_triangle_piercing_a_vertex_neighbor_is_violated() -> None:
    # The second triangle shares only the origin, but its other corners lie on
    # both sides of the first plane, and the crossing segment enters the first.
    mesh, geometry, validity = _embedded_triangles(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
                (1.0, 1.0, -1.0),
                (1.0, 1.0, 1.0),
            )
        ),
        ((0, 1, 2), (0, 3, 4)),
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert certificate.status == "violated"
    assert any(finding.check == "cell_contact" for finding in certificate.findings)


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


@pytest.mark.parametrize(
    "points, cells",
    (
        (
            ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (1.0, 0.0), (0.0, -1.0), (2.0, -1.0)),
            ((0, 1, 2), (3, 4, 5)),
        ),
        (
            ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (1.0, 0.0), (1.0, -1.0)),
            ((0, 1, 2), (0, 4, 3)),
        ),
        (
            ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (1.0, -1.0), (2.0, -1.0)),
            ((0, 1, 2), (2, 3, 4)),
        ),
        (
            ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
            ((0, 1, 2), (3, 4, 5)),
        ),
    ),
    ids=(
        "vertex-on-edge-without-shared-id",
        "collinear-edge-beyond-shared-vertex",
        "shared-vertex-crossing",
        "coincident-edge-without-shared-ids",
    ),
)
def test_planar_contacts_outside_the_shared_stratum_stay_violated(
    points: tuple[tuple[float, float], ...], cells: tuple[tuple[int, int, int], ...]
) -> None:
    # Binary64 pair clearing must never hide an undeclared touching contact.
    mesh, geometry, validity = _embedded_triangles(np.asarray(points), cells)

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(finding.check == "cell_contact" for finding in certificate.findings)


_PLANAR_BASE = ((0, 0), (2, 0), (0, 2))


type _ExactPoint2 = tuple[Fraction, Fraction]


def _exact_triangle(
    points: tuple[tuple[int, int], ...],
) -> tuple[_ExactPoint2, _ExactPoint2, _ExactPoint2]:
    first, second, third = ((Fraction(x), Fraction(y)) for x, y in points)
    return first, second, third


@pytest.mark.parametrize(
    "other, certified",
    (
        (((2, 0), (0, 2), (2, 2)), True),
        (((0, 0), (-2, 1), (-1, -2)), True),
        (((3, 3), (4, 3), (3, 4)), True),
        (((1, 0), (2, -1), (0, -1)), False),
        (((0, 0), (1, 0), (1, -1)), False),
        (((0, 2), (1, -1), (2, -1)), False),
        (((1, 1), (3, 1), (1, 3)), False),
    ),
    ids=(
        "shared-edge",
        "shared-vertex",
        "disjoint",
        "vertex-on-edge",
        "collinear-beyond-vertex",
        "shared-apex-crossing",
        "overlap",
    ),
)
def test_planar_common_hull_screen_certifies_only_common_hull_contacts(
    other: tuple[tuple[int, int], ...], certified: bool
) -> None:
    first, second = _exact_triangle(_PLANAR_BASE), _exact_triangle(other)
    screen = TriangleEnclosures.prepare((first, second))
    pair = (np.asarray([0], dtype=np.intp), np.asarray([1], dtype=np.intp))
    assert screen.common_hull_contacts(*pair).tolist() == [certified]
    assert screen.common_hull_contacts(pair[1], pair[0]).tolist() == [certified]
    common = tuple(point for point in first if point in second)
    contacts = triangle_contact_points2(first, second)
    if certified:
        assert all(point in common for point in contacts) or (
            len(common) == 2
            and all(point_on_segment2(point, *common) for point in contacts)
        )


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
    # The planar exterior completes the degree theorem: no cell-pair contact
    # proof is stated for this single curved cell.
    assert any(
        witness.theorem == "global_degree_one_injectivity"
        for witness in certificate.witnesses
    )
    assert "exact_cell_contacts" not in certificate.evaluated_checks


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


def _unit_hexahedra(
    cells: tuple[tuple[int, int, int], ...],
) -> tuple[CellMesh, CellGeometrySpec, CellValidityCertificate]:
    """Unit cubes at integer lattice offsets, sharing their lattice vertices."""
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.int64)
    rows: dict[tuple[int, int, int], int] = {}
    connectivity = []
    for offset in cells:
        connectivity.append(
            [
                rows.setdefault(
                    (
                        offset[0] + int(corner[0]),
                        offset[1] + int(corner[1]),
                        offset[2] + int(corner[2]),
                    ),
                    len(rows),
                )
                for corner in corners
            ]
        )
    mesh = CellMesh(
        np.asarray(tuple(rows), dtype=np.float64),
        (CellBlock("cubes", "hexahedron", np.asarray(connectivity, dtype=np.int32)),),
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    return mesh, geometry, validity


def test_global_degree_certifies_a_solid_ring_without_cell_pair_proofs() -> None:
    # Eight cubes around an open square hole: one closed shell of genus one.
    mesh, geometry, validity = _unit_hexahedra(
        tuple((x, y, 0) for x in range(3) for y in range(3) if (x, y) != (1, 1))
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert certificate.shell_count == 1
    assert any(
        witness.theorem == "global_degree_one_injectivity"
        for witness in certificate.witnesses
    )
    assert "exact_cell_contacts" not in certificate.evaluated_checks


def test_global_degree_leaves_a_cavity_to_complete_cell_pair_proofs() -> None:
    # A closed cavity gives two shells: one shell's winding argument does not apply.
    mesh, geometry, validity = _unit_hexahedra(
        tuple(
            (x, y, z)
            for x in range(3)
            for y in range(3)
            for z in range(3)
            if (x, y, z) != (1, 1, 1)
        )
    )

    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert certificate.shell_count == 2
    assert "exact_cell_contacts" in certificate.evaluated_checks
    assert not any(
        witness.theorem == "global_degree_one_injectivity"
        for witness in certificate.witnesses
    )


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


def test_integer_markowitz_nullspace_matches_the_exact_sparse_factorization() -> None:
    generator = np.random.default_rng(11)
    checked = {"nullity_one": 0, "other": 0}
    for _ in range(60):
        columns = int(generator.integers(3, 9))
        rank = int(generator.integers(1, columns + 1))
        basis = generator.integers(-4, 5, (rank, columns))
        mixing = generator.integers(-3, 4, (int(generator.integers(rank, 12)), rank))
        dense = mixing @ basis
        # Sparse dyadic and non-dyadic rationals, with exact zeros kept out.
        rows = tuple(
            {
                column: Fraction(int(value), int(generator.choice((1, 2, 3, 8))))
                for column, value in enumerate(row)
                if value
            }
            for row in dense
        )
        factorization = factor_exact_sparse(rows, columns)
        replayed = _rational_markowitz_nullspace(rows, columns)
        if factorization.evidence.rank == columns - 1:
            assert replayed == factorization.nullspace_basis()[0]
            checked["nullity_one"] += 1
        else:
            assert replayed is False
            checked["other"] += 1
        if _full_column_rank_modulo_prime(rows, columns):
            assert factorization.evidence.rank == columns
    assert checked["nullity_one"] and checked["other"]


@dataclass
class _ExactImageSource:
    bank: tuple[tuple[Fraction, ...], ...]
    source_id: str = "interval-image-source"
    source_revision: str = "interval-image-revision"
    source_map_id: str = "interval-image-map"

    def source_coordinates(self) -> tuple[tuple[Fraction, ...], ...]:
        return self.bank


def _curved_separated_image_boxes(
    second_shift: Fraction,
) -> tuple[CellMesh, CellGeometrySpec, tuple[EmbeddingCell, ...]]:
    element = lagrange_element("triangle", 2)
    reference = reference_cell_topology("triangle").vertices
    nodes = lagrange_reference_node_data(element)

    def image(shift: Fraction, point: tuple[Fraction, ...], /) -> tuple[Fraction, ...]:
        u, v = point
        return u + shift - v * (1 - v) / 3, v + u * v / 8

    shifts = (Fraction(0), second_shift)
    points = np.asarray(
        [
            tuple(
                float(value)
                for value in image(shift, tuple(Fraction(float(x)) for x in point))
            )
            for shift in shifts
            for point in reference
        ],
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (CellBlock("cells", "triangle", np.arange(6, dtype=np.int32).reshape(2, 3)),),
    )
    bank_rows: list[tuple[Fraction, ...]] = []
    for shift in shifts:
        for node in nodes:
            if node.exact_point is None:
                raise AssertionError("Quadratic triangle nodes must be exact.")
            point = tuple(
                rational_coefficient(value, role="triangle node")
                for value in node.exact_point
            )
            bank_rows.append(image(shift, point))
    coordinates = np.asarray(bank_rows, dtype=np.float64)
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(len(coordinates), dtype=np.int32).reshape(2, -1)},
        coordinates,
        source_map=_ExactImageSource(tuple(bank_rows)),
        source_mesh=mesh,
    )
    return mesh, geometry, prepare_embedding_cells(mesh, geometry)


@pytest.mark.parametrize("periodic", (False, True))
def test_widened_image_boxes_retry_exact_separation(periodic: bool) -> None:
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.discretization._periodic_topology import PeriodicMeshTopology
    from phydrax.geometry._periodic_mapped_contacts import (
        certify_periodic_mapped_contacts,
    )

    gap = Fraction(1, 2**52)
    translation = Fraction(4) if periodic else Fraction(0)
    shift = Fraction(7, 6) + gap - translation
    mesh, geometry, cells = _curved_separated_image_boxes(shift)
    first, second = cells
    shifted_expression = add(second.coordinates[0], constant(translation, 2))
    first_exact = polynomial_bounds(first.coordinates[0], "simplex", 2)
    second_exact = polynomial_bounds(shifted_expression, "simplex", 2)
    assert first_exact[1] < second_exact[0]
    assert Fraction(second.image_lower[0]) + translation <= Fraction(first.image_upper[0])
    assert first.image_inner_bounds is not None
    assert second.image_inner_bounds is not None
    assert Fraction(first.image_inner_bounds[0][1]) < (
        Fraction(second.image_inner_bounds[0][0]) + translation
    )

    if periodic:
        topology = PeriodicMeshTopology(
            mesh,
            PeriodicCell(np.asarray(((4.0, 0.0),), dtype=np.float64)),
            np.arange(6, dtype=np.int32),
            np.zeros((6, 1), dtype=np.int32),
        )
        result = certify_periodic_mapped_contacts(mesh, cells, topology)
        assert result.status == "certified", result.findings
    else:
        validity = certify_cell_geometry_validity(geometry, mesh=mesh)
        certificate = certify_global_embedding(mesh, geometry, validity)
        assert validity.all_certified
        assert certificate.status == "certified", certificate.findings
