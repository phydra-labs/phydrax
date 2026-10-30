from __future__ import annotations

from fractions import Fraction
from itertools import permutations, product

import numpy as np

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionEvidence,
)
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._nested_reference_types import NestedReferenceWitnesses
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._periodic_restriction import PeriodicRestrictionPremise


def _periodic_cube_tetrahedra() -> tuple[CellMesh, CellGeometrySpec]:
    points = tuple(product((0, 1, 2), repeat=3))
    coordinates = np.asarray(
        tuple(tuple(value * 0.5 for value in point) for point in points),
        dtype=np.float64,
    )
    vertex_index = {point: index for index, point in enumerate(points)}
    rows: list[tuple[int, int, int, int]] = []
    for cube in product((0, 1), repeat=3):
        for permutation in permutations(range(3)):
            path = [cube]
            first = list(cube)
            first[permutation[0]] += 1
            path.append(tuple(first))
            second = first.copy()
            second[permutation[1]] += 1
            path.append(tuple(second))
            path.append(tuple(value + 1 for value in cube))
            row = [vertex_index[point] for point in path]
            inversions = sum(
                permutation[left] > permutation[right]
                for left in range(3)
                for right in range(left + 1, 3)
            )
            if inversions % 2:
                row[1], row[2] = row[2], row[1]
            rows.append((row[0], row[1], row[2], row[3]))
    block = CellBlock("tetrahedra", "tetrahedron", np.asarray(rows, dtype=np.int32))
    lifted = CellMesh(coordinates, (block,))
    representatives = np.asarray(
        [vertex_index[tuple(value % 2 for value in point)] for point in points],
        dtype=np.int32,
    )
    shifts = np.asarray(
        [tuple(value // 2 for value in point) for point in points],
        dtype=np.int32,
    )
    topology = PeriodicMeshTopology(
        lifted, PeriodicCell(np.eye(3, dtype=np.float64)), representatives, shifts
    )
    mesh = CellMesh(coordinates, (block,), periodic_topology=topology)
    return mesh, CellGeometrySpec.affine(mesh)


def _refined_tetrahedra(
    source: CellMesh,
) -> tuple[CellMesh, CellGeometrySpec, NestedReferenceWitnesses]:
    source_topology = source.periodic_topology
    if source_topology is None:
        raise TypeError("The coarse periodic tetrahedral fixture lost its quotient.")
    source_rows = tuple(
        tuple(int(vertex) for vertex in row)
        for row in np.asarray(source.blocks[0].vertices, dtype=np.int64).tolist()
    )
    source_ids = np.asarray(source.blocks[0].global_ids, dtype=np.int64)
    source_coordinates = np.asarray(source.coordinates, dtype=np.float64)
    fine_coordinates = [
        tuple(float(value) for value in row) for row in source_coordinates
    ]
    fine_roots = np.asarray(
        source_topology.vertex_representatives, dtype=np.int64
    ).tolist()
    fine_shifts = np.asarray(source_topology.vertex_shifts, dtype=np.int64).tolist()
    rows: list[tuple[int, int, int, int]] = []
    fine_cell_ids: list[int] = []
    coarse_cell_ids: list[int] = []
    reference_vertices: list[tuple[tuple[Fraction, Fraction, Fraction], ...]] = []
    reference_corners = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    centroid_reference = (
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 4),
    )
    for parent_row, parent in enumerate(source_rows):
        center = tuple(
            sum(source_coordinates[int(parent[index]), axis] for index in range(4)) / 4.0
            for axis in range(3)
        )
        center_index = len(fine_coordinates)
        fine_coordinates.append(center)
        fine_roots.append(center_index)
        fine_shifts.append((0, 0, 0))
        for face_index in range(4):
            face_vertices = tuple(index for index in range(4) if index != face_index)
            local_vertices = (
                center_index,
                *(int(parent[index]) for index in face_vertices),
            )
            child_points = (
                center,
                *(
                    tuple(source_coordinates[int(parent[index])])
                    for index in face_vertices
                ),
            )
            child_reference = (
                centroid_reference,
                *(reference_corners[index] for index in face_vertices),
            )
            edge_rows = tuple(
                tuple(
                    Fraction(float(child_points[row][axis]))
                    - Fraction(float(child_points[0][axis]))
                    for axis in range(3)
                )
                for row in range(1, 4)
            )
            determinant = (
                edge_rows[0][0]
                * (edge_rows[1][1] * edge_rows[2][2] - edge_rows[1][2] * edge_rows[2][1])
                - edge_rows[0][1]
                * (edge_rows[1][0] * edge_rows[2][2] - edge_rows[1][2] * edge_rows[2][0])
                + edge_rows[0][2]
                * (edge_rows[1][0] * edge_rows[2][1] - edge_rows[1][1] * edge_rows[2][0])
            )
            if determinant < 0:
                local_vertices = (
                    local_vertices[0],
                    local_vertices[2],
                    local_vertices[1],
                    local_vertices[3],
                )
                child_reference = (
                    child_reference[0],
                    child_reference[2],
                    child_reference[1],
                    child_reference[3],
                )
            rows.append(
                (
                    local_vertices[0],
                    local_vertices[1],
                    local_vertices[2],
                    local_vertices[3],
                )
            )
            fine_cell_ids.append(1000 + len(rows) - 1)
            coarse_cell_ids.append(int(source_ids[parent_row]))
            reference_vertices.append(child_reference)
    fine_points = np.asarray(fine_coordinates, dtype=np.float64)
    fine_block = CellBlock(
        "tetrahedra",
        "tetrahedron",
        np.asarray(rows, dtype=np.int32),
        global_ids=np.asarray(fine_cell_ids, dtype=np.int64),
    )
    lifted = CellMesh(
        fine_points,
        (fine_block,),
        vertex_global_ids=np.arange(len(fine_points), dtype=np.int64),
    )
    topology = PeriodicMeshTopology(
        lifted,
        source_topology.cell,
        np.asarray(fine_roots, dtype=np.int32),
        np.asarray(fine_shifts, dtype=np.int32),
    )
    target = CellMesh(
        fine_points,
        (fine_block,),
        vertex_global_ids=np.arange(len(fine_points), dtype=np.int64),
        periodic_topology=topology,
    )
    witness = NestedReferenceWitnesses(
        np.asarray(fine_cell_ids, dtype=np.int64),
        np.asarray(coarse_cell_ids, dtype=np.int64),
        np.asarray(
            tuple(
                tuple(tuple(float(value) for value in point) for point in cell)
                for cell in reference_vertices
            ),
            dtype=np.float64,
        ),
        fine_reference_exact_vertices=tuple(reference_vertices),
    )
    return target, CellGeometrySpec.affine(target), witness


def _tetrahedron_volume(points: np.ndarray, /) -> Fraction:
    exact = tuple(tuple(Fraction(float(value)) for value in point) for point in points)
    first = tuple(exact[1][axis] - exact[0][axis] for axis in range(3))
    second = tuple(exact[2][axis] - exact[0][axis] for axis in range(3))
    third = tuple(exact[3][axis] - exact[0][axis] for axis in range(3))
    determinant = (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )
    return abs(determinant) / 6


def test_refined_periodic_tetrahedra_prove_exact_parent_volume_partition() -> None:
    source, source_geometry = _periodic_cube_tetrahedra()
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source)
    source_embedding = certify_global_embedding(source, source_geometry, source_validity)
    target, target_geometry, nested = _refined_tetrahedra(source)
    target_validity = certify_cell_geometry_validity(target_geometry, mesh=target)
    from phydrax.geometry._embedding_cells import prepare_embedding_cells
    from phydrax.geometry._global_embedding import (
        _periodic_facet_trace_certificate_id,
    )

    target_topology = target.periodic_topology
    assert target_topology is not None
    target_trace_id = _periodic_facet_trace_certificate_id(
        target,
        target_geometry,
        prepare_embedding_cells(target, target_geometry),
        target_topology,
    )
    assert target_trace_id is not None
    transition = CellGeometryTransition(
        source,
        source_geometry,
        target,
        target_geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction", refinement_map_count=len(nested.fine_cell_ids)
        ),
        refinement=nested,
    )
    premise = PeriodicRestrictionPremise(
        source,
        source_geometry,
        source_validity,
        source_embedding,
        transition,
    )
    target_embedding = certify_global_embedding(
        target,
        target_geometry,
        target_validity,
        periodic_restriction=premise,
    )
    evidence = target_embedding.periodic_restriction

    source_ids = np.asarray(source.blocks[0].global_ids, dtype=np.int64)
    target_ids = np.asarray(target.blocks[0].global_ids, dtype=np.int64)
    source_vertex_rows = tuple(
        tuple(int(vertex) for vertex in row)
        for row in np.asarray(source.blocks[0].vertices, dtype=np.int64).tolist()
    )
    target_vertex_rows = tuple(
        tuple(int(vertex) for vertex in row)
        for row in np.asarray(target.blocks[0].vertices, dtype=np.int64).tolist()
    )
    independent_parent_volumes = {
        int(identifier): _tetrahedron_volume(
            np.asarray(source.coordinates)[list(source_vertex_rows[row])]
        )
        for row, identifier in enumerate(source_ids)
    }
    independent_child_volumes = {
        int(identifier): _tetrahedron_volume(
            np.asarray(target.coordinates)[list(target_vertex_rows[row])]
        )
        for row, identifier in enumerate(target_ids)
    }
    target_parent_by_child = dict(
        zip(nested.fine_cell_ids.tolist(), nested.coarse_cell_ids.tolist(), strict=True)
    )
    for parent, volume in independent_parent_volumes.items():
        assert (
            sum(
                (
                    independent_child_volumes[child]
                    for child, owner in target_parent_by_child.items()
                    if owner == parent
                ),
                Fraction(0),
            )
            == volume
        )

    assert source_validity.all_certified
    assert source_embedding.status == "certified", source_embedding.findings
    assert target_validity.all_certified
    assert target_embedding.status == "certified", target_embedding.findings
    assert evidence is not None
    assert evidence.direction == "refine"
    assert set(evidence.child_cell_ids) == set(target_ids.tolist())
    assert set(evidence.partition_cell_ids) == set(source_ids.tolist())
    assert sum(
        (
            Fraction(numerator, denominator)
            for numerator, denominator in evidence.child_volumes
        ),
        Fraction(0),
    ) == Fraction(1)
    assert sum(
        (
            Fraction(numerator, denominator)
            for numerator, denominator in evidence.partition_volumes
        ),
        Fraction(0),
    ) == Fraction(1)
    assert {
        "periodic_restriction_source_embedding",
        "periodic_restriction_parent_partition",
        "periodic_restriction_boundary_subdivision",
    } <= set(target_embedding.evaluated_checks)


def test_periodic_mapped_refinement_proves_a_common_subcomplex() -> None:
    from phydrax.geometry._domain_coverage import (
        certify_domain_coverage,
        PiecewiseLinearDomain,
    )
    from phydrax.geometry._embedding_cells import prepare_embedding_cells
    from phydrax.geometry._global_embedding import _periodic_facet_trace_certificate_id
    from phydrax.geometry._periodic_mapped_restriction import (
        _reference_partition,
        _transition_pieces,
        certify_periodic_mapped_restriction,
    )

    lattice = PeriodicCell(
        4.0 * np.eye(3, dtype=np.float64),
        periodic_axes=(True, False, False),
    )
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    source_rows = np.asarray(((0, 1, 2, 3),), dtype=np.int32)

    def make_mesh(
        coordinates: np.ndarray,
        rows: np.ndarray,
        identifiers: np.ndarray,
        roots: np.ndarray,
        shifts: np.ndarray,
    ) -> CellMesh:
        block = CellBlock(
            "tetrahedra",
            "tetrahedron",
            rows,
            global_ids=identifiers,
        )
        vertex_ids = np.arange(len(coordinates), dtype=np.int64)
        lifted = CellMesh(
            coordinates,
            (block,),
            vertex_global_ids=vertex_ids,
        )
        topology = PeriodicMeshTopology(lifted, lattice, roots, shifts)
        return CellMesh(
            coordinates,
            (block,),
            vertex_global_ids=vertex_ids,
            periodic_topology=topology,
        )

    source = make_mesh(
        points,
        source_rows,
        np.asarray((10,), dtype=np.int64),
        np.arange(4, dtype=np.int32),
        np.zeros((4, 3), dtype=np.int32),
    )
    source_geometry = CellGeometrySpec.affine(source)
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source)
    source_embedding = certify_global_embedding(source, source_geometry, source_validity)

    fine_points = np.concatenate(
        (points, np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64))
    )
    fine_rows = np.asarray(
        ((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)),
        dtype=np.int32,
    )
    target = make_mesh(
        fine_points,
        fine_rows,
        np.arange(20, 24, dtype=np.int64),
        np.arange(5, dtype=np.int32),
        np.zeros((5, 3), dtype=np.int32),
    )
    target_topology = target.periodic_topology
    if target_topology is None:
        raise TypeError("The mapped target lost its periodic quotient.")
    target_geometry = CellGeometrySpec.affine(target)
    target_validity = certify_cell_geometry_validity(target_geometry, mesh=target)
    nested = NestedReferenceWitnesses(
        np.asarray((20, 21, 22, 23), dtype=np.int64),
        np.asarray((10, 10, 10, 10), dtype=np.int64),
        fine_points[fine_rows],
    )
    transition = CellGeometryTransition(
        source,
        source_geometry,
        target,
        target_geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction", refinement_map_count=4
        ),
        refinement=nested,
    )
    domain = PiecewiseLinearDomain(
        points.copy(),
        np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))),
        np.asarray(((0, -1),) * 4),
        ("fluid",),
        source_id="periodic-mapped-refinement",
    )
    coverage = certify_domain_coverage(
        source,
        source_geometry,
        domain,
        np.asarray((0,), dtype=np.int64),
        embedding=source_embedding,
    )
    assert coverage.status == "certified", coverage.findings
    relation = _transition_pieces(transition)
    assert relation is not None
    direction, pieces = relation
    assert _reference_partition(source, target, pieces, direction) is not None
    facet_trace_id = _periodic_facet_trace_certificate_id(
        target,
        target_geometry,
        prepare_embedding_cells(target, target_geometry),
        target_topology,
    )
    assert facet_trace_id is not None
    evidence = certify_periodic_mapped_restriction(
        transition,
        source_validity,
        source_embedding,
        target_validity,
        (0,),
        (0, 0, 0, 0),
        domain,
        domain.domain_id,
        facet_trace_id,
    )

    independent_parent = _tetrahedron_volume(points)
    independent_children = sum(
        (
            _tetrahedron_volume(fine_points[list(row)])
            for row in np.asarray(target.blocks[0].vertices, dtype=np.int64).tolist()
        ),
        Fraction(0),
    )

    assert source_validity.all_certified
    assert source_embedding.status == "certified", source_embedding.findings
    assert target_validity.all_certified
    assert evidence is not None
    assert evidence.direction == "refine"
    assert evidence.source_regions == (0,)
    assert evidence.target_regions == (0, 0, 0, 0)
    assert independent_parent == independent_children == Fraction(1, 6)


def _p2_local_entity_keys(
    element: FiniteElementSpec, mesh: CellMesh, cell_row: int
) -> tuple[tuple[int, ...], ...]:
    from phydrax.discretization._reference_cell import reference_cell_topology

    block = mesh.blocks[0]
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_vertices = tuple(
        int(vertex)
        for vertex in np.asarray(block.vertices, dtype=np.int64).tolist()[cell_row]
    )
    entities: list[tuple[int, ...] | None] = [None] * element.local_dof_count
    for local_vertex, local_dofs in enumerate(element.entity_dofs[0]):
        key = (int(vertex_ids[cell_vertices[local_vertex]]),)
        for local_dof in local_dofs:
            entities[int(local_dof)] = key
    topology = reference_cell_topology("tetrahedron")
    for edge_index, local_dofs in enumerate(element.entity_dofs[1]):
        local_edge = topology.entities[1][edge_index]
        key = tuple(
            sorted(
                int(vertex_ids[cell_vertices[local_vertex]])
                for local_vertex in local_edge
            )
        )
        for local_dof in local_dofs:
            entities[int(local_dof)] = key
    if any(entity is None for entity in entities):
        raise AssertionError("P2 tetrahedron DOFs must belong to vertices or edges.")
    return tuple(entity for entity in entities if entity is not None)


def _p2_geometry_from_cell_values(
    mesh: CellMesh,
    element: FiniteElementSpec,
    cell_values: tuple[tuple[tuple[Fraction, ...], ...], ...],
) -> CellGeometrySpec:
    block = mesh.blocks[0]
    routes = np.empty((len(block.vertices), element.local_dof_count), dtype=np.int32)
    coordinate_rows: list[tuple[Fraction, ...]] = []
    coordinate_indices: dict[tuple[int, ...], int] = {}
    for cell_row, dof_values in enumerate(cell_values):
        entities = _p2_local_entity_keys(element, mesh, cell_row)
        for local_dof, (entity, value) in enumerate(
            zip(entities, dof_values, strict=True)
        ):
            coordinate_index = coordinate_indices.get(entity)
            if coordinate_index is None:
                coordinate_index = len(coordinate_rows)
                coordinate_indices[entity] = coordinate_index
                coordinate_rows.append(value)
            elif coordinate_rows[coordinate_index] != value:
                raise AssertionError("Shared P2 entity maps must agree exactly.")
            routes[cell_row, local_dof] = coordinate_index
    return CellGeometrySpec(
        {block.name: element},
        {block.name: routes},
        np.asarray(
            tuple(tuple(float(value) for value in point) for point in coordinate_rows),
            dtype=np.float64,
        ),
    )


def test_curved_p2_periodic_refinement_preserves_degree() -> None:
    from phydrax.discretization._coordinate_enclosure import evaluate_expression
    from phydrax.discretization._exact_algebraic_geometry import rational_coefficient
    from phydrax.discretization._nested_reference_types import exact_reference_vertices
    from phydrax.discretization.fem._reference import (
        lagrange_element,
        lagrange_reference_node_data,
    )
    from phydrax.geometry._embedding_cells import prepare_embedding_cells
    from phydrax.geometry._global_embedding import (
        _periodic_facet_trace_certificate_id,
        GlobalEmbeddingCertificate,
    )

    source, _ = _periodic_cube_tetrahedra()
    element = lagrange_element("tetrahedron", 2)
    nodes = lagrange_reference_node_data(element)
    source_rows = tuple(
        tuple(int(vertex) for vertex in row)
        for row in np.asarray(source.blocks[0].vertices, dtype=np.int64).tolist()
    )
    source_coordinates = np.asarray(source.coordinates)
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    vertex_at = {
        tuple(float(value) for value in point): index
        for index, point in enumerate(source_coordinates)
    }
    curved_edge = tuple(
        sorted(
            (
                int(source_vertex_ids[vertex_at[(0.5, 0.5, 0.0)]]),
                int(source_vertex_ids[vertex_at[(0.5, 0.5, 0.5)]]),
            )
        )
    )
    epsilon = Fraction(1, 4096)
    source_cell_values: list[tuple[tuple[Fraction, ...], ...]] = []
    for cell_row, cell_vertices in enumerate(source_rows):
        cell_points = tuple(
            tuple(Fraction(float(value)) for value in source_coordinates[int(vertex)])
            for vertex in cell_vertices
        )
        entities = _p2_local_entity_keys(
            element,
            source,
            cell_row,
        )
        local_values: list[tuple[Fraction, ...] | None] = [None] * element.local_dof_count
        for node in nodes:
            reference_point_source = node.exact_point
            if reference_point_source is None:
                raise AssertionError("P2 tetrahedron reference nodes must be exact.")
            reference_point = tuple(
                rational_coefficient(value, role="P2 tetrahedron reference node")
                for value in reference_point_source
            )
            physical_point = tuple(
                cell_points[0][axis]
                + sum(
                    (
                        reference_point[local_vertex]
                        * (cell_points[local_vertex + 1][axis] - cell_points[0][axis])
                        for local_vertex in range(3)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            if entities[node.local_dof] == curved_edge:
                physical_point = (
                    physical_point[0],
                    physical_point[1] + epsilon,
                    physical_point[2],
                )
            local_values[node.local_dof] = physical_point
        if any(value is None for value in local_values):
            raise AssertionError("Every P2 local DOF needs a physical control point.")
        source_cell_values.append(
            tuple(value for value in local_values if value is not None)
        )
    source_geometry = _p2_geometry_from_cell_values(
        source, element, tuple(source_cell_values)
    )
    expected_curve = (0.5, 0.5 + float(epsilon), 0.25)
    assert any(
        tuple(float(value) for value in point) == expected_curve
        for point in np.asarray(source_geometry.coordinates)
    )
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source)
    source_cells = prepare_embedding_cells(source, source_geometry)
    source_topology = source.periodic_topology
    assert source_topology is not None
    source_trace_id = _periodic_facet_trace_certificate_id(
        source, source_geometry, source_cells, source_topology
    )
    assert source_trace_id is not None
    source_embedding = certify_global_embedding(source, source_geometry, source_validity)
    source_cells_by_id = {cell.global_id: cell for cell in source_cells}

    target_template, _, nested = _refined_tetrahedra(source)
    exact_reference_bank = exact_reference_vertices(nested)
    target_topology = target_template.periodic_topology
    assert target_topology is not None
    target_coordinates = np.asarray(target_template.coordinates).copy()
    target_rows = tuple(
        tuple(int(vertex) for vertex in row)
        for row in np.asarray(target_template.blocks[0].vertices, dtype=np.int64).tolist()
    )
    moved_centers: dict[int, tuple[Fraction, ...]] = {}
    for cell_row, parent_identifier in enumerate(nested.coarse_cell_ids):
        center_vertex = int(target_rows[cell_row][0])
        parent_cell = source_cells_by_id[int(parent_identifier)]
        center_reference = exact_reference_bank[cell_row][0]
        center_image = tuple(
            rational_coefficient(
                evaluate_expression(component, center_reference),
                role="P2 refinement center image",
            )
            for component in parent_cell.coordinates
        )
        prior_image = moved_centers.get(center_vertex)
        if prior_image is not None and prior_image != center_image:
            raise AssertionError("Refinement child centers must map consistently.")
        moved_centers[center_vertex] = center_image
    for vertex, point in moved_centers.items():
        target_coordinates[vertex] = tuple(float(value) for value in point)
    lifted_target = CellMesh(
        target_coordinates,
        target_template.blocks,
        vertex_global_ids=target_template.vertex_global_ids,
    )
    target_periodic_topology = PeriodicMeshTopology(
        lifted_target,
        target_topology.cell,
        target_topology.vertex_representatives,
        target_topology.vertex_shifts,
    )
    target = CellMesh(
        target_coordinates,
        target_template.blocks,
        vertex_global_ids=target_template.vertex_global_ids,
        periodic_topology=target_periodic_topology,
    )
    target_cell_values: list[tuple[tuple[Fraction, ...], ...]] = []
    for cell_row, parent_identifier in enumerate(nested.coarse_cell_ids):
        parent_cell = source_cells_by_id[int(parent_identifier)]
        parent_reference_vertices = exact_reference_bank[cell_row]
        local_values: list[tuple[Fraction, ...] | None] = [None] * element.local_dof_count
        for node in nodes:
            child_reference_source = node.exact_point
            if child_reference_source is None:
                raise AssertionError("P2 tetrahedron reference nodes must be exact.")
            child_reference = tuple(
                rational_coefficient(value, role="P2 child reference node")
                for value in child_reference_source
            )
            parent_reference = tuple(
                parent_reference_vertices[0][axis]
                + sum(
                    (
                        child_reference[local_vertex]
                        * (
                            parent_reference_vertices[local_vertex + 1][axis]
                            - parent_reference_vertices[0][axis]
                        )
                        for local_vertex in range(3)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            local_values[node.local_dof] = tuple(
                rational_coefficient(
                    evaluate_expression(component, parent_reference),
                    role="P2 child control image",
                )
                for component in parent_cell.coordinates
            )
        if any(value is None for value in local_values):
            raise AssertionError("Every child P2 DOF needs a mapped control point.")
        target_cell_values.append(
            tuple(value for value in local_values if value is not None)
        )
    target_geometry = _p2_geometry_from_cell_values(
        target, element, tuple(target_cell_values)
    )
    target_validity = certify_cell_geometry_validity(target_geometry, mesh=target)
    transition = CellGeometryTransition(
        source,
        source_geometry,
        target,
        target_geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction", refinement_map_count=len(nested.fine_cell_ids)
        ),
        refinement=nested,
    )
    target_embedding = certify_global_embedding(target, target_geometry, target_validity)
    independent_volume = sum(
        (
            _tetrahedron_volume(np.asarray(target.coordinates)[list(vertices)])
            for vertices in target_rows
        ),
        Fraction(0),
    )
    restored = GlobalEmbeddingCertificate.from_stored_fields(target_embedding)

    assert source_validity.all_certified
    assert source_embedding.status == "certified", source_embedding.findings
    assert source_embedding.periodic_mapped_degree is not None
    assert target_validity.all_certified
    assert transition.transition_id
    assert target_embedding.status == "certified", target_embedding.findings
    assert target_embedding.periodic_mapped_degree is not None
    assert target_embedding.periodic_mapped_degree.covering_degree == 1
    assert target_embedding.periodic_mapped_degree.pullback_volume == (1, 1)
    assert independent_volume == Fraction(1)
    assert restored.periodic_mapped_degree == target_embedding.periodic_mapped_degree
