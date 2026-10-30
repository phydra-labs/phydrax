from collections import Counter
from fractions import Fraction
from itertools import permutations, product

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._cell_geometry import (
    PerCellRestrictedCellGeometryElement,
)
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._coordinate_enclosure import axes
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.meshing._hex_generation import extract_volume_hexes


def _unit_tetrahedron() -> CellMesh:
    return CellMesh(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        (CellBlock("tetrahedra", "tetrahedron", np.asarray(((0, 1, 2, 3),))),),
    )


def test_tetrahedral_dual_is_an_explicit_positive_hex_conversion() -> None:
    source = _unit_tetrahedron()
    result = extract_volume_hexes(source)

    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert result.mesh.connectivity.cell_count == 4
    assert result.validity.certified_valid_count == 4
    assert result.validity.invalid_count == result.validity.unresolved_count == 0
    np.testing.assert_array_equal(
        result.mesh.vertex_global_ids[:4], source.vertex_global_ids
    )
    np.testing.assert_array_equal(
        np.diff(result.parent_offsets), np.ones(4, dtype=np.int64)
    )
    np.testing.assert_array_equal(result.parent_cells, np.zeros(4, dtype=np.int64))
    assert result.vertex_supports.shape == (15, 4)
    assert result.source.topology_id == source.topology_id
    result_geometry = result.geometry
    assert result_geometry is not None
    ancestry = result_geometry.restriction_source
    assert ancestry is not None
    assert ancestry.block_source_blocks == {"cells": "tetrahedra"}
    np.testing.assert_array_equal(
        ancestry.block_parent_cell_ids["cells"], np.zeros((4,), dtype=np.int64)
    )
    np.testing.assert_array_equal(
        ancestry.block_parent_vertex_ids["cells"],
        np.tile(np.asarray((0, 1, 2, 3), dtype=np.int64), (4, 1)),
    )
    assert np.all(result.mesh.blocks[0].vertices >= 0)
    assert np.all(result.mesh.blocks[0].vertices < result.mesh.coordinates.shape[0])


def test_dual_hex_accepts_simplex3_without_relabeling_source_ids() -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    vertex_ids = np.asarray((101, 103, 107, 109), dtype=np.int64)
    cell_ids = np.asarray((733,), dtype=np.int64)
    source = CellMesh(
        coordinates,
        (
            CellBlock(
                "simplex-source",
                "simplex:3",
                np.asarray(((0, 1, 2, 3),), dtype=np.int64),
                global_ids=cell_ids,
            ),
        ),
        vertex_global_ids=vertex_ids,
    )
    assert isinstance(source.connectivity, SimplicialConnectivity)

    result = extract_volume_hexes(source)

    assert result.validity.all_certified
    assert [block.cell_kind for block in result.mesh.blocks] == ["hexahedron"]
    geometry = result.geometry
    assert geometry is not None
    restriction = geometry.restriction_source
    assert restriction is not None
    assert restriction.block_source_blocks == {"cells": "simplex-source"}
    np.testing.assert_array_equal(
        restriction.block_parent_cell_ids["cells"],
        np.full((4,), int(cell_ids[0]), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        restriction.block_parent_vertex_ids["cells"],
        np.tile(vertex_ids, (4, 1)),
    )


def test_dual_hex_rejects_untyped_coordinate_geometry() -> None:
    source = _unit_tetrahedron()
    with pytest.raises(TypeError):
        extract_volume_hexes(
            source,
            source_geometry=object(),  # ty: ignore[invalid-argument-type]
        )


def _occupied_tetrahedra(occupied: set[tuple[int, int, int]]) -> CellMesh:
    vertices = sorted(
        {
            tuple(origin[axis] + offset[axis] for axis in range(3))
            for origin in occupied
            for offset in product((0, 1), repeat=3)
        }
    )
    vertex_row = {vertex: row for row, vertex in enumerate(vertices)}
    coordinates = np.asarray(vertices, dtype=np.float64)
    tetrahedra: list[list[int]] = []
    for origin in sorted(occupied):
        for order in permutations(range(3)):
            point = list(origin)
            cell = [vertex_row[tuple(point)]]
            for axis in order:
                point[axis] += 1
                cell.append(vertex_row[tuple(point)])
            points = coordinates[np.asarray(cell)]
            determinant = np.linalg.det(
                np.stack(
                    (
                        points[1] - points[0],
                        points[2] - points[0],
                        points[3] - points[0],
                    ),
                    axis=-1,
                )
            )
            if determinant < 0.0:
                cell[1], cell[2] = cell[2], cell[1]
            tetrahedra.append(cell)
    return CellMesh.from_tetrahedra(coordinates, np.asarray(tetrahedra, dtype=np.int64))


def _independent_hex_volumes(mesh: CellMesh) -> np.ndarray:
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    vertices = np.asarray(mesh.coordinates)[cells]
    reference = np.asarray(reference_cell_topology("hexahedron").vertices)
    samples = 0.5 + np.asarray((-1.0, 1.0)) / (2.0 * np.sqrt(3.0))
    volumes = np.zeros((vertices.shape[0],), dtype=np.float64)
    for point in product(samples, repeat=3):
        reference_point = np.asarray(point, dtype=np.float64)
        factors = np.where(reference == 1.0, reference_point, 1.0 - reference_point)
        derivative = np.empty((8, 3), dtype=np.float64)
        for axis in range(3):
            other_axes = [index for index in range(3) if index != axis]
            derivative[:, axis] = np.where(
                reference[:, axis] == 1.0, 1.0, -1.0
            ) * np.prod(factors[:, other_axes], axis=1)
        jacobians = np.swapaxes(vertices, 1, 2) @ derivative
        determinants = np.linalg.det(jacobians)
        assert np.all(determinants > 0.0)
        volumes += determinants / 8.0
    return volumes


@pytest.mark.parametrize("domain", ("cavity", "nonsweepable"))
def test_dual_hex_closes_positive_cavity_and_non_sweepable_tet_domains(
    domain: str,
) -> None:
    if domain == "cavity":
        occupied = {(x, y, z) for x in range(3) for y in range(3) for z in range(3)} - {
            (1, 1, 1)
        }
    else:
        occupied = {
            (1, 1, 1),
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
        }
    source = _occupied_tetrahedra(occupied)
    result = extract_volume_hexes(source)
    target_cells = np.asarray(result.mesh.blocks[0].vertices, dtype=np.int64)
    source_cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    parent_cells = np.asarray(result.parent_cells, dtype=np.int64)
    points = np.asarray(result.mesh.coordinates)

    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert target_cells.shape == (24 * len(occupied), 8)
    assert result.validity.certified_valid_count == target_cells.shape[0]
    assert np.sum(_independent_hex_volumes(result.mesh)) == pytest.approx(
        len(occupied), rel=1.0e-12
    )
    source_connectivity = source.connectivity
    assert isinstance(source_connectivity, TetrahedralConnectivity)
    source_boundary = {
        tuple(
            sorted(
                (
                    int(face[0]),
                    int(face[1]),
                    int(face[2]),
                )
            )
        )
        for face, boundary in zip(
            source_connectivity.faces,
            source_connectivity.boundary_faces,
            strict=True,
        )
        if bool(boundary)
    }
    target_connectivity = result.mesh.connectivity
    assert isinstance(target_connectivity, HexahedralConnectivity)
    target_boundary = np.asarray(target_connectivity.faces)[
        np.asarray(target_connectivity.boundary_faces, dtype=np.bool_)
    ]
    boundary_counts: Counter[tuple[int, ...]] = Counter()
    source_supports = np.asarray(result.vertex_supports)
    for face in target_boundary:
        support = tuple(
            sorted(
                {
                    int(vertex)
                    for row in face
                    for vertex in source_supports[row]
                    if vertex >= 0
                }
            )
        )
        assert support in source_boundary
        boundary_counts[support] += 1
    assert boundary_counts == Counter({face: 3 for face in source_boundary})
    for target_row, source_parent in enumerate(parent_cells.tolist()):
        parent_vertices = {
            int(value) for value in source_cells[int(source_parent)].tolist()
        }
        for target_vertex in target_cells[target_row].tolist():
            support = {
                int(value)
                for value in source_supports[int(target_vertex)].tolist()
                if int(value) >= 0
            }
            assert support <= parent_vertices
    if domain == "cavity":
        centers = np.mean(points[target_cells], axis=1)
        assert not np.any(np.all((centers > 1.0) & (centers < 2.0), axis=1))


def test_dual_hex_subdivides_a_material_interface_with_shared_quad_faces() -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    source = CellMesh.from_tetrahedra(
        coordinates,
        np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    )
    result = extract_volume_hexes(source)
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, HexahedralConnectivity)
    cells = np.asarray(connectivity.cell_faces, dtype=np.int64)
    faces = np.asarray(connectivity.faces, dtype=np.int64)
    supports = np.asarray(result.vertex_supports, dtype=np.int64)
    parents = np.asarray(result.parent_cells, dtype=np.int64)
    interface_faces = 0
    for face_row in range(faces.shape[0]):
        incident_cells = np.flatnonzero(np.any(cells == face_row, axis=1))
        if (
            incident_cells.size != 2
            or parents[incident_cells[0]] == parents[incident_cells[1]]
        ):
            continue
        source_support = {
            int(supports[int(faces[face_row, local_vertex]), source_row])
            for local_vertex in range(faces.shape[1])
            for source_row in range(supports.shape[1])
            if int(supports[int(faces[face_row, local_vertex]), source_row]) >= 0
        }
        interface_faces += source_support == {0, 1, 2}

    assert interface_faces == 3
    source_regions = np.asarray((0, 1), dtype=np.int64)
    target_regions = source_regions[parents]
    np.testing.assert_array_equal(np.bincount(target_regions), np.asarray((4, 4)))
    assert result.validity.all_certified


def test_curved_tetrahedral_dual_hexes_restrict_exact_source_charts() -> None:
    reference_tetra = np.asarray(reference_cell_topology("tetrahedron").vertices)
    source_points = reference_tetra.copy()
    source_points[:, 2] += source_points[:, 0] * source_points[:, 1] / 8.0
    source = CellMesh.from_tetrahedra(
        source_points,
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
    )
    element = lagrange_element("tetrahedron", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    geometry_nodes = reference_nodes.copy()
    geometry_nodes[:, 2] += geometry_nodes[:, 0] * geometry_nodes[:, 1] / 8.0
    source_geometry = CellGeometrySpec(
        {source.blocks[0].name: element},
        {source.blocks[0].name: np.arange(element.local_dof_count)[None, :]},
        geometry_nodes,
    )

    result = extract_volume_hexes(source, source_geometry=source_geometry)
    result_geometry = result.geometry
    assert result_geometry is not None

    supports = np.asarray(result.vertex_supports, dtype=np.int64)
    exact_parameters = np.asarray(
        [np.mean(reference_tetra[row[row >= 0]], axis=0) for row in supports]
    )
    coordinates = np.asarray(result.mesh.coordinates)
    np.testing.assert_allclose(
        coordinates[:, 2],
        exact_parameters[:, 2] + exact_parameters[:, 0] * exact_parameters[:, 1] / 8.0,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert result.validity.all_certified
    assert result_geometry.restriction_source is not None
    reference_hex = np.asarray(reference_cell_topology("hexahedron").vertices)
    for block in result.mesh.blocks:
        reference_points = np.broadcast_to(
            reference_hex, (block.cell_count, *reference_hex.shape)
        )
        mapped = result_geometry.evaluate(result.mesh, block.global_ids, reference_points)
        np.testing.assert_allclose(
            mapped,
            coordinates[np.asarray(block.vertices)],
            rtol=0.0,
            atol=1.0e-15,
        )

    block = source.blocks[0]
    source_cell_ids = tuple(
        int(value) for value in np.asarray(block.global_ids, dtype=np.int64).tolist()
    )
    wrapped = PerCellRestrictedCellGeometryElement(
        element,
        "tetrahedron",
        source_cell_ids,
        source_cell_ids,
        (axes(3),),
    )
    wrapped_geometry = CellGeometrySpec(
        {block.name: wrapped},
        {block.name: np.arange(element.local_dof_count, dtype=np.int64)[None, :]},
        geometry_nodes,
    )
    wrapped_result = extract_volume_hexes(source, source_geometry=wrapped_geometry)
    assert wrapped_result.validity.all_certified
    np.testing.assert_allclose(
        wrapped_result.mesh.coordinates,
        result.mesh.coordinates,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_multiblock_curved_dual_retains_material_and_root_chart_ancestry() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    blocks = (
        CellBlock(
            "material-a",
            "tetrahedron",
            np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            global_ids=np.asarray((17,), dtype=np.int64),
        ),
        CellBlock(
            "material-b",
            "tetrahedron",
            np.asarray(((0, 2, 1, 4),), dtype=np.int32),
            global_ids=np.asarray((41,), dtype=np.int64),
        ),
    )
    source = CellMesh(points, blocks)
    element = lagrange_element("tetrahedron", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    coefficient_rows: dict[tuple[int, ...], int] = {}
    coefficients: list[np.ndarray] = []
    routes: dict[str, np.ndarray] = {}
    for block in source.blocks:
        block_cells = np.asarray(block.vertices, dtype=np.int64)
        block_routes = np.empty(
            (block.cell_count, element.local_dof_count), dtype=np.int64
        )
        for cell_row in range(block_cells.shape[0]):
            for dof, reference in enumerate(reference_nodes):
                barycentric = np.concatenate(
                    (np.asarray((1.0 - np.sum(reference),)), reference)
                )
                support = tuple(
                    sorted(
                        int(vertex_ids[int(block_cells[cell_row, int(local)])])
                        for local in np.flatnonzero(barycentric).tolist()
                    )
                )
                source_rows = np.asarray(
                    [
                        int(np.flatnonzero(vertex_ids == identifier)[0])
                        for identifier in support
                    ],
                    dtype=np.int64,
                )
                point = np.mean(points[source_rows], axis=0)
                point[2] += point[0] * point[1] / 8.0
                coefficient_row = coefficient_rows.get(support)
                if coefficient_row is None:
                    coefficient_row = len(coefficients)
                    coefficient_rows[support] = coefficient_row
                    coefficients.append(point)
                elif not np.array_equal(coefficients[coefficient_row], point):
                    raise AssertionError(
                        "Shared source entities must share one geometry DOF."
                    )
                block_routes[cell_row, dof] = coefficient_row
        routes[block.name] = block_routes
    source_geometry = CellGeometrySpec(
        {block.name: element for block in source.blocks},
        routes,
        np.asarray(coefficients, dtype=np.float64),
    )

    affine_result = extract_volume_hexes(source)
    affine_geometry = affine_result.geometry
    assert affine_geometry is not None
    affine_restriction = affine_geometry.restriction_source
    assert affine_restriction is not None
    np.testing.assert_array_equal(
        affine_restriction.block_parent_cell_ids["material-a"],
        np.full((4,), 17, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        affine_restriction.block_parent_cell_ids["material-b"],
        np.full((4,), 41, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        affine_restriction.block_parent_vertex_ids["material-a"],
        np.tile(np.asarray((0, 1, 2, 3), dtype=np.int64), (4, 1)),
    )
    np.testing.assert_array_equal(
        affine_restriction.block_parent_vertex_ids["material-b"],
        np.tile(np.asarray((0, 2, 1, 4), dtype=np.int64), (4, 1)),
    )

    result = extract_volume_hexes(source, source_geometry=source_geometry)
    result_geometry = result.geometry
    assert result_geometry is not None

    assert [block.cell_kind for block in result.mesh.blocks] == [
        "hexahedron",
        "hexahedron",
    ]
    assert [block.name for block in result.mesh.blocks] == [
        "material-a",
        "material-b",
    ]
    np.testing.assert_array_equal(
        result.parent_cells, np.repeat(np.asarray((0, 1), dtype=np.int64), 4)
    )
    restriction = result_geometry.restriction_source
    assert restriction is not None
    assert restriction.block_source_blocks == {
        "material-a": "material-a",
        "material-b": "material-b",
    }
    np.testing.assert_array_equal(
        restriction.block_parent_cell_ids["material-a"],
        np.full((4,), 17, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        restriction.block_parent_cell_ids["material-b"],
        np.full((4,), 41, dtype=np.int64),
    )

    source_connectivity = source.connectivity
    assert isinstance(source_connectivity, PolyhedralConnectivity)
    source_edge_rows = np.asarray(source_connectivity.edges, dtype=np.int64)
    source_edge_ids = np.asarray(source.entity_set(1).entity_ids, dtype=np.int64)
    source_edge_row = next(
        row
        for row in range(source_edge_rows.shape[0])
        if (
            int(source_edge_rows[row, 0]),
            int(source_edge_rows[row, 1]),
        )
        in ((0, 1), (1, 0))
    )
    target_edge_ancestry = np.asarray(result.source_edge_ids, dtype=np.int64)
    matching_edges = np.flatnonzero(
        target_edge_ancestry == source_edge_ids[source_edge_row]
    )
    assert matching_edges.size == 2
    numerators = np.asarray(result.source_edge_parameter_numerators, dtype=np.int64)
    denominators = np.asarray(result.source_edge_parameter_denominators, dtype=np.int64)
    fractions = {
        tuple(
            Fraction(int(numerator), int(denominator))
            for numerator, denominator in zip(
                numerators[row], denominators[row], strict=True
            )
        )
        for row in matching_edges.tolist()
    }
    assert fractions == {
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1), Fraction(1, 2)),
    }
    assert result.validity.all_certified

    coordinates = np.asarray(result.mesh.coordinates)
    supports = np.asarray(result.vertex_supports)
    parents = np.asarray(result.parent_cells, dtype=np.int64)
    result_connectivity = result.mesh.connectivity
    assert isinstance(result_connectivity, PolyhedralConnectivity)
    face_offsets = np.asarray(result_connectivity.face_vertex_offsets, dtype=np.int64)
    face_values = np.asarray(result_connectivity.face_vertex_values, dtype=np.int64)
    cell_offsets = np.asarray(result_connectivity.cell_face_offsets, dtype=np.int64)
    cell_values = np.asarray(result_connectivity.cell_face_values, dtype=np.int64)
    face_owners = [[] for _ in range(int(result_connectivity.face_count))]
    for cell_row in range(int(result_connectivity.cell_count)):
        for face_row in cell_values[cell_offsets[cell_row] : cell_offsets[cell_row + 1]]:
            face_owners[int(face_row)].append(cell_row)
    interface_faces = 0
    for face_row, incident in enumerate(face_owners):
        if len(incident) != 2 or parents[incident[0]] == parents[incident[1]]:
            continue
        face = face_values[face_offsets[face_row] : face_offsets[face_row + 1]]
        source_support = {
            int(vertex)
            for target_vertex in face
            for vertex in supports[int(target_vertex)]
            if vertex >= 0
        }
        if source_support == {0, 1, 2}:
            interface_faces += 1
    assert interface_faces == 3
    for vertex_row, support in enumerate(supports):
        source_vertices = support[support >= 0]
        expected = np.mean(points[source_vertices], axis=0)
        expected[2] += expected[0] * expected[1] / 8.0
        np.testing.assert_allclose(
            coordinates[vertex_row], expected, rtol=0.0, atol=1.0e-15
        )
