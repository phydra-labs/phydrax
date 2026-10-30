from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.discretization._cell_geometry import _coordinate_coefficient_identity
from phydrax.discretization._exact_algebraic_geometry import ExactScalar
from phydrax.discretization.fem._reference import (
    FiniteElementSpec,
    lagrange_reference_node_data,
)
from phydrax.meshing._quad_generation import (
    extract_surface_quads,
    prepare_surface_cross_field,
    prepare_surface_quad_layout,
    QuadLayoutObstruction,
)


def _disconnected_square_pairs() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (2.0, 1.0),
        )
    )
    return CellMesh.from_triangles(
        points,
        np.asarray(((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7))),
    )


def _annulus() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 3.0),
            (0.0, 3.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (2.0, 2.0),
            (1.0, 2.0),
        )
    )
    triangles = np.asarray(
        (
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        )
    )
    return CellMesh.from_triangles(points, triangles)


def _triangular_annulus() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0),
            (3.0, 0.0),
            (1.5, 3.0),
            (1.0, 0.8),
            (2.0, 0.8),
            (1.5, 1.8),
        )
    )
    triangles = np.asarray(
        (
            (0, 1, 4), (0, 4, 3),
            (1, 2, 5), (1, 5, 4),
            (2, 0, 3), (2, 3, 5),
        )
    )
    return CellMesh.from_triangles(points, triangles)


def _interior_vertex_grid() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
            (0.0, 2.0), (1.0, 2.0), (2.0, 2.0),
        )
    )
    triangles = np.asarray(
        (
            (0, 1, 4), (0, 4, 3),
            (1, 2, 5), (1, 5, 4),
            (3, 4, 7), (3, 7, 6),
            (4, 5, 8), (4, 8, 7),
        )
    )
    return CellMesh.from_triangles(points, triangles)


def _concave_l() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        )
    )
    triangles = np.asarray(
        ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    )
    return CellMesh.from_triangles(points, triangles)


def _area(mesh: CellMesh) -> float:
    points = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.blocks[0].vertices)
    return float(
        np.sum(
            np.abs(
                np.sum(
                    points[cells, 0] * np.roll(points[cells, 1], -1, axis=1)
                    - points[cells, 1] * np.roll(points[cells, 0], -1, axis=1),
                    axis=1,
                )
            )
            / 2.0
        )
    )

def _boundary_edge_count(mesh: CellMesh) -> int:
    counts: dict[tuple[int, int], int] = {}
    for block in mesh.blocks:
        for cell in np.asarray(block.vertices).tolist():
            for a, b in zip(cell, (*cell[1:], cell[0]), strict=True):
                edge = (min(a, b), max(a, b))
                counts[edge] = counts.get(edge, 0) + 1
    return sum(count == 1 for count in counts.values())


def test_cross_field_recombination_changes_topology_and_retains_exact_ancestry() -> None:
    source = _disconnected_square_pairs()
    connectivity = source.connectivity
    assert isinstance(connectivity, PolygonalConnectivity)
    edges = np.asarray(connectivity.edges)
    first_diagonal = next(
        row for row, edge in enumerate(edges.tolist()) if tuple(sorted(edge)) == (0, 2)
    )

    cross_field = prepare_surface_cross_field(source)
    protected = prepare_surface_cross_field(source, feature_edges=[first_diagonal])
    layout = prepare_surface_quad_layout(source, cross_field)

    assert layout.paired_edge_rows.size == 2
    with pytest.raises(QuadLayoutObstruction) as obstruction:
        prepare_surface_quad_layout(source, protected)
    assert obstruction.value.category == "patch_parity"
    assert obstruction.value.source_cell_rows

    dual = extract_surface_quads(source)
    aligned = extract_surface_quads(source, cross_field=cross_field)
    source_edge_ids = np.asarray(dual.source_edge_ids, dtype=np.int64)
    mapped_source_edges = source_edge_ids >= 0
    assert np.any(mapped_source_edges)
    assert np.all(
        np.isin(
            source_edge_ids[mapped_source_edges],
            np.asarray(source.entity_set(1).entity_ids, dtype=np.int64),
        )
    )
    fractions = np.asarray(dual.source_edge_parameter_numerators, dtype=np.int64)[
        mapped_source_edges
    ]
    denominators = np.asarray(
        dual.source_edge_parameter_denominators, dtype=np.int64
    )[mapped_source_edges]
    assert np.all(denominators > 0)
    assert np.all(fractions <= denominators)
    assert np.any(denominators == 2)
    assert dual.mesh.connectivity.cell_count == 12
    assert aligned.mesh.connectivity.cell_count == 2
    assert aligned.validity.all_certified
    assert aligned.parent_offsets.shape == (3,)
    np.testing.assert_array_equal(np.diff(aligned.parent_offsets), np.full(2, 2))
    np.testing.assert_array_equal(
        np.asarray(aligned.vertex_supports)[:, 0],
        np.arange(source.coordinates.shape[0]),
    )
    target_source_edges = np.asarray(aligned.source_edge_ids, dtype=np.int64)
    assert np.all(target_source_edges >= 0)
    source_edge_entity_ids = np.asarray(source.entity_set(1).entity_ids, dtype=np.int64)
    feature_edge_ids = source_edge_entity_ids[
        np.asarray(cross_field.feature_edges, dtype=np.int64)
    ]
    assert np.all(np.isin(feature_edge_ids, target_source_edges))
    assert np.all(dual.valences[dual.singular_vertices] != 4)
    assert not np.array_equal(dual.mesh.topology_id, aligned.mesh.topology_id)
    assert float(cross_field.field_solve_relative_residual) <= 1.0e-10
    assert cross_field.field_solver == "scipy.sparse.linalg.spsolve"
    assert cross_field.singular_vertices.shape == cross_field.singularity_indices.shape
    assert float(cross_field.field_energy) >= 0.0
    assert _area(dual.mesh) == pytest.approx(2.0)
    assert _area(aligned.mesh) == pytest.approx(2.0)


def test_simplex2_lowering_keeps_source_family_during_quad_extraction() -> None:
    baseline = _disconnected_square_pairs()
    baseline_block = baseline.blocks[0]
    source = CellMesh(
        np.asarray(baseline.coordinates),
        (
            CellBlock(
                baseline_block.name,
                "simplex:2",
                np.asarray(baseline_block.vertices, dtype=np.int64),
                global_ids=np.asarray(baseline_block.global_ids, dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(baseline.vertex_global_ids, dtype=np.int64),
    )
    connectivity = source.connectivity
    assert isinstance(connectivity, SimplicialConnectivity)
    assert connectivity.dimension == 2

    cross_field = prepare_surface_cross_field(source)
    layout = prepare_surface_quad_layout(source, cross_field)
    result = extract_surface_quads(source, cross_field=cross_field)

    assert layout.paired_edge_rows.size == 2
    assert source.blocks[0].cell_kind == "simplex:2"
    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    assert result.validity.all_certified
    assert result.mesh.connectivity.cell_count == 2
    np.testing.assert_array_equal(
        np.sort(np.asarray(result.parent_cells, dtype=np.int64)),
        np.arange(connectivity.cell_count, dtype=np.int64),
    )
    assert _area(result.mesh) == pytest.approx(2.0)


def test_feature_guided_quads_preserve_a_multiply_connected_domain() -> None:
    source = _annulus()
    field = prepare_surface_cross_field(source)
    result = extract_surface_quads(source, cross_field=field)
    topology = result.mesh.topology

    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    assert result.mesh.connectivity.cell_count == 4
    assert result.validity.all_certified
    assert _area(result.mesh) == pytest.approx(8.0)
    assert _boundary_edge_count(result.mesh) == _boundary_edge_count(source)
    assert (
        topology.entities(0).count
        - topology.entities(1).count
        + topology.entities(2).count
    ) == 0
    np.testing.assert_array_equal(
        np.sort(np.asarray(result.parent_cells)),
        np.arange(source.connectivity.cell_count),
    )
    assert result.parent_offsets.shape == (result.mesh.connectivity.cell_count + 1,)
    np.testing.assert_array_equal(np.diff(result.parent_offsets), np.full(4, 2))
    assert result.boundary_vertices.sum() == 8





def test_feature_guided_layout_enforces_interior_regular_valence() -> None:
    source = _interior_vertex_grid()
    field = prepare_surface_cross_field(source)
    result = extract_surface_quads(source, cross_field=field)

    assert result.mesh.connectivity.cell_count == 4
    assert result.validity.all_certified
    assert not bool(np.asarray(result.boundary_vertices)[4])
    assert int(np.asarray(result.valences)[4]) == 4
    assert 4 not in set(np.asarray(result.singular_vertices).tolist())
    np.testing.assert_array_equal(
        np.sort(np.asarray(result.parent_cells)),
        np.arange(source.connectivity.cell_count),
    )


def test_feature_guided_quads_preserve_a_concave_source_domain() -> None:
    source = _concave_l()
    field = prepare_surface_cross_field(source)
    result = extract_surface_quads(
        source, cross_field=field, maximum_field_error=1.5
    )
    topology = result.mesh.topology

    assert result.mesh.connectivity.cell_count == 2
    assert result.validity.all_certified
    assert _area(result.mesh) == pytest.approx(3.0)
    assert _boundary_edge_count(result.mesh) == _boundary_edge_count(source)
    assert (
        topology.entities(0).count
        - topology.entities(1).count
        + topology.entities(2).count
    ) == 1


def test_cross_field_route_reports_unsatisfied_patch_parity() -> None:
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),)),
    )
    field = prepare_surface_cross_field(source)

    with pytest.raises(QuadLayoutObstruction) as obstruction:
        extract_surface_quads(source, cross_field=field)
    assert obstruction.value.category == "patch_parity"
    assert obstruction.value.source_cell_rows == (0,)



def test_cross_field_route_reports_odd_boundary_cycle() -> None:
    source = _triangular_annulus()
    field = prepare_surface_cross_field(source)

    with pytest.raises(QuadLayoutObstruction) as obstruction:
        prepare_surface_quad_layout(source, field)
    assert obstruction.value.category == "boundary_parity"
    assert len(obstruction.value.source_edge_rows) == 3


def test_curved_dual_quads_restrict_exact_source_charts() -> None:
    parameter_vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    source_points = np.column_stack(
        (
            parameter_vertices,
            parameter_vertices[:, 0] * parameter_vertices[:, 1] / 8.0,
        )
    )
    source = CellMesh.from_triangles(source_points, triangles)
    element = lagrange_element("triangle", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinate_rows: dict[tuple[Fraction, Fraction], int] = {}
    coordinate_bank: list[tuple[float, float, float]] = []
    geometry_routes = []
    exact_parameters = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_nodes
    )
    for triangle in triangles:
        vertices = parameter_vertices[triangle]
        route = []
        for reference in exact_parameters:
            u = (
                Fraction.from_float(float(vertices[0, 0]))
                + reference[0]
                * Fraction.from_float(float(vertices[1, 0] - vertices[0, 0]))
                + reference[1]
                * Fraction.from_float(float(vertices[2, 0] - vertices[0, 0]))
            )
            v = (
                Fraction.from_float(float(vertices[0, 1]))
                + reference[0]
                * Fraction.from_float(float(vertices[1, 1] - vertices[0, 1]))
                + reference[1]
                * Fraction.from_float(float(vertices[2, 1] - vertices[0, 1]))
            )
            key = (u, v)
            if key not in coordinate_rows:
                coordinate_rows[key] = len(coordinate_bank)
                coordinate_bank.append((float(u), float(v), float(u * v / 8)))
            route.append(coordinate_rows[key])
        geometry_routes.append(route)
    block_name = source.blocks[0].name
    source_geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: np.asarray(geometry_routes, dtype=np.int32)},
        np.asarray(coordinate_bank, dtype=np.float64),
    )
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source)
    assert source_validity.invalid_count == 0
    assert source_validity.unresolved_count == 0
    expected_source_status = np.full(
        (2,), CellValidityStatus.CERTIFIED_VALID, dtype=np.int32
    )
    np.testing.assert_array_equal(
        source_validity.source_status, expected_source_status
    )
    np.testing.assert_array_equal(
        source_validity.execution_status, expected_source_status
    )

    result = extract_surface_quads(source, source_geometry=source_geometry)

    coordinates = np.asarray(result.mesh.coordinates)
    np.testing.assert_allclose(
        coordinates[:, 2],
        coordinates[:, 0] * coordinates[:, 1] / 8.0,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert result.validity.all_certified
    geometry = result.geometry
    assert geometry is not None
    assert geometry.restriction_source is not None
    for block in result.mesh.blocks:
        reference = np.asarray(reference_cell_topology("quadrilateral").vertices)
        reference_points = np.broadcast_to(
            reference, (block.cell_count, *reference.shape)
        )
        mapped = geometry.evaluate(
            result.mesh, block.global_ids, reference_points
        )
        np.testing.assert_allclose(
            mapped,
            coordinates[np.asarray(block.vertices)],
            rtol=0.0,
            atol=1.0e-15,
        )


def test_folded_lagrange2_source_map_is_invalid() -> None:
    element = lagrange_element("triangle", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    folded_controls = np.column_stack(
        (
            reference_nodes[:, 0],
            reference_nodes[:, 1] * (1.0 - 2.0 * reference_nodes[:, 0]),
        )
    )
    mesh = CellMesh.from_triangles(
        folded_controls[:3],
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    block_name = mesh.blocks[0].name
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: np.arange(reference_nodes.shape[0], dtype=np.int32)[None, :]},
        folded_controls,
    )

    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    assert validity.invalid_count == 1
    assert validity.unresolved_count == 0
    np.testing.assert_array_equal(
        validity.status,
        np.asarray((CellValidityStatus.INVALID,), dtype=np.int32),
    )


def test_unsupported_coordinate_family_remains_unresolved_with_block_identity() -> None:
    canonical = lagrange_element("triangle", 2)
    element = FiniteElementSpec(
        "OpaqueLagrange",
        canonical.cell_kind,
        canonical.degree,
        np.asarray(canonical.reference_nodes),
        canonical.entity_dofs,
        value_spec=canonical.value_spec,
        continuity=canonical.continuity,
        representation=canonical.representation,
    )
    coordinates = np.asarray(canonical.reference_nodes, dtype=np.float64)
    mesh = CellMesh.from_triangles(
        coordinates[:3],
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    block_name = mesh.blocks[0].name
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: np.arange(element.local_dof_count, dtype=np.int32)[None, :]},
        coordinates,
    )

    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    assert validity.invalid_count == 0
    assert validity.unresolved_count == 1
    assert validity.unsupported_block_names == (block_name,)
    np.testing.assert_array_equal(
        validity.status,
        np.asarray((CellValidityStatus.UNRESOLVED,), dtype=np.int32),
    )


@final
class _ExactCoordinateBankSourceMap(StrictModule, NonTrainableState):
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)
    coordinate_bank: tuple[tuple[ExactScalar, ExactScalar, ExactScalar], ...] = eqx.field(
        static=True
    )

    def __init__(
        self,
        source: CellMesh,
        coordinate_bank: tuple[tuple[ExactScalar, ExactScalar, ExactScalar], ...],
        /,
    ) -> None:
        self.source_id = "curved-cross-field-source"
        self.source_revision = source.topology_id
        self.coordinate_bank = coordinate_bank
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "exact-coordinate-bank-source",
                "source": self.source_id,
                "revision": self.source_revision,
                "coordinates": tuple(
                    tuple(_coordinate_coefficient_identity(value) for value in row)
                    for row in coordinate_bank
                ),
            }
        )

    def source_coordinates(
        self, /
    ) -> tuple[tuple[ExactScalar, ExactScalar, ExactScalar], ...]:
        return self.coordinate_bank


def test_curved_cross_field_layout_preserves_one_exact_shared_patch_map() -> None:
    parameters = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    source = CellMesh.from_triangles(
        np.column_stack((parameters, np.zeros((4,), dtype=np.float64))),
        triangles,
    )
    element = lagrange_element("triangle", 4)
    nodal_points = tuple(
        node.exact_point for node in lagrange_reference_node_data(element)
    )
    if any(point is None for point in nodal_points):
        raise ValueError("The curved source requires exact Lagrange node identities.")
    reference_nodes = tuple(point for point in nodal_points if point is not None)
    if len(reference_nodes) != element.local_dof_count:
        raise ValueError("Exact Lagrange node identities must cover every local DOF.")
    coordinate_rows: dict[tuple[ExactScalar, ExactScalar], int] = {}
    coordinate_bank: list[tuple[ExactScalar, ExactScalar, ExactScalar]] = []
    routes = []
    for triangle in triangles:
        vertices = parameters[triangle]
        route = []
        for first, second in reference_nodes:
            u = Fraction.from_float(float(vertices[0, 0])) + first * Fraction.from_float(
                float(vertices[1, 0] - vertices[0, 0])
            ) + second * Fraction.from_float(float(vertices[2, 0] - vertices[0, 0]))
            v = Fraction.from_float(float(vertices[0, 1])) + first * Fraction.from_float(
                float(vertices[1, 1] - vertices[0, 1])
            ) + second * Fraction.from_float(float(vertices[2, 1] - vertices[0, 1]))
            key = (u, v)
            if key not in coordinate_rows:
                coordinate_rows[key] = len(coordinate_bank)
                height = u * (1 - u) * v * (1 - v) / 4
                coordinate_bank.append((u, v, height))
            route.append(coordinate_rows[key])
        routes.append(route)
    block_name = source.blocks[0].name
    exact_coordinate_bank = tuple(coordinate_bank)
    source_map = _ExactCoordinateBankSourceMap(source, exact_coordinate_bank)
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: np.asarray(routes, dtype=np.int32)},
        np.asarray(
            tuple(tuple(float(value) for value in row) for row in coordinate_bank),
            dtype=np.float64,
        ),
        source_map=source_map,
        source_mesh=source,
    )
    cross_field = prepare_surface_cross_field(source)
    dual = extract_surface_quads(source, source_geometry=geometry)

    result = extract_surface_quads(
        source,
        cross_field=cross_field,
        source_geometry=geometry,
    )

    assert dual.mesh.connectivity.cell_count == 6
    assert result.mesh.connectivity.cell_count == 1
    assert result.validity.all_certified
    coordinates = np.asarray(result.mesh.coordinates)
    supports = np.asarray(result.vertex_supports)
    carrier = np.asarray(source.coordinates)
    parameter_points = np.asarray(
        [
            np.mean(carrier[row[row >= 0], :2], axis=0)
            for row in supports
        ]
    )
    expected_height = (
        parameter_points[:, 0]
        * (1.0 - parameter_points[:, 0])
        * parameter_points[:, 1]
        * (1.0 - parameter_points[:, 1])
        / 4.0
    )
    np.testing.assert_allclose(
        coordinates[:, 2], expected_height, rtol=0.0, atol=1.0e-14
    )
    geometry = result.geometry
    assert geometry is not None
    reference = np.asarray(reference_cell_topology("quadrilateral").vertices)
    block = result.mesh.blocks[0]
    mapped = geometry.evaluate(
        result.mesh,
        block.global_ids,
        np.broadcast_to(reference, (block.cell_count, *reference.shape)),
    )
    np.testing.assert_allclose(
        mapped,
        coordinates[np.asarray(block.vertices)],
        rtol=0.0,
        atol=1.0e-14,
    )
