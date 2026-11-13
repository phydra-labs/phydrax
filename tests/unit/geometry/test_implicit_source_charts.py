#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import cast, final, NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.discretization._cell_geometry import (
    CellGeometryElement,
    CellGeometrySpec,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._exact_algebraic_geometry import ExactScalar
from phydrax.discretization.fem._reference import (
    FiniteElementSpec,
    lagrange_reference_node_data,
)
from phydrax.geometry._capabilities import GeometryCapability
from phydrax.geometry._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from phydrax.geometry._contracts import CompiledGeometry, GeometryKernel, GeometryKind
from phydrax.geometry._sampling import RejectionSamplingPlan
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryDistance,
    SourceBoundaryProjectionResult,
    SourceBoundaryQuery,
    SourceBoundarySamples,
    SourceProjectionStatus,
)
from phydrax.geometry._source_fidelity import (
    _implicit_chart_id,
    certify_source_fidelity,
    validate_source_fidelity_witness,
)
from phydrax.geometry._source_identity import source_entity_id
from phydrax.geometry._validity import GeometryValidityEvidence, unrestricted_validity
from phydrax.geometry.design._schema import _ParameterCollector, DesignState
from phydrax.geometry.implicit import (
    _adaptive_discovery as adaptive_discovery,
    _source_charts as source_charts,
    AnalyticImplicitProfile,
)
from phydrax.geometry.implicit._adaptive_discovery import (
    AdaptiveImplicitBoundarySource,
    AdaptiveImplicitSurface,
    discover_adaptive_implicit_surface,
)
from phydrax.geometry.implicit._chart_restriction import (
    ImplicitChartPremise,
    ImplicitChartRows,
)
from phydrax.geometry.implicit._policy import (
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitSurfacePolicy,
)
from phydrax.geometry.implicit._source_charts import (
    _AdaptiveImplicitProjectionResult,
    AdaptiveImplicitRootChartMap,
    certify_adaptive_implicit_charts,
    prepare_adaptive_implicit_root_charts,
)
from phydrax.meshing._curving import _straight_geometry
from phydrax.typing import ConvertibleToArray, Identifier, PRNGKey
from phydrax.units import METER


_STANDARD = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
type GridCorner = tuple[int, int, int]
type GridEdge = tuple[GridCorner, GridCorner]
type SourceCellKey = tuple[int, int, int, int]
type RootEdgeTriangle = tuple[GridEdge, GridEdge, GridEdge]


def _grid_corner(values: np.ndarray, /) -> GridCorner:
    return int(values[0]), int(values[1]), int(values[2])


def _source_cell_key(values: np.ndarray, /) -> SourceCellKey:
    return int(values[0]), int(values[1]), int(values[2]), int(values[3])


def _root_edge_triangle(values: np.ndarray, /) -> RootEdgeTriangle:
    return (
        (_grid_corner(values[0, 0]), _grid_corner(values[0, 1])),
        (_grid_corner(values[1, 0]), _grid_corner(values[1, 1])),
        (_grid_corner(values[2, 0]), _grid_corner(values[2, 1])),
    )


def _triangle_vertex_key(
    vertex_global_ids: np.ndarray, vertices: np.ndarray, /
) -> tuple[int, int, int]:
    values = sorted(
        (
            int(vertex_global_ids[int(vertices[0])]),
            int(vertex_global_ids[int(vertices[1])]),
            int(vertex_global_ids[int(vertices[2])]),
        )
    )
    return values[0], values[1], values[2]


class _BoundAdaptiveSourceQuery(StrictModule, NonTrainableState):
    """Strict test carrier matching the native frame-bound source-query owner."""

    __strict_contract__ = True

    query: AdaptiveImplicitBoundarySource
    authoring_profile: AnalyticImplicitProfile | None
    source_id: Identifier = eqx.field(static=True)
    source_revision: Identifier = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        query: AdaptiveImplicitBoundarySource,
        frame_id: str,
        /,
        *,
        authoring_profile: AnalyticImplicitProfile | None = None,
    ) -> None:
        if authoring_profile is not None:
            if (
                authoring_profile.source_id != query.source_id
                or authoring_profile.source_revision != query.source_revision
                or authoring_profile.coordinate_contract.spatial_id != frame_id
            ):
                raise ValueError("Profile must bind the test source identity and frame.")
            authoring_profile.require_bound(
                query.geometry, authoring_profile.coordinate_contract
            )
        self.query = query
        self.authoring_profile = authoring_profile
        self.source_id = query.source_id
        self.source_revision = query.source_revision
        self.frame_id = frame_id
        self.ambient_dimension = 3

    @property
    def adaptive_source(self) -> AdaptiveImplicitBoundarySource:
        return self.query

    def require_current(self) -> None:
        self.query.require_current()

    def boundary_distance(self, points: ConvertibleToArray, /) -> SourceBoundaryDistance:
        return self.query.boundary_distance(np.asarray(points, dtype=np.float64))

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples:
        return self.query.boundary_cover(maximum_radius)

    def project_boundary(
        self,
        points: ConvertibleToArray,
        /,
        *,
        entity_ids: tuple[str, ...] | None = None,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> SourceBoundaryProjectionResult:
        return self.query.project_boundary(
            np.asarray(points, dtype=np.float64),
            entity_ids=entity_ids,
            occurrence_paths=occurrence_paths,
            frame_id=self.frame_id,
        )


def _adaptive_source(
    *, smooth: bool = True, authoring_profile: bool = False
) -> tuple[
    AdaptiveImplicitSurface,
    _BoundAdaptiveSourceQuery,
    SpatialCoordinateContract,
    CellMesh,
    CellGeometrySpec,
    SourceBoundaryChartWitness,
]:
    geometry = (
        phx.geometry.Superquadric((0.0, 0.0, 0.0), (0.45, 0.50, 0.55)).compile()
        if smooth
        else phx.geometry.Sphere((0.0, 0.0, 0.0), 0.5).compile()
    )
    discovery_domain = np.asarray(
        ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
        if authoring_profile
        else ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    )
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.05 if authoring_profile else 0.12,
        root_tolerance=1.0e-8,
        allow_nonsmooth_field=not smooth,
    )
    surface = discover_adaptive_implicit_surface(
        geometry,
        domain=discovery_domain,
        policy=policy,
        source_id="implicit-root-chart-test",
    )
    if surface.mesh is None:
        raise AssertionError("The bounded implicit fixture must discover a surface mesh.")
    contract = SpatialCoordinateContract(METER)
    adaptive_query = AdaptiveImplicitBoundarySource(surface, frame_id=contract.spatial_id)
    profile = (
        AnalyticImplicitProfile(
            geometry,
            contract,
            source_id=adaptive_query.source_id,
            source_revision=adaptive_query.source_revision,
        )
        if authoring_profile
        else None
    )
    source = _BoundAdaptiveSourceQuery(
        adaptive_query,
        contract.spatial_id,
        authoring_profile=profile,
    )
    source.require_current()
    mesh = CellMesh.from_triangles(
        surface.mesh.vertices, surface.mesh.faces.astype(np.int32)
    )
    geometry_map = CellGeometrySpec.affine(mesh)
    witness = _chart_witness(mesh, geometry_map, source)
    return surface, source, contract, mesh, geometry_map, witness


def _chart_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: _BoundAdaptiveSourceQuery,
    /,
) -> SourceBoundaryChartWitness:
    surface = source.adaptive_source.surface
    scaffold = surface.scaffold
    block = mesh.blocks[0]
    if not isinstance(block, CellBlock):
        raise TypeError("Implicit chart tests require standard triangular target blocks.")
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    cell_rows = np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)
    root_rows = np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64)
    cell_keys = np.asarray(scaffold.cell_indices, dtype=np.int64)[cell_rows]
    root_edges = np.asarray(scaffold.root_edge_ids, dtype=np.int64)[root_rows]
    chart_ids: list[str] = []
    for cell_key, edges in zip(cell_keys, root_edges, strict=True):
        chart_ids.append(
            _implicit_chart_id(
                source.source_id,
                source.source_revision,
                _source_cell_key(np.asarray(cell_key, dtype=np.int64)),
                _root_edge_triangle(np.asarray(edges, dtype=np.int64)),
            )
        )
    parameters = np.broadcast_to(_STANDARD, (cell_ids.size, 3, 2))
    surface_entity_key = source_entity_id(
        source.source_id,
        source.source_revision,
        "implicit-surface",
        "zero-set",
    )
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_revision,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=cell_ids,
        entity_global_ids=cell_ids,
        local_entity_indices=np.full(cell_ids.shape, -1, dtype=np.int32),
        target_reference_vertices=parameters,
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=(surface_entity_key,) * cell_ids.size,
        occurrence_paths=((),) * cell_ids.size,
        source_parameters=parameters,
        orientations=np.ones(cell_ids.shape, dtype=np.int8),
        implicit_cell_keys=cell_keys,
        implicit_root_edge_keys=root_edges,
    )


def _retarget_implicit_witness(
    witness: SourceBoundaryChartWitness,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: ConvertibleToArray,
    /,
    *,
    rows: Sequence[int] | None = None,
    entity_ids: ConvertibleToArray | None = None,
    source_entity_ids: tuple[str, ...] | None = None,
) -> SourceBoundaryChartWitness:
    selected_rows = (
        tuple(range(len(witness.source_chart_ids))) if rows is None else tuple(rows)
    )
    entity_key_rows = (
        witness.source_entity_ids if source_entity_ids is None else source_entity_ids
    )
    target_cells = np.asarray(cell_ids, dtype=np.int64)[list(selected_rows)]
    target_entities = (
        target_cells
        if entity_ids is None
        else np.asarray(entity_ids, dtype=np.int64)[list(selected_rows)]
    )
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        chart_dimension=witness.chart_dimension,
        reference_dimension=witness.reference_dimension,
        cell_global_ids=target_cells,
        entity_global_ids=target_entities,
        local_entity_indices=np.asarray(witness.local_entity_indices)[
            list(selected_rows)
        ],
        target_reference_vertices=np.asarray(witness.target_reference_vertices)[
            list(selected_rows)
        ],
        target_reference_rational_tokens=tuple(
            witness.target_reference_rational_tokens[row] for row in selected_rows
        ),
        source_chart_ids=tuple(witness.source_chart_ids[row] for row in selected_rows),
        source_entity_ids=tuple(entity_key_rows[row] for row in selected_rows),
        occurrence_paths=tuple(witness.occurrence_paths[row] for row in selected_rows),
        source_parameters=np.asarray(witness.source_parameters)[list(selected_rows)],
        source_parameter_rational_tokens=tuple(
            witness.source_parameter_rational_tokens[row] for row in selected_rows
        ),
        orientations=np.asarray(witness.orientations)[list(selected_rows)],
        implicit_cell_keys=np.asarray(witness.implicit_cell_keys)[list(selected_rows)],
        implicit_root_edge_keys=np.asarray(witness.implicit_root_edge_keys)[
            list(selected_rows)
        ],
    )


def _extend_implicit_target_mesh(
    mesh: CellMesh, witness: SourceBoundaryChartWitness, /
) -> tuple[CellMesh, CellGeometrySpec, SourceBoundaryChartWitness]:
    old_block = mesh.blocks[0]
    if not isinstance(old_block, CellBlock):
        raise TypeError(
            "Implicit chart target fixtures require standard triangular blocks."
        )
    old_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    old_cell_keys: dict[int, tuple[int, int, int]] = {
        int(cell_id): _triangle_vertex_key(
            old_vertex_ids, np.asarray(vertices, dtype=np.int64)
        )
        for cell_id, vertices in zip(
            np.asarray(old_block.global_ids, dtype=np.int64),
            np.asarray(old_block.vertices, dtype=np.int64),
            strict=True,
        )
    }
    extra_points = np.asarray(((4.0, 4.0, 4.0), (5.0, 4.0, 4.0), (4.0, 5.0, 4.0)))
    points = np.concatenate((np.asarray(mesh.coordinates), extra_points), axis=0)
    extra_vertices = np.arange(
        np.asarray(mesh.coordinates).shape[0],
        np.asarray(mesh.coordinates).shape[0] + 3,
        dtype=np.int32,
    )
    faces = np.concatenate(
        (
            np.asarray(old_block.vertices, dtype=np.int32),
            extra_vertices[None, :],
        ),
        axis=0,
    )
    target_mesh = CellMesh.from_triangles(points, faces)
    target_geometry = CellGeometrySpec.affine(target_mesh)
    target_vertex_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    target_block = target_mesh.blocks[0]
    if not isinstance(target_block, CellBlock):
        raise TypeError(
            "Implicit chart target fixtures require standard triangular blocks."
        )
    target_cell_by_key: dict[tuple[int, int, int], int] = {
        _triangle_vertex_key(
            target_vertex_ids, np.asarray(vertices, dtype=np.int64)
        ): int(cell_id)
        for cell_id, vertices in zip(
            np.asarray(target_block.global_ids, dtype=np.int64),
            np.asarray(target_block.vertices, dtype=np.int64),
            strict=True,
        )
    }
    target_cell_ids = np.asarray(
        [
            target_cell_by_key[old_cell_keys[int(cell_id)]]
            for cell_id in np.asarray(witness.cell_global_ids, dtype=np.int64)
        ],
        dtype=np.int64,
    )
    return (
        target_mesh,
        target_geometry,
        _retarget_implicit_witness(
            witness, target_mesh, target_geometry, target_cell_ids
        ),
    )


def _affine_root_map(
    mesh: CellMesh,
    source: _BoundAdaptiveSourceQuery,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryChartWitness,
) -> AdaptiveImplicitRootChartMap:
    root_map = prepare_adaptive_implicit_root_charts(
        mesh,
        geometry,
        source,
        witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )
    assert root_map.status == "certified"
    return root_map


@pytest.mark.parametrize(
    ("signed", "minimum", "maximum"),
    (
        ((-4.0, -1.0), 1.0, 4.0),
        ((1.0, 4.0), 1.0, 4.0),
        ((-4.0, 1.0), 0.0, 4.0),
        ((0.0, 0.0), 0.0, 0.0),
    ),
)
def test_signed_field_residual_magnitude_is_outward_enclosed(
    signed: tuple[float, float], minimum: float, maximum: float
) -> None:
    lower, upper = source_charts._absolute_residual_interval(signed)
    assert 0.0 <= lower <= minimum
    assert upper >= maximum


@pytest.mark.parametrize("signed", ((float("nan"), 1.0), (-1.0, float("nan"))))
def test_signed_field_residual_preserves_nan(signed: tuple[float, float]) -> None:
    lower, upper = source_charts._absolute_residual_interval(signed)
    assert np.isnan(lower) and np.isnan(upper)


def test_unbounded_field_residual_remains_unbounded() -> None:
    _, upper = source_charts._absolute_residual_interval((-float("inf"), 1.0))
    assert np.isinf(upper)


def test_off_surface_face_certifies_absolute_field_residual() -> None:
    surface, source, contract, mesh, _affine, witness = _adaptive_source()
    block = mesh.blocks[0]
    if not isinstance(block, CellBlock):
        raise TypeError("Implicit chart tests require standard triangular target blocks.")
    target = CellMesh.from_triangles(
        0.8 * np.asarray(mesh.coordinates, dtype=np.float64),
        np.asarray(block.vertices, dtype=np.int32),
        vertex_global_ids=mesh.vertex_global_ids,
        cell_global_ids=block.global_ids,
    )
    interior = CellGeometrySpec.affine(target)
    selected = _retarget_implicit_witness(
        witness, target, interior, np.asarray(block.global_ids, dtype=np.int64), rows=(0,)
    )
    root_map = prepare_adaptive_implicit_root_charts(
        target, interior, source, selected, frame_id=contract.spatial_id, tolerance=0.5
    )
    face = root_map._faces[0]
    barycentre = (Fraction(1, 3), Fraction(1, 3))
    point = jnp.asarray(
        [
            float(source_charts.evaluate_expression(expression, barycentre))
            for expression in face.face_expressions
        ]
    )
    field = float(surface.geometry.kernel.boundary_field(surface.geometry.state, point))
    assert field < 0.0
    assert face.evidence.status == "certified"
    assert any(piece.field_residual[0] > 0.0 for piece in face.pieces)
    assert face.evidence.field_residual_bound >= abs(field)


@pytest.mark.parametrize("nan_first", (False, True))
def test_nonfinite_piece_residual_never_creates_reusable_face(
    monkeypatch: pytest.MonkeyPatch, nan_first: bool
) -> None:
    from dataclasses import replace

    _, source, contract, mesh, _, witness = _adaptive_source()
    block = mesh.blocks[0]
    if not isinstance(block, CellBlock):
        raise TypeError("Implicit chart tests require standard triangular target blocks.")
    target = CellMesh.from_triangles(
        0.8 * np.asarray(mesh.coordinates, dtype=np.float64),
        np.asarray(block.vertices, dtype=np.int32),
        vertex_global_ids=mesh.vertex_global_ids,
        cell_global_ids=block.global_ids,
    )
    geometry = CellGeometrySpec.affine(target)
    selected = _retarget_implicit_witness(
        witness, target, geometry, np.asarray(block.global_ids, dtype=np.int64), rows=(0,)
    )
    certify = source_charts._certify_face_injectivity

    def inject_nonfinite(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        pieces, margin, detail = yield from certify(*args, **kwargs)
        assert pieces and not detail
        finite = pieces[0]
        nonfinite = replace(finite, field_residual=(float("nan"), float("nan")))
        return ((nonfinite, finite) if nan_first else (finite, nonfinite)), margin, detail

    monkeypatch.setattr(source_charts, "_certify_face_injectivity", inject_nonfinite)
    with pytest.raises(ValueError, match="residual enclosure is not finite"):
        prepare_adaptive_implicit_root_charts(
            target,
            geometry,
            source,
            selected,
            frame_id=contract.spatial_id,
            tolerance=0.5,
        )
    certificate = certify_adaptive_implicit_charts(
        target, geometry, source, selected, tolerance=0.5, maximum_normal_deviation=None
    )
    assert certificate.status == "unresolved"
    assert any(
        "residual enclosure is not finite" in finding.detail
        for finding in certificate.findings
    )


def test_reissued_tetra_boundary_chart_proves_actual_oriented_face() -> None:
    from phydrax.meshing.providers._native_implicit import (
        _implicit_remeshed_volume_chart_witness,
    )
    from phydrax.meshing.providers._native_sources import NativeImplicitSource

    surface, source, contract, _, _, _ = _adaptive_source()
    assert surface.mesh is not None
    scaffold_points = np.asarray(surface.mesh.vertices, dtype=np.float64)
    scaffold_faces = np.asarray(surface.mesh.faces, dtype=np.int64)
    triangle = scaffold_points[scaffold_faces[0]] * 0.999
    points = np.concatenate((np.zeros((1, 3)), triangle))
    cell = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    if np.linalg.det(points[cell[0, 1:]] - points[cell[0, 0]]) < 0:
        cell[0, [2, 3]] = cell[0, [3, 2]]
    mesh = CellMesh(points, (CellBlock("one", "simplex:3", cell),))
    geometry = CellGeometrySpec.affine(mesh)
    authority = NativeImplicitSource(
        surface.geometry, source.source_id, source.source_revision, discovery=surface
    )
    witness = _implicit_remeshed_volume_chart_witness(authority, surface, mesh, geometry)
    root_edges = np.asarray(surface.scaffold.root_edge_ids)[
        np.asarray(surface.scaffold.triangle_root_edge_rows)[0]
    ]
    rows = np.flatnonzero(
        np.all(np.asarray(witness.implicit_root_edge_keys) == root_edges, axis=(1, 2, 3))
    )
    assert len(rows) == 1
    face = _retarget_implicit_witness(
        witness,
        mesh,
        geometry,
        witness.cell_global_ids,
        rows=(int(rows[0]),),
        entity_ids=witness.entity_global_ids,
    )
    root_map = prepare_adaptive_implicit_root_charts(
        mesh,
        geometry,
        source,
        face,
        frame_id=contract.spatial_id,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )
    assert root_map.status == "certified"
    assert root_map.chart_evidence[0].chart_jacobian_lower > 0.0


def _curved_geometry(
    mesh: CellMesh,
    source: _BoundAdaptiveSourceQuery,
    root_map: AdaptiveImplicitRootChartMap,
    degree: int,
) -> tuple[CellGeometrySpec, SourceBoundaryChartWitness]:
    straight = _straight_geometry(mesh, degree)
    elements, routes, coordinates = straight.resolve(mesh)
    elements_by_block: dict[str, CellGeometryElement] = {}
    routes_by_block: dict[str, np.ndarray] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if not isinstance(block, CellBlock):
            raise TypeError(
                "Implicit chart tests require standard triangular target blocks."
            )
        elements_by_block[block.name] = element
        routes_by_block[block.name] = np.asarray(route, dtype=np.int32)
    values = np.array(coordinates, dtype=np.float64, copy=True)
    block = mesh.blocks[0]
    if not isinstance(block, CellBlock):
        raise TypeError("Implicit chart tests require standard triangular target blocks.")
    element = elements_by_block[block.name]
    if not isinstance(element, FiniteElementSpec):
        raise TypeError("Implicit chart tests require Lagrange coordinate elements.")
    route = np.asarray(routes_by_block[block.name], dtype=np.int64)
    assigned: dict[int, np.ndarray] = {}
    nodal_reference_data = lagrange_reference_node_data(element)
    chart_rows: list[int] = []
    references: list[tuple[ExactScalar, ...]] = []
    geometry_rows: list[int] = []
    for row, cell_row in enumerate(range(block.cell_count)):
        for node in nodal_reference_data:
            if node.exact_point is None:
                raise ValueError(
                    "Implicit chart fixture requires exact Lagrange reference nodes."
                )
            chart_rows.append(row)
            references.append(node.exact_point)
            geometry_rows.append(int(route[cell_row, node.local_dof]))
    points = root_map.evaluate_targets(chart_rows, references)
    for geometry_row, point in zip(geometry_rows, points, strict=True):
        previous = assigned.get(geometry_row)
        if previous is not None:
            np.testing.assert_array_equal(previous.view(np.uint64), point.view(np.uint64))
        assigned[geometry_row] = np.array(point, copy=True)
        values[geometry_row] = point
    result = CellGeometrySpec(elements_by_block, routes_by_block, values)
    return result, _chart_witness(mesh, result, source)


def test_adaptive_implicit_p2_p3_root_charts_and_fidelity_replay() -> None:
    _surface, source, contract, mesh, affine, affine_witness = _adaptive_source()
    from phydrax.meshing.providers._native_implicit import _NativeImplicitBoundaryQuery

    native = _NativeImplicitBoundaryQuery(
        source.query, source.source_id, source.source_revision, source.frame_id
    )
    query_leaves = [
        leaf for leaf in jax.tree.leaves(source.query) if isinstance(leaf, Array)
    ]
    assert query_leaves
    for owner in (source, native):
        owner_leaves = jax.tree.leaves(owner)
        assert all(any(leaf is item for item in owner_leaves) for leaf in query_leaves)

    query_points = np.asarray(((0.9, 0.2, 0.1),))
    surface_entity_key = source_entity_id(
        source.source_id,
        source.source_revision,
        "implicit-surface",
        "zero-set",
    )
    global_query = source.project_boundary(query_points)
    assert isinstance(global_query, _AdaptiveImplicitProjectionResult)
    assert global_query.frame_id == contract.spatial_id
    assert np.asarray(global_query.status)[0] == SourceProjectionStatus.UNRESOLVED
    assert global_query.entity_ids() == (surface_entity_key,)
    assert (
        np.asarray(global_query.distance_lower_bounds)[0]
        <= np.asarray(global_query.residuals)[0]
    )
    assert "globally closest" in global_query.reasons[0]
    scoped_query = source.project_boundary(
        query_points,
        entity_ids=(surface_entity_key,),
        occurrence_paths=((),),
    )
    with pytest.raises(ValueError, match="canonical zero-set entity"):
        source.project_boundary(query_points, entity_ids=(source.source_id,))
    assert isinstance(scoped_query, _AdaptiveImplicitProjectionResult)
    assert np.asarray(scoped_query.status)[0] == SourceProjectionStatus.UNRESOLVED
    assert scoped_query.entity_ids() == (surface_entity_key,)
    raw_entity_witness = _retarget_implicit_witness(
        affine_witness,
        mesh,
        affine,
        affine_witness.cell_global_ids,
        source_entity_ids=(source.source_id,) * len(affine_witness.source_chart_ids),
    )
    with pytest.raises(ValueError):
        validate_source_fidelity_witness(mesh, affine, source, raw_entity_witness)

    affine_root_map = _affine_root_map(mesh, source, affine, affine_witness)
    for degree in (2, 3):
        curved, witness = _curved_geometry(mesh, source, affine_root_map, degree)
        chart_map = prepare_adaptive_implicit_root_charts(
            mesh,
            curved,
            source,
            witness,
            frame_id=contract.spatial_id,
            tolerance=0.5,
            maximum_normal_deviation=np.pi,
        )
        assert chart_map.status == "certified"
        assert chart_map.frame_id == contract.spatial_id
        assert all(item.chart_jacobian_lower > 0.0 for item in chart_map.chart_evidence)
        assert all(
            item.chart_injectivity_margin > 0.0 for item in chart_map.chart_evidence
        )
        assert all(
            item.source_cell_key and len(item.root_edge_keys) == 3
            for item in chart_map.chart_evidence
        )
        rows = tuple(range(len(chart_map.chart_evidence)))
        exact_centroids = chart_map.evaluate_targets(
            rows, [(Fraction(1, 3), Fraction(1, 3))] * len(rows)
        )
        authored_centroids = chart_map.evaluate_targets(
            rows, [(1.0 / 3.0, 1.0 / 3.0)] * len(rows)
        )
        assert np.all(np.isfinite(exact_centroids))
        assert np.all(np.isfinite(authored_centroids))
        np.testing.assert_array_equal(
            chart_map.evaluate_target(0, (Fraction(1, 3), Fraction(1, 3))).view(
                np.uint64
            ),
            exact_centroids[0].view(np.uint64),
        )
        if degree == 2:
            # The prepared map is passed explicitly; certification checks it
            # against its operands instead of preparing the charts again.
            fidelity = certify_adaptive_implicit_charts(
                mesh,
                curved,
                source,
                witness,
                tolerance=0.5,
                maximum_normal_deviation=np.pi,
                root_charts=chart_map,
            )
            assert fidelity.status == "certified"
            assert (
                fidelity.witness is not None
                and fidelity.witness.witness_id == witness.witness_id
            )
            assert fidelity.source_to_mesh_upper <= 0.5
            assert fidelity.mesh_to_source_upper <= 0.5
            assert fidelity.normal_deviation_bounds is not None
            assert (
                certify_source_fidelity(
                    mesh,
                    curved,
                    source,
                    tolerance=0.5,
                    maximum_normal_deviation=np.pi,
                    witness=witness,
                    root_charts=chart_map,
                ).certificate_id
                == fidelity.certificate_id
            )
            with pytest.raises(ValueError, match="prepared for a different"):
                certify_adaptive_implicit_charts(
                    mesh,
                    curved,
                    source,
                    witness,
                    tolerance=0.25,
                    maximum_normal_deviation=np.pi,
                    root_charts=chart_map,
                )


def test_adaptive_implicit_on_surface_p2_sphere_charts_certify() -> None:
    _surface, source, contract, mesh, affine, affine_witness = _adaptive_source(
        smooth=False, authoring_profile=True
    )
    affine_root_map = _affine_root_map(mesh, source, affine, affine_witness)
    curved, witness = _curved_geometry(mesh, source, affine_root_map, 2)
    chart_map = prepare_adaptive_implicit_root_charts(
        mesh, curved, source, witness, frame_id=contract.spatial_id
    )
    assert chart_map.status == "certified"
    assert all(item.status == "certified" for item in chart_map.chart_evidence)
    assert all(item.chart_injectivity_margin > 0.0 for item in chart_map.chart_evidence)


# The slab field c - 1/(1 + x0) saturates: its normal-line slope decays like
# 1/(1 + t)^2, so the root x0 = 1/c - 1 lies far beyond every bracket estimated
# from the slope near the face.
_SATURATING_LEVEL = 1.0e-12


@final
class _SaturatingSlabKernel(GeometryKernel):
    """Smooth level-set slab ``{x0 <= 1/c - 1}`` of ``c - 1/(1 + x0)`` on ``x0 > -1``."""

    def geometry_validity(self, state: DesignState, /) -> GeometryValidityEvidence:
        del state
        return unrestricted_validity(contract_id="saturating-slab")

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return frozenset({GeometryCapability.REGION_QUERY})

    @property
    def field_certificate(self) -> FieldCertificate:
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.EXACT,
            sign_reliability=SignReliability.RELIABLE,
            distance_semantics=DistanceSemantics.LEVEL_SET,
            regularity=FieldRegularity.SMOOTH,
            safe_step_factor=None,
            validity_region="first coordinate greater than -1",
            parameter_differentiable=False,
            provenance=("saturating_slab_level_set",),
        )

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        del state
        return _SATURATING_LEVEL - 1.0 / (1.0 + points[..., 0])

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        del state
        return jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0)), points.shape)

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return jnp.asarray(
            (
                (-1.0, -jnp.inf, -jnp.inf),
                (1.0 / _SATURATING_LEVEL - 1.0, jnp.inf, jnp.inf),
            )
        )

    def measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("The saturating slab has no finite measure.")

    def boundary_measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("The saturating slab has no finite boundary measure.")

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: PRNGKey,
        plan: RejectionSamplingPlan | None = None,
    ) -> NoReturn:
        del state, num_points, key, plan
        raise NotImplementedError("The saturating slab advertises region queries only.")

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> NoReturn:
        del state, num_points, key
        raise NotImplementedError("The saturating slab advertises region queries only.")

    def boundary_atlas(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("The saturating slab advertises region queries only.")


def test_implicit_root_bracket_widens_until_a_far_normal_root_is_signed() -> None:
    _, state = _ParameterCollector().finish()
    geometry = CompiledGeometry(_SaturatingSlabKernel(), state)
    surface = discover_adaptive_implicit_surface(
        geometry,
        domain=np.asarray(((-0.5, -1.0, -1.0), (1.0e23, 1.0, 1.0))),
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.25, root_tolerance=1.0e-8
        ),
        source_id="saturating-slab",
    )
    # A flat piece on x0 = 0, where f = c - 1 and grad f = e0: the normal-line
    # parameter is x0 itself and the root sits at t* = 1/c - 1 = 1e12.
    width = Fraction(1, 64)
    face = source_charts._enclose_face(
        (
            {(0, 0): Fraction(0)},
            {(0, 0): Fraction(0), (1, 0): width},
            {(0, 0): Fraction(0), (0, 1): width},
        )
    )
    hessian = source_charts._prepare_hessian(geometry)
    steps = source_charts._face_piece_steps(
        surface, face, source_charts._STANDARD_TRIANGLE, 1, (0, 1, 2), None, None
    )
    request = next(steps)
    bracket_ends: list[float] = []
    while True:
        # Every bracket round queries the tube, its two endpoint boxes and the
        # two shifted piece centres; the first request is the piece itself.
        if len(request.field) == 5:
            bracket_ends.append(float(request.field[2][1][0]))
        (result,) = source_charts._evaluate_box_requests(
            surface.volume.bounds, hessian, [request]
        )
        assert result is not None
        try:
            request = steps.send(result)
        except StopIteration as stop:
            proof, detail, _ = stop.value
            break
    assert proof is not None, detail
    root = 1.0 / _SATURATING_LEVEL - 1.0
    assert proof.root_parameters[0] <= root <= proof.root_parameters[1]
    # Each widening jumps to the monotone sign point of the decayed slope, yet
    # five are needed; four doublings of the first bracket stop 2e10x short.
    widenings = len(bracket_ends) - 1
    assert widenings >= 5
    assert bracket_ends[-2] < root < bracket_ends[-1]
    assert 2.0**4 * bracket_ends[0] < root


_NEAR_BAND_RADIUS = 0.371
_NEAR_BAND_OFFSET = 0.0093


def _near_band_sphere_face(
    inward: float = 0.0, /
) -> tuple[
    source_charts._FieldEnclosures,
    source_charts.PreparedIntervalFunction,
    AdaptiveImplicitSurface,
    source_charts._FaceEnclosure,
    float,
]:
    """A flat face tangent-offset .0093 off a sphere along its diagonal normal.

    This is the shape of a centred remeshed carrier vertex: its plain
    target-to-root bound converges to about .01004, above a .01 tolerance.
    ``inward`` tilts the far vertices toward the sphere, so the face's largest
    deviation is attained only at its corner vertex.
    """
    from types import SimpleNamespace

    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), _NEAR_BAND_RADIUS).compile()
    normal = np.ones(3) / np.sqrt(3.0)
    corner = (_NEAR_BAND_RADIUS + _NEAR_BAND_OFFSET) * normal
    first = np.asarray((1.0, -1.0, 0.0)) / np.sqrt(2.0)
    second = np.cross(normal, first)
    points = [
        tuple(Fraction(float(value)) for value in point)
        for point in (
            corner,
            corner + 0.1 * first - inward * normal,
            corner + 0.1 * second - inward * normal,
        )
    ]

    def coordinate(axis: int, /) -> source_charts.Polynomial:
        terms = {
            (0, 0): points[0][axis],
            (1, 0): points[1][axis] - points[0][axis],
            (0, 1): points[2][axis] - points[0][axis],
        }
        return {key: value for key, value in terms.items() if value}

    expressions: source_charts.Polynomial3 = (
        coordinate(0),
        coordinate(1),
        coordinate(2),
    )
    surface = cast(
        AdaptiveImplicitSurface,
        SimpleNamespace(
            volume=SimpleNamespace(domain=np.asarray(((-1.0,) * 3, (1.0,) * 3)))
        ),
    )
    deviation = (
        float(np.sqrt(float(sum(value * value for value in points[0]))))
        - _NEAR_BAND_RADIUS
    )
    return (
        adaptive_discovery._FieldEnclosures(geometry, "interval"),
        source_charts._prepare_hessian(geometry),
        surface,
        source_charts._enclose_face(expressions),
        deviation,
    )


def _corner_triangle(level: int, /) -> source_charts.TriangleReference:
    size = Fraction(1, 2**level)
    return ((Fraction(0), Fraction(0)), (size, Fraction(0)), (Fraction(0), size))


def _near_band_face_proof(
    tolerance: float, /, inward: float = 0.0
) -> tuple[tuple[source_charts._PieceProof, ...], str]:
    bounds, hessian, surface, face, _ = _near_band_sphere_face(inward)

    def prove(
        triangles: Sequence[source_charts.TriangleReference],
    ) -> tuple[source_charts._PieceSteps, ...]:
        return source_charts._face_piece_proofs(
            surface,
            face,
            triangles,
            source_orientation=1,
            projected_axes=(0, 1, 2),
            tolerance=tolerance,
            maximum_normal_deviation=None,
        )

    (outcome,) = source_charts._run_faces(
        bounds,
        hessian,
        [source_charts._prove_face_triangles(prove, (_corner_triangle(4),))],
    )
    assert outcome is not None
    return outcome


def _proof_reference_area(pieces: Sequence[source_charts._PieceProof]) -> Fraction:
    """Independent rational area of the certified reference subtriangles."""
    total = Fraction(0)
    for proof in pieces:
        first, second, third = proof.triangle
        total += (
            abs(
                (second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0])
            )
            / 2
        )
    return total


def test_proof_neighborhoods_include_contacts_without_bridging_binary64_gaps() -> None:
    lower = np.asarray(
        (
            (-1.0, -1.0),
            (0.0, 0.0),
            (np.nextafter(1.0, np.inf), 0.0),
            (-0.0, -0.0),
            (0.0, 0.0),
        ),
        dtype=np.float64,
    )
    upper = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (0.0, 0.0), (1.0, 1.0)),
        dtype=np.float64,
    )
    queries = np.asarray(((0.0, -0.0), (1.0, 0.5), (3.0, 3.0)), dtype=np.float64)
    tree = source_charts.prepare_bvh(lower, upper, dtype=np.float64)
    neighborhoods = source_charts._proof_box_neighborhoods(tree, queries, queries)
    np.testing.assert_array_equal(neighborhoods[0], (0, 1, 3, 4))
    np.testing.assert_array_equal(neighborhoods[1], (1, 4))
    np.testing.assert_array_equal(neighborhoods[2], np.empty(0, dtype=np.int64))


def test_transversal_neighborhoods_preserve_collision_and_chord_enclosures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_charts, "_PROOF_WORKSET_SIZE", 2)
    bounds, hessian, surface, face, _ = _near_band_sphere_face()

    def prove(
        triangles: Sequence[source_charts.TriangleReference],
    ) -> tuple[source_charts._PieceSteps, ...]:
        return source_charts._face_piece_proofs(
            surface,
            face,
            triangles,
            source_orientation=1,
            projected_axes=(0, 1, 2),
            tolerance=0.01,
            maximum_normal_deviation=None,
        )

    (outcome,) = source_charts._run_faces(
        bounds,
        hessian,
        [
            source_charts._prove_face_triangles(
                prove,
                tuple(_corner_triangle(level) for level in (7, 4, 8, 5, 6)),
            )
        ],
    )
    assert outcome is not None
    pieces, detail = outcome
    assert not detail
    corners = np.asarray(
        [
            [[float(value) for value in point] for point in piece.triangle]
            for piece in pieces
        ],
        dtype=np.float64,
    )
    reference_low, reference_high = corners.min(axis=1), corners.max(axis=1)
    observed: list[int] = []
    for (
        start,
        neighborhoods,
        midpoint_rows,
        hessian_boxes,
    ) in source_charts._transversal_neighborhoods(pieces):
        for offset, (rows, midpoint, (low, high)) in enumerate(
            zip(neighborhoods, midpoint_rows, hessian_boxes, strict=True)
        ):
            index = start + offset
            observed.append(index)
            expected = [
                other
                for other, piece in enumerate(pieces)
                if all(
                    first[0] <= second[1] and second[0] <= first[1]
                    for first, second in zip(
                        pieces[index].root_box, piece.root_box, strict=True
                    )
                )
            ]
            np.testing.assert_array_equal(rows, expected)
            target = np.asarray(
                [pieces[row].target_box for row in expected], dtype=np.float64
            )
            np.testing.assert_array_equal(
                low.view(np.uint64), target[:, :, 0].min(axis=0).view(np.uint64)
            )
            np.testing.assert_array_equal(
                high.view(np.uint64), target[:, :, 1].max(axis=0).view(np.uint64)
            )
            region_low = (
                reference_low[expected].min(axis=0) - 4.0 * np.finfo(np.float64).eps
            )
            region_high = (
                reference_high[expected].max(axis=0) + 4.0 * np.finfo(np.float64).eps
            )
            expected_midpoints = [
                other
                for other in range(len(pieces))
                if all(
                    reference_low[other, axis] <= region_high[axis]
                    and region_low[axis] <= reference_high[other, axis]
                    for axis in range(2)
                )
            ]
            np.testing.assert_array_equal(midpoint, expected_midpoints)
    assert observed == list(range(len(pieces)))


def test_chart_worksets_preserve_face_coverage_and_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_charts, "_PROOF_WORKSET_SIZE", 2)
    bounds, hessian, surface, face, deviation = _near_band_sphere_face()
    levels = (7, 4, 8, 5, 6)

    def prove(
        triangles: Sequence[source_charts.TriangleReference],
    ) -> tuple[source_charts._PieceSteps, ...]:
        return source_charts._face_piece_proofs(
            surface,
            face,
            triangles,
            source_orientation=1,
            projected_axes=(0, 1, 2),
            tolerance=0.01,
            maximum_normal_deviation=None,
        )

    outcomes = source_charts._run_faces(
        bounds,
        hessian,
        [
            source_charts._prove_face_triangles(prove, (_corner_triangle(level),))
            for level in levels
        ],
    )
    for level, outcome in zip(levels, outcomes, strict=True):
        assert outcome is not None, level
        pieces, detail = outcome
        assert detail == "", (level, detail)
        assert _proof_reference_area(pieces) == Fraction(1, 2 ** (2 * level + 1))
        assert min(piece.target_to_source[0] for piece in pieces) <= deviation
        assert deviation <= max(piece.target_to_source[1] for piece in pieces) <= 0.01
        assert all(piece.chart_jacobian_lower > 0.0 for piece in pieces)


def test_chart_worksets_preserve_standalone_query_tail_and_crease_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_charts, "_PROOF_WORKSET_SIZE", 2)
    geometry = phx.geometry.Union(
        (
            phx.geometry.Sphere((-0.5, 0.0, 0.0), 1.0, feature_id="left"),
            phx.geometry.Sphere((0.5, 0.0, 0.0), 1.0, feature_id="right"),
        ),
        feature_id="query-crease",
    ).compile()
    bounds = adaptive_discovery._FieldEnclosures(geometry, "interval")
    hessian = source_charts._prepare_hessian(geometry)
    points = np.asarray(
        (
            (-0.9, 0.2, 0.1),
            (0.2, 0.2, 0.1),
            (0.0, 0.2, 0.1),
            (0.7, 0.2, 0.1),
            (-0.15, 0.2, 0.1),
        ),
        dtype=np.float64,
    )
    queries = tuple(
        source_charts._BoxRequests(field=((point, point),)) for point in points
    )
    _, replies = source_charts._run_piece_steps(
        bounds, hessian, (), queries, branches=True
    )
    expected = np.asarray(
        geometry.kernel.boundary_field(
            geometry.state, jnp.asarray(points, dtype=jnp.float64)
        )
    )
    for index, (reply, value) in enumerate(zip(replies, expected, strict=True)):
        if index == 2:
            assert reply is None
        else:
            assert reply is not None, index
            lower, upper = reply[0][0][0]
            assert lower <= value <= upper, index


def test_near_band_root_chart_plain_bound_does_not_converge_below_tolerance() -> None:
    bounds, hessian, surface, face, deviation = _near_band_sphere_face()
    (steps,) = source_charts._face_piece_proofs(
        surface,
        face,
        (_corner_triangle(15),),
        source_orientation=1,
        projected_axes=(0, 1, 2),
        tolerance=None,
        maximum_normal_deviation=None,
    )
    ((proof, detail, _),), _ = source_charts._run_piece_steps(bounds, hessian, (steps,))
    assert proof is not None, detail
    # The plain step about t = 0 keeps a slope enclosure over the whole normal
    # tube, so even a 2^-15 piece stays above the .01 tolerance.
    assert proof.target_to_source[0] <= deviation < 0.01 < proof.target_to_source[1]


def test_near_band_root_chart_certifies_by_newton_narrowing() -> None:
    pieces, detail = _near_band_face_proof(0.01)
    assert detail == ""
    _, _, _, _, deviation = _near_band_sphere_face()
    assert max(piece.target_to_source[1] for piece in pieces) <= 0.01
    assert max(piece.target_to_source[0] for piece in pieces) <= deviation + 1.0e-4


def test_near_band_root_chart_beyond_tolerance_refuses_as_violated() -> None:
    pieces, _ = _near_band_face_proof(0.0092)
    _, _, _, _, deviation = _near_band_sphere_face()
    assert deviation > 0.0092
    assert _proof_reference_area(pieces) == 0


def test_near_band_root_chart_tie_refuses_at_binary64_resolution() -> None:
    # The tolerance is the corner's own distance up to rounding: no finite
    # subdivision decides it, so refinement narrows to binary64 neighbours
    # around the corner and refuses there, keeping a bounded frontier.
    _, _, _, _, deviation = _near_band_sphere_face(0.05)
    pieces, _ = _near_band_face_proof(deviation, 0.05)
    assert _proof_reference_area(pieces) < Fraction(1, 512)


@pytest.mark.parametrize(
    "tolerance", (0.010001, 0.009999), ids=("inside-bound", "outside-bound")
)
def test_stationary_root_distance_encloses_independent_sphere_extrema(
    tolerance: float,
) -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile()
    bounds = adaptive_discovery._FieldEnclosures(geometry, "interval")
    hessian = source_charts._prepare_hessian(geometry)
    # A small plane patch whose center is the distance minimum. The maximum
    # is at either far vertex, independently of the interval proof.
    width = Fraction(1, 10000)
    normal = (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3))
    tangent_u = (Fraction(2, 3), Fraction(1, 3), Fraction(-2, 3))
    tangent_v = (Fraction(-2, 3), Fraction(2, 3), Fraction(-1, 3))

    def coordinate(axis: int, /) -> source_charts.Polynomial:
        return {
            (0, 0): Fraction(101, 100) * normal[axis]
            - width * (tangent_u[axis] + tangent_v[axis]),
            (1, 0): 3 * width * tangent_u[axis],
            (0, 1): 3 * width * tangent_v[axis],
        }

    face = source_charts._enclose_face((coordinate(0), coordinate(1), coordinate(2)))
    from types import SimpleNamespace

    surface = cast(
        AdaptiveImplicitSurface,
        SimpleNamespace(
            volume=SimpleNamespace(
                domain=np.asarray(((-2.0,) * 3, (2.0,) * 3), dtype=np.float64)
            )
        ),
    )
    steps = source_charts._face_piece_proofs(
        surface,
        face,
        (source_charts._STANDARD_TRIANGLE,),
        source_orientation=1,
        projected_axes=(0, 1, 2),
        tolerance=tolerance,
        maximum_normal_deviation=None,
    )
    ((proof, detail, refine),), _ = source_charts._run_piece_steps(bounds, hessian, steps)
    if tolerance < 0.01:
        assert proof is None
        assert not refine
    else:
        assert proof is not None, detail
        # Decimal sqrt is independent of the source field's JAX/interval route.
        from decimal import Decimal, localcontext

        with localcontext() as context:
            context.prec = 80
            maximum = (
                Decimal(101) ** 2 / Decimal(100) ** 2 + Decimal(5) / Decimal(10000) ** 2
            ).sqrt() - Decimal(1)
            assert Decimal.from_float(proof.target_to_source[0]) <= Decimal("0.01")
            assert Decimal.from_float(proof.target_to_source[1]) >= maximum
        assert proof.target_to_source[1] <= tolerance


def test_local_reverse_cover_proves_complete_meshes_and_refutes_missing_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface, source, _contract, mesh, geometry, witness = _adaptive_source()
    charts = prepare_adaptive_implicit_root_charts(
        mesh, geometry, source, witness, tolerance=0.5, maximum_normal_deviation=np.pi
    )
    faces = charts._faces
    tolerance = 0.05

    maps = source_charts._face_maps(faces)

    def original_blocks(
        face_maps: source_charts._FaceMaps,
        low: np.ndarray,
        high: np.ndarray,
        reach: float,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if not len(low):
            return
        reached = source_charts.prepare_bvh(
            np.nextafter(low - reach, -np.inf),
            np.nextafter(high + reach, np.inf),
            dtype=np.float64,
        )
        for face_rows, box_rows in source_charts.bvh_overlap_pair_blocks(
            face_maps.bvh, reached, include_touching=True
        ):
            yield (
                np.asarray(box_rows, dtype=np.int64),
                np.asarray(face_rows, dtype=np.int64),
            )

    # Compare real native/host BVH streams and enclosed face controls across
    # a small query and a query requiring full plus final-partial pair blocks.
    point = np.asarray(surface.boundary_witnesses, dtype=np.float64)[:1]
    for count in (3, source_charts._REVERSE_COVER_ROUND + 1):
        points = np.repeat(point, count, axis=0)
        old = list(original_blocks(maps, points, points, tolerance))
        new = list(source_charts._box_face_pair_blocks(maps, points, points, tolerance))
        assert old and new
        for column in (0, 1):
            assert np.array_equal(
                np.concatenate([block[column] for block in old]),
                np.concatenate([block[column] for block in new]),
            )
        if count > 3:
            assert any(
                len(boxes) == source_charts._REVERSE_COVER_ROUND for boxes, _ in new
            )
            assert 0 < len(new[-1][0]) < source_charts._REVERSE_COVER_ROUND

        def controls(
            blocks: list[tuple[np.ndarray, np.ndarray]],
        ) -> tuple[np.ndarray, ...]:
            rows = []
            for box_rows, face_rows in blocks:
                first, second = source_charts._nearest_face_parameters(
                    maps, face_rows, points[box_rows]
                )
                low, high = source_charts._face_point_enclosures(
                    maps, face_rows, first, second
                )
                rows.append((first, second, low, high))
            return tuple(np.concatenate([row[i] for row in rows]) for i in range(4))

        for previous, current in zip(controls(old), controls(new), strict=True):
            assert np.array_equal(previous.view(np.uint64), current.view(np.uint64))

    with monkeypatch.context() as patch:
        patch.setattr(source_charts, "_box_face_pair_blocks", original_blocks)
        old_lower, old_upper = source_charts._source_to_mesh_bounds(
            surface, maps, tolerance
        )
    lower, upper = source_charts._source_to_mesh_bounds(surface, maps, tolerance)
    assert (lower, upper) == (old_lower, old_upper)
    assert 0.0 <= lower <= upper <= tolerance

    # Without the faces on the x > 0 side, the zero set there lies about one
    # superquadric radius from every face, so a zero-containing box refutes it.
    kept = [
        face
        for face in faces
        if source_charts._enclosed_box(
            face.face_enclosure, source_charts._STANDARD_TRIANGLE
        )[1][0]
        < 0.0
    ]
    assert 0 < len(kept) < len(faces)
    lower, upper = source_charts._source_to_mesh_bounds(
        surface, source_charts._face_maps(kept), tolerance
    )
    assert lower > tolerance
    assert upper == np.inf


def test_adaptive_implicit_fidelity_retains_scoped_target_extent() -> None:
    _surface, source, _contract, mesh, _affine, witness = _adaptive_source()
    target_mesh, target_geometry, scoped_witness = _extend_implicit_target_mesh(
        mesh, witness
    )
    target_block = target_mesh.blocks[0]
    assert isinstance(target_block, CellBlock)
    target_entities = np.asarray(target_block.global_ids, dtype=np.int64)
    validate_source_fidelity_witness(target_mesh, target_geometry, source, scoped_witness)

    prepared = prepare_adaptive_implicit_root_charts(
        target_mesh,
        target_geometry,
        source,
        scoped_witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )
    assert prepared.status == "certified"
    certificate = certify_adaptive_implicit_charts(
        target_mesh,
        target_geometry,
        source,
        scoped_witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )
    assert certificate.status == "certified"
    assert isinstance(certificate.witness, SourceBoundaryChartWitness)
    assert np.array_equal(
        certificate.witness.entity_global_ids, scoped_witness.entity_global_ids
    )
    assert not set(int(value) for value in target_entities) <= set(
        int(value) for value in scoped_witness.entity_global_ids
    )

    omitted = _retarget_implicit_witness(
        scoped_witness,
        target_mesh,
        target_geometry,
        np.asarray(scoped_witness.cell_global_ids, dtype=np.int64),
        rows=tuple(range(scoped_witness.entity_global_ids.size - 1)),
    )
    with pytest.raises(ValueError, match="omits possible source scaffold triangles"):
        validate_source_fidelity_witness(target_mesh, target_geometry, source, omitted)

    repeated_entities = np.array(scoped_witness.entity_global_ids, copy=True)
    repeated_entities[1] = repeated_entities[0]
    with pytest.raises(ValueError):
        repeated = _retarget_implicit_witness(
            scoped_witness,
            target_mesh,
            target_geometry,
            scoped_witness.cell_global_ids,
            entity_ids=repeated_entities,
        )
        validate_source_fidelity_witness(target_mesh, target_geometry, source, repeated)


def _counting_build_face(monkeypatch: pytest.MonkeyPatch, /) -> list[int]:
    """Record the rows the chart prover is asked to prove."""
    proved: list[int] = []
    build_face = source_charts._build_face

    def counting(setup, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        proved.append(setup.row)
        return build_face(setup, *args, **kwargs)

    monkeypatch.setattr(source_charts, "_build_face", counting)
    return proved


@pytest.mark.parametrize("movement", ("none", "one", "all"))
def test_scoped_fidelity_reuses_only_full_identity_row_proof(
    monkeypatch: pytest.MonkeyPatch, movement: str
) -> None:
    from phydrax.geometry.implicit._chart_restriction import ValidatedImplicitChartParent
    from phydrax.meshing import _certification as certification_owner
    from phydrax.meshing._certification_inputs import MeshCertificationInputs
    from phydrax.meshing._certification_schedule import MeshCertificationSchedule

    _, source, _, mesh, geometry, witness = _adaptive_source()
    parent = certify_source_fidelity(
        mesh, geometry, source, witness=witness, tolerance=0.5
    )
    assert parent.status == "certified" and parent.chart_rows is not None
    validated_parent = ValidatedImplicitChartParent(mesh, geometry, source, parent)
    premise = ImplicitChartPremise(mesh, geometry, parent, None)
    target, target_geometry, target_witness = mesh, geometry, witness
    if movement != "none":
        points = np.array(mesh.coordinates, dtype=np.float64, copy=True)
        if movement == "one":
            points[0] *= 1.0 + 2.0**-12
        else:
            points *= 1.0 + 2.0**-12
        block = mesh.blocks[0]
        assert isinstance(block, CellBlock)
        target = CellMesh.from_triangles(
            points, np.asarray(block.vertices, dtype=np.int32)
        )
        target_geometry = CellGeometrySpec.affine(target)
        target_witness = _retarget_implicit_witness(
            witness, target, target_geometry, np.asarray(witness.cell_global_ids)
        )
    inputs = MeshCertificationInputs(
        target,
        target_geometry,
        MeshCertificationSchedule("surface"),
        source_id=source.source_id,
        source_revision=source.source_revision,
        source=source,
        fidelity_tolerance=0.5,
        fidelity_witness=target_witness,
        scoped_fidelity=((source, target_witness, 0.5),),
        protected_features=(),
    )
    direct = certification_owner.MeshCertificationReport(inputs)
    calls = []
    certify = certification_owner.certify_source_fidelity
    proved = _counting_build_face(monkeypatch)

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(1)
        return certify(*args, **kwargs)

    monkeypatch.setattr(certification_owner, "certify_source_fidelity", counted)
    reused = certification_owner.MeshCertificationReport(
        inputs, fidelity_premise=premise, validated_chart_parent=validated_parent
    )
    assert len(calls) == 1
    if movement == "none":
        assert not proved
    elif movement == "one":
        assert 0 < len(proved) < len(target_witness.source_chart_ids)
    else:
        assert len(proved) == len(target_witness.source_chart_ids)
    assert reused.fidelity is reused.scoped_fidelity[0]
    assert reused.fidelity is not None and direct.fidelity is not None
    assert reused.fidelity.chart_rows is not None
    assert reused.fidelity.chart_restriction is None
    assert reused.fidelity.chart_premise is None
    assert reused.fidelity.certificate_id == direct.fidelity.certificate_id
    assert reused.report_id == direct.report_id


def test_parent_chart_rows_are_reused_by_identity_and_reproduce_the_full_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _surface, source, _contract, mesh, geometry, witness = _adaptive_source()
    from phydrax.geometry._source_fidelity_certificate import SourceFidelityCertificate

    parent = SourceFidelityCertificate(
        mesh,
        geometry,
        source,
        witness=witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )
    assert parent.status == "certified"
    assert parent.chart_rows is not None
    assert bool(np.all(parent.chart_rows.reusable))
    premise = ImplicitChartPremise(mesh, geometry, parent, None)
    from phydrax.geometry.implicit._chart_restriction import ValidatedImplicitChartParent

    replay_count = [0]
    original_revalidate = SourceFidelityCertificate.revalidate

    def counted_revalidate(
        certificate: SourceFidelityCertificate,
        bound_mesh: CellMesh,
        bound_geometry: CellGeometrySpec,
        bound_source: SourceBoundaryQuery,
        /,
    ) -> None:
        replay_count[0] += 1
        original_revalidate(certificate, bound_mesh, bound_geometry, bound_source)

    monkeypatch.setattr(SourceFidelityCertificate, "revalidate", counted_revalidate)
    validated_parent = ValidatedImplicitChartParent(mesh, geometry, source, parent)
    assert replay_count == [1]

    # Unchanged rows of a new mesh: every row is checked by identity.
    target_mesh, target_geometry, target_witness = _extend_implicit_target_mesh(
        mesh, witness
    )
    full = certify_adaptive_implicit_charts(
        target_mesh,
        target_geometry,
        source,
        target_witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )
    proved = _counting_build_face(monkeypatch)
    reused = certify_adaptive_implicit_charts(
        target_mesh,
        target_geometry,
        source,
        target_witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
        chart_parent=premise,
        validated_chart_parent=validated_parent,
    )
    assert proved == []
    assert reused.certificate_id == full.certificate_id
    assert reused.chart_restriction is None

    # A moved vertex changes exactly its incident rows; only those are proved.
    points = np.array(mesh.coordinates, dtype=np.float64, copy=True)
    points[0] *= 1.0 + 2.0**-12
    block = mesh.blocks[0]
    assert isinstance(block, CellBlock)
    moved_mesh = CellMesh.from_triangles(
        points, np.asarray(block.vertices, dtype=np.int32)
    )
    moved_geometry = CellGeometrySpec.affine(moved_mesh)
    moved_witness = _retarget_implicit_witness(
        witness,
        moved_mesh,
        moved_geometry,
        np.asarray(witness.cell_global_ids, dtype=np.int64),
    )
    moved_full = certify_adaptive_implicit_charts(
        moved_mesh,
        moved_geometry,
        source,
        moved_witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )
    proved.clear()
    moved_reused = certify_adaptive_implicit_charts(
        moved_mesh,
        moved_geometry,
        source,
        moved_witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
        chart_parent=premise,
        validated_chart_parent=validated_parent,
    )
    incident = sorted(
        row
        for row, cell in enumerate(np.asarray(moved_witness.cell_global_ids).tolist())
        if 0 in np.asarray(block.vertices)[list(block.global_ids).index(cell)]
    )
    assert incident and sorted(proved) == incident
    assert moved_reused.certificate_id == moved_full.certificate_id

    # Negative: other controls name other proofs, so nothing is reused.
    proved.clear()
    other = certify_adaptive_implicit_charts(
        target_mesh,
        target_geometry,
        source,
        target_witness,
        tolerance=0.25,
        maximum_normal_deviation=np.pi,
        chart_parent=premise,
        validated_chart_parent=validated_parent,
    )
    assert len(proved) == len(target_witness.source_chart_ids)
    assert replay_count == [1]
    assert other.certificate_id != full.certificate_id

    # Negative: altered retained evidence is refused, never reused.
    rows = parent.chart_rows
    tampered_values = np.array(rows.values, copy=True)
    tampered_values[0, 2] = 0.0
    tampered_rows = ImplicitChartRows(rows.keys, rows.reusable, tampered_values)
    object.__setattr__(tampered_rows, "rows_id", rows.rows_id)
    object.__setattr__(parent, "chart_rows", tampered_rows)
    with pytest.raises(ValueError, match="identity"):
        certify_adaptive_implicit_charts(
            target_mesh,
            target_geometry,
            source,
            target_witness,
            tolerance=0.5,
            maximum_normal_deviation=np.pi,
            chart_parent=premise,
        )
    # A forged supplemental row identity must also fail independent replay:
    # row values are stored evidence, so the forgery moves a certified
    # target-to-root enclosure wholly away from its reproof.
    row = int(np.flatnonzero(rows.reusable)[0])
    forged_values = np.array(rows.values, copy=True)
    forged_values[row, 1:3] = 2.0 * rows.values[row, 2] + 1.0
    forged_rows = ImplicitChartRows(rows.keys, rows.reusable, forged_values)
    object.__setattr__(parent, "chart_rows", forged_rows)
    with pytest.raises(ValueError, match="disjoint"):
        parent.revalidate(mesh, geometry, source)


def test_adaptive_implicit_root_chart_requires_smooth_or_profile_regular_source() -> None:
    sphere_surface, sphere_source, contract, mesh, geometry, witness = _adaptive_source(
        smooth=False,
        authoring_profile=True,
    )
    unbound = AdaptiveImplicitBoundarySource(sphere_surface)
    with pytest.raises(ValueError, match="coordinate frame"):
        prepare_adaptive_implicit_root_charts(mesh, geometry, unbound, witness)

    root_map = prepare_adaptive_implicit_root_charts(
        mesh, geometry, sphere_source, witness
    )
    assert root_map.status == "certified"
    unprofiled_sphere = _BoundAdaptiveSourceQuery(
        sphere_source.query, contract.spatial_id
    )
    # Since the non-smooth root charts (ImplicitNonSmoothSources) a field that
    # is neither smooth nor exactly profiled is no longer refused: each face
    # proves smoothness box by box from its interval program's branch
    # decisions. This sphere is smooth on every face's boxes, so every face
    # keeps its root chart.
    unprofiled = prepare_adaptive_implicit_root_charts(
        mesh, geometry, unprofiled_sphere, witness
    )
    assert unprofiled.status == "certified"
    assert {evidence.bound for evidence in unprofiled.chart_evidence} == {"root_chart"}

    cube_geometry = phx.geometry.Box(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), feature_id="cube-kink"
    ).compile()
    cube_surface = discover_adaptive_implicit_surface(
        cube_geometry,
        domain=np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))),
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.25,
            root_tolerance=1.0e-6,
            allow_nonsmooth_field=True,
        ),
        source_id="cube-kink",
    )
    cover = cube_surface.cover
    if cover is None or not cover.established:
        raise AssertionError(
            "Cube SDF discovery must retain its complete interval cover."
        )
    possible_zero_boxes = cover.boxes[cover.possible_zero_mask]
    edge_point = np.asarray((0.5, 0.5, 0.0), dtype=np.float64)
    assert np.any(
        np.all(possible_zero_boxes[:, 0] <= edge_point, axis=1)
        & np.all(edge_point <= possible_zero_boxes[:, 1], axis=1)
    )
    cube_source = AdaptiveImplicitBoundarySource(
        cube_surface, frame_id=contract.spatial_id
    )
    # A cube's possible-zero boxes meet its kinked edges; the root-chart route
    # admits it in branch mode, and a witness of another source is refused.
    with pytest.raises(ValueError):
        prepare_adaptive_implicit_root_charts(mesh, geometry, cube_source, witness)


def test_adaptive_implicit_incomplete_root_ancestry_is_rejected() -> None:
    _surface, source, _contract, mesh, geometry, witness = _adaptive_source()
    bad_edges = np.array(witness.implicit_root_edge_keys, copy=True)
    bad_edges[0, 0, 0, 0] += 1
    broken = SourceBoundaryChartWitness(
        mesh_id=witness.mesh_id,
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        chart_dimension=witness.chart_dimension,
        reference_dimension=witness.reference_dimension,
        cell_global_ids=witness.cell_global_ids,
        entity_global_ids=witness.entity_global_ids,
        local_entity_indices=witness.local_entity_indices,
        target_reference_vertices=witness.target_reference_vertices,
        target_reference_rational_tokens=witness.target_reference_rational_tokens,
        source_chart_ids=witness.source_chart_ids,
        source_entity_ids=witness.source_entity_ids,
        occurrence_paths=witness.occurrence_paths,
        source_parameters=witness.source_parameters,
        source_parameter_rational_tokens=witness.source_parameter_rational_tokens,
        orientations=witness.orientations,
        implicit_cell_keys=witness.implicit_cell_keys,
        implicit_root_edge_keys=bad_edges,
    )
    with pytest.raises(ValueError, match="scaffold triangle|root-cell|root-triangle"):
        prepare_adaptive_implicit_root_charts(mesh, geometry, source, broken)


def test_implicit_witness_refuses_late_duplicate_source_triangle() -> None:
    surface, source, _contract, mesh, geometry, witness = _adaptive_source()
    scaffold = surface.scaffold
    cells = np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)
    edges = np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64)
    keys = np.asarray(scaffold.cell_indices, dtype=np.int64)[cells]
    roots = np.asarray(scaffold.root_edge_ids, dtype=np.int64)[edges]
    matching = np.flatnonzero(
        np.all(keys == np.asarray(witness.implicit_cell_keys)[0], axis=1)
        & np.all(roots == np.asarray(witness.implicit_root_edge_keys)[0], axis=(1, 2, 3))
    )
    assert matching.size == 1
    row = int(matching[0])
    filler = (row + 1) % cells.size
    forged_cells = np.concatenate(
        (cells, np.full(8193, cells[filler], dtype=np.int64), cells[row : row + 1])
    )
    forged_edges = np.concatenate(
        (edges, np.repeat(edges[filler : filler + 1], 8193, axis=0), edges[row : row + 1])
    )
    forged = eqx.tree_at(
        lambda value: (
            value.query.surface.scaffold.triangle_cell_rows,
            value.query.surface.scaffold.triangle_root_edge_rows,
        ),
        source,
        (forged_cells, forged_edges),
    )
    selected = _retarget_implicit_witness(
        witness, mesh, geometry, witness.cell_global_ids, rows=(0,)
    )
    with pytest.raises(ValueError, match="duplicates an exact scaffold triangle"):
        validate_source_fidelity_witness(mesh, geometry, forged, selected)


def test_singular_implicit_target_map_retains_unresolved_fidelity_evidence() -> None:
    _surface, source, _contract, mesh, affine, _witness = _adaptive_source()
    elements, routes, coordinates = affine.resolve(mesh)
    block = mesh.blocks[0]
    if not isinstance(block, CellBlock):
        raise TypeError("Implicit chart tests require standard triangular target blocks.")
    singular = CellGeometrySpec(
        {block.name: elements[0]},
        {block.name: routes[0]},
        np.zeros_like(coordinates, dtype=np.float64),
    )
    singular_witness = _chart_witness(mesh, singular, source)

    certificate = certify_adaptive_implicit_charts(
        mesh,
        singular,
        source,
        singular_witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )

    assert certificate.status == "unresolved"
    assert certificate.witness is not None
    assert certificate.witness.witness_id == singular_witness.witness_id
    assert any(
        finding.check == "implicit_root_chart" and finding.status == "unresolved"
        for finding in certificate.findings
    )


_UNIT_DOMAIN = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=np.float64)


def _singular_issue_count(surface: AdaptiveImplicitSurface, /) -> int:
    issues = np.asarray(surface.evidence.unresolved_issues, dtype=np.int64)
    return int(np.count_nonzero(issues & int(AdaptiveImplicitBoxIssue.SINGULAR)))


def test_adaptive_implicit_discovery_reanchors_a_lattice_vertex_on_a_crossing() -> None:
    # The superquadric's y tip lies on a vertex of the declared lattice; the
    # field changes sign around it, so the vertex root is a lattice coincidence.
    geometry = phx.geometry.Superquadric((0.0, 0.0, 0.0), (0.45, 0.50, 0.55)).compile()
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.12, root_tolerance=1.0e-8
    )
    surface = discover_adaptive_implicit_surface(
        geometry, domain=_UNIT_DOMAIN, policy=policy, source_id="reanchored-tip"
    )

    assert surface.evidence.lattice_anchor > 0
    assert surface.mesh is not None
    assert _singular_issue_count(surface) == 0
    assert not np.any(np.asarray(surface.scaffold.corner_signs) == 0)
    np.testing.assert_array_equal(surface.evidence.declared_domain, _UNIT_DOMAIN)
    anchored = np.asarray(surface.scaffold.domain, dtype=np.float64)
    assert np.all(anchored[0] < _UNIT_DOMAIN[0])
    assert np.all(anchored[1] > _UNIT_DOMAIN[1])
    cell_diagonal = float(
        np.linalg.norm((anchored[1] - anchored[0]) / 2.0**surface.scaffold.target_level)
    )
    assert cell_diagonal <= policy.relative_cell_diameter * float(
        np.linalg.norm(_UNIT_DOMAIN[1] - _UNIT_DOMAIN[0])
    )


def test_adaptive_implicit_discovery_refuses_a_field_singular_cell_on_the_first_lattice() -> (
    None
):
    # A bubble far below the cell size is a possible zero without a corner sign
    # change or a nonvanishing gradient component on every lattice: refused
    # without trying re-anchored lattices.
    geometry = phx.geometry.Sphere((0.013, 0.017, 0.019), 1.0e-4).compile()
    surface = discover_adaptive_implicit_surface(
        geometry,
        domain=_UNIT_DOMAIN,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.12, root_tolerance=1.0e-8
        ),
        source_id="subresolution-bubble",
    )

    assert surface.evidence.lattice_anchor == 0
    assert surface.mesh is None
    assert _singular_issue_count(surface) > 0


def test_adaptive_implicit_discovery_keeps_the_last_refusal_when_anchors_are_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With a single half-cell anchor, the plane x = 0 is a vertex plane of both
    # the declared and the re-anchored lattice; a box face on it keeps a vertex
    # on a sign crossing at every anchor.
    monkeypatch.setattr(adaptive_discovery, "_LATTICE_ANCHORS", ((0.5, 0.5, 0.5),))
    geometry = phx.geometry.Box(
        (0.25, 0.03, 0.04), (0.5, 0.6, 0.7), feature_id="lattice-face"
    ).compile()
    surface = discover_adaptive_implicit_surface(
        geometry,
        domain=_UNIT_DOMAIN,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.25,
            root_tolerance=1.0e-6,
            allow_nonsmooth_field=True,
        ),
        source_id="lattice-face",
    )

    assert surface.evidence.lattice_anchor == 1
    assert surface.mesh is None
    assert _singular_issue_count(surface) > 0
    assert np.any(np.asarray(surface.scaffold.corner_signs) == 0)


def test_restricted_implicit_chart_bounds_agree_with_independent_full_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._cell_geometry_transition_types import (
        _reference_map_groups,
    )
    from phydrax.geometry._source_fidelity import certify_source_fidelity
    from phydrax.geometry.implicit._chart_restriction import (
        ImplicitChartPremise,
        ValidatedImplicitChartParent,
    )
    from phydrax.meshing._adaptation import (
        _retarget_source_boundary_chart_witness,
        execute_mesh_adaptation,
        MarkedMeshAdaptation,
        MeshAdaptationPolicy,
        MeshAdaptationRoute,
        prepare_mesh_adaptation,
    )
    from phydrax.meshing._bisection import BisectionCompatibility
    from phydrax.meshing._canonical import certify_cell_mesh

    _, query, contract, mesh, geometry, witness = _adaptive_source()
    source = certify_cell_mesh(mesh, contract, geometry=geometry)
    parent = certify_source_fidelity(
        mesh,
        geometry,
        query,
        witness=witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(np.asarray(mesh.entity_set(2).entity_ids)),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
            ),
        )
    )
    assert refined.transition is not None
    assert refined.lineage is not None
    transition = refined.transition.geometry_transition
    assert transition is not None and transition.reference_map is not None
    target = refined.target
    target_witness = _retarget_source_boundary_chart_witness(
        source,
        target.mesh,
        target.geometry,
        refined.lineage,
        witness,
        reference_maps=_reference_map_groups(transition.reference_map),
    )
    inherited = certify_source_fidelity(
        target.mesh,
        target.geometry,
        query,
        witness=target_witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
        chart_parent=ImplicitChartPremise(mesh, geometry, parent, transition),
    )
    full = certify_source_fidelity(
        target.mesh,
        target.geometry,
        query,
        witness=target_witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
    )
    assert inherited.chart_restriction is not None
    assert full.chart_restriction is None
    assert inherited.status == full.status == "certified"
    assert inherited.certificate_id != full.certificate_id
    assert inherited.mesh_to_source_lower <= full.mesh_to_source_upper
    assert full.mesh_to_source_lower <= inherited.mesh_to_source_upper
    assert np.all(
        np.asarray(full.deviation_bounds)[:, 0]
        <= np.asarray(inherited.deviation_bounds)[:, 1]
    )
    validated = ValidatedImplicitChartParent(
        target.mesh, target.geometry, query, inherited
    )
    ancestor = validated.ancestor
    assert ancestor is not None and ancestor.certificate is parent
    assert ancestor.certificate.chart_rows is not None
    proved = _counting_build_face(monkeypatch)
    restored = certify_source_fidelity(
        mesh,
        geometry,
        query,
        witness=witness,
        tolerance=0.5,
        maximum_normal_deviation=np.pi,
        chart_parent=ImplicitChartPremise(target.mesh, target.geometry, inherited, None),
        validated_chart_parent=validated,
    )
    assert not proved
    assert restored.certificate_id == parent.certificate_id

    # A nested restriction is not the direct proof and cannot stand in for
    # an otherwise identical scoped request.
    from phydrax.meshing._certification import MeshCertificationReport
    from phydrax.meshing._certification_inputs import MeshCertificationInputs
    from phydrax.meshing._certification_schedule import MeshCertificationSchedule

    unrestricted_parent = certify_source_fidelity(
        mesh, geometry, query, witness=witness, tolerance=0.5
    )
    scoped_inputs = MeshCertificationInputs(
        target.mesh,
        target.geometry,
        MeshCertificationSchedule("surface"),
        source_id=query.source_id,
        source_revision=query.source_revision,
        source=query,
        fidelity_tolerance=0.5,
        fidelity_witness=target_witness,
        scoped_fidelity=((query, target_witness, 0.5),),
        protected_features=(),
    )
    report = MeshCertificationReport(
        scoped_inputs,
        fidelity_premise=ImplicitChartPremise(
            mesh, geometry, unrestricted_parent, transition
        ),
    )
    assert report.fidelity is not None
    assert report.fidelity.chart_restriction is not None
    assert report.scoped_fidelity[0] is not report.fidelity
    independent = certify_source_fidelity(
        target.mesh, target.geometry, query, witness=target_witness, tolerance=0.5
    )
    assert report.scoped_fidelity[0].certificate_id == independent.certificate_id


@pytest.mark.parametrize("case", ("affine", "quadratic", "cubic", "radical"))
def test_face_enclosure_reuses_derivatives_without_changing_exact_bounds(
    case: str,
) -> None:
    from phydrax.discretization._coordinate_enclosure import (
        compose_expression,
        Polynomial,
    )
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    first: source_charts.Polynomial = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(2),
        (0, 1): Fraction(3),
    }
    second: source_charts.Polynomial = {
        (0, 0): Fraction(-1),
        (1, 0): Fraction(1),
    }
    third: source_charts.Polynomial = {(0, 1): Fraction(1)}
    if case == "quadratic":
        first[(2, 0)] = Fraction(1, 7)
        second[(1, 1)] = Fraction(-2, 9)
    elif case == "cubic":
        first[(2, 1)] = Fraction(3, 17)
        third[(0, 3)] = Fraction(-1, 13)
    elif case == "radical":
        first[(1, 0)] = RadicalElement.sqrt(2)
        second[(0, 1)] = RadicalElement.sqrt(3)
    expressions: source_charts.Polynomial3 = (first, second, third)
    arguments: tuple[Polynomial, ...] = (
        {(1, 0): Fraction(1)},
        {(0, 1): Fraction(1)},
    )
    composed = source_charts._triangle_coordinate_expressions(
        expressions, source_charts._STANDARD_TRIANGLE
    )
    for observed, source_expression in zip(composed, expressions, strict=True):
        reference = compose_expression(source_expression, arguments)
        assert isinstance(reference, dict)
        assert tuple(observed.items()) == tuple(reference.items())
    actual = source_charts._enclose_face(expressions)
    bases = tuple(source_charts._enclose_polynomial(value) for value in expressions)
    reference_tangents = tuple(
        tuple(source_charts._enclosed_derivative(value, axis) for axis in (0, 1))
        for value in bases
    )
    reference_curvatures = tuple(
        (
            source_charts._enclosed_derivative(tangent[0], 0),
            source_charts._enclosed_derivative(tangent[0], 1),
            source_charts._enclosed_derivative(tangent[1], 1),
        )
        for tangent in reference_tangents
    )
    for actual_rows, reference_rows in (
        (actual.tangents, reference_tangents),
        (actual.curvatures, reference_curvatures),
    ):
        for actual_row, reference_row in zip(actual_rows, reference_rows, strict=True):
            for observed, expected in zip(actual_row, reference_row, strict=True):
                assert observed.center == expected.center
                assert observed.radii == expected.radii
                assert observed.error == expected.error
                assert source_charts._enclosed_interval(
                    observed, source_charts._STANDARD_TRIANGLE
                ) == source_charts._enclosed_interval(
                    expected, source_charts._STANDARD_TRIANGLE
                )


def test_implicit_leaf_and_lattice_replay_refuse_changed_evidence() -> None:
    from phydrax.geometry._approximate_source_coverage import SourceCoverageUnavailable
    from phydrax.geometry._implicit_source_coverage import (
        _check_leaf_replay,
        _check_scaffold_lattice,
    )
    from phydrax.geometry.implicit._adaptive_polygon_replay import replay_polygon

    surface, _, _, _, _, _ = _adaptive_source()
    assert surface.cover is not None
    bounds = surface.volume.bounds
    _check_leaf_replay(surface, bounds)
    _check_scaffold_lattice(surface, bounds)
    assert replay_polygon(surface, bounds)

    changed = np.array(surface.cover.value_lower, copy=True)
    changed[0] = np.nextafter(changed[0], np.inf)
    forged = eqx.tree_at(lambda value: value.cover.value_lower, surface, changed)
    with pytest.raises(SourceCoverageUnavailable) as error:
        _check_leaf_replay(forged, bounds)
    assert error.value.status == "violated"

    changed = np.array(surface.scaffold.corner_coordinates, copy=True)
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    forged = eqx.tree_at(
        lambda value: value.scaffold.corner_coordinates, surface, changed
    )
    with pytest.raises(SourceCoverageUnavailable) as error:
        _check_scaffold_lattice(forged, bounds)
    assert error.value.status == "violated"

    changed = np.array(surface.scaffold.tetrahedra, copy=True)
    changed[0, 0] = changed[0, 1]
    forged = eqx.tree_at(lambda value: value.scaffold.tetrahedra, surface, changed)
    with pytest.raises(SourceCoverageUnavailable) as error:
        _check_scaffold_lattice(forged, bounds)
    assert error.value.status == "violated"

    changed = np.array(surface.scaffold.root_brackets, copy=True)
    changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.inf)
    forged = eqx.tree_at(lambda value: value.scaffold.root_brackets, surface, changed)
    with pytest.raises(SourceCoverageUnavailable) as error:
        _check_scaffold_lattice(forged, bounds)
    assert error.value.status == "violated"

    changed = np.array(surface.scaffold.triangle_root_edge_rows, copy=True)
    changed[0, [0, 1]] = changed[0, [1, 0]]
    forged = eqx.tree_at(
        lambda value: value.scaffold.triangle_root_edge_rows, surface, changed
    )
    assert not replay_polygon(forged, bounds)

    changed = np.array(surface.boundary_witnesses, copy=True)
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    forged = eqx.tree_at(lambda value: value.boundary_witnesses, surface, changed)
    assert not replay_polygon(forged, bounds)


def test_replayed_implicit_cover_partition_is_exact() -> None:
    from phydrax.geometry._approximate_source_coverage import SourceCoverageUnavailable
    from phydrax.geometry._implicit_source_coverage import _DyadicPartition
    from phydrax.geometry.implicit._adaptive_discovery import _cell_box

    domain = np.asarray(((0.1, 0.1, 0.1), (0.7, 0.7, 0.7)))
    indices = np.asarray(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=np.int64,
    )
    levels = np.ones(len(indices), dtype=np.int64)
    boxes = np.asarray(
        [
            _cell_box((int(index[0]), int(index[1]), int(index[2])), 1, domain)
            for index in indices
        ]
    )
    assert boxes[0, 1, 0] > boxes[4, 0, 0]
    partition = _DyadicPartition(domain, 1, len(indices))
    partition.add(levels, indices, boxes)
    partition.finish()

    mixed_indices = cast(np.ndarray, np.concatenate((indices[1:], indices), axis=0))
    mixed_levels = np.concatenate((np.ones(7, dtype=np.int64), np.full(8, 2)))
    mixed_boxes = np.asarray(
        [
            _cell_box((int(index[0]), int(index[1]), int(index[2])), int(level), domain)
            for index, level in zip(mixed_indices, mixed_levels, strict=True)
        ]
    )
    mixed = _DyadicPartition(domain, 2, len(mixed_indices))
    mixed.add(mixed_levels, mixed_indices, mixed_boxes)
    mixed.finish()

    missing = _DyadicPartition(domain, 1, len(indices) - 1)
    missing.add(levels[:-1], indices[:-1], boxes[:-1])
    with pytest.raises(SourceCoverageUnavailable) as error:
        missing.finish()
    assert error.value.status == "violated"

    duplicate_indices = indices.copy()
    duplicate_boxes = boxes.copy()
    duplicate_indices[-1] = indices[0]
    duplicate_boxes[-1] = boxes[0]
    duplicate = _DyadicPartition(domain, 1, len(indices))
    duplicate.add(levels, duplicate_indices, duplicate_boxes)
    with pytest.raises(SourceCoverageUnavailable) as error:
        duplicate.finish()
    assert error.value.status == "violated"

    narrowed = boxes.copy()
    narrowed[0, 1, 0] = np.nextafter(narrowed[0, 1, 0], -np.inf)
    invalid = _DyadicPartition(domain, 1, len(indices))
    with pytest.raises(SourceCoverageUnavailable) as error:
        invalid.add(levels, indices, narrowed)
    assert error.value.status == "violated"


def test_compact_implicit_topology_and_exact_triangle_lookup() -> None:
    from phydrax.geometry._approximate_source_coverage import SourceCoverageUnavailable
    from phydrax.geometry._implicit_source_coverage import (
        _SourceTriangles,
        _validate_oriented_implicit_triangles,
    )

    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=np.int64)
    _validate_oriented_implicit_triangles(faces)
    index = _SourceTriangles(vertices, faces)
    for row, face in enumerate(faces):
        triangle = tuple(
            tuple(Fraction(float(coordinate)) for coordinate in vertices[vertex])
            for vertex in face[::-1]
        )
        assert index.get(triangle) == row
    perturbed = (
        (Fraction(1, 3), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0)),
    )
    assert index.get(perturbed) is None
    reversed_face = faces.copy()
    reversed_face[0] = reversed_face[0, ::-1]
    with pytest.raises(SourceCoverageUnavailable) as error:
        _validate_oriented_implicit_triangles(reversed_face)
    assert error.value.status == "violated"
    with pytest.raises(SourceCoverageUnavailable):
        _SourceTriangles(vertices, np.concatenate((faces, faces[:1])))


def test_compact_triangles_match_coordinate_identity_and_duplicates() -> None:
    from phydrax.geometry._approximate_source_coverage import SourceCoverageUnavailable
    from phydrax.geometry._implicit_source_coverage import _SourceTriangles

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    faces = np.asarray([[0, 1, 2], [3, 1, 4], [1, 2, 5]])
    reference = {
        tuple(
            sorted(tuple(Fraction(float(v)) for v in vertices[index]) for index in face)
        ): row
        for row, face in enumerate(faces)
    }
    index = _SourceTriangles(vertices, faces)
    for key, row in reference.items():
        assert index.get(key) == row
        assert index.get(key[::-1]) == row
    assert index.get(((Fraction(0),) * 3,) * 3) is None
    with pytest.raises(SourceCoverageUnavailable) as error:
        _SourceTriangles(vertices, np.concatenate((faces, np.asarray([[3, 1, 2]]))))
    assert error.value.status == "violated"

    # Equal coordinates straddling a rank-building chunk remain one identity.
    vertices = np.zeros((65538, 3))
    vertices[:, 0] = np.arange(len(vertices))
    vertices[65536] = vertices[65535]
    faces = np.asarray([[0, 65534, 65535], [0, 65534, 65537]])
    index = _SourceTriangles(vertices, faces)
    triangle = tuple(tuple(Fraction(float(v)) for v in vertices[i]) for i in faces[1])
    assert index.get(triangle) == 1
    with pytest.raises(SourceCoverageUnavailable):
        _SourceTriangles(vertices, np.asarray([[0, 65534, 65535], [0, 65534, 65536]]))


def test_bounded_implicit_indices_match_real_source_rows() -> None:
    surface, _, _, _, _, _ = _adaptive_source()
    scaffold = surface.scaffold
    roots = source_charts._source_root_rows(surface)
    points = source_charts._root_rows_by_point(roots)
    cells = source_charts._source_cell_rows(surface)
    old_roots = {
        source_charts._grid_edge(np.asarray(edge)): (
            np.array(point, copy=True),
            float(error),
        )
        for edge, point, error in zip(
            scaffold.root_edge_ids,
            scaffold.root_points,
            scaffold.root_errors,
            strict=True,
        )
    }
    old_points: dict[bytes, GridEdge] = {}
    for edge in sorted(old_roots):
        old_points.setdefault(old_roots[edge][0].tobytes(), edge)
    old_cells = {
        _source_cell_key(np.asarray(key)): row
        for row, key in enumerate(scaffold.cell_indices)
    }
    roots.prepare(tuple(old_roots))
    points.prepare(tuple(old_points))
    cells.prepare(tuple(old_cells))
    for edge, (point, error) in old_roots.items():
        actual, actual_error = roots[edge]
        assert actual.tobytes() == point.tobytes()
        assert actual_error == error
    for point, edge in old_points.items():
        assert points.get(point) == edge
    for key, row in old_cells.items():
        assert cells.get(key) == row
    assert cells.get((-1, -1, -1, -1)) is None
    assert points.get(np.asarray((np.inf, np.inf, np.inf)).tobytes()) is None


def test_bounded_implicit_indices_duplicate_and_signed_zero() -> None:
    from types import SimpleNamespace

    edges = np.asarray(
        [
            [[0, 0, 0], [1, 0, 0]],
            [[2, 0, 0], [3, 0, 0]],
            [[1, 0, 0], [0, 0, 0]],
            [[4, 0, 0], [5, 0, 0]],
        ],
        dtype=np.int64,
    )
    coords = np.asarray(
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [-0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
    )
    scaffold = SimpleNamespace(
        root_edge_ids=edges,
        root_points=coords,
        root_errors=np.arange(4.0),
        cell_indices=np.asarray([[1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0]]),
    )
    surface = cast(AdaptiveImplicitSurface, SimpleNamespace(scaffold=scaffold))
    roots = source_charts._source_root_rows(surface)
    by_point = source_charts._root_rows_by_point(roots)
    cells = source_charts._source_cell_rows(surface)
    first = source_charts._grid_edge(edges[0])
    second = source_charts._grid_edge(edges[1])
    assert roots[first][0].tobytes() == coords[2].tobytes()
    assert roots[first][1] == 2.0
    assert by_point.get(coords[0].tobytes()) == second
    assert by_point.get(coords[2].tobytes()) == first
    assert cells.get((1, 0, 0, 0)) == 2
    assert cells.get((9, 0, 0, 0)) is None
    assert ((9, 0, 0), (10, 0, 0)) not in roots


def test_bounded_reverse_cover_preserves_all_rows_and_exact_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import math
    from types import SimpleNamespace

    from phydrax.geometry.implicit._adaptive_discovery import AdaptiveImplicitCover

    size = source_charts._REVERSE_COVER_ROUND
    count = 2 * size + 123
    boxes = np.arange(count * 6, dtype=np.float64).reshape(count, 2, 3)
    lower = np.where(np.arange(count) % 3, -1.0, 1.0)
    upper = np.where(np.arange(count) % 5, 1.0, -1.0)
    cover = cast(
        AdaptiveImplicitCover,
        SimpleNamespace(
            boxes=boxes,
            value_lower=lower,
            value_upper=upper,
        ),
    )
    actual = list(source_charts._possible_zero_box_blocks(cover))
    expected = boxes[(lower <= 0.0) & (upper >= 0.0)]
    assert all(len(block) == size for block in actual[:-1])
    assert np.array_equal(np.concatenate(actual), expected)

    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 0] = np.linspace(0.0, 1.1, count)
    errors = np.linspace(0.0, 0.02, count)
    largest_batch = 0

    def distance(
        _maps: object, first: np.ndarray, second: np.ndarray, reach: float
    ) -> np.ndarray:
        nonlocal largest_batch
        assert np.array_equal(first, second)
        largest_batch = max(largest_batch, len(first))
        return np.where(first[:, 0] < reach, first[:, 0], np.inf)

    monkeypatch.setattr(source_charts, "_box_face_distance_lower", distance)
    pending = np.arange(count)
    nearest = np.empty(count)
    reach = 2.0**-6
    while pending.size:
        found = distance(None, points[pending], points[pending], reach)
        settled = (found < reach) | (reach > 1.0)
        nearest[pending[settled]] = np.minimum(found[settled], reach)
        pending = pending[~settled]
        reach = math.nextafter(2.0 * reach, math.inf)
    expected_lower = float(
        np.max(np.maximum(np.nextafter(nearest - errors, -np.inf), 0.0), initial=0.0)
    )
    largest_batch = 0
    maps = cast(source_charts._FaceMaps, None)
    assert (
        source_charts._witness_distance_lower(maps, points, errors, 1.0) == expected_lower
    )
    assert largest_batch <= size


def test_implicit_complete_ancestry_predicate_exactness() -> None:
    from types import SimpleNamespace

    from phydrax.geometry._approximate_source_coverage import (
        _names_each_scaffold_triangle_once,
    )

    cells = np.asarray([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=np.int64)
    edges = np.asarray([[[i, 0, 0], [i + 1, 0, 0]] for i in range(6)], dtype=np.int64)
    triangle_cells = np.asarray([0, 1], dtype=np.int64)
    triangle_edges = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    scaffold = SimpleNamespace(
        cell_indices=cells,
        root_edge_ids=edges,
        triangle_cell_rows=triangle_cells,
        triangle_root_edge_rows=triangle_edges,
    )
    domain = cast(AdaptiveImplicitSurface, SimpleNamespace(scaffold=scaffold))
    witness = cast(
        SourceBoundaryChartWitness,
        SimpleNamespace(
            implicit_cell_keys=cells[::-1].copy(),
            implicit_root_edge_keys=edges[triangle_edges[::-1, ::-1]][:, :, ::-1].copy(),
        ),
    )
    assert _names_each_scaffold_triangle_once(domain, witness)
    witness.implicit_cell_keys[1] = witness.implicit_cell_keys[0]
    witness.implicit_root_edge_keys[1] = witness.implicit_root_edge_keys[0]
    assert not _names_each_scaffold_triangle_once(domain, witness)
    witness.implicit_cell_keys = witness.implicit_cell_keys[:1]
    witness.implicit_root_edge_keys = witness.implicit_root_edge_keys[:1]
    assert not _names_each_scaffold_triangle_once(domain, witness)
    scaffold.triangle_root_edge_rows = triangle_edges.copy()
    scaffold.triangle_root_edge_rows[1, 0] = len(edges)
    with pytest.raises(IndexError):
        _names_each_scaffold_triangle_once(domain, witness)
    scaffold.triangle_root_edge_rows = triangle_edges
    witness.implicit_cell_keys = cells.copy()
    witness.implicit_root_edge_keys = edges[triangle_edges].copy()
    scaffold.triangle_cell_rows = np.asarray([0, 0], dtype=np.int64)
    scaffold.triangle_root_edge_rows = np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    assert not _names_each_scaffold_triangle_once(domain, witness)
    scaffold.triangle_cell_rows = triangle_cells
    scaffold.triangle_root_edge_rows = triangle_edges[:1]
    with pytest.raises(ValueError, match="zip\\(\\) argument 2 is shorter"):
        _names_each_scaffold_triangle_once(domain, witness)
    scaffold.triangle_root_edge_rows = triangle_edges.copy()
    scaffold.triangle_root_edge_rows[1, 0] = len(edges)
    with pytest.raises(IndexError):
        _names_each_scaffold_triangle_once(domain, witness)

    count = 8193
    many_cells = np.zeros((count, 4), dtype=np.int64)
    many_cells[:, 1] = np.arange(count)
    many_edges = np.tile(np.asarray([[0, 1, 2]], dtype=np.int64), (count, 1))
    scaffold.cell_indices = many_cells
    scaffold.triangle_cell_rows = np.arange(count, dtype=np.int64)
    scaffold.triangle_root_edge_rows = many_edges
    witness.implicit_cell_keys = many_cells.copy()
    witness.implicit_root_edge_keys = edges[many_edges].copy()
    assert _names_each_scaffold_triangle_once(domain, witness)
    witness.implicit_cell_keys[-1] = witness.implicit_cell_keys[0]
    assert not _names_each_scaffold_triangle_once(domain, witness)


def test_reverse_cover_searches_past_near_noncertifying_face() -> None:
    from types import SimpleNamespace

    # Exact face-map and source-box operands from the original adaptive sphere
    # capture: the first face box overlaps, but only the second face certifies.
    coefficients = np.asarray(
        [
            [
                [-0.3410944973107536, -0.028708359281904627, -0.008355139944754009],
                [-0.04452625236781594, 0.014747762099195635, 0.16304613827678627],
                [-0.13897106445012897, 0.13867311607951216, 0.10051345576057563],
            ],
            [
                [-0.29229791765256563, -0.048796579658187955, -0.057151719602941964],
                [0.11373552232394626, -0.1582617746917622, 0.004784363585024071],
                [-0.19815942646685542, 0.05918836201672645, 0.15970181777730208],
            ],
        ]
    )
    lower = np.zeros((2, 3, 10))
    upper = np.zeros_like(lower)
    middle = np.zeros_like(lower)
    middle[:, :, :3] = coefficients
    lower[:, :, :3] = coefficients
    upper[:, :, :3] = coefficients
    lower[0, 2, 1] = 0.13867311607951213
    lower[0, 2, 2] = 0.10051345576057562
    lower[1, 1, 1] = -0.15826177469176222
    lower[1, 2, 2] = 0.15970181777730205
    upper[0, 1, 2] = 0.1630461382767863
    box_lower = np.asarray(
        [
            [-0.3698028565926582, -0.04452625236781594, -0.13897106445012897],
            [-0.3494496372555076, -0.04452625236781594, -0.19815942646685542],
        ]
    )
    box_upper = np.asarray(
        [
            [-0.3410944973107536, 0.11851988590897033, -0.00029794837061681423],
            [-0.29229791765256563, 0.11851988590897033, -0.03845760868955334],
        ]
    )
    maps = source_charts._FaceMaps(
        middle,
        lower,
        upper,
        np.zeros((2, 3)),
        box_lower,
        box_upper,
        source_charts.prepare_bvh(box_lower, box_upper, dtype=np.float64),
    )
    box = np.asarray(
        [
            [
                [-0.35052978515625005, 0.02807861328125, -0.11956054687500002],
                [-0.3496240234375, 0.028984375000000003, -0.11865478515625001],
            ]
        ]
    )
    cover = SimpleNamespace(
        boxes=box,
        value_lower=np.asarray([-0.000723988990780966]),
        value_upper=np.asarray([0.0004915189407109777]),
    )
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile()
    bounds = adaptive_discovery._FieldEnclosures(geometry, "interval")
    surface = cast(
        AdaptiveImplicitSurface,
        SimpleNamespace(
            cover=cover,
            volume=SimpleNamespace(bounds=bounds),
            boundary_witnesses=np.empty((0, 3)),
            boundary_errors=np.empty(0),
        ),
    )
    center = np.mean(box, axis=1)
    first = list(
        source_charts._box_face_pair_blocks(maps, center, center, 0.02 * 2.0**-6)
    )
    assert np.array_equal(np.concatenate([faces for _, faces in first]), np.asarray([0]))
    lower_bound, upper_bound = source_charts._source_to_mesh_bounds(surface, maps, 0.02)
    assert lower_bound == 0.0
    assert upper_bound == 0.013992911797843887
