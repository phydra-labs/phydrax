#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from fractions import Fraction
from typing import final, NoReturn

import equinox as eqx
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

    query: AdaptiveImplicitBoundarySource = eqx.field(static=True)
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


def test_adaptive_implicit_on_surface_p2_sphere_charts_certify_in_bounded_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _surface, source, contract, mesh, affine, affine_witness = _adaptive_source(
        smooth=False, authoring_profile=True
    )
    affine_root_map = _affine_root_map(mesh, source, affine, affine_witness)
    curved, witness = _curved_geometry(mesh, source, affine_root_map, 2)
    rounds = 0
    run_piece_steps = source_charts._run_piece_steps

    def counted_rounds(
        bounds: source_charts._FieldEnclosures,
        hessian_program: source_charts.PreparedIntervalFunction,
        steps: Sequence[source_charts._PieceSteps],
        queries: Sequence[source_charts._BoxRequests] = (),
        /,
    ) -> tuple[list[source_charts._PieceOutcome], list[source_charts._BoxResults]]:
        nonlocal rounds
        rounds += 1
        return run_piece_steps(bounds, hessian_program, steps, queries)

    monkeypatch.setattr(source_charts, "_run_piece_steps", counted_rounds)
    # Without fidelity controls every face's endpoint signs and differential
    # close by refinement and bracket widening alone, in a few batched rounds.
    chart_map = prepare_adaptive_implicit_root_charts(
        mesh, curved, source, witness, frame_id=contract.spatial_id
    )
    assert chart_map.status == "certified"
    assert all(item.status == "certified" for item in chart_map.chart_evidence)
    assert all(item.chart_injectivity_margin > 0.0 for item in chart_map.chart_evidence)
    assert rounds <= 4


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


def test_local_reverse_cover_proves_complete_meshes_and_refutes_missing_faces() -> None:
    surface, source, _contract, mesh, geometry, witness = _adaptive_source()
    charts = prepare_adaptive_implicit_root_charts(
        mesh, geometry, source, witness, tolerance=0.5, maximum_normal_deviation=np.pi
    )
    faces = charts._faces
    tolerance = 0.05

    lower, upper = source_charts._source_to_mesh_bounds(
        surface, source_charts._face_maps(faces), tolerance
    )
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
    with pytest.raises(ValueError, match="smooth field"):
        prepare_adaptive_implicit_root_charts(mesh, geometry, unprofiled_sphere, witness)

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
    with pytest.raises(ValueError, match="smooth field"):
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
