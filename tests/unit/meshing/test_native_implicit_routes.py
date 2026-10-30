#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._tensor_support import PreparedTensorGrid
from phydrax.geometry import Sphere, Torus
from phydrax.geometry._contracts import CompiledGeometry
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryDistanceWitness,
)
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.geometry.implicit._restricted_delaunay_coverage import (
    _prove_degree_one_normal_fiber,
    _prove_face_projection_orientation,
    _revalidate_oriented_surface_topology,
    RestrictedDelaunayBoundaryWitness,
    RestrictedDelaunayCoverageUnavailable,
    validate_restricted_delaunay_boundary_witness,
)
from phydrax.meshing._association import GeometryAssociationKind
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._implicit_volume import (
    _inside_seed_rows,
    _root_edge_bank,
    _strict_region_site_mask,
    _topology_matches,
)
from phydrax.meshing._native_construction import NativeMeshConstruction
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native_implicit import prepare_native_implicit
from phydrax.meshing.providers._native_options import (
    NativeMeshingOptions,
    NativeMeshingRoute,
)
from phydrax.meshing.providers._native_sources import NativeImplicitSource


type _Point3 = tuple[Fraction, Fraction, Fraction]
type _Triangle3 = tuple[_Point3, _Point3, _Point3]


def _geometry_and_grid() -> tuple[CompiledGeometry, PreparedTensorGrid]:
    geometry = Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="adaptive-route-sphere",
    ).compile()
    grid = TensorGridPlan(
        tuple(UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))
    return geometry, grid


def _grid_domain(grid: PreparedTensorGrid, /) -> np.ndarray:
    lower: list[Array] = []
    upper: list[Array] = []
    for axis in grid.axes:
        bounds = axis.bounds
        if bounds is None:
            raise AssertionError("Prepared implicit grid axis must retain its bounds.")
        lower.append(bounds[0])
        upper.append(bounds[1])
    return np.asarray((lower, upper), dtype=np.float64)


def _request(
    source: NativeImplicitSource | AnalyticImplicitProfile,
    /,
    *,
    maximum_deviation: float | None,
) -> tuple[SurfaceMeshingSpec, NativeMeshingOptions, SpatialCoordinateContract]:
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            3,
            CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(),
        maximum_deviation=maximum_deviation,
    )
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08,
        root_tolerance=1.0e-8,
    )
    options = NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=policy,
    )
    contract = SpatialCoordinateContract.si()
    return specification, options, contract


def _assert_consumer_metadata(
    source: NativeImplicitSource | AnalyticImplicitProfile,
    specification: SurfaceMeshingSpec,
    construction: NativeMeshConstruction,
    contract: SpatialCoordinateContract,
) -> None:
    mesh = construction.mesh
    boundary = construction.organization.boundary
    assert boundary is not None
    assert boundary.mesh is mesh
    assert boundary.metadata.source_id == source.source_id
    assert boundary.metadata.source_revision == source.source_revision
    assert boundary.metadata.coordinate_contract.spatial_id == contract.spatial_id
    assert (
        boundary.metadata.cell_tags
        == ("implicit-zero-set",) * (mesh.blocks[0].vertices.shape[0])
    )

    source_scope_id = int(source.scope_entity_ids(2)[0])
    source_entity_id, occurrence = source.source_entity_key_for_scope(2, source_scope_id)
    cell_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == mesh.entity_set(2).entity_set_id
    )
    assert cell_association.association_kind is GeometryAssociationKind.IMPLICIT
    assert cell_association.source_id == source.source_id
    assert cell_association.source_revision == source.source_revision
    assert cell_association.source_entity_ids == (source_entity_id,) * len(
        cell_association.source_entity_ids
    )
    assert cell_association.source_occurrence_paths == (occurrence,) * len(
        cell_association.source_occurrence_paths
    )
    assert cell_association.complete

    labels = construction.organization.labels
    assert len(labels) == 1
    assert labels[0].name == "implicit-zero-set"
    np.testing.assert_array_equal(
        labels[0].scope.entity_ids,
        mesh.entity_set(2).entity_ids,
    )
    patches = construction.organization.patches
    assert len(patches) == 1
    assert patches[0].name == "implicit-zero-set"
    np.testing.assert_array_equal(
        patches[0].scope.entity_ids,
        mesh.entity_set(2).entity_ids,
    )

    certification = construction.certification
    assert certification.schedule.route == "surface"
    assert certification.source_id == source.source_id
    assert certification.source_revision == source.source_revision
    assert certification.fidelity_source is not None
    assert certification.fidelity_source.source_id == source.source_id
    assert certification.fidelity_source.source_revision == source.source_revision
    if specification.maximum_deviation is None:
        assert certification.fidelity_tolerance is None
    else:
        assert certification.fidelity_tolerance is not None
        assert certification.fidelity_tolerance <= specification.maximum_deviation
    witness = certification.fidelity_witness
    if not isinstance(witness, SourceBoundaryChartWitness):
        raise AssertionError("Implicit fidelity must retain its source chart witness.")
    assert witness.source_id == source.source_id
    assert witness.source_revision == source.source_revision
    assert witness.source_query_id
    assert np.asarray(witness.implicit_cell_keys).shape == (mesh.blocks[0].cell_count, 4)
    assert np.asarray(witness.implicit_root_edge_keys).shape == (
        mesh.blocks[0].cell_count,
        3,
        2,
        3,
    )
    witness.require_current(mesh, construction.geometry, certification.fidelity_source)
    evidence = certification.fidelity_source.boundary_distance(
        np.asarray(mesh.coordinates, dtype=np.float64)[:4]
    )
    assert evidence.semantics == "certified"
    assert construction.compliance.passed


def test_native_analytic_implicit_profile_normalizes_inside_plan_preparation() -> None:
    geometry, _ = _geometry_and_grid()
    contract = SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        geometry,
        contract,
        source_id="analytic-profile-sphere",
        source_revision="analytic-profile-revision-4",
    )
    specification, options, _ = _request(
        profile,
        maximum_deviation=0.8,
    )

    assert profile.scope_entity_ids(0).size == 0
    assert profile.scope_entity_ids(1).size == 0
    assert profile.scope_entity_ids(2).shape == (1,)
    assert profile.scope_entity_ids(3).shape == (1,)
    assert profile.source_id == "analytic-profile-sphere"
    assert profile.source_revision == "analytic-profile-revision-4"

    construction = prepare_native_implicit(
        profile,
        specification,
        options,
        contract,
    ).execute()

    _assert_consumer_metadata(profile, specification, construction, contract)


def test_native_implicit_grid_source_builds_certified_surface_construction() -> None:
    geometry, grid = _geometry_and_grid()
    source = NativeImplicitSource(
        geometry,
        "implicit-grid-sphere",
        "grid-revision-3",
        grid=grid,
    )
    assert source.scope_entity_ids(0).size == 0
    assert source.scope_entity_ids(1).size == 0
    assert source.scope_entity_ids(2).shape == (1,)
    assert source.scope_entity_ids(3).shape == (1,)
    specification, options, contract = _request(
        source,
        maximum_deviation=0.8,
    )
    construction = prepare_native_implicit(
        source,
        specification,
        options,
        contract,
    ).execute()

    assert construction.mesh.topological_dimension == 2
    assert construction.mesh.ambient_dimension == 3
    assert all(block.cell_kind == "triangle" for block in construction.mesh.blocks)
    np.testing.assert_array_equal(
        construction.geometry.coordinates,
        construction.mesh.coordinates,
    )
    _assert_consumer_metadata(source, specification, construction, contract)


def test_native_implicit_retained_discovery_preserves_zero_set_identity() -> None:
    geometry, grid = _geometry_and_grid()
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08,
        root_tolerance=1.0e-8,
    )
    source_id = "implicit-retained-sphere"
    discovery = discover_adaptive_implicit_surface(
        geometry,
        domain=_grid_domain(grid),
        policy=policy,
        source_id=source_id,
    )
    source = NativeImplicitSource(
        geometry,
        source_id,
        "retained-revision-8",
        discovery=discovery,
    )
    specification, options, contract = _request(
        source,
        maximum_deviation=None,
    )
    construction = prepare_native_implicit(
        source,
        specification,
        options,
        contract,
    ).execute()

    assert discovery.mesh is not None
    np.testing.assert_array_equal(
        np.asarray(construction.mesh.coordinates),
        np.asarray(discovery.mesh.vertices),
    )
    _assert_consumer_metadata(source, specification, construction, contract)


def test_native_implicit_route_rejects_sampled_discovery() -> None:
    geometry, grid = _geometry_and_grid()
    source_id = "implicit-sampled-sphere"
    policy = AdaptiveImplicitSurfacePolicy(
        enclosure="sampled",
        relative_cell_diameter=0.08,
        root_tolerance=1.0e-8,
    )
    discovery = discover_adaptive_implicit_surface(
        geometry,
        domain=_grid_domain(grid),
        policy=policy,
        source_id=source_id,
    )
    source = NativeImplicitSource(
        geometry,
        source_id,
        "sampled-revision-1",
        grid=grid,
        discovery=discovery,
    )
    specification, _, contract = _request(source, maximum_deviation=None)
    options = NativeMeshingOptions("implicit_surface", implicit_policy=policy)

    with pytest.raises(MeshingFailure, match="complete rigorous cover"):
        prepare_native_implicit(source, specification, options, contract)


@pytest.mark.parametrize(
    "route",
    (
        "implicit_restricted_delaunay",
        "implicit_adaptive_tetrahedral",
    ),
)
def test_native_implicit_volume_route_builds_region_mesh(
    route: NativeMeshingRoute,
) -> None:
    geometry, grid = _geometry_and_grid()
    source_id = "implicit-volume-sphere"
    revision = "implicit-volume-revision-4"
    contract = SpatialCoordinateContract.si()
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08,
        root_tolerance=1.0e-8,
    )
    discovery: AdaptiveImplicitSurface | None = None
    if route == "implicit_restricted_delaunay":
        source: NativeImplicitSource | AnalyticImplicitProfile = AnalyticImplicitProfile(
            geometry,
            contract,
            tube_radius=0.5,
            source_id=source_id,
            source_revision=revision,
        )
    else:
        discovery = discover_adaptive_implicit_surface(
            geometry,
            domain=_grid_domain(grid),
            policy=policy,
            source_id=source_id,
        )
        source = NativeImplicitSource(
            geometry,
            source_id,
            revision,
            discovery=discovery,
        )
    boundary_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.8,
    )
    options = NativeMeshingOptions(
        route,
        implicit_policy=policy if route == "implicit_adaptive_tetrahedral" else None,
        volume_schedule=NativeVolumeSchedule(),
    )
    construction = prepare_native_implicit(
        source,
        specification,
        options,
        contract,
    ).execute()

    assert construction.mesh.topological_dimension == 3
    assert construction.mesh.ambient_dimension == 3
    assert all(
        reference_cell_topology(block.cell_kind).family == "tetrahedron"
        for block in construction.mesh.blocks
    )
    assert construction.certification.schedule.route == "volume_implicit"
    assert construction.certification.fidelity_source is not None
    assert construction.certification.fidelity_source.source_id == source.source_id
    assert (
        construction.certification.fidelity_source.source_revision
        == source.source_revision
    )
    assert construction.certification.fidelity_tolerance is not None
    assert specification.maximum_deviation is not None
    assert (
        construction.certification.fidelity_tolerance <= specification.maximum_deviation
    )
    assert construction.certification.cell_regions is not None
    assert np.asarray(construction.certification.cell_regions).shape == (
        construction.mesh.blocks[0].vertices.shape[0],
    )
    assert np.all(np.asarray(construction.certification.cell_regions) == 0)
    boundary = construction.organization.boundary
    assert boundary is not None
    assert boundary.metadata.source_id == source.source_id
    assert boundary.metadata.source_revision == source.source_revision
    assert all(tag == "implicit-zero-set" for tag in boundary.metadata.cell_tags)
    assert len(construction.organization.zones) == 1
    if route == "implicit_restricted_delaunay":
        assert isinstance(source, AnalyticImplicitProfile)
        assert construction.certification.domain is source
        assert construction.certification.fidelity_source is source
        assert construction.organization.zones[0].name == source.region_ids[0]
        distance_witness = construction.certification.fidelity_witness
        coverage_witness = construction.certification.source_coverage_witness
        assert isinstance(distance_witness, SourceBoundaryDistanceWitness)
        assert isinstance(coverage_witness, RestrictedDelaunayBoundaryWitness)
        assert distance_witness.source_id == source.source_id
        assert distance_witness.source_revision == source.source_revision
        assert distance_witness.cover_radius < source.tube_radius
        proof = validate_restricted_delaunay_boundary_witness(
            construction.mesh,
            construction.geometry,
            source,
            construction.certification.cell_regions,
            distance_witness,
            coverage_witness,
        )
        assert proof.region_ids == source.region_ids
        assert proof.target_facet_count > 0
        assert proof.source_facet_count == proof.covered_source_facet_count == 0
        assert coverage_witness.distance_witness_id == distance_witness.witness_id
        forged_binding = eqx.tree_at(
            lambda value: value.mesh_id,
            coverage_witness,
            "forged-restricted-mesh",
        )
        with pytest.raises(ValueError, match="restricted witness"):
            validate_restricted_delaunay_boundary_witness(
                construction.mesh,
                construction.geometry,
                source,
                construction.certification.cell_regions,
                distance_witness,
                forged_binding,
            )
        forged_vertex_ids = np.asarray(
            construction.mesh.vertex_global_ids, dtype=np.int64
        ).copy()
        forged_vertex_ids[0] = int(np.max(forged_vertex_ids)) + 1
        forged_mesh = eqx.tree_at(
            lambda value: value.vertex_global_ids,
            construction.mesh,
            jnp.asarray(forged_vertex_ids),
        )
        with pytest.raises(ValueError, match="Target mesh"):
            validate_restricted_delaunay_boundary_witness(
                forged_mesh,
                construction.geometry,
                source,
                construction.certification.cell_regions,
                distance_witness,
                coverage_witness,
            )
        from phydrax.discretization._cell_geometry import CellGeometrySpec
        from phydrax.discretization._cell_geometry_validity import cell_geometry_id
        from phydrax.discretization._cell_mesh import (
            CellMesh,
            TetrahedralConnectivity,
        )

        source_center = np.asarray(source.center, dtype=np.float64)
        target_coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
        off_source_mesh = CellMesh.from_tetrahedra(
            source_center + 3.0 * (target_coordinates - source_center),
            np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int32),
        )
        off_source_geometry = CellGeometrySpec.affine(off_source_mesh)
        off_source_connectivity = off_source_mesh.connectivity
        if not isinstance(off_source_connectivity, TetrahedralConnectivity):
            raise AssertionError("Off-source tetrahedral mesh lost its connectivity.")
        off_source_faces = off_source_mesh.entity_set(2)
        off_source_boundary_ids = np.asarray(off_source_faces.entity_ids, dtype=np.int64)[
            np.asarray(off_source_connectivity.boundary_faces, dtype=np.bool_)
        ]
        off_source_distance = SourceBoundaryDistanceWitness(
            mesh_id=off_source_mesh.mesh_id,
            geometry_id=cell_geometry_id(off_source_geometry),
            topology_id=off_source_mesh.topology_id,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id=source.profile_id,
            sample_count=distance_witness.sample_count,
            cover_radius=distance_witness.cover_radius,
            target_facet_global_ids=off_source_boundary_ids,
        )
        off_source_coverage = RestrictedDelaunayBoundaryWitness(
            mesh_id=off_source_mesh.mesh_id,
            geometry_id=cell_geometry_id(off_source_geometry),
            topology_id=off_source_mesh.topology_id,
            profile_id=source.profile_id,
            source_id=source.source_id,
            source_revision=source.source_revision,
            domain_id=source.domain_id,
            distance_witness_id=off_source_distance.witness_id,
        )
        with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="normal tube"):
            validate_restricted_delaunay_boundary_witness(
                off_source_mesh,
                off_source_geometry,
                source,
                construction.certification.cell_regions,
                off_source_distance,
                off_source_coverage,
            )
    else:
        assert isinstance(source, NativeImplicitSource)
        assert discovery is not None
        assert construction.certification.domain is discovery
        assert construction.organization.zones[0].name == discovery.region_ids[0]
        witness = construction.certification.fidelity_witness
        assert witness is not None
        fidelity_source = construction.certification.fidelity_source
        assert fidelity_source is not None
        assert isinstance(witness, SourceBoundaryChartWitness)
        assert witness.source_query_id == discovery.boundary_source.source_revision
        source_surface_key = source.source_entity_key_for_scope(
            2, int(source.scope_entity_ids(2)[0])
        )
        assert witness.source_entity_ids == (source_surface_key[0],) * len(
            witness.source_chart_ids
        )
        assert witness.occurrence_paths == (source_surface_key[1],) * len(
            witness.source_chart_ids
        )
        assert np.asarray(witness.implicit_cell_keys).shape == (
            boundary.mesh.blocks[0].cell_count,
            4,
        )
        witness.require_current(
            construction.mesh,
            construction.geometry,
            fidelity_source,
        )
    assert any(
        patch.name == "implicit-zero-set" for patch in construction.organization.patches
    )
    source_surface_id = int(source.scope_entity_ids(2)[0])
    source_surface_key = source.source_entity_key_for_scope(2, source_surface_id)
    boundary_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id
        == construction.mesh.entity_set(2).entity_set_id
    )
    assert source_surface_key[0] in boundary_association.source_entity_ids
    assert source_surface_key[1] in boundary_association.source_occurrence_paths
    assert boundary_association.complete
    assert all(
        association.source_id == source.source_id
        and association.source_revision == source.source_revision
        for association in construction.organization.associations
    )
    assert construction.compliance.passed


def _octahedral_surface(
    center: _Point3,
    radius: Fraction,
    /,
    *,
    vertex_offset: int = 0,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[_Triangle3, ...]]:
    directions: tuple[tuple[int, int, int], ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    rows = {
        (axis, sign): 2 * axis + (0 if sign > 0 else 1)
        for axis in range(3)
        for sign in (-1, 1)
    }
    faces: list[tuple[int, int, int]] = []
    triangles: list[_Triangle3] = []

    def point_for(vertex: int) -> _Point3:
        return (
            center[0] + radius * directions[vertex][0],
            center[1] + radius * directions[vertex][1],
            center[2] + radius * directions[vertex][2],
        )

    for sx, sy, sz in product((-1, 1), repeat=3):
        ids = (rows[(0, sx)], rows[(1, sy)], rows[(2, sz)])
        points = (point_for(ids[0]), point_for(ids[1]), point_for(ids[2]))
        first, second, third = (
            np.asarray(directions[vertex], dtype=np.int64) for vertex in ids
        )
        normal = np.cross(second - first, third - first)
        centroid = first + second + third
        if int(np.dot(normal, centroid)) < 0:
            ids = (ids[0], ids[2], ids[1])
            points = (points[0], points[2], points[1])
        faces.append(
            (
                vertex_offset + ids[0],
                vertex_offset + ids[1],
                vertex_offset + ids[2],
            )
        )
        triangles.append(points)
    return tuple(faces), tuple(triangles)


def test_projection_refuses_whole_face_normal_tangency() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract.si())
    center: _Point3 = (
        Fraction.from_float(float(profile.center[0])),
        Fraction.from_float(float(profile.center[1])),
        Fraction.from_float(float(profile.center[2])),
    )
    radius = Fraction.from_float(profile.radius)
    step = Fraction(1, 100)
    triangle = (
        (center[0] + radius, center[1], center[2]),
        (center[0] + radius + step, center[1], center[2]),
        (center[0] + radius, center[1] + step, center[2]),
    )
    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="tangent"):
        _prove_face_projection_orientation(
            profile,
            triangle,
            (Fraction(0), Fraction(0), Fraction(1)),
            91,
        )


def test_projection_refuses_missing_band_and_hidden_shell() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract.si())
    center: _Point3 = (
        Fraction.from_float(float(profile.center[0])),
        Fraction.from_float(float(profile.center[1])),
        Fraction.from_float(float(profile.center[2])),
    )
    source_radius = Fraction.from_float(profile.radius)
    complete_faces, _complete_triangles = _octahedral_surface(center, source_radius)
    _revalidate_oriented_surface_topology(complete_faces, 2)

    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="edges"):
        _revalidate_oriented_surface_topology(complete_faces[:-1], 2)

    inner_radius = source_radius - Fraction(1, 20)
    outer_radius = source_radius + Fraction(1, 20)
    inner_faces, inner_triangles = _octahedral_surface(center, inner_radius)
    outer_faces, outer_triangles = _octahedral_surface(
        center, outer_radius, vertex_offset=6
    )
    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="components"):
        _revalidate_oriented_surface_topology(inner_faces + outer_faces, 2)
    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="preimages"):
        _prove_degree_one_normal_fiber(
            profile,
            inner_triangles + outer_triangles,
            Fraction.from_float(profile.reach_lower),
        )


def test_restricted_root_bank_batches_and_retains_strict_evidence() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-root-bank",
        source_revision="restricted-root-bank-r1",
    )
    edge_points = (
        ((0.5, 0.0, 0.0), (2.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        ((0.55, -0.6, 0.0), (1.0, -0.2, 0.0)),
        ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    endpoint_bounds = np.asarray(
        [((first, first), (second, second)) for first, second in edge_points],
        dtype=np.float64,
    )
    roots_by_edge = _root_edge_bank(profile, endpoint_bounds, 1.0e-3)
    assert tuple(len(roots) for roots in roots_by_edge) == (1, 0, 0, 2)
    for roots in roots_by_edge:
        for (
            parameter_bounds,
            coordinate_bounds,
            value_bounds,
            derivative_bounds,
            orientation,
        ) in roots:
            assert np.all(np.isfinite(parameter_bounds))
            assert np.all(np.isfinite(coordinate_bounds))
            assert np.all(np.isfinite(value_bounds))
            assert np.all(np.isfinite(derivative_bounds))
            assert derivative_bounds[0] > 0.0 or derivative_bounds[1] < 0.0
            assert orientation in (-1, 1)

    tangent_bounds = np.asarray(
        (
            ((-1.0, 0.0, 0.75), (-1.0, 0.0, 0.75)),
            ((1.0, 0.0, 0.75), (1.0, 0.0, 0.75)),
        ),
        dtype=np.float64,
    )[None, ...]
    with pytest.raises(MeshingFailure):
        _root_edge_bank(profile, tangent_bounds, 1.0e-3)


def test_restricted_surface_topology_ignores_unreferenced_volume_vertices() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-topology-offset-vertices",
        source_revision="restricted-topology-offset-vertices-r1",
    )
    center: _Point3 = (
        Fraction(float(profile.center[0])),
        Fraction(float(profile.center[1])),
        Fraction(float(profile.center[2])),
    )
    faces, _triangles = _octahedral_surface(
        center,
        Fraction.from_float(profile.radius),
        vertex_offset=12,
    )
    assert _topology_matches(profile, np.asarray(faces, dtype=np.int64))


def test_restricted_region_sites_require_strict_side_but_not_cover_clearance() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-region-site-signs",
        source_revision="restricted-region-site-signs-r1",
    )
    candidates = np.asarray(
        (
            (0.75 - 1.0e-8, 0.0, 0.0),
            (0.75, 0.0, 0.0),
            (0.75 + 1.0e-8, 0.0, 0.0),
            (1.6, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    carrier = np.asarray(
        ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        _strict_region_site_mask(profile, candidates, carrier),
        np.asarray((True, False, True, False), dtype=np.bool_),
    )


def test_restricted_inside_witness_uses_a_certified_cell_star() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-witness-star",
        source_revision="restricted-witness-star-r1",
    )
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.25, 0.0, 0.0),
            (0.0, 0.25, 0.0),
            (0.0, 0.0, 0.25),
            (0.0, 0.0, -0.25),
        ),
        dtype=np.float64,
    )
    tetrahedra = np.asarray(
        ((0, 1, 2, 3), (0, 1, 2, 4)),
        dtype=np.int64,
    )
    assert _inside_seed_rows(profile, points, tetrahedra) == (0, 1)
    touching_source = points.copy()
    touching_source[3] = (0.75, 0.0, 0.0)
    with pytest.raises(MeshingFailure, match="boundary-free ball"):
        _inside_seed_rows(
            profile,
            touching_source,
            np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        )


def test_restricted_projection_degree_refuses_an_off_source_boundary() -> None:
    geometry, _grid = _geometry_and_grid()
    profile = AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id="restricted-off-source",
        source_revision="restricted-off-source-r1",
    )
    center: _Point3 = (
        Fraction(float(profile.center[0])),
        Fraction(float(profile.center[1])),
        Fraction(float(profile.center[2])),
    )
    radius = Fraction.from_float(profile.radius)
    tube = Fraction.from_float(profile.reach_lower)
    _faces, triangles = _octahedral_surface(center, radius + 3 * tube)
    with pytest.raises(RestrictedDelaunayCoverageUnavailable, match="preimages"):
        _prove_degree_one_normal_fiber(profile, triangles, tube)


def test_native_restricted_torus_projection_degree_accepts_complete_volume() -> None:
    contract = SpatialCoordinateContract.si()
    geometry = Torus(
        (0.0, 0.0, 0.0),
        0.6,
        1.2,
        feature_id="restricted-projection-degree-torus",
    ).compile()
    profile = AnalyticImplicitProfile(
        geometry,
        contract,
        source_id="restricted-projection-degree-torus",
        source_revision="restricted-projection-degree-torus-revision",
    )
    scope = MeshingScope(
        profile.source_id,
        profile.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        profile.scope_entity_set_id(2),
        profile.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.8,
    )
    construction = prepare_native_implicit(
        profile,
        specification,
        NativeMeshingOptions(
            "implicit_restricted_delaunay",
            volume_schedule=NativeVolumeSchedule(),
        ),
        contract,
    ).execute()
    distance_witness = construction.certification.fidelity_witness
    coverage_witness = construction.certification.source_coverage_witness
    assert isinstance(distance_witness, SourceBoundaryDistanceWitness)
    assert isinstance(coverage_witness, RestrictedDelaunayBoundaryWitness)
    regions = construction.certification.cell_regions
    assert regions is not None
    proof = validate_restricted_delaunay_boundary_witness(
        construction.mesh,
        construction.geometry,
        profile,
        regions,
        distance_witness,
        coverage_witness,
    )
    assert proof.region_ids == ("interior",)
    assert proof.target_facet_count > 0
