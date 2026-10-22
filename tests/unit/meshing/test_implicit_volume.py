"""Implicit domain preparation, query budgets and native publication."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._meshcore import (
    exact_orient3d,
    IncrementalDelaunay3D,
    MeshcoreStatus,
    NativeExecutionScope,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry.implicit._adaptive_discovery import ImplicitVolumeClass
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.geometry.implicit._enclosure import field_bounds
from phydrax.geometry.implicit._policy import ImplicitDiscoveryEnclosure
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingLimits,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._implicit_volume import (
    _protected_insertion_batch,
    _restricted_facet_deviation,
    implicit_source_budget,
    ImplicitVolumeWorkStop,
    prepare_implicit_volume_domain,
)
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._volume_generation import (
    native_volume_execution_budget,
    NativeVolumeSchedule,
)
from phydrax.units import METER
from tests._support.implicit_meshing import (
    assert_box_coverage,
    execution_scope as execution_scope,
    IMPLICIT_DOMAIN,
    implicit_query,
    implicit_specification,
    NATIVE,
    source_budget as source_budget,
)


# Inherited empirical allowance for the independent host barycentric solve
# in the tiny-component fixture; unchanged from its pre-refactor assertions.
BARYCENTRIC_CONTAINMENT_ATOL = 1.0e-12


@NATIVE
def test_certified_inside_box_tetrahedra_cover_declared_domain(
    execution_scope: NativeExecutionScope,
) -> None:
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 3.0).compile(coordinate_unit=METER)
    )
    work = prepare_implicit_volume_domain(
        query,
        implicit_specification(),
        NativeVolumeSchedule(refinement_rounds=2),
        scope=execution_scope,
    )
    assert_box_coverage(work)
    assert work.stop is ImplicitVolumeWorkStop.ALL_CLASSIFIED
    assert np.all(work.classes == int(ImplicitVolumeClass.INSIDE))
    assert np.all(work.value_upper < 0.0)
    assert work.unresolved_tetrahedra.size == 0


@NATIVE
def test_tiny_bounded_component_cannot_be_discarded_by_coarse_samples(
    execution_scope: NativeExecutionScope,
) -> None:
    center = np.asarray((0.37, 0.13, -0.36), dtype=np.float64)
    query = implicit_query(
        phx.geometry.Sphere(tuple(center), 0.04).compile(coordinate_unit=METER)
    )
    work = prepare_implicit_volume_domain(
        query,
        implicit_specification(),
        NativeVolumeSchedule(refinement_rounds=2),
        scope=execution_scope,
    )
    assert_box_coverage(work)
    # Independent barycentric location of the required component center.
    corners = work.points[work.tetrahedra]
    matrix = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
    barycentric = np.linalg.solve(matrix, (center - corners[:, 0])[..., None])[..., 0]
    contains = np.all(barycentric >= -BARYCENTRIC_CONTAINMENT_ATOL, axis=1) & (
        np.sum(barycentric, axis=1) <= 1.0 + BARYCENTRIC_CONTAINMENT_ATOL
    )
    assert np.any(contains)
    assert np.all(work.classes[contains] != int(ImplicitVolumeClass.OUTSIDE))
    assert work.unresolved_tetrahedra.size > 0
    assert work.refinement_rounds == 2
    # No whole-cell inside verdict may contradict the analytic ball enclosure.
    inside_corners = corners[work.inside_tetrahedra]
    assert np.all(np.linalg.norm(inside_corners - center, axis=2) < 0.04)


@NATIVE
def test_geometry_budget_retains_unqueried_domain_cells_with_unbounded_intervals(
    execution_scope: NativeExecutionScope,
) -> None:
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER)
    )
    work = prepare_implicit_volume_domain(
        query,
        implicit_specification(MeshingLimits(maximum_geometry_queries=1)),
        NativeVolumeSchedule(refinement_rounds=2),
        scope=execution_scope,
    )
    assert_box_coverage(work)
    assert work.stop is ImplicitVolumeWorkStop.GEOMETRY_QUERY_LIMIT
    assert work.geometry_queries == 1
    assert np.all(work.classes[~work.queried] == int(ImplicitVolumeClass.UNKNOWN))
    assert np.all(np.isneginf(work.value_lower[~work.queried]))
    assert np.all(np.isposinf(work.value_upper[~work.queried]))
    assert "maximum_scratch_bytes" in work.unenforced_limits


@NATIVE
def test_workset_refuses_changed_source_state_even_with_same_query_cover(
    execution_scope: NativeExecutionScope,
) -> None:
    first = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 3.0).compile(coordinate_unit=METER)
    )
    second = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 4.0).compile(coordinate_unit=METER)
    )
    specification = implicit_specification()
    work = prepare_implicit_volume_domain(
        first, specification, NativeVolumeSchedule(), scope=execution_scope
    )
    work.require_current(first, specification)
    with pytest.raises(ValueError, match="stale"):
        work.require_current(second, specification)


def test_sampled_queries_cannot_supply_certified_domain_work() -> None:
    query = implicit_query(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER),
        "sampled",
    )
    with pytest.raises(ValueError, match=r"prepare_implicit_volume_domain\.query"):
        prepare_implicit_volume_domain(
            query,
            implicit_specification(),
            NativeVolumeSchedule(),
            scope=NativeExecutionScope(MeshingLimits().native_limits()),
        )


@NATIVE
@pytest.mark.parametrize(
    ("enclosure", "query_count"),
    (("sampled", 18), ("lipschitz", 20), ("interval", 4)),
)
def test_enclosure_batches_consume_the_actual_original_query_allowance(
    enclosure: ImplicitDiscoveryEnclosure,
    query_count: int,
) -> None:
    geometry = phx.geometry.Sphere((0.013, -0.021, 0.007), 0.75).compile(
        coordinate_unit=METER
    )
    bounds = field_bounds(geometry, enclosure)
    lower = np.asarray(((-0.2, -0.2, -0.2), (0.1, 0.1, 0.1)))
    upper = lower + 0.1
    with native_volume_execution_budget(
        MeshingLimits(maximum_geometry_queries=query_count),
    ) as execution:
        with implicit_source_budget(
            MeshingLimits(maximum_geometry_queries=query_count),
            scope=execution,
            queries_upper=query_count,
        ) as budget:
            result = bounds.boxes(lower, upper, budget=budget)
    assert execution.evidence is not None
    assert execution.evidence.externally_charged_geometry_queries == query_count
    assert np.all(result.value_lower <= result.value_upper)
    with pytest.raises(MeshingFailure) as failure:
        with native_volume_execution_budget(
            MeshingLimits(maximum_geometry_queries=query_count - 1),
        ) as execution:
            with implicit_source_budget(
                MeshingLimits(maximum_geometry_queries=query_count - 1),
                scope=execution,
                queries_upper=query_count - 1,
            ) as budget:
                bounds.boxes(lower, upper, budget=budget)
    assert failure.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED


@NATIVE
@pytest.mark.parametrize("family", ("sphere", "ring-torus"))
def test_native_provider_publishes_source_certified_restricted_volume(
    family: str,
) -> None:
    meshing = phx.meshing
    if family == "sphere":
        geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(
            coordinate_unit=METER
        )
        radius, major = 0.371, 0.0
    else:
        geometry = phx.geometry.Torus((0.0, 0.0, 0.0), 0.6, 1.0).compile(
            coordinate_unit=METER
        )
        radius, major = 0.2, 0.8
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(1.2 * IMPLICIT_DOMAIN))
    source = meshing.NativeImplicitSource(geometry, grid, "implicit", "state")
    scope = implicit_specification().boundary_scope
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, 0.2, strength=SizeControlStrength.SOFT),
        ),
    )
    result = (
        meshing.NativeMeshingProvider(
            meshing.NativeMeshingOptions("implicit_restricted_delaunay")
        )
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    assert isinstance(result, meshing.CellMeshingResult)
    certificate = result.certification
    assert certificate is not None and certificate.passed
    assert certificate.coverage is not None and certificate.coverage.status == "certified"
    fidelity = certificate.fidelity
    assert fidelity is not None and fidelity.status == "certified"
    projection = fidelity.projection_coverage
    assert projection is not None and projection.status == "certified"
    assert projection.fiber_crossing_classes.count("interior") == 1
    points = np.asarray(result.mesh.coordinates)
    cells = np.asarray(result.mesh.blocks[0].vertices)
    corners = points[cells]
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    assert np.all(volumes > 0.0)
    deviation = projection.forward_upper
    volume = float(np.sum(volumes))
    if family == "sphere":
        lower = 4.0 * np.pi / 3.0 * (radius - deviation) ** 3
        upper = 4.0 * np.pi / 3.0 * (radius + deviation) ** 3
    else:
        lower = 2.0 * np.pi**2 * major * (radius - deviation) ** 2
        upper = 2.0 * np.pi**2 * major * (radius + deviation) ** 2
    assert lower <= volume <= upper


@NATIVE
def test_implicit_boundary_tags_follow_sci_through_native_coefficient_regrouping() -> (
    None
):
    # Source1's fixed acceptance fixture: sphere/.75, count9, [-1.4,1.4]^3.
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="design-sphere"
    ).compile(coordinate_unit=METER)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(9) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(np.asarray(((-1.4,) * 3, (1.4,) * 3), dtype=np.float64))
    revision = canonical_fingerprint(
        {
            "kind": "design-qualification-source",
            "parameters": tuple(np.asarray(value) for value in geometry.state.values),
            "parameter_ids": tuple(str(item) for item in geometry.schema.parameter_ids),
            "lattice_count": 9,
            "bounds": ((-1.4,) * 3, (1.4,) * 3),
        }
    )
    source = phx.meshing.NativeImplicitSource(geometry, grid, "design-sphere", revision)
    scope = phx.meshing.MeshingScope(
        source.source_id,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "design-sphere-boundary",
        np.asarray((0,), dtype=np.int64),
    )
    limits = MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=60_000,
        maximum_faces=20_000,
        maximum_cells=20_000,
        maximum_connectivity_entries=240_000,
        maximum_data_bytes=8 * 1024**3,
        maximum_scratch_bytes=8 * 1024**3,
        maximum_work_units=320_000,
        maximum_cavity_cells=20_000,
        maximum_geometry_queries=1_280_000,
        maximum_wall_seconds=120.0,
    )
    request = phx.meshing.SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 2.8 / 8, strength=SizeControlStrength.SOFT),
        ),
        limits=limits,
    )
    options = phx.meshing.NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            maximum_intersection_pairs=640_000,
        ),
    )
    plan = phx.meshing.NativeMeshingProvider(options).plan(
        source,
        request,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    initial = plan.execute(geometry.state)
    assert initial.boundary is not None
    source_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in initial.boundary.mesh.blocks]
    )
    source_tags = dict(
        zip(source_ids.tolist(), initial.boundary.metadata.cell_tags, strict=True)
    )
    assert initial.certification is not None
    query = initial.certification.request.source
    assert query is not None
    profile = (
        query.profile
        if isinstance(query, phx.geometry.ImplicitProjectionBoundarySource)
        else query
    )
    assert isinstance(profile, AnalyticImplicitProfile)
    adapted = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            initial,
            phx.meshing.MarkedMeshAdaptation(initial.mesh.blocks[0].global_ids),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                limits=request.limits,
                compatibility=phx.meshing.BisectionCompatibility.CONFORMING_CLOSURE,
                association_transfer=phx.meshing.ImplicitAssociationTransfer(
                    profile, maximum_queries=request.limits.maximum_geometry_queries
                ),
            ),
        )
    )
    boundary = adapted.target.boundary
    assert boundary is not None and adapted.lineage is not None
    record = adapted.lineage.entity_lineage(2)
    parents: dict[int, set[str]] = {}
    for parent, child in zip(
        np.asarray(record.source_global_ids),
        np.asarray(record.target_global_ids),
        strict=True,
    ):
        parents.setdefault(int(child), set()).add(source_tags[int(parent)])
    target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in boundary.mesh.blocks]
    )
    assert set(target_ids.tolist()) == set(parents)
    for identifier, tag in zip(target_ids, boundary.metadata.cell_tags, strict=True):
        assert parents[int(identifier)] == {tag}
    assert boundary.metadata.source_id == initial.boundary.metadata.source_id
    assert boundary.metadata.source_revision == initial.boundary.metadata.source_revision
    assert tuple(value.name for value in boundary.selections) == tuple(
        value.name for value in initial.boundary.selections
    )
    assert tuple(value.name for value in boundary.interfaces) == tuple(
        value.name for value in initial.boundary.interfaces
    )


def test_restricted_facet_distance_encloses_triangle_without_axis_box_overrefinement(
    source_budget: CoordinateEnclosureBudget,
) -> None:
    radius, offset, fidelity = 0.371, 0.06, 0.02
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), radius).compile(coordinate_unit=METER)
    profile = AnalyticImplicitProfile(
        geometry, phx.SpatialCoordinateContract.si(), budget=source_budget
    )
    radial = np.sqrt(radius * radius - offset * offset)
    points = np.asarray(
        ((radius, 0.0, 0.0), (radial, offset, 0.0), (radial, 0.0, offset)),
        dtype=np.float64,
    )
    # Both facets share their vertices: charge each source point only once.
    facets = np.asarray(((0, 1, 2), (0, 2, 1)), dtype=np.int64)
    triangles = points[facets]
    boxes = profile.field_bounds.boxes(
        np.min(triangles, axis=1), np.max(triangles, axis=1), budget=source_budget
    )
    box_deviation = np.maximum(np.abs(boxes.value_lower), np.abs(boxes.value_upper))
    diameter = np.full(
        (2,), np.nextafter(np.sqrt(2.0) * offset, np.inf), dtype=np.float64
    )
    before = source_budget.ticket.remaining_geometry_queries
    deviation = _restricted_facet_deviation(
        profile, points, facets, box_deviation, diameter, budget=source_budget
    )
    assert before - source_budget.ticket.remaining_geometry_queries == 3
    assert np.all(box_deviation > 0.25 * fidelity)
    assert np.all(diameter > fidelity)
    assert np.all(deviation < 0.25 * fidelity)
    axis = np.linspace(0.0, 1.0, 25, dtype=np.float64)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    selected = first + second <= 1.0
    weights = np.stack(
        (1.0 - first[selected] - second[selected], first[selected], second[selected]),
        axis=1,
    )
    probes = weights @ points
    independent_distance = np.abs(np.linalg.norm(probes, axis=1) - radius)
    assert np.all(independent_distance <= deviation[0])


def test_restricted_facet_distance_retains_box_bound_outside_hessian_tube(
    source_budget: CoordinateEnclosureBudget,
) -> None:
    profile = AnalyticImplicitProfile(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER),
        phx.SpatialCoordinateContract.si(),
        budget=source_budget,
    )
    points = np.asarray(
        ((0.371, 0.0, 0.0), (-0.371, 0.0, 0.0), (0.0, 0.371, 0.0)),
        dtype=np.float64,
    )
    facets = np.asarray(((0, 1, 2),), dtype=np.int64)
    boxes = profile.field_bounds.boxes(
        np.min(points, axis=0)[None], np.max(points, axis=0)[None], budget=source_budget
    )
    box_deviation = np.maximum(np.abs(boxes.value_lower), np.abs(boxes.value_upper))
    deviation = _restricted_facet_deviation(
        profile,
        points,
        facets,
        box_deviation,
        np.asarray((0.742,), dtype=np.float64),
        budget=source_budget,
    )
    np.testing.assert_array_equal(deviation, box_deviation)
    # The source passes through the vertices but the triangle contains the
    # medial axis; a vertex-only distance verdict would incorrectly return zero.
    assert deviation[0] >= profile.radius


def test_restricted_insertion_batch_preserves_entire_empty_balls() -> None:
    candidates = np.asarray(
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.5, 0.0, 0.0), (1.75, 0.0, 0.0)),
        dtype=np.float64,
    )
    radii = np.ones((4,), dtype=np.float64)
    expected = candidates[[0, 2]]
    for scale in (0.5, 1.0, 2.0):
        accepted = _protected_insertion_batch(scale * candidates, scale * radii)
        np.testing.assert_array_equal(accepted, scale * expected)
        assert np.linalg.norm(accepted[0] - accepted[1]) >= scale


@NATIVE
def test_restricted_insertion_makes_geometric_progress_with_deterministic_counters() -> (
    None
):
    axis = np.asarray((-1.0, 1.0), dtype=np.float64)
    seeds = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    candidates = np.asarray(
        ((0.0, 0.0, 0.0), (0.125, 0.0, 0.0), (0.75, 0.0, 0.0)),
        dtype=np.float64,
    )
    batch = _protected_insertion_batch(candidates, np.full((3,), 0.5, dtype=np.float64))
    np.testing.assert_array_equal(batch, candidates[[0, 2]])
    observations: list[tuple[np.ndarray, np.ndarray, tuple[tuple[str, int], ...]]] = []
    for _ in range(2):
        state = IncrementalDelaunay3D(
            seeds, max_vertices=16, max_tetrahedra=128, max_cavity=64
        )
        try:
            before = state.statistics()
            _, status = state.insert(batch, work_limit=100_000)
            after = state.statistics()
            snapshot = state.finalize()
            np.testing.assert_array_equal(status, MeshcoreStatus.OK)
            assert after.live_vertices - before.live_vertices == batch.shape[0]
            assert (
                after.committed_insertions - before.committed_insertions == batch.shape[0]
            )
            assert after.refused_insertions == before.refused_insertions == 0
            assert after.work > before.work
            assert after.vertex_ids <= 16 and after.finite_cells <= 128
            corners = snapshot.points[snapshot.cells]
            assert np.all(
                exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
                > 0
            )
            observations.append((snapshot.points, snapshot.cells, after.items()))
        finally:
            state.close()
    np.testing.assert_array_equal(observations[0][0], observations[1][0])
    np.testing.assert_array_equal(observations[0][1], observations[1][1])
    assert observations[0][2] == observations[1][2]


@NATIVE
def test_restricted_insertion_keeps_work_refusal_and_invalid_input_truthful() -> None:
    seeds = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    state = IncrementalDelaunay3D(seeds, max_vertices=8, max_tetrahedra=32, max_cavity=16)
    try:
        before = state.statistics()
        cells = state.finalize().cells
        vertices, status = state.insert(
            np.asarray(((0.125, 0.125, 0.125),), dtype=np.float64), work_limit=1
        )
        np.testing.assert_array_equal(vertices, -1)
        np.testing.assert_array_equal(status, MeshcoreStatus.CAPACITY_EXCEEDED)
        after = state.statistics()
        assert after.live_vertices == before.live_vertices
        assert after.committed_insertions == before.committed_insertions
        assert after.refused_insertions == before.refused_insertions + 1
        np.testing.assert_array_equal(state.finalize().cells, cells)
        with pytest.raises(ValueError, match="finite"):
            state.insert(np.asarray(((np.nan, 0.0, 0.0),), dtype=np.float64))
        assert state.statistics().items() == after.items()
    finally:
        state.close()


@NATIVE
def test_restricted_lf03_preserves_original_capacity_and_hard_fidelity() -> None:
    # LF-03's authored request is independent of the qualification runner.
    radius, fidelity, capacity = 0.371, 0.02, 20_000
    domain = np.asarray(((-1.2 * radius,) * 3, (1.2 * radius,) * 3), dtype=np.float64)
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), radius, feature_id="qualification-implicit-sphere"
    ).compile(coordinate_unit=METER)
    revision = canonical_fingerprint(
        {
            "kind": "qualification-analytic-sphere",
            "radius": radius,
            "center": (0.0, 0.0, 0.0),
            "state": tuple(np.asarray(value) for value in geometry.state.values),
            "domain": domain,
            "coordinates": phx.SpatialCoordinateContract.si().spatial_id,
        }
    )
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(domain, dtype=jnp.float64))
    source = phx.meshing.NativeImplicitSource(
        geometry, grid, "qualification-implicit-sphere", revision
    )
    scope = phx.meshing.MeshingScope(
        source.source_id,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "implicit-zero-set",
        np.asarray((0,), dtype=np.int64),
    )
    limits = MeshingLimits(
        maximum_vertices=capacity,
        maximum_edges=6 * capacity,
        maximum_faces=4 * capacity,
        maximum_cells=capacity,
        maximum_connectivity_entries=16 * capacity,
        maximum_data_bytes=8_589_934_592,
        maximum_scratch_bytes=8_589_934_592,
        maximum_work_units=256 * capacity,
        maximum_cavity_cells=capacity,
        maximum_geometry_queries=256 * capacity,
        maximum_wall_seconds=120.0,
    )
    h = 2.0 * radius / 4
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(
                scope, h, maximum_size=1.5 * h, strength=SizeControlStrength.HARD
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(relative_tolerance=0.5),
        protected_features=(
            phx.meshing.ProtectedFeature(
                scope,
                phx.meshing.FeatureKind.SURFACE,
                maximum_deviation=fidelity,
                hard=True,
            ),
        ),
        region_seeds=(
            phx.meshing.RegionSeed(
                np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
                "interior",
                "uniform-material",
                phx.meshing.RegionRole.USER,
            ),
        ),
        limits=limits,
    )
    options = phx.meshing.NativeMeshingOptions(
        "implicit_restricted_delaunay",
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=2.0,
            minimum_dihedral_degrees=10.0,
            refinement_rounds=8,
            improvement_passes=8,
        ),
    )
    assert specification.limits.maximum_cells == capacity
    assert specification.protected_features[0].maximum_deviation == fidelity
    result = (
        phx.meshing.NativeMeshingProvider(options)
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    certificate = result.certification
    assert certificate is not None and certificate.passed
    assert certificate.fidelity is not None and certificate.fidelity.status == "certified"
    projection = certificate.fidelity.projection_coverage
    assert projection is not None and projection.status == "certified"
    assert projection.forward_upper <= fidelity
    assert result.mesh.coordinates.shape[0] <= specification.limits.maximum_vertices
    assert result.mesh.blocks[0].vertices.shape[0] <= specification.limits.maximum_cells
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, phx.discretization.TetrahedralConnectivity)
    points = np.asarray(result.mesh.coordinates)
    boundary_edges = np.asarray(connectivity.edges)[
        np.asarray(connectivity.boundary_edges)
    ]
    lengths = np.linalg.norm(
        points[boundary_edges[:, 1]] - points[boundary_edges[:, 0]], axis=1
    )
    p50, p95 = np.quantile(lengths, (0.5, 0.95))
    assert np.all(np.abs(np.asarray((p50, p95)) - h) <= 0.5 * h)
    assert np.max(lengths) <= 1.5 * h * (1.0 + 0.5)
    assert result.compliance.passed
    execution = result.execution_evidence
    assert execution is not None
    execution.require_valid()
    assert execution.total_work_units <= limits.maximum_work_units
    next_metric_size = h / (2.0 ** (1.0 / 3.0))
    assert np.min(lengths) >= next_metric_size / np.sqrt(2.0)


@NATIVE
def test_sphere_geodesic_seeds_meet_joint_hard_size_and_fidelity(
    execution_scope: NativeExecutionScope,
) -> None:
    from phydrax.meshing._implicit_volume import (
        _sphere_geodesic_lattice,
        _sphere_seed_sites,
        prepare_restricted_implicit_volume,
    )
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.implicit import (
        IMPLICIT_QUALIFICATION_CASES,
        _implicit_qualification_source,
        _implicit_volume_request,
    )

    args = QualificationControls(implicit_case="sphere-refinement", adaptation_rounds=2)
    geometry, source, domain, witness = _implicit_qualification_source(
        args, IMPLICIT_QUALIFICATION_CASES[args.implicit_case]
    )
    fidelity = args.implicit_fidelity / 4.0
    specification = _implicit_volume_request(args, source, witness, fidelity)
    prepared = prepare_restricted_implicit_volume(
        geometry,
        domain,
        specification,
        NativeVolumeSchedule(refinement_rounds=8),
        phx.SpatialCoordinateContract.si(),
        scope=execution_scope,
    )
    with implicit_source_budget(
        specification.limits,
        scope=execution_scope,
        queries_upper=execution_scope.remaining().remaining_geometry_queries,
    ) as budget:
        points, cover = _sphere_seed_sites(
            prepared, specification.limits.maximum_vertices, budget=budget
        )
        directions, faces = _sphere_geodesic_lattice(4, 1, prepared, budget=budget)
    assert points.shape[0] == 10 * (4 * 4 + 4 + 1) + 2
    assert np.array_equal(points, args.implicit_radius * directions)
    edges = np.unique(
        np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2)), axis=1), axis=0
    )
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    control = specification.size_controls[0]
    assert isinstance(control, UniformSizeControl)
    assert np.all(
        np.abs(np.quantile(lengths, (0.5, 0.95)) - control.target_size)
        <= specification.size_compliance.tolerance(control.target_size)
    )
    assert cover <= prepared.seed_cover_radius
    assert np.max(np.abs(np.linalg.norm(points, axis=1) - args.implicit_radius)) <= fidelity
    corners = points[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    plane_distances = np.abs(np.sum(normals * corners[:, 0], axis=1)) / np.linalg.norm(
        normals, axis=1
    )
    assert args.implicit_radius - np.min(plane_distances) <= fidelity
