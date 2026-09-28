import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.applications.foams import (
    FoamDynamicsPlan,
    FoamDynamicsState,
    FoamMaterialPlan,
    PreparedFoamDynamics,
    PreparedVortexSheetAir,
    RegionPressureAirPlan,
    VortexSheetAirPlan,
    VortexSheetAirStatus,
    VortexSheetCurvatureRelationCapacityError,
    VortexSheetCurvatureRelationStatus,
)
from phydrax.geometry.multiregion_surface import (
    EdgeSplitProposal,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceSeed,
    PreparedMultiRegionSurface,
    seed_double_bubble,
    seed_sphere,
)


SIGMA = 0.025


def _prepared_seed(
    seed: MultiRegionSurfaceSeed,
    *,
    resource_id: str,
    headroom: float = 1.0,
    event_capacity: int = 0,
    region_capacity: int | None = None,
    surface_tension_scale: float = 1.0,
    time_step: float = 1.0e-5,
    maximum_curvature_routes: int | None = None,
) -> tuple[PreparedVortexSheetAir, FoamDynamicsState]:
    base = seed.capacity_plan(
        resource_id=resource_id,
        headroom=headroom,
        event_capacity=event_capacity,
    )
    capacity = (
        base
        if region_capacity is None
        else MultiRegionSurfaceCapacityPlan(
            vertex_capacity=base.vertex_capacity,
            edge_capacity=base.edge_capacity,
            face_capacity=base.face_capacity,
            region_capacity=region_capacity,
            region_pair_capacity=base.region_pair_capacity,
            maximum_edge_valence=base.maximum_edge_valence,
            maximum_vertex_region_pairs=base.maximum_vertex_region_pairs,
            resource_id=resource_id,
            event_capacity=base.event_capacity,
            coordinate_dtype=base.coordinate_dtype,
            index_dtype=base.index_dtype,
        )
    )
    topology = seed.topology(capacity)
    surface_state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    volume = surface.region_volumes(surface_state.positions)[
        jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    ]
    dynamics_state = FoamDynamicsState(surface_state)
    dynamics = PreparedFoamDynamics(
        FoamDynamicsPlan(
            route="film-inertia",
            time_step=time_step,
            areal_mass=1.0,
            volume_tolerance=1.0e-10,
        ),
        surface,
        FoamMaterialPlan.soap_film(topology.region_ids, SIGMA),
        RegionPressureAirPlan.incompressible(volume),
        dynamics_state,
    )
    prepared = VortexSheetAirPlan(
        air_density=1.2,
        time_step=time_step,
        surface_tension_scale=surface_tension_scale,
        core_radius_fraction=0.4,
        fmm_depth=1,
        fmm_leaf_capacity=64,
        maximum_fmm_relative_error=0.15,
        maximum_curvature_routes=maximum_curvature_routes,
    ).prepare(dynamics, dynamics_state)
    return prepared, dynamics_state


def _problem(
    *,
    subdivisions: int = 0,
    headroom: float = 1.0,
    event_capacity: int = 0,
    region_capacity: int | None = None,
    surface_tension_scale: float = 1.0,
    time_step: float = 1.0e-5,
    maximum_curvature_routes: int | None = None,
) -> tuple[PreparedVortexSheetAir, FoamDynamicsState]:
    return _prepared_seed(
        seed_sphere(1.0, subdivisions=subdivisions),
        resource_id="vortex-sheet-air-test",
        headroom=headroom,
        event_capacity=event_capacity,
        region_capacity=region_capacity,
        surface_tension_scale=surface_tension_scale,
        time_step=time_step,
        maximum_curvature_routes=maximum_curvature_routes,
    )


def _bounded_dense_slot_curvature(
    prepared: PreparedVortexSheetAir, positions: Array, /
) -> np.ndarray:
    topology = prepared.topology
    points = np.asarray(positions)
    edges = np.asarray(topology.edges, dtype=np.int64)
    wedges = prepared.dynamics.surface.junction_wedges(positions)
    angles = np.asarray(wedges.angles)
    regions = np.asarray(wedges.regions, dtype=np.int64)
    valid = np.asarray(wedges.valid)
    dense = np.zeros((topology.vertex_capacity, topology.region_count))
    for edge_index in range(topology.edge_count):
        edge = edges[edge_index]
        length = np.linalg.norm(points[edge[1]] - points[edge[0]])
        for wedge_index in np.flatnonzero(valid[edge_index]):
            contribution = 0.5 * length * (np.pi - angles[edge_index, wedge_index])
            dense[edge, regions[edge_index, wedge_index]] += contribution
    pair_slots = np.maximum(np.asarray(topology.vertex_pair_slots), 0)
    pair_regions = np.asarray(topology.region_pairs)[pair_slots]
    vertices = np.arange(topology.vertex_capacity)[:, None, None]
    return dense[vertices, pair_regions]


def _dense_source_reference(
    prepared: PreparedVortexSheetAir, positions: Array, /
) -> Array:
    topology = prepared.topology
    slot_curvature = jnp.asarray(_bounded_dense_slot_curvature(prepared, positions))
    pair_slots = jnp.maximum(topology.vertex_pair_slots, 0)
    areas = prepared.dynamics.surface.slot_areas(positions)
    raw = (
        prepared.plan.surface_tension_scale
        * (0.5 * prepared.pair_tension[pair_slots])
        * (slot_curvature[..., 0] - slot_curvature[..., 1])
        / (
            prepared.plan.air_density
            * jnp.where(topology.slot_active, areas, 1.0)
        )
    )
    return prepared.gauge(raw)[0]


def _quad_cell_seed() -> MultiRegionSurfaceSeed:
    outer = np.asarray(
        (
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        )
    )
    points = np.concatenate((np.zeros((1, 3)), outer), axis=0)
    cell_centers = -0.25 * outer
    faces: list[tuple[int, int, int]] = []
    labels: list[tuple[int, int]] = []
    for first in range(4):
        for second in range(first + 1, 4):
            remaining = [value for value in range(4) if value not in (first, second)]
            triangle = [0, first + 1, second + 1]
            corners = points[triangle]
            normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            left, right = remaining
            if np.dot(normal, cell_centers[right] - cell_centers[left]) < 0.0:
                left, right = right, left
            faces.append((triangle[0], triangle[1], triangle[2]))
            labels.append((left, right))
    for region in range(4):
        triangle = [vertex + 1 for vertex in range(4) if vertex != region]
        corners = points[triangle]
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        face_center = np.mean(corners, axis=0)
        if np.dot(normal, face_center - cell_centers[region]) < 0.0:
            triangle[1], triangle[2] = triangle[2], triangle[1]
        faces.append((triangle[0], triangle[1], triangle[2]))
        labels.append((region, 4))
    return MultiRegionSurfaceSeed(
        points,
        np.asarray(faces),
        np.asarray(labels),
        ("q0", "q1", "q2", "q3", "ambient"),
        ("finite", "finite", "finite", "finite", "boundary"),
        source="vortex-sheet-quad-cell",
    )


def test_pairwise_gauge_and_intrinsic_sheet_strength_identity() -> None:
    prepared, dynamics = _problem()
    topology = prepared.topology
    values = jnp.arange(
        topology.vertex_capacity * topology.slot_width, dtype=jnp.float64
    ).reshape((topology.vertex_capacity, topology.slot_width))

    state = prepared.initialize(dynamics, values)
    gauged, means = prepared.gauge(state.circulation)
    gradient, strength = prepared.intrinsic_sheet_gradient(
        gauged, state.surface.positions
    )
    geometry = prepared.dynamics.surface.evaluate(
        state.surface, prepared.dynamics.face_tension
    )
    unit_normal = geometry.face_normals
    canonical_normal = unit_normal * topology.face_pair_signs[:, None]

    assert float(jnp.max(jnp.abs(means))) < 1.0e-14
    np.testing.assert_allclose(jnp.sum(gradient * unit_normal, axis=1), 0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        strength,
        jnp.cross(canonical_normal, gradient),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_zero_surface_tension_keeps_circulation_and_fmm_matches_direct() -> None:
    prepared, dynamics = _problem(surface_tension_scale=0.0, time_step=1.0e-12)
    topology = prepared.topology
    vertex_values = 1.0e-4 * dynamics.surface.positions[:, 2]
    circulation = jnp.broadcast_to(
        vertex_values[:, None], (topology.vertex_capacity, topology.slot_width)
    )
    state = prepared.initialize(dynamics, circulation)

    result = prepared.fixed_topology_step(state)

    assert result.successful, (
        int(result.evidence.status),
        bool(result.evidence.fmm_successful),
        float(result.evidence.fmm_relative_l2_error),
        bool(result.evidence.ccd_certified),
        float(result.evidence.volume_residual),
        int(result.evidence.constraint_rank),
    )
    assert int(result.evidence.status) == VortexSheetAirStatus.COMPLETED
    np.testing.assert_array_equal(result.state.circulation, state.circulation)
    assert float(result.evidence.circulation_conservation_residual) == 0.0
    assert float(result.evidence.maximum_gauge_residual) < 1.0e-14
    assert bool(result.evidence.direct_reference_evaluated)
    assert float(result.evidence.fmm_relative_l2_error) < 0.15
    assert float(result.evidence.volume_residual) < 1.0e-10
    assert bool(result.evidence.ccd_certified)
    assert result.evidence.no_circulation_smoothing
    assert float(result.evidence.minimum_core_radius) > 0.0
    assert int(result.evidence.constraint_rank) == 1


def test_curvature_tension_source_uses_pairwise_gauge() -> None:
    prepared, dynamics = _problem(subdivisions=1)
    points = dynamics.surface.positions
    radius = jnp.linalg.norm(points, axis=1)
    cosine = jnp.where(radius > 0.0, points[:, 2] / radius, 0.0)
    mode = 0.5 * (3.0 * cosine * cosine - 1.0)
    deformed = points * (1.0 + 0.03 * mode)[:, None]
    changed_surface = dynamics.surface.with_positions(deformed)
    changed = FoamDynamicsState(changed_surface)
    surface = PreparedMultiRegionSurface(prepared.topology, changed_surface)
    volume = surface.region_volumes(deformed)[
        jnp.asarray(prepared.topology.finite_region_indices, dtype=jnp.int32)
    ]
    changed_dynamics = PreparedFoamDynamics(
        prepared.dynamics.plan,
        surface,
        prepared.dynamics.material,
        RegionPressureAirPlan.incompressible(volume),
        changed,
    )
    changed_prepared = prepared.plan.prepare(changed_dynamics, changed)
    state = changed_prepared.initialize(changed)

    source = changed_prepared.circulation_source(state)
    _, means = changed_prepared.gauge(source)

    assert float(jnp.linalg.norm(source)) > 0.0
    assert float(jnp.max(jnp.abs(means))) < 1.0e-13


def test_signed_curvature_source_matches_bounded_dense_reference_and_scales() -> None:
    prepared, dynamics = _problem(subdivisions=1)
    points = dynamics.surface.positions
    radius = jnp.linalg.norm(points, axis=1)
    cosine = jnp.where(radius > 0.0, points[:, 2] / radius, 0.0)
    mode = 0.5 * (3.0 * cosine * cosine - 1.0)
    deformed = points * (1.0 + 0.03 * mode)[:, None]
    state = prepared.initialize(
        FoamDynamicsState(dynamics.surface.with_positions(deformed))
    )

    expected = _dense_source_reference(prepared, deformed)
    source = prepared.circulation_source(state)
    np.testing.assert_allclose(source, expected, rtol=2.0e-14, atol=2.0e-14)

    scaled_positions = 2.0 * deformed
    scaled = prepared.initialize(
        FoamDynamicsState(dynamics.surface.with_positions(scaled_positions))
    )
    curvature = _bounded_dense_slot_curvature(prepared, deformed)
    scaled_curvature = _bounded_dense_slot_curvature(prepared, scaled_positions)
    scaled_source = prepared.circulation_source(scaled)
    np.testing.assert_allclose(
        scaled_curvature, 2.0 * curvature, rtol=2.0e-14, atol=2.0e-14
    )
    np.testing.assert_allclose(scaled_source, 0.5 * source, rtol=2.0e-13, atol=2.0e-13)


def test_curvature_relation_retention_is_independent_of_region_capacity() -> None:
    small, _ = _problem(region_capacity=2)
    large, _ = _problem(region_capacity=256)
    small_relation = small.curvature_slots
    large_relation = large.curvature_slots
    small_bytes = sum(
        np.asarray(value).nbytes
        for value in (
            small_relation.source_indices,
            small_relation.target_indices,
            small_relation.valid,
            small.curvature_route_regions,
        )
    )
    large_bytes = sum(
        np.asarray(value).nbytes
        for value in (
            large_relation.source_indices,
            large_relation.target_indices,
            large_relation.valid,
            large.curvature_route_regions,
        )
    )

    assert small.topology.region_capacity == 2
    assert large.topology.region_capacity == 256
    assert small_relation.target_size == (
        small.topology.vertex_capacity * small.topology.slot_width * 2
    )
    assert large_relation.target_size == small_relation.target_size
    assert large_relation.capacity == small_relation.capacity
    assert large_bytes == small_bytes
    assert small.curvature_relation_evidence.logical_retained_bytes == small_bytes
    assert large.curvature_relation_evidence.logical_retained_bytes == large_bytes


def test_curvature_relation_capacity_overflow_is_refused_with_evidence() -> None:
    with pytest.raises(VortexSheetCurvatureRelationCapacityError) as caught:
        _problem(maximum_curvature_routes=1)

    evidence = caught.value.evidence
    assert evidence.status is VortexSheetCurvatureRelationStatus.ROUTE_CAPACITY_EXCEEDED
    assert not evidence.accepted
    assert evidence.required_routes > evidence.route_capacity
    assert evidence.route_capacity == 1
    assert evidence.retained_routes == 0
    assert evidence.logical_retained_bytes == 0


def test_triple_junction_curvature_preserves_canonical_pair_signs() -> None:
    prepared, dynamics = _prepared_seed(
        seed_double_bubble(1.0, 0.75, ring_points=12),
        resource_id="vortex-sheet-triple",
    )
    points = dynamics.surface.positions
    deformed = points * (1.0 + 0.025 * points[:, 2])[:, None]
    state = prepared.initialize(
        FoamDynamicsState(dynamics.surface.with_positions(deformed))
    )

    source = prepared.circulation_source(state)
    expected = _dense_source_reference(prepared, deformed)

    assert prepared.topology.valence_width == 3
    assert float(jnp.max(jnp.abs(source))) > 0.0
    np.testing.assert_allclose(source, expected, rtol=2.0e-13, atol=2.0e-13)


def test_quad_junction_curvature_preserves_canonical_pair_signs() -> None:
    prepared, dynamics = _prepared_seed(
        _quad_cell_seed(),
        resource_id="vortex-sheet-quad",
    )
    points = dynamics.surface.positions
    displacement = jnp.asarray((0.07, -0.03, 0.04), dtype=points.dtype)
    deformed = points.at[0].add(displacement)
    state = prepared.initialize(
        FoamDynamicsState(dynamics.surface.with_positions(deformed))
    )

    source = prepared.circulation_source(state)
    expected = _dense_source_reference(prepared, deformed)

    assert int(jnp.max(jnp.sum(prepared.topology.slot_active, axis=1))) == 6
    assert float(jnp.max(jnp.abs(source))) > 0.0
    np.testing.assert_allclose(source, expected, rtol=2.0e-13, atol=2.0e-13)


def test_curvature_source_jit_matches_bounded_dense_reference() -> None:
    prepared, dynamics = _problem(subdivisions=1)
    state = prepared.initialize(dynamics)
    source = eqx.filter_jit(
        lambda compiled, value: compiled.circulation_source(value)
    )(prepared, state)

    np.testing.assert_allclose(
        source,
        _dense_source_reference(prepared, state.surface.positions),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_topology_epoch_transfer_conserves_circulation_content() -> None:
    prepared, dynamics = _problem(
        headroom=2.0,
        event_capacity=8,
        surface_tension_scale=0.0,
        time_step=1.0e-12,
    )
    topology = prepared.topology
    circulation = jnp.broadcast_to(
        1.0e-4 * dynamics.surface.positions[:, 2, None],
        (topology.vertex_capacity, topology.slot_width),
    )
    state = prepared.initialize(dynamics, circulation)
    edge = np.asarray(topology.edges[3], dtype=np.int64)
    ids = np.asarray(topology.vertex_global_ids, dtype=np.int64)[edge]

    result = prepared.advance(
        state,
        events=(EdgeSplitProposal((int(ids[0]), int(ids[1]))),),
    )

    assert result.successful
    assert bool(result.evidence.topology_changed)
    assert bool(result.evidence.rebuild_required)
    assert int(result.evidence.target_epoch) == int(result.evidence.source_epoch) + 1
    assert float(result.evidence.circulation_transfer_residual) < 1.0e-12
    active = np.asarray(result.topology.slot_active)
    pairs = np.asarray(result.topology.vertex_pair_slots)
    values = np.asarray(result.state.circulation)
    for pair in range(result.topology.region_pair_count):
        mask = active & (pairs == pair)
        assert float(np.mean(values[mask])) == pytest.approx(0.0, abs=1.0e-13)
