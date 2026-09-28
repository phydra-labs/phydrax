import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.foams import (
    FoamConstraintBasisStatus,
    FoamDynamicsPlan,
    FoamDynamicsRoute,
    FoamDynamicsState,
    FoamDynamicsStatus,
    FoamMaterialPlan,
    PreparedFoamDynamics,
    RegionPressureAirPlan,
)
from phydrax.bubble_dynamics import (
    BubbleEnvironment,
    CaloricIdealBubbleGasLaw,
)
from phydrax.geometry.multiregion_surface import (
    EdgeSplitProposal,
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
    seed_double_bubble,
    seed_sphere,
)
from phydrax.linalg import LinearSolveStatus


SIGMA = 0.025


def _sphere_problem(
    air: RegionPressureAirPlan,
    *,
    route: FoamDynamicsRoute = "overdamped",
    time_step: float = 1.0e-3,
    projection_iterations: int = 6,
    headroom: float = 1.0,
    event_capacity: int = 0,
) -> tuple[
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
    FoamDynamicsState,
    PreparedFoamDynamics,
]:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(
        seed.capacity_plan(
            resource_id="foam-dynamics-test",
            headroom=headroom,
            event_capacity=event_capacity,
        )
    )
    surface_state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    material = FoamMaterialPlan.soap_film(topology.region_ids, SIGMA)
    gas = None
    if air.route == "compartment-gas":
        volumes = surface.region_volumes(surface_state.positions)[
            jnp.asarray(topology.finite_region_indices)
        ]
        gas = air.initialize(volumes, jnp.asarray((0.12,)))
    state = FoamDynamicsState(surface_state, gas=gas)
    plan = FoamDynamicsPlan(
        route=route,
        time_step=time_step,
        steps=1,
        friction=5.0,
        areal_mass=2.0,
        projection_iterations=projection_iterations,
        volume_tolerance=1.0e-10,
        energy_tolerance=1.0e-8,
    )
    return (
        topology,
        surface,
        state,
        PreparedFoamDynamics(plan, surface, material, air, state),
    )


def _disconnected_tetra_bubbles(count: int, /) -> MultiRegionSurfaceSeed:
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    labels: list[tuple[int, int]] = []
    for region in range(count):
        offset = len(points)
        shift = 3.0 * region
        points.extend(
            (
                (shift, 0.0, 0.0),
                (shift + 1.0, 0.0, 0.0),
                (shift, 1.0, 0.0),
                (shift, 0.0, 1.0),
            )
        )
        faces.extend(
            (
                (offset + 1, offset + 2, offset + 3),
                (offset, offset + 3, offset + 2),
                (offset, offset + 1, offset + 3),
                (offset, offset + 2, offset + 1),
            )
        )
        labels.extend(((region, count),) * 4)
    return MultiRegionSurfaceSeed(
        np.asarray(points),
        np.asarray(faces),
        np.asarray(labels),
        tuple(f"bubble-{index}" for index in range(count)) + ("ambient",),
        ("finite",) * count + ("boundary",),
        source=f"disconnected-tetra-bubbles-{count}",
    )


def _edge_split(topology: MultiRegionSurfaceTopology, /) -> EdgeSplitProposal:
    edge = np.asarray(topology.edges[0], dtype=np.int64)
    ids = np.asarray(topology.vertex_global_ids, dtype=np.int64)[edge]
    return EdgeSplitProposal((int(ids[0]), int(ids[1])))


def test_volume_operator_actions_match_bounded_dense_reference_and_adjoint() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-volume-operator"))
    initial = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, initial)
    volume = surface.region_volumes(initial.positions)[topology.finite_region_indices[0]]
    air = RegionPressureAirPlan.incompressible(jnp.asarray((volume,)))
    _, surface, state, prepared = _sphere_problem(air)
    positions = state.surface.positions
    tangent = jnp.linspace(-0.3, 0.4, positions.size, dtype=positions.dtype).reshape(
        positions.shape
    )
    cotangent = jnp.asarray((1.7,), dtype=positions.dtype)

    operator = prepared.volume_operator(positions)
    dense = jax.jacrev(
        lambda values: surface.region_volumes(values)[prepared.finite_slots]
    )(positions)
    expected_jvp = np.tensordot(
        np.asarray(dense),
        np.asarray(tangent),
        axes=((1, 2), (0, 1)),
    )
    expected_vjp = np.tensordot(
        np.asarray(cotangent),
        np.asarray(dense),
        axes=(0, 0),
    )

    np.testing.assert_allclose(operator.linearization.primal, jnp.asarray((volume,)))
    np.testing.assert_allclose(operator.mv(tangent), expected_jvp, rtol=1.0e-13)
    np.testing.assert_allclose(
        operator.transpose_mv(cotangent),
        expected_vjp,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        jnp.vdot(operator.mv(tangent), cotangent),
        jnp.vdot(tangent, operator.transpose_mv(cotangent)),
        rtol=1.0e-14,
        atol=1.0e-15,
    )
    evidence = prepared.constraint_basis.evidence
    assert evidence.partition_count == 1
    assert evidence.closed_partition_count == 0
    assert evidence.constrained_count == evidence.numerical_rank == 1
    assert evidence.rank_check_actions == 2
    assert evidence.accepted


def test_many_active_regions_prepare_with_constraint_scaled_retained_evidence() -> None:
    count = 8
    seed = _disconnected_tetra_bubbles(count)
    topology = seed.topology(
        seed.capacity_plan(resource_id="foam-many-volume-constraints")
    )
    surface_state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    finite = jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    targets = surface.region_volumes(surface_state.positions)[finite]
    prepared = PreparedFoamDynamics(
        FoamDynamicsPlan(
            time_step=1.0e-5,
            maximum_constraint_entries=count**2,
            maximum_constraint_rank_actions=2 * count,
            maximum_constraint_preparation_bytes=1_000_000,
        ),
        surface,
        FoamMaterialPlan.soap_film(topology.region_ids, SIGMA),
        RegionPressureAirPlan.incompressible(targets),
        FoamDynamicsState(surface_state),
    )

    evidence = prepared.constraint_basis.evidence
    dense_jacobian_bytes = (
        count * surface_state.positions.size * np.dtype(np.float64).itemsize
    )
    assert evidence.partition_count == count
    assert evidence.closed_partition_count == 0
    assert evidence.constrained_count == evidence.numerical_rank == count
    assert evidence.required_constraint_entries == count**2
    assert evidence.rank_check_actions == 2 * count
    assert evidence.logical_retained_bytes == 2 * count * np.dtype(np.int32).itemsize
    assert evidence.preparation_bytes < dense_jacobian_bytes
    assert prepared.constrained_region_ids == topology.finite_region_ids
    assert prepared.dependent_region_ids == ()


def test_many_compressible_compartments_skip_constraint_basis_and_step() -> None:
    count = 513
    seed = _disconnected_tetra_bubbles(count)
    topology = seed.topology(
        seed.capacity_plan(resource_id="foam-many-compressible-compartments")
    )
    surface_state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    finite = jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    volumes = surface.region_volumes(surface_state.positions)[finite]
    air = RegionPressureAirPlan.compressible(
        CaloricIdealBubbleGasLaw(1.4),
        BubbleEnvironment(0.1, 293.15),
    )
    gas = air.initialize(volumes, jnp.full(volumes.shape, 0.12))
    assert gas is not None
    state = FoamDynamicsState(surface_state, gas=gas)
    plan = FoamDynamicsPlan(
        time_step=1.0e-8,
        maximum_constraint_entries=1,
        maximum_constraint_rank_actions=1,
        maximum_constraint_preparation_bytes=1,
    )
    material = FoamMaterialPlan.soap_film(topology.region_ids, SIGMA)
    prepared = PreparedFoamDynamics(
        plan,
        surface,
        material,
        air,
        state,
    )

    basis = prepared.constraint_basis
    evidence = basis.evidence
    assert evidence.status == FoamConstraintBasisStatus.NOT_APPLICABLE
    assert evidence.accepted
    assert evidence.finite_region_count == count
    assert evidence.constrained_count == evidence.dependent_count == 0
    assert evidence.numerical_rank == 0
    assert evidence.rank_condition is None
    assert evidence.required_constraint_entries == 0
    assert evidence.rank_check_actions == 0
    assert evidence.preparation_bytes == 0
    assert evidence.logical_retained_bytes == 0
    assert basis.finite_region_slots == ()
    assert prepared.constrained_rows == prepared.dependent_rows == ()
    assert prepared.constrained_slots.size == 0
    repeated = PreparedFoamDynamics(plan, surface, material, air, state)
    assert repeated.constraint_basis.basis_id == basis.basis_id
    assert repeated.prepared_id == prepared.prepared_id

    tangent = jnp.ones_like(surface_state.positions)
    weights = jnp.ones((count,), dtype=surface_state.positions.dtype)
    volume_operator = prepared.volume_operator(surface_state.positions)
    assert jnp.all(jnp.isfinite(volume_operator.linearization.primal))
    assert jnp.all(jnp.isfinite(volume_operator.mv(tangent)))
    assert jnp.all(jnp.isfinite(volume_operator.transpose_mv(weights)))

    result = prepared.advance(state)

    assert result.successful
    assert result.evidence.air is not None
    assert jnp.isfinite(result.evidence.pressure_work)


def test_volume_operator_handles_large_declared_capacity_with_action_outputs() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    compact_topology = seed.topology(
        seed.capacity_plan(resource_id="foam-volume-large-target")
    )
    compact_state = seed.state(compact_topology)
    compact = PreparedMultiRegionSurface(compact_topology, compact_state)
    volume = compact.region_volumes(compact_state.positions)[
        compact_topology.finite_region_indices[0]
    ]
    air = RegionPressureAirPlan.incompressible(jnp.asarray((volume,)))
    _, _, state, prepared = _sphere_problem(air, headroom=32.0)
    tangent = jnp.ones_like(state.surface.positions)
    cotangent = jnp.ones((prepared.finite_slots.size,), dtype=tangent.dtype)

    def actions(
        positions: jax.Array, direction: jax.Array, weights: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        operator = prepared.volume_operator(positions)
        return (
            operator.linearization.primal,
            operator.mv(direction),
            operator.transpose_mv(weights),
        )

    primal, jvp, vjp = jax.jit(actions)(state.surface.positions, tangent, cotangent)

    assert primal.shape == (prepared.finite_slots.size,)
    assert jvp.shape == primal.shape
    assert vjp.shape == state.surface.positions.shape
    assert jnp.all(jnp.isfinite(primal))
    assert jnp.all(jnp.isfinite(jvp))
    assert jnp.all(jnp.isfinite(vjp))


def test_accepted_fixed_topology_step_executes_proposed_events() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-event-target"))
    initial = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, initial)
    volume = surface.region_volumes(initial.positions)[topology.finite_region_indices[0]]
    air = RegionPressureAirPlan.incompressible(jnp.asarray((volume,)))
    topology, _, state, prepared = _sphere_problem(
        air,
        time_step=1.0e-5,
        headroom=2.0,
        event_capacity=4,
    )

    result = prepared.advance(state, events=(_edge_split(topology),))

    assert result.successful
    assert result.evidence.event is not None
    assert result.evidence.event.committed
    assert bool(result.evidence.topology_changed)
    assert not bool(result.evidence.rollback)
    assert result.topology.epoch == topology.epoch + 1


def test_overdamped_incompressible_step_preserves_volume_and_decreases_energy() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-dynamics-volume"))
    initial = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, initial)
    volume = surface.region_volumes(initial.positions)[topology.finite_region_indices[0]]
    air = RegionPressureAirPlan.incompressible(jnp.asarray((volume,)))
    _, _, state, prepared = _sphere_problem(air)

    result = prepared.advance(state)

    assert result.successful
    assert int(result.evidence.status) == FoamDynamicsStatus.COMPLETED
    assert float(result.evidence.volume_residual) < 1.0e-10
    assert bool(result.evidence.energy_nonincreasing)
    assert float(result.evidence.viscous_dissipation) >= 0.0
    assert int(result.evidence.constraint_rank) == 1
    assert np.isfinite(float(result.evidence.constraint_condition))
    assert int(result.evidence.constraint_linear_status) == LinearSolveStatus.SUCCESS
    assert bool(result.evidence.ccd_certified)
    assert not bool(result.evidence.derivative_available)


@pytest.mark.parametrize(
    (
        "maximum_entries",
        "maximum_actions",
        "maximum_bytes",
        "expected_status",
    ),
    (
        (1, 1_024, 64 * 1024 * 1024, "CONSTRAINT_ENTRIES_EXCEEDED"),
        (1_000_000, 1, 64 * 1024 * 1024, "RANK_CHECK_ACTIONS_EXCEEDED"),
        (1_000_000, 1_024, 1, "PREPARATION_BYTES_EXCEEDED"),
    ),
    ids=("entries", "actions", "bytes"),
)
def test_constraint_basis_resource_limit_refuses_before_derivative_work(
    monkeypatch: pytest.MonkeyPatch,
    maximum_entries: int,
    maximum_actions: int,
    maximum_bytes: int,
    expected_status: str,
) -> None:
    seed = seed_double_bubble(1.0, 1.0, ring_points=8)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-constraint-resource"))
    surface_state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    finite_slots = jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    targets = surface.region_volumes(surface_state.positions)[finite_slots]

    def forbidden_linearize(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Derivative work ran before resource admission.")

    monkeypatch.setattr(jax, "linearize", forbidden_linearize)
    with pytest.raises(ValueError, match=expected_status):
        PreparedFoamDynamics(
            FoamDynamicsPlan(
                time_step=1.0e-5,
                maximum_constraint_entries=maximum_entries,
                maximum_constraint_rank_actions=maximum_actions,
                maximum_constraint_preparation_bytes=maximum_bytes,
            ),
            surface,
            FoamMaterialPlan.soap_film(topology.region_ids, SIGMA),
            RegionPressureAirPlan.incompressible(targets),
            FoamDynamicsState(surface_state),
        )


def test_compressible_caloric_cells_conserve_amount_and_close_energy_rate_ledger() -> (
    None
):
    air = RegionPressureAirPlan.compressible(
        CaloricIdealBubbleGasLaw(1.4),
        BubbleEnvironment(0.1, 293.15),
    )
    _, _, state, prepared = _sphere_problem(air, time_step=2.0e-4)
    assert state.gas is not None and state.gas.internal_energy is not None
    initial_amount = jnp.sum(state.gas.amount)

    result = prepared.advance(state)

    assert result.successful
    assert result.state.gas is not None
    assert result.state.gas.internal_energy is not None
    assert state.gas.internal_energy is not None
    assert result.evidence.air is not None
    np.testing.assert_allclose(
        jnp.sum(result.state.gas.amount), initial_amount, rtol=1.0e-14
    )
    assert abs(float(result.evidence.air.amount_residual)) < 1.0e-20
    assert abs(float(result.evidence.air.internal_energy_residual)) < 1.0e-14
    assert bool(result.evidence.air.admissible)
    # Adiabatic internal-energy rate is exactly minus pressure work.
    delta_internal = jnp.sum(result.state.gas.internal_energy - state.gas.internal_energy)
    np.testing.assert_allclose(
        delta_internal + result.evidence.pressure_work,
        0.0,
        rtol=2.0e-3,
        atol=1.0e-12,
    )


def test_film_inertia_uses_shake_rattle_and_reports_projection_work() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-inertia-volume"))
    initial = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, initial)
    volume = surface.region_volumes(initial.positions)[topology.finite_region_indices[0]]
    air = RegionPressureAirPlan.incompressible(jnp.asarray((volume,)))
    _, _, state, prepared = _sphere_problem(
        air,
        route="film-inertia",
        time_step=1.0e-4,
    )

    result = prepared.advance(state)

    assert result.successful
    assert float(result.evidence.volume_residual) < 1.0e-9
    assert np.isfinite(float(result.evidence.projection_work))
    assert int(result.evidence.constraint_rank) == 1
    assert np.isfinite(float(result.evidence.constraint_condition))
    assert int(result.evidence.constraint_linear_status) == LinearSolveStatus.SUCCESS


def test_failed_volume_projection_rolls_back_state() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-dynamics-rollback"))
    initial = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, initial)
    volume = surface.region_volumes(initial.positions)[topology.finite_region_indices[0]]
    air = RegionPressureAirPlan.incompressible(jnp.asarray((1.5 * volume,)))
    _, _, state, prepared = _sphere_problem(
        air,
        time_step=1.0e-3,
        projection_iterations=1,
    )

    result = prepared.advance(state)

    assert not result.successful
    assert bool(result.evidence.rollback)
    assert jnp.array_equal(result.state.surface.positions, state.surface.positions)
    assert jnp.array_equal(result.state.surface.velocities, state.surface.velocities)


def test_failed_fixed_topology_step_does_not_execute_proposed_events() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-failure-target"))
    initial = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, initial)
    volume = surface.region_volumes(initial.positions)[topology.finite_region_indices[0]]
    air = RegionPressureAirPlan.incompressible(jnp.asarray((1.5 * volume,)))
    topology, _, state, prepared = _sphere_problem(
        air,
        time_step=1.0e-3,
        projection_iterations=1,
        headroom=2.0,
        event_capacity=4,
    )

    result = prepared.advance(state, events=(_edge_split(topology),))

    assert int(result.evidence.status) == FoamDynamicsStatus.MECHANICS_FAILED
    assert result.topology is topology
    assert result.state is state
    assert result.topology.epoch == topology.epoch
    assert result.evidence.event is None
    assert not bool(result.evidence.topology_changed)
    assert bool(result.evidence.rollback)
    assert float(result.evidence.elapsed_time) == 0.0


def test_ccd_rejected_fixed_topology_step_does_not_execute_proposed_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-ccd-target"))
    initial = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, initial)
    volume = surface.region_volumes(initial.positions)[topology.finite_region_indices[0]]
    air = RegionPressureAirPlan.incompressible(jnp.asarray((volume,)))
    topology, _, state, prepared = _sphere_problem(
        air,
        time_step=1.0e-5,
        headroom=2.0,
        event_capacity=4,
    )

    def reject_motion(
        _prepared: PreparedFoamDynamics,
        _start: jax.Array,
        _end: jax.Array,
        /,
    ) -> tuple[bool, float]:
        return False, 0.25

    monkeypatch.setattr(PreparedFoamDynamics, "certify_motion", reject_motion)
    result = prepared.advance(state, events=(_edge_split(topology),))

    assert int(result.evidence.status) == FoamDynamicsStatus.CCD_FAILED
    assert result.topology is topology
    assert result.state is state
    assert result.topology.epoch == topology.epoch
    assert result.evidence.event is None
    assert not bool(result.evidence.topology_changed)
    assert bool(result.evidence.rollback)
    assert not bool(result.evidence.ccd_certified)
    assert float(result.evidence.minimum_ccd_time_of_impact) == pytest.approx(0.25)
    assert float(result.evidence.elapsed_time) == 0.0


def test_fixed_topology_jvp_matches_centered_difference() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-dynamics-jvp"))
    initial = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, initial)
    volume = surface.region_volumes(initial.positions)[topology.finite_region_indices[0]]
    air = RegionPressureAirPlan.incompressible(jnp.asarray((volume,)))
    _, _, state, prepared = _sphere_problem(air, time_step=1.0e-5)
    direction = jnp.zeros_like(state.surface.positions).at[0, 0].set(1.0e-3)

    def advance(positions: jax.Array) -> jax.Array:
        changed = FoamDynamicsState(
            state.surface.with_positions(positions),
            unresolved_rim_content=state.unresolved_rim_content,
            time=state.time,
        )
        return prepared.fixed_topology_step(changed)[0].surface.positions

    _, tangent = jax.jvp(advance, (state.surface.positions,), (direction,))
    epsilon = 1.0e-4
    finite_difference = (
        advance(state.surface.positions + epsilon * direction)
        - advance(state.surface.positions - epsilon * direction)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(tangent, finite_difference, rtol=5.0e-4, atol=2.0e-7)


def test_air_plan_refuses_mixed_or_mismatched_routes() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        RegionPressureAirPlan()
    with pytest.raises(ValueError, match="exactly one"):
        RegionPressureAirPlan(
            target_volumes=jnp.ones((1,)),
            gas_law=CaloricIdealBubbleGasLaw(1.4),
            environment=BubbleEnvironment(1.0, 293.15),
        )
    with pytest.raises(ValueError, match="every finite region"):
        RegionPressureAirPlan.incompressible(jnp.ones((2,))).require_region_count(1)
