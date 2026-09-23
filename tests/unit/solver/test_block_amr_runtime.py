#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.solver._block_amr_runtime import (
    AMRTimeSchedulePlan,
    BlockAMRAdvancePhase,
    BlockAMRRuntimePlan,
)
from phydrax.solver._finite_volume_topology_events import (
    FiniteVolumeTopologyEventRequest,
    TopologyEventKind,
    TopologyEventStatus,
)


class _PositiveScalarSystem(phx.equations.AbstractConservationSystem):
    base: phx.equations.ScalarConservationSystem

    def __init__(self, dimension, flux, wave_speed, /, *, system_id):
        base = phx.equations.ScalarConservationSystem(
            dimension,
            flux,
            wave_speed,
            system_id=system_id,
        )
        self.base = base
        self.dimension = base.dimension
        self.component_names = base.component_names
        self.system_id = base.system_id

    def physical_flux(self, state, axis, args=None, /):
        return self.base.physical_flux(state, axis, args)

    def max_wave_speed(self, left, right, axis, args=None, /):
        return self.base.max_wave_speed(left, right, axis, args)

    def signal_bounds(self, left, right, axis, args=None, /):
        return self.base.signal_bounds(left, right, axis, args)

    def normal_signal_bounds(self, left, right, normal, args=None, /):
        return self.base.normal_signal_bounds(left, right, normal, args)

    def conserved_to_primitive(self, state, /):
        return self.base.conserved_to_primitive(state)

    def primitive_to_conserved(self, primitive, /):
        return self.base.primitive_to_conserved(primitive)

    def reflect_state(self, state, axis, /):
        return self.base.reflect_state(state, axis)

    def admissible(self, state, /):
        value = jnp.asarray(state)
        return jnp.all(jnp.isfinite(value), axis=-1) & (value[..., 0] >= 0.0)


def _prepared(levels=3, *, precision=None):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    level_plans = [phx.discretization.BlockLevelPlan(0, (4,), 4, refinement_ratio=2)]
    if levels >= 2:
        level_plans.append(
            phx.discretization.BlockLevelPlan(1, (2,), 16, refinement_ratio=2)
        )
    if levels == 3:
        level_plans.append(phx.discretization.BlockLevelPlan(2, (2,), 32))
    hierarchy = phx.discretization.BlockHierarchyPlan(grid, tuple(level_plans))
    return phx.discretization.FDAMRHierarchyPlan(
        hierarchy,
        precision=precision,
    ).prepare()


def _topology(prepared):
    initial = prepared.initial_topology()
    level_count = len(initial.plan.levels)
    if level_count == 1:
        return initial
    coarse_tags = jnp.zeros((4, 4), dtype="bool")
    coarse_tags = coarse_tags.at[1:3].set(True)
    if level_count == 2:
        result = prepared.compile_topology(initial, (coarse_tags,))
        assert result.status.successful
        return result.topology
    empty_middle = jnp.zeros((16, 2), dtype="bool")
    middle = prepared.compile_topology(
        initial,
        (coarse_tags, empty_middle),
    ).topology
    middle_tags = jnp.zeros((16, 2), dtype="bool").at[3, 1].set(True)
    result = prepared.compile_topology(middle, (coarse_tags, middle_tags))
    assert result.status.successful
    return result.topology


def _hierarchy_state(topology, values=1.0):
    if isinstance(values, (int, float)):
        per_level = (float(values),) * len(topology.plan.levels)
    else:
        per_level = tuple(values)
    levels = []
    for level_plan, metadata, fill_value in zip(
        topology.plan.levels,
        topology.levels,
        per_level,
        strict=True,
    ):
        array = jnp.zeros(
            (level_plan.maximum_blocks, *level_plan.block_shape, 1),
            dtype=jnp.float64,
        )
        active = metadata.active.reshape(
            (level_plan.maximum_blocks,) + (1,) * len(level_plan.block_shape) + (1,)
        )
        array = jnp.where(active, jnp.full_like(array, fill_value), array)
        levels.append(phx.discretization.BlockLevelState(level_plan, metadata, array))
    return phx.discretization.BlockHierarchyState(topology, tuple(levels))


def _boundaries(callback=None):
    boundary = (
        phx.discretization.ExtrapolationBoundary()
        if callback is None
        else phx.discretization.PrescribedStateBoundary(
            callback,
            boundary_id="block-amr-observed-boundary",
        )
    )
    return phx.discretization.FiniteVolumeBoundarySet(
        ("x",),
        (phx.discretization.FiniteVolumeBoundaryPair(boundary, boundary),),
    )


def _system(*, flux=None, positive=False, system_id="block-amr-runtime-scalar"):
    flux_ = (lambda state, axis, args: state) if flux is None else flux
    cls = _PositiveScalarSystem if positive else phx.equations.ScalarConservationSystem
    return cls(
        1,
        flux_,
        lambda left, right, axis, args: jnp.zeros(left.shape[:-1]),
        system_id=system_id,
    )


def _runtime(
    prepared,
    topology,
    *,
    schedule=None,
    subcycling=None,
    system=None,
    source=None,
    source_id=None,
    boundaries=None,
    specialist=None,
    specialist_id=None,
    indicator=None,
    indicator_id=None,
    precision=None,
):
    finite_volume = phx.discretization.BlockAMRFiniteVolumePlan(
        prepared,
        _system() if system is None else system,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
        _boundaries() if boundaries is None else boundaries,
        source=source,
        source_id=source_id,
        precision=precision,
    )
    plan = BlockAMRRuntimePlan(
        finite_volume,
        schedule,
        subcycling=subcycling,
        specialist_synchronization=specialist,
        specialist_id=specialist_id,
        indicator=indicator,
        indicator_id=indicator_id,
    )
    return plan.prepare(topology)


def test_one_level_schedule_degenerates_to_one_exact_ssprk_interval():
    prepared = _prepared(1)
    topology = _topology(prepared)
    schedule = AMRTimeSchedulePlan(prepared)
    runtime = _runtime(prepared, topology, schedule=schedule)
    state = runtime.initial_state(_hierarchy_state(topology), time=1.25)

    result = runtime.advance(state, 0.2)

    assert bool(result.accepted)
    assert schedule.edge_substeps == ()
    assert schedule.level_substeps_per_root == (1,)
    assert result.level_attempt_order == (0,)
    assert result.synchronization_order == ()
    assert result.edge_accepted_ledgers == ()
    assert result.flux_registers == ()
    np.testing.assert_allclose(result.stage_times[0], (1.25, 1.45, 1.35))
    np.testing.assert_allclose(result.runtime_state.time, 1.45)
    np.testing.assert_allclose(result.accepted_step_size, 0.2)


def test_three_level_subcycling_uses_exact_stage_times_and_deepest_first_sync():
    prepared = _prepared(3)
    topology = _topology(prepared)
    runtime = _runtime(prepared, topology)
    state = runtime.initial_state(_hierarchy_state(topology), time=0.0)

    result = runtime.advance(state, 0.4)

    assert bool(result.accepted)
    assert runtime.plan.schedule.edge_substeps == (2, 2)
    assert runtime.plan.schedule.level_substeps_per_root == (1, 2, 4)
    assert result.level_attempt_order == (0, 1, 2, 2, 1, 2, 2)
    assert result.synchronization_order == (1, 1, 0)
    expected = (
        (0.0, 0.4, 0.2),
        (0.0, 0.2, 0.1),
        (0.0, 0.1, 0.05),
        (0.1, 0.2, 0.15),
        (0.2, 0.4, 0.3),
        (0.2, 0.3, 0.25),
        (0.3, 0.4, 0.35),
    )
    np.testing.assert_allclose(np.asarray(result.stage_times), expected)
    np.testing.assert_array_equal(
        result.runtime_state.level_accepted_steps,
        np.asarray((1, 2, 4), dtype=np.int32),
    )
    assert result.runtime_state.time == result.accepted_ledgers[0].end_time
    assert result.runtime_state.time == result.edge_accepted_ledgers[-1].end_time
    for level, ledger in zip(
        result.level_attempt_order,
        result.accepted_ledgers,
        strict=True,
    ):
        expected_block_ids = tuple(
            route.block_id
            for route in runtime.dynamics.face_routes
            if route.level == level
        )
        assert tuple(block.block_id for block in ledger.blocks) == expected_block_ids
        begin = runtime.dynamics.level_cell_offsets[level]
        end = (
            runtime.dynamics.level_cell_offsets[level + 1]
            if level + 1 < len(runtime.dynamics.level_cell_offsets)
            else ledger.cell_count
        )
        outside_source = np.asarray(ledger.source_integral).copy()
        outside_source[begin:end] = 0.0
        np.testing.assert_array_equal(outside_source, 0.0)


def test_no_subcycling_retains_fillpatch_and_deepest_first_synchronization():
    prepared = _prepared(3)
    topology = _topology(prepared)
    schedule = AMRTimeSchedulePlan(prepared, subcycling=False)
    runtime = _runtime(prepared, topology, schedule=schedule)
    state = runtime.initial_state(_hierarchy_state(topology))

    result = runtime.advance(state, 0.4)

    assert bool(result.accepted)
    assert schedule.edge_substeps == (1, 1)
    assert schedule.level_substeps_per_root == (1, 1, 1)
    assert result.level_attempt_order == (0, 1, 2)
    assert result.synchronization_order == (1, 0)
    assert all(ledger.start_time == 0.0 for ledger in result.accepted_ledgers)
    assert all(ledger.end_time == 0.4 for ledger in result.accepted_ledgers)


def test_coarse_old_new_endpoints_drive_fine_stage_temporal_interpolation():
    observed = []

    def boundary(time, interior, coordinates, outward_normal, args):
        del coordinates, outward_normal
        args["time"] = float(np.asarray(time))
        return interior

    def flux(state, axis, args):
        del axis
        if state.shape[0] == 16:
            observed.append((args["time"], float(np.max(np.asarray(state)))))
        return jnp.zeros_like(state)

    def source(time, state, coordinates, args):
        del time, coordinates, args
        return jnp.ones_like(state) if state.shape[1] == 4 else jnp.zeros_like(state)

    prepared = _prepared(2)
    topology = _topology(prepared)
    runtime = _runtime(
        prepared,
        topology,
        system=_system(flux=flux, system_id="block-amr-temporal-fill"),
        source=source,
        source_id="source:coarse-unit-rate",
        boundaries=_boundaries(boundary),
    )
    state = runtime.initial_state(_hierarchy_state(topology, 0.0))

    result = runtime.advance(state, 0.2, {})

    assert bool(result.accepted)
    child_stage_times = (0.0, 0.1, 0.05, 0.1, 0.2, 0.15)
    for expected in child_stage_times:
        assert any(
            np.isclose(time, expected) and np.isclose(maximum, expected)
            for time, maximum in observed
        )


def test_accepted_ledgers_are_contiguous_and_consumed_without_an_extra_dt():
    prepared = _prepared(3)
    topology = _topology(prepared)
    runtime = _runtime(
        prepared,
        topology,
        system=_system(system_id="block-amr-ledger-advection"),
    )
    state = runtime.initial_state(_hierarchy_state(topology))

    result = runtime.advance(state, 0.4)

    assert bool(result.accepted)
    assert len(result.edge_accepted_ledgers) == 3
    assert len(result.flux_registers) == 3
    np.testing.assert_allclose(
        [float(register.accumulated_time) for register in result.flux_registers],
        (0.2, 0.2, 0.4),
    )
    for register in result.flux_registers:
        np.testing.assert_allclose(register.mismatch(), 0.0, atol=2e-14)
        assert np.max(np.abs(np.asarray(register.coarse_flux))) == pytest.approx(
            float(register.accumulated_time)
        )
    for ledger, register in zip(
        result.edge_accepted_ledgers,
        result.flux_registers,
        strict=True,
    ):
        assert ledger.end_time - ledger.start_time == register.accumulated_time
        assert register.owner_id == runtime.conservation.plan_id


def test_deepest_first_restriction_updates_only_covered_cells_and_conserves_composite():
    calls = []

    def specialist(level, state, time, step_size, args):
        del args
        calls.append((level, float(time), float(step_size)))
        return state

    prepared = _prepared(3)
    topology = _topology(prepared)
    runtime = _runtime(
        prepared,
        topology,
        system=_system(
            flux=lambda state, axis, args: jnp.zeros_like(state),
            system_id="block-amr-zero-flux",
        ),
        specialist=specialist,
        specialist_id="specialist:test-order",
    )
    state = runtime.initial_state(_hierarchy_state(topology, (1.0, 7.0, 9.0)))

    result = runtime.advance(state, 0.1)

    assert bool(result.accepted)
    assert [level for level, _, _ in calls] == [1, 1, 0]
    np.testing.assert_allclose([step for _, _, step in calls], (0.05, 0.05, 0.1))
    level_one = np.asarray(result.runtime_state.hierarchy_state.levels[1].values)
    level_one_covered = np.asarray(runtime.covered_cell_masks[1])
    np.testing.assert_allclose(level_one[level_one_covered], 9.0)
    root = np.asarray(result.runtime_state.hierarchy_state.levels[0].values)
    root_covered = np.asarray(runtime.covered_cell_masks[0])
    root_active = np.broadcast_to(
        np.asarray(topology.levels[0].active)[:, None], root_covered.shape
    )
    np.testing.assert_allclose(root[..., 0][root_active & ~root_covered], 1.0)
    assert np.min(root[..., 0][root_covered]) >= 7.0 - 1.0e-6
    np.testing.assert_allclose(result.composite_conservation_defect, 0.0, atol=2e-14)


def test_late_finest_rejection_rolls_back_every_level_counter_and_journal_bitwise():
    def source(time, state, coordinates, args):
        del coordinates, args
        if state.shape[0] != 32:
            return jnp.zeros_like(state)
        return jnp.where(time > 0.3, -100.0, 0.0) * jnp.ones_like(state)

    prepared = _prepared(3)
    topology = _topology(prepared)
    runtime = _runtime(
        prepared,
        topology,
        system=_system(
            flux=lambda state, axis, args: jnp.zeros_like(state),
            positive=True,
            system_id="block-amr-positive-rejection",
        ),
        source=source,
        source_id="source:late-finest-rejection",
    )
    state = runtime.initial_state(_hierarchy_state(topology))

    result = runtime.advance(state, 0.4)

    assert not bool(result.accepted)
    assert int(result.failed_level) == 2
    assert int(result.failed_phase) == int(BlockAMRAdvancePhase.ADMISSIBILITY)
    assert int(result.runtime_state.last_status) == int(result.failed_phase)
    assert result.accepted_step_size == 0.0
    assert result.runtime_state.time == state.time
    np.testing.assert_array_equal(
        result.runtime_state.level_accepted_steps,
        state.level_accepted_steps,
    )
    for actual, expected in zip(
        result.runtime_state.hierarchy_state.levels,
        state.hierarchy_state.levels,
        strict=True,
    ):
        np.testing.assert_array_equal(actual.values, expected.values)
    for name, expected in state.topology_journal.archive_arrays().items():
        np.testing.assert_array_equal(
            result.runtime_state.topology_journal.archive_arrays()[name],
            expected,
        )
    for ledger in result.accepted_ledgers:
        np.testing.assert_array_equal(
            ledger.source_integral,
            np.zeros_like(np.asarray(ledger.source_integral)),
        )
        assert all(jnp.all(block.flux_integral == 0.0) for block in ledger.blocks)


def test_topology_indicator_is_deferred_until_a_synchronized_accepted_root_endpoint():
    calls = []

    def indicator(state, args):
        del args
        calls.append((int(state.accepted_step), float(state.time)))
        return FiniteVolumeTopologyEventRequest(
            TopologyEventKind.AMR_REGRID,
            state.hierarchy_state.topology.epoch.epoch_id,
            state.hierarchy_state.plan.plan_id,
            reason="accepted synchronized hierarchy",
        )

    prepared = _prepared(2)
    topology = _topology(prepared)
    runtime = _runtime(
        prepared,
        topology,
        indicator=indicator,
        indicator_id="indicator:test-accepted-root",
    )
    state = runtime.initial_state(_hierarchy_state(topology))

    result = runtime.advance(state, 0.2)

    assert bool(result.accepted)
    assert calls == [(1, 0.2)]
    assert result.topology_event_request is not None
    assert int(result.topology_status) == int(TopologyEventStatus.PENDING)
    assert int(result.runtime_state.topology_journal.count) == 1
    assert (
        result.runtime_state.topology_journal.current_epoch_id == topology.epoch.epoch_id
    )

    rejected_calls = []

    def rejected_indicator(state, args):
        rejected_calls.append((state, args))
        return None

    def rejecting_source(time, state, coordinates, args):
        del time, coordinates, args
        return (
            -100.0 * jnp.ones_like(state)
            if state.shape[0] == 16
            else jnp.zeros_like(state)
        )

    rejected_runtime = _runtime(
        prepared,
        topology,
        system=_system(
            flux=lambda state, axis, args: jnp.zeros_like(state),
            positive=True,
            system_id="block-amr-event-rejection",
        ),
        source=rejecting_source,
        source_id="source:event-rejection",
        indicator=rejected_indicator,
        indicator_id="indicator:must-not-run",
    )
    rejected_state = rejected_runtime.initial_state(_hierarchy_state(topology))
    rejected = rejected_runtime.advance(rejected_state, 0.2)

    assert not bool(rejected.accepted)
    assert rejected_calls == []
    assert int(rejected.runtime_state.topology_journal.count) == 0


def test_fixed_epoch_runtime_is_jittable_differentiable_and_checkpoint_replayable():
    def source(time, state, coordinates, rate):
        del time, coordinates
        return rate * state

    prepared = _prepared(1)
    topology = _topology(prepared)
    runtime = _runtime(
        prepared,
        topology,
        system=_system(
            flux=lambda state, axis, args: jnp.zeros_like(state),
            system_id="block-amr-fixed-epoch-ad",
        ),
        source=source,
        source_id="source:linear-fixed-epoch",
    )
    state = runtime.initial_state(_hierarchy_state(topology))
    dt = 0.05

    def step(rate):
        advanced = runtime.advance(state, dt, rate)
        return advanced.runtime_state.hierarchy_state.levels[0].values

    rate = jnp.asarray(0.2)
    eager = step(rate)
    compiled = eqx.filter_jit(step)(rate)
    replayed = jax.checkpoint(step)(rate)
    derivative = jax.grad(lambda value: jnp.sum(step(value)))(rate)
    expected = 16.0 * (dt + float(rate) * dt**2 + 0.5 * float(rate) ** 2 * dt**3)

    np.testing.assert_allclose(compiled, eager, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(replayed, eager, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(derivative, expected, rtol=2e-12, atol=2e-12)
    assert jnp.isfinite(derivative)


def test_prepared_runtime_refuses_stale_epoch_and_route_artifacts():
    prepared = _prepared(2)
    topology = _topology(prepared)
    runtime = _runtime(prepared, topology)
    state = runtime.initial_state(_hierarchy_state(topology))
    stale_topology = prepared.initial_topology()

    with pytest.raises(ValueError, match="stale topology epoch"):
        runtime.initial_state(_hierarchy_state(stale_topology))

    other_runtime = _runtime(
        prepared,
        topology,
        source=lambda time, value, coordinates, args: jnp.zeros_like(value),
        source_id="source:different-route-artifacts",
    )
    with pytest.raises(ValueError, match="prepared routes"):
        other_runtime.initial_state(
            state.hierarchy_state,
            journal=state.topology_journal,
        )
