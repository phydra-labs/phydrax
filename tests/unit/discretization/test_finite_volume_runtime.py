#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _runtime(
    cells=64,
    *,
    cfl=0.4,
    retries=4,
    source=None,
    mapped=False,
    interface_solver=None,
):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    if mapped:
        discretization = phx.discretization.MappedFiniteVolumePlan(
            discretization,
            lambda point: point,
            mapping_id="runtime-identity",
        ).prepare()
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.SupersonicOutflowBoundary(),
        phx.discretization.SupersonicOutflowBoundary(),
    )
    problem = phx.equations.ConservationProblemIR(
        "runtime-sod",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,)),
        source=source,
        source_id=None if source is None else "runtime-test-source",
    )
    solver = interface_solver
    if solver is None:
        solver = (
            phx.discretization.RusanovFluxPlan()
            if mapped
            else phx.discretization.HLLCFluxPlan()
        )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        solver,
        positivity=phx.discretization.ConvexStateLimiterPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(
            cfl=cfl, maximum_retries=retries, reduction_factor=0.5
        ),
    )
    x = grid.structured_axes[0].interval_centers
    primitive = jnp.stack(
        (
            jnp.where(x < 0.5, 1.0, 0.125),
            jnp.zeros_like(x),
            jnp.where(x < 0.5, 1.0, 0.1),
        ),
        axis=-1,
    )
    return runtime, system.primitive_to_conserved(primitive)


def _assert_exact_journal(left, right):
    assert left.journal_id == right.journal_id
    assert left.to_archive_record() == right.to_archive_record()
    for name, left_array in left.archive_arrays().items():
        np.testing.assert_array_equal(left_array, right.archive_arrays()[name])


def test_einfeldt_fallback_is_consistent_and_has_finite_bounds():
    system = phx.equations.EulerSystem()
    state = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, 1.0]]))
    result = phx.discretization.EinfeldtHLLFluxPlan().face_flux(system, state, state, 0)
    np.testing.assert_allclose(
        result.normal_flux, system.physical_flux(state, 0), rtol=1e-12
    )
    assert jnp.all(jnp.isfinite(result.max_speed))


def test_global_flux_blending_preserves_conservation_and_admissibility():
    system = phx.equations.EulerSystem()
    fallback = system.primitive_to_conserved(
        jnp.asarray([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    )
    high = fallback.at[0, 0].set(-0.1)
    high = high.at[1].add(fallback[0] - high[0])
    result = phx.discretization.FluxPositivityPlan().limit_candidate(
        system, high, fallback
    )

    assert result.report.activated
    assert jnp.all(system.admissible(result.state))
    np.testing.assert_allclose(
        jnp.sum(result.state, axis=0), jnp.sum(high, axis=0), atol=2e-10
    )


def test_runtime_initialization_binds_static_content_and_round_trips_averages():
    runtime, average = _runtime(cells=20)
    state = runtime.initialize_state(average, 0.25, 0.001)
    volumes = runtime.dynamics.effective_volumes.reshape((-1,))

    np.testing.assert_allclose(state.cell_average(), average)
    np.testing.assert_allclose(
        state.content_state.conservative_content,
        average * volumes[:, None],
    )
    np.testing.assert_array_equal(
        state.content_state.active_cell_mask,
        jnp.ones((average.shape[0],), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(
        state.content_state.effective_cell_volumes,
        volumes,
    )
    assert state.time == state.content_state.time
    assert "conservative_state" not in vars(state)
    assert "time" not in vars(state)
    assert state.content_state.topology_epoch_id == runtime.topology_epoch_id
    assert state.content_state.geometry_layout_id == runtime.geometry_layout_id
    assert state.content_state.evidence_policy_id == runtime.evidence_policy_id
    assert state.content_state.geometry_version == 0
    assert state.content_state.evidence_version == 0
    journal = state.topology_journal
    assert journal.capacity == 1
    assert int(journal.count) == 0
    assert not bool(journal.overflowed)
    assert journal.epoch_table == (runtime.initial_topology_epoch,)
    assert journal.current_epoch_id == state.content_state.topology_epoch_id
    assert journal.current_epoch_id == runtime.topology_epoch_id
    assert runtime.initial_topology_epoch.parent_epoch_id is None
    assert (
        runtime.initial_topology_epoch.prepared_id
        == runtime.dynamics.discretization.prepared_id
    )


def test_runtime_accepts_admissible_step_and_advances_state_atomically():
    runtime, state = _runtime()
    initial = runtime.initialize_state(state, 0.0, 0.002)
    result = runtime.advance(initial)

    assert result.accepted
    assert result.runtime_state.accepted_step == 1
    assert result.runtime_state.time > initial.time
    assert jnp.all(
        runtime.dynamics.system.admissible(result.runtime_state.cell_average())
    )
    assert jnp.all(jnp.isfinite(result.runtime_state.cell_average()))
    accepted_average = runtime._candidate(
        initial.time,
        state,
        result.accepted_step_size,
        None,
    ).state
    np.testing.assert_allclose(
        result.runtime_state.cell_average(),
        accepted_average,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_array_equal(
        result.runtime_state.content_state.effective_cell_volumes,
        initial.content_state.effective_cell_volumes,
    )
    assert (
        result.runtime_state.content_state.geometry_version
        == initial.content_state.geometry_version
    )
    assert (
        result.runtime_state.content_state.evidence_version
        == initial.content_state.evidence_version
    )
    _assert_exact_journal(
        result.runtime_state.topology_journal,
        initial.topology_journal,
    )
    cell_content_change = (
        result.runtime_state.content_state.conservative_content
        - initial.content_state.conservative_content
    )
    ledger = result.accepted_flux_integrals
    np.testing.assert_allclose(
        ledger.scatter_content_integral(),
        cell_content_change,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(ledger.source_integral, 0.0, atol=2e-11)
    source_sum, boundary_sum, net_cell_sum = ledger.conservation_sums()
    np.testing.assert_allclose(
        net_cell_sum,
        jnp.sum(cell_content_change, axis=0),
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        source_sum - boundary_sum,
        net_cell_sum,
        rtol=2e-11,
        atol=2e-11,
    )


def test_runtime_rejects_invalid_initial_state_without_mutating_content():
    runtime, state = _runtime(retries=1)
    invalid = state.at[0, 0].set(-1.0)
    initial = runtime.initialize_state(invalid, 0.0, 0.01)
    result = runtime.advance(initial)

    assert not result.accepted
    assert result.runtime_state.last_status == int(
        phx.solver.FiniteVolumeRunStatus.INVALID_INITIAL_STATE
    )
    np.testing.assert_array_equal(
        result.runtime_state.content_state.conservative_content,
        initial.content_state.conservative_content,
    )
    np.testing.assert_array_equal(
        result.runtime_state.content_state.effective_cell_volumes,
        initial.content_state.effective_cell_volumes,
    )
    np.testing.assert_array_equal(result.runtime_state.time, initial.time)
    assert (
        result.runtime_state.content_state.geometry_version
        == initial.content_state.geometry_version
    )
    assert (
        result.runtime_state.content_state.evidence_version
        == initial.content_state.evidence_version
    )
    _assert_exact_journal(
        result.runtime_state.topology_journal,
        initial.topology_journal,
    )
    accepted_result = runtime.advance(
        runtime.initialize_state(state, 0.0, initial.step_size)
    )
    rejected_ledger = result.accepted_flux_integrals
    accepted_ledger = accepted_result.accepted_flux_integrals
    assert tuple(
        (block.block_id, block.route_id) for block in rejected_ledger.blocks
    ) == tuple((block.block_id, block.route_id) for block in accepted_ledger.blocks)
    for block in rejected_ledger.blocks:
        np.testing.assert_array_equal(
            block.flux_integral,
            jnp.zeros_like(block.flux_integral),
        )
    np.testing.assert_array_equal(
        rejected_ledger.source_integral,
        jnp.zeros_like(rejected_ledger.source_integral),
    )


def test_runtime_step_is_jittable_and_status_is_bounded():
    runtime, state = _runtime(cells=32)
    initial = runtime.initialize_state(state, 0.0, 0.001)
    result = eqx.filter_jit(runtime.advance)(initial)

    assert result.runtime_state.last_status in (
        int(phx.solver.FiniteVolumeRunStatus.SUCCESS),
        int(phx.solver.FiniteVolumeRunStatus.RECOVERED_REJECTION),
    )
    assert result.retries <= runtime.policy.maximum_retries


def test_face_local_positivity_blending_preserves_shared_flux_conservation():
    cells = 6
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    primitive = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 1.0]), (cells, 3))
    state = system.primitive_to_conserved(primitive)
    low_flux = jnp.zeros((cells + 1, 3))
    high_flux = low_flux.at[3, 0].set(100.0)
    result = phx.discretization.FluxPositivityPlan().limit_face_fluxes(
        system,
        state,
        (high_flux,),
        (low_flux,),
        jnp.zeros_like(state),
        jnp.asarray(0.1),
        discretization,
    )

    assert result.report.activated
    assert jnp.all(system.admissible(result.state))
    assert result.face_blend_factors[0][3] < 1.0
    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes[..., None] * result.state, axis=0),
        jnp.sum(discretization.cell_volumes[..., None] * state, axis=0),
        atol=2e-11,
    )


def test_runtime_exposes_one_stable_accepted_flux_integral_ledger():
    runtime, state = _runtime(cells=24)
    result = runtime.advance(runtime.initialize_state(state, 0.0, 0.001))
    second = runtime.advance(result.runtime_state)

    assert result.accepted
    ledger = result.accepted_flux_integrals
    assert ledger.units == "content"
    assert len(ledger.blocks) == 1
    assert ledger.blocks[0].flux_integral.shape == (25, 3)
    assert jnp.all(jnp.isfinite(ledger.blocks[0].flux_integral))
    assert tuple((block.block_id, block.route_id) for block in ledger.blocks) == tuple(
        (block.block_id, block.route_id)
        for block in second.accepted_flux_integrals.blocks
    )
    assert "accepted_" + "integrated_fluxes" not in vars(result)


def test_mapped_runtime_high_order_fallback_and_ledger_routes_are_deterministic():
    left_runtime, left_average = _runtime(cells=18, mapped=True)
    right_runtime, right_average = _runtime(cells=18, mapped=True)
    assert isinstance(
        left_runtime.fallback_dynamics.method.interface_solver,
        phx.discretization.EinfeldtHLLFluxPlan,
    )
    left_initial = left_runtime.initialize_state(left_average, 0.0, 0.001)
    right_initial = right_runtime.initialize_state(right_average, 0.0, 0.001)
    left = left_runtime.advance(left_initial)
    right = right_runtime.advance(right_initial)

    assert left.accepted
    assert right.accepted
    assert tuple(
        (block.block_id, block.route_id) for block in left.accepted_flux_integrals.blocks
    ) == tuple(
        (block.block_id, block.route_id) for block in right.accepted_flux_integrals.blocks
    )
    np.testing.assert_allclose(
        left.accepted_flux_integrals.scatter_content_integral(),
        (
            left.runtime_state.content_state.conservative_content
            - left_initial.content_state.conservative_content
        ),
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        left.accepted_flux_integrals.source_integral,
        0.0,
        atol=2e-11,
    )


def test_mapped_runtime_rejects_hllc_flux():
    runtime, average = _runtime(
        cells=18,
        mapped=True,
        interface_solver=phx.discretization.HLLCFluxPlan(),
    )
    initial = runtime.initialize_state(average, 0.0, 0.001)

    with pytest.raises(
        ValueError,
        match=(
            r"Mapped finite volumes currently require Rusanov, HLL, "
            r"or Einfeldt HLL flux\."
        ),
    ):
        runtime.advance(initial)


def test_static_accepted_ledger_accounts_source_and_boundary_content():
    source_vector = jnp.asarray((0.0, 0.0, 0.2))

    def source(time, state, coordinates, args):
        del time, coordinates, args
        return jnp.broadcast_to(source_vector, state.shape)

    runtime, state = _runtime(cells=24, source=source)
    initial = runtime.initialize_state(state, 0.0, 0.001)
    result = runtime.advance(initial)

    assert result.accepted
    ledger = result.accepted_flux_integrals
    content_change = (
        result.runtime_state.content_state.conservative_content
        - initial.content_state.conservative_content
    )
    np.testing.assert_allclose(
        ledger.scatter_content_integral(),
        content_change,
        rtol=2e-11,
        atol=2e-11,
    )
    expected_source_integral = (
        result.accepted_step_size
        * initial.content_state.effective_cell_volumes[:, None]
        * source_vector
    )
    np.testing.assert_allclose(
        ledger.source_integral,
        expected_source_integral,
        rtol=2e-10,
        atol=2e-11,
    )
    source_sum, boundary_sum, net_cell_sum = ledger.conservation_sums()
    expected_source = jnp.sum(expected_source_integral, axis=0)
    np.testing.assert_allclose(source_sum, expected_source, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(
        source_sum - boundary_sum,
        net_cell_sum,
        rtol=2e-11,
        atol=2e-11,
    )


def test_unstructured_accepted_step_coupling_sets_journal_capacity():
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    system = phx.equations.EulerSystem(2)
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        component_names=system.component_names,
    ).prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "journal-capacity",
        "state",
        system,
        boundaries,
    )
    coupling = phx.discretization.finite_volume.UnstructuredFiniteVolumeCouplingPlan(
        topology_event_capacity=3,
        topology_event_policy="accepted_step",
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        coupling=coupling,
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    primitive = jnp.asarray(((1.0, 0.1, -0.05, 1.0),))
    state = runtime.initialize_state(
        system.primitive_to_conserved(primitive),
        0.0,
        1e-3,
    )

    assert state.topology_journal.capacity == 3
    assert state.topology_journal.epoch_table == (runtime.initial_topology_epoch,)
    assert state.topology_journal.current_epoch_id == runtime.topology_epoch_id
