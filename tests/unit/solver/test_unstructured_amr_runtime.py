#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume._flux_ledger import (
    FiniteVolumeAcceptedFluxIntegralBlock,
    FiniteVolumeAcceptedFluxIntegralLedger,
)
from phydrax.solver._finite_volume_runtime import PreparedFiniteVolumeRuntime
from phydrax.solver._unstructured_amr_runtime import (
    _AMRCoarseTemporalStageTrace,
    PreparedUnstructuredAMRRuntime,
)


def _grid_plan(system, nx, ny):
    vertices = np.asarray(
        [(2.0 * i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            cells.append((lower_left, lower_right, upper_right, upper_left))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(cells, dtype=np.int32),
        vertex_global_ids=np.arange(1000, 1000 + vertices.shape[0]),
        cell_global_ids=np.arange(2000, 2000 + len(cells)),
        component_names=system.component_names,
    )


def _level_runtime(plan, system, label, topology_event_policy="accepted_step"):
    discretization = plan.prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        f"amr-runtime:{label}",
        "state",
        system,
        boundaries,
    )
    event_capacity = 4 if topology_event_policy == "accepted_step" else 0
    coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        topology_event_capacity=event_capacity,
        topology_event_policy=topology_event_policy,
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
    return discretization, runtime


def _runtime(
    topology_event_policy="accepted_step",
    *,
    fine_topology_event_policy=None,
):
    system = phx.equations.EulerSystem(2)
    coarse_plan = _grid_plan(system, 2, 1)
    fine_plan = _grid_plan(system, 4, 2)
    coarse, coarse_runtime = _level_runtime(
        coarse_plan, system, "coarse", topology_event_policy
    )
    fine, fine_runtime = _level_runtime(
        fine_plan,
        system,
        "fine",
        (
            topology_event_policy
            if fine_topology_event_policy is None
            else fine_topology_event_policy
        ),
    )
    parent = np.asarray((0, 0, 1, 1, 0, 0, 1, 1), dtype=np.int32)
    prolongation = phx.discretization.UnstructuredConservativeRemapPlan(
        coarse,
        fine,
        np.arange(fine.cell_count + 1, dtype=np.int32),
        parent,
        fine.cell_volumes,
        method="nested-constant-prolongation",
        provenance="unit-test",
    )
    restriction = phx.discretization.UnstructuredConservativeRemapPlan(
        fine,
        coarse,
        np.asarray((0, 4, 8), dtype=np.int32),
        np.asarray((0, 1, 4, 5, 2, 3, 6, 7), dtype=np.int32),
        np.asarray((0.25,) * 8),
        method="nested-volume-restriction",
        provenance="unit-test",
    )
    hierarchy = phx.discretization.UnstructuredAMRHierarchyPlan(
        coarse,
        fine,
        prolongation,
        restriction,
        maximum_refined_cells=1,
        coarse_interface_route_ids=(
            coarse_runtime.static_flux_rate_block_templates[0].route_id,
        ),
        fine_interface_route_ids=(
            fine_runtime.static_flux_rate_block_templates[0].route_id,
        ),
    )
    return (
        system,
        coarse,
        fine,
        PreparedUnstructuredAMRRuntime(
            hierarchy,
            coarse_runtime,
            fine_runtime,
            refinement_ratio=2,
        ),
    )


def _uniform(system, cell_count, velocity=(0.15, -0.05)):
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, *velocity, 1.0)),
        (cell_count, system.component_count),
    )
    return system.primitive_to_conserved(primitive)


def test_amr_constant_free_stream_and_exact_two_substeps():
    system, coarse, _, runtime = _runtime()
    coarse_average = _uniform(system, coarse.cell_count, velocity=(0.0, 0.0))
    state = runtime.initialize_state(
        coarse_average,
        0.0,
        1.0e-4,
        indicator=jnp.asarray((1.0, 0.0)),
        threshold=jnp.asarray(0.5),
    )
    single = runtime.coarse_runtime.advance(state.coarse_state)
    np.testing.assert_allclose(
        single.runtime_state.cell_average(),
        coarse_average,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    new_selection = runtime.hierarchy.select(jnp.asarray((0.0, 1.0)), jnp.asarray(0.5))
    np.testing.assert_array_equal(state.selection.coarse_refined, (True, False))
    np.testing.assert_array_equal(new_selection.coarse_refined, (False, True))
    result = runtime.advance(state, selection=new_selection)
    np.testing.assert_allclose(
        result.reflux_report.correction,
        0.0,
        atol=2.0e-11,
    )
    assert bool(result.regrid_committed)
    assert result.successor_runtime is not None
    assert (
        result.coarse_state.topology_journal.current_epoch_id
        != state.coarse_state.topology_journal.current_epoch_id
    )
    assert (
        result.fine_state.topology_journal.current_epoch_id
        != state.fine_state.topology_journal.current_epoch_id
    )
    assert (
        result.coarse_state.topology_journal.journal_id
        != state.coarse_state.topology_journal.journal_id
    )
    assert (
        result.fine_state.topology_journal.journal_id
        != state.fine_state.topology_journal.journal_id
    )
    assert result.topology_event_request is not None
    assert (
        result.coarse_state.topology_journal.payload_ids[0]
        == result.topology_event_request.payload_id
    )
    assert (
        result.fine_state.topology_journal.payload_ids[0]
        == result.topology_event_request.payload_id
    )
    assert len(result.fine_substep_ledgers) == 2
    np.testing.assert_allclose(
        result.coarse_state.cell_average(), coarse_average, rtol=2.0e-9, atol=2.0e-11
    )
    np.testing.assert_allclose(
        result.fine_state.cell_average(),
        runtime.hierarchy.prolong(coarse_average),
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    continued = result.successor_runtime.advance(result.runtime_state)
    assert bool(continued.accepted)


def test_amr_runtime_rejects_mismatched_level_event_policies_at_construction():
    with pytest.raises(ValueError, match="both use accepted-step topology events"):
        _runtime("accepted_step", fine_topology_event_policy="disabled")


def test_amr_explicit_selection_change_requires_events_before_stepping(monkeypatch):
    system, coarse, _, runtime = _runtime("disabled")
    state = runtime.initialize_state(
        _uniform(system, coarse.cell_count),
        0.0,
        1.0e-4,
        indicator=jnp.asarray((1.0, 0.0)),
        threshold=jnp.asarray(0.5),
    )
    successor = runtime.hierarchy.select(jnp.asarray((0.0, 1.0)), jnp.asarray(0.5))
    advance_called = False

    def forbidden_advance(*args, **kwargs):
        nonlocal advance_called
        advance_called = True
        raise AssertionError("level advance must not run")

    monkeypatch.setattr(type(runtime.coarse_runtime), "advance", forbidden_advance)
    with pytest.raises(ValueError, match="Explicit AMR selection change requires"):
        runtime.advance(state, selection=successor)
    assert not advance_called
    assert int(state.coarse_state.accepted_step) == 0
    assert int(state.fine_state.accepted_step) == 0


def test_amr_disabled_indicator_change_rolls_back_levels_and_journals():
    system, coarse, _, runtime = _runtime("disabled")
    state = runtime.initialize_state(
        _uniform(system, coarse.cell_count, velocity=(0.0, 0.0)),
        0.0,
        1.0e-4,
        indicator=jnp.asarray((1.0, 0.0)),
        threshold=jnp.asarray(0.5),
    )
    result = runtime.advance(
        state,
        indicator=jnp.asarray((0.0, 1.0)),
        threshold=jnp.asarray(0.5),
    )

    assert not bool(result.accepted)
    assert result.coarse_advance is not None
    assert bool(result.coarse_advance.accepted)
    assert result.orchestration_failure is not None
    assert "rolled back" in result.orchestration_failure
    assert result.runtime_state is state
    assert result.coarse_state.topology_journal is state.coarse_state.topology_journal
    assert result.fine_state.topology_journal is state.fine_state.topology_journal
    assert (
        result.coarse_state.topology_journal.journal_id
        == state.coarse_state.topology_journal.journal_id
    )
    assert (
        result.fine_state.topology_journal.journal_id
        == state.fine_state.topology_journal.journal_id
    )
    np.testing.assert_array_equal(
        result.selection.coarse_refined, state.selection.coarse_refined
    )
    np.testing.assert_array_equal(
        result.selection.fine_active, state.selection.fine_active
    )
    assert result.topology_event_request is None
    assert not bool(result.regrid_committed)
    np.testing.assert_allclose(result.accepted_step_size, 0.0)
    np.testing.assert_allclose(result.runtime_state.time, state.time)


def test_amr_disabled_unchanged_fixed_selection_remains_valid():
    system, coarse, _, runtime = _runtime("disabled")
    indicator = jnp.asarray((1.0, 0.0))
    state = runtime.initialize_state(
        _uniform(system, coarse.cell_count, velocity=(0.0, 0.0)),
        0.0,
        1.0e-4,
        indicator=indicator,
        threshold=jnp.asarray(0.5),
    )
    result = runtime.advance(state, selection=state.selection)

    assert bool(result.accepted)
    assert result.orchestration_failure is None
    assert result.topology_event_request is None
    assert not bool(result.regrid_committed)
    np.testing.assert_array_equal(
        result.selection.coarse_refined, state.selection.coarse_refined
    )
    np.testing.assert_array_equal(
        result.selection.fine_active, state.selection.fine_active
    )
    assert (
        result.coarse_state.topology_journal.journal_id
        == state.coarse_state.topology_journal.journal_id
    )
    assert (
        result.fine_state.topology_journal.journal_id
        == state.fine_state.topology_journal.journal_id
    )


def test_amr_ghost_fill_uses_temporal_coarse_content():
    system, coarse, _, runtime = _runtime()
    state = runtime.initialize_state(
        _uniform(system, coarse.cell_count),
        0.0,
        1.0e-4,
        indicator=jnp.asarray((1.0, 0.0)),
        threshold=jnp.asarray(0.5),
    )
    coarse_end = state.coarse_state.content_state.with_content(
        state.coarse_state.content_state.conservative_content + 0.25,
        time=jnp.asarray(1.0e-4, dtype=state.coarse_state.time.dtype),
    )
    coarse_end_state = phx.solver.FiniteVolumeRuntimeState(
        coarse_end,
        state.coarse_state.topology_journal,
        state.coarse_state.step_size,
        accepted_step=state.coarse_state.accepted_step,
    )
    filled = runtime.fill_fine_ghost(
        state.fine_state,
        state.coarse_state,
        coarse_end_state,
        state.selection,
        0.5,
    )
    expected = runtime.hierarchy.prolong(
        0.5
        * (
            state.coarse_state.content_state.cell_average()
            + coarse_end_state.content_state.cell_average()
        )
    )
    inactive = ~state.selection.fine_active
    np.testing.assert_allclose(
        filled.cell_average()[inactive], expected[inactive], rtol=1e-12, atol=1e-12
    )


def test_amr_stage_provider_prescribes_all_substep_nodes_and_jits():
    system, coarse, fine, runtime = _runtime()
    state = runtime.initialize_state(
        _uniform(system, coarse.cell_count, velocity=(0.0, 0.0)),
        0.0,
        1.0e-4,
        indicator=jnp.asarray((1.0, 0.0)),
        threshold=jnp.asarray(0.5),
    )
    coarse_start_average = state.coarse_state.cell_average()
    coarse_end_average = coarse_start_average.at[:, 0].add(
        jnp.asarray((0.2, 0.4), dtype=coarse_start_average.dtype)
    )
    coarse_end_content = (
        coarse_end_average
        * state.coarse_state.content_state.effective_cell_volumes[:, None]
    )
    coarse_end = runtime._state_with_content(
        state.coarse_state,
        coarse_end_content,
        time=state.coarse_state.time + state.coarse_state.step_size,
    )
    current_fine_average = state.fine_state.cell_average()
    stale_ghost_average = _uniform(system, fine.cell_count, velocity=(-0.3, 0.2))
    owned = state.selection.fine_active
    stale_average = jnp.where(
        owned[:, None],
        current_fine_average,
        stale_ghost_average,
    )
    stale_fine = runtime._state_with_content(
        state.fine_state,
        stale_average * state.fine_state.content_state.effective_cell_volumes[:, None],
    )

    recorded = []
    providers = []
    for substep in range(runtime.refinement_ratio):
        provider = runtime._fine_stage_state_provider(
            state.coarse_state,
            coarse_end,
            state.selection,
            substep,
        )
        providers.append(provider)
        for node in (0.0, 1.0, 0.5):
            fraction = (substep + node) / runtime.refinement_ratio
            stage_time = state.coarse_state.time + fraction * (
                coarse_end.time - state.coarse_state.time
            )
            recorded.append((fraction, np.asarray(provider(stage_time, stale_average))))

    for fraction, stage_average in recorded:
        expected = runtime.hierarchy.prolong(
            (1.0 - fraction) * coarse_start_average + fraction * coarse_end_average
        )
        np.testing.assert_allclose(
            stage_average[~np.asarray(owned)],
            np.asarray(expected)[~np.asarray(owned)],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            stage_average[np.asarray(owned)],
            np.asarray(stale_average)[np.asarray(owned)],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    assert providers[0].provider_id != providers[1].provider_id
    bound = runtime.fine_runtime.with_stage_state_provider(providers[0])
    assert bound.runtime_id != runtime.fine_runtime.runtime_id
    jitted = eqx.filter_jit(bound.advance)(stale_fine)
    assert bool(jitted.accepted)


def _record_amr_stage_traces(monkeypatch):
    records = []
    original = _AMRCoarseTemporalStageTrace.__call__

    def recording_call(self, stage_time, state, /):
        provided = original(self, stage_time, state)

        def append_record(time, incoming, outgoing, start, end, owned, t0, t1):
            records.append(
                tuple(
                    np.asarray(value)
                    for value in (
                        time,
                        incoming,
                        outgoing,
                        start,
                        end,
                        owned,
                        t0,
                        t1,
                    )
                )
            )

        jax.debug.callback(
            append_record,
            stage_time,
            state,
            provided,
            self.fine_start_average,
            self.fine_end_average,
            self.fine_owned_mask,
            self.coarse_start_time,
            self.coarse_end_time,
        )
        return provided

    monkeypatch.setattr(
        _AMRCoarseTemporalStageTrace,
        "__call__",
        recording_call,
    )
    return records


def test_amr_time_varying_coarse_trace_drives_stages_ledgers_and_reflux(monkeypatch):
    records = _record_amr_stage_traces(monkeypatch)
    system, coarse, _, runtime = _runtime()
    primitive = jnp.asarray(
        (
            (1.1, 0.25, -0.05, 1.0),
            (0.8, -0.15, 0.10, 0.7),
        )
    )
    coarse_average = system.primitive_to_conserved(primitive)
    state = runtime.initialize_state(
        coarse_average,
        0.0,
        1.0e-4,
        indicator=jnp.asarray((1.0, 0.0)),
        threshold=jnp.asarray(0.5),
    )

    result = runtime.advance(state)
    jax.effects_barrier()

    assert bool(result.accepted)
    assert not np.array_equal(
        np.asarray(result.coarse_advance.runtime_state.cell_average()),
        np.asarray(coarse_average),
    )
    assert len(result.fine_substep_ledgers) == runtime.refinement_ratio
    assert result.fine_accepted_flux_integrals is not None
    assert result.fine_accepted_flux_integrals.units == "content"
    assert result.reflux_register.route_id == runtime.hierarchy.interface_route_id
    np.testing.assert_allclose(
        result.reflux_report.maximum_budget_defect,
        0.0,
        atol=1.0e-12,
    )

    assert records
    observed_times = []
    for time, incoming, outgoing, start, end, owned, t0, t1 in records:
        fraction = float((time - t0) / (t1 - t0))
        expected = (1.0 - fraction) * start + fraction * end
        owned = owned.astype(bool)
        np.testing.assert_allclose(
            outgoing[~owned], expected[~owned], rtol=1.0e-11, atol=1.0e-12
        )
        np.testing.assert_allclose(
            outgoing[owned], incoming[owned], rtol=1.0e-11, atol=1.0e-12
        )
        observed_times.append(float(time))

    start_time = float(state.time)
    accepted_dt = float(result.accepted_step_size)
    required_stage_times = start_time + accepted_dt * np.asarray(
        (0.0, 0.5, 0.25, 0.5, 1.0, 0.75)
    )
    for required in required_stage_times:
        assert any(
            np.isclose(time, required, rtol=0.0, atol=1.0e-12) for time in observed_times
        )


def test_amr_fine_trace_failure_rolls_back_both_levels(monkeypatch):
    system, coarse, _, runtime = _runtime()
    state = runtime.initialize_state(
        _uniform(system, coarse.cell_count, velocity=(0.0, 0.0)),
        0.0,
        1.0e-4,
        indicator=jnp.asarray((1.0, 0.0)),
        threshold=jnp.asarray(0.5),
    )
    original_advance = PreparedFiniteVolumeRuntime.advance
    provider_ids = []

    def fail_second_fine_substep(self, runtime_state, args=None, /):
        result = original_advance(self, runtime_state, args)
        provider = self.stage_state_provider
        if provider is None:
            return result
        provider_ids.append(provider.provider_id)
        if len(provider_ids) == 2:
            return eqx.tree_at(
                lambda advance: advance.accepted,
                result,
                jnp.asarray(False),
            )
        return result

    monkeypatch.setattr(
        PreparedFiniteVolumeRuntime,
        "advance",
        fail_second_fine_substep,
    )
    result = runtime.advance(state)

    assert not bool(result.accepted)
    assert result.runtime_state is state
    assert len(provider_ids) == 2
    assert provider_ids[0] != provider_ids[1]
    assert len(result.fine_advances) == 2
    assert len(result.fine_substep_ledgers) == 1
    assert result.coarse_state.topology_journal is state.coarse_state.topology_journal
    assert result.fine_state.topology_journal is state.fine_state.topology_journal
    np.testing.assert_allclose(result.reflux_register.integrated_correction, 0.0)


def test_amr_selection_is_stable_and_reports_capacity_overflow():
    _, _, _, runtime = _runtime()
    first = eqx.filter_jit(runtime.hierarchy.select)(
        jnp.asarray((1.0, 1.0)), jnp.asarray(0.5)
    )
    second = runtime.hierarchy.select(jnp.asarray((1.0, 1.0)), jnp.asarray(0.5))
    np.testing.assert_array_equal(first.coarse_refined, (True, False))
    np.testing.assert_array_equal(first.fine_active, second.fine_active)
    assert bool(first.capacity_overflow)


def test_amr_vof_transfer_and_reflux_budget_are_conservative():
    system, coarse, _, runtime = _runtime()
    alpha = jnp.asarray((1.0, 0.2))
    fine_alpha = runtime.transfer_volume_fraction(alpha)
    np.testing.assert_allclose(runtime.restrict_volume_fraction(fine_alpha), alpha)
    state = runtime.initialize_state(
        _uniform(system, coarse.cell_count),
        0.0,
        1.0e-4,
        indicator=jnp.asarray((1.0, 0.0)),
        threshold=jnp.asarray(0.5),
    )
    result = runtime.advance(state)
    assert bool(result.accepted)
    np.testing.assert_allclose(result.reflux_report.maximum_budget_defect, 0.0)
    assert result.reflux_register.integrated_correction.shape == (
        coarse.cell_count,
        system.component_count,
    )


def test_amr_non_interface_ledger_rates_do_not_enter_reflux_scatter():
    runtime = object.__new__(PreparedUnstructuredAMRRuntime)
    object.__setattr__(
        runtime,
        "hierarchy",
        SimpleNamespace(coarse_fine_interface_map=np.asarray(((0, 0),))),
    )
    block = FiniteVolumeAcceptedFluxIntegralBlock(
        jnp.asarray(((2.0,), (17.0,))),
        jnp.asarray((0, 1), dtype=jnp.int32),
        jnp.asarray((1, -1), dtype=jnp.int32),
        jnp.asarray((True, True)),
        "synthetic-route",
        "physical",
    )
    ledger = FiniteVolumeAcceptedFluxIntegralLedger(
        (block,),
        jnp.zeros((2, 1)),
        jnp.asarray((True, True)),
        geometry_family_id="synthetic-family",
        geometry_layout_id="synthetic-layout",
        stage_geometry_versions=(jnp.asarray(0),) * 3,
        start_geometry_version=jnp.asarray(0),
        end_geometry_version=jnp.asarray(0),
        evidence_policy_id="synthetic-evidence",
        stage_evidence_versions=(jnp.asarray(0),) * 3,
        start_evidence_version=jnp.asarray(0),
        end_evidence_version=jnp.asarray(0),
        start_topology_epoch_id="synthetic-epoch",
        end_topology_epoch_id="synthetic-epoch",
        start_time=jnp.asarray(0.0),
        end_time=jnp.asarray(1.0),
        accepted_step=jnp.asarray(1),
    )
    scattered = runtime._interface_scatter(ledger, jnp.asarray((True, False)))
    np.testing.assert_allclose(scattered, ((-2.0,), (2.0,)))
