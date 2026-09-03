#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cardiovascular.electrophysiology._bidomain import (
    BidomainFEMPlan,
    BidomainSolveStatus,
    BidomainStepInputs,
    HeartOnlyBidomainRoute,
    HeartTorsoBidomainRoute,
    initialize_bidomain_state,
    step_bidomain,
    step_proportional_monodomain_limit,
    zero_bidomain_inputs,
)
from phydrax.applications.cardiovascular.electrophysiology._conduction_network import (
    initialize_purkinje_state,
    make_purkinje_stimulus_batch,
    propagate_purkinje,
    PurkinjeEventKind,
    PurkinjeNetworkPlan,
    with_purkinje_edge_block,
)
from phydrax.applications.cardiovascular.electrophysiology._eikonal import (
    AnisotropicEikonalPlan,
    FiniteElementEikonalRoute,
    GraphEikonalRoute,
    solve_anisotropic_eikonal,
)
from phydrax.applications.cardiovascular.electrophysiology._pacing import (
    DemandPacingControllerPlan,
    evaluate_pacing_protocol,
    evaluate_pmj_exchange,
    initialize_demand_pacing_controller,
    PacingProtocol,
    PMJExchangePlan,
    schedule_pmj_activations,
    step_demand_pacing_controller,
    TissuePacingTarget,
)


def test_anisotropic_graph_and_fem_eikonal_match_analytic_travel_times():
    graph = GraphEikonalRoute(
        jnp.asarray((10, 20, 30)),
        jnp.asarray((100, 101)),
        jnp.asarray(((0, 1), (1, 2))),
    )
    positions = jnp.asarray(((0.0, 0.0), (2.0, 0.0), (5.0, 0.0)))
    squared_velocity = jnp.asarray(((4.0, 0.0), (0.0, 1.0)))
    result = solve_anisotropic_eikonal(
        AnisotropicEikonalPlan(graph, positions, squared_velocity).prepare(),
        jnp.asarray((0,)),
        jnp.asarray((0.0,)),
    )
    np.testing.assert_allclose(result.arrival_time_ms, (0.0, 1.0, 2.5), atol=1.0e-6)
    np.testing.assert_array_equal(result.predecessor_index, (-1, 0, 1))
    assert bool(result.evidence.successful)
    assert bool(result.evidence.fixed_topology_derivative_valid)

    fem = FiniteElementEikonalRoute(
        jnp.asarray((10, 20, 30)),
        jnp.asarray((200,)),
        jnp.asarray(((0, 1, 2),)),
    )
    triangle_positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)))
    fem_result = solve_anisotropic_eikonal(
        AnisotropicEikonalPlan(fem, triangle_positions, jnp.eye(2)).prepare(),
        jnp.asarray((0, 1)),
        jnp.asarray((0.0, 0.0)),
    )
    np.testing.assert_allclose(fem_result.arrival_time_ms, (0.0, 0.0, 1.0), atol=1.0e-6)
    assert bool(fem_result.evidence.successful)


def test_eikonal_source_time_remains_differentiable():
    route = GraphEikonalRoute(
        jnp.asarray((10, 20, 30)),
        jnp.asarray((100, 101)),
        jnp.asarray(((0, 1), (1, 2))),
    )
    prepared = AnisotropicEikonalPlan(
        route,
        jnp.asarray(((0.0,), (1.0,), (2.0,))),
        jnp.asarray(((1.0,),)),
    ).prepare()
    derivative = jax.grad(
        lambda source_time: solve_anisotropic_eikonal(
            prepared,
            jnp.asarray((0,)),
            jnp.asarray((source_time,)),
        ).arrival_time_ms[2]
    )(jnp.asarray(3.0))
    np.testing.assert_allclose(derivative, 1.0)


def _line_network(*, event_capacity=24):
    return PurkinjeNetworkPlan(
        jnp.asarray((10, 20, 30)),
        jnp.asarray((100, 101)),
        jnp.asarray(((0, 1), (1, 2))),
        jnp.asarray((1.0, 2.0)),
        5.0,
        event_capacity=event_capacity,
        stimulus_capacity=4,
    )


def test_purkinje_delay_refractory_block_and_deterministic_events():
    plan = _line_network()
    state = initialize_purkinje_state(plan)
    first = propagate_purkinje(
        plan,
        state,
        make_purkinje_stimulus_batch(plan, (1,), (0,), (0.0,)),
    )
    active = np.asarray(first.events.active)
    kinds = np.asarray(first.events.kind)[active]
    times = np.asarray(first.events.time_ms)[active]
    np.testing.assert_allclose(
        times[kinds != int(PurkinjeEventKind.EDGE_BLOCK)], (0.0, 1.0, 3.0)
    )
    np.testing.assert_allclose(first.state.latest_activation_time_ms, (0.0, 1.0, 3.0))
    assert bool(first.evidence.deterministic_order)

    refractory = propagate_purkinje(
        plan,
        first.state,
        make_purkinje_stimulus_batch(plan, (20,), (0,), (4.0,)),
    )
    assert int(refractory.evidence.refractory_rejection_count) == 1
    np.testing.assert_allclose(
        refractory.state.latest_activation_time_ms, first.state.latest_activation_time_ms
    )

    blocked_state = with_purkinje_edge_block(
        plan, refractory.state, jnp.asarray((False, True))
    )
    blocked = propagate_purkinje(
        plan,
        blocked_state,
        make_purkinje_stimulus_batch(plan, (30,), (0,), (10.0,)),
    )
    assert int(blocked.evidence.blocked_wave_count) == 1
    assert float(blocked.state.latest_activation_time_ms[2]) == 3.0


def test_purkinje_event_overflow_exposes_candidate_and_fails_closed():
    plan = _line_network(event_capacity=1)
    initial = initialize_purkinje_state(plan)
    result = propagate_purkinje(
        plan,
        initial,
        make_purkinje_stimulus_batch(plan, (1,), (0,), (0.0,)),
    )
    assert bool(result.evidence.overflowed)
    assert not bool(result.evidence.successful)
    assert np.isneginf(float(result.state.latest_activation_time_ms[0]))
    assert float(result.candidate_state.latest_activation_time_ms[0]) == 0.0


def test_pmj_requires_successful_propagation_from_bound_network():
    bound_network = _line_network(event_capacity=1)
    rejected = propagate_purkinje(
        bound_network,
        initialize_purkinje_state(bound_network),
        make_purkinje_stimulus_batch(bound_network, (1,), (0,), (0.0,)),
    )
    pmj = PMJExchangePlan(
        jnp.asarray((900,)),
        jnp.asarray((0,)),
        jnp.asarray((0,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.2,)),
        purkinje_plan=bound_network,
        tissue_node_count=1,
        event_capacity=2,
    )
    with pytest.raises(ValueError, match="unsuccessful Purkinje"):
        schedule_pmj_activations(pmj, rejected, jnp.asarray((-jnp.inf,)))
    with pytest.raises(TypeError, match="PurkinjePropagationResult"):
        schedule_pmj_activations(pmj, rejected.events, jnp.asarray((-jnp.inf,)))

    other_network = _line_network(event_capacity=8)
    accepted_other = propagate_purkinje(
        other_network,
        initialize_purkinje_state(other_network),
        make_purkinje_stimulus_batch(other_network, (1,), (0,), (0.0,)),
    )
    with pytest.raises(ValueError, match="another PMJ-bound plan"):
        schedule_pmj_activations(pmj, accepted_other, jnp.asarray((-jnp.inf,)))


def test_purkinje_antiparallel_waves_collide_before_node_arrival():
    plan = PurkinjeNetworkPlan(
        jnp.asarray((10, 20)),
        jnp.asarray((100,)),
        jnp.asarray(((0, 1),)),
        jnp.asarray((2.0,)),
        10.0,
        event_capacity=8,
        stimulus_capacity=2,
    )
    result = propagate_purkinje(
        plan,
        initialize_purkinje_state(plan),
        make_purkinje_stimulus_batch(plan, (1, 2), (0, 1), (0.0, 0.0)),
    )
    active = np.asarray(result.events.active)
    kinds = np.asarray(result.events.kind)[active]
    times = np.asarray(result.events.time_ms)[active]
    assert np.sum(kinds == int(PurkinjeEventKind.WAVE_COLLISION)) == 1
    np.testing.assert_allclose(
        times[kinds == int(PurkinjeEventKind.WAVE_COLLISION)], (1.0,)
    )
    assert int(result.evidence.activation_count) == 2
    assert not bool(result.evidence.fixed_event_sequence_derivative_valid)


def test_pmj_exchange_timing_support_and_pacing_controller_state():
    network = _line_network()
    propagated = propagate_purkinje(
        network,
        initialize_purkinje_state(network),
        make_purkinje_stimulus_batch(network, (1,), (0,), (0.0,)),
    )
    pmj = PMJExchangePlan(
        jnp.asarray((900,)),
        jnp.asarray((2,)),
        jnp.asarray((1,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.2,)),
        purkinje_plan=network,
        tissue_node_count=2,
        event_capacity=4,
    )
    exchange = evaluate_pmj_exchange(
        pmj,
        jnp.asarray((0.0, 0.0, 10.0)),
        jnp.asarray((0.0, -10.0)),
    )
    np.testing.assert_allclose(exchange.junction_current_purkinje_to_tissue_uA, (4.0,))
    np.testing.assert_allclose(exchange.purkinje_current_uA, (0.0, 0.0, -4.0))
    np.testing.assert_allclose(exchange.tissue_current_uA, (0.0, 4.0))
    assert bool(exchange.evidence.conservative)

    scheduled = schedule_pmj_activations(
        pmj, propagated, jnp.asarray((-jnp.inf, -jnp.inf))
    )
    assert int(scheduled.evidence.accepted_count) == 1
    np.testing.assert_allclose(
        scheduled.activations.activation_time_ms[scheduled.activations.active], (3.5,)
    )
    np.testing.assert_array_equal(
        scheduled.activations.tissue_node_index[scheduled.activations.active], (1,)
    )

    target = TissuePacingTarget(2, jnp.asarray((1,)))
    protocol = PacingProtocol(
        target,
        jnp.asarray((7,)),
        jnp.asarray((5.0,)),
        jnp.asarray((2.0,)),
        jnp.asarray((12.0,)),
    )
    np.testing.assert_allclose(
        evaluate_pacing_protocol(protocol, 6.0).nodal_stimulus_uA_per_mm3,
        (0.0, 12.0),
    )

    controller_plan = DemandPacingControllerPlan(
        1,
        escape_interval_ms=1000.0,
        target_sensed_interval_ms=800.0,
        minimum_cycle_length_ms=500.0,
        maximum_cycle_length_ms=1200.0,
        feedback_gain=0.5,
        duration_ms=2.0,
        amplitude_uA_per_mm3=20.0,
    )
    controller = initialize_demand_pacing_controller(controller_plan, 0.0)
    sensed = step_demand_pacing_controller(
        controller_plan, controller, 700.0, sensed_activation_time_ms=700.0
    )
    assert not bool(sensed.command.emitted)
    paced = step_demand_pacing_controller(controller_plan, sensed.state, 1700.0)
    assert bool(paced.command.emitted)
    assert int(paced.state.command_count) == 1
    assert float(paced.command.start_time_ms) == 1700.0


def _heart_only_bidomain():
    plan = BidomainFEMPlan(
        HeartOnlyBidomainRoute(),
        jnp.asarray((10, 20, 30)),
        jnp.asarray((100, 101)),
        jnp.asarray(((0.0,), (1.0,), (2.0,))),
        jnp.asarray(((0, 1), (1, 2))),
        jnp.asarray(((1.0,),)),
        jnp.asarray(((2.0,),)),
        dt_ms=0.1,
        membrane_capacitance_uF_per_mm3=1.0,
        residual_tolerance=2.0e-5,
        gauge_tolerance_mV=2.0e-5,
        source_compatibility_tolerance_uA=2.0e-5,
    )
    return plan.prepare()


def test_bidomain_gauge_nullspace_block_residual_and_monodomain_limit():
    prepared = _heart_only_bidomain()
    state = initialize_bidomain_state(prepared, jnp.asarray((0.0, 1.0, 0.0)))
    inputs = zero_bidomain_inputs(prepared)
    bidomain = step_bidomain(prepared, state, inputs)
    monodomain = step_proportional_monodomain_limit(prepared, state, inputs, 2.0)
    assert bool(bidomain.evidence.successful)
    assert bool(bidomain.evidence.gauge.fixed_gauge)
    assert float(bidomain.evidence.gauge.ungauged_nullspace_residual) < 1.0e-6
    assert float(bidomain.evidence.block_residual.relative_norm) < 2.0e-5
    assert bool(bidomain.evidence.preconditioner.finite)
    assert bool(monodomain.evidence.successful)
    np.testing.assert_allclose(
        bidomain.state.transmembrane_voltage_mV,
        monodomain.transmembrane_voltage_mV,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    incompatible = BidomainStepInputs(
        inputs.ionic_current_uA_per_mm3,
        inputs.transmembrane_stimulus_uA_per_mm3,
        jnp.ones((prepared.heart_node_count,)),
        inputs.torso_source_uA_per_mm3,
    )
    rejected = step_bidomain(prepared, state, incompatible)
    assert int(rejected.evidence.status) & int(
        BidomainSolveStatus.INCOMPATIBLE_EXTERNAL_SOURCE
    )
    np.testing.assert_allclose(
        rejected.state.transmembrane_voltage_mV, state.transmembrane_voltage_mV
    )
    rejected_limit = jax.jit(
        lambda candidate_inputs: step_proportional_monodomain_limit(
            prepared, state, candidate_inputs, 2.0
        )
    )(incompatible)
    assert not bool(rejected_limit.evidence.successful)


def test_heart_torso_interface_is_monolithic_and_flux_conservative():
    route = HeartTorsoBidomainRoute(
        jnp.asarray((1000, 1001)),
        jnp.asarray((1100,)),
        jnp.asarray(((1.0,), (2.0,))),
        jnp.asarray(((0, 1),)),
        jnp.asarray(((0.5,),)),
        jnp.asarray((1200,)),
        jnp.asarray(((1, 0),)),
        jnp.asarray((3.0,)),
    )
    prepared = BidomainFEMPlan(
        route,
        jnp.asarray((10, 20)),
        jnp.asarray((100,)),
        jnp.asarray(((0.0,), (1.0,))),
        jnp.asarray(((0, 1),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray(((2.0,),)),
        dt_ms=0.1,
        membrane_capacitance_uF_per_mm3=1.0,
        residual_tolerance=2.0e-5,
        gauge_tolerance_mV=2.0e-5,
        source_compatibility_tolerance_uA=2.0e-5,
    ).prepare()
    state = initialize_bidomain_state(prepared, jnp.asarray((0.0, 1.0)))
    result = step_bidomain(prepared, state, zero_bidomain_inputs(prepared))
    assert bool(result.evidence.successful)
    assert bool(result.evidence.interface.supported)
    np.testing.assert_allclose(
        prepared.interface_coupling_matrix,
        ((0.0, 0.0), (3.0, 0.0)),
    )
    assert np.isfinite(float(result.evidence.interface.interface_current_norm_uA))
    assert float(result.evidence.interface.flux_balance_error_uA) == 0.0
    combined = jnp.concatenate(
        (result.state.extracellular_potential_mV, result.state.torso_potential_mV)
    )
    assert abs(float(prepared.gauge_weights @ combined)) < 2.0e-5
