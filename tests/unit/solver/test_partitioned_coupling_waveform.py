#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


cpl = phx.solver.coupling


def _waveform_capabilities():
    return cpl.CouplingSubsystemCapabilities(
        jit=True,
        differentiable=True,
        deterministic_replay=True,
        fixed_topology=True,
        supports_endpoint=False,
        supports_waveform=True,
    )


def test_barycentric_waveform_interpolation_is_exact_and_capacity_padded():
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="waveform-scalar")
    source_plan = cpl.CouplingWaveformPlan(4, 2, (0.0, 0.5, 1.0))
    target_plan = cpl.CouplingWaveformPlan(5, 2, (0.0, 0.25, 0.75, 1.0))
    source_grid = source_plan.initial_grid()
    target_grid = target_plan.initial_grid()
    waveform = cpl.CouplingWaveform(
        source_grid,
        jnp.asarray([[0.0], [0.25], [1.0], [0.0]], dtype=jnp.float64),
        space,
    )

    transferred = cpl.BarycentricCouplingTemporalTransfer(2).interpolate(
        waveform, target_grid, space
    )

    assert jnp.allclose(
        transferred.values[:, 0],
        jnp.asarray([0.0, 0.0625, 0.5625, 1.0, 0.0]),
    )
    assert not transferred.grid.active[-1]
    assert transferred.values[-1, 0] == 0.0


def _waveform_graph(*, parameterized=False):
    waveform_plan = cpl.CouplingWaveformPlan(
        3, 1, (0.0, 0.5, 1.0), plan_id="canonical-coupling-grid"
    )
    grid = waveform_plan.initial_grid()
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="waveform-interface")
    a_input = cpl.CouplingPort(
        "a-input",
        "input",
        space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )
    a_output = cpl.CouplingPort(
        "a-output",
        "output",
        space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )
    b_input = cpl.CouplingPort(
        "b-input",
        "input",
        space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )
    b_output = cpl.CouplingPort(
        "b-output",
        "output",
        space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )

    def advance_a(window, state, inputs, args):
        del window, state, args
        waveform = cpl.CouplingWaveform(grid, 0.5 * inputs[0].values, space)
        return cpl.CouplingSubsystemResult(
            waveform.values[-1], (waveform,), successful=True, status=0
        )

    def advance_b(window, state, inputs, args):
        del window, state
        forcing = args if parameterized else jnp.asarray(1.0)
        waveform = cpl.CouplingWaveform(grid, 0.5 * (inputs[0].values + forcing), space)
        return cpl.CouplingSubsystemResult(
            waveform.values[-1], (waveform,), successful=True, status=0
        )

    a = cpl.CallableCouplingSubsystem(
        advance_a,
        subsystem_id="a",
        input_ports=(a_input,),
        output_ports=(a_output,),
        capabilities=_waveform_capabilities(),
    )
    b = cpl.CallableCouplingSubsystem(
        advance_b,
        subsystem_id="b",
        input_ports=(b_input,),
        output_ports=(b_output,),
        capabilities=_waveform_capabilities(),
    )
    graph = cpl.CouplingGraph(
        (a, b),
        (
            cpl.CouplingExchange("a-to-b", "a-output", "b-input"),
            cpl.CouplingExchange("b-to-a", "b-output", "a-input"),
        ),
    )
    zero = cpl.CouplingWaveform.constant(grid, jnp.zeros(1, dtype=jnp.float64), space)
    return graph, (jnp.zeros(1), jnp.zeros(1)), (zero, zero)


def _waveform_fixed_point_policy():
    return cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(
            acceleration=phx.nonlinear.AndersonAcceleration(history=4)
        ),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-10,
            relative_residual=0.0,
            maximum_steps=40,
        ),
        (
            cpl.CouplingTolerance("a-input", absolute=1e-9),
            cpl.CouplingTolerance("b-input", absolute=1e-9),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )


def test_waveform_fixed_point_and_jit_certify_every_canonical_sample():
    graph, states, values = _waveform_graph()
    prepared = cpl.prepare_coupling(
        graph, states, values, policy=_waveform_fixed_point_policy()
    )

    result = eqx.filter_jit(cpl.advance_coupling_window)(
        prepared, prepared.reference_state, 1.0, None
    )

    assert bool(result.successful)
    assert bool(result.converged)
    assert jnp.allclose(
        result.accepted_state.exchange_values[0].values,
        jnp.full((3, 1), 1.0 / 3.0),
        atol=1e-8,
    )
    assert jnp.allclose(
        result.accepted_state.exchange_values[1].values,
        jnp.full((3, 1), 2.0 / 3.0),
        atol=1e-8,
    )


def test_fixed_grid_subcycling_adapter_samples_each_substep_endpoint():
    waveform_plan = cpl.CouplingWaveformPlan(
        3, 1, (0.0, 0.5, 1.0), plan_id="subcycle-grid"
    )
    grid = waveform_plan.initial_grid()
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="subcycle-scalar")
    input_port = cpl.CouplingPort(
        "input",
        "input",
        space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )
    output_port = cpl.CouplingPort(
        "output",
        "output",
        space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )

    def substep(window, state, inputs, args):
        del args
        candidate = state + window.size * inputs[0]
        return cpl.CouplingSubsystemResult(
            candidate, (candidate,), successful=True, status=0, work=1
        )

    subsystem = cpl.FixedGridSubcyclingSubsystem(
        substep,
        lambda state, inputs, args: (state,),
        subsystem_id="subcycling",
        input_ports=(input_port,),
        output_ports=(output_port,),
        differentiable=True,
    )
    input_waveform = cpl.CouplingWaveform.constant(grid, jnp.ones(1), space)
    result = subsystem.advance_window(
        cpl.CouplingWindow(0, 0.0, 1.0),
        jnp.zeros(1),
        (input_waveform,),
        None,
    )

    assert bool(result.successful)
    assert int(result.work) == 2
    assert jnp.allclose(result.candidate_state, 1.0)
    assert jnp.allclose(result.outputs[0].values[:, 0], jnp.asarray([0.0, 0.5, 1.0]))


def test_fixed_grid_subcycling_stops_work_after_the_first_failed_substep():
    waveform_plan = cpl.CouplingWaveformPlan(
        3, 1, (0.0, 0.5, 1.0), plan_id="failing-subcycle-grid"
    )
    grid = waveform_plan.initial_grid()
    space = phx.linalg.ArraySpace(
        (1,), dtype=jnp.float64, space_id="failing-subcycle-scalar"
    )
    input_port = cpl.CouplingPort(
        "failing-input",
        "input",
        space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )
    output_port = cpl.CouplingPort(
        "failing-output",
        "output",
        space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )

    def fail(window, state, inputs, args):
        del window, inputs, args
        candidate = state + 1.0
        return cpl.CouplingSubsystemResult(
            candidate, (candidate,), successful=False, status=9, work=1
        )

    subsystem = cpl.FixedGridSubcyclingSubsystem(
        fail,
        lambda state, inputs, args: (state,),
        subsystem_id="failing-subcycling",
        input_ports=(input_port,),
        output_ports=(output_port,),
        differentiable=True,
    )
    input_waveform = cpl.CouplingWaveform.constant(grid, jnp.ones(1), space)

    result = subsystem.advance_window(
        cpl.CouplingWindow(0, 0.0, 1.0),
        jnp.zeros(1),
        (input_waveform,),
        None,
    )

    assert not bool(result.successful)
    assert int(result.status) == 9
    assert int(result.work) == 1
    assert jnp.allclose(result.candidate_state, 1.0)
    assert jnp.allclose(result.outputs[0].values[:, 0], jnp.asarray([0.0, 1.0, 1.0]))


def test_waveform_implicit_root_derivative_uses_the_fixed_sample_grid():
    graph, states, values = _waveform_graph(parameterized=True)
    policy = cpl.ImplicitCouplingPolicy(
        phx.nonlinear.NewtonKrylov(),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-12,
            relative_residual=0.0,
            maximum_steps=12,
        ),
        (
            cpl.CouplingTolerance("a-input", absolute=1e-10),
            cpl.CouplingTolerance("b-input", absolute=1e-10),
        ),
    )
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=policy,
        differentiation=cpl.CouplingDifferentiationPolicy("implicit"),
        args=jnp.asarray(1.0, dtype=jnp.float64),
    )

    def observable(parameter):
        result = cpl.advance_coupling_window(
            prepared, prepared.reference_state, 1.0, parameter
        )
        return result.accepted_state.exchange_values[0].values[-1, 0]

    value, derivative = jax.value_and_grad(observable)(
        jnp.asarray(1.0, dtype=jnp.float64)
    )
    assert float(value) == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert float(derivative) == pytest.approx(1.0 / 3.0, abs=1e-8)


def _waveform_field_space(name):
    topology = phx.discretization.TensorTopology(("x",), (2,))
    support = phx.discretization.DiscreteSupport(topology, 1, f"{name}-waveform-support")
    layout = phx.discretization.TensorDofLayout(("x",), (2,))
    vectors = phx.linalg.ArraySpace((2,), space_id=f"{name}-waveform-vectors")
    return phx.discretization.DiscreteFieldSpace(
        name,
        support.support_id,
        layout,
        vectors,
        representation="point_value",
    )


def test_field_transfer_is_applied_samplewise_to_waveform_exchanges():
    waveform_plan = cpl.CouplingWaveformPlan(
        3, 1, (0.0, 0.5, 1.0), plan_id="field-waveform-grid"
    )
    grid = waveform_plan.initial_grid()
    source_space = _waveform_field_space("source")
    target_space = _waveform_field_space("target")
    matrix = jnp.asarray([[1.0, 0.25], [0.5, 1.0]])
    transfer = phx.discretization.FieldTransfer(
        source_space,
        target_space,
        phx.linalg.DenseLinearOperator(
            matrix,
            source=source_space.vector_space,
            target=target_space.vector_space,
        ),
        adjoint_operator=phx.linalg.DenseLinearOperator(
            matrix.T,
            source=target_space.vector_space,
            target=source_space.vector_space,
        ),
        properties=phx.discretization.TransferProperties(adjoint_paired=True),
    )
    source_input = cpl.CouplingPort(
        "source-input",
        "input",
        source_space.vector_space,
        field_space=source_space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )
    source_output = cpl.CouplingPort(
        "source-output",
        "output",
        source_space.vector_space,
        field_space=source_space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )
    target_input = cpl.CouplingPort(
        "target-input",
        "input",
        target_space.vector_space,
        field_space=target_space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )
    target_output = cpl.CouplingPort(
        "target-output",
        "output",
        target_space.vector_space,
        field_space=target_space,
        waveform_plan=waveform_plan,
        temporal_transfer=cpl.BarycentricCouplingTemporalTransfer(1),
        reference_scale=1.0,
    )
    source_values = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    target_values = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    source_waveform = cpl.CouplingWaveform(grid, source_values, source_space.vector_space)
    target_waveform = cpl.CouplingWaveform(grid, target_values, target_space.vector_space)
    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (source_waveform,), successful=True, status=0
        ),
        subsystem_id="source",
        input_ports=(source_input,),
        output_ports=(source_output,),
        capabilities=_waveform_capabilities(),
    )
    target = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (target_waveform,), successful=True, status=0
        ),
        subsystem_id="target",
        input_ports=(target_input,),
        output_ports=(target_output,),
        capabilities=_waveform_capabilities(),
    )
    graph = cpl.CouplingGraph(
        (source, target),
        (
            cpl.CouplingExchange(
                "forward",
                "source-output",
                "target-input",
                transfer=transfer,
            ),
            cpl.CouplingExchange(
                "adjoint",
                "target-output",
                "source-input",
                transfer=transfer,
                use_adjoint=True,
            ),
        ),
    )
    zero_source = cpl.CouplingWaveform.constant(
        grid, jnp.zeros(2), source_space.vector_space
    )
    zero_target = cpl.CouplingWaveform.constant(
        grid, jnp.zeros(2), target_space.vector_space
    )
    prepared = cpl.prepare_coupling(
        graph,
        (jnp.zeros(1), jnp.zeros(1)),
        (zero_target, zero_source),
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
    )

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    adjoint_values = result.accepted_state.exchange_values[0].values
    forward_values = result.accepted_state.exchange_values[1].values
    assert jnp.allclose(forward_values, source_values @ matrix.T)
    assert jnp.allclose(adjoint_values, target_values @ matrix)


def test_waveform_adaptation_activates_one_candidate_and_requests_growth():
    adaptation = cpl.CouplingWaveformAdaptationPolicy(
        (0.25, 0.75), observable_tolerance=0.1
    )
    plan = cpl.CouplingWaveformPlan(
        3, 1, (0.0, 1.0), adaptation=adaptation, plan_id="adaptive-grid"
    )
    refined, evidence, request = cpl.adapt_coupling_waveform_grid(
        plan, plan.initial_grid(), jnp.asarray((0.2, 2.0)), "temperature"
    )
    assert evidence.activated
    assert refined.sample_count == 3
    assert jnp.allclose(refined.nodes, jnp.asarray((0.0, 0.75, 1.0)))
    _, exhausted, request = cpl.adapt_coupling_waveform_grid(
        plan, refined, jnp.asarray((2.0, 2.0)), "temperature"
    )
    assert exhausted.capacity_exhausted
    assert request.required_samples == 4


def test_coupling_epoch_transition_is_explicit_and_atomic():
    graph, states, values = _waveform_graph()
    prepared = cpl.prepare_coupling(
        graph, states, values, policy=_waveform_fixed_point_policy()
    )
    current_epoch = cpl.PreparedCouplingEpoch(
        prepared,
        ("a-epoch-0", "b-epoch-0"),
        ("waveform-capacity-0",),
    )
    target_epoch = cpl.PreparedCouplingEpoch(
        prepared,
        ("a-epoch-1", "b-epoch-1"),
        ("waveform-capacity-1",),
    )
    request = cpl.CouplingTopologyRequest(
        True,
        jnp.asarray((1, 1), dtype=jnp.int32),
        jnp.asarray((3, 3), dtype=jnp.int32),
        1,
    )
    identities = (
        cpl.IdentityCouplingEpochTransfer(),
        cpl.IdentityCouplingEpochTransfer(),
    )
    transition = cpl.CouplingEpochTransitionPlan(
        identities,
        identities,
        (),
        (),
        source_subsystem_ids=prepared.reference_state.subsystem_ids,
        target_subsystem_ids=prepared.reference_state.subsystem_ids,
        source_exchange_ids=prepared.reference_state.exchange_ids,
        target_exchange_ids=prepared.reference_state.exchange_ids,
        transition_id="identity-epoch-transition",
    )
    accepted = cpl.transition_coupling_epoch(
        current_epoch,
        prepared.reference_state,
        target_epoch,
        transition,
        request,
        accepted_window=True,
    )

    assert accepted.successful
    assert accepted.epoch.epoch_id == target_epoch.epoch_id
    assert accepted.state.subsystem_ids == prepared.reference_state.subsystem_ids
    assert accepted.state.exchange_ids == prepared.reference_state.exchange_ids

    failed_transfer = cpl.CallableCouplingEpochTransfer(
        lambda value, args: cpl.CouplingEpochTransferResult(value, jnp.asarray(False)),
        transfer_id="failed-retained-state-transfer",
    )
    failed_plan = cpl.CouplingEpochTransitionPlan(
        (failed_transfer, cpl.IdentityCouplingEpochTransfer()),
        identities,
        (),
        (),
        source_subsystem_ids=prepared.reference_state.subsystem_ids,
        target_subsystem_ids=prepared.reference_state.subsystem_ids,
        source_exchange_ids=prepared.reference_state.exchange_ids,
        target_exchange_ids=prepared.reference_state.exchange_ids,
        transition_id="failed-epoch-transition",
    )
    rejected = cpl.transition_coupling_epoch(
        current_epoch,
        prepared.reference_state,
        target_epoch,
        failed_plan,
        request,
        accepted_window=True,
    )

    assert not rejected.successful
    assert rejected.epoch.epoch_id == current_epoch.epoch_id
    assert rejected.state is prepared.reference_state

    ignored = cpl.transition_coupling_epoch(
        current_epoch,
        prepared.reference_state,
        target_epoch,
        transition,
        request,
        accepted_window=False,
    )
    assert not ignored.successful
    assert ignored.epoch.epoch_id == current_epoch.epoch_id
    assert ignored.state is prepared.reference_state
