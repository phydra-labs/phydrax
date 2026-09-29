#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cpl = phx.solver.coupling


def _waveform_capabilities() -> Any:
    return cpl.CouplingSubsystemCapabilities(
        jit=True,
        differentiable=True,
        deterministic_replay=True,
        fixed_topology=True,
        supports_endpoint=False,
        supports_waveform=True,
    )


def _linear_interpolation() -> Any:
    return cpl.CouplingTemporalConversion(
        "interpolate", transfer=cpl.BarycentricCouplingTemporalTransfer(1)
    )


def test_partitioned_coupling_waveform_scenario_1() -> None:
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
    waveform_plan = cpl.CouplingWaveformPlan(
        3, 1, (0.0, 0.5, 1.0), plan_id="field-waveform-grid"
    )
    grid = waveform_plan.initial_grid()
    source_space = _waveform_field_space("source")
    target_space = _waveform_field_space("target")
    matrix = jnp.asarray([[1.0, 0.25], [0.5, 1.0]])
    adjoint = phx.linalg.DenseLinearOperator(
        matrix.T,
        source=target_space.vector_space,
        target=source_space.vector_space,
    )
    transfer = phx.discretization.FieldTransfer(
        source_space,
        target_space,
        phx.linalg.DenseLinearOperator(
            matrix,
            source=source_space.vector_space,
            target=target_space.vector_space,
        ),
        dual_pullback_operator=adjoint,
        hilbert_adjoint_operator=adjoint,
        properties=phx.discretization.TransferProperties(adjoint_paired=True),
    )
    source_input = cpl.CouplingPort(
        "source-input",
        "input",
        source_space.vector_space,
        field_space=source_space,
        waveform_plan=waveform_plan,
        reference_scale=1.0,
    )
    source_output = cpl.CouplingPort(
        "source-output",
        "output",
        source_space.vector_space,
        field_space=source_space,
        waveform_plan=waveform_plan,
        reference_scale=1.0,
    )
    target_input = cpl.CouplingPort(
        "target-input",
        "input",
        target_space.vector_space,
        field_space=target_space,
        waveform_plan=waveform_plan,
        reference_scale=1.0,
    )
    target_output = cpl.CouplingPort(
        "target-output",
        "output",
        target_space.vector_space,
        field_space=target_space,
        waveform_plan=waveform_plan,
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
                temporal=_linear_interpolation(),
            ),
            cpl.CouplingExchange(
                "adjoint",
                "target-output",
                "source-input",
                transfer=transfer,
                use_adjoint=True,
                temporal=_linear_interpolation(),
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


def _waveform_graph(*, parameterized: Any = False) -> Any:
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
        reference_scale=1.0,
    )
    a_output = cpl.CouplingPort(
        "a-output",
        "output",
        space,
        waveform_plan=waveform_plan,
        reference_scale=1.0,
    )
    b_input = cpl.CouplingPort(
        "b-input",
        "input",
        space,
        waveform_plan=waveform_plan,
        reference_scale=1.0,
    )
    b_output = cpl.CouplingPort(
        "b-output",
        "output",
        space,
        waveform_plan=waveform_plan,
        reference_scale=1.0,
    )

    def advance_a(window: Any, state: Any, inputs: Any, args: Any) -> Any:
        del window, state, args
        waveform = cpl.CouplingWaveform(grid, 0.5 * inputs[0].values, space)
        return cpl.CouplingSubsystemResult(
            waveform.values[-1], (waveform,), successful=True, status=0
        )

    def advance_b(window: Any, state: Any, inputs: Any, args: Any) -> Any:
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
            cpl.CouplingExchange(
                "a-to-b", "a-output", "b-input", temporal=_linear_interpolation()
            ),
            cpl.CouplingExchange(
                "b-to-a", "b-output", "a-input", temporal=_linear_interpolation()
            ),
        ),
    )
    zero = cpl.CouplingWaveform.constant(grid, jnp.zeros(1, dtype=jnp.float64), space)
    return graph, (jnp.zeros(1), jnp.zeros(1)), (zero, zero)


def _waveform_fixed_point_policy() -> Any:
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


def test_fixed_grid_subcycling_adapter_samples_each_substep_endpoint() -> None:
    waveform_plan = cpl.CouplingWaveformPlan(
        4, 1, (0.0, 0.5, 1.0), plan_id="subcycle-grid"
    )
    grid = waveform_plan.initial_grid()
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="subcycle-scalar")
    input_port = cpl.CouplingPort(
        "input",
        "input",
        space,
        waveform_plan=waveform_plan,
        reference_scale=1.0,
    )
    output_port = cpl.CouplingPort(
        "output",
        "output",
        space,
        waveform_plan=waveform_plan,
        reference_scale=1.0,
    )

    def substep(window: Any, state: Any, inputs: Any, args: Any) -> Any:
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
    assert jnp.allclose(result.outputs[0].values[:, 0], jnp.asarray([0.0, 0.5, 1.0, 0.0]))
    assert not result.outputs[0].grid.active[-1]


def test_fixed_grid_subcycling_stops_work_after_the_first_failed_substep() -> None:
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
        reference_scale=1.0,
    )
    output_port = cpl.CouplingPort(
        "failing-output",
        "output",
        space,
        waveform_plan=waveform_plan,
        reference_scale=1.0,
    )

    def fail(window: Any, state: Any, inputs: Any, args: Any) -> Any:
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


def test_waveform_implicit_root_derivative_uses_the_fixed_sample_grid() -> None:
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

    def observable(parameter: Any) -> Any:
        result = cpl.advance_coupling_window(
            prepared, prepared.reference_state, 1.0, parameter
        )
        return result.accepted_state.exchange_values[0].values[-1, 0]

    value, derivative = jax.value_and_grad(observable)(
        jnp.asarray(1.0, dtype=jnp.float64)
    )
    assert float(value) == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert float(derivative) == pytest.approx(1.0 / 3.0, abs=1e-8)


def _waveform_field_space(name: Any) -> Any:
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


def test_coupling_epoch_transition_contracts() -> None:
    graph, states, values = _waveform_graph()
    prepared = cpl.prepare_coupling(
        graph, states, values, policy=_waveform_fixed_point_policy()
    )
    current_epoch = cpl.PreparedCouplingEpoch(
        prepared,
        ("a-epoch-0", "b-epoch-0"),
        ("waveform-capacity-0",),
        participant_epoch_codes=(0, 0),
        waveform_required_samples=(2, 2),
        topology_code=0,
    )
    target_epoch = cpl.PreparedCouplingEpoch(
        prepared,
        ("a-epoch-1", "b-epoch-1"),
        ("waveform-capacity-1",),
        participant_epoch_codes=(1, 1),
        waveform_required_samples=(3, 3),
        topology_code=1,
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
    graph, states, values = _waveform_graph()
    prepared = cpl.prepare_coupling(
        graph, states, values, policy=_waveform_fixed_point_policy()
    )
    current = cpl.PreparedCouplingEpoch(
        prepared,
        ("a-0", "b-0"),
        ("wave-0",),
        participant_epoch_codes=(0, 0),
        waveform_required_samples=(2, 2),
        topology_code=0,
    )
    target = cpl.PreparedCouplingEpoch(
        prepared,
        ("a-1", "b-1"),
        ("wave-1",),
        participant_epoch_codes=(1, 1),
        waveform_required_samples=(3, 3),
        topology_code=1,
    )
    identity = cpl.IdentityCouplingEpochTransfer()
    transition = cpl.CouplingEpochTransitionPlan(
        (identity, identity),
        (identity, identity),
        (),
        (),
        source_subsystem_ids=prepared.reference_state.subsystem_ids,
        target_subsystem_ids=prepared.reference_state.subsystem_ids,
        source_exchange_ids=prepared.reference_state.exchange_ids,
        target_exchange_ids=prepared.reference_state.exchange_ids,
        transition_id="stale-request",
    )
    # ty: ignore[invalid-argument-type]
    stale = cpl.CouplingTopologyRequest(True, (1, 0), (3, 3), 1)
    result = cpl.transition_coupling_epoch(
        current,
        prepared.reference_state,
        target,
        transition,
        stale,
        accepted_window=True,
    )
    assert not result.successful
    assert result.epoch.epoch_id == current.epoch_id
    assert result.state is prepared.reference_state


def _mixed_graph(hold: Any, sample: Any) -> Any:
    plan = cpl.CouplingWaveformPlan(3, 2, (0.0, 0.25, 1.0), plan_id="mixed-grid")
    grid = plan.initial_grid()
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="mixed-scalar")
    source_waveform = cpl.CouplingWaveform(
        grid, jnp.asarray([[1.0], [2.0], [5.0]], dtype=jnp.float64), space
    )

    def advance_endpoint(window: Any, state: Any, inputs: Any, args: Any) -> Any:
        del window, state, args
        return cpl.CouplingSubsystemResult(
            inputs[0], (jnp.asarray([2.5], dtype=jnp.float64),), successful=True, status=0
        )

    def advance_waveform(window: Any, state: Any, inputs: Any, args: Any) -> Any:
        del window, state, args
        return cpl.CouplingSubsystemResult(
            inputs[0].values, (source_waveform,), successful=True, status=0
        )

    endpoint = cpl.CallableCouplingSubsystem(
        advance_endpoint,
        subsystem_id="endpoint",
        input_ports=(
            cpl.CouplingPort("endpoint-input", "input", space, reference_scale=1.0),
        ),
        output_ports=(
            cpl.CouplingPort("endpoint-output", "output", space, reference_scale=1.0),
        ),
        capabilities=cpl.CouplingSubsystemCapabilities(
            jit=True, differentiable=True, deterministic_replay=True, fixed_topology=True
        ),
    )
    waveform = cpl.CallableCouplingSubsystem(
        advance_waveform,
        subsystem_id="waveform",
        input_ports=(
            cpl.CouplingPort(
                "waveform-input", "input", space, waveform_plan=plan, reference_scale=1.0
            ),
        ),
        output_ports=(
            cpl.CouplingPort(
                "waveform-output",
                "output",
                space,
                waveform_plan=plan,
                reference_scale=1.0,
            ),
        ),
        capabilities=_waveform_capabilities(),
    )
    graph = cpl.CouplingGraph(
        (endpoint, waveform),
        (
            cpl.CouplingExchange(
                "endpoint-to-waveform", "endpoint-output", "waveform-input", temporal=hold
            ),
            cpl.CouplingExchange(
                "waveform-to-endpoint",
                "waveform-output",
                "endpoint-input",
                temporal=sample,
            ),
        ),
    )
    zero = jnp.zeros((1,), dtype=jnp.float64)
    return cpl.prepare_coupling(
        graph,
        (zero, jnp.zeros((3, 1), dtype=jnp.float64)),
        (cpl.CouplingWaveform.constant(grid, zero, space), zero),
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
    )


def test_undeclared_or_mismatched_temporal_conversions_are_refused() -> None:
    hold = cpl.CouplingTemporalConversion("hold")
    sample = cpl.CouplingTemporalConversion("sample-end")

    with pytest.raises(ValueError, match="requires temporal conversion 'hold'"):
        _mixed_graph(None, sample)
    with pytest.raises(ValueError, match="requires temporal conversion 'sample-end'"):
        _mixed_graph(hold, None)
    with pytest.raises(
        ValueError, match="requires temporal conversion 'hold' but declares 'sample-end'"
    ):
        _mixed_graph(sample, sample)
    with pytest.raises(
        ValueError,
        match="requires temporal conversion 'sample-end' but declares 'interpolate'",
    ):
        _mixed_graph(hold, _linear_interpolation())


def test_hold_and_sample_end_deliver_the_declared_endpoint_values() -> None:
    prepared = _mixed_graph(
        cpl.CouplingTemporalConversion("hold"),
        cpl.CouplingTemporalConversion("sample-end"),
    )
    step = eqx.filter_jit(cpl.advance_coupling_window)

    first = step(prepared, prepared.reference_state, 1.0)
    state = first.accepted_state
    held = state.exchange_values[state.exchange_ids.index("endpoint-to-waveform")]
    sampled = state.exchange_values[state.exchange_ids.index("waveform-to-endpoint")]

    assert bool(first.successful)
    assert jnp.array_equal(held.grid.nodes, jnp.asarray([0.0, 0.25, 1.0]))
    assert jnp.array_equal(held.values, jnp.full((3, 1), 2.5))
    assert jnp.array_equal(sampled, jnp.asarray([5.0]))
    second = step(prepared, state, 1.0).accepted_state
    endpoint_state = second.participant_states[second.subsystem_ids.index("endpoint")]
    waveform_state = second.participant_states[second.subsystem_ids.index("waveform")]
    assert jnp.array_equal(endpoint_state, jnp.asarray([5.0]))
    assert jnp.array_equal(waveform_state, jnp.full((3, 1), 2.5))


_FLUX = phx.units.derived_unit(
    "W/m²", ((phx.units.JOULE, 1), (phx.units.METER, -2), (phx.units.SECOND, -1))
)
_HEAT = phx.units.JOULE_PER_SQUARE_METER
_AREA = phx.units.derived_unit("m²", ((phx.units.METER, 2),))


def _integrated_flux(
    integrated: Any, time_unit: Any = phx.units.SECOND, plan: Any = None
) -> Any:
    plan = (
        cpl.CouplingWaveformPlan(3, 2, (0.0, 0.25, 1.0), plan_id="flux-grid")
        if plan is None
        else plan
    )
    grid = plan.initial_grid()
    measure = phx.discretization.DiscreteMeasure(
        "surface-area", "flux-surface", "flux-cells", jnp.asarray([1.0, 3.0])
    )
    space = phx.linalg.ArraySpace((2,), dtype=jnp.float64, space_id="flux-cells")
    field = phx.discretization.DiscreteFieldSpace(
        "surface-heat",
        "flux-surface",
        phx.discretization.EntityDofLayout("flux-cells", 2, 2),
        space,
        representation="cell_average",
    )
    measurement = cpl.CouplingMeasurement.from_measure(measure, space, _AREA)
    nodes = grid.nodes[:, None]
    # Cell rates 1 + 2s + 3s² and 2 + 6s² on the normalized window s ∈ [0, 1].
    rates = jnp.concatenate(
        (1.0 + 2.0 * nodes + 3.0 * nodes**2, 2.0 + 6.0 * nodes**2), axis=1
    )
    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (cpl.CouplingWaveform(grid, rates, space),), successful=True, status=0
        ),
        subsystem_id="flux",
        output_ports=(
            cpl.CouplingPort(
                "flux-output",
                "output",
                space,
                field_space=field,
                waveform_plan=plan,
                quantity=cpl.CouplingQuantity("surface_heat_flux", _FLUX),
                measurement=measurement,
                reference_scale=1.0,
            ),
        ),
        capabilities=_waveform_capabilities(),
    )
    store = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state + inputs[0], (), successful=True, status=0
        ),
        subsystem_id="store",
        input_ports=(
            cpl.CouplingPort(
                "heat-input",
                "input",
                space,
                field_space=field,
                quantity=cpl.CouplingQuantity("surface_heat", _HEAT),
                measurement=measurement,
                temporal_kind="interval_integral",
                reference_scale=1.0,
            ),
        ),
        capabilities=cpl.CouplingSubsystemCapabilities(
            jit=True, differentiable=True, deterministic_replay=True, fixed_topology=True
        ),
    )
    exchange = cpl.CouplingExchange(
        "heat",
        "flux-output",
        "heat-input",
        temporal=cpl.CouplingTemporalConversion(
            "integrate", integrated_quantity=integrated
        ),
    )
    zero = jnp.zeros((2,), dtype=jnp.float64)
    return cpl.prepare_coupling(
        cpl.CouplingGraph((source, store), (exchange,), time_unit=time_unit),
        (zero, zero),
        (zero,),
        policy=cpl.ExplicitCouplingPolicy(
            cpl.CouplingSweep("gauss-seidel", subsystem_order=("flux", "store"))
        ),
    )


def test_integrate_delivers_the_exact_window_amount_and_ledger() -> None:
    prepared = _integrated_flux(cpl.CouplingQuantity("surface_heat", _HEAT))

    result = eqx.filter_jit(cpl.advance_coupling_window)(
        prepared, prepared.reference_state, 2.0
    )

    # ∫₀¹ (1 + 2s + 3s²) ds = 3 and ∫₀¹ (2 + 6s²) ds = 4 over a 2 s window.
    amount = 2.0 * np.asarray([3.0, 4.0])
    area_integral = float(np.dot([1.0, 3.0], amount))
    state = result.accepted_state
    assert bool(result.successful)
    np.testing.assert_allclose(state.exchange_values[0], amount, rtol=1e-13)
    np.testing.assert_allclose(
        state.participant_states[state.subsystem_ids.index("store")], amount, rtol=1e-13
    )
    np.testing.assert_allclose(
        result.accepted_exchange_budget, [[-area_integral, area_integral]], rtol=1e-13
    )
    with pytest.raises(ValueError, match="rate dimension times time"):
        _integrated_flux(cpl.CouplingQuantity("surface_heat", _FLUX))


def test_integrate_is_exact_for_the_reconstruction_degree_not_the_metric_order() -> None:
    # A cubic reconstruction with a one-point residual metric: the booked amount
    # must still be the exact integral of the quadratic rates (3 and 4 per unit s).
    plan = cpl.CouplingWaveformPlan(
        4, 3, (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0), metric_order=1, plan_id="cubic-grid"
    )
    prepared = _integrated_flux(
        cpl.CouplingQuantity("surface_heat", _HEAT), phx.units.SECOND, plan
    )

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 2.0)

    amount = 2.0 * np.asarray([3.0, 4.0])
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.accepted_state.exchange_values[0], amount, rtol=1e-13
    )
    np.testing.assert_allclose(
        result.accepted_exchange_budget[0, 1], np.dot([1.0, 3.0], amount), rtol=1e-13
    )


def test_integrate_books_the_window_in_the_graph_clock_unit() -> None:
    # A 2 ms window of W/m² rates delivers 1e-3 times the 2 s window's J/m².
    prepared = _integrated_flux(
        cpl.CouplingQuantity("surface_heat", _HEAT), phx.units.MILLISECOND
    )

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 2.0)

    amount = 2.0e-3 * np.asarray([3.0, 4.0])
    area_integral = float(np.dot([1.0, 3.0], amount))
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.accepted_state.exchange_values[0], amount, rtol=1e-13
    )
    np.testing.assert_allclose(
        result.accepted_exchange_budget, [[-area_integral, area_integral]], rtol=1e-13
    )
    with pytest.raises(ValueError, match="declare the coupling clock time_unit"):
        _integrated_flux(cpl.CouplingQuantity("surface_heat", _HEAT), None)
    with pytest.raises(ValueError, match="unit of time"):
        _integrated_flux(cpl.CouplingQuantity("surface_heat", _HEAT), phx.units.METER)


def test_temporal_conversion_refuses_inconsistent_declarations() -> None:
    heat = cpl.CouplingQuantity("surface_heat", _HEAT)

    with pytest.raises(ValueError, match="exactly one temporal transfer"):
        cpl.CouplingTemporalConversion("interpolate")
    with pytest.raises(ValueError, match="integrated_quantity"):
        cpl.CouplingTemporalConversion("integrate")
    with pytest.raises(ValueError, match="carries no transfer"):
        cpl.CouplingTemporalConversion(
            "hold", transfer=cpl.BarycentricCouplingTemporalTransfer(1)
        )
    with pytest.raises(ValueError, match="carries no transfer"):
        cpl.CouplingTemporalConversion("sample-end", integrated_quantity=heat)
