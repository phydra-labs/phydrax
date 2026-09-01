#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


cpl = phx.solver.coupling


def _capabilities(*, differentiable=True, waveform=False):
    return cpl.CouplingSubsystemCapabilities(
        jit=True,
        differentiable=differentiable,
        deterministic_replay=True,
        fixed_topology=True,
        supports_endpoint=not waveform,
        supports_waveform=waveform,
    )


def _linear_graph(*, fail_b=False, parameterized=False, count_state=False):
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="coupled-scalar")
    a_input = cpl.CouplingPort("a-input", "input", space, reference_scale=1.0)
    a_output = cpl.CouplingPort("a-output", "output", space, reference_scale=1.0)
    b_input = cpl.CouplingPort("b-input", "input", space, reference_scale=1.0)
    b_output = cpl.CouplingPort("b-output", "output", space, reference_scale=1.0)

    def advance_a(window, state, inputs, args):
        del window, args
        value = 0.5 * inputs[0]
        candidate = state + 1.0 if count_state else value
        return cpl.CouplingSubsystemResult(candidate, (value,), successful=True, status=0)

    def advance_b(window, state, inputs, args):
        forcing = args if parameterized else jnp.asarray(1.0, dtype=inputs[0].dtype)
        value = 0.5 * (inputs[0] + forcing)
        candidate = state + 1.0 if count_state else value
        successful = jnp.asarray(True)
        if fail_b:
            successful = window.index == 0
        return cpl.CouplingSubsystemResult(
            candidate,
            (value,),
            successful=successful,
            status=jnp.where(successful, 0, 17),
        )

    subsystem_a = cpl.CallableCouplingSubsystem(
        advance_a,
        subsystem_id="a",
        input_ports=(a_input,),
        output_ports=(a_output,),
        capabilities=_capabilities(),
    )
    subsystem_b = cpl.CallableCouplingSubsystem(
        advance_b,
        subsystem_id="b",
        input_ports=(b_input,),
        output_ports=(b_output,),
        capabilities=_capabilities(),
    )
    graph = cpl.CouplingGraph(
        (subsystem_b, subsystem_a),
        (
            cpl.CouplingExchange("a-to-b", "a-output", "b-input"),
            cpl.CouplingExchange("b-to-a", "b-output", "a-input"),
        ),
    )
    states = (jnp.zeros((1,), dtype=jnp.float64),) * 2
    values = (jnp.zeros((1,), dtype=jnp.float64),) * 2
    return graph, states, values


def _implicit_policy(*, maximum_steps=40, absolute=1e-10):
    return cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(
            acceleration=phx.nonlinear.AndersonAcceleration(history=4)
        ),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=absolute,
            relative_residual=0.0,
            maximum_steps=maximum_steps,
        ),
        (
            cpl.CouplingTolerance("a-input", absolute=1e-9),
            cpl.CouplingTolerance("b-input", absolute=1e-9),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )


def test_graph_identity_is_declaration_order_invariant_and_compiles_one_scc():
    graph, states, values = _linear_graph()
    reordered = cpl.CouplingGraph(
        tuple(reversed(graph.subsystems)),
        tuple(reversed(graph.exchanges)),
    )

    assert graph.graph_id == reordered.graph_id
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=_implicit_policy(),
    )
    assert prepared.report.subsystem_ids == ("a", "b")
    assert prepared.report.exchange_ids == ("a-to-b", "b-to-a")
    assert len(prepared.stages) == 1
    assert prepared.stages[0].cyclic
    assert prepared.report.resources.interface_size == 2


def test_graph_rejects_mismatched_direct_spaces_and_missing_driver():
    first = phx.linalg.ArraySpace((1,), space_id="first")
    second = phx.linalg.ArraySpace((1,), space_id="second")
    output = cpl.CouplingPort("output", "output", first, reference_scale=1.0)
    input_ = cpl.CouplingPort("input", "input", second, reference_scale=1.0)
    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (state,), successful=True, status=0
        ),
        subsystem_id="source",
        output_ports=(output,),
        capabilities=_capabilities(),
    )
    target = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (), successful=True, status=0
        ),
        subsystem_id="target",
        input_ports=(input_,),
        capabilities=_capabilities(),
    )
    graph = cpl.CouplingGraph(
        (source, target),
        (cpl.CouplingExchange("bad", "output", "input"),),
    )
    with pytest.raises(ValueError, match="vector-space identity"):
        cpl.prepare_coupling(
            graph,
            (jnp.zeros(1), jnp.zeros(1)),
            (jnp.zeros(1),),
            policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        )


def test_explicit_jacobi_completes_without_claiming_convergence():
    graph, states, values = _linear_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
    )

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert bool(result.successful)
    assert not bool(result.converged)
    assert float(result.accepted_state.time) == pytest.approx(1.0)
    assert jnp.allclose(result.accepted_state.exchange_values[0], 0.0)
    assert jnp.allclose(result.accepted_state.exchange_values[1], 0.5)
    assert jnp.allclose(
        result.diagnostics.exchange_residual_norms,
        jnp.asarray([0.0, 0.5]),
    )


def test_explicit_gauss_seidel_order_changes_the_single_sweep():
    graph, states, values = _linear_graph()
    policy = cpl.ExplicitCouplingPolicy(
        cpl.CouplingSweep("gauss-seidel", subsystem_order=("b", "a"))
    )
    prepared = cpl.prepare_coupling(graph, states, values, policy=policy)

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert bool(result.successful)
    assert jnp.allclose(result.accepted_state.exchange_values[0], 0.25)
    assert jnp.allclose(result.accepted_state.exchange_values[1], 0.5)


def test_implicit_anderson_certifies_the_physical_interface_root_under_jit():
    graph, states, values = _linear_graph()
    prepared = cpl.prepare_coupling(graph, states, values, policy=_implicit_policy())

    step = eqx.filter_jit(cpl.advance_coupling_window)
    result = step(prepared, prepared.reference_state, 1.0, None)

    assert bool(result.successful)
    assert bool(result.converged)
    assert jnp.allclose(
        result.accepted_state.exchange_values[0], jnp.asarray([1.0 / 3.0]), atol=1e-8
    )
    assert jnp.allclose(
        result.accepted_state.exchange_values[1], jnp.asarray([2.0 / 3.0]), atol=1e-8
    )
    assert jnp.all(result.diagnostics.exchange_certified)


def test_iteration_exhaustion_keeps_the_window_checkpoint():
    graph, states, values = _linear_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=_implicit_policy(maximum_steps=1, absolute=1e-30),
    )

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert not bool(result.successful)
    assert not bool(result.converged)
    assert int(result.status) == int(cpl.CouplingStatus.WORK_EXHAUSTED)
    assert float(result.accepted_state.time) == pytest.approx(0.0)
    assert jnp.allclose(result.accepted_state.exchange_values[0], 0.0)
    assert not jnp.allclose(result.candidate_state.exchange_values[1], 0.0)


def test_participant_failure_rolls_back_the_entire_window_and_rollout_stops():
    graph, states, values = _linear_graph(fail_b=True)
    problem = cpl.CouplingProblem(
        graph,
        states,
        values,
        cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        t0=0.0,
        t1=3.0,
        window_size=1.0,
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
    )

    solution = cpl.solve_coupling(
        problem,
        rollout=cpl.CouplingRolloutPlan(retention="trajectory"),
    )

    assert not bool(solution.successful)
    assert float(solution.final_state.time) == pytest.approx(1.0)
    assert solution.retained_valid.tolist() == [True, True, False, False]
    assert int(solution.statuses[1]) == int(cpl.CouplingStatus.PARTICIPANT_FAILURE)
    assert jnp.all(solution.participant_evaluations[2] == 0)


def test_final_physical_certification_can_reject_a_loose_nonlinear_success():
    graph, states, values = _linear_graph()
    policy = cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=0.4,
            relative_residual=0.0,
            maximum_steps=10,
        ),
        (
            cpl.CouplingTolerance("a-input", absolute=1e-12),
            cpl.CouplingTolerance("b-input", absolute=1e-12),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )
    prepared = cpl.prepare_coupling(graph, states, values, policy=policy)

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert not bool(result.successful)
    assert int(result.status) == int(cpl.CouplingStatus.CERTIFICATION_FAILURE)
    assert float(result.accepted_state.time) == pytest.approx(0.0)


def test_implicit_iterations_replay_the_window_checkpoint_instead_of_chaining_state():
    graph, states, values = _linear_graph(count_state=True)
    prepared = cpl.prepare_coupling(graph, states, values, policy=_implicit_policy())

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert bool(result.successful)
    assert all(
        jnp.allclose(state, 1.0) for state in result.accepted_state.participant_states
    )
    assert int(result.diagnostics.participant_evaluations[0]) > 1


def test_numeric_refresh_preserves_plan_identity_and_increments_version():
    graph, states, values = _linear_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
    )

    refreshed_graph, _, _ = _linear_graph(parameterized=True)
    refreshed = cpl.refresh_coupling(prepared, refreshed_graph, args=jnp.asarray(1.0))
    result = cpl.advance_coupling_window(
        refreshed,
        refreshed.reference_state,
        1.0,
        jnp.asarray(2.0),
    )

    assert refreshed.plan_id == prepared.plan_id
    assert int(refreshed.numeric_version) == 1
    assert jnp.allclose(result.accepted_state.exchange_values[1], 1.0)


def _field_space(name):
    topology = phx.discretization.TensorTopology(("x",), (3,))
    support = phx.discretization.DiscreteSupport(topology, 1, f"{name}-support")
    layout = phx.discretization.TensorDofLayout(("x",), (3,))
    vector_space = phx.linalg.ArraySpace((3,), space_id=f"{name}-vectors")
    return phx.discretization.DiscreteFieldSpace(
        name,
        support.support_id,
        layout,
        vector_space,
        representation="point_value",
    )


def test_forward_and_paired_adjoint_field_exchanges_preserve_virtual_work():
    source_space = _field_space("source")
    target_space = _field_space("target")
    matrix = jnp.asarray([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]])
    forward = phx.linalg.DenseLinearOperator(
        matrix,
        source=source_space.vector_space,
        target=target_space.vector_space,
    )
    adjoint = phx.linalg.DenseLinearOperator(
        matrix.T,
        source=target_space.vector_space,
        target=source_space.vector_space,
    )
    transfer = phx.discretization.FieldTransfer(
        source_space,
        target_space,
        forward,
        adjoint_operator=adjoint,
        properties=phx.discretization.TransferProperties(
            constant_preserving=True,
            adjoint_paired=True,
            exact_on=("constants",),
        ),
    )
    source_output = cpl.CouplingPort(
        "source-output",
        "output",
        source_space.vector_space,
        field_space=source_space,
        reference_scale=1.0,
    )
    target_input = cpl.CouplingPort(
        "target-input",
        "input",
        target_space.vector_space,
        field_space=target_space,
        reference_scale=1.0,
    )
    target_output = cpl.CouplingPort(
        "target-output",
        "output",
        target_space.vector_space,
        field_space=target_space,
        reference_scale=1.0,
    )
    source_input = cpl.CouplingPort(
        "source-input",
        "input",
        source_space.vector_space,
        field_space=source_space,
        reference_scale=1.0,
    )
    source_value = jnp.asarray([1.0, 2.0, 3.0])
    target_value = jnp.asarray([2.0, -1.0, 0.5])

    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (source_value,), successful=True, status=0
        ),
        subsystem_id="source",
        input_ports=(source_input,),
        output_ports=(source_output,),
        capabilities=_capabilities(),
    )
    target = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (target_value,), successful=True, status=0
        ),
        subsystem_id="target",
        input_ports=(target_input,),
        output_ports=(target_output,),
        capabilities=_capabilities(),
    )
    graph = cpl.CouplingGraph(
        (source, target),
        (
            cpl.CouplingExchange(
                "forward",
                "source-output",
                "target-input",
                transfer=transfer,
                requirement=cpl.CouplingTransferRequirement(constant_preserving=True),
            ),
            cpl.CouplingExchange(
                "adjoint",
                "target-output",
                "source-input",
                transfer=transfer,
                use_adjoint=True,
                requirement=cpl.CouplingTransferRequirement(adjoint_paired=True),
            ),
        ),
    )
    prepared = cpl.prepare_coupling(
        graph,
        (jnp.zeros(1), jnp.zeros(1)),
        (jnp.zeros(3), jnp.zeros(3)),
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
    )
    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    mapped_source = result.accepted_state.exchange_values[0]
    mapped_target = result.accepted_state.exchange_values[1]
    assert jnp.allclose(mapped_target, matrix @ source_value)
    assert jnp.allclose(mapped_source, matrix.T @ target_value)
    assert jnp.vdot(mapped_target, target_value) == pytest.approx(
        float(jnp.vdot(source_value, mapped_source))
    )
