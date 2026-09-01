#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


cpl = phx.solver.coupling


def _parameterized_graph():
    space = phx.linalg.ArraySpace(
        (1,), dtype=jnp.float64, space_id="differentiable-interface"
    )
    a_input = cpl.CouplingPort("a-input", "input", space, reference_scale=1.0)
    a_output = cpl.CouplingPort("a-output", "output", space, reference_scale=1.0)
    b_input = cpl.CouplingPort("b-input", "input", space, reference_scale=1.0)
    b_output = cpl.CouplingPort("b-output", "output", space, reference_scale=1.0)
    capabilities = cpl.CouplingSubsystemCapabilities(
        jit=True,
        differentiable=True,
        deterministic_replay=True,
        fixed_topology=True,
    )

    def advance_a(window, state, inputs, parameter):
        del window, state, parameter
        value = 0.5 * inputs[0]
        return cpl.CouplingSubsystemResult(value, (value,), successful=True, status=0)

    def advance_b(window, state, inputs, parameter):
        del window, state
        value = 0.5 * (inputs[0] + parameter)
        return cpl.CouplingSubsystemResult(value, (value,), successful=True, status=0)

    a = cpl.CallableCouplingSubsystem(
        advance_a,
        subsystem_id="a",
        input_ports=(a_input,),
        output_ports=(a_output,),
        capabilities=capabilities,
    )
    b = cpl.CallableCouplingSubsystem(
        advance_b,
        subsystem_id="b",
        input_ports=(b_input,),
        output_ports=(b_output,),
        capabilities=capabilities,
    )
    graph = cpl.CouplingGraph(
        (a, b),
        (
            cpl.CouplingExchange("a-to-b", "a-output", "b-input"),
            cpl.CouplingExchange("b-to-a", "b-output", "a-input"),
        ),
    )
    states = (jnp.zeros(1, dtype=jnp.float64),) * 2
    values = (jnp.zeros(1, dtype=jnp.float64),) * 2
    return graph, states, values


def _root_policy():
    return cpl.ImplicitCouplingPolicy(
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


def test_explicit_algorithmic_derivative_matches_single_sweep_map():
    graph, states, values = _parameterized_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
        args=jnp.asarray(1.0, dtype=jnp.float64),
    )

    def observable(parameter):
        result = cpl.advance_coupling_window(
            prepared, prepared.reference_state, 1.0, parameter
        )
        return result.accepted_state.exchange_values[1][0]

    assert float(jax.grad(observable)(jnp.asarray(1.0))) == pytest.approx(0.5)


def test_explicit_coupling_vectorizes_over_runtime_parameters():
    graph, states, values = _parameterized_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
        args=jnp.asarray(1.0, dtype=jnp.float64),
    )

    values = jax.vmap(
        lambda parameter: cpl.advance_coupling_window(
            prepared, prepared.reference_state, 1.0, parameter
        ).accepted_state.exchange_values[1][0]
    )(jnp.asarray([1.0, 2.0], dtype=jnp.float64))

    assert jnp.allclose(values, jnp.asarray([0.5, 1.0]))


def test_none_differentiation_policy_stops_returned_state_gradients():
    graph, states, values = _parameterized_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        differentiation=cpl.CouplingDifferentiationPolicy("none"),
        args=jnp.asarray(1.0, dtype=jnp.float64),
    )

    derivative = jax.grad(
        lambda parameter: cpl.advance_coupling_window(
            prepared, prepared.reference_state, 1.0, parameter
        ).accepted_state.exchange_values[1][0]
    )(jnp.asarray(1.0))

    assert float(derivative) == pytest.approx(0.0)


def test_implicit_root_derivative_matches_analytic_coupled_solution():
    graph, states, values = _parameterized_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=_root_policy(),
        differentiation=cpl.CouplingDifferentiationPolicy("implicit"),
        args=jnp.asarray(1.0, dtype=jnp.float64),
    )

    def observable(parameter):
        result = cpl.advance_coupling_window(
            prepared, prepared.reference_state, 1.0, parameter
        )
        return result.accepted_state.exchange_values[0][0]

    value, tangent = jax.jvp(
        observable,
        (jnp.asarray(1.0, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )
    reverse = jax.grad(observable)(jnp.asarray(1.0, dtype=jnp.float64))

    assert float(value) == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert float(tangent) == pytest.approx(1.0 / 3.0, abs=1e-8)
    assert float(reverse) == pytest.approx(float(tangent), abs=1e-9)


def test_implicit_root_derivative_composes_across_checkpointed_rollout():
    graph, states, values = _parameterized_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=_root_policy(),
        differentiation=cpl.CouplingDifferentiationPolicy("implicit"),
        args=jnp.asarray(1.0, dtype=jnp.float64),
    )
    rollout = cpl.CouplingRolloutPlan(
        retention="final",
        replay=phx.solver.FixedStepReplayPolicy("step"),
    )

    def observable(parameter):
        solution = rollout.rollout(
            prepared,
            window_count=2,
            window_size=1.0,
            args=parameter,
        )
        return solution.final_state.exchange_values[0][0]

    derivative = jax.grad(observable)(jnp.asarray(1.0, dtype=jnp.float64))

    assert float(derivative) == pytest.approx(1.0 / 3.0, abs=1e-8)


def test_implicit_differentiation_rejects_fixed_point_anderson():
    graph, states, values = _parameterized_graph()
    policy = cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(),
        phx.nonlinear.NonlinearTermination(maximum_steps=10),
        (
            cpl.CouplingTolerance("a-input", absolute=1e-8),
            cpl.CouplingTolerance("b-input", absolute=1e-8),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )

    with pytest.raises(ValueError, match="general-root"):
        cpl.prepare_coupling(
            graph,
            states,
            values,
            policy=policy,
            differentiation=cpl.CouplingDifferentiationPolicy("implicit"),
            args=jnp.asarray(1.0, dtype=jnp.float64),
        )
