#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable fixed-window partitioning of two coupled scalar oscillators."""

import jax
import jax.numpy as jnp

import phydrax as phx


cpl = phx.solver.coupling


space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="oscillator-interface")
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


def advance_a(window, state, inputs, forcing):
    del forcing
    candidate = state + window.size * (-state + 0.5 * inputs[0])
    return cpl.CouplingSubsystemResult(
        candidate, (candidate,), successful=True, status=0, work=1
    )


def advance_b(window, state, inputs, forcing):
    candidate = state + window.size * (-state + 0.5 * inputs[0] + forcing)
    return cpl.CouplingSubsystemResult(
        candidate, (candidate,), successful=True, status=0, work=1
    )


a = cpl.CallableCouplingSubsystem(
    advance_a,
    subsystem_id="oscillator-a",
    input_ports=(a_input,),
    output_ports=(a_output,),
    capabilities=capabilities,
)
b = cpl.CallableCouplingSubsystem(
    advance_b,
    subsystem_id="oscillator-b",
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
    (jnp.zeros(1), jnp.zeros(1)),
    (jnp.zeros(1), jnp.zeros(1)),
    policy=policy,
    differentiation=cpl.CouplingDifferentiationPolicy("implicit"),
    args=jnp.asarray(1.0, dtype=jnp.float64),
    problem_id="coupled-oscillators",
)


def final_a(forcing):
    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0, forcing)
    return result.accepted_state.participant_states[0][0]


value, derivative = jax.value_and_grad(final_a)(jnp.asarray(1.0, dtype=jnp.float64))
result = cpl.advance_coupling_window(
    prepared,
    prepared.reference_state,
    1.0,
    jnp.asarray(1.0, dtype=jnp.float64),
)
print("successful:", bool(result.successful))
print("converged:", bool(result.converged))
print("interface residuals:", result.diagnostics.exchange_residual_norms)
print("oscillator A:", value)
print("d oscillator A / d forcing:", derivative)
