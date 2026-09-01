#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


cpl = phx.solver.coupling


def _graph(alpha: float, *, parameterized: bool = False):
    space = phx.linalg.ArraySpace(
        (1,), dtype=jnp.float64, space_id=f"qualification-interface-{alpha}"
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

    def advance_a(window, state, inputs, forcing):
        del window, state, forcing
        value = alpha * inputs[0]
        return cpl.CouplingSubsystemResult(
            value, (value,), successful=True, status=0, work=1
        )

    def advance_b(window, state, inputs, forcing):
        del window, state
        amplitude = forcing if parameterized else jnp.asarray(1.0)
        value = alpha * inputs[0] + (1.0 - alpha) * amplitude
        return cpl.CouplingSubsystemResult(
            value, (value,), successful=True, status=0, work=1
        )

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
    return (
        cpl.CouplingGraph(
            (a, b),
            (
                cpl.CouplingExchange("a-to-b", "a-output", "b-input"),
                cpl.CouplingExchange("b-to-a", "b-output", "a-input"),
            ),
        ),
        (jnp.zeros(1, dtype=jnp.float64),) * 2,
        (jnp.zeros(1, dtype=jnp.float64),) * 2,
    )


def _fixed_point_policy(accelerated: bool):
    acceleration = phx.nonlinear.AndersonAcceleration(history=5) if accelerated else None
    return cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(acceleration=acceleration),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-9,
            relative_residual=0.0,
            maximum_steps=500,
        ),
        (
            cpl.CouplingTolerance("a-input", absolute=1e-8),
            cpl.CouplingTolerance("b-input", absolute=1e-8),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )


def _run_fixed_point(alpha: float, *, accelerated: bool):
    graph, states, values = _graph(alpha)
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=_fixed_point_policy(accelerated),
        problem_id=("qualification-anderson" if accelerated else "qualification-picard"),
    )
    step = eqx.filter_jit(cpl.advance_coupling_window)
    warm = step(prepared, prepared.reference_state, 1.0, None)
    jax.block_until_ready(warm.accepted_state.exchange_values[0])
    start = perf_counter()
    result = step(prepared, prepared.reference_state, 1.0, None)
    jax.block_until_ready(result.accepted_state.exchange_values[0])
    elapsed = perf_counter() - start
    return result, elapsed


def _explicit_defect(alpha: float) -> float:
    graph, states, values = _graph(alpha)
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
        problem_id="qualification-explicit",
    )
    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)
    return float(jnp.max(result.diagnostics.exchange_residual_norms))


def _derivative_error(alpha: float) -> float:
    graph, states, values = _graph(alpha, parameterized=True)
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
        problem_id="qualification-derivative",
    )

    def observable(forcing):
        result = cpl.advance_coupling_window(
            prepared, prepared.reference_state, 1.0, forcing
        )
        return result.accepted_state.exchange_values[0][0]

    derivative = jax.grad(observable)(jnp.asarray(1.0, dtype=jnp.float64))
    exact = alpha / (1.0 + alpha)
    return abs(float(derivative) - exact)


def run_qualification() -> dict[str, object]:
    alpha = 0.9
    picard, picard_seconds = _run_fixed_point(alpha, accelerated=False)
    anderson, anderson_seconds = _run_fixed_point(alpha, accelerated=True)
    explicit_defect = _explicit_defect(alpha)
    derivative_error = _derivative_error(alpha)
    exact_a = alpha / (1.0 + alpha)
    solution_error = abs(float(anderson.accepted_state.exchange_values[0][0]) - exact_a)
    final_residual = float(jnp.max(anderson.diagnostics.exchange_residual_norms))
    finite = all(
        jnp.isfinite(jnp.asarray(value))
        for value in (
            explicit_defect,
            derivative_error,
            solution_error,
            final_residual,
            picard_seconds,
            anderson_seconds,
        )
    )
    passed = (
        bool(picard.successful)
        and bool(anderson.successful)
        and solution_error < 1e-9
        and final_residual < 1e-9
        and derivative_error < 1e-8
        and int(anderson.diagnostics.participant_evaluations[0])
        <= int(picard.diagnostics.participant_evaluations[0])
        and bool(finite)
    )
    return {
        "coupling_coefficient": alpha,
        "explicit_interface_defect": explicit_defect,
        "picard_iterations": int(picard.diagnostics.coupling_iterations),
        "anderson_iterations": int(anderson.diagnostics.coupling_iterations),
        "picard_participant_evaluations": int(
            picard.diagnostics.participant_evaluations[0]
        ),
        "anderson_participant_evaluations": int(
            anderson.diagnostics.participant_evaluations[0]
        ),
        "picard_execution_seconds": picard_seconds,
        "anderson_execution_seconds": anderson_seconds,
        "solution_error": solution_error,
        "final_interface_residual": final_residual,
        "implicit_derivative_error": derivative_error,
        "finite": bool(finite),
        "passed": bool(passed),
    }


def main() -> None:
    evidence = run_qualification()
    destination = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "partitioned_coupling.json"
    )
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not evidence["passed"]:
        raise SystemExit("Partitioned coupling qualification failed.")


if __name__ == "__main__":
    main()
