#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic correctness and runtime benchmarks for continuous learned transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import Normal as FlowJAXNormal

import phydrax as phx
from benchmarks._runtime import (
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


class _LinearField(eqx.Module):
    matrix: jnp.ndarray
    shift: jnp.ndarray

    def __init__(self, matrix, shift=None):
        self.matrix = jnp.asarray(matrix, dtype=float)
        self.shift = (
            jnp.zeros((self.matrix.shape[0],))
            if shift is None
            else jnp.asarray(shift, dtype=float)
        )

    def __call__(self, time, state, args):
        del time, args
        return self.matrix @ state + self.shift


class _EndpointVelocity(eqx.Module):
    def __call__(self, state, time):
        del time
        return state


def _timings(function, arguments, *, repetitions: int):
    jitted = eqx.filter_jit(function)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, first_execution_seconds = measure_synchronized(lambda: compiled(*arguments))
    _, steady = measure_repeated(
        lambda: compiled(*arguments),
        warmup=0,
        repeats=int(repetitions),
    )
    return result, {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_execution_seconds": first_execution_seconds,
        "steady": steady.to_seconds_dict(),
    }


def _normal(dimension: int, *, location=None, covariance=None):
    family = phx.uq.MultivariateNormalFamily(dimension)
    resolved_location = (
        jnp.zeros((dimension,)) if location is None else jnp.asarray(location)
    )
    resolved_covariance = (
        jnp.eye(dimension) if covariance is None else jnp.asarray(covariance)
    )
    return family.law_from_location_covariance(resolved_location, resolved_covariance)


def _flow(matrix, *, shift=None):
    matrix = jnp.asarray(matrix, dtype=float)
    dimension = int(matrix.shape[0])
    system = phx.dynamics.ContinuousSystem(
        _LinearField(matrix, shift),
        state_layout=phx.dynamics.StateLayout((dimension,)),
        system_id=f"benchmark-linear-flow-{dimension}",
    )
    evolution = phx.solver.DiffraxEvolution(
        system,
        rtol=2e-8,
        atol=2e-10,
        max_steps=4096,
    )
    transport = phx.transport.ContinuousTransport(_normal(dimension), evolution)
    return phx.transport.ContinuousFlowLaw(
        transport, max_exact_dimension=max(32, dimension)
    )


def benchmark_linear_case(
    *, dimension: int, sample_count: int, repetitions: int, seed: int
) -> dict[str, Any]:
    rates = jnp.linspace(-0.2, 0.3, dimension)
    flow = _flow(jnp.diag(rates))
    key = jr.key(seed)
    values = jr.normal(key, (sample_count, dimension))
    target = _normal(dimension, covariance=jnp.diag(jnp.exp(2.0 * rates)))

    density_result, density_timing = _timings(
        lambda current, points: current.log_prob_with_diagnostics(points),
        (flow, values),
        repetitions=repetitions,
    )
    sample_result, sample_timing = _timings(
        lambda current, current_key: current.sample_with_diagnostics(
            current_key, (sample_count,)
        ),
        (flow, key),
        repetitions=repetitions,
    )
    exact_target = target.log_prob(values)
    expected_inverse_volume = -jnp.sum(rates)
    stochastic = phx.transport.estimate_continuous_flow_log_prob(
        flow.transport,
        values[0],
        jr.fold_in(key, 1),
        policy=phx.operators.StochasticTracePolicy(32),
    )
    return {
        "case": "analytic-diagonal-linear",
        "dimension": dimension,
        "sample_count": sample_count,
        "density_timing": density_timing,
        "sample_timing": sample_timing,
        "maximum_log_density_error": float(
            jnp.max(jnp.abs(density_result.log_prob - exact_target))
        ),
        "maximum_log_volume_error": float(
            jnp.max(jnp.abs(density_result.log_volume - expected_inverse_volume))
        ),
        "all_density_solves_valid": bool(density_result.successful),
        "all_sample_solves_valid": bool(sample_result.successful),
        "median_accepted_steps": float(jnp.median(density_result.accepted_steps)),
        "maximum_rejected_steps": int(jnp.max(density_result.rejected_steps)),
        "stochastic_log_density_error": float(
            jnp.abs(stochastic.log_prob - flow.log_prob(values[0]))
        ),
        "stochastic_standard_error": float(stochastic.standard_error),
    }


def benchmark_translation_case(
    *, dimension: int, sample_count: int, repetitions: int, seed: int
) -> dict[str, Any]:
    offset = jnp.linspace(-1.0, 1.0, dimension)
    flow = _flow(jnp.zeros((dimension, dimension)), shift=offset)
    key = jr.key(seed)
    transported, transport_timing = _timings(
        lambda current, current_key: current.sample_with_diagnostics(
            current_key, (sample_count,)
        ),
        (flow, key),
        repetitions=repetitions,
    )
    flowjax = FlowJAXNormal(offset, jnp.ones((dimension,)))
    baseline, baseline_timing = _timings(
        lambda distribution, current_key: distribution.sample(
            current_key, sample_shape=(sample_count,)
        ),
        (flowjax, key),
        repetitions=repetitions,
    )
    expected = transported.source_states + offset
    transported_mean = jnp.mean(transported.final_states, axis=0)
    baseline_mean = jnp.mean(baseline, axis=0)
    return {
        "case": "translation-sampling",
        "dimension": dimension,
        "sample_count": sample_count,
        "continuous_transport_timing": transport_timing,
        "flowjax_analytic_baseline_timing": baseline_timing,
        "maximum_endpoint_error": float(
            jnp.max(jnp.abs(transported.final_states - expected))
        ),
        "continuous_sample_mean_error": float(jnp.linalg.norm(transported_mean - offset)),
        "flowjax_sample_mean_error": float(jnp.linalg.norm(baseline_mean - offset)),
        "all_solves_valid": bool(transported.successful),
    }


def benchmark_endpoint_physics_case(*, repetitions: int) -> dict[str, Any]:
    state_domain = phx.domain.HyperRectangle(
        jnp.asarray((-10.0,)),
        jnp.asarray((10.0,)),
        label="x",
    )
    domain = state_domain @ phx.domain.TimeInterval(0.0, 1.0)
    velocity = domain.Function("x", "t")(_EndpointVelocity())
    endpoints = phx.transport.EndpointCouplingSample(
        source=jnp.ones((1, 1)),
        target=jnp.full((1, 1), jnp.e),
        source_indices=jnp.zeros((1,), dtype=jnp.int32),
        target_indices=jnp.zeros((1,), dtype=jnp.int32),
        valid=jnp.ones((1,), dtype=bool),
        log_weights=jnp.zeros((1,)),
        context={"desired": jnp.full((1, 1), jnp.e)},
        coupling_id="benchmark-exponential-endpoint",
        provenance="analytic",
    )
    interpolant = phx.transport.LinearEndpointInterpolant((1,))
    functional = phx.terms.CallableFlowEndpointFunctional(
        lambda value, context: jnp.sum(jnp.square(value - context["desired"])),
        event_shape=(1,),
        functional_id="analytic-exponential-endpoint-error",
    )
    batch = phx.terms.FlowMatchingBatch(
        state=jnp.ones((1, 1)),
        time=jnp.zeros((1,)),
        target_velocity=jnp.zeros((1, 1)),
        valid=jnp.ones((1,), dtype=bool),
        log_weights=jnp.zeros((1,)),
        context={"desired": jnp.full((1, 1), jnp.e)},
        evaluation_key=jr.key(91),
        source_indices=jnp.zeros((1,), dtype=jnp.int32),
        target_indices=jnp.zeros((1,), dtype=jnp.int32),
        interpolant_id=interpolant.interpolant_id,
        coupling_id=endpoints.coupling_id,
        policy_id="analytic-time-zero",
        batch_id="analytic-exponential-endpoint",
    )

    def term(steps):
        return phx.terms.PhysicsFlowMatchingTerm(
            "velocity",
            endpoints,
            interpolant,
            functional,
            phx.terms.FlowEndpointRolloutPolicy(steps, rematerialize=steps > 1),
        )

    def physics_objective(current, function, materialized):
        return current.objective_components(
            {"velocity": function},
            batch=materialized,
        )[1]

    one_value, one_timing = _timings(
        physics_objective,
        (term(1), velocity, batch),
        repetitions=repetitions,
    )
    four_value, four_timing = _timings(
        physics_objective,
        (term(4), velocity, batch),
        repetitions=repetitions,
    )
    return {
        "case": "terminal-physics-euler-refinement",
        "one_step_endpoint_energy": float(one_value),
        "four_step_endpoint_energy": float(four_value),
        "one_step_timing": one_timing,
        "four_step_timing": four_timing,
        "refinement_reduced_error": bool(four_value < one_value),
    }


def run_continuous_transport_benchmarks(*, quick: bool = False) -> dict[str, Any]:
    dimensions = (2,) if quick else (2, 8, 16)
    sample_count = 8 if quick else 128
    repetitions = 2 if quick else 5
    cases = []
    for index, dimension in enumerate(dimensions):
        cases.append(
            benchmark_linear_case(
                dimension=dimension,
                sample_count=sample_count,
                repetitions=repetitions,
                seed=20260822 + index,
            )
        )
        cases.append(
            benchmark_translation_case(
                dimension=dimension,
                sample_count=sample_count,
                repetitions=repetitions,
                seed=20260852 + index,
            )
        )
    cases.append(benchmark_endpoint_physics_case(repetitions=repetitions))
    return {
        "schema": "phydrax-continuous-transport-benchmark-v1",
        "backend": jax.default_backend(),
        "quick": bool(quick),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    report = run_continuous_transport_benchmarks(quick=arguments.quick)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n")
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
