#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp

import phydrax as phx
from benchmarks._runtime import measure_lower_and_compile, measure_repeated
from phydrax._frozendict import frozendict


def _timed(operation, /, *, repeats: int) -> tuple[jax.Array, float, float]:
    jitted = jax.jit(operation)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(),
        lambda lowered: lowered.compile(),
    )
    value, distribution = measure_repeated(
        compiled,
        warmup=1,
        repeats=repeats,
    )
    compile_ms = 1_000.0 * (
        compilation.lowering_seconds + compilation.compilation_seconds
    )
    return value, compile_ms, 1_000.0 * float(distribution.mean_seconds)


def _time_point(value: float):
    return frozendict({"t": cx.Field(jnp.asarray(value), dims=())})


def _space_time_point(space: float, time_: float):
    return frozendict(
        {
            "x": cx.Field(jnp.asarray([space]), dims=(None,)),
            "t": cx.Field(jnp.asarray(time_), dims=()),
        }
    )


def _convolution_record(*, repeats: int) -> dict[str, Any]:
    domain = phx.domain.TimeInterval(0.0, 2.0)
    function = domain.Function("t")(lambda time_: jnp.sin(time_))
    convolution = phx.operators.time_convolution(
        lambda lag: jnp.exp(-lag),
        function,
        rule=phx.integration.GaussLegendreRule(64),
    )
    time_ = 1.234
    point = _time_point(time_)
    value, compile_ms, steady_ms = _timed(
        lambda: convolution(point).data,
        repeats=repeats,
    )
    reference = 0.5 * (math.sin(time_) - math.cos(time_) + math.exp(-time_))
    return {
        "case": "deterministic-time-convolution",
        "value": float(value),
        "reference": reference,
        "absolute_error": abs(float(value) - reference),
        "compile_ms": compile_ms,
        "steady_ms": steady_ms,
        "num_inner_evaluations": 64,
        "successful": abs(float(value) - reference) < 2e-10,
    }


def _integral_heat_record(*, repeats: int) -> dict[str, Any]:
    domain = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.TimeInterval(0.0, 1.0)
    solution = domain.Function("x", "t")(
        lambda x, time_: jnp.exp(-(jnp.pi**2) * time_) * jnp.sin(jnp.pi * x[0])
    )
    initial = domain.Function("x")(lambda x: jnp.sin(jnp.pi * x[0]))
    right_hand_side = phx.operators.laplacian(solution, var="x")
    history = phx.operators.time_convolution(
        lambda lag: jnp.ones_like(lag),
        right_hand_side,
        rule=phx.integration.GaussLegendreRule(64),
    )
    residual = solution - initial - history
    point = _space_time_point(0.5, 0.4)
    value, compile_ms, steady_ms = _timed(
        lambda: residual(point).data,
        repeats=repeats,
    )
    error = abs(float(value))
    return {
        "case": "integral-heat-residual-nonzero-initial",
        "value": float(value),
        "absolute_error": error,
        "compile_ms": compile_ms,
        "steady_ms": steady_ms,
        "num_inner_evaluations": 64,
        "successful": error < 2e-10,
    }


def _caputo_record(alpha: float, power: float, *, repeats: int) -> dict[str, Any]:
    start = 0.3
    endpoint = 1.1
    domain = phx.domain.TimeInterval(start, 1.5)
    function = domain.Function("t")(lambda time_: (time_ - start) ** power)
    derivative = phx.operators.caputo_time_fractional(
        function,
        alpha=alpha,
        mode="gj",
        order=128,
    )
    point = _time_point(endpoint)
    value, compile_ms, steady_ms = _timed(
        lambda: derivative(point).data,
        repeats=repeats,
    )
    reference = float(
        jsp.gamma(power + 1.0)
        / jsp.gamma(power + 1.0 - alpha)
        * (endpoint - start) ** (power - alpha)
    )
    error = abs(float(value) - reference)
    return {
        "case": f"caputo-power-alpha-{alpha}",
        "value": float(value),
        "reference": reference,
        "absolute_error": error,
        "compile_ms": compile_ms,
        "steady_ms": steady_ms,
        "num_inner_evaluations": 128,
        "successful": error < 3e-4,
    }


def _randomized_moment_record(*, seed: int) -> dict[str, Any]:
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    component = domain.component()
    parameter = 0.4
    function = domain.Function("x")(lambda x: parameter * x)
    condition = phx.conditions.Moment(
        "u",
        component,
        lambda value: value,
        target=0.5,
    )
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.MonteCarloPlan(2),
    )
    unbiased = phx.terms.RandomizedMomentPenalty(
        condition,
        source,
        num_realizations=2,
        loss_mode="u_statistic",
    )
    plug_in = phx.terms.RandomizedMomentPenalty(
        condition,
        source,
        num_realizations=2,
        loss_mode="plug_in",
    )
    keys = tuple(jr.split(jr.key(seed), 512))
    functions = {"u": function}
    batches = tuple(unbiased.sample(key=key) for key in keys)
    unbiased_values = jnp.stack(
        tuple(unbiased.loss(functions, batch=batch) for batch in batches)
    )
    plug_in_values = jnp.stack(
        tuple(plug_in.loss(functions, batch=batch) for batch in batches)
    )
    jax.block_until_ready((unbiased_values, plug_in_values))
    unbiased_mean = float(jnp.mean(unbiased_values))
    plug_in_mean = float(jnp.mean(plug_in_values))
    true_objective = (0.5 * parameter - 0.5) ** 2
    expected_plugin = true_objective + parameter**2 / 48.0
    return {
        "case": "randomized-moment-estimator-bias",
        "true_objective": true_objective,
        "u_statistic_mean": unbiased_mean,
        "plug_in_mean": plug_in_mean,
        "expected_plugin_mean": expected_plugin,
        "num_batches": len(keys),
        "inner_samples_per_estimate": 2,
        "num_estimates_per_batch": 2,
        "u_statistic_absolute_error": abs(unbiased_mean - true_objective),
        "plug_in_absolute_error": abs(plug_in_mean - expected_plugin),
        "successful": (
            abs(unbiased_mean - true_objective) < 8e-3
            and abs(plug_in_mean - expected_plugin) < 8e-3
            and plug_in_mean > unbiased_mean
        ),
    }


def run_integral_physics_benchmarks(
    *,
    repeats: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    if int(repeats) < 1:
        raise ValueError("repeats must be positive.")
    records = [
        _convolution_record(repeats=int(repeats)),
        _integral_heat_record(repeats=int(repeats)),
        _caputo_record(0.5, 2.0, repeats=int(repeats)),
        _caputo_record(1.5, 3.0, repeats=int(repeats)),
        _randomized_moment_record(seed=int(seed)),
    ]
    return {
        "benchmark": "integral-physics",
        "schema_version": 1,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "repeats": int(repeats),
        "seed": int(seed),
        "successful": all(record["successful"] for record in records),
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark integral and nonlocal physics-learning primitives."
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = run_integral_physics_benchmarks(
        repeats=arguments.repeats,
        seed=arguments.seed,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
