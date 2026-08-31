#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


@dataclass(frozen=True)
class _Measured:
    value: Any
    first_ms: float
    steady_ms: float


def _measure(function: Callable[[], Any], repeats: int) -> _Measured:
    value, first_seconds = measure_synchronized(function)
    value, distribution = measure_repeated(
        function,
        warmup=0,
        repeats=repeats,
    )
    return _Measured(
        value,
        1_000.0 * first_seconds,
        1_000.0 * float(distribution.mean_seconds),
    )


def _ode(rate: float, *, problem_id: str):
    return phx.solver.DifferentialProblem(
        lambda time, state, parameter: -parameter * state,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=1.0,
        args=jnp.asarray(rate),
        problem_id=problem_id,
    )


def _dae(rate: float):
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, parameter: state_rate + parameter * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="benchmark:time-integrator:dae",
    )
    return phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray([1.0]),
        args=jnp.asarray(rate),
        problem_id="benchmark:time-integrator:dae-problem",
    )


def _record(name: str, measured: _Measured, terminal: Any, exact: float, successful: Any):
    terminal_value = float(jnp.asarray(terminal))
    return {
        "name": name,
        "successful": bool(jnp.asarray(successful)),
        "terminal": terminal_value,
        "absolute_error": abs(terminal_value - exact),
        "first_ms": measured.first_ms,
        "steady_ms": measured.steady_ms,
    }


def run_time_integrator_benchmarks(*, steps: int, repeats: int) -> dict[str, Any]:
    times = jnp.linspace(0.0, 1.0, steps + 1)
    grid = phx.dynamics.TimeGrid(times, time_id=f"benchmark:time:{steps}")
    exact = float(jnp.exp(-1.0))
    records = []

    explicit_problem = _ode(1.0, problem_id="benchmark:time:explicit")
    tsit = _measure(
        lambda: phx.solver.solve_diffrax(explicit_problem, save_times=times), repeats
    )
    records.append(
        _record(
            "diffrax-tsit5", tsit, tsit.value.states[-1, 0], exact, tsit.value.successful
        )
    )

    split_problem = phx.solver.SplitDifferentialProblem(
        lambda time, state, args: args[0] * state,
        lambda time, state, args: args[1] * state,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=1.0,
        args=jnp.asarray([1.0, -2.0]),
        problem_id="benchmark:time:imex",
    )
    imex = _measure(
        lambda: phx.solver.solve_diffrax(
            split_problem, save_times=times, dt0=1.0 / steps
        ),
        repeats,
    )
    records.append(
        _record(
            "diffrax-kencarp4",
            imex,
            imex.value.states[-1, 0],
            exact,
            imex.value.successful,
        )
    )

    dae_problem = _dae(1.0)
    for order in (2, 5):
        measured = _measure(
            lambda order=order: phx.solver.solve_dae(
                dae_problem,
                grid,
                policy=phx.solver.DAESolvePolicy(method=phx.solver.BDFMethod(order)),
            ),
            repeats,
        )
        records.append(
            _record(
                f"native-bdf{order}",
                measured,
                measured.value.states[-1, 0],
                exact,
                measured.value.successful,
            )
        )

    theta = _measure(
        lambda: phx.solver.solve_dae(
            dae_problem,
            grid,
            policy=phx.solver.DAESolvePolicy(
                method=phx.solver.ThetaMethod(0.5, endpoint=True)
            ),
        ),
        repeats,
    )
    records.append(
        _record(
            "native-crank-nicolson",
            theta,
            theta.value.states[-1, 0],
            exact,
            theta.value.successful,
        )
    )

    rosenbrock = _measure(
        lambda: phx.solver.solve_rosenbrock(explicit_problem, grid), repeats
    )
    records.append(
        _record(
            "native-ra34pw2",
            rosenbrock,
            rosenbrock.value.states[-1, 0],
            exact,
            rosenbrock.value.successful,
        )
    )

    irk = _measure(
        lambda: phx.solver.solve_implicit_runge_kutta(
            explicit_problem,
            grid,
            method=phx.solver.GaussLegendreIRK(2),
        ),
        repeats,
    )
    records.append(
        _record(
            "native-gauss4", irk, irk.value.states[-1, 0], exact, irk.value.successful
        )
    )

    second_order_system = phx.dynamics.SecondOrderDifferentialSystem(
        lambda time, configuration, velocity, acceleration, omega: (
            acceleration + omega**2 * configuration
        ),
        state_shape=(1,),
        system_id="benchmark:time:oscillator",
    )
    second_order_problem = phx.dynamics.SecondOrderDifferentialProblem(
        second_order_system,
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        initial_acceleration=jnp.asarray([-1.0]),
        args=jnp.asarray(1.0),
        problem_id="benchmark:time:oscillator-problem",
    )
    alpha = _measure(
        lambda: phx.solver.solve_generalized_alpha(second_order_problem, grid), repeats
    )
    records.append(
        _record(
            "native-generalized-alpha",
            alpha,
            alpha.value.configurations[-1, 0],
            float(jnp.cos(1.0)),
            alpha.value.successful,
        )
    )

    partition = phx.solver.StatePartition(
        {
            "slow": jnp.asarray([True, False]),
            "fast": jnp.asarray([False, True]),
        }
    )
    partitioned_problem = phx.solver.PartitionedDifferentialProblem(
        lambda time, state, args: jnp.asarray([-state[0], 0.0]),
        lambda time, state, args: jnp.asarray([0.0, -5.0 * state[1]]),
        jnp.ones((2,)),
        t0=0.0,
        t1=1.0,
        partition=partition,
        problem_id="benchmark:time:partitioned",
    )
    multirate = _measure(
        lambda: phx.solver.solve_multirate(
            partitioned_problem,
            grid,
            method=phx.solver.MultiratePartitionedRK(3, refinement_ratio=3),
        ),
        repeats,
    )
    records.append(
        {
            "name": "native-mprk3",
            "successful": bool(multirate.value.successful),
            "terminal": [float(value) for value in multirate.value.states[-1]],
            "absolute_error": float(
                jnp.max(
                    jnp.abs(
                        multirate.value.states[-1]
                        - jnp.asarray([jnp.exp(-1.0), jnp.exp(-5.0)])
                    )
                )
            ),
            "first_ms": multirate.first_ms,
            "steady_ms": multirate.steady_ms,
        }
    )

    return {
        "schema_version": "phydrax-time-integrator-benchmark-v1",
        "configuration": {"steps": steps, "repeats": repeats},
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
        },
        "cases": records,
        "passed": all(record["successful"] for record in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 1:
        raise ValueError("steps must be at least two and repeats must be positive.")
    report = run_time_integrator_benchmarks(steps=args.steps, repeats=args.repeats)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
