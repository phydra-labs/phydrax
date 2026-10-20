#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


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
        # ty: ignore[invalid-argument-type]
        1_000.0 * float(distribution.mean_seconds),
    )


def _ode(rate: float, *, problem_id: str) -> Any:
    return phx.solver.DifferentialProblem(
        lambda time, state, parameter: -parameter * state,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=1.0,
        args=jnp.asarray(rate),
        problem_id=problem_id,
    )


def _dae(rate: float) -> Any:
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


def _record(
    name: str, measured: _Measured, terminal: Any, exact: float, successful: Any
) -> Any:
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


def _fixed_decay(time: jax.Array, state: jax.Array, rate: jax.Array) -> jax.Array:
    del time
    return -rate * state


def _fixed_case(
    *, steps: int, state_size: int, max_steps: int, dtype: str, repeats: int
) -> dict[str, Any]:
    """Time one bounded public-API solve and its reverse parameter derivative."""
    host_dtype = np.dtype(dtype)

    def prepare() -> tuple[np.ndarray, ...]:
        return (
            np.linspace(0.5, 1.5, state_size, dtype=host_dtype),
            np.full((state_size,), 1.25, dtype=host_dtype),
            np.asarray([0.5, 1.0], dtype=host_dtype),
            np.asarray([0.0, 1.0], dtype=host_dtype),
            np.asarray(1.0 / steps, dtype=host_dtype),
        )

    host_inputs, preparation_seconds = measure_host(prepare)
    # Interval, save schedule, and step count are fixed benchmark configuration,
    # exactly as in qualification; state and rates remain dynamic numerical inputs.
    runtime_inputs = host_inputs[:2]
    inputs, transfer_seconds = measure_synchronized(
        lambda: tuple(jax.device_put(value) for value in runtime_inputs)
    )
    precision, policy_preparation_seconds = measure_host(
        lambda: phx.solver.TemporalPrecisionPolicy(
            coefficient_dtype=dtype,
            state_dtype=dtype,
            stage_dtype=dtype,
            accumulation_dtype=dtype,
            residual_dtype=dtype,
            decision_dtype=dtype,
            checkpoint_dtype=dtype,
            output_dtype=dtype,
        )
    )

    def run(
        initial: jax.Array,
        rate: jax.Array,
    ) -> dict[str, jax.Array]:
        def objective(
            parameter: jax.Array,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            problem = phx.solver.DifferentialProblem(
                _fixed_decay,
                initial,
                t0=0.0,
                t1=1.0,
                args=parameter,
                problem_id="benchmark:time:heun-fixed",
            )
            solution = phx.solver.solve_diffrax(
                problem,
                save_times=jnp.asarray([0.5, 1.0], dtype=dtype),
                solver=dfx.Heun(),
                stepsize_controller=dfx.ConstantStepSize(),
                dt0=1.0 / steps,
                max_steps=max_steps,
                precision=precision,
            )
            evidence = solution.temporal_evidence
            if evidence is None:
                raise RuntimeError("The fixed recurrence must retain temporal evidence.")
            result = {
                "states": solution.states,
                "times": solution.times,
                "terminal_time": solution.terminal_time,
                "successful": solution.successful,
                "backend_successful": solution.backend_successful,
                "terminal_successful": solution.terminal_successful,
                "event_terminated": solution.event_terminated,
                "valid": solution.valid,
                "adaptive": jnp.asarray(evidence.adaptive, dtype=jnp.bool_),
                "num_steps": solution.stats["num_steps"],
                "num_accepted_steps": solution.stats["num_accepted_steps"],
                "num_rejected_steps": solution.stats["num_rejected_steps"],
            }
            # Independent rates make each component of this reverse gradient
            # the derivative of its own terminal state, without a dense Jacobian.
            return jnp.sum(solution.states[-1]), result

        (_, result), derivative = jax.value_and_grad(objective, has_aux=True)(rate)
        return {**result, "parameter_derivative": derivative}

    entry = jax.jit(run)
    compiled, timing = measure_lower_and_compile(
        lambda: entry.lower(*inputs), lambda lowered: lowered.compile()
    )
    memory = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-heun-fixed-value-and-reverse-derivative",
        unavailable_reason="The selected compiler did not expose all cost/memory fields.",
    )
    first, dispatch_seconds = measure_host(lambda: compiled(*inputs))
    _, synchronization_seconds = measure_synchronized(lambda: first)
    result, warm = measure_repeated(lambda: compiled(*inputs), warmup=0, repeats=repeats)
    host, readback_seconds = measure_synchronized(lambda: jax.device_get(result))
    checks = _fixed_checks(host, host_inputs, steps=steps, dtype=dtype)
    return {
        "route": "heun-fixed",
        "dtype": dtype,
        "steps": steps,
        "state_size": state_size,
        "max_steps": max_steps,
        "interval": [0.0, 1.0],
        "schedule_binding": "fixed interval, save times, and step size",
        "derivative": "reverse gradient of sum of terminal independent states",
        "devices": sorted(
            {
                f"{device.platform}:{device.id}:{device.device_kind}"
                for value in result.values()
                for device in value.devices()
            }
        ),
        "status": {
            name: np.asarray(host[name]).tolist()
            for name in (
                "successful",
                "backend_successful",
                "terminal_successful",
                "event_terminated",
                "adaptive",
                "num_steps",
                "num_accepted_steps",
                "num_rejected_steps",
            )
        },
        "checks": checks,
        "passed": all(checks["gates"].values()),
        "timing": {
            "host_input_preparation_seconds": preparation_seconds,
            "host_policy_preparation_seconds": policy_preparation_seconds,
            "input_transfer_and_sync_seconds": transfer_seconds,
            "lowering_seconds": timing.lowering_seconds,
            "compilation_seconds": timing.compilation_seconds,
            "first_dispatch_seconds": dispatch_seconds,
            "first_completion_wait_seconds": synchronization_seconds,
            "first_execution_seconds": dispatch_seconds + synchronization_seconds,
            "warm_synchronized": warm.to_seconds_dict(),
            "host_readback_seconds": readback_seconds,
        },
        "memory": {
            "compiler": asdict(memory),
            "logical_input_bytes": logical_array_bytes(inputs),
            "logical_output_bytes": logical_array_bytes(result),
            "scope": "logical payloads and compiler estimates; not allocator/RSS totals",
        },
    }


def _fixed_checks(
    result: dict[str, Any],
    inputs: tuple[np.ndarray, ...],
    *,
    steps: int,
    dtype: str,
) -> dict[str, Any]:
    """Compare rounded inputs with independent host FP64 recurrence and ODE."""
    initial, rate, times, _, step = (
        np.asarray(value, dtype=np.float64) for value in inputs
    )
    z = rate * step
    amplification = 1.0 - z + 0.5 * z * z
    recurrence = np.stack(
        [initial * amplification ** (steps // 2), initial * amplification**steps]
    )
    derivative = initial * steps * amplification ** (steps - 1) * (z - 1.0) * step
    analytical = initial[None, :] * np.exp(-times[:, None] * rate[None, :])
    analytical_derivative = -initial * np.exp(-rate)
    states = np.asarray(result["states"], dtype=np.float64)
    gradient = np.asarray(result["parameter_derivative"], dtype=np.float64)
    # Same rounding bounds as test_metal_models' qualified recurrence. The
    # analytical ODE check adds the independently computed Heun truncation error,
    # rather than disguising O(h^2) discretization as backend roundoff.
    value_rtol = 8.0 * (steps + 1) * np.finfo(np.dtype(dtype)).eps
    derivative_rtol = 16.0 * (steps + 1) * np.finfo(np.dtype(dtype)).eps
    state_error = np.abs(states - recurrence)
    derivative_error = np.abs(gradient - derivative)
    analytical_error = np.abs(states - analytical)
    analytical_gradient_error = np.abs(gradient - analytical_derivative)
    gates = {
        "successful": bool(np.all(result["successful"])),
        "backend_successful": bool(np.all(result["backend_successful"])),
        "terminal_successful": bool(np.all(result["terminal_successful"])),
        "valid": bool(np.all(result["valid"])),
        "no_event": not bool(np.any(result["event_terminated"])),
        "fixed_controller": not bool(result["adaptive"]),
        "exact_work": (
            int(result["num_steps"]) == steps
            and int(result["num_accepted_steps"]) == steps
            and int(result["num_rejected_steps"]) == 0
        ),
        "terminal_time": bool(np.asarray(result["terminal_time"]) == 1.0),
        "save_times": bool(np.array_equal(result["times"], inputs[2])),
        "dtype": all(
            np.asarray(result[name]).dtype == np.dtype(dtype)
            for name in ("states", "times", "terminal_time", "parameter_derivative")
        ),
        "recurrence_value": bool(np.all(state_error <= value_rtol * np.abs(recurrence))),
        "recurrence_derivative": bool(
            np.all(derivative_error <= derivative_rtol * np.abs(derivative))
        ),
        "analytical_value": bool(
            np.all(
                analytical_error
                <= np.abs(recurrence - analytical) + value_rtol * np.abs(recurrence)
            )
        ),
        "analytical_derivative": bool(
            np.all(
                analytical_gradient_error
                <= np.abs(derivative - analytical_derivative)
                + derivative_rtol * np.abs(derivative)
            )
        ),
    }
    return {
        "gates": gates,
        "terminal_state_endpoints": [float(states[-1, 0]), float(states[-1, -1])],
        "parameter_derivative_endpoints": [float(gradient[0]), float(gradient[-1])],
        "value_relative_roundoff_bound": float(value_rtol),
        "derivative_relative_roundoff_bound": float(derivative_rtol),
        "recurrence_value_max_absolute_error": float(np.max(state_error)),
        "recurrence_derivative_max_absolute_error": float(np.max(derivative_error)),
        "analytical_value_max_absolute_error": float(np.max(analytical_error)),
        "analytical_derivative_max_absolute_error": float(
            np.max(analytical_gradient_error)
        ),
        "analytical_value_discretization_max_absolute_error": float(
            np.max(np.abs(recurrence - analytical))
        ),
        "analytical_derivative_discretization_max_absolute_error": float(
            np.max(np.abs(derivative - analytical_derivative))
        ),
        "reference": "NumPy host float64 on the identical rounded inputs",
    }


def run_fixed_time_integrator_benchmarks(
    *,
    step_counts: list[int],
    state_sizes: list[int],
    max_steps: int | None,
    dtype: str,
    repeats: int,
) -> dict[str, Any]:
    if not phx.config.x64_enabled():
        raise ValueError("The qualified fixed recurrence requires x64 to remain enabled.")
    if jax.default_backend() == "mps" and dtype != "float32":
        raise ValueError("Metal's fixed recurrence requires explicit --dtype float32.")
    if not step_counts or not state_sizes or repeats < 1:
        raise ValueError("Step/state sweeps must be nonempty and repeats positive.")
    if any(steps < 2 or steps > 512 or steps & (steps - 1) for steps in step_counts):
        raise ValueError("Fixed step counts must be powers of two in [2, 512].")
    if any(size < 1 or size > 65536 for size in state_sizes):
        raise ValueError("State capacities must be in [1, 65536].")
    if max_steps is not None and not 1 <= max_steps <= 512:
        raise ValueError("max_steps must be in [1, 512].")
    records = [
        _fixed_case(
            steps=steps,
            state_size=size,
            max_steps=steps if max_steps is None else max_steps,
            dtype=dtype,
            repeats=repeats,
        )
        for steps in step_counts
        for size in state_sizes
    ]
    driver = Path(__file__).resolve()
    return {
        "identity": capture_benchmark_identity(
            driver.parents[1], driver, records[0]
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "configuration": {
            "route": "heun-fixed",
            "dtype": dtype,
            "step_counts": step_counts,
            "state_sizes": state_sizes,
            "max_steps": max_steps,
            "repeats": repeats,
            "control_flow": "compiled Diffrax fixed recurrence and reverse differentiation",
        },
        "cases": records,
        "passed": all(record["passed"] for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--route",
        choices=("all", "heun-fixed"),
        default="all",
        help="Keep the default broad suite, or select only the qualified fixed recurrence.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--step-counts",
        type=int,
        nargs="+",
        help="heun-fixed sweep: powers of two from 2 through 512; defaults to --steps.",
    )
    parser.add_argument(
        "--state-sizes",
        type=int,
        nargs="+",
        default=None,
        help="heun-fixed state capacities from 1 through 65536; default: 1 128 4096.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Explicit bounded solver workspace/budget; default: the selected step count.",
    )
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 1:
        raise ValueError("steps must be at least two and repeats must be positive.")
    if args.route == "all":
        if jax.default_backend() == "mps":
            parser.error("Metal requires --route heun-fixed --dtype float32.")
        if args.dtype != "float64":
            parser.error(
                "The broad suite retains its existing implicit float64 contract."
            )
        if (
            args.step_counts is not None
            or args.state_sizes is not None
            or args.max_steps is not None
        ):
            parser.error("Capacity controls require --route heun-fixed.")
        report = run_time_integrator_benchmarks(steps=args.steps, repeats=args.repeats)
    else:
        report = run_fixed_time_integrator_benchmarks(
            step_counts=args.step_counts or [args.steps],
            state_sizes=args.state_sizes or [1, 128, 4096],
            max_steps=args.max_steps,
            dtype=args.dtype,
            repeats=args.repeats,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.route == "heun-fixed" and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
