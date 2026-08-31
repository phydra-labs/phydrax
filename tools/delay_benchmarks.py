#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import (
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)


def _constant_delay_problem(drift, history, delays, /, **kwargs):
    delay_values = jnp.asarray(delays).reshape((-1,))
    delay_terms = tuple(
        phx.solver.ConstantDelay(f"delay_{index}", delay_values[index])
        for index in range(int(delay_values.size))
    )
    return phx.solver.DelayDifferentialProblem(
        drift,
        history,
        delay_terms,
        **kwargs,
    )


def _measure(
    operation: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    argument: jax.Array,
    /,
    *,
    repeats: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], dict[str, Any]]:
    jitted = jax.jit(operation)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(argument),
        lambda lowered: lowered.compile(),
    )
    output, first_execution_seconds = measure_synchronized(lambda: compiled(argument))
    output, distribution = measure_repeated(
        lambda: compiled(argument),
        warmup=0,
        repeats=repeats,
    )
    return output, {
        "execution_mode": "jit",
        "compile_ms": 1_000.0
        * (compilation.lowering_seconds + compilation.compilation_seconds),
        "first_execution_ms": 1_000.0 * first_execution_seconds,
        "steady_ms": 1_000.0 * float(distribution.mean_seconds),
    }


def _measure_eager(
    operation: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    argument: jax.Array,
    /,
    *,
    repeats: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], dict[str, Any]]:
    output, first_execution_seconds = measure_synchronized(lambda: operation(argument))
    output, distribution = measure_repeated(
        lambda: operation(argument),
        warmup=0,
        repeats=repeats,
    )
    return output, {
        "execution_mode": "eager-host-orchestrated",
        "compile_ms": None,
        "first_execution_ms": 1_000.0 * first_execution_seconds,
        "steady_ms": 1_000.0 * float(distribution.mean_seconds),
    }


def _record(
    method: str,
    observed: tuple[jax.Array, jax.Array, jax.Array],
    expected: jax.Array,
    timings: dict[str, Any],
    /,
) -> dict[str, Any]:
    state, accepted, rejected = observed
    return {
        "method": method,
        "terminal_max_abs_error": float(jnp.max(jnp.abs(state - expected))),
        "num_accepted_steps": int(accepted),
        "num_rejected_steps": int(rejected),
        **timings,
    }


def run_benchmarks(
    *,
    repeats: int = 5,
    state_dim: int = 32,
    fixed_steps: int = 4096,
    family_steps: int = 128,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    if state_dim < 1:
        raise ValueError("state_dim must be at least one.")
    if fixed_steps < 10:
        raise ValueError("fixed_steps must be at least ten.")
    if family_steps < 10:
        raise ValueError("family_steps must be at least ten.")

    base = jnp.linspace(0.5, 1.5, state_dim)
    delays = jnp.asarray([0.2, 0.37, 0.5])
    weights = jnp.asarray([0.2, 0.3, 0.5])
    smooth_t1 = 2.0
    smooth_rate = jnp.asarray(0.4)
    smooth_expected = jnp.exp(smooth_rate * smooth_t1) * base

    def ordinary_smooth(rate):
        solution = dfx.diffeqsolve(
            dfx.ODETerm(lambda time, state, args: args * state),
            dfx.Tsit5(),
            t0=0.0,
            t1=smooth_t1,
            dt0=None,
            y0=base,
            args=rate,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-8),
        )
        return (
            solution.ys[0],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    def delay_smooth(rate):
        problem = _constant_delay_problem(
            lambda time, state, delayed, args: jnp.tensordot(
                args * weights * jnp.exp(args * delays),
                delayed.stacked,
                axes=((0,), (0,)),
            ),
            lambda time, args: jnp.exp(args * time) * base,
            delays,
            t0=0.0,
            t1=smooth_t1,
            args=rate,
        )
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([smooth_t1]),
            max_steps=4096,
        )
        return (
            solution.states[0],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    def fixed_delay_smooth(rate):
        problem = _constant_delay_problem(
            lambda time, state, delayed, args: jnp.tensordot(
                args * weights * jnp.exp(args * delays),
                delayed.stacked,
                axes=((0,), (0,)),
            ),
            lambda time, args: jnp.exp(args * time) * base,
            delays,
            t0=0.0,
            t1=smooth_t1,
            args=rate,
        )
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([smooth_t1]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=smooth_t1 / fixed_steps,
            max_steps=fixed_steps + 16,
        )
        return (
            solution.states[0],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    stiff_delay = 0.2
    stiff_t1 = 0.5
    stiff_expected = jnp.exp(-stiff_t1) * base

    def implicit_stiff(scale):
        problem = _constant_delay_problem(
            lambda time, state, delayed, args: (
                -1000.0 * state + 999.0 * jnp.exp(-stiff_delay) * delayed[0]
            ),
            lambda time, args: jnp.exp(-time) * base * args,
            jnp.asarray([stiff_delay]),
            t0=0.0,
            t1=stiff_t1,
            args=scale,
        )
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([stiff_t1]),
            solver=dfx.Kvaerno5(),
            max_steps=4096,
        )
        return (
            solution.states[0],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    def fixed_stiff(scale):
        problem = _constant_delay_problem(
            lambda time, state, delayed, args: (
                -1000.0 * state + 999.0 * jnp.exp(-stiff_delay) * delayed[0]
            ),
            lambda time, args: jnp.exp(-time) * base * args,
            jnp.asarray([stiff_delay]),
            t0=0.0,
            t1=stiff_t1,
            args=scale,
        )
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([stiff_t1]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=stiff_t1 / fixed_steps,
            max_steps=fixed_steps + 16,
        )
        return (
            solution.states[0],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    family_t1 = 1.0
    family_times = jnp.linspace(0.0, family_t1, family_steps + 1)
    family_expected = jnp.exp(smooth_rate * family_t1) * base
    family_lags = jnp.asarray([0.1, 0.2, 0.3])
    family_weights = jnp.asarray([0.2, 0.3, 0.5])

    def functional_delay(rate):
        functional = phx.solver.FunctionalDelay(
            "window",
            lambda time, state, history, args: (
                jnp.tensordot(
                    family_weights,
                    history.values(family_lags),
                    axes=((0,), (0,)),
                )
                / jnp.sum(family_weights * jnp.exp(-args * family_lags))
            ),
            (0.1, 0.3),
            discontinuity_lags=family_lags,
        )
        problem = phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: args * memory["window"],
            lambda time, args: jnp.exp(args * time) * base,
            (functional,),
            t0=0.0,
            t1=family_t1,
            args=rate,
        )
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([family_t1]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=family_t1 / family_steps,
            max_steps=family_steps + 16,
        )
        return (
            solution.states[0],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    neutral_delay = 0.2
    neutral_weight = 0.2

    def transformed_neutral(rate):
        coefficient = rate * (1.0 - neutral_weight * jnp.exp(-rate * neutral_delay))
        problem = phx.solver.NeutralDelayProblem(
            lambda time, memory, args: neutral_weight * memory["past"],
            lambda time, state, memory, args: coefficient * state,
            lambda time, args: jnp.exp(args * time) * base,
            (phx.solver.ConstantDelay("past", neutral_delay),),
            t0=0.0,
            t1=family_t1,
            args=rate,
        )
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([family_t1]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=family_t1 / family_steps,
            max_steps=family_steps + 16,
        )
        return (
            solution.states[0],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    rough_delay = 4.0 / family_steps
    rough_control = phx.stochastic.GeometricRoughPath.from_values(
        family_times,
        family_times[:, None],
    )

    def rough_delay_davie(rate):
        problem = phx.solver.RoughDelayDifferentialProblem(
            lambda time, state, memory, args: (
                args * jnp.exp(args * rough_delay) * memory["past"]
            )[..., None],
            lambda time, args: jnp.exp(args * time) * base,
            (phx.solver.ConstantDelay("past", rough_delay),),
            t0=0.0,
            driver_dimension=1,
            args=rate,
        )
        solution = phx.solver.solve_rough_delay(
            problem,
            rough_control,
            solver=phx.solver.Davie(),
        )
        return (
            solution.states[-1],
            jnp.asarray(solution.control.num_steps),
            jnp.asarray(0),
        )

    jump_events = phx.stochastic.JumpEventBatch(
        jnp.asarray([0.25, 0.75]),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.ones((2, 1)),
        jnp.ones((2,), dtype=bool),
        jnp.asarray(0, dtype=jnp.int32),
        mark_shape=(1,),
    )

    jump_base_problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.zeros_like(state),
        lambda time, args: base,
        (phx.solver.ConstantDelay("past", 0.2),),
        t0=0.0,
        t1=family_t1,
        args=smooth_rate,
    )
    jump_problem = phx.solver.JumpDelayProblem(
        jump_base_problem,
        lambda time, state, memory, channel, mark, args: state + args * mark,
        mark_shape=(1,),
    )

    def prescribed_jump_delay(_rate):
        solution = phx.solver.solve_jump_delay(
            jump_problem,
            jump_events,
            save_times=jnp.asarray([family_t1]),
            solver=dfx.Euler(),
            dt0=family_t1 / family_steps,
            max_steps=family_steps + 16,
        )
        return (
            solution.states[0],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    def convolution_volterra(rate):
        problem = phx.solver.ConvolutionVolterraProblem(
            lambda time, state, args: args * state,
            base,
            t0=0.0,
            t1=family_t1,
            args=rate,
        )
        solution = phx.solver.solve_convolution_volterra(
            problem,
            times=family_times,
        )
        return (
            solution.states[-1],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    def caputo_order_one(rate):
        problem = phx.solver.CaputoFractionalProblem(
            lambda time, state, args: args * state,
            base,
            1.0,
            t0=0.0,
            t1=family_t1,
            args=rate,
        )
        solution = phx.solver.solve_caputo_fractional(
            problem,
            times=family_times,
        )
        return (
            solution.states[-1],
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
        )

    records = {}
    observed, timings = _measure(ordinary_smooth, smooth_rate, repeats=repeats)
    records["ordinary_tsit5_smooth"] = _record(
        "diffrax.Tsit5",
        observed,
        smooth_expected,
        timings,
    )
    observed, timings = _measure(delay_smooth, smooth_rate, repeats=repeats)
    records["retarded_tsit5_smooth"] = _record(
        "solve_diffrax_delay(Tsit5)",
        observed,
        smooth_expected,
        timings,
    )
    observed, timings = _measure(fixed_delay_smooth, smooth_rate, repeats=repeats)
    records["fixed_euler_smooth"] = _record(
        "solve_diffrax_delay(Euler)",
        observed,
        smooth_expected,
        timings,
    )
    scale = jnp.asarray(1.0)
    observed, timings = _measure(implicit_stiff, scale, repeats=repeats)
    records["retarded_kvaerno5_stiff"] = _record(
        "solve_diffrax_delay(Kvaerno5)",
        observed,
        stiff_expected,
        timings,
    )
    observed, timings = _measure(fixed_stiff, scale, repeats=repeats)
    records["fixed_euler_stiff"] = _record(
        "solve_diffrax_delay(Euler)",
        observed,
        stiff_expected,
        timings,
    )

    family_records = {}
    observed, timings = _measure(functional_delay, smooth_rate, repeats=repeats)
    family_records["functional_delay_euler"] = _record(
        "solve_diffrax_delay(FunctionalDelay, Euler)",
        observed,
        family_expected,
        timings,
    )
    observed, timings = _measure(transformed_neutral, smooth_rate, repeats=repeats)
    family_records["transformed_neutral_euler"] = _record(
        "solve_diffrax_delay(NeutralDelayProblem, Euler)",
        observed,
        family_expected,
        timings,
    )
    observed, timings = _measure_eager(
        rough_delay_davie,
        smooth_rate,
        repeats=repeats,
    )
    family_records["rough_delay_davie"] = _record(
        "solve_rough_delay(Davie)",
        observed,
        family_expected,
        timings,
    )
    observed, timings = _measure_eager(
        prescribed_jump_delay,
        smooth_rate,
        repeats=repeats,
    )
    family_records["prescribed_jump_delay"] = _record(
        "solve_jump_delay(Euler)",
        observed,
        base + 2.0 * smooth_rate,
        timings,
    )
    observed, timings = _measure(convolution_volterra, smooth_rate, repeats=repeats)
    family_records["convolution_volterra"] = _record(
        "solve_convolution_volterra",
        observed,
        family_expected,
        timings,
    )
    observed, timings = _measure(caputo_order_one, smooth_rate, repeats=repeats)
    family_records["caputo_order_one"] = _record(
        "solve_caputo_fractional(order=1)",
        observed,
        family_expected,
        timings,
    )

    def rolling_memory(horizon):
        problem = _constant_delay_problem(
            lambda time, state, delayed, args: jnp.tensordot(
                args * weights * jnp.exp(args * delays),
                delayed.stacked,
                axes=((0,), (0,)),
            ),
            lambda time, args: jnp.exp(args * time) * base,
            delays,
            t0=0.0,
            t1=horizon,
            args=smooth_rate,
        )
        capacity = phx.solver.fixed_delay_history_capacity(
            jnp.max(delays),
            0.01,
        )
        started = time.perf_counter()
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([horizon]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=0.01,
            history_mode="rolling",
            history_capacity=capacity,
            max_steps=None,
            throw=True,
        )
        synchronize(solution.states)
        wall_ms = 1e3 * (time.perf_counter() - started)
        expected = jnp.exp(smooth_rate * horizon) * base
        return {
            "horizon": horizon,
            "wall_ms": wall_ms,
            "terminal_max_abs_error": float(
                jnp.max(jnp.abs(solution.states[0] - expected))
            ),
            "num_accepted_steps": int(solution.stats["num_accepted_steps"]),
            "history_capacity": int(solution.stats["history_capacity"]),
            "active_history_entries": int(solution.stats["history_max_occupancy"]),
            "active_history_bytes": int(solution.stats["active_history_bytes"]),
        }

    def segmented_memory(horizon):
        problem = _constant_delay_problem(
            lambda time, state, delayed, args: jnp.tensordot(
                args * weights * jnp.exp(args * delays),
                delayed.stacked,
                axes=((0,), (0,)),
            ),
            lambda time, args: jnp.exp(args * time) * base,
            delays,
            t0=0.0,
            t1=horizon,
            args=smooth_rate,
        )
        started = time.perf_counter()
        solution = phx.solver.solve_diffrax_delay_segmented(
            problem,
            save_times=jnp.asarray([horizon]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=0.01,
            max_steps_per_segment=64,
            throw=True,
        )
        synchronize(solution.states)
        wall_ms = 1e3 * (time.perf_counter() - started)
        expected = jnp.exp(smooth_rate * horizon) * base
        return {
            "horizon": horizon,
            "wall_ms": wall_ms,
            "terminal_max_abs_error": float(
                jnp.max(jnp.abs(solution.states[0] - expected))
            ),
            "num_segments": int(solution.stats["num_segments"]),
            "num_accepted_steps": int(solution.stats["num_accepted_steps"]),
            "history_capacity": int(solution.stats["history_capacity"]),
            "active_history_entries": int(solution.continuation.active_history.size),
            "active_history_bytes": int(solution.stats["active_history_bytes"]),
        }

    short_memory = segmented_memory(2.0)
    long_memory = segmented_memory(8.0)
    short_rolling_memory = rolling_memory(2.0)
    long_rolling_memory = rolling_memory(8.0)
    memory_behavior = {
        "method": "solve_diffrax_delay_segmented(Euler)",
        "execution_mode": "host-segmented",
        "short": short_memory,
        "long": long_memory,
        "capacity_independent_of_horizon": (
            short_memory["history_capacity"] == long_memory["history_capacity"]
        ),
        "allocated_bytes_independent_of_horizon": (
            short_memory["active_history_bytes"] == long_memory["active_history_bytes"]
        ),
        "whole_solve_rolling": {
            "method": "solve_diffrax_delay(Euler, history_mode='rolling')",
            "execution_mode": "jit-whole-solve",
            "short": short_rolling_memory,
            "long": long_rolling_memory,
            "capacity_independent_of_horizon": (
                short_rolling_memory["history_capacity"]
                == long_rolling_memory["history_capacity"]
            ),
            "allocated_bytes_independent_of_horizon": (
                short_rolling_memory["active_history_bytes"]
                == long_rolling_memory["active_history_bytes"]
            ),
        },
    }
    return {
        "configuration": {
            "backend": jax.default_backend(),
            "repeats": repeats,
            "state_dim": state_dim,
            "num_delays": int(delays.size),
            "fixed_steps": fixed_steps,
            "family_steps": family_steps,
            "dtype": str(base.dtype),
        },
        "benchmarks": records,
        "family_benchmarks": family_records,
        "memory_behavior": memory_behavior,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed/adaptive Diffrax delay solves and bounded history."
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--fixed-steps", type=int, default=4096)
    parser.add_argument("--family-steps", type=int, default=128)
    arguments = parser.parse_args()
    report = run_benchmarks(
        repeats=arguments.repeats,
        state_dim=arguments.state_dim,
        fixed_steps=arguments.fixed_steps,
        family_steps=arguments.family_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
