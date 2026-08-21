#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic serial-versus-causal recurrent execution benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _ready(value: Any) -> Any:
    return jax.block_until_ready(value)


def _timings(function, argument, *, repetitions: int):
    compiled = jax.jit(function)
    started = time.perf_counter()
    first = compiled(argument)
    _ready(first)
    cold = time.perf_counter() - started
    warm = []
    for _ in range(int(repetitions)):
        started = time.perf_counter()
        value = compiled(argument)
        _ready(value)
        warm.append(time.perf_counter() - started)
    execution = statistics.median(warm)
    return first, {
        "cold_seconds": cold,
        "compile_seconds": max(0.0, cold - execution),
        "execute_seconds": execution,
    }


def _cell(kind: str, width: int, key):
    if kind == "rnn":
        return phx.nn.layers.RNNCell(width, width, key=key)
    if kind == "gru":
        return phx.nn.layers.GRUCell(width, width, key=key)
    if kind == "lstm":
        return phx.nn.layers.LSTMCell(width, width, key=key)
    raise ValueError(f"Unknown recurrent cell kind {kind!r}.")


def benchmark_case(
    *,
    cell_kind: str,
    sequence_length: int,
    hidden_size: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    root = jr.key(seed)
    cell_key, input_key = jr.split(root)
    cell = _cell(cell_kind, hidden_size, cell_key)
    inputs = jr.normal(input_key, (sequence_length, hidden_size))
    batch = phx.nn.layers.RecurrentBatch(
        inputs,
        jnp.ones((sequence_length,), dtype=bool),
    )
    termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=2e-6,
        relative_residual=2e-6,
        maximum_steps=64,
    )
    causal_config = phx.nn.layers.CausalRecurrentConfig(
        method=phx.nonlinear.CausalLevenbergMarquardt(
            linearization=phx.nonlinear.CausalLinearizationPolicy("diagonal-exact")
        ),
        termination=termination,
    )

    def serial_function(current):
        return phx.nn.layers.run_recurrent(current, batch).outputs

    def causal_function(current):
        return phx.nn.layers.run_causal_recurrent(
            current,
            batch,
            config=causal_config,
        ).outputs

    serial, serial_timing = _timings(
        serial_function,
        cell,
        repetitions=repetitions,
    )
    causal, causal_timing = _timings(
        causal_function,
        cell,
        repetitions=repetitions,
    )
    causal_result = phx.nn.layers.run_causal_recurrent(
        cell,
        batch,
        config=causal_config,
    )
    serial_gradient = jax.grad(lambda current: jnp.sum(serial_function(current)))(cell)
    causal_gradient = jax.grad(lambda current: jnp.sum(causal_function(current)))(cell)
    gradient_error = max(
        float(jnp.max(jnp.abs(left - right)))
        for left, right in zip(
            jax.tree.leaves(serial_gradient),
            jax.tree.leaves(causal_gradient),
            strict=True,
        )
    )
    output_error = float(jnp.max(jnp.abs(serial - causal)))
    speedup = serial_timing["execute_seconds"] / causal_timing["execute_seconds"]
    outer_iterations = int(causal_result.diagnostics.causal.iteration_count)
    final_residual = causal_result.diagnostics.causal.residual_norm[
        max(outer_iterations - 1, 0)
    ]
    return {
        "cell": cell_kind,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "dtype": str(inputs.dtype),
        "serial": serial_timing,
        "causal": causal_timing,
        "warm_speedup": speedup,
        "maximum_output_error": output_error,
        "maximum_gradient_error": gradient_error,
        "outer_iterations": outer_iterations,
        "maximum_direct_residual": float(final_residual),
        "transition_evaluations": int(
            causal_result.diagnostics.causal.transition_evaluations
        ),
        "jvp_evaluations": int(causal_result.diagnostics.causal.jvp_evaluations),
        "fallback_used": bool(causal_result.diagnostics.fallback_used),
    }


def run_recurrent_benchmarks(*, quick: bool = False) -> dict[str, Any]:
    lengths = (128,) if quick else (128, 512, 2048, 8192)
    widths = (16,) if quick else (16, 64, 256)
    kinds = ("rnn",) if quick else ("rnn", "gru", "lstm")
    repetitions = 2 if quick else 5
    cases = []
    index = 0
    for kind in kinds:
        for length in lengths:
            for width in widths:
                index += 1
                cases.append(
                    benchmark_case(
                        cell_kind=kind,
                        sequence_length=length,
                        hidden_size=width,
                        repetitions=repetitions,
                        seed=20260821 + index,
                    )
                )
    return {
        "schema": "phydrax-recurrent-causal-benchmark-v1",
        "backend": jax.default_backend(),
        "quick": bool(quick),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    report = run_recurrent_benchmarks(quick=arguments.quick)
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
