#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Synchronized artificial-LIF forward and surrogate-backward sequence scaling.

Run: python -m tools.artificial_lif_benchmarks --quick
The full sweep varies sequence length and hidden width independently. Timings
separate lowering, compilation, first execution, and repeated warm execution.
Compiler memory estimates are not process/device peak-memory measurements.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from benchmarks._runtime import (
    compiler_evidence,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


def _measure(function, arguments, *, repetitions):
    jitted = jax.jit(function)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(*arguments), lambda lowered: lowered.compile()
    )
    first, first_seconds = measure_synchronized(lambda: compiled(*arguments))
    _, steady = measure_repeated(
        lambda: compiled(*arguments), warmup=0, repeats=repetitions
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Backend did not expose compiler cost or memory analysis.",
    )
    return first, {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_execution_seconds": first_seconds,
        "steady": steady.to_seconds_dict(),
        "compiler": asdict(evidence),
        "compiler_estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
    }


def benchmark_case(*, length: int, width: int, repetitions: int) -> dict:
    cell_key, input_key = jr.split(jr.key(2026))
    cell = phx.nn.layers.ArtificialLIFCell(
        width,
        width,
        time_constant_ms=5.0,
        threshold=0.5,
        key=cell_key,
    )
    cell = eqx.tree_at(lambda current: current.bias, cell, jnp.full((width,), 0.8))
    model = phx.nn.models.RecurrentSequenceModel(cell)
    inputs = jr.normal(input_key, (length, width), dtype=jnp.float32)
    valid = jnp.ones((length,), dtype=bool)
    time = jnp.arange(length, dtype=jnp.float32)

    def forward(current, values):
        return current(phx.nn.layers.RecurrentBatch(values, valid, time=time))

    def objective(current, values):
        spikes = forward(current, values)
        return jnp.mean(spikes)

    outputs, forward_timing = _measure(forward, (model, inputs), repetitions=repetitions)
    (loss, gradient), backward_timing = _measure(
        jax.value_and_grad(objective), (model, inputs), repetitions=repetitions
    )
    gradient_leaves = jax.tree.leaves(gradient)
    gradient_norm = jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in gradient_leaves))
    if not bool(jnp.isfinite(loss) & jnp.isfinite(gradient_norm)):
        raise RuntimeError(
            "Artificial LIF benchmark produced non-finite primal or gradient."
        )
    return {
        "sequence_length": length,
        "input_size": width,
        "hidden_size": width,
        "dtype": str(inputs.dtype),
        "spike_fraction": float(jnp.mean(outputs)),
        "surrogate_gradient_norm": float(gradient_norm),
        "forward": forward_timing,
        "value_and_surrogate_gradient": backward_timing,
    }


def run_benchmarks(*, quick: bool = False) -> dict:
    lengths = (32, 128) if quick else (128, 512, 2048, 8192)
    widths = (8,) if quick else (16, 64, 128)
    repetitions = 2 if quick else 5
    return {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "quick": quick,
        "surrogate": "fast_sigmoid",
        "surrogate_width": 1.0,
        "detach_reset": False,
        "cases": [
            benchmark_case(length=length, width=width, repetitions=repetitions)
            for width in widths
            for length in lengths
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    payload = json.dumps(run_benchmarks(quick=arguments.quick), indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n")
        print(arguments.output)


if __name__ == "__main__":
    main()
