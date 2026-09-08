"""Clock/event neural-network execution with actual spike and queue evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

from phydrax.applications import electrophysiology as ep


def _runtime(cell_count: int, steps: int, mode: str):
    model = ep.LeakyIntegrateAndFire(1.0, 0.1, -65.0, -64.0, -65.0, refractory_ms=0.25)
    cells = tuple(
        ep.NeuralCellPlan(f"cell-{index}", model) for index in range(cell_count)
    )
    delay = 1.0 if mode == "clock" else 0.75
    connections = tuple(
        ep.SynapseConnection(
            f"ring-{source}",
            source,
            0,
            (source + 1) % cell_count,
            0,
            ep.CurrentSynapse(3.0, -0.2),
            delay_ms=delay,
            weight=1.0,
        )
        for source in range(cell_count)
    )
    return ep.NeuralNetworkPlan(
        cells,
        ep.SynapseNetworkPlan(
            (1,) * cell_count,
            cell_count,
            delay,
            1.0,
            connections=connections,
            execution=mode,
        ),
        queue_capacity=4 * cell_count,
        spike_capacity=2 * cell_count * steps,
        recording_capacity=steps,
        maximum_events_per_step=8,
        root_subdivisions=1,
    ).prepare()


def _measure(cell_count, steps, mode, warmup, repeats):
    runtime = _runtime(cell_count, steps, mode)
    state = ep.initialize_neural_network(runtime)
    inputs = ep.zero_neural_inputs(runtime, dtype=state.time_ms.dtype)
    inputs = ep.NeuralNetworkInputs(
        jnp.full((cell_count,), 2.0),
        inputs.voltage_clamp_mask,
        inputs.voltage_clamp_target_mV,
        inputs.modulation,
    )
    function = jax.jit(
        lambda initial: ep.run_neural_network(runtime, initial, steps, inputs=inputs)
    )
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(state), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(state), warmup=warmup, repeats=repeats
    )
    relation_bytes = sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(result.state.relations)
        if eqx.is_array(leaf)
    )
    queue_bytes = sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(result.state.queue)
        if eqx.is_array(leaf)
    )
    mean = execution.mean_seconds
    return {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_seconds_dict(),
        "cell_steps_per_second": None
        if mean in (None, 0.0)
        else cell_count * steps / mean,
        "emitted_spikes": int(jnp.sum(result.emitted_spikes)),
        "delivered_messages": int(jnp.sum(result.delivered_messages)),
        "maximum_queue_occupancy": int(jnp.max(result.maximum_queue_occupancy)),
        "all_steps_successful": bool(jnp.all(result.status == 0)),
        "all_event_sensitivities_valid": bool(jnp.all(result.sensitivity_valid)),
        "relation_state_bytes": relation_bytes,
        "queue_state_bytes": queue_bytes,
        "final_voltage_range_mV": [
            float(jnp.min(result.voltage_mV[-1])),
            float(jnp.max(result.voltage_mV[-1])),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--modes", nargs="+", choices=("clock", "event"), default=("clock", "event")
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.cells, args.steps, args.repeats) <= 0 or args.warmup < 0:
        raise ValueError(
            "Cells, steps and repeats must be positive; warmup must be nonnegative."
        )
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "cells": args.cells,
            "relations": args.cells,
            "steps": args.steps,
            "modes": args.modes,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "modes": {
            mode: _measure(args.cells, args.steps, mode, args.warmup, args.repeats)
            for mode in args.modes
        },
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
