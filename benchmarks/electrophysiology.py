from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

from phydrax.applications import electrophysiology as ep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("steps/repeats must be positive and warmup nonnegative")

    morphology = ep.parse_swc_text(
        """
        1 1 0 0 0 6 -1
        2 3 20 0 0 2 1
        3 3 60 15 0 1 2
        4 3 60 -15 0 1 2
        """,
        "benchmark-cell",
    ).morphology.prepare()
    membrane = ep.MembraneProgram((ep.PassiveLeak(0.3, -65.0), ep.HodgkinHuxleyNaK()))
    cable = ep.CableSolverPlan(
        0.025,
        scheme="crank-nicolson",
        residual_tolerance=1.0e-9,
    ).prepare(morphology, membrane)
    state = ep.initialize_cable_state(cable, jnp.full((4,), -65.0))
    zeros = jnp.zeros((4,))
    inputs = ep.CableStepInputs(
        zeros.at[0].set(0.2),
        zeros,
        zeros,
        jnp.zeros((4,), dtype=bool),
        zeros,
    )

    def run(initial):
        def step(carry, _):
            result = ep.step_cable(cable, carry, inputs)
            return result.state, result.evidence.successful

        return jax.lax.scan(step, initial, xs=None, length=args.steps)

    function = jax.jit(run)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(state), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(state), warmup=args.warmup, repeats=args.repeats
    )
    final_state, successful = result
    mean_seconds = execution.mean_seconds
    rate = None if mean_seconds in (None, 0.0) else 4 * args.steps / mean_seconds
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "compartments": 4,
            "steps": args.steps,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "identities": {
            "morphology": morphology.plan.plan_id,
            "cable": cable.plan.plan_id,
            "units": ep.ELECTROPHYSIOLOGY_UNITS.units_id,
        },
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": execution.to_seconds_dict(),
        "compartment_steps_per_second": rate,
        "physics": {
            "finite_voltage": bool(jnp.all(jnp.isfinite(final_state.voltage_mV))),
            "successful": bool(jnp.all(successful)),
            "minimum_voltage_mV": float(jnp.min(final_state.voltage_mV)),
            "maximum_voltage_mV": float(jnp.max(final_state.voltage_mV)),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
