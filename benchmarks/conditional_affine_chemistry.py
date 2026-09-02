#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import capture_environment


def _prepared(pair_count: int):
    species = tuple(
        name for index in range(pair_count) for name in (f"A{index}", f"B{index}")
    )
    elements = tuple(f"E{index}" for index in range(pair_count))
    composition = jnp.zeros((pair_count, 2 * pair_count), dtype=jnp.int32)
    for index in range(pair_count):
        composition = composition.at[index, 2 * index : 2 * index + 2].set(1)
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        species,
        (phx.equations.ChemicalPhaseKind.GAS,) * (2 * pair_count),
        jnp.ones((2 * pair_count,)),
        elements,
        composition,
        jnp.zeros((2 * pair_count,), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((2 * pair_count,), 10.0),
        jnp.zeros((2 * pair_count,)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=2000.0,
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "conditional-affine-benchmark",
        schema,
        thermodynamics,
        tuple(
            phx.equations.ChemicalReactionSpec(
                f"A{index}->B{index}",
                {f"A{index}": 1.0},
                {f"B{index}": 1.0},
                phx.equations.ArrheniusRatePlan(float(index + 1)),
            )
            for index in range(pair_count)
        ),
    ).prepare()
    return phx.equations.ChemicalConditionalAffinePlan(species).prepare(mechanism)


def _case(pair_count: int, batch_size: int, repetitions: int):
    prepared = _prepared(pair_count)
    state = jnp.zeros((batch_size, 2 * pair_count))
    state = state.at[:, 0::2].set(1.0)
    drivers = phx.equations.ChemicalConditionalAffineDrivers(
        jnp.zeros((batch_size, 0)),
        jnp.full((batch_size,), 500.0),
        jnp.full((batch_size,), 101325.0),
    )
    duration = jnp.full((batch_size,), 1.0e-2)
    action = eqx.filter_jit(prepared.advance)

    start = time.perf_counter()
    first = action(state, drivers, duration)
    jax.block_until_ready(first.candidate_state)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    for _ in range(repetitions):
        result = action(state, drivers, duration)
    jax.block_until_ready(result.candidate_state)
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions
    return {
        "pair_count": pair_count,
        "state_size": 2 * pair_count,
        "batch_size": batch_size,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "transitions_per_second": 1000.0 * batch_size / execution_ms,
        "maximum_invariant_residual": float(jnp.max(jnp.abs(result.element_residual))),
        "minimum_species": float(jnp.min(result.candidate_state)),
        "successful": bool(jnp.all(result.successful)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-counts", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 128, 1024])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/conditional_affine_chemistry.json"),
    )
    arguments = parser.parse_args()
    cases = [
        _case(pair_count, batch_size, arguments.repeats)
        for pair_count in arguments.pair_counts
        for batch_size in arguments.batch_sizes
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_successful": all(bool(case["successful"]) for case in cases),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
