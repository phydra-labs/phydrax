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


def _model(component_count: int):
    names = tuple(f"X{index}" for index in range(component_count))
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        names,
        (phx.equations.ChemicalPhaseKind.GAS,) * component_count,
        jnp.linspace(0.002, 0.044, component_count),
        names,
        jnp.eye(component_count, dtype=jnp.int32),
        jnp.zeros((component_count,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    cv = jnp.full((component_count, 1), 2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT)
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        cv,
        jnp.zeros((component_count,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=5000.0,
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, thermodynamics)
    return phx.equations.HomogeneousHelmholtzPlan(
        ideal, phx.equations.ZeroResidualHelmholtzTerm(schema)
    )


def _case(component_count: int, batch_size: int) -> dict[str, float | int | bool]:
    model = _model(component_count)
    temperature = jnp.linspace(300.0, 1500.0, batch_size)
    molar_density = jnp.linspace(1.0, 50.0, batch_size)
    composition = jnp.full((batch_size, component_count), 1.0 / component_count)
    action = eqx.filter_jit(model.evaluate)

    start = time.perf_counter()
    first = action(temperature, molar_density, composition)
    jax.block_until_ready(first.pressure)
    compile_ms = 1000.0 * (time.perf_counter() - start)

    repetitions = 10
    start = time.perf_counter()
    for _ in range(repetitions):
        result = action(temperature, molar_density, composition)
    jax.block_until_ready(result.pressure)
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions
    return {
        "component_count": component_count,
        "batch_size": batch_size,
        "compile_and_first_ms": compile_ms,
        "execution_ms": execution_ms,
        "maximum_pressure": float(jnp.max(result.pressure)),
        "successful": bool(jnp.all(result.evidence.successful)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-counts", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 1024])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    cases = [
        _case(component_count, batch_size)
        for component_count in arguments.component_counts
        for batch_size in arguments.batch_sizes
    ]
    payload = {
        "cases": cases,
        "all_successful": all(case["successful"] for case in cases),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
