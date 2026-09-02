#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    velocities = jnp.asarray(
        tuple(itertools.product((-2.0, -1.0, 0.0, 1.0, 2.0), repeat=3))
    )
    quadrature = phx.equations.MolecularVelocityQuadrature(
        velocities, jnp.ones((velocities.shape[0],)), 1
    )
    multipliers = jnp.asarray((-2.0, 0.1, -0.05, 0.02, -0.8))
    population = jnp.exp(quadrature.moment_features @ multipliers)
    target = quadrature.moments(population)
    action = eqx.filter_jit(
        phx.equations.PositiveDiscreteMaxwellianPlan(quadrature).solve
    )
    start = time.perf_counter()
    first = action(target)
    jax.block_until_ready(first.population)
    compile_ms = 1000.0 * (time.perf_counter() - start)
    repetitions = 10
    start = time.perf_counter()
    for _ in range(repetitions):
        result = action(target)
    jax.block_until_ready(result.population)
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions
    payload = {
        "velocity_count": quadrature.velocity_count,
        "compile_and_first_ms": compile_ms,
        "execution_ms": execution_ms,
        "maximum_moment_residual": float(jnp.max(jnp.abs(result.moment_residual))),
        "successful": bool(result.successful),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
