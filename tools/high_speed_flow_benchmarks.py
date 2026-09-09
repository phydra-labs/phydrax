#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.equations import (
    ChemicalPhaseKind,
    ChemicalSpeciesSchema,
    ConstantTransport,
    HomogeneousHelmholtzPlan,
    HomogeneousMixtureCompressibleNavierStokesSystem,
    IdealGasReferenceHelmholtzTerm,
    PolynomialSpeciesThermodynamicsPlan,
    SpalartAllmarasNegativePlan,
    UNIVERSAL_GAS_CONSTANT,
    ZeroResidualHelmholtzTerm,
)


def _system():
    schema = ChemicalSpeciesSchema.from_unique_species(
        ("air",),
        (ChemicalPhaseKind.GAS,),
        jnp.asarray((0.02897,)),
        ("air",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    calorics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((2.5 * UNIVERSAL_GAS_CONSTANT,)),
        jnp.asarray((0.0,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=5000.0,
    )
    thermodynamics = HomogeneousHelmholtzPlan(
        IdealGasReferenceHelmholtzTerm(schema, calorics),
        ZeroResidualHelmholtzTerm(schema),
    )
    return HomogeneousMixtureCompressibleNavierStokesSystem(
        thermodynamics, ConstantTransport(1.8e-5, 0.026), 2
    )


def _timed(function, arguments, iterations):
    start = time.perf_counter()
    compiled = function(*arguments)
    jax.block_until_ready(compiled)
    compile_seconds = time.perf_counter() - start
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        value = function(*arguments)
        jax.block_until_ready(value)
        samples.append(time.perf_counter() - start)
    return compile_seconds, float(np.median(samples))


def benchmark(*, smoke: bool, iterations: int) -> dict[str, object]:
    nx, ny = (16, 8) if smoke else (128, 64)
    system = _system()
    x = jnp.linspace(0.0, 1.0, nx)[:, None]
    y = jnp.linspace(0.0, 1.0, ny)[None, :]
    primitive = jnp.stack(
        (
            jnp.ones((nx, ny)),
            100.0 + 5.0 * jnp.broadcast_to(y, (nx, ny)),
            2.0 * jnp.broadcast_to(x, (nx, ny)),
            300.0 + 20.0 * jnp.broadcast_to(x * y, (nx, ny)),
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    gradient = jnp.zeros(state.shape + (2,), dtype=state.dtype)
    gradient = gradient.at[..., 1, 0].set(1.0)
    gradient = gradient.at[..., 2, 1].set(1.0)
    viscous = eqx.filter_jit(system.viscous_flux)
    compile_seconds, median_seconds = _timed(viscous, (state, gradient), iterations)

    sa = SpalartAllmarasNegativePlan()
    sa_function = eqx.filter_jit(sa.evaluate)
    density = jnp.ones((nx, ny))
    molecular = jnp.full((nx, ny), 1.8e-5)
    working = jnp.full((nx, ny), 3.0e-5)
    velocity_gradient = jnp.zeros((nx, ny, 2, 2)).at[..., 0, 1].set(10.0)
    working_gradient = jnp.zeros((nx, ny, 2))
    wall_distance = jnp.broadcast_to(0.01 + y, (nx, ny))
    sa_compile, sa_median = _timed(
        sa_function,
        (
            density,
            molecular,
            working,
            velocity_gradient,
            working_gradient,
            wall_distance,
        ),
        iterations,
    )
    cells = nx * ny
    return {
        "benchmark": "high-speed-flow-kernels",
        "shape": [nx, ny],
        "iterations": iterations,
        "viscous_flux": {
            "compile_seconds": compile_seconds,
            "median_seconds": median_seconds,
            "cell_evaluations_per_second": cells / median_seconds,
        },
        "sa_negative": {
            "compile_seconds": sa_compile,
            "median_seconds": sa_median,
            "cell_evaluations_per_second": cells / sa_median,
        },
        "finite": all(
            np.isfinite(value)
            for value in (compile_seconds, median_seconds, sa_compile, sa_median)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.iterations <= 0:
        raise ValueError("iterations must be positive.")
    report = benchmark(smoke=arguments.smoke, iterations=arguments.iterations)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if not report["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
