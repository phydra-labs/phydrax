#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark a 10,000-point activated muscle-block constitutive update."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle import continuum


def _material():
    architecture = continuum.UniformFiberArchitecturePlan(
        "benchmark-longitudinal"
    ).prepare(jnp.asarray((1.0, 0.0, 0.0)))
    return continuum.EngelhardtGasam2025Plan("benchmark-gasam-2025").prepare(
        continuum.EngelhardtGasam2025Parameters.published_multiload_fit(),
        architecture,
        0.65,
    )


def _deformations(point_count):
    coordinate = jnp.linspace(0.0, 1.0, point_count)
    stretch = 0.72 + 0.66 * coordinate
    transverse = stretch ** -0.5
    deformation = jnp.zeros((point_count, 3, 3))
    deformation = deformation.at[:, 0, 0].set(stretch)
    deformation = deformation.at[:, 1, 1].set(transverse)
    deformation = deformation.at[:, 2, 2].set(transverse)
    shear = 0.04 * jnp.sin(2.0 * jnp.pi * coordinate)
    return deformation.at[:, 0, 1].set(shear)


def _case(point_count, repetitions):
    material = _material()
    deformations = _deformations(point_count)

    @eqx.filter_jit
    def evaluate(values):
        def point(value):
            return material.evaluate(value, 0.0).first_piola

        return jax.vmap(point)(values)

    start = time.perf_counter()
    first = evaluate(deformations)
    jax.block_until_ready(first)
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = first
    for _ in range(repetitions):
        result = evaluate(deformations)
    jax.block_until_ready(result)
    steady_seconds = (time.perf_counter() - start) / repetitions
    return {
        "material_points": point_count,
        "activation": 0.65,
        "fiber_stretch_range": [0.72, 1.38],
        "first_execution_seconds": first_seconds,
        "steady_execution_seconds": steady_seconds,
        "material_points_per_second": point_count / steady_seconds,
        "first_piola_checksum_pa": float(jnp.sum(result)),
        "finite": bool(jnp.all(jnp.isfinite(result))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/skeletal_muscle_continuum.json"),
    )
    arguments = parser.parse_args()
    point_count = 128 if arguments.smoke else 10_000
    repetitions = 2 if arguments.smoke else arguments.repetitions
    case = _case(point_count, repetitions)
    result = {
        "backend": jax.default_backend(),
        "source": "Engelhardt et al. 2025 GASAM",
        "case": case,
        "all_valid": case["finite"],
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not result["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
