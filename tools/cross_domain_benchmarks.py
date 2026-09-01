"""Benchmarks for reconciled cross-domain gravity and observation cores."""

from __future__ import annotations

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _block(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(function, *args):
    start = time.perf_counter()
    value = function(*args)
    _block(value)
    return value, time.perf_counter() - start


def main() -> None:
    count = 64
    coordinate = (jnp.arange(count, dtype=float) + 0.5) / count
    positions = jnp.stack(
        (
            coordinate,
            jnp.mod(13.0 * coordinate, 1.0),
            jnp.mod(29.0 * coordinate, 1.0),
        ),
        axis=-1,
    )
    masses = jnp.ones((count,)) / count
    kernel = phx.solver.NewtonianPairKernel(1.0, softening=0.01)
    direct = phx.solver.DirectParticleGravityPlan(kernel)
    direct_function = eqx.filter_jit(direct.evaluate)
    _, direct_compile = _measure(direct_function, positions, masses)
    direct_result, direct_steady = _measure(direct_function, positions, masses)

    tree_plan = phx.solver.ParticleOctreePlan3D((1.0, 1.0, 1.0), 3)
    tree = tree_plan.prepare(positions, masses)
    bh = phx.solver.BarnesHutGravityPlan(1.0, softening=0.01, opening_angle=0.5)
    bh_function = eqx.filter_jit(bh.evaluate)
    _, bh_compile = _measure(bh_function, tree)
    bh_result, bh_steady = _measure(bh_function, tree)

    source = phx.observation.CoordinateLayout(tuple(f"s{index}" for index in range(256)))
    target = phx.observation.CoordinateLayout(tuple(f"d{index}" for index in range(128)))
    response = phx.observation.LinearObservationPlan(
        jnp.ones((128, 256)) / 256.0, source, target
    )
    product = phx.observation.TheoryVector(jnp.ones((256,)), source, "benchmark")
    response_function = eqx.filter_jit(response.apply)
    _, response_compile = _measure(response_function, product)
    response_result, response_steady = _measure(response_function, product)

    report = {
        "particles": count,
        "direct_compile_seconds": direct_compile,
        "direct_steady_seconds": direct_steady,
        "direct_finite": bool(direct_result[1].successful),
        "barnes_hut_compile_seconds": bh_compile,
        "barnes_hut_steady_seconds": bh_steady,
        "barnes_hut_finite": bool(bh_result.successful),
        "response_shape": [128, 256],
        "response_compile_seconds": response_compile,
        "response_steady_seconds": response_steady,
        "response_finite": bool(jnp.all(jnp.isfinite(response_result.values))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
