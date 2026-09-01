"""Compile and steady-state benchmarks for cosmology boundary closures."""

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
    cosmo = phx.applications.cosmology
    source = cosmo.CoordinateLayout(tuple(f"t{i}" for i in range(256)))
    target = cosmo.CoordinateLayout(tuple(f"d{i}" for i in range(128)))
    matrix = jnp.reshape(jnp.arange(128 * 256, dtype=float), (128, 256)) / (128 * 256)
    observation = cosmo.LinearObservationPlan(matrix, source, target)
    theory = cosmo.TheoryVector(jnp.ones((256,)), source, "benchmark")
    observation_function = eqx.filter_jit(observation.apply)
    _, observation_compile = _measure(observation_function, theory)
    observed, observation_steady = _measure(observation_function, theory)

    ewald = cosmo.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        softening=0.01,
        alpha=5.0,
        real_shells=1,
        reciprocal_modes=3,
    )
    coordinate = (jnp.arange(16, dtype=float) + 0.5) / 16.0
    positions = jnp.stack(
        (
            coordinate,
            jnp.mod(3.0 * coordinate, 1.0),
            jnp.mod(5.0 * coordinate, 1.0),
        ),
        axis=-1,
    )
    ewald_function = eqx.filter_jit(lambda value: ewald.evaluate(value, jnp.ones((16,))))
    _, ewald_compile = _measure(ewald_function, positions)
    ewald_result, ewald_steady = _measure(ewald_function, positions)

    pixel_count = 12
    pixels = jnp.repeat(jnp.arange(pixel_count), 4)
    angles = jnp.tile(
        jnp.asarray([0.0, jnp.pi / 4.0, jnp.pi / 2.0, 3.0 * jnp.pi / 4.0]), pixel_count
    )
    pointing = cosmo.CmbPointingProduct(
        pixels,
        angles,
        jnp.zeros_like(pixels, dtype=bool),
        jnp.tile(jnp.arange(4), pixel_count),
        pixel_count=pixel_count,
    )
    tod = cosmo.CmbTodProduct(
        jnp.ones((pixels.size,)),
        pointing,
        jnp.asarray(1.0),
        jnp.asarray(0.1),
        "benchmark-tod",
    )
    mapmaking = cosmo.CmbMapmakingPlan(pixel_count)
    map_function = eqx.filter_jit(mapmaking.solve)
    _, map_compile = _measure(map_function, tod)
    map_result, map_steady = _measure(map_function, tod)

    report = {
        "observation_shape": list(matrix.shape),
        "observation_compile_seconds": observation_compile,
        "observation_steady_seconds": observation_steady,
        "observation_finite": bool(jnp.all(jnp.isfinite(observed.values))),
        "ewald_particles": int(positions.shape[0]),
        "ewald_compile_seconds": ewald_compile,
        "ewald_steady_seconds": ewald_steady,
        "ewald_finite": bool(ewald_result.successful),
        "map_pixels": pixel_count,
        "map_samples": int(pixels.size),
        "map_compile_seconds": map_compile,
        "map_steady_seconds": map_steady,
        "map_successful": bool(map_result.successful),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
