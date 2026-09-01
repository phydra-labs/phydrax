"""Cold and warm benchmarks for astronomy closure kernels."""

import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _timed(function, *args):
    start = time.perf_counter()
    result = function(*args)
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def main():
    solver = phx.solver.IAS15Plan(relative_tolerance=1e-9, absolute_tolerance=1e-11)
    integrate = jax.jit(
        lambda position, velocity: (
            solver.solve(
                lambda time, q, v, args: -q,
                position,
                velocity,
                jnp.linspace(0.0, 1.0, 17),
            ).position
        )
    )
    _, ias_compile = _timed(integrate, jnp.asarray([1.0]), jnp.asarray([0.0]))
    _, ias_warm = _timed(integrate, jnp.asarray([1.0]), jnp.asarray([0.0]))

    wcs = phx.applications.astrophysics.TangentSipWcsPlan(
        jnp.asarray([1.0, 0.5]),
        jnp.asarray([100.0, 100.0]),
        jnp.eye(2),
        jnp.zeros((2, 2)),
        jnp.zeros((2, 2)),
    )
    sky = jnp.stack(
        (jnp.linspace(0.99, 1.01, 65536), jnp.linspace(0.49, 0.51, 65536)),
        axis=-1,
    )
    project = jax.jit(jax.vmap(lambda value: wcs.world_to_pixel(value).coordinates))
    _, wcs_compile = _timed(project, sky)
    pixels, wcs_warm = _timed(project, sky)
    report = {
        "kind": "astronomy-closure-benchmark",
        "device": str(jax.devices()[0]),
        "ias15_compile_seconds": ias_compile,
        "ias15_warm_seconds": ias_warm,
        "wcs_points": int(sky.shape[0]),
        "wcs_compile_seconds": wcs_compile,
        "wcs_warm_seconds": wcs_warm,
        "wcs_finite_fraction": float(jnp.mean(jnp.isfinite(pixels))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
