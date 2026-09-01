"""Cold and warm benchmarks for native astrodynamics kernels."""

import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _timed(function, *args):
    start = time.perf_counter()
    value = function(*args)
    jax.block_until_ready(value)
    return value, time.perf_counter() - start


def main():
    astro = phx.applications.astrodynamics
    context = astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0, 0.0), "TT")),
        astro.FrameDefinition("central", "inertial", pseudo_inertial=True),
    )
    state = astro.CartesianOrbitState(
        jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([0.0, 1.0, 0.0]), context
    )
    batch = 4096
    elapsed = jnp.linspace(0.0, 2.0 * jnp.pi, batch)
    propagate = jax.jit(
        jax.vmap(
            lambda dt: astro.propagate_universal_kepler(state, dt, 1.0).state.packed()
        )
    )
    _, compile_seconds = _timed(propagate, elapsed)
    values, warm_seconds = _timed(propagate, elapsed)
    lambert = jax.jit(
        lambda tof: (
            astro.solve_lambert(
                jnp.asarray([1.0, 0.0, 0.0]),
                jnp.asarray([0.0, 1.0, 0.0]),
                tof,
                1.0,
                context,
                astro.LambertPlan(),
            ).departure_velocity
        )
    )
    _, lambert_compile = _timed(lambert, jnp.asarray(0.5 * jnp.pi))
    _, lambert_warm = _timed(lambert, jnp.asarray(0.5 * jnp.pi))
    report = {
        "kind": "astrodynamics-benchmark",
        "device": str(jax.devices()[0]),
        "dtype": str(values.dtype),
        "batch": batch,
        "universal_compile_seconds": compile_seconds,
        "universal_warm_seconds": warm_seconds,
        "lambert_compile_seconds": lambert_compile,
        "lambert_warm_seconds": lambert_warm,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
