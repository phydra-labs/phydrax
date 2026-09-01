"""Cold and warm benchmarks for transit photometry."""

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
    physics = phx.applications.astrophysics
    occultation = physics.CircularOccultationPlan(
        physics.PolynomialLimbDarkenedDisk(jnp.asarray([0.3, 0.2]))
    )
    separation = jnp.linspace(0.0, 1.2, 65536)
    evaluate = jax.jit(lambda values: occultation.evaluate(values, 0.1).relative_flux)
    _, compile_seconds = _timed(evaluate, separation)
    flux, warm_seconds = _timed(evaluate, separation)
    gradient = jax.jit(jax.grad(lambda values: jnp.sum(evaluate(values))))
    _, gradient_compile = _timed(gradient, separation)
    derivative, gradient_warm = _timed(gradient, separation)
    report = {
        "kind": "astrophysics-transit-benchmark",
        "device": str(jax.devices()[0]),
        "dtype": str(flux.dtype),
        "cadences": int(separation.size),
        "compile_seconds": compile_seconds,
        "warm_seconds": warm_seconds,
        "gradient_compile_seconds": gradient_compile,
        "gradient_warm_seconds": gradient_warm,
        "finite_gradient_fraction": float(jnp.mean(jnp.isfinite(derivative))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
