#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measured contact distance, gradient, and HVP microbenchmarks."""

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _timed(function, argument, repeats=20):
    compiled = eqx.filter_jit(function)
    started = time.perf_counter()
    value = compiled(argument)
    jax.block_until_ready(value)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        value = compiled(argument)
    jax.block_until_ready(value)
    return value, compile_seconds, (time.perf_counter() - started) / repeats


def main():
    count = 4096
    parameter = jnp.linspace(0.05, 0.95, count)
    points = jnp.stack(
        (parameter, 0.25 * parameter, jnp.full_like(parameter, 0.1)), axis=-1
    )
    triangle = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    def energy(value):
        distance = phx.discretization.point_triangle_distance(
            value,
            jnp.broadcast_to(triangle[0], value.shape),
            jnp.broadcast_to(triangle[1], value.shape),
            jnp.broadcast_to(triangle[2], value.shape),
        )
        return jnp.sum(distance.squared_distance)

    values, value_compile, value_seconds = _timed(
        lambda value: (
            phx.discretization.point_triangle_distance(
                value,
                jnp.broadcast_to(triangle[0], value.shape),
                jnp.broadcast_to(triangle[1], value.shape),
                jnp.broadcast_to(triangle[2], value.shape),
            ).squared_distance
        ),
        points,
    )
    gradient, gradient_compile, gradient_seconds = _timed(jax.grad(energy), points)
    direction = jnp.ones_like(points) / jnp.sqrt(points.size)
    hvp, hvp_compile, hvp_seconds = _timed(
        lambda value: jax.jvp(jax.grad(energy), (value,), (direction,))[1],
        points,
    )
    print(
        json.dumps(
            {
                "benchmark": "contact-primitives",
                "device": str(jax.devices()[0]),
                "dtype": str(points.dtype),
                "batch_size": count,
                "value_compile_seconds": value_compile,
                "value_seconds": value_seconds,
                "gradient_compile_seconds": gradient_compile,
                "gradient_seconds": gradient_seconds,
                "hvp_compile_seconds": hvp_compile,
                "hvp_seconds": hvp_seconds,
                "minimum_squared_distance": float(jnp.min(values)),
                "gradient_norm": float(jnp.sqrt(jnp.sum(gradient * gradient))),
                "hvp_norm": float(jnp.sqrt(jnp.sum(hvp * hvp))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
