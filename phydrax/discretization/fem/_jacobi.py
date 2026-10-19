from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def jacobi_jax(degree: int, alpha: int, beta: int, points: Array) -> Array:

    previous = jnp.ones_like(points)
    if degree == 0:
        return previous
    x = 2.0 * points - 1.0
    current = 0.5 * (alpha - beta + (alpha + beta + 2) * x)
    for n in range(2, degree + 1):
        ab = alpha + beta
        numerator = (
            (2 * n + ab - 1)
            * ((2 * n + ab) * (2 * n + ab - 2) * x + alpha * alpha - beta * beta)
            * current
        )
        numerator -= 2 * (n + alpha - 1) * (n + beta - 1) * (2 * n + ab) * previous
        following = numerator / (2 * n * (n + ab) * (2 * n + ab - 2))
        previous, current = current, following
    return current


def jacobi_derivative_jax(degree: int, alpha: int, beta: int, points: Array) -> Array:

    return (
        jnp.zeros_like(points)
        if degree == 0
        else (degree + alpha + beta + 1)
        * jacobi_jax(degree - 1, alpha + 1, beta + 1, points)
    )
