#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._manifold import AbstractGeodesicManifold


class FrechetMeanResult(StrictModule):
    """Fixed-iteration intrinsic mean and final tangent residual."""

    point: Array
    residual_norm: Array
    iterations: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        point: ArrayLike,
        residual_norm: ArrayLike,
        /,
        *,
        iterations: int,
    ):
        self.point = jnp.asarray(point)
        self.residual_norm = jnp.asarray(residual_norm)
        self.iterations = int(iterations)
        self.method_id = "fixed-karcher-mean"


def _points_and_weights(
    geometry: AbstractGeodesicManifold,
    points: ArrayLike,
    weights: ArrayLike | None,
    /,
) -> tuple[Array, Array]:
    values = jnp.asarray(points)
    expected = geometry.point_shape
    rank = len(expected)
    if values.ndim != rank + 1 or values.shape[1:] != expected:
        raise ValueError(
            f"Intrinsic samples must have shape (samples, {expected}); got {values.shape}."
        )
    if weights is None:
        probabilities = jnp.full(
            (values.shape[0],),
            1.0 / float(values.shape[0]),
            dtype=values.real.dtype,
        )
    else:
        probabilities = jnp.asarray(weights, dtype=values.real.dtype)
        if probabilities.shape != (values.shape[0],):
            raise ValueError("Intrinsic sample weights must match the sample axis.")
        if not bool(jnp.all(jnp.isfinite(probabilities) & (probabilities >= 0))):
            raise ValueError("Intrinsic sample weights must be finite and nonnegative.")
        total = jnp.sum(probabilities)
        if not bool(total > 0):
            raise ValueError("Intrinsic sample weights must have positive mass.")
        probabilities = probabilities / total
    return values, probabilities


def frechet_objective(
    geometry: AbstractGeodesicManifold,
    candidate: ArrayLike,
    points: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
) -> Array:
    """Return one half of the weighted squared-distance objective."""
    if not isinstance(geometry, AbstractGeodesicManifold):
        raise TypeError("geometry must be an AbstractGeodesicManifold.")
    values, probabilities = _points_and_weights(geometry, points, weights)
    distances = jax.vmap(lambda point: geometry.squared_distance(candidate, point))(
        values
    )
    return 0.5 * jnp.sum(probabilities * distances)


def frechet_mean(
    geometry: AbstractGeodesicManifold,
    points: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    initial: ArrayLike | None = None,
    iterations: int = 16,
    step_size: float = 1.0,
) -> FrechetMeanResult:
    """Compute a fixed-iteration Karcher mean inside one convex normal region."""
    if not isinstance(geometry, AbstractGeodesicManifold):
        raise TypeError("geometry must be an AbstractGeodesicManifold.")
    count = int(iterations)
    if count < 0:
        raise ValueError("iterations must be non-negative.")
    step = float(step_size)
    if not (0.0 < step <= 1.0):
        raise ValueError("step_size must lie in (0, 1].")
    values, probabilities = _points_and_weights(geometry, points, weights)
    start = values[0] if initial is None else jnp.asarray(initial)
    if start.shape != geometry.point_shape:
        raise ValueError(
            f"Initial intrinsic mean must have shape {geometry.point_shape}."
        )

    def average_log(candidate: Array) -> Array:
        logs = jax.vmap(lambda point: geometry.log(candidate, point))(values)
        reshape = (values.shape[0],) + (1,) * len(geometry.point_shape)
        return jnp.sum(probabilities.reshape(reshape) * logs, axis=0)

    def update(_, candidate: Array) -> Array:
        return geometry.exp(candidate, step * average_log(candidate))

    point = jax.lax.fori_loop(0, count, update, start)
    residual = average_log(point)
    return FrechetMeanResult(
        point,
        geometry.norm(point, residual),
        iterations=count,
    )


__all__ = ["FrechetMeanResult", "frechet_mean", "frechet_objective"]
