#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._manifold import AbstractGeodesicManifold


class FrechetMeanResult(StrictModule):
    """Fixed-iteration intrinsic mean and final tangent residual."""

    point: Array
    residual_norm: Array
    precision_evidence: PrecisionEvidenceEnvelope
    iterations: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        point: ArrayLike,
        residual_norm: ArrayLike,
        /,
        *,
        iterations: int,
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
    ):
        point_ = jnp.asarray(point)
        evidence = (
            GeometryPrecisionPolicy().evidence_for(point_)
            if precision_evidence is None
            else precision_evidence
        )
        if not isinstance(evidence, PrecisionEvidenceEnvelope):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        self.point = point_
        self.residual_norm = jnp.asarray(residual_norm)
        self.precision_evidence = evidence
        self.iterations = int(iterations)
        self.method_id = "fixed-karcher-mean"


def _points_and_weights(
    geometry: AbstractGeodesicManifold,
    points: ArrayLike,
    weights: ArrayLike | None,
    precision: GeometryPrecisionPolicy,
    /,
) -> tuple[Array, Array]:
    values = jnp.asarray(points)
    expected = geometry.point_shape
    rank = len(expected)
    if values.ndim != rank + 1 or values.shape[1:] != expected:
        raise ValueError(
            f"Intrinsic samples must have shape (samples, {expected}); got {values.shape}."
        )
    precision.validate_coordinates(values)
    if weights is None:
        probabilities = jnp.full(
            (values.shape[0],),
            1.0 / float(values.shape[0]),
            dtype=values.real.dtype,
        )
        probabilities = precision.decision(probabilities)
    else:
        probabilities = precision.decision(jnp.asarray(weights, dtype=values.real.dtype))
        if probabilities.shape != (values.shape[0],):
            raise ValueError("Intrinsic sample weights must match the sample axis.")
        if not bool(jnp.all(jnp.isfinite(probabilities) & (probabilities >= 0))):
            raise ValueError("Intrinsic sample weights must be finite and nonnegative.")
        total = precision.sum(probabilities)
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
    precision: GeometryPrecisionPolicy | None = None,
) -> Array:
    """Return one half of the weighted squared-distance objective."""
    if not isinstance(geometry, AbstractGeodesicManifold):
        raise TypeError("geometry must be an AbstractGeodesicManifold.")
    precision_ = GeometryPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, GeometryPrecisionPolicy):
        raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
    values, probabilities = _points_and_weights(
        geometry,
        points,
        weights,
        precision_,
    )
    candidate_ = precision_.compute(candidate)
    computed_values = precision_.compute(values)
    distances = jax.vmap(lambda point: geometry.squared_distance(candidate_, point))(
        computed_values
    )
    return precision_.decision(
        0.5 * precision_.sum(probabilities * precision_.accumulation(distances))
    )


def frechet_mean(
    geometry: AbstractGeodesicManifold,
    points: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    initial: ArrayLike | None = None,
    iterations: int = 16,
    step_size: float = 1.0,
    precision: GeometryPrecisionPolicy | None = None,
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
    precision_ = GeometryPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, GeometryPrecisionPolicy):
        raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
    values, probabilities = _points_and_weights(
        geometry,
        points,
        weights,
        precision_,
    )
    start = values[0] if initial is None else jnp.asarray(initial)
    if start.shape != geometry.point_shape:
        raise ValueError(
            f"Initial intrinsic mean must have shape {geometry.point_shape}."
        )
    if start.dtype != values.dtype:
        raise TypeError("Initial intrinsic mean and samples must have one dtype.")
    computed_values = precision_.compute(values)

    def average_log(candidate: Array) -> Array:
        candidate_ = precision_.compute(candidate)
        logs = jax.vmap(lambda point: geometry.log(candidate_, point))(computed_values)
        reshape = (values.shape[0],) + (1,) * len(geometry.point_shape)
        weighted = precision_.accumulation(probabilities.reshape(reshape)) * (
            precision_.accumulation(logs)
        )
        return precision_.sum(weighted, axis=0)

    def update(_, candidate: Array) -> Array:
        next_candidate = geometry.exp(
            precision_.compute(candidate),
            step * average_log(candidate),
        )
        return jnp.asarray(next_candidate, dtype=candidate.dtype)

    point = jax.lax.fori_loop(0, count, update, start)
    residual = average_log(point)
    residual_norm = precision_.decision(
        geometry.norm(precision_.compute(point), residual)
    )
    return FrechetMeanResult(
        precision_.output(point),
        residual_norm,
        iterations=count,
        precision_evidence=precision_.evidence_for(values),
    )


__all__ = ["FrechetMeanResult", "frechet_mean", "frechet_objective"]
