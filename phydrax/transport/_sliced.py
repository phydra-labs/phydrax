#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._univariate import _probabilities, _validate_p, _wasserstein_cost_1d


class SlicedWassersteinResult(StrictModule):
    """Finite-projection sliced Wasserstein estimate and projection provenance."""

    value: Array
    projection_distances: Array
    projections: Array
    p: float = eqx.field(static=True)
    sampling: str = eqx.field(static=True)

    @property
    def num_projections(self) -> int:
        return int(self.projections.shape[0])


def sliced_wasserstein_distance(
    source: ArrayLike,
    target: ArrayLike,
    /,
    *,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    p: float = 2.0,
    num_projections: int = 128,
    key: Array | None = None,
    projections: ArrayLike | None = None,
) -> SlicedWassersteinResult:
    """Estimate sliced Wasserstein distance using exact projected transport."""
    source_points = _events(source, name="source")
    target_points = _events(target, name="target")
    if source_points.shape[1] != target_points.shape[1]:
        raise ValueError("Source and target events must have equal feature size.")
    exponent = _validate_p(p)
    source_probabilities = _probabilities(
        source_weights,
        source_points.shape[0],
        name="source_weights",
        dtype=source_points.dtype,
    )
    target_probabilities = _probabilities(
        target_weights,
        target_points.shape[0],
        name="target_weights",
        dtype=target_points.dtype,
    )
    if projections is None:
        count = int(num_projections)
        if count < 1:
            raise ValueError("num_projections must be positive.")
        if key is None:
            raise ValueError("A PRNG key is required when projections are not supplied.")
        directions = jr.normal(
            key,
            (count, source_points.shape[1]),
            dtype=source_points.dtype,
        )
        sampling = "random-normal"
    else:
        if key is not None:
            raise ValueError("Supply either key or projections, not both.")
        directions = jnp.asarray(projections, dtype=source_points.dtype)
        if directions.ndim != 2 or directions.shape[0] == 0:
            raise ValueError("projections must be a nonempty rank-two array.")
        if directions.shape[1] != source_points.shape[1]:
            raise ValueError("Projection feature size must match event feature size.")
        sampling = "explicit"
    directions = eqx.error_if(
        directions,
        jnp.any(~jnp.isfinite(directions)),
        "Projection directions must be finite.",
    )
    norms = jnp.linalg.norm(directions, axis=1)
    norms = eqx.error_if(
        norms,
        jnp.any(~jnp.isfinite(norms)) | jnp.any(norms <= 0.0),
        "Projection directions must have finite positive norm.",
    )
    directions = directions / norms[:, None]
    source_projected = source_points @ directions.T
    target_projected = target_points @ directions.T

    def projection_cost(source_values, target_values):
        return _wasserstein_cost_1d(
            source_values,
            target_values,
            source_probabilities,
            target_probabilities,
            p=exponent,
        )

    costs = jax.vmap(projection_cost, in_axes=(1, 1))(
        source_projected,
        target_projected,
    )
    distances = costs ** (1.0 / exponent)
    value = jnp.mean(costs) ** (1.0 / exponent)
    return SlicedWassersteinResult(
        value=value,
        projection_distances=distances,
        projections=directions,
        p=exponent,
        sampling=sampling,
    )


def _events(values: ArrayLike, /, *, name: str) -> Array:
    result = jnp.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must have nonempty shape (sample, feature).")
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)),
        f"{name} must contain only finite events.",
    )


__all__ = ["SlicedWassersteinResult", "sliced_wasserstein_distance"]
