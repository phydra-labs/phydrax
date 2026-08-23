#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...metrix import AbstractGeodesicManifold
from ._interpolant import AbstractEndpointInterpolant, EndpointInterpolantEvaluation


class GeodesicEndpointInterpolant(AbstractEndpointInterpolant):
    """Exact endpoint geodesic inside one convex normal region."""

    geometry: AbstractGeodesicManifold
    event_shape: tuple[int, ...] = eqx.field(static=True)
    source_coordinate: Array
    target_coordinate: Array
    interpolant_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: AbstractGeodesicManifold,
        /,
        *,
        source_coordinate: ArrayLike = 0.0,
        target_coordinate: ArrayLike = 1.0,
        interpolant_id: str | None = None,
    ):
        if not isinstance(geometry, AbstractGeodesicManifold):
            raise TypeError("geometry must be an AbstractGeodesicManifold.")
        source = jnp.asarray(source_coordinate, dtype=float).reshape(())
        target = jnp.asarray(target_coordinate, dtype=float).reshape(())
        if not bool(jnp.isfinite(source) & jnp.isfinite(target) & (target > source)):
            raise ValueError(
                "Geodesic interpolant coordinates must be finite and ordered."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "geodesic-endpoint-interpolant-v1",
                    "geometry": geometry.manifold_id,
                    "source_coordinate": float(source),
                    "target_coordinate": float(target),
                }
            )
            if interpolant_id is None
            else str(interpolant_id)
        )
        if not identifier:
            raise ValueError("interpolant_id must be non-empty.")
        self.geometry = geometry
        self.event_shape = geometry.point_shape
        self.source_coordinate = source
        self.target_coordinate = target
        self.interpolant_id = identifier

    def evaluate(
        self,
        time: ArrayLike,
        source: ArrayLike,
        target: ArrayLike,
        /,
    ) -> EndpointInterpolantEvaluation:
        source_array = jnp.asarray(source)
        target_array = jnp.asarray(target, dtype=source_array.dtype)
        if source_array.shape != target_array.shape:
            raise ValueError("Geodesic source and target shapes must match.")
        rank = len(self.event_shape)
        if source_array.ndim < rank or source_array.shape[-rank:] != self.event_shape:
            raise ValueError("Geodesic endpoints must match the manifold point shape.")
        leading = source_array.shape[:-rank] if rank else source_array.shape
        time_array = jnp.asarray(time, dtype=source_array.real.dtype)
        if time_array.shape == ():
            time_array = jnp.broadcast_to(time_array, leading)
        elif time_array.shape != leading:
            raise ValueError(
                "Geodesic time must be scalar or match endpoint leading axes."
            )
        time_array = eqx.error_if(
            time_array,
            jnp.any(
                (time_array < self.source_coordinate)
                | (time_array > self.target_coordinate)
            ),
            "Geodesic interpolation time lies outside its coordinate interval.",
        )
        count = prod(leading) if leading else 1
        flat_source = source_array.reshape((count,) + self.event_shape)
        flat_target = target_array.reshape((count,) + self.event_shape)
        flat_time = time_array.reshape((count,))
        duration = self.target_coordinate - self.source_coordinate

        def evaluate_one(start: Array, end: Array, coordinate: Array):
            logarithm = self.geometry.log(start, end)
            weight = (coordinate - self.source_coordinate) / duration

            def path(local_weight: Array) -> Array:
                return self.geometry.exp(start, local_weight * logarithm)

            state, velocity = jax.jvp(path, (weight,), (1.0 / duration,))
            finite = (
                jnp.all(jnp.isfinite(start))
                & jnp.all(jnp.isfinite(end))
                & jnp.all(jnp.isfinite(state))
                & jnp.all(jnp.isfinite(velocity))
            )
            return state, velocity, finite

        state, velocity, valid = jax.vmap(evaluate_one)(
            flat_source, flat_target, flat_time
        )
        return EndpointInterpolantEvaluation(
            time=time_array,
            state=state.reshape(leading + self.event_shape),
            conditional_velocity=velocity.reshape(leading + self.event_shape),
            valid=valid.reshape(leading),
            event_shape=self.event_shape,
            interpolant_id=self.interpolant_id,
        )


__all__ = ["GeodesicEndpointInterpolant"]
