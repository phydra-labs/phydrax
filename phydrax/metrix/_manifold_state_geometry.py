#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._manifold import AbstractGeodesicManifold
from ._state_geometry import AbstractStateGeometry


class GeodesicManifoldStateGeometry(AbstractStateGeometry):
    """Equal-ambient four-space adapter for an exact geodesic manifold."""

    manifold: AbstractGeodesicManifold
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, manifold: AbstractGeodesicManifold, /):
        if not isinstance(manifold, AbstractGeodesicManifold):
            raise TypeError("manifold must be an AbstractGeodesicManifold.")
        self.manifold = manifold
        self.geometry_id = f"state-geometry:{manifold.manifold_id}:geodesic"
        self.retraction_method = "geodesic-exponential"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = manifold.transport_is_isometric
        self.supports_commutator_free = False

    def contains(self, state: ArrayLike, /) -> Array:
        return self.manifold.contains(state)

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        return self.manifold.project_tangent(state, vector)

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        return self.manifold.exp(
            state, self.manifold.project_tangent(state, local_tangent)
        )

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        return self.manifold.log(state, point)

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        anchor = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        direction = jnp.asarray(local_velocity)
        return jax.jvp(
            lambda value: self.retract(anchor, value),
            (local,),
            (direction,),
        )[1]

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        anchor = jnp.asarray(state)
        target = jnp.asarray(point)
        velocity = self.manifold.project_tangent(target, tangent)
        return jax.jvp(
            lambda value: self.inverse_retract(anchor, value),
            (target,),
            (velocity,),
        )[1]

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        anchor = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        target_cotangent = jnp.asarray(cotangent)
        return jax.linear_transpose(
            lambda direction: self.retraction_jvp(anchor, local, direction),
            jnp.zeros_like(local),
        )(target_cotangent)[0]

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        source = jnp.asarray(state)
        target = jnp.asarray(point)
        step = self.manifold.log(source, target)
        return self.manifold.transport(source, step, target, tangent)

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        source = jnp.asarray(state)
        target = jnp.asarray(point)
        target_cotangent = jnp.asarray(cotangent)
        return jax.linear_transpose(
            lambda tangent: self.transport_tangent(source, target, tangent),
            jnp.zeros_like(source),
        )(target_cotangent)[0]

    def cut_locus_margin(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        source = jnp.asarray(state)
        target = jnp.asarray(point)
        if source.shape != target.shape:
            raise ValueError("Geodesic chart points must have matching shapes.")
        return jnp.asarray(1.0, dtype=source.dtype)


__all__ = ["GeodesicManifoldStateGeometry"]
