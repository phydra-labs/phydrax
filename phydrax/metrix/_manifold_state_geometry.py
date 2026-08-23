#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, ArrayLike

from ._manifold import AbstractGeodesicManifold
from ._state_geometry import AbstractStateGeometry


class GeodesicManifoldStateGeometry(AbstractStateGeometry):
    """Ambient-tangent state adapter for an exact geodesic manifold."""

    manifold: AbstractGeodesicManifold
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, manifold: AbstractGeodesicManifold, /):
        if not isinstance(manifold, AbstractGeodesicManifold):
            raise TypeError("manifold must be an AbstractGeodesicManifold.")
        self.manifold = manifold
        self.geometry_id = f"state-geometry:{manifold.manifold_id}:geodesic"
        self.retraction_method = "geodesic-exponential"
        self.trivial = False
        self.supports_exact_pullback = False
        self.supports_commutator_free = False

    def contains(self, state: ArrayLike, /) -> Array:
        return self.manifold.contains(state)

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        return self.manifold.project_tangent(state, vector)

    def to_local(self, state: ArrayLike, tangent: ArrayLike, /) -> Array:
        return self.manifold.project_tangent(state, tangent)

    def from_local(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        return self.manifold.project_tangent(state, local_tangent)

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        return self.manifold.exp(state, self.to_local(state, local_tangent))

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        return self.manifold.log(state, point)

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        del state, local_tangent, tangent
        raise ValueError(
            "GeodesicManifoldStateGeometry does not claim an exact exp pullback."
        )


__all__ = ["GeodesicManifoldStateGeometry"]
