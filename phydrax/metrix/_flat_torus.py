#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ._state_geometry import AbstractStateGeometry


def _same_shape(value: Array, reference: Array, role: str, /) -> None:
    if value.shape != reference.shape:
        raise ValueError(f"{role} must have shape {reference.shape}; got {value.shape}.")


class FlatTorusStateGeometry(AbstractStateGeometry):
    """Principal-coordinate geometry for a flat torus with one common period."""

    period: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, period: float, /, *, geometry_id: str | None = None):
        value = float(period)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("period must be finite and positive.")
        self.period = value
        self.geometry_id = (
            canonical_fingerprint({"kind": "flat-torus-geometry", "period": value})
            if geometry_id is None
            else str(geometry_id)
        )
        if not self.geometry_id:
            raise ValueError("geometry_id must be non-empty.")
        self.retraction_method = "wrapped-addition"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def wrap(self, value: ArrayLike, /) -> Array:
        """Map real coordinates into ``[-period / 2, period / 2)``."""
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("Flat-torus coordinates must be real floating arrays.")
        return jnp.mod(array + 0.5 * self.period, self.period) - 0.5 * self.period

    def contains(self, state: ArrayLike, /) -> Array:
        array = jnp.asarray(state)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            return jnp.asarray(False)
        half = 0.5 * self.period
        return (
            jnp.all(jnp.isfinite(array)) & jnp.all(array >= -half) & jnp.all(array < half)
        )

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        state_array = jnp.asarray(state)
        vector_array = jnp.asarray(vector)
        _same_shape(vector_array, state_array, "Flat-torus tangent")
        return vector_array

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        _same_shape(local, state_array, "Flat-torus local tangent")
        return self.wrap(state_array + local)

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        _same_shape(point_array, state_array, "Flat-torus retraction point")
        return self.wrap(point_array - state_array)

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        velocity = jnp.asarray(local_velocity)
        _same_shape(local, state_array, "Flat-torus local tangent")
        _same_shape(velocity, local, "Flat-torus local velocity")
        return velocity

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        vector = jnp.asarray(tangent)
        _same_shape(point_array, state_array, "Flat-torus retraction point")
        _same_shape(vector, state_array, "Flat-torus tangent")
        return vector

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        covector = jnp.asarray(cotangent)
        _same_shape(local, state_array, "Flat-torus local tangent")
        _same_shape(covector, state_array, "Flat-torus cotangent")
        return covector

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        vector = jnp.asarray(tangent)
        _same_shape(point_array, state_array, "Flat-torus transport point")
        _same_shape(vector, state_array, "Flat-torus tangent")
        return vector

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        covector = jnp.asarray(cotangent)
        _same_shape(point_array, state_array, "Flat-torus transport point")
        _same_shape(covector, state_array, "Flat-torus cotangent")
        return covector

    def cut_locus_margin(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        _same_shape(point_array, state_array, "Flat-torus chart point")
        displacement = self.inverse_retract(state_array, point_array)
        return jnp.asarray(0.5 * self.period, dtype=state_array.dtype) - jnp.max(
            jnp.abs(displacement)
        )


__all__ = ["FlatTorusStateGeometry"]
