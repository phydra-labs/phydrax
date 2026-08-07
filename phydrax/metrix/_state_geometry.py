#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _same_shape(value: Array, reference: Array, name: str, /) -> None:
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must preserve state shape {reference.shape}; got {value.shape}."
        )


class AbstractStateGeometry(StrictModule):
    """Retraction geometry for an array-valued differential-equation state.

    Vector fields keep the ordinary ``(time, state, args) -> state-shaped array``
    contract. A geometry projects those ambient arrays onto the tangent space and
    expresses them in local, state-shaped coordinates used by geometric solvers.
    """

    geometry_id: AbstractAttribute[str]
    retraction_method: AbstractAttribute[str]
    trivial: AbstractAttribute[bool]

    @abstractmethod
    def contains(self, state: ArrayLike, /) -> Array:
        """Return one scalar boolean indicating membership in the state space."""
        raise NotImplementedError

    @abstractmethod
    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        """Project a state-shaped ambient vector onto the tangent space at state."""
        raise NotImplementedError

    @abstractmethod
    def to_local(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        """Express a tangent vector in state-shaped local coordinates."""
        raise NotImplementedError

    @abstractmethod
    def from_local(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        """Convert state-shaped local coordinates to an ambient tangent vector."""
        raise NotImplementedError

    @abstractmethod
    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        """Map local coordinates at state back onto the state space."""
        raise NotImplementedError

    @abstractmethod
    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        """Return local coordinates at state for a nearby point."""
        raise NotImplementedError

    @abstractmethod
    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        """Pull a tangent at ``retract(state, local_tangent)`` into local velocity."""
        raise NotImplementedError

    def local_retraction(self, state: ArrayLike, /) -> LocalRetraction:
        """Bind this geometry's retraction and pullback to one base point."""
        return LocalRetraction(self, state)

    def interpolate(
        self,
        left: ArrayLike,
        right: ArrayLike,
        weight: ArrayLike,
        /,
    ) -> Array:
        """Interpolate on the state space along the local retraction from left."""
        left_array = jnp.asarray(left)
        right_array = jnp.asarray(right)
        _same_shape(right_array, left_array, "Interpolation endpoint")
        local = self.inverse_retract(left_array, right_array)
        return self.retract(left_array, jnp.asarray(weight) * local)


class LocalRetraction(StrictModule):
    """A geometry retraction bound to a validated base point.

    ``evaluate`` maps state-shaped local coordinates to the state space;
    ``pullback`` converts an ambient tangent at that point into the derivative of
    those local coordinates. The identifiers record the resolved geometry method.
    """

    geometry: AbstractStateGeometry
    base_point: Array
    retraction_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(self, geometry: AbstractStateGeometry, base_point: ArrayLike, /):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("LocalRetraction geometry must be an AbstractStateGeometry.")
        base = jnp.asarray(base_point)
        membership = jnp.asarray(geometry.contains(base), dtype=bool)
        if membership.shape != ():
            raise ValueError("State geometry contains() must return a scalar boolean.")
        base = eqx.error_if(
            base,
            ~membership,
            "LocalRetraction base point is outside the state geometry.",
        )
        self.geometry = geometry
        self.base_point = base
        self.retraction_id = f"{geometry.geometry_id}:local-retraction"
        self.resolved_method = geometry.retraction_method

    def evaluate(self, local_tangent: ArrayLike, /) -> Array:
        local = jnp.asarray(local_tangent)
        _same_shape(local, self.base_point, "Local retraction coordinates")
        point = jnp.asarray(self.geometry.retract(self.base_point, local))
        _same_shape(point, self.base_point, "Local retraction")
        return point

    def __call__(self, local_tangent: ArrayLike, /) -> Array:
        return self.evaluate(local_tangent)

    def pullback(
        self,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        local = jnp.asarray(local_tangent)
        vector = jnp.asarray(tangent)
        _same_shape(local, self.base_point, "Local retraction coordinates")
        _same_shape(vector, self.base_point, "Retraction tangent")
        velocity = jnp.asarray(
            self.geometry.pullback(self.base_point, local, vector)
        )
        _same_shape(velocity, self.base_point, "Local retraction pullback")
        return velocity


class EuclideanStateGeometry(AbstractStateGeometry):
    """Identity geometry for unconstrained array-valued states."""

    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_id: str = "state-geometry:euclidean",
    ):
        self.geometry_id = _identifier(geometry_id, "geometry_id")
        self.retraction_method = "addition"
        self.trivial = True

    def contains(self, state: ArrayLike, /) -> Array:
        return jnp.all(jnp.isfinite(jnp.asarray(state)))

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        vector_array = jnp.asarray(vector)
        _same_shape(vector_array, state_array, "Euclidean tangent")
        return vector_array

    def to_local(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(state, tangent)

    def from_local(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(state, local_tangent)

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = self.from_local(state_array, local_tangent)
        return state_array + local

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        _same_shape(point_array, state_array, "Euclidean retraction point")
        return point_array - state_array

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        _same_shape(local, state_array, "Euclidean local tangent")
        return self.to_local(state_array + local, tangent)


__all__ = [
    "AbstractStateGeometry",
    "EuclideanStateGeometry",
    "LocalRetraction",
]
