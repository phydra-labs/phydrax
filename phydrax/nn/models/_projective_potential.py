#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class ProjectiveInvariantPotential(StrictModule):
    """Real scalar potential over normalized Hermitian-projector features."""

    network: eqx.nn.MLP
    homogeneous_dimension: int = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)

    def __init__(
        self,
        homogeneous_dimension: int,
        key: Array,
        /,
        *,
        width: int = 32,
        depth: int = 2,
        potential_id: str = "projective-invariant-potential",
    ):
        dimension = int(homogeneous_dimension)
        if dimension < 2:
            raise ValueError("homogeneous_dimension must be at least two.")
        if int(width) < 1 or int(depth) < 1:
            raise ValueError("width and depth must be positive.")
        identifier = str(potential_id)
        if not identifier:
            raise ValueError("potential_id must be non-empty.")
        feature_dimension = 2 * dimension * dimension
        self.network = eqx.nn.MLP(
            feature_dimension,
            1,
            int(width),
            int(depth),
            activation=jax.nn.tanh,
            final_activation=lambda value: value,
            key=key,
        )
        self.homogeneous_dimension = dimension
        self.potential_id = identifier

    def features(self, homogeneous_point: ArrayLike, /) -> Array:
        point = jnp.asarray(homogeneous_point)
        expected = (self.homogeneous_dimension,)
        if point.shape != expected:
            raise ValueError(f"Homogeneous point must have shape {expected}.")
        norm_squared = jnp.real(jnp.vdot(point, point))
        projector = point[:, None] * jnp.conj(point[None, :]) / norm_squared
        return jnp.concatenate((jnp.real(projector).ravel(), jnp.imag(projector).ravel()))

    def __call__(self, homogeneous_point: ArrayLike, /) -> Array:
        return jnp.asarray(self.network(self.features(homogeneous_point))).reshape(())

    def invariance_residual(
        self,
        homogeneous_point: ArrayLike,
        /,
        *,
        scale: complex = 0.7 + 0.4j,
    ) -> Array:
        point = jnp.asarray(homogeneous_point)
        return jnp.abs(self(scale * point) - self(point))


__all__ = ["ProjectiveInvariantPotential"]
