#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._shallow_water import PreparedShallowWaterBathymetry


BalancedShallowWaterBackend: TypeAlias = Literal[
    "triangle-unstructured", "sbp", "global-spectral", "dgsem"
]
ScalarDerivative = Callable[[Array, int, Any], Array]


class PreparedBalancedShallowWaterLowering(StrictModule, NonTrainableState):
    """Backend-neutral equilibrium split bound to one discrete derivative."""

    gravity: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    bathymetry: PreparedShallowWaterBathymetry
    derivative: ScalarDerivative
    backend: BalancedShallowWaterBackend = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    lowering_id: str = eqx.field(static=True)

    def residual(self, state: ArrayLike, args: Any = None, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[:-1] != self.bathymetry.values.shape:
            raise ValueError("Balanced state does not match bathymetry support.")
        if value.shape[-1] != self.dimension + 1:
            raise ValueError("Balanced state has the wrong component count.")
        depth = value[..., 0]
        discharge = value[..., 1:]
        velocity = discharge / jnp.where(depth[..., None] > 0, depth[..., None], 1)
        velocity = jnp.where(depth[..., None] > 0, velocity, 0)
        # The prepared bed precision defines the discrete equilibrium manifold.
        surface = depth.astype(self.bathymetry.values.dtype) + self.bathymetry.values
        mass_rate = jnp.zeros_like(depth)
        momentum_rate = jnp.zeros_like(discharge)
        for axis in range(self.dimension):
            mass_rate = mass_rate - self.derivative(discharge[..., axis], axis, args)
            advective = discharge * velocity[..., axis, None]
            momentum_rate = momentum_rate - jnp.stack(
                tuple(
                    self.derivative(advective[..., component], axis, args)
                    for component in range(self.dimension)
                ),
                axis=-1,
            )
        pressure = jnp.stack(
            tuple(self.derivative(surface, axis, args) for axis in range(self.dimension)),
            axis=-1,
        )
        momentum_rate = momentum_rate - self.gravity * depth[..., None] * pressure
        return jnp.concatenate((mass_rate[..., None], momentum_rate), axis=-1)


def _lower(
    backend: BalancedShallowWaterBackend,
    bathymetry: PreparedShallowWaterBathymetry,
    derivative: ScalarDerivative,
    dimension: int,
    gravity: float,
    geometry_id: str,
) -> PreparedBalancedShallowWaterLowering:
    if not isinstance(bathymetry, PreparedShallowWaterBathymetry):
        raise TypeError("Balanced lowering requires prepared bathymetry.")
    if not callable(derivative):
        raise TypeError("Balanced lowering derivative must be callable.")
    dimension_ = int(dimension)
    gravity_ = float(gravity)
    geometry = str(geometry_id)
    if dimension_ not in (1, 2) or gravity_ <= 0 or not geometry:
        raise ValueError("Balanced lowering dimension/gravity/geometry is invalid.")
    if bathymetry.geometry_id != geometry:
        raise ValueError("Bathymetry belongs to a different backend geometry.")
    return PreparedBalancedShallowWaterLowering(
        gravity_,
        dimension_,
        bathymetry,
        derivative,
        backend,
        geometry,
        canonical_fingerprint(
            {
                "kind": "balanced-shallow-water-lowering",
                "backend": backend,
                "bathymetry": bathymetry.bed_id,
                "geometry": geometry,
                "dimension": dimension_,
                "gravity": gravity_,
            }
        ),
    )


def lower_triangle_unstructured_shallow_water(*args, **kwargs):
    return _lower("triangle-unstructured", *args, **kwargs)


def lower_sbp_shallow_water(*args, **kwargs):
    return _lower("sbp", *args, **kwargs)


def lower_global_spectral_shallow_water(*args, **kwargs):
    return _lower("global-spectral", *args, **kwargs)


def lower_dgsem_shallow_water(*args, **kwargs):
    return _lower("dgsem", *args, **kwargs)


__all__ = [
    "BalancedShallowWaterBackend",
    "PreparedBalancedShallowWaterLowering",
    "lower_dgsem_shallow_water",
    "lower_global_spectral_shallow_water",
    "lower_sbp_shallow_water",
    "lower_triangle_unstructured_shallow_water",
]
