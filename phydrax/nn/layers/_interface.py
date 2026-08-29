#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ...geometry import regularized_heaviside_values
from .._base import _AbstractBaseModel
from .._keys import EvalKey


InterfaceDistanceSemantics = Literal["level_set", "signed_distance"]


class InterfaceFeatureLift(_AbstractBaseModel):
    """Augment coordinates with interface distance, cusp, and side features.

    The lift keeps interface location out of downstream network weights. A general
    level set is locally normalized by its spatial gradient; a certified signed
    distance can be used directly. The compact side feature uses the same cosine
    regularization as the public level-set calculus.
    """

    level_set: Callable[[Array], Array]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    distance_semantics: InterfaceDistanceSemantics = eqx.field(static=True)
    distance_clip: float = eqx.field(static=True)
    side_width: float = eqx.field(static=True)
    gradient_floor: float = eqx.field(static=True)
    include_coordinates: bool = eqx.field(static=True)
    include_signed_distance: bool = eqx.field(static=True)
    include_cusp: bool = eqx.field(static=True)
    include_side: bool = eqx.field(static=True)

    def __init__(
        self,
        level_set: Callable[[Array], Array],
        in_size: int,
        /,
        *,
        distance_semantics: InterfaceDistanceSemantics = "level_set",
        distance_clip: float = 1.0,
        side_width: float = 0.1,
        gradient_floor: float = 1.0e-12,
        include_coordinates: bool = True,
        include_signed_distance: bool = True,
        include_cusp: bool = True,
        include_side: bool = True,
    ):
        if not callable(level_set):
            raise TypeError("level_set must be callable.")
        dimension = int(in_size)
        if dimension <= 0:
            raise ValueError("in_size must be positive.")
        if distance_semantics not in ("level_set", "signed_distance"):
            raise ValueError(
                "distance_semantics must be 'level_set' or 'signed_distance'."
            )
        clip = _positive_finite(distance_clip, "distance_clip")
        width = _positive_finite(side_width, "side_width")
        floor = _positive_finite(gradient_floor, "gradient_floor")
        flags = (
            bool(include_coordinates),
            bool(include_signed_distance),
            bool(include_cusp),
            bool(include_side),
        )
        if not any(flags):
            raise ValueError("InterfaceFeatureLift must emit at least one feature.")
        extra = sum(flags[1:])
        self.level_set = level_set
        self.in_size = dimension
        self.out_size = (dimension if flags[0] else 0) + extra
        self.distance_semantics = distance_semantics
        self.distance_clip = clip
        self.side_width = width
        self.gradient_floor = floor
        (
            self.include_coordinates,
            self.include_signed_distance,
            self.include_cusp,
            self.include_side,
        ) = flags

    def _scalar_level_set(self, point: Array, key: EvalKey, /) -> Array:
        value = (
            self.level_set(point, key=key)
            if isinstance(self.level_set, AbstractArrayModel)
            else self.level_set(point)
        )
        scalar = jnp.asarray(value)
        if scalar.shape != () or jnp.iscomplexobj(scalar):
            raise ValueError("level_set must return one real scalar per coordinate.")
        return scalar

    def _distance(self, point: Array, key: EvalKey, /) -> Array:
        value = self._scalar_level_set(point, key)
        if self.distance_semantics == "signed_distance":
            distance = value
        else:
            gradient = jax.grad(
                lambda coordinate: self._scalar_level_set(coordinate, key)
            )(point)
            magnitude = jnp.sqrt(jnp.sum(gradient * gradient))
            distance = value / jnp.maximum(magnitude, self.gradient_floor)
        return jnp.clip(distance, -self.distance_clip, self.distance_clip)

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        point = jnp.asarray(x)
        if point.shape != (self.in_size,) or jnp.iscomplexobj(point):
            raise ValueError(
                f"InterfaceFeatureLift input must be one real vector of shape "
                f"{(self.in_size,)}, got {point.shape}."
            )
        distance = self._distance(point, key)
        features = []
        if self.include_coordinates:
            features.append(point)
        if self.include_signed_distance:
            features.append(distance[None])
        if self.include_cusp:
            features.append(jnp.abs(distance)[None])
        if self.include_side:
            side = regularized_heaviside_values(distance, width=self.side_width)
            features.append(side[None])
        return jnp.concatenate(tuple(features), axis=-1)


def _positive_finite(value: float, name: str, /) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


__all__ = ["InterfaceDistanceSemantics", "InterfaceFeatureLift"]
