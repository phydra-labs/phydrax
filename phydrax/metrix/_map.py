#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import CoordinateChart
from ._utils import _pointwise_array, _pointwise_jacfwd


class _ComposedDifferentiableMap(StrictModule):
    first: Callable[[Array], Array]
    second: Callable[[Array], Array]

    def __init__(
        self,
        first: Callable[[Array], Array],
        second: Callable[[Array], Array],
        /,
    ):
        self.first = first
        self.second = second

    def __call__(self, coordinates: Array, /) -> Array:
        return self.second(self.first(coordinates))


class DifferentiableMap(StrictModule):
    """Directed differentiable map between possibly unequal-dimensional charts."""

    source: CoordinateChart
    target: CoordinateChart
    map_function: Callable[[Array], Array]

    def __init__(
        self,
        source: CoordinateChart,
        target: CoordinateChart,
        map: Callable[[Array], Array],
        /,
    ):
        if not isinstance(source, CoordinateChart) or not isinstance(
            target, CoordinateChart
        ):
            raise TypeError("DifferentiableMap source and target must be charts.")
        if not callable(map):
            raise TypeError("Differentiable map must be callable.")
        self.source = source
        self.target = target
        self.map_function = map

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        result = _pointwise_array(
            self.map_function,
            coordinates,
            self.source.dimension,
        )
        if result.shape[-1:] != (self.target.dimension,):
            raise ValueError(
                "Differentiable map output must have trailing dimension "
                f"{self.target.dimension}; got {result.shape}."
            )
        return result

    def jacobian(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_jacfwd(
            self.map_function,
            coordinates,
            self.source.dimension,
        )

    def hessian(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_jacfwd(
            jax.jacfwd(self.map_function),
            coordinates,
            self.source.dimension,
        )

    def pushforward(
        self,
        vector: ArrayLike,
        source_coordinates: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(vector)
        points = jnp.asarray(source_coordinates)
        expected = points.shape[:-1] + (self.source.dimension,)
        if values.shape != expected:
            raise ValueError(
                f"Source tangent vector must have shape {expected}; got {values.shape}."
            )
        return oe.contract("...ai,...i->...a", self.jacobian(points), values)

    def pullback_covector(
        self,
        covector: ArrayLike,
        source_coordinates: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(covector)
        points = jnp.asarray(source_coordinates)
        expected = points.shape[:-1] + (self.target.dimension,)
        if values.shape != expected:
            raise ValueError(
                f"Target covector must have shape {expected}; got {values.shape}."
            )
        return oe.contract("...ai,...a->...i", self.jacobian(points), values)

    def compose(self, after: DifferentiableMap, /) -> DifferentiableMap:
        """Return ``after ∘ self`` after checking the intermediate chart."""
        if not isinstance(after, DifferentiableMap):
            raise TypeError("after must be a DifferentiableMap.")
        if not self.target.compatible_with(after.source):
            raise ValueError(
                "Cannot compose maps with mismatched intermediate charts: "
                f"{self.target.name!r} and {after.source.name!r}."
            )
        return DifferentiableMap(
            self.source,
            after.target,
            _ComposedDifferentiableMap(self.map_function, after.map_function),
        )

    def minimum_singular_value(self, coordinates: ArrayLike, /) -> Array:
        """Return the smallest Jacobian singular value at each supplied point."""
        values = jnp.linalg.svd(self.jacobian(coordinates), compute_uv=False)
        return values[..., -1]


__all__ = ["DifferentiableMap"]
