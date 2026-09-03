#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

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
        return ein.contract("...ai,...i->...a", self.jacobian(points), values)

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
        return ein.contract("...ai,...a->...i", self.jacobian(points), values)

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


class Immersion(StrictModule):
    """Differentiable-map candidate with injective differential."""

    map: DifferentiableMap

    def __init__(
        self,
        source: CoordinateChart,
        target: CoordinateChart,
        map: Callable[[Array], Array],
        /,
    ):
        if source.dimension > target.dimension:
            raise ValueError(
                "An immersion source dimension must not exceed its target dimension."
            )
        self.map = DifferentiableMap(source, target, map)

    @property
    def source(self) -> CoordinateChart:
        return self.map.source

    @property
    def target(self) -> CoordinateChart:
        return self.map.target

    @property
    def map_function(self) -> Callable[[Array], Array]:
        return self.map.map_function

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        return self.map(coordinates)

    def jacobian(self, coordinates: ArrayLike, /) -> Array:
        return self.map.jacobian(coordinates)

    def hessian(self, coordinates: ArrayLike, /) -> Array:
        return self.map.hessian(coordinates)

    def pushforward(
        self,
        vector: ArrayLike,
        source_coordinates: ArrayLike,
        /,
    ) -> Array:
        return self.map.pushforward(vector, source_coordinates)

    def pullback_covector(
        self,
        covector: ArrayLike,
        source_coordinates: ArrayLike,
        /,
    ) -> Array:
        return self.map.pullback_covector(covector, source_coordinates)

    def minimum_singular_value(self, coordinates: ArrayLike, /) -> Array:
        return self.map.minimum_singular_value(coordinates)


class ImmersionValidationReport(StrictModule):
    """Sampled rank diagnostics for an immersion candidate."""

    valid: Array
    finite: Array
    minimum_singular_value: Array

    def __init__(
        self,
        *,
        valid: ArrayLike,
        finite: ArrayLike,
        minimum_singular_value: ArrayLike,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.minimum_singular_value = jnp.asarray(minimum_singular_value)


def validate_immersion(
    immersion: Immersion,
    points: ArrayLike,
    /,
    *,
    singular_value_tolerance: float = 1e-10,
    raise_on_error: bool = True,
) -> ImmersionValidationReport:
    """Validate full column rank at representative source points."""
    if not isinstance(immersion, Immersion):
        raise TypeError("validate_immersion requires an Immersion.")
    if singular_value_tolerance < 0.0:
        raise ValueError("singular_value_tolerance must be non-negative.")
    jacobian = immersion.jacobian(points)
    finite = jnp.all(jnp.isfinite(jacobian))
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    minimum = jnp.min(singular_values)
    valid = finite & (minimum > singular_value_tolerance)
    report = ImmersionValidationReport(
        valid=valid,
        finite=finite,
        minimum_singular_value=minimum,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Immersion validation failed: "
            f"finite={bool(jax.device_get(finite))}, "
            "minimum_singular_value="
            f"{float(jax.device_get(minimum))}."
        )
    return report


__all__ = [
    "DifferentiableMap",
    "Immersion",
    "ImmersionValidationReport",
    "validate_immersion",
]
