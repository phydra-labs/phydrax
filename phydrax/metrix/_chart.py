#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._utils import _pointwise_array, _pointwise_jacfwd


class CoordinateChart(StrictModule):
    """Static identity for one ordered local coordinate representation.

    A chart deliberately does not own bounds, periodicity, sampling, or a physical
    unit system. Those are domain-level concerns.
    """

    name: str
    coordinates: tuple[str, ...]

    def __init__(self, name: str, coordinates: Sequence[str], /):
        name_ = str(name)
        coordinates_ = tuple(str(coordinate) for coordinate in coordinates)
        if not name_:
            raise ValueError("Chart name must be non-empty.")
        if not coordinates_:
            raise ValueError("A coordinate chart must have positive dimension.")
        if any(not coordinate for coordinate in coordinates_):
            raise ValueError("Coordinate names must be non-empty.")
        if len(set(coordinates_)) != len(coordinates_):
            raise ValueError("Coordinate names must be unique within a chart.")
        self.name = name_
        self.coordinates = coordinates_

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    def compatible_with(self, other: CoordinateChart, /) -> bool:
        return self.name == other.name and self.coordinates == other.coordinates


class _IdentityMap(StrictModule):
    def __call__(self, coordinates: Array, /) -> Array:
        return coordinates


class _ComposedMap(StrictModule):
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


class ChartTransition(StrictModule):
    """Directed coordinate map from ``source`` coordinates to ``target`` coordinates."""

    source: CoordinateChart
    target: CoordinateChart
    map_function: Callable[[Array], Array]
    inverse_function: Callable[[Array], Array] | None

    def __init__(
        self,
        source: CoordinateChart,
        target: CoordinateChart,
        map: Callable[[Array], Array],
        /,
        *,
        inverse: Callable[[Array], Array] | None = None,
    ):
        if not callable(map):
            raise TypeError("Chart transition map must be callable.")
        if inverse is not None and not callable(inverse):
            raise TypeError("Chart transition inverse must be callable when supplied.")
        self.source = source
        self.target = target
        self.map_function = map
        self.inverse_function = inverse

    @classmethod
    def identity(cls, chart: CoordinateChart, /) -> ChartTransition:
        identity = _IdentityMap()
        return cls(chart, chart, identity, inverse=identity)

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        result = _pointwise_array(
            self.map_function,
            coordinates,
            self.source.dimension,
        )
        if result.shape[-1:] != (self.target.dimension,):
            raise ValueError(
                "Chart transition output must have trailing dimension "
                f"{self.target.dimension}; got {result.shape}."
            )
        return result

    def inverse(self, coordinates: ArrayLike, /) -> Array:
        if self.inverse_function is None:
            raise ValueError(
                f"Chart transition {self.source.name!r} -> {self.target.name!r} "
                "does not provide an inverse."
            )
        result = _pointwise_array(
            self.inverse_function,
            coordinates,
            self.target.dimension,
        )
        if result.shape[-1:] != (self.source.dimension,):
            raise ValueError(
                "Inverse chart transition output must have trailing dimension "
                f"{self.source.dimension}; got {result.shape}."
            )
        return result

    def jacobian(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_jacfwd(
            self.map_function,
            coordinates,
            self.source.dimension,
        )

    def inverse_jacobian(self, coordinates: ArrayLike, /) -> Array:
        if self.inverse_function is None:
            raise ValueError(
                f"Chart transition {self.source.name!r} -> {self.target.name!r} "
                "does not provide an inverse."
            )
        return _pointwise_jacfwd(
            self.inverse_function,
            coordinates,
            self.target.dimension,
        )

    def hessian(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_jacfwd(
            jax.jacfwd(self.map_function),
            coordinates,
            self.source.dimension,
        )

    def compose(self, after: ChartTransition, /) -> ChartTransition:
        """Return ``after ∘ self`` after checking the intermediate chart."""

        if not self.target.compatible_with(after.source):
            raise ValueError(
                "Cannot compose chart transitions with mismatched intermediate charts: "
                f"{self.target.name!r} and {after.source.name!r}."
            )
        inverse = None
        if self.inverse_function is not None and after.inverse_function is not None:
            inverse = _ComposedMap(after.inverse_function, self.inverse_function)
        return ChartTransition(
            self.source,
            after.target,
            _ComposedMap(self.map_function, after.map_function),
            inverse=inverse,
        )
