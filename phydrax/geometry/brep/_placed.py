#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authored affine pose operations on original exact CAD source definitions.

An authored orthogonal placement (including a reflected occurrence) is accepted
within the placement owner's binary64 orthogonality premise; its coefficients
are never normalized or canceled as though they formed an exactly orthogonal
matrix. Common-pose cancellation uses exact coefficient equality and a
binary-rational invertibility proof instead.
"""

from __future__ import annotations

from fractions import Fraction
from typing import final, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core

from ...typing import ConvertibleToArray, Float64, parse
from .._interval_enclosure import interval_add, interval_multiply
from ._intersection_curve import IntersectionCurve
from ._patches import AbstractCurve, AbstractSurfacePatch


def _exact_determinant(rotation: np.ndarray, /) -> Fraction:
    rows = tuple(tuple(Fraction(float(value)) for value in row) for row in rotation)
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _validate_pose(
    rotation: ConvertibleToArray, translation: ConvertibleToArray, /
) -> tuple[Array, Array]:
    rotation_host = np.asarray(rotation, dtype=np.float64)
    translation_host = np.asarray(translation, dtype=np.float64)
    if rotation_host.shape != (3, 3) or translation_host.shape != (3,):
        raise ValueError("A source placement requires a 3x3 rotation and 3-vector.")
    if not np.all(np.isfinite(rotation_host)) or not np.all(
        np.isfinite(translation_host)
    ):
        raise ValueError("A source placement must be finite.")
    if (
        np.max(np.abs(rotation_host.T @ rotation_host - np.eye(3, dtype=np.float64)))
        > 1.0e-12
        or _exact_determinant(rotation_host) == 0
    ):
        raise ValueError("A source placement rotation must be orthogonal and invertible.")
    return (
        parse(
            jnp.asarray(rotation_host, dtype=jnp.float64),
            Float64[Literal[3], Literal[3]],
            "rotation",
        ),
        parse(
            jnp.asarray(translation_host, dtype=jnp.float64),
            Float64[Literal[3]],
            "translation",
        ),
    )


def source_transform_bounds(
    rotation: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward image of source-coordinate intervals, with coordinate axis first.

    The three multiply/add operations use the shared interval substrate. This
    also encloses unbounded source jets without a sampled or nominal inverse.
    """
    if rotation.shape != (3, 3) or lower.shape != upper.shape or lower.shape[0] != 3:
        raise ValueError(
            "Source transform bounds require a 3x3 map and matching three-coordinate intervals."
        )
    coefficient_shape = (3,) + (1,) * (lower.ndim - 1)
    coefficient = rotation[:, 0].reshape(coefficient_shape)
    result = interval_multiply((coefficient, coefficient), (lower[0], upper[0]))
    for axis in (1, 2):
        coefficient = rotation[:, axis].reshape(coefficient_shape)
        result = interval_add(
            result,
            interval_multiply(
                (coefficient, coefficient),
                (lower[axis], upper[axis]),
            ),
        )
    return result


def _placed_box(
    rotation: Array, translation: Array, source_box: np.ndarray, /
) -> np.ndarray:
    bounds = source_transform_bounds(np.asarray(rotation), source_box[0], source_box[1])
    shift = np.asarray(translation)
    low, high = interval_add(bounds, (shift, shift))
    return np.stack((low, high))


@final
class PlacedSurface(AbstractSurfacePatch):
    """Original surface and immutable authored world-pose operation tree."""

    __strict_contract__ = True
    definition: AbstractSurfacePatch
    rotation: Float64[Literal[3], Literal[3]]
    translation: Float64[Literal[3]]

    def __init__(
        self,
        definition: AbstractSurfacePatch,
        rotation: ConvertibleToArray,
        translation: ConvertibleToArray,
    ) -> None:
        if not isinstance(definition, AbstractSurfacePatch):
            raise TypeError("Placed surfaces require an exact surface source definition.")
        rotation_, translation_ = _validate_pose(rotation, translation)
        self.definition = definition
        self.rotation = rotation_
        self.translation = translation_

    def evaluate(self, parameters: Array, /) -> Array:
        return self.definition.evaluate(parameters) @ self.rotation.T + self.translation

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.definition.periods

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        return self.definition.degenerate_isolines(parameter_box)

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        return self.definition.validate_parameter_box(parameter_box)

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        return self.definition.is_c1_on(parameter_box)

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        return _placed_box(
            self.rotation, self.translation, self.definition.bounding_box(parameter_box)
        )

    def derivative_bounds_batch(
        self,
        parameter_boxes: ConvertibleToArray,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.definition.derivative_bounds_batch(parameter_boxes, order=order)
        low, high = source_transform_bounds(
            np.asarray(self.rotation), np.moveaxis(low, 1, 0), np.moveaxis(high, 1, 0)
        )
        return np.moveaxis(low, 0, 1), np.moveaxis(high, 0, 1)


@final
class PlacedCurve(AbstractCurve):
    """Original three-dimensional edge source and authored pose operation tree."""

    __strict_contract__ = True
    definition: AbstractCurve | IntersectionCurve
    rotation: Float64[Literal[3], Literal[3]]
    translation: Float64[Literal[3]]

    def __init__(
        self,
        definition: AbstractCurve | IntersectionCurve,
        rotation: ConvertibleToArray,
        translation: ConvertibleToArray,
    ) -> None:
        if not isinstance(definition, (AbstractCurve, IntersectionCurve)):
            raise TypeError("Placed curves require an exact curve source definition.")
        if isinstance(definition, AbstractCurve) and definition.ambient_dimension != 3:
            raise ValueError("Placed edge definitions must be three-dimensional.")
        rotation_, translation_ = _validate_pose(rotation, translation)
        self.definition = definition
        self.rotation = rotation_
        self.translation = translation_

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        if isinstance(self.definition, AbstractCurve):
            return self.definition.period
        return float(self.definition.num_charts) if self.definition.closed else None

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        if isinstance(self.definition, AbstractCurve):
            return self.definition.parameter_domain
        return 0.0, float(self.definition.num_charts)

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        return self.definition.validate_range(first, last)

    def is_c1_on(self, first: float, last: float, /) -> bool:
        return self.definition.is_c1_on(first, last)

    def evaluate(self, parameters: Array, /) -> Array:
        if isinstance(self.definition, IntersectionCurve):
            if not self.definition.fully_certified:
                raise ValueError(
                    "Placed intersection edges require certified source charts."
                )
            result = self.definition.evaluate(parameters)
            if not isinstance(result.parameter_bound, jax_core.Tracer) and not np.all(
                np.isfinite(np.asarray(result.parameter_bound))
            ):
                raise ValueError(
                    "Placed intersection query has unresolved source bounds."
                )
            points = result.point
        else:
            points = self.definition.evaluate(parameters)
        return points @ self.rotation.T + self.translation

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        return _placed_box(
            self.rotation, self.translation, self.definition.bounding_box(first, last)
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.definition.derivative_bounds(first, last, order=order)
        return source_transform_bounds(np.asarray(self.rotation), low, high)


def same_source_pose(
    first: PlacedSurface | PlacedCurve, second: PlacedSurface | PlacedCurve, /
) -> bool:
    """Prove one identical invertible authored affine map for residual pullback.

    This proves equality-zero conjugation, not equality of the definitions or
    that the transpose is an inverse. It is a host-side source admission proof.
    """
    if not isinstance(first, (PlacedSurface, PlacedCurve)) or not isinstance(
        second, (PlacedSurface, PlacedCurve)
    ):
        raise TypeError("Source pose equivalence requires canonical placed carriers.")
    rotation = np.asarray(first.rotation)
    translation = np.asarray(first.translation)
    return bool(
        np.array_equal(
            rotation.view(np.uint64), np.asarray(second.rotation).view(np.uint64)
        )
        and np.array_equal(
            translation.view(np.uint64), np.asarray(second.translation).view(np.uint64)
        )
        and np.all(np.isfinite(rotation))
        and np.all(np.isfinite(translation))
        and _exact_determinant(rotation) != 0
    )
