#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact-definition affine placement wrappers for native B-Rep carriers."""

from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint
from ...linalg._exact import determinant_exact
from ...typing import checked, ConvertibleToArray
from .._interval_enclosure import interval_add, interval_multiply
from .._rational import fraction_down, fraction_up, rational_vector
from ._assembly import BRepAffinePlacement
from ._patches import AbstractCurve, AbstractSurfacePatch


def validate_pose(rotation: ArrayLike, translation: ArrayLike, /) -> tuple[Array, Array]:
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    if r.shape != (3, 3) or t.shape != (3,):
        raise ValueError("Placement pose must have rotation (3, 3) and translation (3,).")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
        raise ValueError("Placement pose coefficients must be finite.")
    exact = determinant_exact(tuple(rational_vector(row) for row in r))
    if exact == 0:
        raise ValueError(
            "Placement linear map must be exactly invertible over represented coefficients."
        )
    return jnp.asarray(r, dtype=jnp.float64), jnp.asarray(t, dtype=jnp.float64)


def source_transform_bounds(
    rotation: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward interval image with coordinate-first bounds."""
    r = np.asarray(rotation, dtype=np.float64)
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if r.shape != (3, 3) or lo.shape != hi.shape or lo.ndim < 1 or lo.shape[0] != 3:
        raise ValueError(
            "Placed source bounds require a 3x3 rotation and aligned coordinate-first intervals."
        )
    shape = (3,) + (1,) * (lo.ndim - 1)
    result = interval_multiply(
        (r[:, 0].reshape(shape), r[:, 0].reshape(shape)), (lo[0], hi[0])
    )
    for axis in (1, 2):
        coefficient = r[:, axis].reshape(shape)
        result = interval_add(
            result, interval_multiply((coefficient, coefficient), (lo[axis], hi[axis]))
        )
    return np.asarray(result[0], dtype=np.float64), np.asarray(
        result[1], dtype=np.float64
    )


def _affine_interval_transform(
    affine: BRepAffinePlacement,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
    *,
    include_translation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    lower_, upper_ = (
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    if lower_.shape != upper_.shape or lower_.ndim < 1 or lower_.shape[0] != 3:
        raise ValueError(
            "Placed source bounds require aligned coordinate-first intervals."
        )
    image_lower = np.empty_like(lower_)
    image_upper = np.empty_like(upper_)
    for row in range(3):
        interval = (np.zeros_like(lower_[0]), np.zeros_like(upper_[0]))
        for axis in range(3):
            coefficient = (
                np.asarray(fraction_down(affine.matrix[row][axis]), dtype=np.float64),
                np.asarray(fraction_up(affine.matrix[row][axis]), dtype=np.float64),
            )
            interval = interval_add(
                interval, interval_multiply(coefficient, (lower_[axis], upper_[axis]))
            )
        if include_translation:
            translation = (
                np.asarray(fraction_down(affine.matrix[row][3]), dtype=np.float64),
                np.asarray(fraction_up(affine.matrix[row][3]), dtype=np.float64),
            )
            interval = interval_add(interval, translation)
        image_lower[row], image_upper[row] = interval
    return image_lower, image_upper


def _exact_affine(
    rotation: Array,
    translation: Array,
    affine: BRepAffinePlacement | None,
    /,
) -> BRepAffinePlacement:
    matrix = np.asarray(rotation, dtype=np.float64)
    offset = np.asarray(translation, dtype=np.float64)
    if matrix.shape != (3, 3) or offset.shape != (3,):
        raise ValueError(
            "Placement affine realization requires a 3x3 linear map and 3-vector."
        )
    if affine is None:
        return BRepAffinePlacement.from_matrix(
            (
                (
                    Fraction.from_float(float(matrix[0, 0])),
                    Fraction.from_float(float(matrix[0, 1])),
                    Fraction.from_float(float(matrix[0, 2])),
                    Fraction.from_float(float(offset[0])),
                ),
                (
                    Fraction.from_float(float(matrix[1, 0])),
                    Fraction.from_float(float(matrix[1, 1])),
                    Fraction.from_float(float(matrix[1, 2])),
                    Fraction.from_float(float(offset[1])),
                ),
                (
                    Fraction.from_float(float(matrix[2, 0])),
                    Fraction.from_float(float(matrix[2, 1])),
                    Fraction.from_float(float(matrix[2, 2])),
                    Fraction.from_float(float(offset[2])),
                ),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
            )
        )
    if not isinstance(affine, BRepAffinePlacement):
        raise TypeError("placement affine must be a BRepAffinePlacement.")
    expected = np.asarray(
        [[float(value) for value in row] for row in affine.matrix],
        dtype=np.float64,
    )
    if not np.array_equal(expected[:3, :3], matrix) or not np.array_equal(
        expected[:3, 3], offset
    ):
        raise ValueError(
            "Numeric placement realization must round its exact affine source."
        )
    return affine


@eqx.filter_jit
def _placed_surface_evaluate(
    definition: AbstractSurfacePatch,
    rotation: Array,
    translation: Array,
    parameters: Array,
) -> Array:
    return definition.evaluate(parameters) @ rotation.T + translation


@eqx.filter_jit
def _placed_curve_evaluate(
    definition: AbstractCurve, rotation: Array, translation: Array, parameters: Array
) -> Array:
    return definition.evaluate(parameters) @ rotation.T + translation


class PlacedSurface(AbstractSurfacePatch):
    affine: BRepAffinePlacement = eqx.field(static=True)
    definition: AbstractSurfacePatch
    rotation: Array
    translation: Array

    @checked
    def __init__(
        self,
        definition: AbstractSurfacePatch,
        rotation: ArrayLike,
        translation: ArrayLike,
        /,
        *,
        affine: BRepAffinePlacement | None = None,
    ) -> None:
        self.rotation, self.translation = validate_pose(rotation, translation)
        self.affine = _exact_affine(self.rotation, self.translation, affine)
        self.definition = definition

    def evaluate(self, parameters: Array, /) -> Array:
        return _placed_surface_evaluate(
            self.definition, self.rotation, self.translation, parameters
        )

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.definition.periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.definition.native_period_symbols()

    @property
    def parameter_domain(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return self.definition.parameter_domain

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> Array:
        return self.definition.validate_parameter_box(parameter_box)

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        return self.definition.is_c1_on(parameter_box)

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = np.asarray(
            self.definition.bounding_box(self.validate_parameter_box(parameter_box)),
            dtype=np.float64,
        )
        lower, upper = _affine_interval_transform(
            self.affine,
            box[0].reshape((3, 1)),
            box[1].reshape((3, 1)),
            include_translation=True,
        )
        return np.stack((lower[:, 0], upper[:, 0]))

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Placed surface derivative bounds support order=1 only.")
        lower, upper = self.definition.derivative_bounds(
            self.validate_parameter_box(parameter_box), order=order
        )
        return _affine_interval_transform(
            self.affine, lower, upper, include_translation=False
        )

    def identity_payload(self) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload

        return {
            "kind": "placed-surface",
            "definition": carrier_payload(self.definition),
            "affine": self.affine.identity_payload(),
            "rotation": array_tree_fingerprint(self.rotation),
            "translation": array_tree_fingerprint(self.translation),
        }


class PlacedCurve(AbstractCurve):
    definition: AbstractCurve
    affine: BRepAffinePlacement = eqx.field(static=True)
    rotation: Array
    translation: Array

    @checked
    def __init__(
        self,
        definition: AbstractCurve,
        rotation: ArrayLike,
        translation: ArrayLike,
        /,
        *,
        affine: BRepAffinePlacement | None = None,
    ) -> None:
        if not isinstance(definition, AbstractCurve) or definition.ambient_dimension != 3:
            raise ValueError(
                "PlacedCurve.definition must be a three-dimensional AbstractCurve."
            )
        self.rotation, self.translation = validate_pose(rotation, translation)
        self.affine = _exact_affine(self.rotation, self.translation, affine)
        self.definition = definition

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        return self.definition.period

    def native_period_symbol(self, /) -> str | None:
        return self.definition.native_period_symbol()

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.definition.parameter_domain

    def evaluate(self, parameters: Array, /) -> Array:
        return _placed_curve_evaluate(
            self.definition, self.rotation, self.translation, parameters
        )

    def is_c1_on(self, first: float, last: float, /) -> bool:
        return self.definition.is_c1_on(first, last)

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        box = self.definition.bounding_box(first, last)
        lower, upper = _affine_interval_transform(
            self.affine, box[0, :, None], box[1, :, None], include_translation=True
        )
        return np.stack((lower[:, 0], upper[:, 0]))

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = self.definition.derivative_bounds(first, last, order=order)
        transformed = _affine_interval_transform(
            self.affine, lower[:, None], upper[:, None], include_translation=False
        )
        return transformed[0][:, 0], transformed[1][:, 0]

    def identity_payload(self) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload

        return {
            "kind": "placed-curve",
            "definition": carrier_payload(self.definition),
            "affine": self.affine.identity_payload(),
            "rotation": array_tree_fingerprint(self.rotation),
            "translation": array_tree_fingerprint(self.translation),
        }


def same_source_pose(
    first: PlacedSurface | PlacedCurve,
    second: PlacedSurface | PlacedCurve,
    /,
) -> bool:
    return type(first) is type(second) and first.affine.matrix == second.affine.matrix


__all__ = [
    "PlacedCurve",
    "PlacedSurface",
    "same_source_pose",
    "source_transform_bounds",
    "validate_pose",
]
