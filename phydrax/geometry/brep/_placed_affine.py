#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact analytic proof descriptors; authoritative placed operations remain intact."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ._patches import AbstractCurve, AbstractSurfacePatch, LineCurve, PlanePatch
from ._placed import PlacedCurve, PlacedSurface


def _mapped_vector(
    rotation: np.ndarray, value: np.ndarray, translation: np.ndarray | None, /
) -> np.ndarray | None:
    result = []
    for row in range(3):
        exact = sum(
            (
                Fraction(float(rotation[row, column])) * Fraction(float(value[column]))
                for column in range(3)
            ),
            Fraction(0),
        )
        if translation is not None:
            exact += Fraction(float(translation[row]))
        represented = float(exact)
        if not np.isfinite(represented) or Fraction(represented) != exact:
            return None
        result.append(represented)
    return np.asarray(result, dtype=np.float64)


def exact_line_descriptor(curve: AbstractCurve, /) -> LineCurve | None:
    if isinstance(curve, LineCurve):
        return curve
    if not isinstance(curve, PlacedCurve) or not isinstance(
        curve.definition, AbstractCurve
    ):
        return None
    source = exact_line_descriptor(curve.definition)
    if source is None:
        return None
    origin = _mapped_vector(
        np.asarray(curve.rotation),
        np.asarray(source.origin),
        np.asarray(curve.translation),
    )
    direction = _mapped_vector(
        np.asarray(curve.rotation), np.asarray(source.direction), None
    )
    if origin is None or direction is None:
        return None
    return LineCurve(origin, direction)


def exact_plane_descriptor(patch: AbstractSurfacePatch, /) -> PlanePatch | None:
    if isinstance(patch, PlanePatch):
        return patch
    if not isinstance(patch, PlacedSurface):
        return None
    source = exact_plane_descriptor(patch.definition)
    if source is None:
        return None
    matrix = np.asarray(patch.rotation)
    origin = _mapped_vector(
        matrix, np.asarray(source.origin), np.asarray(patch.translation)
    )
    first = _mapped_vector(matrix, np.asarray(source.first_axis), None)
    second = _mapped_vector(matrix, np.asarray(source.second_axis), None)
    if origin is None or first is None or second is None:
        return None
    return PlanePatch(origin, first, second)
