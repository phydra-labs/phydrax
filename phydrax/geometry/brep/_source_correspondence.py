#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic source correspondence and authored-pose bounds."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import phydrax.geometry.brep._intersection_curve as intersection_curve_owner

from ..._strict import StrictModule
from ...typing import checked
from ._exact import square_root_interval
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    CircleCurve,
    CylinderPatch,
    PlanePatch,
)


def norm_upper(components: np.ndarray, /) -> float:
    if np.any(np.isnan(components)):
        return np.inf
    squared = np.float64(0.0)
    for component in components:
        term = np.nextafter(component * component, np.inf)
        squared = np.nextafter(squared + term, np.inf)
    return float(np.nextafter(np.sqrt(squared), np.inf))


def fraction_norm_upper(components: tuple[Fraction, ...], /) -> float:
    squared = sum((value * value for value in components), Fraction(0))
    return (
        0.0
        if squared == 0
        else float(np.nextafter(np.sqrt(np.nextafter(float(squared), np.inf)), np.inf))
    )


class _HarmonicCoefficients(StrictModule):
    center: tuple[Fraction, ...]
    cosine: tuple[Fraction, ...]
    sine: tuple[Fraction, ...]


def _circle_harmonics(source: CircleCurve) -> _HarmonicCoefficients:
    radius = Fraction(float(source.radius))
    return _HarmonicCoefficients(
        tuple(Fraction(float(value)) for value in np.asarray(source.center)),
        tuple(radius * Fraction(float(value)) for value in np.asarray(source.first_axis)),
        tuple(
            radius * Fraction(float(value)) for value in np.asarray(source.second_axis)
        ),
    )


def _affine_harmonics(
    source: intersection_curve_owner.AffinePCurve, coefficients: _HarmonicCoefficients
) -> _HarmonicCoefficients:
    matrix, offset = source.matrix, source.offset

    def image(vector: tuple[Fraction, ...], translate: bool) -> tuple[Fraction, ...]:
        return tuple(
            (offset[axis] if translate else Fraction(0))
            + sum(
                (
                    coefficient * value
                    for coefficient, value in zip(row, vector, strict=True)
                ),
                Fraction(0),
            )
            for axis, row in enumerate(matrix)
        )

    return _HarmonicCoefficients(
        image(coefficients.center, True),
        image(coefficients.cosine, False),
        image(coefficients.sine, False),
    )


def _pcurve_harmonics(source: AbstractCurve) -> _HarmonicCoefficients | None:
    match source:
        case CircleCurve():
            return _circle_harmonics(source)
        case intersection_curve_owner.AffinePCurve(curve=AbstractCurve() as curve):
            coefficients = _pcurve_harmonics(curve)
            return (
                None if coefficients is None else _affine_harmonics(source, coefficients)
            )
        case _:
            return None


def _plane_harmonics(
    surface: PlanePatch, coefficients: _HarmonicCoefficients
) -> _HarmonicCoefficients:
    axes = tuple(
        tuple(Fraction(float(value)) for value in np.asarray(axis))
        for axis in (surface.first_axis, surface.second_axis)
    )
    origin = tuple(Fraction(float(value)) for value in np.asarray(surface.origin))
    center, cosine, sine = (
        tuple(
            (origin[dimension] if component == 0 else Fraction(0))
            + sum(
                (axes[axis][dimension] * vector[axis] for axis in range(2)),
                Fraction(0),
            )
            for dimension in range(3)
        )
        for component, vector in enumerate(
            (coefficients.center, coefficients.cosine, coefficients.sine)
        )
    )
    return _HarmonicCoefficients(center, cosine, sine)


def _cylinder_harmonics(
    surface: CylinderPatch, pcurve: AbstractCurve
) -> _HarmonicCoefficients | None:
    coefficients = intersection_curve_owner.affine_curve_coefficients(
        pcurve, np.zeros((2,), dtype=np.float64)
    )
    if (
        coefficients is None
        or coefficients[0][0] != 0
        or coefficients[1] != (Fraction(1), Fraction(0))
    ):
        return None
    height = coefficients[0][1]
    center = tuple(
        Fraction(float(a)) + height * Fraction(float(b))
        for a, b in zip(np.asarray(surface.origin), np.asarray(surface.axis), strict=True)
    )
    radius = Fraction(float(surface.radius))
    return _HarmonicCoefficients(
        center,
        tuple(
            radius * Fraction(float(value)) for value in np.asarray(surface.first_axis)
        ),
        tuple(
            radius * Fraction(float(value)) for value in np.asarray(surface.second_axis)
        ),
    )


@checked
def circle_correspondence(
    curve: AbstractCurve | None, pcurve: AbstractCurve, surface: AbstractSurfacePatch, /
) -> float | None:
    """Return a deviation bound from full-curve harmonic source coefficients."""
    match curve:
        case CircleCurve():
            coefficients = _circle_harmonics(curve)
        case _:
            return None
    match surface:
        case PlanePatch():
            parameters = _pcurve_harmonics(pcurve)
            if parameters is None:
                return None
            expected = _plane_harmonics(surface, parameters)
        case CylinderPatch():
            expected = _cylinder_harmonics(surface, pcurve)
            if expected is None:
                return None
        case _:
            return None
    bound = 0.0
    for actual, target in (
        (coefficients.center, expected.center),
        (coefficients.cosine, expected.cosine),
        (coefficients.sine, expected.sine),
    ):
        contribution = fraction_norm_upper(
            tuple(a - b for a, b in zip(actual, target, strict=True))
        )
        bound = (
            bound
            if contribution == 0.0
            else float(np.nextafter(bound + contribution, np.inf))
        )
    return bound


@checked
def plane_correspondence(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: PlanePatch,
    first: float,
    last: float,
    point: np.ndarray,
    /,
) -> float | None:
    """Return a deviation bound from affine coefficients or a shared rational basis."""
    origin = tuple(Fraction(float(value)) for value in np.asarray(surface.origin))
    axes = tuple(
        tuple(Fraction(float(value)) for value in np.asarray(axis))
        for axis in (surface.first_axis, surface.second_axis)
    )
    expected, parameters = (
        intersection_curve_owner.affine_curve_coefficients(curve, point),
        intersection_curve_owner.affine_curve_coefficients(
            pcurve, np.zeros((2,), dtype=np.float64)
        ),
    )
    if expected is not None and parameters is not None:
        constant = tuple(
            origin[dimension]
            + sum(
                (axes[axis][dimension] * parameters[0][axis] for axis in range(2)),
                Fraction(0),
            )
            - expected[0][dimension]
            for dimension in range(3)
        )
        derivative = tuple(
            sum(
                (axes[axis][dimension] * parameters[1][axis] for axis in range(2)),
                Fraction(0),
            )
            - expected[1][dimension]
            for dimension in range(3)
        )
        return max(
            fraction_norm_upper(
                tuple(
                    a + Fraction(parameter) * b
                    for a, b in zip(constant, derivative, strict=True)
                )
            )
            for parameter in (first, last)
        )
    match (curve, pcurve):
        case (BSplineCurve() as spline, BSplineCurve() as parameter_spline):
            curve, pcurve = spline, parameter_spline
        case _:
            return None
    if (
        curve.degree != pcurve.degree
        or not np.array_equal(np.asarray(curve.knots), np.asarray(pcurve.knots))
        or not np.array_equal(np.asarray(curve.weights), np.asarray(pcurve.weights))
        or np.any(np.asarray(curve.weights) <= 0.0)
    ):
        return None
    residuals = [
        tuple(
            origin[dimension]
            + sum(
                (axes[axis][dimension] * Fraction(float(uv[axis])) for axis in range(2)),
                Fraction(0),
            )
            - Fraction(float(xyz[dimension]))
            for dimension in range(3)
        )
        for xyz, uv in zip(
            np.asarray(curve.control_points),
            np.asarray(pcurve.control_points),
            strict=True,
        )
    ]
    return max(fraction_norm_upper(residual) for residual in residuals)


def placed_deviation_bound(rotation: np.ndarray, deviation: float, /) -> float:
    """Return a Gram/Gershgorin operator bound for an authored binary source pose."""
    if deviation == 0.0:
        return 0.0
    columns = tuple(
        tuple(Fraction(float(rotation[row, column])) for row in range(3))
        for column in range(3)
    )
    gram = tuple(
        tuple(
            sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))
            for second in columns
        )
        for first in columns
    )
    squared = max(sum((abs(value) for value in row), Fraction(0)) for row in gram)
    scale = square_root_interval(squared)[1]
    return float(np.nextafter(scale * deviation, np.inf))
