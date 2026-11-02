#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Whole-carrier displacement bounds for explicit native BRep repairs.

The bounds compare the same native parameter on both carriers.  Thus they
also bound every restriction of that parameterization, without depending on
samples or on a particular trim.  They use the exact source coefficients,
not their rounded numeric realizations.
"""

from __future__ import annotations

from fractions import Fraction

from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
    RadicalElement,
)
from ...linalg._hermitian_spectral import fraction_sqrt_interval
from ._patches import CircleCurve, TorusPatch


def _absolute_upper(value: ExactScalar, /) -> Fraction:
    lower, upper = coefficient_interval(value, bits=128)
    return max(abs(lower), abs(upper))


def _norm_upper(values: tuple[ExactScalar, ...], /) -> Fraction:
    components = tuple(_absolute_upper(value) for value in values)
    return fraction_sqrt_interval(
        sum((value * value for value in components), Fraction(0))
    )[1]


def _frame_norm_upper(axes: tuple[tuple[ExactScalar, ...], ...], /) -> Fraction:
    """Bound the operator norm by the maximum absolute Gram row sum.

    This does not assume that the exact axes are orthonormal: native numeric
    shadows may be normalized while their exact coefficients are not.  For an
    exactly orthonormal frame the bound is exactly one.
    """
    gram = tuple(
        tuple(
            sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))
            for right in axes
        )
        for left in axes
    )
    squared = max(
        sum((_absolute_upper(value) for value in row), Fraction(0)) for row in gram
    )
    return fraction_sqrt_interval(squared)[1]


def vertex_displacement_bound(
    source: tuple[ExactScalar, ...], target: tuple[ExactScalar, ...], /
) -> Fraction:
    """Return a rational upper bound for the distance between exact points."""
    if not isinstance(source, tuple) or not isinstance(target, tuple):
        raise TypeError("Vertex coordinates must be tuples of exact scalars.")
    if not source or len(source) != len(target):
        raise ValueError("Vertex coordinates must have the same nonzero dimension.")
    if any(
        not isinstance(value, (Fraction, RadicalElement)) for value in (*source, *target)
    ):
        raise TypeError("Vertex coordinates must be exact scalars.")
    return _norm_upper(tuple(a - b for a, b in zip(source, target, strict=True)))


def circle_displacement_bound(source: CircleCurve, target: CircleCurve, /) -> Fraction:
    """Bound displacement everywhere on two aligned exact circle carriers.

    The exact parameter axes must agree; other differences are unsupported.
    The result is an outward rational bound, suitable for exact comparison with
    a rational construction tolerance.
    """
    if type(source) is not CircleCurve or type(target) is not CircleCurve:
        raise TypeError("Circle displacement requires native CircleCurve carriers.")
    if (
        len(source.center_exact) != len(target.center_exact)
        or source.first_axis_exact != target.first_axis_exact
        or source.second_axis_exact != target.second_axis_exact
    ):
        raise ValueError("Circle displacement requires identical exact parameter axes.")
    center = vertex_displacement_bound(source.center_exact, target.center_exact)
    radius = _absolute_upper(source.radius_exact - target.radius_exact)
    frame = _frame_norm_upper((source.first_axis_exact, source.second_axis_exact))
    return center + radius * frame


def torus_displacement_bound(source: TorusPatch, target: TorusPatch, /) -> Fraction:
    """Bound displacement everywhere on two aligned exact torus carriers.

    The frame is bound to the actual carriers and compared exactly.  The
    expression bounds the center translation, the major-radius contribution,
    and the minor-radius contribution over both full angular periods.
    """
    if type(source) is not TorusPatch or type(target) is not TorusPatch:
        raise TypeError("Torus displacement requires native TorusPatch carriers.")
    if (
        source.first_axis_exact != target.first_axis_exact
        or source.second_axis_exact != target.second_axis_exact
        or source.axis_exact != target.axis_exact
    ):
        raise ValueError("Torus displacement requires identical exact parameter axes.")
    center = vertex_displacement_bound(source.center_exact, target.center_exact)
    major = _absolute_upper(source.major_radius_exact - target.major_radius_exact)
    minor = _absolute_upper(source.minor_radius_exact - target.minor_radius_exact)
    radial = _frame_norm_upper((source.first_axis_exact, source.second_axis_exact))
    tube = _frame_norm_upper(
        (source.first_axis_exact, source.second_axis_exact, source.axis_exact)
    )
    return center + major * radial + minor * tube


def torus_parameter_displacement_bound(
    carrier: TorusPatch,
    longitude_shift: ExactScalar,
    tube_angle_shift: ExactScalar,
    /,
) -> Fraction:
    """Bound motion under fixed angular shifts everywhere on a torus.

    Shifts are exact radians.  This bounds the parameterization, not whether
    two trimmed domains actually correspond under those shifts; the caller
    must establish that correspondence separately.
    """
    if type(carrier) is not TorusPatch:
        raise TypeError("Parameter displacement requires a native TorusPatch.")
    if not all(
        isinstance(value, (Fraction, RadicalElement))
        for value in (longitude_shift, tube_angle_shift)
    ):
        raise TypeError("Parameter shifts must be exact scalars.")
    radial = _frame_norm_upper((carrier.first_axis_exact, carrier.second_axis_exact))
    tube = _frame_norm_upper(
        (carrier.first_axis_exact, carrier.second_axis_exact, carrier.axis_exact)
    )
    major = _absolute_upper(carrier.major_radius_exact)
    minor = _absolute_upper(carrier.minor_radius_exact)
    return (
        _absolute_upper(longitude_shift) * (major + minor) * radial
        + _absolute_upper(tube_angle_shift) * minor * tube
    )
