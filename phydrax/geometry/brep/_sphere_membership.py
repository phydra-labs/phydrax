#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-exact polynomial membership of a complete affine sphere chart."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ._patches import AbstractSurfacePatch, sphere_source_equivalence
from ._placed import PlacedSurface


def _determinant(matrix: tuple[tuple[Fraction, Fraction, Fraction], ...], /) -> Fraction:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _square_root_interval(value: Fraction, /) -> tuple[float, float]:
    if value == 0:
        return 0.0, 0.0
    lower = upper = float(value)
    if Fraction(lower) > value:
        lower = float(np.nextafter(lower, -np.inf))
    if Fraction(upper) < value:
        upper = float(np.nextafter(upper, np.inf))
    lo, hi = float(np.sqrt(lower)), float(np.sqrt(upper))
    if Fraction(lo) ** 2 > value:
        lo = float(np.nextafter(lo, -np.inf))
    if Fraction(hi) ** 2 < value:
        hi = float(np.nextafter(hi, np.inf))
    return lo, hi


@dataclass(frozen=True, slots=True)
class SphereMembership:
    inside: bool
    on_surface: bool
    distance_lower: float
    distance_upper: float


@dataclass(frozen=True, slots=True)
class _SphereSource:
    center: tuple[Fraction, ...]
    basis: tuple[tuple[Fraction, Fraction, Fraction], ...]
    determinant: Fraction
    radius: Fraction
    radius_bounds: tuple[float, float]
    scale_bounds: tuple[float, float]


def prepare_full_sphere(patch: AbstractSurfacePatch, /) -> _SphereSource | None:
    """Retain exact radial and affine coefficients without replacing the source."""
    placements: list[PlacedSurface] = []
    while isinstance(patch, PlacedSurface):
        placements.append(patch)
        patch = patch.definition
    equivalence = sphere_source_equivalence(patch)
    if equivalence is None:
        return None
    original, signed_radius = equivalence
    radius = abs(signed_radius)
    if radius == 0:
        raise ValueError("A complete sphere solid requires a nonzero source radius.")
    axes = np.stack(
        (
            np.asarray(original.first_axis),
            np.asarray(original.second_axis),
            np.asarray(original.axis),
        ),
        axis=1,
    )
    basis = tuple(
        (Fraction(float(row[0])), Fraction(float(row[1])), Fraction(float(row[2])))
        for row in axes
    )
    center = tuple(Fraction(float(value)) for value in np.asarray(original.center))
    for placement in reversed(placements):
        rotation = np.asarray(placement.rotation)
        transform = tuple(
            tuple(Fraction(float(rotation[i, j])) for j in range(3)) for i in range(3)
        )
        center = tuple(
            sum((transform[i][j] * center[j] for j in range(3)), Fraction(0))
            + Fraction(float(placement.translation[i]))
            for i in range(3)
        )
        basis = tuple(
            (
                sum((transform[i][j] * basis[j][0] for j in range(3)), Fraction(0)),
                sum((transform[i][j] * basis[j][1] for j in range(3)), Fraction(0)),
                sum((transform[i][j] * basis[j][2] for j in range(3)), Fraction(0)),
            )
            for i in range(3)
        )
    determinant = _determinant(basis)
    if determinant == 0:
        raise ValueError("A complete sphere solid requires a nonsingular source frame.")
    gram = tuple(
        tuple(
            sum((basis[k][i] * basis[k][j] for k in range(3)), Fraction(0))
            for j in range(3)
        )
        for i in range(3)
    )
    upper_squared = max(sum((abs(value) for value in row), Fraction(0)) for row in gram)
    lower_squared = min(
        gram[i][i] - sum((abs(gram[i][j]) for j in range(3) if j != i), Fraction(0))
        for i in range(3)
    )
    if lower_squared > 0:
        scale_lower = _square_root_interval(lower_squared)[0]
    else:
        exact_lower = abs(determinant) / upper_squared
        scale_lower = float(exact_lower)
        if Fraction(scale_lower) > exact_lower:
            scale_lower = float(np.nextafter(scale_lower, -np.inf))
    scale_upper = _square_root_interval(upper_squared)[1]
    radius_lower = radius_upper = float(radius)
    if Fraction(radius_lower) > radius:
        radius_lower = float(np.nextafter(radius_lower, -np.inf))
    if Fraction(radius_upper) < radius:
        radius_upper = float(np.nextafter(radius_upper, np.inf))
    return _SphereSource(
        center,
        basis,
        determinant,
        radius,
        (radius_lower, radius_upper),
        (scale_lower, scale_upper),
    )


def classify_full_sphere(source: _SphereSource, point: np.ndarray, /) -> SphereMembership:
    """Exact membership and distance bounds for the prepared original source.

    The caller establishes complete angular/latitude chart coverage and closed
    solid incidence. Placements retain exact affine coefficients, not assumed
    orthogonality; nonrepresentable offset radii retain directed bounds.
    """
    basis, determinant = source.basis, source.determinant
    delta = tuple(
        Fraction(float(value)) - origin
        for value, origin in zip(point, source.center, strict=True)
    )
    numerators = []
    for column in range(3):
        replaced = tuple(
            (
                delta[row] if column == 0 else basis[row][0],
                delta[row] if column == 1 else basis[row][1],
                delta[row] if column == 2 else basis[row][2],
            )
            for row in range(3)
        )
        numerators.append(_determinant(replaced))
    squared = sum((value * value for value in numerators), Fraction(0)) / (
        determinant * determinant
    )
    sign = squared - source.radius * source.radius
    lo, hi = _square_root_interval(squared)
    radius_lower, radius_upper = source.radius_bounds
    radial_lower = max(0.0, radius_lower - hi, lo - radius_upper)
    radial_lower = max(0.0, float(np.nextafter(radial_lower, -np.inf)))
    radial_upper = float(
        np.nextafter(max(abs(lo - radius_upper), abs(hi - radius_lower)), np.inf)
    )
    scale_lower, scale_upper = source.scale_bounds
    return SphereMembership(
        sign <= 0,
        sign == 0,
        max(0.0, float(np.nextafter(radial_lower * scale_lower, -np.inf))),
        float(np.nextafter(radial_upper * scale_upper, np.inf)),
    )


__all__ = ["SphereMembership", "prepare_full_sphere", "classify_full_sphere"]
