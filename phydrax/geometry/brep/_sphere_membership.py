#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Polynomial membership and directed distance bounds for full affine sphere charts."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import jax
import numpy as np

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._rational import rational_dot, rational_vector
from ._coordinate_request import coordinate_source_request
from ._exact import rational_determinant, square_root_interval
from ._patches import AbstractSurfacePatch, sphere_source_equivalence
from ._placed import PlacedSurface


@dataclass(frozen=True, slots=True)
class SphereMembership:
    inside: bool
    on_surface: bool
    distance_lower: float
    distance_upper: float


@dataclass(frozen=True, slots=True)
class SphereSource:
    center: tuple[Fraction, ...]
    basis: tuple[tuple[Fraction, Fraction, Fraction], ...]
    determinant: Fraction
    radius: Fraction
    radius_bounds: tuple[float, float]
    scale_bounds: tuple[float, float]


@checked
def prepare_full_sphere(
    patch: AbstractSurfacePatch, /, *, budget: CoordinateEnclosureBudget | None = None
) -> SphereSource | None:
    """Prepare radial and affine coefficients for a full sphere chart."""
    if budget is None:
        with coordinate_source_request((patch,)) as budget:
            return prepare_full_sphere(patch, budget=budget)
    patch = jax.device_get(patch)
    placements: list[PlacedSurface] = []
    while True:
        match patch:
            case PlacedSurface():
                placements.append(patch)
                patch = patch.definition
            case _:
                break
    equivalence = sphere_source_equivalence(patch, budget=budget)
    if equivalence is None:
        return None
    original, signed_radius = equivalence
    radius = abs(signed_radius)
    if radius == 0:
        raise ValueError("prepare_full_sphere.patch.radius must be nonzero; got 0.")
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
    center = rational_vector(np.asarray(original.center))
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
    determinant = rational_determinant(*basis)
    if determinant == 0:
        raise ValueError(
            "prepare_full_sphere.patch.frame must be nonsingular; got determinant 0."
        )
    gram = tuple(
        tuple(
            rational_dot(
                tuple(basis[k][i] for k in range(3)),
                tuple(basis[k][j] for k in range(3)),
            )
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
        scale_lower = square_root_interval(lower_squared)[0]
    else:
        exact_lower = abs(determinant) / upper_squared
        scale_lower = float(exact_lower)
        if Fraction(scale_lower) > exact_lower:
            scale_lower = float(np.nextafter(scale_lower, -np.inf))
    scale_upper = square_root_interval(upper_squared)[1]
    radius_lower = radius_upper = float(radius)
    if Fraction(radius_lower) > radius:
        radius_lower = float(np.nextafter(radius_lower, -np.inf))
    if Fraction(radius_upper) < radius:
        radius_upper = float(np.nextafter(radius_upper, np.inf))
    return SphereSource(
        center=center,
        basis=basis,
        determinant=determinant,
        radius=radius,
        radius_bounds=(radius_lower, radius_upper),
        scale_bounds=(scale_lower, scale_upper),
    )


@checked
def classify_full_sphere(source: SphereSource, point: np.ndarray, /) -> SphereMembership:
    """Classify membership and enclose distance for the prepared source sphere.

    The caller establishes full angular/latitude chart coverage and closed
    solid incidence. Placements retain affine coefficients without assuming
    orthogonality; nonrepresentable offset radii retain directed bounds.
    """
    basis, determinant = source.basis, source.determinant
    delta = tuple(
        value - origin
        for value, origin in zip(rational_vector(point), source.center, strict=True)
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
        numerators.append(rational_determinant(*replaced))
    squared = sum((value * value for value in numerators), Fraction(0)) / (
        determinant * determinant
    )
    sign = squared - source.radius * source.radius
    lo, hi = square_root_interval(squared)
    radius_lower, radius_upper = source.radius_bounds
    radial_lower = max(0.0, radius_lower - hi, lo - radius_upper)
    radial_lower = max(0.0, float(np.nextafter(radial_lower, -np.inf)))
    radial_upper = float(
        np.nextafter(max(abs(lo - radius_upper), abs(hi - radius_lower)), np.inf)
    )
    scale_lower, scale_upper = source.scale_bounds
    return SphereMembership(
        inside=sign <= 0,
        on_surface=sign == 0,
        distance_lower=max(0.0, float(np.nextafter(radial_lower * scale_lower, -np.inf))),
        distance_upper=float(np.nextafter(radial_upper * scale_upper, np.inf)),
    )


__all__ = ["SphereMembership", "prepare_full_sphere", "classify_full_sphere"]
