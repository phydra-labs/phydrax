#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Whole-cell embedding evidence for affine source parameter triangles.

This is only a cell-local theorem. It does not prove agreement with another
cell, source trim containment, or a periodic/pole identification. In particular,
no rounded angle is promoted to a symbolic source period or pole.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._interval_enclosure import interval_multiply, interval_subtract
from .brep._boolean_core import chart_add, chart_scale, ChartScalar
from .brep._patches import AbstractSurfacePatch
from .brep._source_embedding import _sheet_sign, certify_source_embedding, SourceEmbeddingProof
from .brep._trim_helpers import period_scalar_bounds, PeriodOffset


type ExactTriangle = tuple[
    tuple[Fraction, Fraction], tuple[Fraction, Fraction], tuple[Fraction, Fraction]
]


@dataclass(frozen=True, slots=True, eq=False)
class AffineSurfaceCellProof:
    """A nonsingular affine triangle contained in an injective source sheet."""

    corners: ExactTriangle
    determinant: Fraction
    sheet: SourceEmbeddingProof

    def replay(self, surface: AbstractSurfacePatch, corners: ExactTriangle, /) -> bool:
        if corners != self.corners:
            return False
        current = certify_affine_surface_cell(surface, corners)
        return (
            current is not None
            and current.determinant == self.determinant
            and current.sheet.source_id == self.sheet.source_id
            and current.sheet.bounds == self.sheet.bounds
            and self.sheet.replay(surface, self.sheet.bounds)
        )


def certify_affine_surface_cell(
    surface: AbstractSurfacePatch, corners: ExactTriangle, /
) -> AffineSurfaceCellProof | None:
    """Prove that the complete affine triangle is a regular embedded cell.

    Coordinates must be the exact represented chart values, not rounded shadows
    of some other source values. The containing rectangle is conservative; an
    unresolved rectangle (including a pole or a full deck) is refused. This
    does not authorize gluing or assert that the triangle lies inside a trim.
    """
    if (
        not isinstance(corners, tuple)
        or len(corners) != 3
        or any(
            not isinstance(point, tuple)
            or len(point) != 2
            or any(not isinstance(value, Fraction) for value in point)
            for point in corners
        )
    ):
        return None
    a, b, c = corners
    determinant = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if not determinant:
        return None
    bounds = (
        (min(point[0] for point in corners), min(point[1] for point in corners)),
        (max(point[0] for point in corners), max(point[1] for point in corners)),
    )
    sheet = certify_source_embedding(surface, bounds)
    if sheet is None:
        return None
    return AffineSurfaceCellProof(corners, determinant, sheet)


type PeriodicTriangle = tuple[
    tuple[ChartScalar, ChartScalar],
    tuple[ChartScalar, ChartScalar],
    tuple[ChartScalar, ChartScalar],
]


@dataclass(frozen=True, slots=True, eq=False)
class PeriodicAffineSurfaceCellProof:
    """Regular affine source triangle with exact symbolic native periods."""

    corners: PeriodicTriangle
    orientation: int
    sheet: SourceEmbeddingProof

    def replay(self, surface: AbstractSurfacePatch, corners: PeriodicTriangle, /) -> bool:
        if corners != self.corners:
            return False
        current = certify_periodic_affine_surface_cell(surface, corners)
        return (
            current is not None
            and current.orientation == self.orientation
            and current.sheet.source_id == self.sheet.source_id
            and current.sheet.bounds == self.sheet.bounds
            and self.sheet.replay(surface, self.sheet.bounds)
        )


def certify_periodic_affine_surface_cell(
    surface: AbstractSurfacePatch, corners: PeriodicTriangle, /
) -> PeriodicAffineSurfaceCellProof | None:
    """Prove a source cell with rational-plus-symbolic-turn UV coordinates.

    Its complete affine image lies in an injective carrier rectangle. This
    theorem does not authorize the native-turn tokens, source trimming, or any
    gluing; the caller proves those independently from current occurrences.
    """
    if (
        not isinstance(corners, tuple)
        or len(corners) != 3
        or any(
            not isinstance(point, tuple)
            or len(point) != 2
            or any(
                not isinstance(value, Fraction)
                and not (
                    isinstance(value, PeriodOffset)
                    and isinstance(value.rational, Fraction)
                    and isinstance(value.turns, Fraction)
                )
                for value in point
            )
            for point in corners
        )
    ):
        return None

    def difference(a: ChartScalar, b: ChartScalar, /) -> ChartScalar:
        return chart_add(a, chart_scale(b, Fraction(-1)))

    a, b, c = corners
    u1, v1 = difference(b[0], a[0]), difference(b[1], a[1])
    u2, v2 = difference(c[0], a[0]), difference(c[1], a[1])
    determinant = interval_subtract(
        interval_multiply(period_scalar_bounds(u1), period_scalar_bounds(v2)),
        interval_multiply(period_scalar_bounds(v1), period_scalar_bounds(u2)),
    )
    if float(determinant[0]) > 0:
        orientation = 1
    elif float(determinant[1]) < 0:
        orientation = -1
    else:
        return None
    lower: list[ChartScalar] = []
    upper: list[ChartScalar] = []
    for axis in range(2):
        low = high = corners[0][axis]
        for point in corners[1:]:
            lo_sign = _sheet_sign(difference(point[axis], low))
            hi_sign = _sheet_sign(difference(point[axis], high))
            if lo_sign is None or hi_sign is None:
                return None
            if lo_sign < 0:
                low = point[axis]
            if hi_sign > 0:
                high = point[axis]
        lower.append(low)
        upper.append(high)
    bounds = ((lower[0], lower[1]), (upper[0], upper[1]))
    sheet = certify_source_embedding(surface, bounds)
    if sheet is None:
        return None
    return PeriodicAffineSurfaceCellProof(corners, orientation, sheet)
