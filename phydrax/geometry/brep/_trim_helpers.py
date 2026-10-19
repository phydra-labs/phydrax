#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rational parameter arithmetic shared by trims, pcurves, and Boolean arrangements."""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from types import NotImplementedType
from typing import overload

import numpy as np

from .._interval_enclosure import interval_add, interval_multiply
from .._rational import fraction_down, fraction_up


def fraction_bounds(value: Fraction, /) -> tuple[np.ndarray, np.ndarray]:
    """Return scalar-array directed endpoints for an exact rational."""
    return np.asarray(fraction_down(value)), np.asarray(fraction_up(value))


@dataclasses.dataclass(frozen=True)
class PeriodOffset:
    """Represent a scalar in Q + Q*(2*pi) with rational offset and turn coefficients."""

    rational: Fraction
    turns: Fraction

    @overload
    def __add__(
        self, other: Fraction | int | PeriodOffset
    ) -> Fraction | PeriodOffset: ...

    @overload
    def __add__(self, other: object) -> Fraction | PeriodOffset | NotImplementedType: ...

    def __add__(self, other: object) -> Fraction | PeriodOffset | NotImplementedType:
        if isinstance(other, PeriodOffset):
            return period_offset(self.rational + other.rational, self.turns + other.turns)
        if isinstance(other, (Fraction, int)):
            return period_offset(self.rational + other, self.turns)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self) -> Fraction | PeriodOffset:
        return period_offset(-self.rational, -self.turns)

    @overload
    def __sub__(
        self, other: Fraction | int | PeriodOffset
    ) -> Fraction | PeriodOffset: ...

    @overload
    def __sub__(self, other: object) -> Fraction | PeriodOffset | NotImplementedType: ...

    def __sub__(self, other: object) -> Fraction | PeriodOffset | NotImplementedType:
        if isinstance(other, (Fraction, int, PeriodOffset)):
            return self + -other
        return NotImplemented

    @overload
    def __rsub__(
        self, other: Fraction | int | PeriodOffset
    ) -> Fraction | PeriodOffset: ...

    @overload
    def __rsub__(self, other: object) -> Fraction | PeriodOffset | NotImplementedType: ...

    def __rsub__(self, other: object) -> Fraction | PeriodOffset | NotImplementedType:
        if isinstance(other, (Fraction, int, PeriodOffset)):
            return -self + other
        return NotImplemented

    @overload
    def __mul__(self, other: Fraction | int) -> Fraction | PeriodOffset: ...

    @overload
    def __mul__(self, other: object) -> Fraction | PeriodOffset | NotImplementedType: ...

    def __mul__(self, other: object) -> Fraction | PeriodOffset | NotImplementedType:
        if isinstance(other, (Fraction, int)):
            return period_offset(self.rational * other, self.turns * other)
        return NotImplemented

    __rmul__ = __mul__


def period_offset(rational: Fraction, turns: Fraction, /) -> Fraction | PeriodOffset:
    return (
        Fraction(rational)
        if not turns
        else PeriodOffset(Fraction(rational), Fraction(turns))
    )


def period_scalar_bounds(
    value: Fraction | PeriodOffset, /
) -> tuple[np.ndarray, np.ndarray]:
    """Directed enclosure of the authored Q + Q*(2*pi) expression."""
    if isinstance(value, Fraction):
        return fraction_bounds(value)
    period = (
        np.asarray(np.nextafter(math.tau, -np.inf)),
        np.asarray(np.nextafter(math.tau, np.inf)),
    )
    lower, upper = interval_add(
        fraction_bounds(value.rational),
        interval_multiply(fraction_bounds(value.turns), period),
    )
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
