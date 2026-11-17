#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Query-local interval isolation of simple roots with exact fallback."""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jax import Array

from ...algebraic._exact_univariate import RealRootIsolator
from .._interval_enclosure import prepare_interval_function, PreparedIntervalFunction


_MAX_DEGREE = 12
_MAX_BOXES = 4096


def _horner(values: Array, /) -> Array:
    x = values[0]
    value = values[_MAX_DEGREE + 1]
    derivative = jnp.zeros_like(x)
    for index in range(_MAX_DEGREE, 0, -1):
        derivative = derivative * x + value
        value = value * x + values[index]
    return jnp.stack((value, derivative))


@lru_cache(maxsize=1)
def _program() -> PreparedIntervalFunction:
    return prepare_interval_function(
        _horner, _MAX_DEGREE + 2, batch_capacity=1, execution="batched"
    )


def _integer_coefficients(coefficients: tuple[Fraction, ...], /) -> tuple[int, ...]:
    denominator = math.lcm(*(value.denominator for value in coefficients))
    integers = tuple(
        value.numerator * (denominator // value.denominator) for value in coefficients
    )
    divisor = math.gcd(*integers)
    return tuple(value // divisor for value in integers)


def _sign(coefficients: tuple[int, ...], x: Fraction, /) -> int:
    numerator, denominator = x.numerator, x.denominator
    value = coefficients[-1]
    power = denominator
    for coefficient in reversed(coefficients[:-1]):
        value = value * numerator + coefficient * power
        power *= denominator
    return (value > 0) - (value < 0)


def _possibly_repeated(coefficients: tuple[int, ...], /) -> bool:
    """Screen repeated roots modulo a prime; uncertainty selects exact isolation."""
    prime = 65521
    polynomial = [value % prime for value in coefficients]
    if not polynomial[-1]:
        return True
    derivative = [(index * value) % prime for index, value in enumerate(polynomial)][1:]
    while derivative and derivative[-1] == 0:
        derivative.pop()
    while derivative:
        remainder = polynomial[:]
        inverse = pow(derivative[-1], -1, prime)
        while len(remainder) >= len(derivative):
            factor = remainder[-1] * inverse % prime
            offset = len(remainder) - len(derivative)
            for index, value in enumerate(derivative):
                remainder[offset + index] = (
                    remainder[offset + index] - factor * value
                ) % prime
            while remainder and remainder[-1] == 0:
                remainder.pop()
        polynomial, derivative = derivative, remainder
    return len(polynomial) > 1


def _coefficient_bounds(
    coefficients: tuple[int, ...], /
) -> tuple[np.ndarray, np.ndarray] | None:
    shift = max(abs(value).bit_length() for value in coefficients)
    if any(value and shift - abs(value).bit_length() > 1073 for value in coefficients):
        return None
    lower = np.zeros((_MAX_DEGREE + 2,), dtype=np.float64)
    upper = np.zeros_like(lower)
    for index, value in enumerate(coefficients, start=1):
        exact = Fraction(value, 1 << shift)
        rounded = float(exact)
        represented = Fraction.from_float(rounded)
        lower[index] = np.nextafter(rounded, -np.inf) if represented > exact else rounded
        upper[index] = np.nextafter(rounded, np.inf) if represented < exact else rounded
    return lower, upper


def _root_bound(coefficients: tuple[int, ...], /) -> Fraction:
    degree = len(coefficients) - 1
    leading = abs(coefficients[-1])
    exponent = 0
    for index, value in enumerate(coefficients[:-1]):
        if value:
            distance = degree - index
            bits = abs(value).bit_length() - leading.bit_length() + 1
            exponent = max(exponent, (bits + distance - 1) // distance)
    # If |a_i| <= |a_n| R**(n-i), the sum of all lower terms at
    # |x| >= 2R is strictly smaller than |a_n x**n|. R is dyadic.
    return Fraction(1 << (exponent + 1))


def _float_endpoint(value: Fraction, *, lower: bool) -> float | None:
    if abs(value.numerator).bit_length() - value.denominator.bit_length() > 1022:
        return None
    rounded = float(value)
    represented = Fraction.from_float(rounded)
    if (lower and represented > value) or (not lower and represented < value):
        rounded = float(np.nextafter(rounded, -np.inf if lower else np.inf))
    return rounded if math.isfinite(rounded) else None


class IntervalRootIsolator:
    """Isolate every simple real root; ambiguous polynomials use exact Sturm isolation."""

    def __init__(self, coefficients: tuple[Fraction, ...], /) -> None:
        self._coefficients = _integer_coefficients(coefficients)
        self._exact = RealRootIsolator(coefficients)
        self._boxes: list[tuple[Fraction, Fraction, int, int]] | None = None
        self._failed = False

    def _fallback(self, reason: str, /) -> None:
        self._failed = True
        self._boxes = None

    def _refine(
        self, box: tuple[Fraction, Fraction, int, int], /
    ) -> tuple[Fraction, Fraction, int, int]:
        lo, hi, left_sign, right_sign = box
        midpoint = (lo + hi) / 2
        if lo == hi:
            return box
        middle_sign = _sign(self._coefficients, midpoint)
        if not middle_sign:
            return midpoint, midpoint, 0, 0
        if middle_sign == left_sign:
            return midpoint, hi, middle_sign, right_sign
        return lo, midpoint, left_sign, middle_sign

    def _initial(self) -> None:
        if len(self._coefficients) - 1 > _MAX_DEGREE:
            self._fallback("degree")
            return
        bounds = _coefficient_bounds(self._coefficients)
        if _possibly_repeated(self._coefficients):
            self._fallback("possible multiplicity")
            return
        if bounds is None:
            self._fallback("coefficients")
            return
        bound = _root_bound(self._coefficients)
        if _float_endpoint(bound, lower=False) is None:
            self._fallback("root bound")
            return
        pending = [(-bound, bound)]
        certified: list[tuple[Fraction, Fraction, int, int]] = []
        count = 0
        while pending:
            count += len(pending)
            if count > _MAX_BOXES:
                self._fallback("box limit")
                return
            low = np.broadcast_to(bounds[0], (len(pending), _MAX_DEGREE + 2)).copy()
            high = np.broadcast_to(bounds[1], low.shape).copy()
            for row, (left, right) in enumerate(pending):
                lfloat = _float_endpoint(left, lower=True)
                rfloat = _float_endpoint(right, lower=False)
                if lfloat is None or rfloat is None:
                    self._fallback("endpoint precision")
                    return
                low[row, 0], high[row, 0] = lfloat, rfloat
            lower, upper = _program().evaluate(low, high)
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                self._fallback("nonfinite enclosure")
                return
            following: list[tuple[Fraction, Fraction]] = []
            for row, (left, right) in enumerate(pending):
                if lower[row, 0] > 0 or upper[row, 0] < 0:
                    continue
                left_sign = _sign(self._coefficients, left)
                right_sign = _sign(self._coefficients, right)
                derivative_nonzero = lower[row, 1] > 0 or upper[row, 1] < 0
                if derivative_nonzero:
                    if not left_sign or not right_sign or left_sign != right_sign:
                        wide_left = Fraction.from_float(float(low[row, 0]))
                        wide_right = Fraction.from_float(float(high[row, 0]))
                        if (wide_left < left or wide_right > right) and any(
                            (other_left < left and other_right > wide_left)
                            or (other_right > right and other_left < wide_right)
                            for other_left, other_right in (
                                *(
                                    box
                                    for index, box in enumerate(pending)
                                    if index != row
                                ),
                                *((box[0], box[1]) for box in certified),
                            )
                        ):
                            self._fallback("overlapping enclosure")
                            return
                    if not left_sign:
                        certified.append((left, left, 0, 0))
                    elif not right_sign:
                        certified.append((right, right, 0, 0))
                    elif left_sign != right_sign:
                        certified.append((left, right, left_sign, right_sign))
                    continue
                midpoint = (left + right) / 2
                following.extend(((left, midpoint), (midpoint, right)))
            pending = following
        self._boxes = sorted(set(certified))

    def isolate(
        self, *, tolerance: Fraction, recognize_rational_roots: bool = True
    ) -> tuple[tuple[Fraction, Fraction, int], ...]:
        if self._boxes is None and not self._failed:
            self._initial()
        if not self._failed:
            boxes = self._boxes
            if boxes is None:
                raise RuntimeError("interval root isolation did not produce boxes")
            for index, box in enumerate(boxes):
                while box[1] - box[0] > tolerance or (
                    index and boxes[index - 1][1] >= box[0]
                ):
                    if box[0] == box[1]:
                        self._fallback("overlapping roots")
                        break
                    box = self._refine(box)
                if self._failed:
                    break
                boxes[index] = box
            if not self._failed:
                return tuple((lo, hi, 1) for lo, hi, _, _ in boxes)
        return self._exact.isolate(
            tolerance=tolerance, recognize_rational_roots=recognize_rational_roots
        )
