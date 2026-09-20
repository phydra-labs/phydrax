#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from fractions import Fraction
from math import gcd, lcm


Polynomial = tuple[Fraction, ...]


def _trim(values: Polynomial, /) -> Polynomial:
    result = list(values)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def _degree(values: Polynomial, /) -> int:
    return len(_trim(values)) - 1


def _derivative(values: Polynomial, /) -> Polynomial:
    if len(values) <= 1:
        return (Fraction(0),)
    return _trim(tuple(index * values[index] for index in range(1, len(values))))


def _evaluate(values: Polynomial, point: Fraction, /) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(values):
        result = result * point + coefficient
    return result


def _divmod(left: Polynomial, right: Polynomial, /) -> tuple[Polynomial, Polynomial]:
    numerator = list(_trim(left))
    denominator = _trim(right)
    if denominator == (0,):
        raise ZeroDivisionError("Polynomial division by zero.")
    if len(numerator) < len(denominator):
        return (Fraction(0),), tuple(numerator)
    quotient = [Fraction(0)] * (len(numerator) - len(denominator) + 1)
    while len(numerator) >= len(denominator) and any(numerator):
        offset = len(numerator) - len(denominator)
        scale = numerator[-1] / denominator[-1]
        quotient[offset] = scale
        for index, coefficient in enumerate(denominator):
            numerator[offset + index] -= scale * coefficient
        numerator = list(_trim(tuple(numerator)))
    return _trim(tuple(quotient)), _trim(tuple(numerator))


def _exact_quotient(left: Polynomial, right: Polynomial, /) -> Polynomial:
    quotient, remainder = _divmod(left, right)
    if remainder != (0,):
        raise RuntimeError("Expected an exact polynomial quotient.")
    return quotient


def _monic(values: Polynomial, /) -> Polynomial:
    values_ = _trim(values)
    if values_ == (0,):
        return values_
    leading = values_[-1]
    return tuple(value / leading for value in values_)


def _gcd(left: Polynomial, right: Polynomial, /) -> Polynomial:
    left_ = _trim(left)
    right_ = _trim(right)
    while right_ != (0,):
        _, remainder = _divmod(left_, right_)
        left_, right_ = right_, remainder
    return _monic(left_)


def _square_free_factors(values: Polynomial, /) -> tuple[tuple[Polynomial, int], ...]:
    polynomial = _monic(values)
    repeated = _gcd(polynomial, _derivative(polynomial))
    remaining = _exact_quotient(polynomial, repeated)
    multiplicity = 1
    factors: list[tuple[Polynomial, int]] = []
    while _degree(remaining) > 0:
        shared = _gcd(remaining, repeated)
        factor = _exact_quotient(remaining, shared)
        if _degree(factor) > 0:
            factors.append((_monic(factor), multiplicity))
        remaining = shared
        repeated = _exact_quotient(repeated, shared)
        multiplicity += 1
    return tuple(factors)


def _positive_divisors(value: int, /) -> tuple[int, ...]:
    value_ = abs(int(value))
    if value_ == 0:
        return (0,)
    divisors = set()
    candidate = 1
    while candidate * candidate <= value_:
        if value_ % candidate == 0:
            divisors.add(candidate)
            divisors.add(value_ // candidate)
        candidate += 1
    return tuple(sorted(divisors))


def _integer_coefficients(values: Polynomial, /) -> tuple[int, ...]:
    denominator = 1
    for value in values:
        denominator = lcm(denominator, value.denominator)
    integers = [int(value * denominator) for value in values]
    common = 0
    for value in integers:
        common = gcd(common, abs(value))
    common = max(common, 1)
    return tuple(value // common for value in integers)


def _remove_rational_roots(
    values: Polynomial,
    /,
) -> tuple[Polynomial, tuple[Fraction, ...]]:
    polynomial = _trim(values)
    roots: list[Fraction] = []
    while _degree(polynomial) > 0 and polynomial[0] == 0:
        roots.append(Fraction(0))
        polynomial = _trim(polynomial[1:])
    if _degree(polynomial) <= 0:
        return polynomial, tuple(roots)
    integers = _integer_coefficients(polynomial)
    numerators = _positive_divisors(integers[0])
    denominators = _positive_divisors(integers[-1])
    candidates = sorted(
        {
            Fraction(sign * numerator, denominator)
            for numerator in numerators
            for denominator in denominators
            if denominator
            for sign in (-1, 1)
        }
    )
    for root in candidates:
        if _evaluate(polynomial, root) != 0:
            continue
        polynomial = _exact_quotient(polynomial, (-root, Fraction(1)))
        roots.append(root)
    return polynomial, tuple(sorted(roots))


def _sturm_sequence(values: Polynomial, /) -> tuple[Polynomial, ...]:
    first = _trim(values)
    second = _derivative(first)
    sequence = [first, second]
    while sequence[-1] != (0,):
        _, remainder = _divmod(sequence[-2], sequence[-1])
        if remainder == (0,):
            break
        sequence.append(tuple(-value for value in remainder))
    return tuple(sequence)


def _variations(sequence: tuple[Polynomial, ...], point: Fraction, /) -> int:
    signs = []
    for polynomial in sequence:
        value = _evaluate(polynomial, point)
        if value:
            signs.append(1 if value > 0 else -1)
    return sum(left != right for left, right in zip(signs[:-1], signs[1:], strict=True))


def _root_count(
    sequence: tuple[Polynomial, ...], lower: Fraction, upper: Fraction, /
) -> int:
    return _variations(sequence, lower) - _variations(sequence, upper)


def _irrational_intervals(
    values: Polynomial,
    tolerance: Fraction,
    /,
) -> tuple[tuple[Fraction, Fraction], ...]:
    if _degree(values) <= 0:
        return ()
    leading = abs(values[-1])
    bound = Fraction(1) + max((abs(value) / leading for value in values[:-1]), default=0)
    bound = Fraction((bound.numerator + bound.denominator - 1) // bound.denominator + 1)
    lower = -bound
    upper = bound
    while _evaluate(values, lower) == 0 or _evaluate(values, upper) == 0:
        lower -= 1
        upper += 1
    sequence = _sturm_sequence(values)
    pending = [(lower, upper, _root_count(sequence, lower, upper))]
    intervals: list[tuple[Fraction, Fraction]] = []
    while pending:
        left, right, count = pending.pop()
        if count == 0:
            continue
        if count == 1 and right - left <= tolerance:
            intervals.append((left, right))
            continue
        midpoint = (left + right) / 2
        left_count = _root_count(sequence, left, midpoint)
        right_count = count - left_count
        pending.append((midpoint, right, right_count))
        pending.append((left, midpoint, left_count))
    return tuple(sorted(intervals))


def isolate_real_roots(
    coefficients: tuple[int | Fraction, ...],
    /,
    *,
    tolerance: Fraction,
) -> tuple[tuple[Fraction, Fraction, int], ...]:
    polynomial = _trim(tuple(Fraction(value) for value in coefficients))
    if polynomial == (0,):
        raise ValueError("A nonzero coefficient sequence is required.")
    result: list[tuple[Fraction, Fraction, int]] = []
    for factor, multiplicity in _square_free_factors(polynomial):
        remainder, rational_roots = _remove_rational_roots(factor)
        result.extend((root, root, multiplicity) for root in rational_roots)
        result.extend(
            (lower, upper, multiplicity)
            for lower, upper in _irrational_intervals(remainder, tolerance)
        )
    return tuple(sorted(result, key=lambda item: (item[0], item[1])))


__all__ = ["isolate_real_roots"]
