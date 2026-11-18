#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Compare isolation with a Fraction-only reference implementation."""

from fractions import Fraction as F
from math import gcd, lcm
from random import Random

from phydrax.algebraic._exact_univariate import isolate_real_roots, RealRootIsolator


type Poly = tuple[F, ...]
type Interval = tuple[F, F, int]


def _trim(p: Poly) -> Poly:
    values = list(p)
    while len(values) > 1 and not values[-1]:
        values.pop()
    return tuple(values)


def _divmod_poly(a: Poly, b: Poly) -> tuple[Poly, Poly]:
    remainder = list(_trim(a))
    b = _trim(b)
    quotient = [F(0)] * max(1, len(remainder) - len(b) + 1)
    while len(remainder) >= len(b) and any(remainder):
        offset = len(remainder) - len(b)
        coefficient = remainder[-1] / b[-1]
        quotient[offset] = coefficient
        for i, value in enumerate(b):
            remainder[offset + i] -= coefficient * value
        remainder = list(_trim(tuple(remainder)))
    return _trim(tuple(quotient)), _trim(tuple(remainder))


def _quotient(a: Poly, b: Poly) -> Poly:
    q, r = _divmod_poly(a, b)
    if r != (0,):
        raise ArithmeticError("reference division is not exact")
    return q


def _monic(p: Poly) -> Poly:
    p = _trim(p)
    return tuple(v / p[-1] for v in p) if p != (0,) else p


def _gcd_poly(a: Poly, b: Poly) -> Poly:
    while b != (0,):
        a, b = b, _divmod_poly(a, b)[1]
    return _monic(a)


def _derivative(p: Poly) -> Poly:
    return _trim(tuple(i * p[i] for i in range(1, len(p))) or (F(0),))


def _factors(p: Poly) -> list[tuple[Poly, int]]:
    p = _monic(p)
    repeated = _gcd_poly(p, _derivative(p))
    remaining = _quotient(p, repeated)
    factors = []
    multiplicity = 1
    while len(remaining) > 1:
        shared = _gcd_poly(remaining, repeated)
        factor = _quotient(remaining, shared)
        if len(factor) > 1:
            factors.append((_monic(factor), multiplicity))
        remaining = shared
        repeated = _quotient(repeated, shared)
        multiplicity += 1
    return factors


def _evaluate(p: Poly, x: F) -> F:
    result = F(0)
    for coefficient in reversed(p):
        result = result * x + coefficient
    return result


def _sequence(p: Poly) -> list[Poly]:
    sequence = [p, _derivative(p)]
    while sequence[-1] != (0,):
        remainder = _divmod_poly(sequence[-2], sequence[-1])[1]
        if remainder == (0,):
            break
        sequence.append(tuple(-v for v in remainder))
    return sequence


def _variation(sequence: list[Poly], x: F) -> int:
    values = [_evaluate(p, x) for p in sequence]
    signs = [v > 0 for v in values if v]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _integer_coefficients(p: Poly) -> tuple[int, ...]:
    denominator = lcm(*(v.denominator for v in p))
    values = tuple(v.numerator * (denominator // v.denominator) for v in p)
    content = gcd(*values)
    return tuple(v // content for v in values)


def _no_rational_roots(p: Poly) -> bool:
    coefficients = _integer_coefficients(p)
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if coefficients[-1] % prime and all(
            sum(value * pow(x, i, prime) for i, value in enumerate(coefficients)) % prime
            for x in range(prime)
        ):
            return True
    return False


def _isolate_factor(p: Poly, tolerance: F, recognize: bool) -> tuple[tuple[F, F], ...]:
    exact: list[tuple[F, F]] = []
    while len(p) > 1:
        if len(p) == 2:
            root = -p[0] / p[1]
            exact.append((root, root))
            break
        sequence = _sequence(p)
        denominator = abs(_integer_coefficients(p)[-1])
        separation = F(1, 2 * denominator * denominator)
        bound = F(1) + max(abs(v / p[-1]) for v in p[:-1])
        bound = F((bound.numerator + bound.denominator - 1) // bound.denominator + 1)
        lower, upper = -bound, bound
        while not _evaluate(p, lower) or not _evaluate(p, upper):
            lower -= 1
            upper += 1
        pending = [
            (lower, upper, _variation(sequence, lower) - _variation(sequence, upper))
        ]
        intervals: list[tuple[F, F]] = []
        found = None
        accept = not recognize or _no_rational_roots(p)
        while pending:
            left, right, count = pending.pop()
            if not count:
                continue
            midpoint = (left + right) / 2
            if not _evaluate(p, midpoint):
                found = midpoint
                break
            if count == 1 and right - left <= tolerance:
                if accept:
                    intervals.append((left, right))
                    continue
                if right - left < separation:
                    candidate = midpoint.limit_denominator(denominator)
                    if left <= candidate <= right and not _evaluate(p, candidate):
                        found = candidate
                        break
                    intervals.append((left, right))
                    continue
            left_count = _variation(sequence, left) - _variation(sequence, midpoint)
            pending.append((midpoint, right, count - left_count))
            pending.append((left, midpoint, left_count))
        if found is None:
            return tuple(sorted((*exact, *intervals)))
        exact.append((found, found))
        p = _quotient(p, (-found, F(1)))
    return tuple(sorted(exact))


def _reference(
    coefficients: tuple[int | F, ...], tolerance: F, recognize: bool
) -> tuple[Interval, ...]:
    factors = _factors(_trim(tuple(F(v) for v in coefficients)))
    while True:
        result = sorted(
            (left, right, multiplicity)
            for p, multiplicity in factors
            for left, right in _isolate_factor(p, tolerance, recognize)
        )
        if all(a[1] < b[0] for a, b in zip(result, result[1:])):
            return tuple(result)
        tolerance /= 2


def _multiply(a: Poly, b: Poly) -> Poly:
    result = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] += x * y
    return tuple(result)


def test_randomized_fraction_reference_parity() -> None:
    rng = Random(194708)
    polynomials: list[Poly] = []
    for degree in range(1, 13):
        # A product provides repeated and midpoint roots as well as complex roots.
        p = (F(rng.choice((-3, 2)), rng.randint(1, 5)),)
        for _ in range(degree):
            root = F(rng.randint(-3, 3), rng.randint(1, 3))
            p = _multiply(p, (-root, F(1)))
        polynomials.append(p)
        # Independent dense rational coefficients exercise general remainders.
        polynomials.append(
            tuple(F(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(degree))
            + (F(1),)
        )
    for p in polynomials:
        isolator = RealRootIsolator(p)
        for recognize in (False, True):
            tolerance = F(1, 8)
            expected = _reference(p, tolerance, recognize)
            assert (
                isolator.isolate(tolerance=tolerance, recognize_rational_roots=recognize)
                == expected
            )
            assert (
                isolate_real_roots(
                    p, tolerance=tolerance, recognize_rational_roots=recognize
                )
                == expected
            )


def test_large_coefficients_and_endpoint_roots() -> None:
    huge = 10**60 + 39
    cases = (
        (F(0), F(-2), F(0), F(1)),
        (F(-huge, 7), F(0), F(huge, 7)),
        (F(2 * huge, 11), F(-3 * huge, 11), F(0), F(huge, 11)),
    )
    for p in cases:
        for recognize in (False, True):
            assert isolate_real_roots(
                p, tolerance=F(1, 16), recognize_rational_roots=recognize
            ) == _reference(p, F(1, 16), recognize)


def test_sturm_remainder_sign_fault_changes_observable_roots() -> None:
    # The rational remainder of x²-2 divided by 2x is -2. Omitting its
    # required negation makes the Sturm variation difference zero at the bounds.
    p = (F(-2), F(0), F(1))
    expected = ((F(-3, 2), F(-1), 1), (F(1), F(3, 2), 1))
    wrong_sequence = [p, _derivative(p), (F(-2),)]
    assert _variation(wrong_sequence, F(-4)) - _variation(wrong_sequence, F(4)) == 0
    assert _reference(p, F(1, 2), False) == expected
    assert (
        isolate_real_roots(p, tolerance=F(1, 2), recognize_rational_roots=False)
        == expected
    )
