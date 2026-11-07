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


def _primitive_remainder(left: tuple[int, ...], right: tuple[int, ...], /) -> tuple[int, ...]:
    """Positive multiple of the rational remainder, with primitive integer content."""
    if right == (0,):
        raise ZeroDivisionError("Polynomial division by zero.")
    numerator = list(left)
    while len(numerator) >= len(right) and any(numerator):
        offset = len(numerator) - len(right)
        leading = numerator[-1]
        divisor_leading = right[-1]
        scale = abs(divisor_leading)
        factor = leading if divisor_leading > 0 else -leading
        numerator = [scale * value for value in numerator]
        for index, value in enumerate(right):
            numerator[offset + index] -= factor * value
        while len(numerator) > 1 and numerator[-1] == 0:
            numerator.pop()
        content = 0
        for value in numerator:
            content = gcd(content, abs(value))
        if content > 1:
            numerator = [value // content for value in numerator]
    return tuple(numerator)


def _gcd(left: Polynomial, right: Polynomial, /) -> Polynomial:
    left_ = _integer_coefficients(_trim(left))
    right_ = _integer_coefficients(_trim(right))
    while right_ != (0,):
        left_, right_ = right_, _primitive_remainder(left_, right_)
    return _monic(tuple(Fraction(value) for value in left_))


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


def _has_no_rational_roots_modulo_prime(coefficients: tuple[int, ...], /) -> bool:
    # For a primitive integer polynomial, a rational root a/b has b dividing
    # the leading coefficient. At a prime not dividing that coefficient it
    # therefore reduces to a root modulo p. Failure to find one is a proof,
    # not a heuristic; inconclusive primes leave reconstruction unchanged.
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if coefficients[-1] % prime == 0:
            continue
        for point in range(prime):
            value = 0
            for coefficient in reversed(coefficients):
                value = (value * point + coefficient) % prime
            if value == 0:
                break
        else:
            return True
    return False


def _sturm_sequence(values: Polynomial, /) -> tuple[tuple[int, ...], ...]:
    # Positive rescaling preserves every Sturm sign and avoids carrying the
    # potentially enormous common rational denominators through evaluation.
    first = _integer_coefficients(values)
    second = _integer_coefficients(_derivative(tuple(Fraction(value) for value in first)))
    sequence = [first, second]
    while sequence[-1] != (0,):
        remainder = _primitive_remainder(sequence[-2], sequence[-1])
        if remainder == (0,):
            break
        sequence.append(tuple(-value for value in remainder))
    return tuple(sequence)


class _SturmState:
    """Exact Sturm isolation state of one monic square-free polynomial.

    Every derived quantity is a pure function of the polynomial. Polynomial signs
    and Sturm variation counts at a point are memoized for the lifetime of the
    owning isolator only, so repeated subdivision under a finer tolerance
    revisits the same dyadic nodes without re-evaluating them.
    """

    __slots__ = (
        "_deflations",
        "_no_rational_roots",
        "_signs",
        "_variations",
        "lower",
        "maximum_root_denominator",
        "polynomial",
        "rational_separation",
        "sequence",
        "upper",
    )

    def __init__(self, polynomial: Polynomial, /) -> None:
        self.polynomial = polynomial
        self._deflations: dict[Fraction, _SturmState] = {}
        self._signs: dict[Fraction, int] = {}
        self._variations: dict[Fraction, int] = {}
        self._no_rational_roots: bool | None = None
        if _degree(polynomial) <= 1:
            return
        integer_coefficients = _integer_coefficients(polynomial)
        # A reduced rational root's denominator divides the primitive leading
        # coefficient. Distinct such rationals with denominators <= Q differ by
        # at least 1/Q**2; a narrower interval therefore permits continued-
        # fraction reconstruction, which is accepted only after exact substitution.
        self.maximum_root_denominator = abs(integer_coefficients[-1])
        self.rational_separation = Fraction(
            1, 2 * self.maximum_root_denominator * self.maximum_root_denominator
        )
        self.sequence = _sturm_sequence(polynomial)
        leading = abs(polynomial[-1])
        bound = Fraction(1) + max(
            (abs(value) / leading for value in polynomial[:-1]),
            default=Fraction(0),
        )
        bound = Fraction(
            (bound.numerator + bound.denominator - 1) // bound.denominator + 1
        )
        lower, upper = -bound, bound
        while not self.sign(lower) or not self.sign(upper):
            lower -= 1
            upper += 1
        self.lower, self.upper = lower, upper

    def no_rational_roots(self) -> bool:
        if self._no_rational_roots is None:
            self._no_rational_roots = _has_no_rational_roots_modulo_prime(
                self.sequence[0]
            )
        return self._no_rational_roots

    def _values(self, point: Fraction, rows: int, /) -> list[int]:
        # Every Sturm polynomial has integer coefficients. Evaluate it over the
        # positive common denominator q**degree with integer Horner; a dyadic
        # denominator multiplies by shifts instead of big-integer products.
        numerator, denominator = point.numerator, point.denominator
        shift = denominator.bit_length() - 1
        dyadic = denominator == 1 << shift
        if not dyadic:
            powers = [1]
            for _ in range(len(self.sequence[0]) - 1):
                powers.append(powers[-1] * denominator)
        values = []
        for polynomial in self.sequence[:rows]:
            value = polynomial[-1]
            if dyadic:
                offset = shift
                for coefficient in reversed(polynomial[:-1]):
                    value = value * numerator + (coefficient << offset)
                    offset += shift
            else:
                for power, coefficient in enumerate(reversed(polynomial[:-1]), 1):
                    value = value * numerator + coefficient * powers[power]
            values.append(value)
        return values

    def sign(self, point: Fraction, /) -> int:
        """Sign of the polynomial at ``point`` (the first row is a positive multiple)."""
        cached = self._signs.get(point)
        if cached is None:
            (value,) = self._values(point, 1)
            self._signs[point] = cached = (value > 0) - (value < 0)
        return cached

    def _variation(self, point: Fraction, /) -> int:
        cached = self._variations.get(point)
        if cached is None:
            values = self._values(point, len(self.sequence))
            signs = [value > 0 for value in values if value]
            cached = sum(
                left != right for left, right in zip(signs[:-1], signs[1:], strict=True)
            )
            self._variations[point] = cached
        return cached

    def root_count(self, lower: Fraction, upper: Fraction, /) -> int:
        """Number of distinct real roots in ``(lower, upper]`` by Sturm's theorem."""
        return self._variation(lower) - self._variation(upper)

    def deflate(self, root: Fraction, /) -> _SturmState:
        child = self._deflations.get(root)
        if child is None:
            child = _SturmState(_exact_quotient(self.polynomial, (-root, Fraction(1))))
            self._deflations[root] = child
        return child


def _sturm_root_intervals(
    state: _SturmState,
    tolerance: Fraction,
    /,
    *,
    recognize_rational_roots: bool,
) -> tuple[tuple[Fraction, Fraction], ...]:
    exact_roots: list[tuple[Fraction, Fraction]] = []
    while _degree(state.polynomial) > 0:
        polynomial = state.polynomial
        if _degree(polynomial) == 1:
            root = -polynomial[0] / polynomial[1]
            exact_roots.append((root, root))
            break
        can_accept_interval = not recognize_rational_roots or state.no_rational_roots()
        lower, upper = state.lower, state.upper
        pending = [(lower, upper, state.root_count(lower, upper))]
        intervals: list[tuple[Fraction, Fraction]] = []
        exact_root: Fraction | None = None
        while pending:
            left, right, count = pending.pop()
            if count == 0:
                continue
            midpoint = (left + right) / 2
            if not state.sign(midpoint):
                exact_root = midpoint
                break
            if count == 1 and right - left <= tolerance:
                if can_accept_interval:
                    intervals.append((left, right))
                    continue
                if right - left < state.rational_separation:
                    reconstructed = midpoint.limit_denominator(
                        state.maximum_root_denominator
                    )
                    if left <= reconstructed <= right and not state.sign(reconstructed):
                        exact_root = reconstructed
                        break
                    intervals.append((left, right))
                    continue
            # Every endpoint is a nonzero bound or an already tested nonzero
            # midpoint. With exactly one simple root inside, Sturm's count on the
            # left half is therefore exactly whether the polynomial changes sign.
            left_count = (
                int(state.sign(left) != state.sign(midpoint))
                if count == 1
                else state.root_count(left, midpoint)
            )
            right_count = count - left_count
            pending.append((midpoint, right, right_count))
            pending.append((left, midpoint, left_count))
        if exact_root is None:
            return tuple(sorted((*exact_roots, *intervals)))
        exact_roots.append((exact_root, exact_root))
        state = state.deflate(exact_root)
    return tuple(sorted(exact_roots))


class RealRootIsolator:
    """Exact real-root isolation of one fixed polynomial at any tolerance.

    ``isolate`` returns exactly what :func:`isolate_real_roots` returns for the
    same coefficients and arguments. The square-free factorization, Sturm
    sequences and every exact sign evaluation are computed once per isolator and
    reused across calls, so refining one polynomial under shrinking tolerances
    does not repeat finished subdivision. The memo is owned by this object and
    bound to its polynomial; no state is shared between isolators.
    """

    __slots__ = ("_factors", "_polynomial", "_results")

    def __init__(self, coefficients: tuple[int | Fraction, ...], /) -> None:
        polynomial = _trim(tuple(Fraction(value) for value in coefficients))
        if polynomial == (0,):
            raise ValueError("A nonzero coefficient sequence is required.")
        self._polynomial = polynomial
        self._factors: tuple[tuple[_SturmState, int], ...] | None = None
        self._results: dict[
            tuple[Fraction, bool], tuple[tuple[Fraction, Fraction, int], ...]
        ] = {}

    def isolate(
        self, *, tolerance: Fraction, recognize_rational_roots: bool = True
    ) -> tuple[tuple[Fraction, Fraction, int], ...]:
        """Isolate all real roots with multiplicity in pairwise disjoint intervals.

        By default every rational root is identified as a singleton. If
        ``recognize_rational_roots`` is false, nonsingleton intervals may also
        contain rational roots; callers must not infer irrationality from them.
        Dyadic roots encountered during subdivision still produce singletons.
        """
        key = (tolerance, bool(recognize_rational_roots))
        cached = self._results.get(key)
        if cached is not None:
            return cached
        if self._factors is None:
            self._factors = tuple(
                (_SturmState(factor), multiplicity)
                for factor, multiplicity in _square_free_factors(self._polynomial)
            )
        width = key[0]
        while True:
            result = sorted(
                (lower, upper, multiplicity)
                for state, multiplicity in self._factors
                for lower, upper in _sturm_root_intervals(
                    state, width, recognize_rational_roots=key[1]
                )
            )
            # Different multiplicity factors (and rational roots removed before
            # re-isolation) must still have globally separated, ordered intervals.
            if all(left[1] < right[0] for left, right in zip(result, result[1:])):
                self._results[key] = cached = tuple(result)
                return cached
            width /= 2


def isolate_real_roots(
    coefficients: tuple[int | Fraction, ...],
    /,
    *,
    tolerance: Fraction,
    recognize_rational_roots: bool = True,
) -> tuple[tuple[Fraction, Fraction, int], ...]:
    """Isolate all real roots with multiplicity in pairwise disjoint intervals.

    By default every rational root is identified as a singleton. If
    ``recognize_rational_roots`` is false, nonsingleton intervals may also
    contain rational roots; callers must not infer irrationality from them.
    Dyadic roots encountered during subdivision still produce singletons.
    Callers refining one polynomial repeatedly should hold a
    :class:`RealRootIsolator` instead.
    """
    return RealRootIsolator(coefficients).isolate(
        tolerance=tolerance, recognize_rational_roots=recognize_rational_roots
    )


__all__ = ["RealRootIsolator", "isolate_real_roots"]
