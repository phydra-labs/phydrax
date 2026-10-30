from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction

import numpy as np


type RationalScalar = int | np.integer | Fraction
type RadicalTerms = tuple[tuple[int, Fraction], ...]
type ExactScalar = Fraction | RadicalElement


def _rational(value: RationalScalar) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if type(value) is int or isinstance(value, np.integer):
        return Fraction(int(value))
    raise TypeError(f"Expected an exact rational scalar, got {type(value).__name__}.")


def _squarefree(value: int, /) -> tuple[int, int]:
    """Return ``outside, inside`` with ``value == outside**2 * inside``."""
    if type(value) is not int or value <= 0:
        raise ValueError("Square-free reduction requires a positive integer.")
    root = math.isqrt(value)
    if root * root == value:
        return root, 1
    outside = inside = 1
    divisor = 2
    while divisor * divisor <= value:
        exponent = 0
        while value % divisor == 0:
            value //= divisor
            exponent += 1
        if exponent:
            outside *= divisor ** (exponent // 2)
            if exponent % 2:
                inside *= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    return outside, inside * value


def _canonical(terms: Iterable[tuple[int, Fraction]], /) -> RadicalTerms:
    coefficients: dict[int, Fraction] = {}
    for radicand, coefficient in terms:
        outside, inside = _squarefree(int(radicand))
        coefficients[inside] = (
            coefficients.get(inside, Fraction(0)) + outside * coefficient
        )
    return tuple(sorted((n, c) for n, c in coefficients.items() if c))


def _field_terms(value: RadicalElement | RationalScalar, /) -> RadicalTerms:
    if isinstance(value, RadicalElement):
        return value.terms
    rational = _rational(value)
    return ((1, rational),) if rational else ()


def _short_sign(terms: RadicalTerms, /) -> int | None:
    if not terms:
        return 0
    first, a = terms[0]
    sign_a = 1 if a > 0 else -1
    if len(terms) == 1:
        return sign_a
    if len(terms) > 2:
        return None
    second, b = terms[1]
    sign_b = 1 if b > 0 else -1
    if sign_a == sign_b:
        return sign_a
    norm = a * a * first - b * b * second
    return sign_a * ((norm > 0) - (norm < 0))


def _difference_sign(
    left: RadicalTerms, other: RadicalElement | RationalScalar, /
) -> int:
    coefficients = dict(left)
    for radicand, coefficient in _field_terms(other):
        coefficients[radicand] = coefficients.get(radicand, Fraction(0)) - coefficient
    terms = tuple(sorted((n, c) for n, c in coefficients.items() if c))
    direct = _short_sign(terms)
    return RadicalElement._from_canonical(terms).sign() if direct is None else direct


class RadicalElement:
    """Exact real multiquadratic scalar in a canonical square-free radical basis."""

    __slots__ = ("terms",)

    terms: RadicalTerms

    def __init__(
        self,
        value: RationalScalar = 0,
        /,
        *,
        terms: Iterable[tuple[int, RationalScalar]] = (),
    ) -> None:
        coefficients: list[tuple[int, Fraction]] = []
        rational = _rational(value)
        if rational:
            coefficients.append((1, rational))
        for radicand, coefficient in terms:
            coefficients.append((int(radicand), _rational(coefficient)))
        self.terms = _canonical(coefficients)

    @classmethod
    def _from_canonical(cls, terms: Iterable[tuple[int, Fraction]], /) -> RadicalElement:
        result = cls.__new__(cls)
        coefficients: dict[int, Fraction] = {}
        for radicand, coefficient in terms:
            coefficients[radicand] = coefficients.get(radicand, Fraction(0)) + coefficient
        result.terms = tuple(sorted((n, c) for n, c in coefficients.items() if c))
        return result

    @classmethod
    def sqrt(cls, value: RationalScalar, /) -> RadicalElement:
        rational = _rational(value)
        if rational < 0:
            raise ValueError("A real radical requires a nonnegative rational radicand.")
        if not rational:
            return cls()
        outside, inside = _squarefree(rational.numerator * rational.denominator)
        return cls._from_canonical(((inside, Fraction(outside, rational.denominator)),))

    @classmethod
    def from_stored_fields(cls, value: RadicalElement, /) -> RadicalElement:
        if type(value) is not cls:
            raise TypeError("Stored algebraic coefficients must be RadicalElement.")
        renewed = cls(terms=value.terms)
        if renewed.terms != value.terms:
            raise ValueError("Stored radical terms are not canonical.")
        return renewed

    @property
    def coefficient_bits(self) -> int:
        return max(
            (
                max(
                    n.bit_length(),
                    abs(c.numerator).bit_length(),
                    c.denominator.bit_length(),
                )
                for n, c in self.terms
            ),
            default=1,
        )

    def __bool__(self) -> bool:
        return bool(self.terms)

    def __hash__(self) -> int:
        if not self.terms:
            return hash(Fraction(0))
        if len(self.terms) == 1 and self.terms[0][0] == 1:
            return hash(self.terms[0][1])
        return hash(self.terms)

    def __eq__(self, other: object, /) -> bool:
        if isinstance(other, RadicalElement):
            return self.terms == other.terms
        if isinstance(other, (Fraction, int, np.integer)):
            return self.terms == _field_terms(other)
        return NotImplemented

    def __neg__(self) -> RadicalElement:
        return self._from_canonical((n, -c) for n, c in self.terms)

    def __pos__(self) -> RadicalElement:
        return self

    def __add__(self, other: RadicalElement | RationalScalar, /) -> RadicalElement:
        return self._from_canonical((*self.terms, *_field_terms(other)))

    def __radd__(self, other: RationalScalar, /) -> RadicalElement:
        return self + other

    def __sub__(self, other: RadicalElement | RationalScalar, /) -> RadicalElement:
        return self._from_canonical(
            (*self.terms, *((n, -c) for n, c in _field_terms(other)))
        )

    def __rsub__(self, other: RationalScalar, /) -> RadicalElement:
        return self._from_canonical(
            (*((n, -c) for n, c in self.terms), *_field_terms(other))
        )

    def __mul__(self, other: RadicalElement | RationalScalar, /) -> RadicalElement:
        if not isinstance(other, RadicalElement):
            scalar = _rational(other)
            return self._from_canonical((n, c * scalar) for n, c in self.terms)
        terms = []
        for left, a in self.terms:
            for right, b in other.terms:
                common = math.gcd(left, right)
                terms.append(((left // common) * (right // common), a * b * common))
        return self._from_canonical(terms)

    def __rmul__(self, other: RationalScalar, /) -> RadicalElement:
        return self * other

    def _split(self) -> tuple[RadicalElement, RadicalElement, int]:
        span = {1}
        subfield = {1}
        generator = 1
        for radicand, _ in self.terms:
            if radicand in span:
                continue
            subfield, generator = span, radicand
            span = span | {
                (radicand // math.gcd(radicand, value))
                * (value // math.gcd(radicand, value))
                for value in span
            }
        if generator == 1:
            raise ArithmeticError(
                "A nonrational radical split needs an extension generator."
            )
        left, right = [], []
        for radicand, coefficient in self.terms:
            if radicand in subfield:
                left.append((radicand, coefficient))
            else:
                common = math.gcd(radicand, generator)
                reduced = (radicand // common) * (generator // common)
                right.append((reduced, coefficient * Fraction(common, generator)))
        return self._from_canonical(left), self._from_canonical(right), generator

    def reciprocal(self) -> RadicalElement:
        if not self:
            raise ZeroDivisionError(
                "Cannot invert zero in an algebraic coefficient field."
            )
        if len(self.terms) == 1:
            radicand, coefficient = self.terms[0]
            return self._from_canonical(((radicand, 1 / (coefficient * radicand)),))
        a, b, generator = self._split()
        norm = a * a - generator * b * b
        conjugate = a - b * self._from_canonical(((generator, Fraction(1)),))
        return conjugate * norm.reciprocal()

    def __truediv__(self, other: RadicalElement | RationalScalar, /) -> RadicalElement:
        if isinstance(other, RadicalElement):
            return self * other.reciprocal()
        return self * (1 / _rational(other))

    def __rtruediv__(self, other: RationalScalar, /) -> RadicalElement:
        return self.reciprocal() * other

    def __pow__(self, exponent: int, /) -> RadicalElement:
        if type(exponent) is not int:
            raise TypeError("Algebraic exponents must be integers.")
        if exponent < 0:
            return self.reciprocal() ** -exponent
        result, base = RadicalElement(1), self
        while exponent:
            if exponent & 1:
                result *= base
            exponent >>= 1
            if exponent:
                base *= base
        return result

    def sign(self) -> int:
        direct = _short_sign(self.terms)
        if direct is not None:
            return direct
        a, b, generator = self._split()
        sign_a, sign_b = a.sign(), b.sign()
        if not sign_a:
            return sign_b
        if not sign_b or sign_a == sign_b:
            return sign_a
        return sign_a * (a * a - generator * b * b).sign()

    def __lt__(self, other: RadicalElement | RationalScalar, /) -> bool:
        return _difference_sign(self.terms, other) < 0

    def __le__(self, other: RadicalElement | RationalScalar, /) -> bool:
        return _difference_sign(self.terms, other) <= 0

    def __gt__(self, other: RadicalElement | RationalScalar, /) -> bool:
        return _difference_sign(self.terms, other) > 0

    def __ge__(self, other: RadicalElement | RationalScalar, /) -> bool:
        return _difference_sign(self.terms, other) >= 0

    def __abs__(self) -> RadicalElement:
        return -self if self.sign() < 0 else self

    def interval(self, /, *, bits: int = 80) -> tuple[Fraction, Fraction]:
        """Rational outward enclosure, proved by integer square-root bounds."""
        if isinstance(bits, bool) or not isinstance(bits, int) or bits < 1:
            raise ValueError("interval bits must be a positive integer.")
        lower = upper = Fraction(0)
        scale = 1 << bits
        for radicand, coefficient in self.terms:
            root = math.isqrt(radicand << (2 * bits))
            lo = Fraction(root, scale)
            hi = (
                lo if root * root == radicand << (2 * bits) else Fraction(root + 1, scale)
            )
            if coefficient < 0:
                lo, hi = hi, lo
            lower += coefficient * lo
            upper += coefficient * hi
        return lower, upper

    def rounded(self, /) -> tuple[float, Fraction]:
        """Correctly rounded finite binary64 view and a rational absolute error bound."""
        if not self:
            return 0.0, Fraction(0)
        if len(self.terms) == 1 and self.terms[0][0] == 1:
            value = self.terms[0][1]
            representative = float(value)
            return representative, abs(value - Fraction(representative))
        maximum = float(np.finfo(np.float64).max)
        overflow_midpoint = Fraction(maximum) + Fraction(2) ** 970
        bits = 80
        while True:
            lower, upper = self.interval(bits=bits)
            if lower >= overflow_midpoint or upper <= -overflow_midpoint:
                raise OverflowError("Algebraic value exceeds finite binary64 range.")
            if lower <= -overflow_midpoint or upper >= overflow_midpoint:
                bits *= 2
                continue
            lo, hi = float(lower), float(upper)
            if lo == hi and math.copysign(1.0, lo) == math.copysign(1.0, hi):
                carrier = Fraction(lo)
                return lo, max(abs(lower - carrier), abs(upper - carrier))
            bits *= 2

    def __float__(self) -> float:
        return self.rounded()[0]

    def __floor__(self) -> int:
        bits = 80
        lower, upper = self.interval(bits=bits)
        while upper - lower >= 1:
            bits *= 2
            lower, upper = self.interval(bits=bits)
        result = math.floor(lower)
        while self >= result + 1:
            result += 1
        return result

    def __ceil__(self) -> int:
        return -math.floor(-self)


def rational_coefficient(value: ExactScalar, /, *, role: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, RadicalElement):
        if not value.terms:
            return Fraction(0)
        if len(value.terms) == 1 and value.terms[0][0] == 1:
            return value.terms[0][1]
        raise ValueError(
            f"{role} must be rational, got a nonrational radical coefficient."
        )
    raise TypeError("Exact coefficient must be Fraction or RadicalElement.")


def coefficient_payload(
    value: ExactScalar, /
) -> tuple[str, int, int] | tuple[str, tuple[tuple[int, int, int], ...]]:
    """Encode exact coefficients as canonical JSON-compatible source identity."""
    if isinstance(value, RadicalElement) and (
        not value.terms or (len(value.terms) == 1 and value.terms[0][0] == 1)
    ):
        value = rational_coefficient(value, role="source identity coefficient")
    if isinstance(value, Fraction):
        return "rational", value.numerator, value.denominator
    if isinstance(value, RadicalElement):
        return "radical", tuple(
            (radicand, coefficient.numerator, coefficient.denominator)
            for radicand, coefficient in value.terms
        )
    raise TypeError("Exact coefficient must be Fraction or RadicalElement.")


def coefficient_interval(
    value: ExactScalar, /, *, bits: int = 80
) -> tuple[Fraction, Fraction]:
    return (
        value.interval(bits=bits) if isinstance(value, RadicalElement) else (value, value)
    )


def coefficient_endpoint(
    value: ExactScalar,
    /,
    *,
    upper: bool,
    saturate: bool = False,
) -> float:
    """Directed binary64 endpoint of an exact rational or algebraic scalar."""
    from ..geometry._rational import fraction_endpoint

    if isinstance(value, Fraction):
        return fraction_endpoint(value, upper=upper, saturate=saturate)
    if saturate:
        maximum = float(np.finfo(np.float64).max)
        if value > Fraction(maximum):
            return math.inf if upper else maximum
        if value < -Fraction(maximum):
            return -maximum if upper else -math.inf
    rounded = value.rounded()[0]
    carrier = Fraction(rounded)
    if upper and value > carrier:
        return float(np.nextafter(rounded, math.inf))
    if not upper and value < carrier:
        return float(np.nextafter(rounded, -math.inf))
    return rounded


def coefficient_rounding_error(value: ExactScalar, representative: float, /) -> Fraction:
    rounded = float(value) if isinstance(value, RadicalElement) else float(value)
    return abs(Fraction(rounded) - Fraction(representative)) + (
        value.rounded()[1]
        if isinstance(value, RadicalElement)
        else abs(value - Fraction(rounded))
    )


def coefficient_bits(value: ExactScalar | int | np.integer, /) -> int:
    if isinstance(value, RadicalElement):
        return value.coefficient_bits
    rational = _rational(value)
    return max(1, abs(rational.numerator).bit_length(), rational.denominator.bit_length())


__all__ = [
    "AlgebraicCoordinateBank",
    "ExactScalar",
    "RadicalElement",
    "coefficient_bits",
    "coefficient_endpoint",
    "coefficient_interval",
    "coefficient_payload",
    "coefficient_rounding_error",
    "rational_coefficient",
]

AlgebraicCoordinateBank = tuple[tuple[RadicalElement, ...], ...]
