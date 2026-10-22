# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real multiquadratic field coefficients and authenticated coordinate banks."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from typing import assert_never, final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..typing import checked
from ._coordinate_budget import (
    coordinate_retention_preparation_bounds,
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ._reference_cell_types import CellGeometryElement


type RadicalTerms = tuple[tuple[int, Fraction], ...]
type RationalScalar = int | np.integer | Fraction
type AlgebraicLayouts = tuple[tuple[str, str, int, str, str], ...]


def rounded_rational_square_root(value: Fraction, /) -> float:
    """Round a nonnegative rational square root to nearest-even binary64.

    The square root's binade determines its binary64 spacing (including
    subnormals). Integer square root gives the lower representable endpoint;
    comparison with the *exact squared midpoint* decides rounding, including
    ties, without a precision heuristic or factoring the rational radicand.
    """
    numerator, denominator = value.numerator, value.denominator
    if numerator < 0:
        raise ValueError("Rational square root must have a nonnegative radicand")
    if numerator == 0:
        return 0.0
    logarithm = numerator.bit_length() - denominator.bit_length()
    if (
        numerator < denominator << logarithm
        if logarithm >= 0
        else numerator << -logarithm < denominator
    ):
        logarithm -= 1
    exponent = logarithm // 2
    if exponent > 1023:
        return math.inf
    spacing = max(exponent - 52, -1074)
    if spacing < 0:
        scaled_numerator = numerator << (-2 * spacing)
        scaled_denominator = denominator
    else:
        scaled_numerator = numerator
        scaled_denominator = denominator << (2 * spacing)
    lower = math.isqrt(scaled_numerator // scaled_denominator)
    comparison = 4 * scaled_numerator - scaled_denominator * (2 * lower + 1) ** 2
    if comparison > 0 or comparison == 0 and lower % 2:
        lower += 1
    try:
        return math.ldexp(float(lower), spacing)
    except OverflowError:
        return math.inf


def _squarefree(
    value: int, /, *, budget: CoordinateEnclosureBudget | None = None
) -> tuple[int, int]:
    """Return integers ``outside, inside`` with value = outside² inside."""
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"RadicalElement.radicand must be a positive integer; got {value!r}"
        )
    root = math.isqrt(value)
    if root * root == value:
        return root, 1
    outside, inside, divisor = 1, 1, 2
    while divisor * divisor <= value:
        if budget is not None:
            budget.reserve(1)
        exponent = 0
        while value % divisor == 0:
            if budget is not None:
                budget.reserve(1)
            value //= divisor
            exponent += 1
        outside *= divisor ** (exponent // 2)
        if exponent % 2:
            inside *= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    return outside, inside * value


def _rational(value: RationalScalar) -> Fraction:
    match value:
        case Fraction():
            return value
        case int() if type(value) is int:
            return Fraction(value)
        case np.integer():
            return Fraction(int(value))
        case _:
            raise TypeError(
                f"RadicalElement.coefficient must be an integer or Fraction; got {value!r}"
            )


def _field_terms(value: RadicalElement | RationalScalar, /) -> RadicalTerms:
    match value:
        case RadicalElement():
            return value.terms
        case Fraction() | int() | np.integer():
            scalar = _rational(value)
            return ((1, scalar),) if scalar else ()
        case _:
            assert_never(value)


def _short_terms_sign(terms: RadicalTerms, /) -> int | None:
    """Decide at most two radical terms by comparing rational squared magnitudes."""
    if not terms:
        return 0
    first, a = terms[0]
    sa = 1 if a > 0 else -1
    if len(terms) == 1:
        return sa
    if len(terms) > 2:
        return None
    second, b = terms[1]
    sb = 1 if b > 0 else -1
    if sa == sb:
        return sa
    norm = a * a * first - b * b * second
    return sa * ((norm > 0) - (norm < 0))


def _difference_sign(
    left: RadicalTerms, other: RadicalElement | RationalScalar, /
) -> int:
    coefficients = dict(left)
    for radicand, coefficient in _field_terms(other):
        coefficients[radicand] = coefficients.get(radicand, Fraction(0)) - coefficient
    terms = tuple(sorted((n, c) for n, c in coefficients.items() if c))
    direct = _short_terms_sign(terms)
    return RadicalElement._from_canonical(terms).sign() if direct is None else direct


@dataclass(frozen=True, slots=True)
class _CanonicalRadicalTerms:
    terms: RadicalTerms


@final
class RadicalElement(StrictModule, NonTrainableState):
    """Store a real multiquadratic value in a squarefree radical basis.

    Distinct positive squarefree radicands are linearly independent over the
    rationals. The empty tuple is zero, and radicand one is the rational part.
    All arithmetic and comparisons preserve this normalized field representation.
    """

    terms: RadicalTerms = eqx.field(static=True)

    def __init__(
        self,
        value: RationalScalar | _CanonicalRadicalTerms = 0,
        /,
        *,
        terms: RadicalTerms = (),
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        if isinstance(value, _CanonicalRadicalTerms):
            if terms:
                raise ValueError(
                    "RadicalElement.terms must use one construction input; got two banks"
                )
            self.terms = value.terms
            return
        coefficients: dict[int, Fraction] = {}
        rational = _rational(value)
        if rational:
            coefficients[1] = rational
        for radicand, coefficient in terms:
            outside, inside = _squarefree(radicand, budget=budget)
            coefficients[inside] = coefficients.get(
                inside, Fraction(0)
            ) + outside * _rational(coefficient)
        self.terms = tuple(sorted((n, c) for n, c in coefficients.items() if c))

    @classmethod
    def _from_canonical(cls, terms: Iterable[tuple[int, Fraction]], /) -> RadicalElement:
        coefficients: dict[int, Fraction] = {}
        for n, c in terms:
            coefficients[n] = coefficients.get(n, Fraction(0)) + c
        return cls(
            _CanonicalRadicalTerms(
                tuple(sorted((n, c) for n, c in coefficients.items() if c))
            )
        )

    @classmethod
    def sqrt(
        cls,
        value: RationalScalar,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> RadicalElement:
        """Return the positive square root of a nonnegative rational value."""
        rational = _rational(value)
        if rational < 0:
            raise ValueError(
                f"RadicalElement.sqrt.value must be nonnegative; got {rational!r}"
            )
        if not rational:
            return cls()
        numerator, numerator_inside = _squarefree(rational.numerator, budget=budget)
        denominator, denominator_inside = _squarefree(rational.denominator, budget=budget)
        return cls._from_canonical(
            (
                (
                    numerator_inside * denominator_inside,
                    Fraction(numerator, denominator * denominator_inside),
                ),
            )
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls,
        value: RadicalElement,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> RadicalElement:
        renewed = cls(terms=value.terms, budget=budget)
        if renewed.terms != value.terms:
            raise ValueError(
                "RadicalElement.terms must retain normalized nonzero squarefree coefficients; got an unnormalized restored bank"
            )
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
        match other:
            case RadicalElement():
                return self.terms == other.terms
            case Fraction() | int() | np.integer():
                return self.terms == _field_terms(other)
            case _:
                return NotImplemented

    def __neg__(self) -> RadicalElement:
        return RadicalElement._from_canonical((n, -c) for n, c in self.terms)

    def __pos__(self) -> RadicalElement:
        return self

    def __add__(self, other: RadicalElement | RationalScalar, /) -> RadicalElement:
        return RadicalElement._from_canonical(self.terms + _field_terms(other))

    def __radd__(self, other: RationalScalar, /) -> RadicalElement:
        return self + other

    def __sub__(self, other: RadicalElement | RationalScalar, /) -> RadicalElement:
        return RadicalElement._from_canonical(
            (*self.terms, *((n, -c) for n, c in _field_terms(other)))
        )

    def __rsub__(self, other: RationalScalar, /) -> RadicalElement:
        return RadicalElement._from_canonical(
            (*((n, -c) for n, c in self.terms), *_field_terms(other))
        )

    def __mul__(self, other: RadicalElement | RationalScalar, /) -> RadicalElement:
        if not isinstance(other, RadicalElement):
            scalar = _rational(other)
            return RadicalElement._from_canonical((n, c * scalar) for n, c in self.terms)
        terms = []
        for left, a in self.terms:
            for right, b in other.terms:
                common = math.gcd(left, right)
                terms.append(((left // common) * (right // common), a * b * common))
        return RadicalElement._from_canonical(terms)

    def __rmul__(self, other: RationalScalar, /) -> RadicalElement:
        return self * other

    def _split(self) -> tuple[RadicalElement, RadicalElement, int]:
        # Generate square classes by gcd multiplication. Choosing the final
        # independent generator gives a quadratic extension of the preceding
        # span, without factoring normalized radicands during arithmetic.
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
            raise RuntimeError(
                "RadicalElement.quadratic_split must contain a nonrational coefficient; got a rational value"
            )
        left, right = [], []
        for radicand, coefficient in self.terms:
            if radicand in subfield:
                left.append((radicand, coefficient))
            else:
                common = math.gcd(radicand, generator)
                reduced = (radicand // common) * (generator // common)
                right.append((reduced, coefficient * Fraction(common, generator)))
        return (
            RadicalElement._from_canonical(left),
            RadicalElement._from_canonical(right),
            generator,
        )

    def reciprocal(self) -> RadicalElement:
        """Invert by recursive quadratic conjugation in a smaller field."""
        if not self:
            raise ZeroDivisionError("RadicalElement.divisor must be nonzero; got zero")
        if len(self.terms) == 1:
            radicand, coefficient = self.terms[0]
            return RadicalElement._from_canonical(
                ((radicand, 1 / (coefficient * radicand)),)
            )
        a, b, generator = self._split()
        norm = a * a - generator * b * b
        conjugate = a - b * RadicalElement._from_canonical(((generator, Fraction(1)),))
        return conjugate * norm.reciprocal()

    def __truediv__(self, other: RadicalElement | RationalScalar, /) -> RadicalElement:
        if isinstance(other, RadicalElement):
            return self * other.reciprocal()
        return self * (1 / _rational(other))

    def __rtruediv__(self, other: RationalScalar, /) -> RadicalElement:
        return self.reciprocal() * other

    def __pow__(self, exponent: int, /) -> RadicalElement:
        if type(exponent) is not int:
            raise TypeError(
                f"RadicalElement.exponent must be an integer; got {exponent!r}"
            )
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
        """Return the real sign using recursive quadratic norms without rounding."""
        direct = _short_terms_sign(self.terms)
        if direct is not None:
            return direct
        a, b, generator = self._split()
        sa, sb = a.sign(), b.sign()
        if not sa:
            return sb
        if not sb or sa == sb:
            return sa
        return sa * (a * a - generator * b * b).sign()

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

    def interval(
        self,
        /,
        *,
        bits: int = 80,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[Fraction, Fraction]:
        """Enclose the value by rational square-root endpoints proved by squaring."""
        bits = positive_integer(bits, "RadicalElement.interval.bits")
        if budget is not None:
            budget.reserve(
                4 * len(self.terms),
                8 * len(self.terms) * (256 + (self.coefficient_bits + 2 * bits + 7) // 8),
            )
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

    def rounded(
        self, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> tuple[float, Fraction]:
        """Return round-to-nearest-even binary64 and a rational absolute error bound."""
        if not self:
            return 0.0, Fraction(0)
        if len(self.terms) == 1 and self.terms[0][0] == 1:
            value = self.terms[0][1]
            representative = float(value)
            return representative, abs(value - Fraction(representative))
        bits = 80
        info = np.finfo(np.float64)
        overflow_midpoint = Fraction(float(info.max)) + Fraction(2) ** (
            info.maxexp - info.nmant - 2
        )
        while True:
            lower, upper = self.interval(bits=bits, budget=budget)
            if lower >= overflow_midpoint or upper <= -overflow_midpoint:
                raise OverflowError(
                    "RadicalElement.representative must fit finite binary64; got an overflowing field value"
                )
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

    def rounding_error(self) -> Fraction:
        return self.rounded()[1]

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


type AlgebraicCoordinateBank = tuple[tuple[RadicalElement, ...], ...]


@checked
def rational_coefficient(
    value: Fraction | RadicalElement,
    /,
    *,
    role: str,
) -> Fraction:
    """Return a rational subfield coefficient without rounding its value."""
    match value:
        case Fraction():
            return value
        case RadicalElement():
            if not value.terms:
                return Fraction(0)
            if len(value.terms) == 1 and value.terms[0][0] == 1:
                return value.terms[0][1]
            raise ValueError(
                f"{role} must lie in the rational subfield; got a nonrational radical coefficient"
            )
        case _:
            assert_never(value)


def coefficient_interval(
    value: Fraction | RadicalElement,
    /,
    *,
    bits: int = 80,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[Fraction, Fraction]:
    """Return rational endpoints without replacing an algebraic coefficient."""
    return (
        value.interval(bits=bits, budget=budget)
        if isinstance(value, RadicalElement)
        else (value, value)
    )


def coefficient_endpoint(
    value: Fraction | RadicalElement,
    /,
    *,
    upper: bool,
    saturate: bool = False,
    budget: CoordinateEnclosureBudget | None = None,
) -> float:
    """Return a directed binary64 endpoint of a real field coefficient."""
    from ..geometry._rational import fraction_endpoint

    if isinstance(value, Fraction):
        return fraction_endpoint(value, upper=upper, saturate=saturate)
    if saturate:
        maximum = float(np.finfo(np.float64).max)
        if value > Fraction(maximum):
            return math.inf if upper else maximum
        if value < -Fraction(maximum):
            return -maximum if upper else -math.inf
    rounded = value.rounded(budget=budget)[0]
    if upper:
        return (
            float(np.nextafter(rounded, math.inf))
            if value > Fraction(rounded)
            else rounded
        )
    return (
        float(np.nextafter(rounded, -math.inf)) if value < Fraction(rounded) else rounded
    )


def coefficient_rounding_error(
    value: Fraction | RadicalElement,
    representative: float,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> Fraction:
    """Bound distance from a field coefficient to a supplied binary64 carrier."""
    if isinstance(value, RadicalElement):
        rounded, error = value.rounded(budget=budget)
        return error + abs(Fraction(rounded) - Fraction(representative))
    return abs(value - Fraction(representative))


def coefficient_bits(value: Fraction | RadicalElement | int | np.integer, /) -> int:
    match value:
        case RadicalElement():
            return value.coefficient_bits
        case Fraction():
            return max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        case int() | np.integer():
            return max(1, abs(int(value)).bit_length())
        case _:
            raise TypeError(
                f"coefficient_bits.value must be an exact rational or radical scalar; got {type(value).__name__}"
            )


def _field_degree(radicands: set[int], /) -> int:
    """Return the square-class rank without factoring the radicands."""
    atoms: list[int] = []
    for radicand in sorted(radicands - {1}):
        remaining = radicand
        for index in range(len(atoms)):
            atom = atoms[index]
            common = math.gcd(atom, remaining)
            if common == 1:
                continue
            if common != atom:
                atoms[index] = common
                atoms.append(atom // common)
            remaining //= common
        if remaining != 1:
            atoms.append(remaining)
    pivots: dict[int, int] = {}
    for radicand in sorted(radicands - {1}):
        vector = sum(
            1 << index for index, atom in enumerate(atoms) if radicand % atom == 0
        )
        while vector:
            pivot = vector.bit_length() - 1
            previous = pivots.get(pivot)
            if previous is None:
                pivots[pivot] = vector
                break
            vector ^= previous
    return 1 << len(pivots)


def reserve_algebraic_work(
    values: Iterable[
        Fraction
        | RadicalElement
        | int
        | np.integer
        | tuple[Fraction | RadicalElement | int | np.integer, ...]
        | np.ndarray
    ],
    work: int,
    count: int,
    degree: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Admit coefficient products before performing field arithmetic."""
    bits, terms = 1, 1
    radicands: set[int] = set()
    for entry in values:
        row = entry if isinstance(entry, (tuple, np.ndarray)) else (entry,)
        for value in row:
            bits = max(bits, coefficient_bits(value))
            if isinstance(value, RadicalElement):
                terms = max(terms, len(value.terms))
                radicands.update(n for n, _ in value.terms)
                bits = max(
                    bits,
                    value.coefficient_bits
                    + sum(c.denominator.bit_length() for _, c in value.terms),
                )
    field_degree = _field_degree(radicands)
    products = min(terms ** min(degree, field_degree.bit_length()), field_degree)
    # A scalar product visits at most D² rational pairs. Recursive quadratic
    # norms/signs and reciprocal elimination fit a cubic field-degree bound.
    scalar_work = max(products, field_degree**3)
    height = max(degree, field_degree) * bits + max(1, products.bit_length()) + 1
    budget.reserve(
        work * scalar_work,
        count * field_degree * (256 + 4 * ((height + 7) // 8)),
    )


def _rounding_precision_bound(value: RadicalElement, /) -> int:
    # Clear coefficient denominators and binary64 midpoint denominators. The
    # resulting nonzero algebraic integer has a nonzero integer field norm.
    # Bound the remaining conjugates by the sum of their absolute terms.
    count = len(value.terms)
    denominator_bits = sum(c.denominator.bit_length() for _, c in value.terms)
    height = denominator_bits + 2 * value.coefficient_bits + 2102 + count.bit_length()
    degree = _field_degree({n for n, _ in value.terms})
    return degree * height + value.coefficient_bits + count.bit_length() + 2


def algebraic_source_limits(
    coordinates: tuple[tuple[RadicalElement | Fraction | int, ...], ...],
    /,
) -> tuple[int, int]:
    """Derive source work and coefficient-bit limits from the field-valued input bank."""
    work, bits = 1, 1
    for row in coordinates:
        for scalar in row:
            value = (
                scalar if isinstance(scalar, RadicalElement) else RadicalElement(scalar)
            )
            bits = max(bits, value.coefficient_bits)
            refinements = max(1, _rounding_precision_bound(value).bit_length())
            work += 4 * len(value.terms) * refinements + 1
    return work, bits


@dataclass(frozen=True, slots=True)
class PreparedExactAlgebraicGeometry:
    vertices: AlgebraicCoordinateBank
    on_entity: tuple[bool, ...]
    source_id: str
    operation_count: int
    maximum_integer_bits: int
    maximum_rounding_error: Fraction


@final
class ExactAlgebraicCellGeometrySource(StrictModule, NonTrainableState):
    """Bind field coefficients, nominal layouts, and a declared source identity."""

    coefficients: AlgebraicCoordinateBank = eqx.field(static=True)
    layouts: AlgebraicLayouts = eqx.field(static=True)
    source_identity: str = eqx.field(static=True)
    authority_binding: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        coordinates: tuple[tuple[RadicalElement | Fraction | int, ...], ...],
        layouts: AlgebraicLayouts,
        /,
        *,
        source_identity: str,
        maximum_work: int,
        maximum_bits: int,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        from . import _cell_geometry as geometry_owner

        maximum_work = positive_integer(maximum_work, "maximum_work")
        maximum_bits = positive_integer(maximum_bits, "maximum_bits")
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError(
                f"ExactAlgebraicCellGeometrySource.source_identity must be nonempty; got {source_identity!r}"
            )
        if (
            not coordinates
            or not coordinates[0]
            or any(len(row) != len(coordinates[0]) for row in coordinates)
        ):
            raise ValueError(
                "ExactAlgebraicCellGeometrySource.coefficients must be a nonempty rectangular field-valued bank; got an empty or ragged bank"
            )
        if (
            not layouts
            or tuple(sorted(layouts)) != layouts
            or len({row[0] for row in layouts}) != len(layouts)
        ):
            raise ValueError(
                f"ExactAlgebraicCellGeometrySource.layouts must be nonempty, unique and sorted; got {layouts!r}"
            )
        for name, kind, degree, identity, route_id in layouts:
            geometry_owner.coordinate_lagrange_element(kind, degree)
            if not name or not identity or not route_id:
                raise ValueError(
                    "ExactAlgebraicCellGeometrySource.layouts must retain block, element and route identities; got an empty identity"
                )
        bank = tuple(
            tuple(
                RadicalElement.from_stored_fields(value, budget=budget)
                if isinstance(value, RadicalElement)
                else RadicalElement(value, budget=budget)
                for value in row
            )
            for row in coordinates
        )
        bits = max(value.coefficient_bits for row in bank for value in row)
        if bits > maximum_bits:
            raise CoordinateEnclosureResourceError(
                "coefficient_work", maximum_bits, bits, 0
            )
        terms = sum(len(value.terms) + 1 for row in bank for value in row)
        basis = (bank, layouts, source_identity)
        if budget is None:
            work_bound, memory_bound = coordinate_retention_preparation_bounds(basis)
            preparation = geometry_owner.owned_coordinate_preparation(
                terms + work_bound, memory_bound
            )
        else:
            preparation = nullcontext(budget)
        with preparation as admission:
            admission.reserve(terms)
            admission.retain_basis(basis)
        self.coefficients, self.layouts = bank, layouts
        self.source_identity = source_identity
        self.maximum_work, self.maximum_bits = maximum_work, maximum_bits
        self.authority_binding = self._binding()

    def _binding(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "exact-algebraic-cell-source",
                "source_identity": self.source_identity,
                "coefficients": tuple(
                    tuple(
                        tuple((n, c.numerator, c.denominator) for n, c in value.terms)
                        for value in row
                    )
                    for row in self.coefficients
                ),
                "layouts": self.layouts,
                "maximum_work": self.maximum_work,
                "maximum_bits": self.maximum_bits,
            }
        )

    @property
    def source_id(self) -> str:
        return self.authority_binding

    @property
    def field_degree(self) -> int:
        return _field_degree(
            {n for row in self.coefficients for value in row for n, _ in value.terms}
        )

    def preparation_query_upper(self) -> int:
        return 0

    def _preparation_resources(self) -> tuple[int, int, int]:
        count = sum(len(value.terms) + 1 for row in self.coefficients for value in row)
        largest = max(len(value.terms) + 1 for row in self.coefficients for value in row)
        precision = max(
            80,
            max(_rounding_precision_bound(value) for row in self.coefficients for value in row),
        )
        retained = (
            4096 * (count + len(self.layouts) + 1)
            + 128 * count * ((self.maximum_bits + 7) // 8)
        )
        scalar = 128 * largest * ((max(self.maximum_bits, precision) + 7) // 8)
        work = algebraic_source_limits(self.coefficients)[0] + 128 * (count + len(self.layouts) + 1)
        return work, retained, scalar

    def preparation_bounds(self) -> tuple[int, int]:
        work, retained, scalar = self._preparation_resources()
        # Keep the original complete rounding work; only one interval circuit
        # and the retained maximum-error scalar coexist with the numeric bank.
        return work, retained + 2 * scalar

    @classmethod
    @checked
    def from_stored_fields(
        cls,
        value: ExactAlgebraicCellGeometrySource,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> ExactAlgebraicCellGeometrySource:
        renewed = cls(
            value.coefficients,
            value.layouts,
            source_identity=value.source_identity,
            maximum_work=value.maximum_work,
            maximum_bits=value.maximum_bits,
            budget=budget,
        )
        if renewed.authority_binding != value.authority_binding:
            raise ValueError(
                "ExactAlgebraicCellGeometrySource.authority_binding must authenticate restored coefficients, layouts and source identity; got a mismatched binding"
            )
        return renewed

    def representatives(self, *, budget: CoordinateEnclosureBudget) -> np.ndarray:
        if self._binding() != self.authority_binding:
            raise ValueError(
                "ExactAlgebraicCellGeometrySource.authority_binding must authenticate coefficients, layouts and source identity; got a mismatched binding"
            )
        count = sum(len(row) for row in self.coefficients)
        with budget.temporary_scope():
            budget.reserve(count, 8 * count)
            numeric = np.empty((len(self.coefficients), len(self.coefficients[0])), dtype=np.float64)
            for row, values in enumerate(self.coefficients):
                for axis, value in enumerate(values):
                    with budget.temporary_scope():
                        numeric[row, axis] = value.rounded(budget=budget)[0]
            budget.retain_basis((numeric,))
        return numeric

    def prepare(
        self, rounded_vertices: ArrayLike, /, *, budget: CoordinateEnclosureBudget
    ) -> PreparedExactAlgebraicGeometry:
        if self._binding() != self.authority_binding:
            raise ValueError(
                "ExactAlgebraicCellGeometrySource.authority_binding must authenticate coefficients, layouts and source identity; got a mismatched binding"
            )
        rounded = np.asarray(rounded_vertices, dtype=np.float64)
        shape = (len(self.coefficients), len(self.coefficients[0]))
        if rounded.shape != shape or not np.all(np.isfinite(rounded)):
            raise ValueError(
                f"ExactAlgebraicCellGeometrySource.representatives must be a finite aligned bank; got shape {rounded.shape!r} for {shape!r}"
            )
        from .._fingerprint import array_tree_fingerprint

        key = (self.source_id, canonical_fingerprint(array_tree_fingerprint(rounded)))
        cached = budget.algebraic_witness_cache.get(key)
        if cached is not None:
            return cached
        work_upper, retained_upper, scalar_upper = self._preparation_resources()
        budget.admit_work_bound(work_upper, storage_upper=retained_upper + 2 * scalar_upper)
        start = budget.work_units
        with budget.temporary_scope():
            budget.reserve(shape[0] * shape[1], 8 * shape[0] * shape[1] + scalar_upper)
            representatives = np.empty(shape, dtype=np.float64)
            maximum_error = Fraction(0)
            for row, values in enumerate(self.coefficients):
                for axis, value in enumerate(values):
                    # Intervals and rejected precision refinements die with
                    # this scalar. Only its float and the running maximum error
                    # survive; no all-coordinate (float, error) bank is built.
                    with budget.temporary_scope():
                        representative, error = value.rounded(budget=budget)
                        representatives[row, axis] = representative
                        maximum_error = max(maximum_error, error)
            if not np.array_equal(representatives.view(np.uint64), rounded.view(np.uint64)):
                raise ValueError(
                    "ExactAlgebraicCellGeometrySource.representatives must correctly round the declared field coefficients; got differing binary64 bytes"
                )
            work = budget.work_units - start
            if work > self.maximum_work:
                raise CoordinateEnclosureResourceError(
                    "coefficient_work", self.maximum_work, work, 0
                )
            prepared = PreparedExactAlgebraicGeometry(
                self.coefficients,
                (False,) * shape[0],
                self.source_id,
                work,
                max(value.coefficient_bits for row in self.coefficients for value in row),
                maximum_error,
            )
            budget.retain_basis(
                (key, self.coefficients, prepared.on_entity, prepared.maximum_rounding_error, prepared)
            )
            budget.algebraic_witness_cache[key] = prepared
        return prepared

    @checked
    def reindexed(
        self,
        indices: ArrayLike,
        elements: Mapping[str, CellGeometryElement],
        routes: Mapping[str, ArrayLike],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> ExactAlgebraicCellGeometrySource:
        from ._exact_rational_geometry import rational_source_layouts

        selected = np.asarray(indices)
        if (
            selected.ndim != 1
            or selected.dtype.kind not in "iu"
            or np.any(selected < 0)
            or np.any(selected >= len(self.coefficients))
            or np.any(selected[1:] <= selected[:-1])
        ):
            raise ValueError(
                "ExactAlgebraicCellGeometrySource.indices must be sorted unique declared coefficient rows; got an invalid route"
            )
        if self._binding() != self.authority_binding:
            raise ValueError(
                "ExactAlgebraicCellGeometrySource.authority_binding must authenticate coefficients before reindexing; got a mismatched binding"
            )
        return ExactAlgebraicCellGeometrySource(
            tuple(self.coefficients[int(index)] for index in selected),
            rational_source_layouts(elements, routes),
            source_identity=self.source_identity,
            maximum_work=self.maximum_work,
            maximum_bits=self.maximum_bits,
            budget=budget,
        )
