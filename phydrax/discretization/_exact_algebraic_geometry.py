from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


if TYPE_CHECKING:
    from ._periodic_cell import PeriodicCell


type RationalScalar = int | np.integer | Fraction
type RadicalTerms = tuple[tuple[int, Fraction], ...]
type ExactScalar = Fraction | RadicalElement


def _rational(value: RationalScalar) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if type(value) is int or isinstance(value, np.integer):
        return Fraction(int(value))
    raise TypeError(f"Expected an exact rational scalar, got {type(value).__name__}.")


def _prime_uint64(value: int, /) -> bool:
    """Deterministic Miller–Rabin decision for nonnegative 64-bit integers.

    The seven bases are the complete unsigned-64-bit witness set published by
    Jim Sinclair and summarized at https://miller-rabin.appspot.com/.
    """
    if value >= (1 << 64):
        raise ValueError("The deterministic word-prime test requires a 64-bit input.")
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0:
        return False
    odd = value - 1
    shifts = 0
    while odd % 2 == 0:
        odd //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        residue = pow(base % value, odd, value)
        if residue in (0, 1, value - 1):
            continue
        for _ in range(shifts - 1):
            residue = residue * residue % value
            if residue == value - 1:
                break
        else:
            return False
    return True


def _squarefree(value: int, /) -> tuple[int, int]:
    """Return ``outside, inside`` with ``value == outside**2 * inside``."""
    if type(value) is not int or value <= 0:
        raise ValueError("Square-free reduction requires a positive integer.")
    root = math.isqrt(value)
    if root * root == value:
        return root, 1
    if value < (1 << 64) and _prime_uint64(value):
        return 1, value
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
            if value < (1 << 64) and _prime_uint64(value):
                return outside, inside * value
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
    """Exact real multiquadratic scalar in a canonical square-free radical basis.

    Instances are immutable values: ``terms`` is assigned once by the canonical
    constructors and attribute assignment or deletion is refused afterwards.
    """

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
        object.__setattr__(self, "terms", _canonical(coefficients))

    @classmethod
    def _from_canonical(cls, terms: Iterable[tuple[int, Fraction]], /) -> RadicalElement:
        result = cls.__new__(cls)
        coefficients: dict[int, Fraction] = {}
        for radicand, coefficient in terms:
            coefficients[radicand] = coefficients.get(radicand, Fraction(0)) + coefficient
        object.__setattr__(
            result,
            "terms",
            tuple(sorted((n, c) for n, c in coefficients.items() if c)),
        )
        return result

    def __setattr__(self, name: str, value: object, /) -> None:
        raise AttributeError("RadicalElement values are immutable.")

    def __delattr__(self, name: str, /) -> None:
        raise AttributeError("RadicalElement values are immutable.")

    def __reduce__(self) -> tuple[object, tuple[RadicalTerms]]:
        return _restore_radical, (self.terms,)

    def __copy__(self) -> RadicalElement:
        return self

    def __deepcopy__(self, memo: dict[int, object], /) -> RadicalElement:
        return self

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

    def __repr__(self) -> str:
        # Content-only: static PyTree metadata strings enter source identities.
        return f"RadicalElement(terms={self.terms!r})"

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


def _restore_radical(terms: RadicalTerms, /) -> RadicalElement:
    """Rebuild one pickled or copied radical, refusing noncanonical terms."""
    restored = RadicalElement(terms=terms)
    if restored.terms != tuple(terms):
        raise ValueError("Stored radical terms are not canonical.")
    return restored


def canonical_exact_scalar(value: ExactScalar, /, *, role: str) -> ExactScalar:
    """Return the canonical exact scalar: rational values as ``Fraction``.

    A ``RadicalElement`` with a purely rational value is the same field element
    as its ``Fraction``; only genuinely irrational values keep radical terms.
    """
    if isinstance(value, Fraction):
        return value
    if type(value) is RadicalElement:
        radical = RadicalElement.from_stored_fields(value)
        if not radical.terms:
            return Fraction(0)
        if len(radical.terms) == 1 and radical.terms[0][0] == 1:
            return radical.terms[0][1]
        return radical
    raise TypeError(f"{role} must be Fraction or RadicalElement values.")


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


class ExactPeriodicAlgebraicSource(StrictModule, NonTrainableState):
    """Immutable exact vertex bank witnessing a periodic affine deck action.

    The retained bank consists of canonical JSON-compatible scalar payloads that
    also form its source identity. Accessors reconstruct exact scalar leaves.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    vertex_global_ids: Array
    rounded_vertices: Array
    _coefficient_payloads: tuple = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: Iterable[Iterable[ExactScalar | RationalScalar]],
        vertex_global_ids: ArrayLike,
        source_id: str,
        source_revision: str,
        /,
    ) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Exact periodic source ID must be nonempty.")
        if not isinstance(source_revision, str) or not source_revision.strip():
            raise ValueError("Exact periodic source revision must be nonempty.")
        ids = np.asarray(vertex_global_ids)
        rows = tuple(tuple(row) for row in coefficients)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or ids.size == 0
            or np.any(ids < 0)
            or np.any(ids > np.iinfo(np.int64).max)
            or np.unique(ids).size != ids.size
            or len(rows) != ids.size
            or not rows[0]
            or any(len(row) != len(rows[0]) for row in rows)
        ):
            raise ValueError("Exact periodic coordinate rows and vertex IDs must align.")
        exact = tuple(
            tuple(
                RadicalElement.from_stored_fields(value)
                if isinstance(value, RadicalElement)
                else RadicalElement(value)
                for value in row
            )
            for row in rows
        )
        payloads = tuple(
            tuple(coefficient_payload(value) for value in row) for row in exact
        )
        rounded = np.asarray(
            [[float(value) for value in row] for row in exact], dtype=np.float64
        )
        if not np.all(np.isfinite(rounded)):
            raise ValueError(
                "Exact periodic coordinates require finite binary64 shadows."
            )
        self.source_id = source_id.strip()
        self.source_revision = source_revision.strip()
        self.vertex_global_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.rounded_vertices = jnp.asarray(rounded)
        self._coefficient_payloads = payloads
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "exact-periodic-algebraic-source",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "vertex_global_ids": tuple(int(value) for value in ids),
                "coefficients": payloads,
            }
        )

    @property
    def coefficients(self) -> tuple[tuple[RadicalElement, ...], ...]:
        return tuple(
            tuple(
                RadicalElement(Fraction(payload[1], payload[2]))
                if payload[0] == "rational"
                else RadicalElement(
                    terms=((n, Fraction(num, den)) for n, num, den in payload[1])
                )
                for payload in row
            )
            for row in self._coefficient_payloads
        )

    def source_coordinates(
        self, indices: ArrayLike | None = None, /
    ) -> tuple[tuple[RadicalElement, ...], ...]:
        exact = self.coefficients
        if indices is None:
            return exact
        rows = np.asarray(indices)
        if rows.ndim != 1 or rows.dtype.kind not in "iu":
            raise TypeError(
                "Exact periodic coordinate selection must be an integer vector."
            )
        if np.any(rows < 0) or np.any(rows >= len(exact)):
            raise IndexError(
                "Exact periodic coordinate selection is outside the vertex bank."
            )
        return tuple(exact[int(row)] for row in rows)

    def require_periodic_lift(
        self,
        coordinates: ArrayLike,
        cell: PeriodicCell,
        representatives: ArrayLike,
        shifts: ArrayLike,
        /,
    ) -> None:
        """Prove the full integer-winding deck action on exact source rows."""
        from ._periodic_exact import exact_periodic_element, identification_orders

        current = type(self).from_stored_fields(self)
        exact = current.source_coordinates()
        points = np.asarray(coordinates, dtype=np.float64)
        if (
            points.shape != (len(exact), len(exact[0]))
            or points.shape[1] != cell.ambient_dimension
            or not np.array_equal(points, np.asarray(current.rounded_vertices))
        ):
            raise ValueError(
                "Exact periodic coordinates differ from their canonical shadows."
            )
        roots = np.asarray(representatives)
        exponents = np.asarray(shifts)
        rank = len(identification_orders(cell))
        if (
            roots.shape != (len(exact),)
            or roots.dtype.kind not in "iu"
            or exponents.shape != (len(exact), rank)
            or exponents.dtype.kind not in "iu"
            or np.any(roots < 0)
            or np.any(roots >= len(exact))
        ):
            raise ValueError("Exact periodic orbit rows are malformed.")
        roots = roots.astype(np.int64)
        if np.any(roots[roots] != roots) or np.any(exponents[roots] != 0):
            raise ValueError("Periodic representatives must be zero-winding roots.")
        transforms = {}
        for row, root in enumerate(roots):
            key = tuple(int(value) for value in exponents[row])
            if key not in transforms:
                transforms[key] = exact_periodic_element(cell, exponents[row])
            matrix = transforms[key]
            base = exact[int(root)] + (RadicalElement(1),)
            expected = tuple(
                sum(
                    (
                        matrix[axis][column] * base[column]
                        for column in range(points.shape[1] + 1)
                    ),
                    RadicalElement(),
                )
                for axis in range(points.shape[1])
            )
            if expected != exact[row]:
                raise ValueError(
                    "Exact periodic coordinates violate their declared deck action."
                )

    @classmethod
    def from_stored_fields(
        cls, value: ExactPeriodicAlgebraicSource, /
    ) -> ExactPeriodicAlgebraicSource:
        if type(value) is not cls:
            raise TypeError("value must be ExactPeriodicAlgebraicSource.")
        rebuilt = cls(
            value.coefficients,
            value.vertex_global_ids,
            value.source_id,
            value.source_revision,
        )
        if (
            rebuilt._coefficient_payloads != value._coefficient_payloads
            or rebuilt.source_map_id != value.source_map_id
            or not np.array_equal(
                np.asarray(rebuilt.rounded_vertices), np.asarray(value.rounded_vertices)
            )
        ):
            raise ValueError(
                "Stored exact periodic source does not reproduce its source leaves."
            )
        return rebuilt


__all__ = [
    "AlgebraicCoordinateBank",
    "ExactPeriodicAlgebraicSource",
    "ExactScalar",
    "RadicalElement",
    "canonical_exact_scalar",
    "coefficient_bits",
    "coefficient_endpoint",
    "coefficient_interval",
    "coefficient_payload",
    "coefficient_rounding_error",
    "rational_coefficient",
]

AlgebraicCoordinateBank = tuple[tuple[RadicalElement, ...], ...]
