#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only decimal interval arithmetic and strict positivity certificates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
    localcontext,
    ROUND_CEILING,
    ROUND_FLOOR,
)
from typing import TypeAlias

from .._fingerprint import canonical_fingerprint


DecimalLike: TypeAlias = Decimal | int | float | str


def _decimal(value: DecimalLike, /) -> Decimal:
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, float):
        result = Decimal(str(value))
    else:
        result = Decimal(value)
    if not result.is_finite():
        raise ValueError("Validated decimal values must be finite.")
    return result


def _directed_binary(
    left: Decimal,
    right: Decimal,
    operation: str,
    precision: int,
    rounding: str,
    /,
) -> Decimal:
    with localcontext() as context:
        context.prec = precision
        context.rounding = rounding
        if operation == "add":
            return left + right
        if operation == "subtract":
            return left - right
        if operation == "multiply":
            return left * right
        if operation == "divide":
            return left / right
    raise ValueError("Unknown directed decimal operation.")


@dataclass(frozen=True, slots=True)
class DecimalInterval:
    """Closed real interval with outward-rounded decimal operations."""

    lower: Decimal
    upper: Decimal
    precision: int
    interval_id: str

    def __init__(
        self,
        lower: DecimalLike,
        upper: DecimalLike | None = None,
        /,
        *,
        precision: int = 80,
    ):
        lower_ = _decimal(lower)
        upper_ = lower_ if upper is None else _decimal(upper)
        precision_ = int(precision)
        if precision_ < 18:
            raise ValueError("Decimal interval precision must be at least 18 digits.")
        if lower_ > upper_:
            raise ValueError("Decimal interval endpoints must be ordered.")
        content = {
            "kind": "decimal-interval",
            "lower": str(lower_),
            "upper": str(upper_),
            "precision": precision_,
        }
        object.__setattr__(self, "lower", lower_)
        object.__setattr__(self, "upper", upper_)
        object.__setattr__(self, "precision", precision_)
        object.__setattr__(self, "interval_id", canonical_fingerprint(content))

    @classmethod
    def point(cls, value: DecimalLike, /, *, precision: int = 80) -> DecimalInterval:
        return cls(value, precision=precision)

    @property
    def width(self) -> Decimal:
        return self.upper - self.lower

    @property
    def midpoint(self) -> Decimal:
        with localcontext() as context:
            context.prec = self.precision
            return (self.lower + self.upper) / Decimal(2)

    @property
    def contains_zero(self) -> bool:
        return self.lower <= 0 <= self.upper

    def contains(self, value: DecimalLike, /) -> bool:
        resolved = _decimal(value)
        return self.lower <= resolved <= self.upper

    def abs_upper(self) -> Decimal:
        return max(abs(self.lower), abs(self.upper))

    def _coerce(self, other: DecimalInterval | DecimalLike, /) -> DecimalInterval:
        if isinstance(other, DecimalInterval):
            return other
        return DecimalInterval.point(other, precision=self.precision)

    def __neg__(self) -> DecimalInterval:
        return DecimalInterval(-self.upper, -self.lower, precision=self.precision)

    def __add__(self, other: DecimalInterval | DecimalLike) -> DecimalInterval:
        rhs = self._coerce(other)
        precision = max(self.precision, rhs.precision)
        lower = _directed_binary(
            self.lower,
            rhs.lower,
            "add",
            precision,
            ROUND_FLOOR,
        )
        upper = _directed_binary(
            self.upper,
            rhs.upper,
            "add",
            precision,
            ROUND_CEILING,
        )
        return DecimalInterval(lower, upper, precision=precision)

    __radd__ = __add__

    def __sub__(self, other: DecimalInterval | DecimalLike) -> DecimalInterval:
        return self + (-self._coerce(other))

    def __rsub__(self, other: DecimalInterval | DecimalLike) -> DecimalInterval:
        return self._coerce(other) - self

    def __mul__(self, other: DecimalInterval | DecimalLike) -> DecimalInterval:
        rhs = self._coerce(other)
        precision = max(self.precision, rhs.precision)
        lower_products = (
            _directed_binary(a, b, "multiply", precision, ROUND_FLOOR)
            for a in (self.lower, self.upper)
            for b in (rhs.lower, rhs.upper)
        )
        upper_products = (
            _directed_binary(a, b, "multiply", precision, ROUND_CEILING)
            for a in (self.lower, self.upper)
            for b in (rhs.lower, rhs.upper)
        )
        return DecimalInterval(
            min(lower_products),
            max(upper_products),
            precision=precision,
        )

    __rmul__ = __mul__

    def reciprocal(self) -> DecimalInterval:
        if self.contains_zero:
            raise ZeroDivisionError("Cannot invert an interval containing zero.")
        lower = _directed_binary(
            Decimal(1),
            self.upper,
            "divide",
            self.precision,
            ROUND_FLOOR,
        )
        upper = _directed_binary(
            Decimal(1),
            self.lower,
            "divide",
            self.precision,
            ROUND_CEILING,
        )
        if lower > upper:
            lower, upper = upper, lower
        return DecimalInterval(lower, upper, precision=self.precision)

    def __truediv__(self, other: DecimalInterval | DecimalLike) -> DecimalInterval:
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other: DecimalInterval | DecimalLike) -> DecimalInterval:
        return self._coerce(other) * self.reciprocal()

    def __pow__(self, exponent: int) -> DecimalInterval:
        power = int(exponent)
        if power < 0:
            return (self.reciprocal()) ** (-power)
        result = DecimalInterval.point(1, precision=self.precision)
        base = self
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result


@dataclass(frozen=True, slots=True)
class IntervalPSDCertificate:
    """Strict diagonal-dominance certificate for a symmetric interval matrix."""

    dimension: int
    lower_margin: Decimal
    symmetric: bool
    positive_semidefinite: bool
    precision: int
    certificate_id: str


def certify_interval_psd(
    matrix: Sequence[Sequence[DecimalInterval | DecimalLike]],
    /,
    *,
    precision: int = 80,
) -> IntervalPSDCertificate:
    """Certify PSD by a rigorous Gershgorin lower bound.

    This intentionally abstains for PSD matrices that are not certifiable by strict
    interval diagonal dominance. It never upgrades an inconclusive matrix.
    """

    rows = tuple(tuple(row) for row in matrix)
    dimension = len(rows)
    if dimension == 0 or any(len(row) != dimension for row in rows):
        raise ValueError("Interval PSD certificates require a non-empty square matrix.")
    values = tuple(
        tuple(
            item
            if isinstance(item, DecimalInterval)
            else DecimalInterval.point(item, precision=precision)
            for item in row
        )
        for row in rows
    )
    symmetric = all(
        values[i][j].lower == values[j][i].lower
        and values[i][j].upper == values[j][i].upper
        for i in range(dimension)
        for j in range(dimension)
    )
    margins: list[Decimal] = []
    for i in range(dimension):
        radius = sum(
            (values[i][j].abs_upper() for j in range(dimension) if j != i),
            Decimal(0),
        )
        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_FLOOR
            margins.append(values[i][i].lower - radius)
    lower_margin = min(margins)
    positive = symmetric and lower_margin >= 0
    content = {
        "kind": "interval-psd-certificate",
        "dimension": dimension,
        "lower_margin": str(lower_margin),
        "symmetric": symmetric,
        "positive_semidefinite": positive,
        "precision": int(precision),
        "matrix": [
            [[str(value.lower), str(value.upper)] for value in row] for row in values
        ],
    }
    return IntervalPSDCertificate(
        dimension,
        lower_margin,
        symmetric,
        positive,
        int(precision),
        canonical_fingerprint(content),
    )


def evaluate_interval_polynomial(
    coefficients: Sequence[DecimalInterval | DecimalLike],
    argument: DecimalInterval | DecimalLike,
    /,
    *,
    precision: int = 80,
) -> DecimalInterval:
    """Evaluate ascending-order coefficients by outward-rounded Horner iteration."""

    values = tuple(coefficients)
    if not values:
        raise ValueError("Interval polynomial coefficients must be non-empty.")
    point = (
        argument
        if isinstance(argument, DecimalInterval)
        else DecimalInterval.point(argument, precision=precision)
    )
    result = DecimalInterval.point(0, precision=max(precision, point.precision))
    try:
        for coefficient in reversed(values):
            result = result * point + coefficient
    except InvalidOperation as error:
        raise ValueError(
            "Interval polynomial evaluation was numerically invalid."
        ) from error
    return result


__all__ = [
    "DecimalInterval",
    "DecimalLike",
    "IntervalPSDCertificate",
    "certify_interval_psd",
    "evaluate_interval_polynomial",
]
