#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from operator import index

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._resources import AlgebraResourceBudget


RationalPair = tuple[int, int]
AlgebraTerm = tuple[int, int, int, int, int]


def _fraction(value: RationalPair | int | Fraction, /) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("Algebra rational values must not be Boolean.")
    if isinstance(value, tuple):
        if len(value) != 2 or any(isinstance(item, bool) for item in value):
            raise TypeError("Rational pairs must contain integer numerator/denominator.")
        numerator = index(value[0])
        denominator = index(value[1])
        if denominator == 0:
            raise ValueError("Algebra rational denominator must be nonzero.")
        return Fraction(numerator, denominator)
    return Fraction(index(value), 1)


def _pair(value: Fraction, /) -> RationalPair:
    return int(value.numerator), int(value.denominator)


class AlgebraRationalVector(StrictModule, NonTrainableState):
    entries: tuple[RationalPair, ...] = eqx.field(static=True)
    vector_id: str = eqx.field(static=True)

    def __init__(self, entries: Sequence[RationalPair | int | Fraction], /):
        values = tuple(_fraction(value) for value in entries)
        if not values:
            raise ValueError("Algebra rational vectors must be non-empty.")
        self.entries = tuple(_pair(value) for value in values)
        self.vector_id = canonical_fingerprint(
            {"kind": "algebra-rational-vector-v1", "entries": self.entries}
        )

    @property
    def fractions(self) -> tuple[Fraction, ...]:
        return tuple(
            Fraction(numerator, denominator) for numerator, denominator in self.entries
        )


class AlgebraRationalMap(StrictModule, NonTrainableState):
    rows: tuple[tuple[RationalPair, ...], ...] = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(self, rows: Sequence[Sequence[RationalPair | int | Fraction]], /):
        values = tuple(tuple(_fraction(value) for value in row) for row in rows)
        if not values or any(len(row) != len(values) for row in values):
            raise ValueError("Algebra rational maps must be non-empty and square.")
        self.rows = tuple(tuple(_pair(value) for value in row) for row in values)
        self.map_id = canonical_fingerprint(
            {"kind": "algebra-rational-map-v1", "rows": self.rows}
        )

    @property
    def dimension(self) -> int:
        return len(self.rows)

    @property
    def fractions(self) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(Fraction(numerator, denominator) for numerator, denominator in row)
            for row in self.rows
        )

    def apply(self, vector: Sequence[Fraction], /) -> tuple[Fraction, ...]:
        values = tuple(vector)
        if len(values) != self.dimension:
            raise ValueError("Algebra map input has the wrong coordinate dimension.")
        return tuple(
            sum(
                (
                    coefficient * value
                    for coefficient, value in zip(row, values, strict=True)
                ),
                Fraction(0),
            )
            for row in self.fractions
        )


class AlgebraStructureTable(StrictModule, NonTrainableState):
    coordinate_dimension: int = eqx.field(static=True)
    terms: tuple[AlgebraTerm, ...] = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate_dimension: int,
        terms: Sequence[AlgebraTerm],
        /,
        *,
        budget: AlgebraResourceBudget,
    ):
        if isinstance(coordinate_dimension, bool):
            raise TypeError("Algebra coordinate dimension must be an integer.")
        dimension = index(coordinate_dimension)
        budget.admit_coordinates(dimension)
        budget.admit_product_pairs(dimension * dimension)
        combined: dict[tuple[int, int, int], Fraction] = {}
        for raw in terms:
            if len(raw) != 5 or any(isinstance(value, bool) for value in raw):
                raise TypeError("Algebra terms must be five integer values.")
            left, right, output, numerator, denominator = tuple(
                index(value) for value in raw
            )
            if not (
                0 <= left < dimension
                and 0 <= right < dimension
                and 0 <= output < dimension
            ):
                raise ValueError("Algebra term index lies outside the coordinate basis.")
            coefficient = _fraction((numerator, denominator))
            key = (left, right, output)
            combined[key] = combined.get(key, Fraction(0)) + coefficient
        normalized = tuple(
            (left, right, output, coefficient.numerator, coefficient.denominator)
            for (left, right, output), coefficient in sorted(combined.items())
            if coefficient
        )
        plan_bytes = len(normalized) * 5 * 8
        budget.admit_product(len(normalized), plan_bytes)
        self.coordinate_dimension = dimension
        self.terms = normalized
        self.table_id = canonical_fingerprint(
            {
                "kind": "algebra-structure-table-v1",
                "dimension": dimension,
                "terms": normalized,
            }
        )

    @property
    def term_count(self) -> int:
        return len(self.terms)

    def basis_product(self, left: int, right: int, /) -> tuple[Fraction, ...]:
        if isinstance(left, bool) or isinstance(right, bool):
            raise TypeError("Algebra basis indices must be integers.")
        left_ = index(left)
        right_ = index(right)
        if not (
            0 <= left_ < self.coordinate_dimension
            and 0 <= right_ < self.coordinate_dimension
        ):
            raise ValueError("Algebra basis index lies outside the coordinate basis.")
        output = [Fraction(0) for _ in range(self.coordinate_dimension)]
        for left_index, right_index, output_index, numerator, denominator in self.terms:
            if left_index == left_ and right_index == right_:
                output[output_index] += Fraction(numerator, denominator)
        return tuple(output)

    def multiply(
        self,
        left: Sequence[Fraction],
        right: Sequence[Fraction],
        /,
    ) -> tuple[Fraction, ...]:
        left_ = tuple(left)
        right_ = tuple(right)
        if (
            len(left_) != self.coordinate_dimension
            or len(right_) != self.coordinate_dimension
        ):
            raise ValueError("Algebra product input has the wrong coordinate dimension.")
        output = [Fraction(0) for _ in range(self.coordinate_dimension)]
        for left_index, right_index, output_index, numerator, denominator in self.terms:
            output[output_index] += (
                left_[left_index] * right_[right_index] * Fraction(numerator, denominator)
            )
        return tuple(output)


__all__ = [
    "AlgebraRationalMap",
    "AlgebraRationalVector",
    "AlgebraStructureTable",
    "AlgebraTerm",
    "RationalPair",
]
