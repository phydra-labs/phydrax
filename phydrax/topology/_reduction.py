#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ._coefficients import PrimeField
from ._complex import CompactBoundary
from ._resources import TopologyResourceError, TopologyResourcePolicy


FieldVector = dict[int, int]
RationalVector = dict[int, Fraction]


@dataclass
class ReductionStats:
    operations: int = 0
    peak_entries: int = 0
    representative_entries: int = 0
    maximum_bit_length: int = 0

    def observe(
        self,
        columns: list[FieldVector] | tuple[FieldVector, ...],
        policy: TopologyResourcePolicy,
        /,
    ) -> None:
        entries = sum(len(column) for column in columns)
        self.peak_entries = max(self.peak_entries, entries)
        if entries > policy.max_reduction_entries:
            raise TopologyResourceError(
                "Exact topology reduction exceeded max_reduction_entries."
            )

    def operation(self, policy: TopologyResourcePolicy, count: int = 1, /) -> None:
        self.operations += int(count)
        if self.operations > policy.max_operations:
            raise TopologyResourceError(
                "Exact topology reduction exceeded max_operations."
            )

    def representatives(
        self,
        values: list[FieldVector] | tuple[FieldVector, ...],
        policy: TopologyResourcePolicy,
        /,
    ) -> None:
        self.representative_entries = max(
            self.representative_entries,
            sum(len(value) for value in values),
        )
        if self.representative_entries > policy.max_representative_entries:
            raise TopologyResourceError(
                "Exact topology representatives exceed max_representative_entries."
            )

    def rational(self, value: Fraction, policy: TopologyResourcePolicy, /) -> None:
        bits = max(value.numerator.bit_length(), value.denominator.bit_length())
        self.maximum_bit_length = max(self.maximum_bit_length, bits)
        if bits > policy.max_rational_bit_length:
            raise TopologyResourceError(
                "Exact rational reduction exceeded max_rational_bit_length."
            )


def field_columns(boundary: CompactBoundary, field: PrimeField, /) -> list[FieldVector]:
    columns = [dict() for _ in range(boundary.column_count)]
    for row, column, coefficient in zip(
        np.asarray(boundary.row_indices),
        np.asarray(boundary.column_indices),
        np.asarray(boundary.coefficients),
        strict=True,
    ):
        target = columns[int(column)]
        value = field.add(target.get(int(row), 0), int(coefficient))
        if value:
            target[int(row)] = value
        elif int(row) in target:
            del target[int(row)]
    return columns


def transpose_columns(
    columns: list[FieldVector],
    row_count: int,
    field: PrimeField,
    /,
) -> list[FieldVector]:
    transposed = [dict() for _ in range(int(row_count))]
    for column_index, column in enumerate(columns):
        for row, value in column.items():
            normalized = field.normalize(value)
            if normalized:
                transposed[row][column_index] = normalized
    return transposed


def _add_scaled(
    target: FieldVector,
    source: FieldVector,
    scale: int,
    field: PrimeField,
    stats: ReductionStats,
    policy: TopologyResourcePolicy,
    /,
) -> None:
    normalized_scale = field.normalize(scale)
    if normalized_scale == 0:
        return
    for index, source_value in source.items():
        value = field.subtract(
            target.get(index, 0),
            field.multiply(normalized_scale, source_value),
        )
        if value:
            target[index] = value
        elif index in target:
            del target[index]
        stats.operation(policy)


def _scale(
    vector: FieldVector,
    scale: int,
    field: PrimeField,
    stats: ReductionStats,
    policy: TopologyResourcePolicy,
    /,
) -> None:
    for index in tuple(vector):
        vector[index] = field.multiply(vector[index], scale)
        stats.operation(policy)


@dataclass
class ColumnReduction:
    reduced: list[FieldVector]
    transformations: list[FieldVector] | None
    pivot_to_column: dict[int, int]
    zero_columns: tuple[int, ...]
    stats: ReductionStats


def reduce_columns(
    columns: list[FieldVector],
    field: PrimeField,
    policy: TopologyResourcePolicy,
    /,
    *,
    track_transformations: bool,
    stats: ReductionStats | None = None,
) -> ColumnReduction:
    state = ReductionStats() if stats is None else stats
    reduced: list[FieldVector] = []
    transformations = [] if track_transformations else None
    pivot_to_column: dict[int, int] = {}
    zeros = []
    for column_index, source_column in enumerate(columns):
        column = dict(source_column)
        transform = {column_index: 1} if track_transformations else None
        while column:
            pivot = max(column)
            owner = pivot_to_column.get(pivot)
            if owner is None:
                inverse = field.inverse(column[pivot])
                _scale(column, inverse, field, state, policy)
                if transform is not None:
                    _scale(transform, inverse, field, state, policy)
                pivot_to_column[pivot] = column_index
                break
            factor = column[pivot]
            _add_scaled(
                column,
                reduced[owner],
                factor,
                field,
                state,
                policy,
            )
            if transform is not None and transformations is not None:
                _add_scaled(
                    transform,
                    transformations[owner],
                    factor,
                    field,
                    state,
                    policy,
                )
        if not column:
            zeros.append(column_index)
        reduced.append(column)
        if transformations is not None and transform is not None:
            transformations.append(transform)
        state.observe(reduced, policy)
        if transformations is not None:
            state.representatives(transformations, policy)
    return ColumnReduction(
        reduced,
        transformations,
        pivot_to_column,
        tuple(zeros),
        state,
    )


class FieldVectorBasis:
    """Incremental exact basis in one finite-field coordinate space."""

    def __init__(
        self,
        field: PrimeField,
        policy: TopologyResourcePolicy,
        stats: ReductionStats,
    ):
        self.field = field
        self.policy = policy
        self.stats = stats
        self._vectors: dict[int, FieldVector] = {}

    @property
    def dimension(self) -> int:
        return len(self._vectors)

    def reduce(self, value: FieldVector, /) -> FieldVector:
        vector = dict(value)
        while vector:
            pivot = max(vector)
            basis = self._vectors.get(pivot)
            if basis is None:
                break
            _add_scaled(
                vector,
                basis,
                vector[pivot],
                self.field,
                self.stats,
                self.policy,
            )
        return vector

    def add(self, value: FieldVector, /) -> tuple[bool, FieldVector]:
        vector = self.reduce(value)
        if not vector:
            return False, vector
        pivot = max(vector)
        inverse = self.field.inverse(vector[pivot])
        _scale(vector, inverse, self.field, self.stats, self.policy)
        self._vectors[pivot] = vector
        self.stats.observe(list(self._vectors.values()), self.policy)
        return True, vector


def homology_representatives(
    boundary: list[FieldVector],
    incoming_boundary: list[FieldVector],
    field: PrimeField,
    policy: TopologyResourcePolicy,
    /,
    *,
    stats: ReductionStats | None = None,
) -> tuple[tuple[FieldVector, ...], int, int, ReductionStats]:
    state = ReductionStats() if stats is None else stats
    outgoing = reduce_columns(
        boundary,
        field,
        policy,
        track_transformations=True,
        stats=state,
    )
    if outgoing.transformations is None:
        raise RuntimeError("Kernel extraction requires tracked transformations.")
    cycles = [outgoing.transformations[index] for index in outgoing.zero_columns]
    boundary_basis = FieldVectorBasis(field, policy, state)
    for column in incoming_boundary:
        boundary_basis.add(column)
    boundary_rank = boundary_basis.dimension
    quotient_basis = FieldVectorBasis(field, policy, state)
    for column in incoming_boundary:
        quotient_basis.add(column)
    representatives = []
    for cycle in cycles:
        independent, canonical = quotient_basis.add(cycle)
        if independent:
            representatives.append(canonical)
    state.representatives(representatives, policy)
    return (
        tuple(representatives),
        len(outgoing.pivot_to_column),
        boundary_rank,
        state,
    )


def field_rank(
    columns: list[FieldVector],
    field: PrimeField,
    policy: TopologyResourcePolicy,
    /,
    *,
    stats: ReductionStats | None = None,
) -> tuple[int, ReductionStats]:
    reduction = reduce_columns(
        columns,
        field,
        policy,
        track_transformations=False,
        stats=stats,
    )
    return len(reduction.pivot_to_column), reduction.stats


def rational_rank(
    boundary: CompactBoundary,
    policy: TopologyResourcePolicy,
    /,
) -> tuple[int, ReductionStats]:
    stats = ReductionStats()
    basis: dict[int, RationalVector] = {}
    columns: list[RationalVector] = [dict() for _ in range(boundary.column_count)]
    for row, column, coefficient in zip(
        np.asarray(boundary.row_indices),
        np.asarray(boundary.column_indices),
        np.asarray(boundary.coefficients),
        strict=True,
    ):
        target = columns[int(column)]
        target[int(row)] = target.get(int(row), Fraction(0)) + Fraction(int(coefficient))
        if target[int(row)] == 0:
            del target[int(row)]
    for source in columns:
        vector = dict(source)
        while vector:
            pivot = max(vector)
            owner = basis.get(pivot)
            if owner is None:
                coefficient = vector[pivot]
                for index in tuple(vector):
                    vector[index] /= coefficient
                    stats.rational(vector[index], policy)
                    stats.operation(policy)
                basis[pivot] = vector
                break
            factor = vector[pivot]
            for index, owner_value in owner.items():
                value = vector.get(index, Fraction(0)) - factor * owner_value
                stats.rational(value, policy)
                stats.operation(policy)
                if value:
                    vector[index] = value
                elif index in vector:
                    del vector[index]
        entries = sum(len(value) for value in basis.values())
        stats.peak_entries = max(stats.peak_entries, entries)
        if entries > policy.max_reduction_entries:
            raise TopologyResourceError(
                "Exact rational reduction exceeded max_reduction_entries."
            )
    return len(basis), stats


def verify_boundary_composition(
    lower: list[FieldVector],
    upper: list[FieldVector],
    field: PrimeField,
    policy: TopologyResourcePolicy,
    /,
) -> ReductionStats:
    stats = ReductionStats()
    for upper_column in upper:
        composition: FieldVector = {}
        for middle, coefficient in upper_column.items():
            if middle >= len(lower):
                raise ValueError("Consecutive compact boundary dimensions do not match.")
            source = lower[middle]
            for row, value in source.items():
                total = field.add(
                    composition.get(row, 0),
                    field.multiply(coefficient, value),
                )
                if total:
                    composition[row] = total
                elif row in composition:
                    del composition[row]
                stats.operation(policy)
        if composition:
            raise ValueError("Compact boundaries violate boundary-of-boundary zero.")
    return stats


def integer_columns(boundary: CompactBoundary, /) -> list[dict[int, int]]:
    """Return coalesced arbitrary-precision integer boundary columns."""
    columns: list[dict[int, int]] = [{} for _ in range(boundary.column_count)]
    for row, column, coefficient in zip(
        np.asarray(boundary.row_indices),
        np.asarray(boundary.column_indices),
        np.asarray(boundary.coefficients),
        strict=True,
    ):
        target = columns[int(column)]
        value = target.get(int(row), 0) + int(coefficient)
        if value:
            target[int(row)] = value
        elif int(row) in target:
            del target[int(row)]
    return columns


def verify_integer_boundary_composition(
    lower: list[dict[int, int]],
    upper: list[dict[int, int]],
    policy: TopologyResourcePolicy,
    /,
) -> ReductionStats:
    """Verify consecutive boundaries over the integers without overflow."""
    stats = ReductionStats()
    for upper_column in upper:
        composition: dict[int, int] = {}
        for middle, coefficient in upper_column.items():
            if middle >= len(lower):
                raise ValueError("Consecutive compact boundary dimensions do not match.")
            for row, value in lower[middle].items():
                total = composition.get(row, 0) + coefficient * value
                if total:
                    composition[row] = total
                elif row in composition:
                    del composition[row]
                stats.operation(policy)
        if composition:
            raise ValueError("Compact boundaries violate boundary-of-boundary zero.")
    return stats


__all__ = [
    "ColumnReduction",
    "FieldVector",
    "ReductionStats",
    "field_columns",
    "field_rank",
    "homology_representatives",
    "integer_columns",
    "rational_rank",
    "reduce_columns",
    "transpose_columns",
    "verify_boundary_composition",
    "verify_integer_boundary_composition",
]
