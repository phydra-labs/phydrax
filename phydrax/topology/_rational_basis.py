#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._complex import CellSubcomplex, compact_boundary
from ._resources import TopologyResourceError, TopologyResourcePolicy


class RationalClassBasis(StrictModule, NonTrainableState):
    """Sparse-storage exact rational homology class representatives."""

    cell_indices: Array
    generator_indices: Array
    numerators: tuple[int, ...] = eqx.field(static=True)
    denominators: tuple[int, ...] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    generator_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        vectors: tuple[tuple[Fraction, ...], ...],
        complex: CellSubcomplex,
        /,
    ):
        cells = []
        generators = []
        numerators = []
        denominators = []
        ambient = np.asarray(complex.layout.compact_to_ambient[int(degree)])
        for generator, vector in enumerate(vectors):
            for compact, value in enumerate(vector):
                if value:
                    cells.append(int(ambient[compact]))
                    generators.append(generator)
                    numerators.append(value.numerator)
                    denominators.append(value.denominator)
        self.cell_indices = jnp.asarray(np.asarray(cells, dtype=np.int32))
        self.generator_indices = jnp.asarray(np.asarray(generators, dtype=np.int32))
        self.numerators = tuple(numerators)
        self.denominators = tuple(denominators)
        self.degree = int(degree)
        self.generator_count = len(vectors)
        self.source_id = complex.subcomplex_id
        self.basis_id = canonical_fingerprint(
            {
                "kind": "rational-class-basis",
                "degree": int(degree),
                "source": complex.subcomplex_id,
                "cells": cells,
                "generators": generators,
                "numerators": numerators,
                "denominators": denominators,
            }
        )

    def dense(self, cell_count: int, /) -> np.ndarray:
        values = np.zeros((int(cell_count), self.generator_count), dtype=object)
        for cell, generator, numerator, denominator in zip(
            np.asarray(self.cell_indices),
            np.asarray(self.generator_indices),
            self.numerators,
            self.denominators,
            strict=True,
        ):
            values[int(cell), int(generator)] = Fraction(numerator, denominator)
        return values


class RationalHomologyBasisResult(StrictModule, NonTrainableState):
    """Exact rational free-class bases by degree."""

    bases: tuple[RationalClassBasis, ...]
    source_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(self, bases: tuple[RationalClassBasis, ...], /, *, source_id: str):
        self.bases = bases
        self.source_id = str(source_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "rational-homology-basis-result",
                "source": self.source_id,
                "bases": [value.basis_id for value in bases],
            }
        )

    def degree(self, degree: int, /) -> RationalClassBasis:
        for basis in self.bases:
            if basis.degree == int(degree):
                return basis
        raise KeyError(f"No rational homology basis exists in degree {degree}.")


def _dense_boundary(complex: CellSubcomplex, degree: int, /):
    boundary = compact_boundary(complex, degree)
    matrix = np.zeros((boundary.row_count, boundary.column_count), dtype=object)
    for row, column, coefficient in zip(
        np.asarray(boundary.row_indices),
        np.asarray(boundary.column_indices),
        np.asarray(boundary.coefficients),
        strict=True,
    ):
        matrix[int(row), int(column)] = Fraction(int(coefficient))
    return matrix


def _rref(matrix):
    values = np.asarray(matrix, dtype=object).copy()
    row = 0
    pivots = []
    for column in range(values.shape[1]):
        pivot = next(
            (index for index in range(row, values.shape[0]) if values[index, column]),
            None,
        )
        if pivot is None:
            continue
        values[[row, pivot]] = values[[pivot, row]]
        values[row] = [value / values[row, column] for value in values[row]]
        for other in range(values.shape[0]):
            if other != row and values[other, column]:
                factor = values[other, column]
                values[other] = [
                    left - factor * right
                    for left, right in zip(values[other], values[row], strict=True)
                ]
        pivots.append(column)
        row += 1
        if row == values.shape[0]:
            break
    return values, tuple(pivots)


def _nullspace(matrix):
    rref, pivots = _rref(matrix)
    free = [column for column in range(matrix.shape[1]) if column not in pivots]
    vectors = []
    for column in free:
        vector = [Fraction(0)] * matrix.shape[1]
        vector[column] = Fraction(1)
        for row, pivot in enumerate(pivots):
            vector[pivot] = -rref[row, column]
        vectors.append(tuple(vector))
    return tuple(vectors)


def _rank(columns):
    if not columns:
        return 0
    matrix = np.asarray(columns, dtype=object).T
    return len(_rref(matrix)[1])


def compute_rational_homology_basis(
    complex: CellSubcomplex,
    /,
    *,
    resources: TopologyResourcePolicy | None = None,
) -> RationalHomologyBasisResult:
    """Compute exact rational representatives of the free homology classes."""
    policy = TopologyResourcePolicy() if resources is None else resources
    boundaries = tuple(
        _dense_boundary(complex, degree) for degree in range(complex.max_degree + 1)
    )
    if sum(value.size for value in boundaries) > policy.max_reduction_entries:
        raise TopologyResourceError("Rational basis dense workspace exceeds policy.")
    bases = []
    for degree in range(complex.max_degree + 1):
        cycles = _nullspace(boundaries[degree])
        incoming = (
            [
                tuple(boundaries[degree + 1][:, column])
                for column in range(boundaries[degree + 1].shape[1])
            ]
            if degree < complex.max_degree
            else []
        )
        span = list(incoming)
        rank = _rank(span)
        representatives = []
        for cycle in cycles:
            candidate_rank = _rank(span + [cycle])
            if candidate_rank > rank:
                representatives.append(cycle)
                span.append(cycle)
                rank = candidate_rank
        for vector in representatives:
            for value in vector:
                bits = max(value.numerator.bit_length(), value.denominator.bit_length())
                if bits > policy.max_rational_bit_length:
                    raise TopologyResourceError(
                        "Rational class basis exceeds coefficient bit-length policy."
                    )
        bases.append(RationalClassBasis(degree, tuple(representatives), complex))
    return RationalHomologyBasisResult(tuple(bases), source_id=complex.subcomplex_id)


__all__ = [
    "RationalClassBasis",
    "RationalHomologyBasisResult",
    "compute_rational_homology_basis",
]
