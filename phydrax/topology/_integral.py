#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..algebraic._exact_integer import (
    invert_unimodular,
    smith_normal_decomposition,
    smith_rank,
)
from ._complex import CellSubcomplex, compact_boundary
from ._resources import TopologyResourceError, TopologyResourcePolicy


class IntegralHomologyDegree(StrictModule, NonTrainableState):
    """Free rank and positive torsion invariant factors in one degree."""

    degree: int = eqx.field(static=True)
    free_rank: int = eqx.field(static=True)
    torsion_invariants: tuple[int, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(self, degree: int, free_rank: int, torsion: Sequence[int], /):
        values = tuple(abs(int(value)) for value in torsion if abs(int(value)) > 1)
        if int(free_rank) < 0:
            raise ValueError("Integral homology free rank must be non-negative.")
        if any(right % left for left, right in zip(values[:-1], values[1:], strict=True)):
            raise ValueError("Integral torsion invariant factors must divide successors.")
        self.degree = int(degree)
        self.free_rank = int(free_rank)
        self.torsion_invariants = values
        self.result_id = canonical_fingerprint(
            {
                "kind": "integral-homology-degree",
                "degree": int(degree),
                "free_rank": int(free_rank),
                "torsion": list(values),
            }
        )


class IntegralHomologyResult(StrictModule, NonTrainableState):
    """Exact integral homology invariants computed with unimodular transforms."""

    degrees: tuple[IntegralHomologyDegree, ...]
    source_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(self, degrees, /, *, source_id: str, backend: str):
        values = tuple(degrees)
        self.degrees = values
        self.source_id = str(source_id)
        self.backend = str(backend)
        self.result_id = canonical_fingerprint(
            {
                "kind": "integral-homology-result",
                "source": self.source_id,
                "backend": self.backend,
                "degrees": [value.result_id for value in values],
            }
        )

    def degree(self, degree: int, /) -> IntegralHomologyDegree:
        for value in self.degrees:
            if value.degree == int(degree):
                return value
        raise KeyError(f"No integral homology result exists in degree {degree}.")


def _matrix(complex: CellSubcomplex, degree: int, /):
    boundary = compact_boundary(complex, degree)
    values = np.zeros((boundary.row_count, boundary.column_count), dtype=object)
    for row, column, coefficient in zip(
        np.asarray(boundary.row_indices),
        np.asarray(boundary.column_indices),
        np.asarray(boundary.coefficients),
        strict=True,
    ):
        values[int(row), int(column)] = int(coefficient)
    return values


def _rank_from_smith(smith: np.ndarray, /) -> int:
    return smith_rank(smith)


def compute_integral_homology(
    complex: CellSubcomplex,
    /,
    *,
    resources: TopologyResourcePolicy | None = None,
) -> IntegralHomologyResult:
    """Compute exact free ranks and torsion via chain-compatible Smith transforms."""
    policy = TopologyResourcePolicy() if resources is None else resources
    total_dense = sum(
        complex.layout.counts[degree]
        * (complex.layout.counts[degree - 1] if degree else 0)
        for degree in range(complex.max_degree + 1)
    )
    if total_dense > policy.max_reduction_entries:
        raise TopologyResourceError("Integral Smith workspace exceeds resource policy.")
    boundaries = tuple(
        _matrix(complex, degree) for degree in range(complex.max_degree + 1)
    )
    results = []
    for degree, boundary in enumerate(boundaries):
        if boundary.shape[0] == 0:
            smith = boundary
            right = np.eye(boundary.shape[1], dtype=object)
            rank_boundary = 0
        else:
            smith, _, right = smith_normal_decomposition(boundary)
            rank_boundary = _rank_from_smith(smith)
        kernel_dimension = boundary.shape[1] - rank_boundary
        incoming = (
            boundaries[degree + 1]
            if degree < complex.max_degree
            else np.zeros((boundary.shape[1], 0), dtype=object)
        )
        transformed = invert_unimodular(right) @ incoming
        if np.any(transformed[:rank_boundary, :] != 0):
            raise RuntimeError("Incoming boundary does not lie in the exact kernel.")
        quotient = transformed[rank_boundary:, :]
        if quotient.shape[0] == 0 or quotient.shape[1] == 0:
            quotient_smith = quotient
            rank_incoming = 0
        else:
            quotient_smith, _, _ = smith_normal_decomposition(quotient)
            rank_incoming = _rank_from_smith(quotient_smith)
        torsion = tuple(
            int(quotient_smith[index, index])
            for index in range(rank_incoming)
            if abs(int(quotient_smith[index, index])) > 1
        )
        results.append(
            IntegralHomologyDegree(
                degree,
                kernel_dimension - rank_incoming,
                torsion,
            )
        )
    return IntegralHomologyResult(
        results,
        source_id=complex.subcomplex_id,
        backend="phydrax-exact-smith-normal-decomposition",
    )


__all__ = [
    "IntegralHomologyDegree",
    "IntegralHomologyResult",
    "compute_integral_homology",
]
