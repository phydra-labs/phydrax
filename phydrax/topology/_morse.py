#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._integer import ExactChainComplex, ExactIntegerCOO
from ._resources import TopologyResourcePolicy


class MorseReductionResult(StrictModule, NonTrainableState):
    """Exact elementary unit cancellation and its reduced chain complex."""

    source: ExactChainComplex
    reduced: ExactChainComplex
    degree: int = eqx.field(static=True)
    lower_cell: int = eqx.field(static=True)
    upper_cell: int = eqx.field(static=True)
    pivot: int = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: ExactChainComplex,
        reduced: ExactChainComplex,
        degree: int,
        lower_cell: int,
        upper_cell: int,
        pivot: int,
        /,
    ):
        self.source = source
        self.reduced = reduced
        self.degree = int(degree)
        self.lower_cell = int(lower_cell)
        self.upper_cell = int(upper_cell)
        self.pivot = int(pivot)
        self.result_id = canonical_fingerprint(
            {
                "kind": "morse-reduction-result",
                "source": source.complex_id,
                "reduced": reduced.complex_id,
                "degree": int(degree),
                "lower": int(lower_cell),
                "upper": int(upper_cell),
                "pivot": int(pivot),
            }
        )


def _from_dense(matrix, /, *, source_id: str, target_id: str):
    rows, columns = np.nonzero(np.asarray(matrix, dtype=object))
    return ExactIntegerCOO(
        matrix.shape[0],
        matrix.shape[1],
        rows.astype(np.int32),
        columns.astype(np.int32),
        tuple(
            int(matrix[row, column]) for row, column in zip(rows, columns, strict=True)
        ),
        source_id=source_id,
        target_id=target_id,
    )


def cancel_unit_pair(
    complex: ExactChainComplex,
    degree: int,
    lower_cell: int,
    upper_cell: int,
    /,
    *,
    resources: TopologyResourcePolicy | None = None,
) -> MorseReductionResult:
    """Cancel one unit boundary pair by exact Schur-complement elimination."""
    policy = TopologyResourcePolicy() if resources is None else resources
    degree_ = int(degree)
    if degree_ <= 0 or degree_ >= len(complex.boundaries):
        raise ValueError("Morse cancellation degree must be positive and represented.")
    lower = int(lower_cell)
    upper = int(upper_cell)
    matrices = [boundary.dense(resources=policy) for boundary in complex.boundaries]
    boundary = matrices[degree_]
    if lower < 0 or lower >= boundary.shape[0] or upper < 0 or upper >= boundary.shape[1]:
        raise ValueError("Morse cancellation cell index is out of range.")
    pivot = int(boundary[lower, upper])
    if abs(pivot) != 1:
        raise ValueError("Integral Morse cancellation requires a unit pivot ±1.")
    inverse = pivot
    remaining_rows = [index for index in range(boundary.shape[0]) if index != lower]
    remaining_columns = [index for index in range(boundary.shape[1]) if index != upper]
    updated = np.empty((len(remaining_rows), len(remaining_columns)), dtype=object)
    for row_position, row in enumerate(remaining_rows):
        for column_position, column in enumerate(remaining_columns):
            updated[row_position, column_position] = int(boundary[row, column]) - int(
                boundary[row, upper]
            ) * inverse * int(boundary[lower, column])
    matrices[degree_] = updated
    matrices[degree_ - 1] = np.delete(matrices[degree_ - 1], lower, axis=1)
    if degree_ + 1 < len(matrices):
        matrices[degree_ + 1] = np.delete(matrices[degree_ + 1], upper, axis=0)
    boundaries = []
    identifier = f"morse:{complex.complex_id}:{degree_}:{lower}:{upper}"
    for current_degree, matrix in enumerate(matrices):
        boundaries.append(
            _from_dense(
                matrix,
                source_id=f"{identifier}:degree:{current_degree}",
                target_id=f"{identifier}:degree:{current_degree - 1}",
            )
        )
    reduced = ExactChainComplex(boundaries, complex_id=identifier)
    return MorseReductionResult(
        complex,
        reduced,
        degree_,
        lower,
        upper,
        pivot,
    )


__all__ = ["MorseReductionResult", "cancel_unit_pair"]
