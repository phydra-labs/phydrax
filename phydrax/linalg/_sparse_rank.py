#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded host sparse row-rank profiles for fixed-topology constraints.

This owner supplies a tolerance-defined deterministic pivot profile, not an
algebraic-rank certificate or a singular-value estimate. Consumers validate
selected equations with native numerical factors and audit every original
constraint. No global dense design or Gram matrix is materialized here.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..typing import Dim, Float64, Int32, Scalar, Size
from ._policies import RankPolicy
from ._sparse_contract import AbstractSparseLinearOperator


class _SparseIndependentRowDim(Dim):
    """Selected rows in one native sparse rank profile."""


class _SparseDependentRowDim(Dim):
    """Rows dropped by a native tolerance-defined pivot profile."""


@final
class SparseRowRankPolicy(StrictModule):
    """Entry-scaled pivot cutoff and separate input, fill and work capacities.

    RankPolicy supplies absolute/relative cutoffs. Here the scale is the
    largest stored entry, explicitly not the largest singular value. A missing
    relative cutoff resolves to max(shape)*machine epsilon of the input dtype.
    """

    rank: RankPolicy
    maximum_rows: int = eqx.field(static=True)
    maximum_input_nonzeros: int = eqx.field(static=True)
    maximum_factor_nonzeros: int = eqx.field(static=True)
    maximum_elimination_work: int = eqx.field(static=True)

    def __init__(
        self,
        rank: RankPolicy | None = None,
        *,
        maximum_rows: int = 2_000_000,
        maximum_input_nonzeros: int = 2_000_000,
        maximum_factor_nonzeros: int = 2_000_000,
        maximum_elimination_work: int = 2_000_000,
    ) -> None:
        rank_ = RankPolicy() if rank is None else rank
        if not isinstance(rank_, RankPolicy):
            raise TypeError("rank must be the native RankPolicy.")
        capacities = (
            maximum_rows,
            maximum_input_nonzeros,
            maximum_factor_nonzeros,
            maximum_elimination_work,
        )
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in capacities):
            raise TypeError("Sparse row-rank capacities must be host integers.")
        if any(v < 1 for v in capacities):
            raise ValueError("Sparse row-rank capacities must be positive.")
        self.rank = rank_
        self.maximum_rows = int(maximum_rows)
        self.maximum_input_nonzeros = int(maximum_input_nonzeros)
        self.maximum_factor_nonzeros = int(maximum_factor_nonzeros)
        self.maximum_elimination_work = int(maximum_elimination_work)


@final
class SparseRowRankRefusal(ValueError):
    """An explicit native preparation refusal, retaining limit/work evidence."""

    reason: str
    observed: int
    limit: int

    def __init__(self, reason: str, observed: int, limit: int, /) -> None:
        super().__init__(
            f"Sparse row-rank preparation refuses {reason}: observed {observed}, limit {limit}."
        )
        self.reason = reason
        self.observed = int(observed)
        self.limit = int(limit)


@final
class SparseRowRankEvidence(StrictModule):
    __strict_contract__ = True
    selected_rows: Int32[_SparseIndependentRowDim]
    unselected_rows: Int32[_SparseDependentRowDim]
    pivot_magnitudes: Float64[_SparseIndependentRowDim]
    rank: Size[_SparseIndependentRowDim] = eqx.field(static=True)
    pivot_threshold: Float64[Scalar]
    entry_scale: Float64[Scalar]
    row_count: int = eqx.field(static=True)
    column_count: int = eqx.field(static=True)
    input_nonzeros: int = eqx.field(static=True)
    factor_nonzeros: int = eqx.field(static=True)
    elimination_work: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    policy: SparseRowRankPolicy
    tolerance_defined: bool = eqx.field(static=True, default=True)
    non_claim: str = eqx.field(
        static=True,
        default="pivot profile is neither exact algebraic rank nor a singular-value bound",
    )


def prepare_sparse_row_rank(
    operator: AbstractSparseLinearOperator,
    policy: SparseRowRankPolicy | None = None,
    /,
) -> SparseRowRankEvidence:
    """Select independent rows by bounded sparse host elimination.

    Rows and pivot columns are visited in canonical ascending order. Coalesced
    native sparse storage owns duplicate-route accumulation. This is an eager
    preparation boundary: numerical tracers, batches and global dense storage
    are refused. Dynamic consumers retain this pattern only while their native
    factor/constraint audits continue to satisfy the declared acceptance.
    """
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("Sparse row-rank preparation requires a native sparse operator.")
    selected_policy = SparseRowRankPolicy() if policy is None else policy
    if not isinstance(selected_policy, SparseRowRankPolicy):
        raise TypeError("policy must be a SparseRowRankPolicy.")
    storage = operator.sparse_storage()
    rows, columns = storage.shape
    if operator.batch_shape or storage.values.ndim != 1:
        raise ValueError("Sparse row-rank preparation requires one unbatched matrix.")
    if not storage.canonical or not storage.sorted_indices:
        raise ValueError(
            "Sparse row-rank preparation requires canonical sorted native CSR storage."
        )
    if isinstance(storage.values, jax_core.Tracer):
        raise TypeError(
            "Sparse row rank is a host preparation boundary, not a traced runtime operation."
        )
    if rows > selected_policy.maximum_rows:
        raise SparseRowRankRefusal("row capacity", rows, selected_policy.maximum_rows)
    nnz = storage.values.size
    if nnz > selected_policy.maximum_input_nonzeros:
        raise SparseRowRankRefusal(
            "input nonzeros", nnz, selected_policy.maximum_input_nonzeros
        )
    values = np.asarray(storage.values)
    indices = np.asarray(storage.indices)
    pointers = np.asarray(storage.indptr)
    if not np.all(np.isfinite(values)):
        raise SparseRowRankRefusal(
            "nonfinite input", int(np.count_nonzero(~np.isfinite(values))), 0
        )
    scale = float(np.max(np.abs(values), initial=0.0))
    relative = selected_policy.rank.relative_cutoff
    if relative is None:
        relative = max(rows, columns, 1) * float(np.finfo(values.real.dtype).eps)
    absolute = (
        0.0
        if selected_policy.rank.absolute_cutoff is None
        else selected_policy.rank.absolute_cutoff
    )
    threshold = float(absolute + relative * scale)
    if not isfinite(threshold):
        raise ValueError("Sparse row-rank cutoff is not finite.")
    pivots: dict[int, dict[int, complex | float]] = {}
    selected: list[int] = []
    magnitudes: list[float] = []
    retained_entries = 0
    work = 0
    for row in range(rows):
        candidate = {
            int(indices[position]): values[position].item()
            for position in range(int(pointers[row]), int(pointers[row + 1]))
            if abs(values[position]) > threshold
        }
        while candidate:
            column = min(candidate)
            value = candidate[column]
            if abs(value) <= threshold:
                del candidate[column]
                continue
            if column not in pivots:
                pivot = {key: entry / value for key, entry in candidate.items()}
                retained_entries += len(pivot)
                if retained_entries > selected_policy.maximum_factor_nonzeros:
                    raise SparseRowRankRefusal(
                        "factor fill",
                        retained_entries,
                        selected_policy.maximum_factor_nonzeros,
                    )
                pivots[column] = pivot
                selected.append(row)
                magnitudes.append(float(abs(value)))
                break
            pivot = pivots[column]
            work += len(pivot)
            if work > selected_policy.maximum_elimination_work:
                raise SparseRowRankRefusal(
                    "elimination work", work, selected_policy.maximum_elimination_work
                )
            for key, entry in pivot.items():
                updated = candidate.get(key, 0.0) - value * entry
                if not isfinite(abs(updated)):
                    raise ValueError(
                        "Sparse row-rank elimination produced nonfinite arithmetic."
                    )
                if abs(updated) <= threshold:
                    candidate.pop(key, None)
                else:
                    candidate[key] = updated
    rank = len(selected)
    if selected_policy.rank.require_full_rank and rank != min(rows, columns):
        raise SparseRowRankRefusal(
            "tolerance-defined rank deficiency", rank, min(rows, columns)
        )
    selected_rows = np.asarray(selected, dtype=np.int32)
    keep = np.zeros(rows, dtype=np.bool_)
    keep[selected_rows] = True
    unselected = np.flatnonzero(~keep).astype(np.int32)
    evidence_id = canonical_fingerprint(
        {
            "kind": "native-sparse-row-rank",
            "operator": operator.operator_id,
            "numeric-storage": array_tree_fingerprint((values, indices, pointers)),
            "selected-rows": selected,
            "threshold": threshold,
            "work": work,
            "fill": retained_entries,
        }
    )
    return SparseRowRankEvidence(
        selected_rows=jnp.asarray(selected_rows),
        unselected_rows=jnp.asarray(unselected),
        pivot_magnitudes=jnp.asarray(magnitudes, dtype=jnp.float64),
        rank=rank,
        pivot_threshold=jnp.asarray(threshold, dtype=jnp.float64),
        entry_scale=jnp.asarray(scale, dtype=jnp.float64),
        row_count=rows,
        column_count=columns,
        input_nonzeros=nnz,
        factor_nonzeros=retained_entries,
        elimination_work=work,
        operator_id=operator.operator_id,
        evidence_id=evidence_id,
        policy=selected_policy,
    )
