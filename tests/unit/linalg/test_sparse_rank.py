# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.linalg import ArraySpace, RankPolicy
from phydrax.linalg._sparse_rank import (
    prepare_sparse_row_rank,
    SparseRowRankPolicy,
    SparseRowRankRefusal,
)
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def _operator(
    rows: list[int],
    columns: list[int],
    values: list[float],
    shape: tuple[int, int],
) -> SparseCoordinateOperator:
    relation = EdgeRelation(
        np.asarray(columns, dtype=np.int32),
        np.asarray(rows, dtype=np.int32),
        source_size=shape[1],
        target_size=shape[0],
    )
    return SparseCoordinateOperator(
        relation,
        jnp.asarray(values, dtype=jnp.float64),
        source=ArraySpace(
            (shape[1],), dtype=jnp.float64, space_id="sparse-rank-regression:source"
        ),
        target=ArraySpace(
            (shape[0],), dtype=jnp.float64, space_id="sparse-rank-regression:target"
        ),
    )


def test_dependent_rows_are_removed_after_native_duplicate_coalescing() -> None:
    operator = _operator([0, 0, 1, 2], [0, 0, 0, 1], [0.5, 0.5, 2.0, 1.0], (3, 2))
    evidence = prepare_sparse_row_rank(
        operator, SparseRowRankPolicy(RankPolicy(relative_cutoff=1e-12))
    )
    np.testing.assert_array_equal(evidence.selected_rows, [0, 2])
    np.testing.assert_array_equal(evidence.unselected_rows, [1])
    assert evidence.rank == 2
    assert evidence.elimination_work == 1
    assert evidence.factor_nonzeros == 2
    np.testing.assert_allclose(evidence.pivot_magnitudes, [1.0, 1.0])


def test_near_dependent_pivots_obey_declared_cutoff_not_algebraic_rank() -> None:
    operator = _operator([0, 0, 1, 1], [0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0 + 1e-10], (2, 2))
    coarse = prepare_sparse_row_rank(
        operator, SparseRowRankPolicy(RankPolicy(relative_cutoff=1e-8))
    )
    fine = prepare_sparse_row_rank(
        operator, SparseRowRankPolicy(RankPolicy(relative_cutoff=1e-12))
    )
    assert coarse.rank == 1
    assert fine.rank == 2
    assert coarse.tolerance_defined
    np.testing.assert_allclose(coarse.pivot_threshold, (1.0 + 1e-10) * 1e-8)
    with pytest.raises(SparseRowRankRefusal, match="rank deficiency"):
        prepare_sparse_row_rank(
            operator,
            SparseRowRankPolicy(RankPolicy(relative_cutoff=1e-8, require_full_rank=True)),
        )


def test_sparse_elimination_refuses_actual_symbolic_fill_and_work_limits() -> None:
    # Row 1 starts with only column 0; eliminating row 0 creates columns 1,2.
    operator = _operator([0, 0, 0, 1], [0, 1, 2, 0], [1.0, 1.0, 1.0, 1.0], (2, 3))
    with pytest.raises(SparseRowRankRefusal, match="factor fill") as refusal:
        prepare_sparse_row_rank(operator, SparseRowRankPolicy(maximum_factor_nonzeros=4))
    assert refusal.value.observed == 5
    assert refusal.value.limit == 4
    with pytest.raises(SparseRowRankRefusal, match="elimination work") as work_refusal:
        prepare_sparse_row_rank(operator, SparseRowRankPolicy(maximum_elimination_work=2))
    assert work_refusal.value.observed == 3
    admitted = prepare_sparse_row_rank(
        operator,
        SparseRowRankPolicy(maximum_factor_nonzeros=5, maximum_elimination_work=3),
    )
    assert admitted.rank == 2
    assert admitted.factor_nonzeros == 5
    assert admitted.elimination_work == 3


def test_nonfinite_values_and_input_capacity_are_not_rank_zero_successes() -> None:
    operator = _operator([0], [0], [np.nan], (1, 1))
    with pytest.raises(SparseRowRankRefusal, match="nonfinite input"):
        prepare_sparse_row_rank(operator)
    finite = _operator([0, 0], [0, 1], [1.0, 1.0], (1, 2))
    with pytest.raises(SparseRowRankRefusal, match="input nonzeros"):
        prepare_sparse_row_rank(finite, SparseRowRankPolicy(maximum_input_nonzeros=1))
