#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host symbolic analysis for sparse LU with partial pivoting.

Partial pivoting chooses rows numerically, so the factor pattern is unknown
before numeric factorization. George and Ng (SIAM J. Sci. Stat. Comput. 6,
390–409, 1985) proved that for a nonsingular ``A`` with a zero-free diagonal
the patterns of ``L`` and ``U`` in ``P A Q = L U`` are contained in the
Cholesky factor ``R`` of ``(AQ)ᵀ(AQ)`` for every pivot sequence. Pivoting is
indifferent to the initial row order, so structural nonsingularity (a row
matching that makes the diagonal zero-free) suffices, and
``nnz(L) + nnz(U) ≤ 2 nnz(R)`` bounds the factor before any value is read.

The analysis owns the column ordering ``Q``: an approximate minimum-degree
ordering of ``AᵀA`` on the quotient graph whose initial elements are the rows
of ``A`` (the COLAMD formulation; Amestoy, Davis and Duff, SIAM J. Matrix
Anal. Appl. 17, 1996, for the approximate degrees, element absorption and
indistinguishable-variable detection). ``nnz(R)`` comes from the column
elimination tree (Liu, SIAM J. Matrix Anal. Appl. 11, 1990) and the
Gilbert–Ng–Peyton column counts of ``(AQ)ᵀ(AQ)`` (SIAM J. Matrix Anal. Appl.
15, 1994), both computed from the rows of ``A`` without forming ``AᵀA``.
"""

from __future__ import annotations

from hashlib import sha256

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._strict import StrictModule
from ._sparse_contract import SparseStorage
from ._sparse_ordering import _column_minimum_degree, _validated_inverse, _WorkMeter


# SuperLU stores row indices and column pointers as C ``int``.
_SUPERLU_INDEX_BYTES = np.dtype(np.int32).itemsize


class SparseLUSymbolicAnalysis(StrictModule):
    """Owned column ordering and static fill bound for pivoted sparse LU.

    ``column_permutation[k]`` is the original column eliminated ``k``-th; the
    provider factors ``A[:, column_permutation]``. ``factor_nnz_bound`` bounds
    the stored entries of ``L`` and ``U`` (diagonals included) for every
    partial-pivoting sequence when ``structurally_nonsingular``; otherwise it
    is the dense ``n²`` bound. ``ordering_work`` is the metered work of the
    column ordering.
    """

    column_permutation: Array
    factor_nnz_bound: int = eqx.field(static=True)
    structurally_nonsingular: bool = eqx.field(static=True)
    size: int = eqx.field(static=True)
    input_nnz: int = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)
    ordering_work: int = eqx.field(static=True)

    def factor_bytes(self, value_itemsize: int, batch_count: int, /) -> int:
        """Bytes of the SuperLU factors: values and row indices per entry."""
        per_factor = (
            self.factor_nnz_bound * (value_itemsize + _SUPERLU_INDEX_BYTES)
            + 2 * (self.size + 1) * _SUPERLU_INDEX_BYTES
        )
        return batch_count * per_factor

    def workspace_bytes(self, value_itemsize: int, /) -> int:
        """Column-permuted CSC copy of ``A`` plus SuperLU's ``O(n)`` work arrays."""
        permuted_copy = (
            self.input_nnz * (value_itemsize + _SUPERLU_INDEX_BYTES)
            + (self.size + 1) * _SUPERLU_INDEX_BYTES
        )
        # Row/column permutations, etree, postorder, supernode maps, markers,
        # and the dense column panel: under 16 integer and 2 value arrays of n.
        work_arrays = self.size * (16 * _SUPERLU_INDEX_BYTES + 2 * value_itemsize)
        return permuted_copy + work_arrays


def _row_lists(indices: np.ndarray, indptr: np.ndarray, rows: int, /) -> list[list[int]]:
    return [indices[indptr[row] : indptr[row + 1]].tolist() for row in range(rows)]


def _column_etree(column_rows: list[list[int]], rows: int, /) -> list[int]:
    """Elimination tree of ``BᵀB`` from the row patterns of ``B``'s columns."""
    size = len(column_rows)
    parent = [-1] * size
    ancestor = [-1] * size
    previous = [-1] * rows
    for column, pattern in enumerate(column_rows):
        for row in pattern:
            node = previous[row]
            while node != -1 and node < column:
                following = ancestor[node]
                ancestor[node] = column
                if following == -1:
                    parent[node] = column
                node = following
            previous[row] = column
    return parent


def _postorder(parent: list[int], /) -> list[int]:
    children: list[list[int]] = [[] for _ in parent]
    roots: list[int] = []
    for node, ancestor in enumerate(parent):
        (roots if ancestor == -1 else children[ancestor]).append(node)
    order: list[int] = []
    stack = [(root, False) for root in reversed(roots)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(children[node]))
    return order


def _gram_column_counts(
    row_columns: list[list[int]],
    parent: list[int],
    post: list[int],
    /,
) -> list[int]:
    """Column counts of the Cholesky factor of ``BᵀB`` (Gilbert–Ng–Peyton).

    A row of ``B`` is a clique of ``BᵀB`` whose columns lie on one root path of
    the elimination tree, so only its first column in postorder can start a
    row subtree; each row is visited there once, giving ``O(nnz(B) α)`` work.
    """
    size = len(parent)
    first = [-1] * size
    delta = [0] * size
    for position, node in enumerate(post):
        delta[node] = 1 if first[node] == -1 else 0
        while node != -1 and first[node] == -1:
            first[node] = position
            node = parent[node]
    position_of = [0] * size
    for position, node in enumerate(post):
        position_of[node] = position
    starts: list[list[int]] = [[] for _ in range(size)]
    for row, pattern in enumerate(row_columns):
        if pattern:
            starts[post[min(position_of[column] for column in pattern)]].append(row)
    max_first = [-1] * size
    previous_leaf = [-1] * size
    ancestor = list(range(size))
    for node in post:
        if parent[node] != -1:
            delta[parent[node]] -= 1
        for row in starts[node]:
            for column in row_columns[row]:
                # ``node`` is a leaf of row subtree ``column`` only when its
                # subtree starts after every leaf already seen for that row.
                if column <= node or first[node] <= max_first[column]:
                    continue
                max_first[column] = first[node]
                leaf = previous_leaf[column]
                previous_leaf[column] = node
                delta[node] += 1
                if leaf == -1:
                    continue
                root = leaf
                while root != ancestor[root]:
                    root = ancestor[root]
                while leaf != root:
                    following = ancestor[leaf]
                    ancestor[leaf] = root
                    leaf = following
                delta[root] -= 1
        if parent[node] != -1:
            ancestor[node] = parent[node]
    for node in post:
        if parent[node] != -1:
            delta[parent[node]] += delta[node]
    return delta


# Symbolic analysis reads the concrete host pattern even when a caller plans
# inside an ambient trace.
@jax.ensure_compile_time_eval()
def analyze_sparse_lu(storage: SparseStorage, /) -> SparseLUSymbolicAnalysis:
    """Own the column ordering and bound the pivoted LU fill of ``storage``."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import structural_rank

    rows, columns = storage.shape
    if rows != columns:
        raise ValueError("Sparse LU symbolic analysis requires a square pattern.")
    indices = np.asarray(storage.indices, dtype=np.int64)
    indptr = np.asarray(storage.indptr, dtype=np.int64)
    row_lists = _row_lists(indices, indptr, rows)
    # LU analysis has no symbolic work cap; its factor bound is charged by plans.
    meter = _WorkMeter(None)
    permutation = np.asarray(
        _column_minimum_degree(row_lists, columns, meter), dtype=np.int64
    )
    inverse = _validated_inverse(permutation)
    permuted_rows = [sorted(inverse[row].tolist()) for row in map(np.asarray, row_lists)]
    permuted_columns: list[list[int]] = [[] for _ in range(columns)]
    for row, pattern in enumerate(permuted_rows):
        for column in pattern:
            permuted_columns[column].append(row)
    parent = _column_etree(permuted_columns, rows)
    post = _postorder(parent)
    counts = _gram_column_counts(permuted_rows, parent, post)
    pattern = sp.csr_matrix(
        (np.ones(indices.size, dtype=np.int8), indices, indptr), shape=(rows, columns)
    )
    # Static metadata: a host bool, not a NumPy scalar.
    nonsingular = bool(structural_rank(pattern) == columns)
    bound = 2 * sum(counts) if nonsingular else rows * columns
    payload = b"|".join(
        (
            np.asarray(storage.shape, dtype=np.int64).tobytes(),
            indices.tobytes(),
            indptr.tobytes(),
            permutation.tobytes(),
        )
    )
    return SparseLUSymbolicAnalysis(
        column_permutation=jnp.asarray(permutation, dtype=storage.indices.dtype),
        factor_nnz_bound=bound,
        structurally_nonsingular=nonsingular,
        size=columns,
        input_nnz=storage.nnz,
        analysis_id=sha256(payload).hexdigest(),
        ordering_work=meter.total,
    )


__all__ = ["SparseLUSymbolicAnalysis", "analyze_sparse_lu"]
