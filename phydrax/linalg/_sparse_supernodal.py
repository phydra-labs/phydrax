#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Supernodal multifrontal sparse Cholesky: host symbolic analysis and kernels.

The symbolic analysis follows the classical pipeline on the symmetrically
ordered pattern ``P A Pᵀ``: elimination tree (Liu, ACM TOMS 12, 1986),
postorder, column counts by row-subtree skeletons (Gilbert, Ng and Peyton,
SIAM J. Matrix Anal. Appl. 15, 1994), fundamental supernodes, and relaxed
amalgamation with the CHOLMOD default thresholds (Chen, Davis, Hager and
Rajamanickam, ACM TOMS 35, 2008). Each supernode stores one dense
column-major ``rows x columns`` panel of ``L``; the numeric factorization is
multifrontal (Duff and Reid, ACM TOMS 9, 1983): every front is assembled from
the input entries and the children's update matrices, then factored with dense
``potrf``/``trsm``/``syrk`` (``herk``) blocks. Triangular solves sweep the
same panels with dense ``trsm``/``gemm`` blocks for all right-hand sides at
once.

Everything here is concrete host work on NumPy arrays. Callers reach the
kernels through `jax.pure_callback` with explicit result shapes, so one
symbolic analysis serves every refreshed value set of its pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
from scipy.linalg import blas, lapack


# CHOLMOD relaxed-amalgamation defaults: always merge up to ``_RELAX[0]``
# columns; otherwise merge while the zero fraction of the merged panel stays
# below ``_ZERO[i]`` for at most ``_RELAX[i + 1]`` columns, or below ``_ZERO[2]``.
_RELAX = (4, 16, 48)
_ZERO = (0.8, 0.1, 0.05)

STATUS_SUCCESS = 0
STATUS_NONPOSITIVE_PIVOT = 2
STATUS_NONFINITE = 3
TRIANGULAR_SUCCESS = 0
TRIANGULAR_NONFINITE = 2
# Pivot kinds of one refresh, per elimination column: the front's blocked
# ``potrf`` accepted it, the pivot-by-pivot route accepted it, or it is a
# replaced or continued constant.
PIVOT_BLOCKED = 0
PIVOT_ACCEPTED = 1
PIVOT_CONSTANT = 2


def retained_index_bytes(size: int, supernodes: int, rows: int, entries: int, /) -> int:
    """Host index bytes of a `SupernodalAnalysis` from its counted sizes.

    ``rows`` is the summed row-structure length and ``entries`` the number of
    scattered input pairs; all indices are int64 and flags one byte.
    """
    updates = rows - size
    return (
        8 * (2 * size + 4 * (supernodes + 1) + rows + updates + supernodes)
        + supernodes
        + 17 * entries
    )


@dataclass(frozen=True, eq=False)
class SupernodalAnalysis:
    """Immutable host symbolic factorization of one ordered symmetric pattern.

    ``elimination`` maps elimination position to original index (the ordering
    composed with an elimination-tree postorder, which leaves fill unchanged);
    ``supernode_columns[s] : supernode_columns[s + 1]`` are the columns of
    supernode ``s`` in elimination positions. ``rows[row_offsets[s] :
    row_offsets[s + 1]]`` is its sorted row structure, its own columns first.
    Its panel ``L[rows, columns]`` occupies ``value_offsets[s] :
    value_offsets[s + 1]`` of the factor values in column-major order.
    ``relative[update_offsets[s] : update_offsets[s + 1]]`` locates the update
    rows of ``s`` (its rows below its columns) inside its parent's rows, and
    ``child_counts[s]`` children of ``s`` precede it in postorder.
    ``input_sources`` and ``input_targets`` scatter one stored entry of every
    symmetric input pair into the panels; ``input_conjugate`` marks entries
    read from the upper triangle.
    """

    size: int
    elimination: np.ndarray
    supernode_columns: np.ndarray
    row_offsets: np.ndarray
    rows: np.ndarray
    value_offsets: np.ndarray
    update_offsets: np.ndarray
    relative: np.ndarray
    contiguous_update: np.ndarray
    child_counts: np.ndarray
    input_sources: np.ndarray
    input_targets: np.ndarray
    input_conjugate: np.ndarray
    diagonal_targets: np.ndarray
    structural_nnz: int
    factor_size: int
    factor_flops: int
    solve_flops: int
    front_entries: int
    stack_entries: int
    analysis_id: str

    @property
    def supernode_count(self) -> int:
        return int(self.supernode_columns.size - 1)

    @property
    def index_bytes(self) -> int:
        return retained_index_bytes(
            self.size,
            self.supernode_count,
            int(self.rows.size),
            int(self.input_sources.size),
        )

    def __hash__(self) -> int:
        return hash(self.analysis_id)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SupernodalAnalysis)
            and other.analysis_id == self.analysis_id
        )


@dataclass(frozen=True)
class SymbolicCounts:
    """Elimination tree, postorder and column counts of an ordered pattern."""

    parent: np.ndarray
    postorder: np.ndarray
    column_counts: np.ndarray
    lower_indptr: np.ndarray
    lower_indices: np.ndarray
    source_rows: np.ndarray
    source_columns: np.ndarray
    source_positions: np.ndarray
    source_conjugate: np.ndarray


def symmetric_entries(
    size: int, indices: np.ndarray, indptr: np.ndarray, inverse: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One stored entry per symmetric pair of ``P A Pᵀ``, lower entries preferred.

    Returns ordered ``(row >= column)`` coordinates, the CSR position of the
    chosen stored entry, and whether it was read from the upper triangle.
    """
    stored_rows = np.repeat(np.arange(size, dtype=np.int64), np.diff(indptr))
    ordered_rows = inverse[stored_rows]
    ordered_columns = inverse[indices]
    conjugate = ordered_rows < ordered_columns
    high = np.maximum(ordered_rows, ordered_columns)
    low = np.minimum(ordered_rows, ordered_columns)
    keys = high * size + low
    order = np.lexsort((conjugate, keys))
    sorted_keys = keys[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    chosen = order[first]
    return high[chosen], low[chosen], chosen, conjugate[chosen]


def _elimination_tree(
    size: int, row_indptr: list[int], row_columns: list[int], /
) -> list[int]:
    """Liu's elimination tree from the strictly lower rows of ``P A Pᵀ``."""
    parent = [-1] * size
    ancestor = [-1] * size
    for row in range(size):
        for position in range(row_indptr[row], row_indptr[row + 1]):
            node = row_columns[position]
            while node != -1 and node < row:
                following = ancestor[node]
                ancestor[node] = row
                if following == -1:
                    parent[node] = row
                node = following
    return parent


def _postorder(parent: np.ndarray, /) -> np.ndarray:
    """Depth-first postorder visiting children in ascending order."""
    size = parent.size
    head = [-1] * size
    following = [-1] * size
    # Prepend children in descending order so traversal visits them ascending.
    parent_list = parent.tolist()
    for node in range(size - 1, -1, -1):
        owner = parent_list[node]
        if owner != -1:
            following[node] = head[owner]
            head[owner] = node
    order: list[int] = []
    stack: list[int] = []
    for root in range(size):
        if parent_list[root] != -1:
            continue
        stack.append(root)
        while stack:
            node = stack[-1]
            child = head[node]
            if child == -1:
                order.append(node)
                stack.pop()
            else:
                head[node] = following[child]
                stack.append(child)
    return np.asarray(order, dtype=np.int64)


def _column_counts(
    parent: list[int],
    column_indptr: list[int],
    column_rows: list[int],
    /,
) -> np.ndarray:
    """Column counts of ``L`` (diagonal included) on a postordered tree.

    Gilbert–Ng–Peyton skeleton counting, CSparse ``cs_counts`` specialised to
    a symmetric pattern whose labels are already a postorder.
    """
    size = len(parent)
    first = list(range(size))
    for node in range(size):
        owner = parent[node]
        if owner != -1 and first[node] < first[owner]:
            first[owner] = first[node]
    delta = [1 if first[node] == node else 0 for node in range(size)]
    ancestor = list(range(size))
    maximum_first = [-1] * size
    previous_leaf = [-1] * size
    for column in range(size):
        owner = parent[column]
        if owner != -1:
            delta[owner] -= 1
        first_column = first[column]
        for position in range(column_indptr[column], column_indptr[column + 1]):
            row = column_rows[position]
            if first_column <= maximum_first[row]:
                continue
            maximum_first[row] = first_column
            previous = previous_leaf[row]
            previous_leaf[row] = column
            delta[column] += 1
            if previous == -1:
                continue
            root = previous
            while root != ancestor[root]:
                root = ancestor[root]
            node = previous
            while node != root:
                upper = ancestor[node]
                ancestor[node] = root
                node = upper
            delta[root] -= 1
        if owner != -1:
            ancestor[column] = owner
    counts = np.asarray(delta, dtype=np.int64)
    for node in range(size):
        owner = parent[node]
        if owner != -1:
            counts[owner] += counts[node]
    return counts


def symbolic_counts(
    size: int, indices: np.ndarray, indptr: np.ndarray, inverse: np.ndarray, /
) -> SymbolicCounts:
    """Elimination tree, postorder and column counts of ``P A Pᵀ`` in O(nnz α)."""
    high, low, positions, conjugate = symmetric_entries(size, indices, indptr, inverse)
    off = high != low
    # Strictly lower rows (for the tree), then relabel by the tree postorder.
    row_order = np.lexsort((low[off], high[off]))
    row_indptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(np.bincount(high[off], minlength=size), out=row_indptr[1:])
    parent = np.asarray(
        _elimination_tree(size, row_indptr.tolist(), low[off][row_order].tolist()),
        dtype=np.int64,
    )
    postorder = _postorder(parent)
    relabel = np.empty(size, dtype=np.int64)
    relabel[postorder] = np.arange(size, dtype=np.int64)
    post_parent = np.full(size, -1, dtype=np.int64)
    rooted = parent[postorder] != -1
    post_parent[rooted] = relabel[parent[postorder][rooted]]
    high = relabel[high]
    low = relabel[low]
    # Strictly lower columns in postorder labels: rows below each column.
    off = high != low
    column_order = np.lexsort((high[off], low[off]))
    lower_indptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(np.bincount(low[off], minlength=size), out=lower_indptr[1:])
    lower_indices = high[off][column_order]
    counts = _column_counts(
        post_parent.tolist(), lower_indptr.tolist(), lower_indices.tolist()
    )
    return SymbolicCounts(
        parent=post_parent,
        postorder=postorder,
        column_counts=counts,
        lower_indptr=lower_indptr,
        lower_indices=lower_indices,
        source_rows=high,
        source_columns=low,
        source_positions=positions,
        source_conjugate=conjugate,
    )


def _trapezoid(columns: int, rows: int, /) -> int:
    return columns * rows - columns * (columns - 1) // 2


def relaxed_supernodes(
    parent: np.ndarray, counts: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Fundamental supernodes merged by relaxed amalgamation.

    Returns the column boundaries of the final supernodes and their row counts.
    A child merges into its parent only when its columns immediately precede
    the parent's, so every supernode stays a contiguous column range.
    """
    size = parent.size
    if size == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    child_count = np.bincount(parent[parent >= 0], minlength=size)
    nodes = np.arange(size - 1)
    fused = (
        (parent[:-1] == nodes + 1)
        & (counts[:-1] == counts[1:] + 1)
        & (child_count[1:] == 1)
    )
    starts = np.concatenate(([0], np.flatnonzero(~fused) + 1)).astype(np.int64)
    bounds = np.concatenate((starts, [size])).astype(np.int64)
    fundamental = starts.size
    owner = np.repeat(np.arange(fundamental, dtype=np.int64), np.diff(bounds))
    last = bounds[1:] - 1
    super_parent = np.where(parent[last] >= 0, owner[np.maximum(parent[last], 0)], -1)
    columns = np.diff(bounds).tolist()
    row_counts = counts[starts].tolist()
    structural = np.add.reduceat(counts, starts).tolist() if size else []
    super_parent_list = super_parent.tolist()
    # Group of each fundamental supernode: union-find toward the topmost merge.
    group = list(range(fundamental))
    group_first = list(range(fundamental))
    group_columns = list(columns)
    group_rows = list(row_counts)
    group_structural = list(structural)

    def find(node: int) -> int:
        root = node
        while group[root] != root:
            root = group[root]
        while group[node] != root:
            group[node], node = root, group[node]
        return root

    for child in range(fundamental - 2, -1, -1):
        owner_node = super_parent_list[child]
        if owner_node == -1:
            continue
        target = find(owner_node)
        if group_first[target] != child + 1:
            continue
        merged_columns = columns[child] + group_columns[target]
        merged_rows = columns[child] + group_rows[target]
        stored = _trapezoid(merged_columns, merged_rows)
        zeros = stored - (structural[child] + group_structural[target])
        fraction = zeros / stored
        if not (
            merged_columns <= _RELAX[0]
            or (merged_columns <= _RELAX[1] and fraction < _ZERO[0])
            or (merged_columns <= _RELAX[2] and fraction < _ZERO[1])
            or fraction < _ZERO[2]
        ):
            continue
        group[child] = target
        group_first[target] = child
        group_columns[target] = merged_columns
        group_rows[target] = merged_rows
        group_structural[target] += structural[child]
    roots = sorted(
        {find(node) for node in range(fundamental)}, key=group_first.__getitem__
    )
    boundaries = np.asarray(
        [int(bounds[group_first[root]]) for root in roots] + [size], dtype=np.int64
    )
    return boundaries, np.asarray([group_rows[root] for root in roots], dtype=np.int64)


def supernodal_structure(
    counts: SymbolicCounts, boundaries: np.ndarray, row_counts: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row structures, row offsets, supernodal parents and child counts."""
    supernodes = boundaries.size - 1
    owner = np.repeat(np.arange(supernodes, dtype=np.int64), np.diff(boundaries))
    last = boundaries[1:] - 1
    super_parent = np.where(
        counts.parent[last] >= 0, owner[np.maximum(counts.parent[last], 0)], -1
    )
    row_offsets = np.zeros(supernodes + 1, dtype=np.int64)
    np.cumsum(row_counts, out=row_offsets[1:])
    rows = np.empty(int(row_offsets[-1]), dtype=np.int64)
    pending: list[list[np.ndarray]] = [[] for _ in range(supernodes)]
    lower_indptr = counts.lower_indptr
    lower_indices = counts.lower_indices
    for node in range(supernodes):
        first, stop = int(boundaries[node]), int(boundaries[node + 1])
        below = lower_indices[lower_indptr[first] : lower_indptr[stop]]
        parts = [below[below >= stop], *pending[node]]
        pending[node] = []
        merged = np.unique(np.concatenate(parts))
        structure = np.concatenate((np.arange(first, stop, dtype=np.int64), merged))
        if structure.size != row_counts[node]:
            raise AssertionError("Supernodal row structure disagrees with column counts.")
        rows[row_offsets[node] : row_offsets[node + 1]] = structure
        owner_node = int(super_parent[node])
        if owner_node != -1:
            above = structure[stop - first :]
            pending[owner_node].append(above[above >= boundaries[owner_node + 1]])
    child_counts = np.bincount(super_parent[super_parent >= 0], minlength=supernodes)
    return rows, row_offsets, super_parent, child_counts


def analyze_supernodal(
    size: int,
    indices: np.ndarray,
    indptr: np.ndarray,
    permutation: np.ndarray,
    counts: SymbolicCounts,
    boundaries: np.ndarray,
    row_counts: np.ndarray,
    /,
) -> SupernodalAnalysis:
    """Assemble the immutable supernodal analysis from counted structure."""
    rows, row_offsets, super_parent, child_counts = supernodal_structure(
        counts, boundaries, row_counts
    )
    supernodes = boundaries.size - 1
    widths = np.diff(boundaries)
    heights = np.diff(row_offsets)
    value_offsets = np.zeros(supernodes + 1, dtype=np.int64)
    np.cumsum(widths * heights, out=value_offsets[1:])
    update_sizes = heights - widths
    update_offsets = np.zeros(supernodes + 1, dtype=np.int64)
    np.cumsum(update_sizes, out=update_offsets[1:])
    relative = np.empty(int(update_offsets[-1]), dtype=np.int64)
    contiguous = np.zeros(supernodes, dtype=bool)
    for node in range(supernodes):
        owner_node = int(super_parent[node])
        if owner_node == -1:
            continue
        update = rows[row_offsets[node] + widths[node] : row_offsets[node + 1]]
        target = rows[row_offsets[owner_node] : row_offsets[owner_node + 1]]
        located = np.searchsorted(target, update)
        relative[update_offsets[node] : update_offsets[node + 1]] = located
        contiguous[node] = located.size == 0 or (
            int(located[-1] - located[0]) == located.size - 1
        )
    # Panel position of every chosen input entry (row >= column, postorder labels).
    owner = np.repeat(np.arange(supernodes, dtype=np.int64), widths)
    node_of_entry = owner[counts.source_columns]
    keys = np.repeat(np.arange(supernodes, dtype=np.int64), heights) * size + rows
    entry_keys = node_of_entry * size + counts.source_rows
    located = np.searchsorted(keys, entry_keys)
    local_rows = located - row_offsets[node_of_entry]
    local_columns = counts.source_columns - boundaries[node_of_entry]
    targets = (
        value_offsets[node_of_entry] + local_rows + heights[node_of_entry] * local_columns
    )
    diagonal_columns = np.arange(size, dtype=np.int64)
    diagonal_nodes = owner[diagonal_columns] if size else diagonal_columns
    local = diagonal_columns - boundaries[diagonal_nodes] if size else diagonal_columns
    diagonal_targets = (
        value_offsets[diagonal_nodes] + local + heights[diagonal_nodes] * local
    )
    # Dense kernel flop counts: potrf k³/3, trsm u k², syrk/herk k u (u + 1).
    factor_flops = int(
        np.sum(
            widths**3 // 3
            + widths**2 * update_sizes
            + widths * update_sizes * (update_sizes + 1)
        )
    )
    solve_flops = int(4 * value_offsets[-1])
    # Multifrontal stack: update matrices live from their creation until the
    # parent assembles them; children directly precede their parent.
    live = 0
    peak = 0
    stack: list[int] = []
    front = 0
    for node in range(supernodes):
        for _ in range(int(child_counts[node])):
            live -= stack.pop()
        front = max(front, int(heights[node]) ** 2)
        peak = max(peak, live + int(heights[node]) ** 2)
        entries = int(update_sizes[node]) ** 2
        stack.append(entries)
        live += entries
    elimination = permutation[counts.postorder]
    payload = b"|".join(
        (
            elimination.tobytes(),
            boundaries.tobytes(),
            row_offsets.tobytes(),
            rows.tobytes(),
        )
    )
    return SupernodalAnalysis(
        size=size,
        elimination=elimination,
        supernode_columns=boundaries,
        row_offsets=row_offsets,
        rows=rows,
        value_offsets=value_offsets,
        update_offsets=update_offsets,
        relative=relative,
        contiguous_update=contiguous,
        child_counts=child_counts.astype(np.int64),
        input_sources=counts.source_positions.astype(np.int64),
        input_targets=targets.astype(np.int64),
        input_conjugate=counts.source_conjugate.astype(bool),
        diagonal_targets=diagonal_targets.astype(np.int64),
        structural_nnz=int(counts.column_counts.sum()),
        factor_size=int(value_offsets[-1]),
        factor_flops=factor_flops,
        solve_flops=solve_flops,
        front_entries=front,
        stack_entries=peak,
        analysis_id=sha256(payload).hexdigest(),
    )


def _kernels(dtype: np.dtype, /):  # noqa: ANN202
    complex_ = np.issubdtype(dtype, np.complexfloating)
    (potrf,) = lapack.get_lapack_funcs(("potrf",), dtype=dtype)
    trsm, gemm = blas.get_blas_funcs(("trsm", "gemm"), dtype=dtype)
    (rank_update,) = blas.get_blas_funcs(("herk" if complex_ else "syrk",), dtype=dtype)
    return complex_, potrf, trsm, gemm, rank_update


@dataclass
class _PivotState:
    status: int = STATUS_SUCCESS
    minimum_pivot: float = float("inf")
    minimum_position: int = -1
    replaced: int = 0
    rejected: int = 0

    def observe(self, magnitude: float, position: int, /) -> None:
        """Record a pivot magnitude; ``position < 0`` marks a constant pivot."""
        if magnitude < self.minimum_pivot:
            self.minimum_pivot = magnitude
            self.minimum_position = position


def _unblocked_front(
    front: np.ndarray,
    kinds: np.ndarray,
    columns: int,
    offset: int,
    state: _PivotState,
    pivot_tolerance: float,
    allow_replacement: bool,
    replacement_value: float,
    /,
) -> None:
    """Pivot-by-pivot elimination of a front with typed pivot evidence.

    Reproduces the scalar route's per-pivot acceptance, replacement and status
    rules exactly; it runs only for fronts whose blocked factorization met a
    rejected or non-finite pivot. ``kinds`` receives each pivot's
    `PIVOT_ACCEPTED` or `PIVOT_CONSTANT` (replaced or continued) mark; constant
    pivots have vanishing derivatives. ``offset`` is the front's first panel
    position.
    """
    real_dtype = np.finfo(front.dtype).dtype
    epsilon = float(np.finfo(real_dtype).eps)
    replacement = max(replacement_value, pivot_tolerance)
    height = front.shape[0]
    for pivot_index in range(columns):
        pivot = front[pivot_index, pivot_index]
        finite = bool(np.isfinite(pivot))
        real = float(np.real(pivot)) if finite else float("nan")
        imaginary_tolerance = max(pivot_tolerance, 64.0 * epsilon * max(1.0, abs(real)))
        acceptable = (
            finite
            and real > pivot_tolerance
            and abs(np.imag(pivot)) <= imaginary_tolerance
        )
        if not finite:
            state.status = STATUS_NONFINITE
        elif not acceptable and state.status == STATUS_SUCCESS:
            state.status = STATUS_NONPOSITIVE_PIVOT
        if not acceptable:
            state.rejected += 1
        use_replacement = (not acceptable) and allow_replacement
        if use_replacement and state.status == STATUS_NONPOSITIVE_PIVOT:
            state.status = STATUS_SUCCESS
        if acceptable:
            effective = real
        elif use_replacement:
            effective = replacement
        else:
            effective = 1.0
        state.replaced += int(use_replacement)
        state.observe(
            abs(effective),
            offset + pivot_index * (height + 1) if acceptable else -1,
        )
        kinds[pivot_index] = PIVOT_ACCEPTED if acceptable else PIVOT_CONSTANT
        root = np.sqrt(effective)
        column = front[pivot_index + 1 :, pivot_index]
        column /= root
        front[pivot_index, pivot_index] = root
        front[pivot_index + 1 :, pivot_index + 1 :] -= np.outer(column, np.conj(column))


def _scatter_input(
    analysis: SupernodalAnalysis, values: np.ndarray, complex_: bool, /
) -> np.ndarray:
    panels = np.zeros(analysis.factor_size, dtype=values.dtype)
    gathered = values[analysis.input_sources]
    if complex_:
        gathered = np.where(analysis.input_conjugate, np.conj(gathered), gathered)
    panels[analysis.input_targets] = gathered
    return panels


def _extend_add(
    front: np.ndarray,
    update: np.ndarray,
    located: np.ndarray,
    contiguous: bool,
    /,
) -> None:
    if contiguous:
        first = int(located[0])
        end = first + located.size
        front[first:end, first:end] += update
    else:
        front[np.ix_(located, located)] += update


@dataclass(frozen=True)
class FactorResult:
    """Panels, pivot evidence and per-column pivot kinds of one refresh.

    ``minimum_position`` is the panel position of the minimum accepted pivot
    (``-1`` when the minimum is a replaced or continued constant pivot).
    ``pivot_kinds`` holds one `PIVOT_BLOCKED`, `PIVOT_ACCEPTED` or
    `PIVOT_CONSTANT` mark per elimination column; `tangent_values` reads it to
    differentiate the panels without refactoring.
    """

    panels: np.ndarray
    status: int
    minimum_pivot: float
    minimum_position: int
    replaced: int
    rejected: int
    pivot_kinds: np.ndarray


def factor_values(
    analysis: SupernodalAnalysis,
    values: np.ndarray,
    diagonal_shift: float,
    pivot_tolerance: float,
    allow_replacement: bool,
    replacement_value: float,
    /,
) -> FactorResult:
    """Multifrontal numeric factorization of one value vector."""
    dtype = values.dtype
    complex_, potrf, trsm, _, rank_update = _kernels(dtype)
    panels = _scatter_input(analysis, values, complex_)
    if diagonal_shift:
        panels[analysis.diagonal_targets] += diagonal_shift
    kinds = np.full(analysis.size, PIVOT_BLOCKED, dtype=np.int8)
    state = _PivotState()
    boundaries = analysis.supernode_columns.tolist()
    value_offsets = analysis.value_offsets.tolist()
    row_offsets = analysis.row_offsets.tolist()
    update_offsets = analysis.update_offsets.tolist()
    child_counts = analysis.child_counts.tolist()
    contiguous = analysis.contiguous_update.tolist()
    relative = analysis.relative
    real_dtype = np.finfo(dtype).dtype
    epsilon = float(np.finfo(real_dtype).eps)
    stack: list[tuple[int, np.ndarray]] = []
    transpose = 2 if complex_ else 1
    for node in range(len(boundaries) - 1):
        columns = boundaries[node + 1] - boundaries[node]
        height = row_offsets[node + 1] - row_offsets[node]
        start, stop = value_offsets[node], value_offsets[node + 1]
        front = np.zeros((height, height), dtype=dtype, order="F")
        front[:, :columns] = panels[start:stop].reshape((columns, height)).T
        for _ in range(child_counts[node]):
            child, update = stack.pop()
            if update.size == 0:
                continue
            located = relative[update_offsets[child] : update_offsets[child + 1]]
            _extend_add(front, update, located, contiguous[child])
        diagonal = np.diagonal(front)[:columns]
        blocked = True
        if complex_:
            real = np.abs(diagonal.real)
            tolerance = np.maximum(
                pivot_tolerance, 64.0 * epsilon * np.maximum(1.0, real)
            )
            blocked = bool(np.all(np.abs(diagonal.imag) <= tolerance))
        factor = None
        if blocked:
            factor, info = potrf(front[:columns, :columns], lower=1, clean=1)
            if info != 0:
                blocked = False
            else:
                pivots = np.abs(np.diagonal(factor)) ** 2
                if pivot_tolerance > 0.0 and not np.all(pivots > pivot_tolerance):
                    blocked = False
                else:
                    smallest = int(np.argmin(pivots))
                    state.observe(
                        float(pivots[smallest]), start + smallest * (height + 1)
                    )
        if blocked and factor is not None:
            panel = np.empty((height, columns), dtype=dtype, order="F")
            panel[:columns] = factor
            if height > columns:
                lower = trsm(
                    1.0,
                    factor,
                    front[columns:, :columns],
                    side=1,
                    lower=1,
                    trans_a=transpose,
                )
                panel[columns:] = lower
                update = rank_update(
                    -1.0,
                    lower,
                    beta=1.0,
                    c=np.asfortranarray(front[columns:, columns:]),
                    lower=1,
                    trans=0,
                )
            else:
                update = np.empty((0, 0), dtype=dtype)
        else:
            first = boundaries[node]
            _unblocked_front(
                front,
                kinds[first : first + columns],
                columns,
                start,
                state,
                pivot_tolerance,
                allow_replacement,
                replacement_value,
            )
            panel = np.asfortranarray(front[:, :columns])
            panel[:columns] = np.tril(panel[:columns])
            update = np.asfortranarray(np.tril(front[columns:, columns:]))
        panels[start:stop] = panel.reshape(-1, order="F")
        stack.append((node, update))
    if not np.all(np.isfinite(panels)):
        state.status = STATUS_NONFINITE
    return FactorResult(
        panels=panels,
        status=state.status,
        minimum_pivot=state.minimum_pivot,
        minimum_position=state.minimum_position,
        replaced=state.replaced,
        rejected=state.rejected,
        pivot_kinds=kinds,
    )


def tangent_values(
    analysis: SupernodalAnalysis,
    panels: np.ndarray,
    pivot_kinds: np.ndarray,
    seed: np.ndarray,
    /,
) -> np.ndarray:
    """Panel tangent ``dL`` of refreshed panels for a panel-pattern seed ``dF``.

    Solves the linearized factor identity ``tril(dL Lᴴ + L dLᴴ) = dF`` on the
    factor pattern by one multifrontal sweep over the stored panels, without
    refactoring: ``dL₁₁ = L₁₁ Φ(L₁₁⁻¹ dF₁₁ L₁₁⁻ᴴ)`` (``Φ`` keeps the strict
    lower triangle and half the diagonal), ``dL₂₁ = (dF₂₁ − L₂₁ dL₁₁ᴴ) L₁₁⁻ᴴ``
    and the update tangent ``dF₂₂ − dL₂₁ L₂₁ᴴ − L₂₁ dL₂₁ᴴ``. Fronts the refresh
    eliminated pivot by pivot repeat that order; a constant pivot ignores its
    diagonal seed, so its column tangent only rescales. The seed may carry
    fill-pattern entries, which is how higher derivatives recurse.
    """
    dtype = np.result_type(panels.dtype, seed.dtype)
    complex_, _, trsm, _, _ = _kernels(dtype)
    panels = panels.astype(dtype, copy=False)
    tangents = np.zeros(analysis.factor_size, dtype=dtype)
    boundaries = analysis.supernode_columns.tolist()
    value_offsets = analysis.value_offsets.tolist()
    row_offsets = analysis.row_offsets.tolist()
    update_offsets = analysis.update_offsets.tolist()
    child_counts = analysis.child_counts.tolist()
    contiguous = analysis.contiguous_update.tolist()
    relative = analysis.relative
    stack: list[tuple[int, np.ndarray]] = []
    transpose = 2 if complex_ else 1
    for node in range(len(boundaries) - 1):
        first = boundaries[node]
        columns = boundaries[node + 1] - first
        height = row_offsets[node + 1] - row_offsets[node]
        start, stop = value_offsets[node], value_offsets[node + 1]
        panel = panels[start:stop].reshape((columns, height)).T
        front = np.zeros((height, height), dtype=dtype, order="F")
        front[:, :columns] = seed[start:stop].reshape((columns, height)).T
        for _ in range(child_counts[node]):
            child, update = stack.pop()
            if update.size == 0:
                continue
            located = relative[update_offsets[child] : update_offsets[child + 1]]
            _extend_add(front, update, located, contiguous[child])
        if pivot_kinds[first] == PIVOT_BLOCKED:
            factor = np.asfortranarray(panel[:columns])
            block = front[:columns, :columns]
            hermitian = np.tril(block) + np.tril(block, -1).conj().T
            scaled = trsm(1.0, factor, hermitian, lower=1)
            scaled = trsm(1.0, factor, scaled, side=1, lower=1, trans_a=transpose)
            phi = np.tril(scaled, -1) + np.diag(0.5 * np.real(np.diagonal(scaled)))
            factor_tangent = factor @ phi
            panel_tangent = np.empty((height, columns), dtype=dtype, order="F")
            panel_tangent[:columns] = factor_tangent
            if height > columns:
                lower = np.asfortranarray(panel[columns:])
                lower_tangent = trsm(
                    1.0,
                    factor,
                    front[columns:, :columns] - lower @ factor_tangent.conj().T,
                    side=1,
                    lower=1,
                    trans_a=transpose,
                )
                panel_tangent[columns:] = lower_tangent
                product = lower_tangent @ lower.conj().T
                update = np.asfortranarray(
                    front[columns:, columns:] - np.tril(product + product.conj().T)
                )
            else:
                update = np.empty((0, 0), dtype=dtype)
        else:
            for pivot_index in range(columns):
                root = float(np.real(panel[pivot_index, pivot_index]))
                column = panel[pivot_index + 1 :, pivot_index]
                root_tangent = (
                    float(np.real(front[pivot_index, pivot_index])) / (2.0 * root)
                    if pivot_kinds[first + pivot_index] == PIVOT_ACCEPTED
                    else 0.0
                )
                front[pivot_index, pivot_index] = root_tangent
                column_tangent = front[pivot_index + 1 :, pivot_index]
                column_tangent -= column * root_tangent
                column_tangent /= root
                front[pivot_index + 1 :, pivot_index + 1 :] -= np.outer(
                    column_tangent, np.conj(column)
                ) + np.outer(column, np.conj(column_tangent))
            panel_tangent = np.asfortranarray(front[:, :columns])
            panel_tangent[:columns] = np.tril(panel_tangent[:columns])
            update = np.asfortranarray(np.tril(front[columns:, columns:]))
        tangents[start:stop] = panel_tangent.reshape(-1, order="F")
        stack.append((node, update))
    return tangents


def symmetric_product(
    analysis: SupernodalAnalysis,
    left: np.ndarray,
    right: np.ndarray,
    /,
) -> np.ndarray:
    """Panels of ``tril(X Yᴴ + Y Xᴴ)`` for two panel-stored lower factors.

    The Cholesky fill pattern is closed under this product: column ``k``
    couples only rows of its own supernode's structure, so each supernode's
    outer products land in its panel and, through the multifrontal update
    matrices, in its ancestors' panels.
    """
    dtype = np.result_type(left.dtype, right.dtype)
    product_panels = np.zeros(analysis.factor_size, dtype=dtype)
    boundaries = analysis.supernode_columns.tolist()
    value_offsets = analysis.value_offsets.tolist()
    row_offsets = analysis.row_offsets.tolist()
    update_offsets = analysis.update_offsets.tolist()
    child_counts = analysis.child_counts.tolist()
    contiguous = analysis.contiguous_update.tolist()
    relative = analysis.relative
    stack: list[tuple[int, np.ndarray]] = []
    for node in range(len(boundaries) - 1):
        columns = boundaries[node + 1] - boundaries[node]
        height = row_offsets[node + 1] - row_offsets[node]
        start, stop = value_offsets[node], value_offsets[node + 1]
        first_panel = np.asfortranarray(
            left[start:stop].reshape((columns, height)).T, dtype=dtype
        )
        second_panel = np.asfortranarray(
            right[start:stop].reshape((columns, height)).T, dtype=dtype
        )
        product = first_panel @ second_panel.conj().T
        front = np.asfortranarray(product + product.conj().T)
        for _ in range(child_counts[node]):
            child, update = stack.pop()
            if update.size == 0:
                continue
            located = relative[update_offsets[child] : update_offsets[child + 1]]
            _extend_add(front, update, located, contiguous[child])
        panel = np.asfortranarray(front[:, :columns])
        panel[:columns] = np.tril(panel[:columns])
        product_panels[start:stop] = panel.reshape(-1, order="F")
        stack.append((node, np.asfortranarray(np.tril(front[columns:, columns:]))))
    return product_panels


def solve_values(
    analysis: SupernodalAnalysis,
    panels: np.ndarray,
    right_hand_side: np.ndarray,
    /,
) -> tuple[np.ndarray, int, int]:
    """Solve ``A X = B`` for an ``(n, k)`` block with the stored panels.

    Returns the solution and the forward and backward sweep statuses.
    """
    dtype = np.result_type(panels.dtype, right_hand_side.dtype)
    complex_, _, trsm, gemm, _ = _kernels(dtype)
    panels = panels.astype(dtype, copy=False)
    work = np.asfortranarray(right_hand_side[analysis.elimination].astype(dtype))
    boundaries = analysis.supernode_columns.tolist()
    value_offsets = analysis.value_offsets.tolist()
    row_offsets = analysis.row_offsets.tolist()
    rows = analysis.rows
    supernodes = len(boundaries) - 1
    transpose = 2 if complex_ else 1
    views = []
    for node in range(supernodes):
        columns = boundaries[node + 1] - boundaries[node]
        height = row_offsets[node + 1] - row_offsets[node]
        panel = (
            panels[value_offsets[node] : value_offsets[node + 1]]
            .reshape((columns, height))
            .T
        )
        views.append((boundaries[node], columns, height, panel))
    for node, (first, columns, height, panel) in enumerate(views):
        block = trsm(1.0, panel[:columns], work[first : first + columns], lower=1)
        work[first : first + columns] = block
        if height > columns:
            below = rows[row_offsets[node] + columns : row_offsets[node + 1]]
            work[below] -= panel[columns:] @ block
    forward = TRIANGULAR_SUCCESS if np.all(np.isfinite(work)) else TRIANGULAR_NONFINITE
    for node in range(supernodes - 1, -1, -1):
        first, columns, height, panel = views[node]
        block = work[first : first + columns]
        if height > columns:
            below = rows[row_offsets[node] + columns : row_offsets[node + 1]]
            block = block - gemm(1.0, panel[columns:], work[below], trans_a=transpose)
        work[first : first + columns] = trsm(
            1.0, panel[:columns], block, lower=1, trans_a=transpose
        )
    backward = TRIANGULAR_SUCCESS if np.all(np.isfinite(work)) else TRIANGULAR_NONFINITE
    solution = np.empty_like(work)
    solution[analysis.elimination] = work
    return solution, forward, backward


SupernodalKind = Literal["cholesky"]

__all__ = [
    "SupernodalAnalysis",
    "SymbolicCounts",
    "analyze_supernodal",
    "factor_values",
    "relaxed_supernodes",
    "solve_values",
    "symbolic_counts",
    "symmetric_entries",
    "symmetric_product",
    "tangent_values",
]
