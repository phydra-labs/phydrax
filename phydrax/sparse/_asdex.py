#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from jaxtyping import Array

from ..linalg import AbstractVectorSpace
from ._coloring import (
    SparseColoring,
    SparseDerivativeKind,
    SparseDerivativeMode,
)
from ._pattern import SparsePattern


def compile_asdex_coloring(
    function: Callable[[Array, Any], Array],
    coordinates: Array,
    sample_args: Any,
    /,
    *,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    pattern: SparsePattern | None,
    derivative_kind: SparseDerivativeKind,
    mode: SparseDerivativeMode | None,
    symmetric: bool,
) -> SparseColoring:
    """Compile ASDEX detection/coloring into a portable Phydrax artifact."""

    import asdex

    def bound_function(value: Array) -> Array:
        return function(value, sample_args)

    if derivative_kind == "jacobian":
        if mode is not None and mode not in ("fwd", "rev"):
            raise ValueError("Jacobian mode must be 'fwd', 'rev', or None.")
        if pattern is None:
            colored = asdex.jacobian_coloring(
                bound_function,
                coordinates,
                argnums=0,
                mode=mode,
                symmetric=symmetric,
            )
            pattern_ = _normalize_pattern(
                colored.sparsity,
                expected_shape=(target.size, source.size),
                symmetric=symmetric,
            )
        else:
            sparsity = asdex.SparsityPattern.from_coo(
                np.asarray(pattern.rows),
                np.asarray(pattern.cols),
                pattern.shape,
            )
            colored = asdex.jacobian_coloring_from_sparsity(
                sparsity,
                mode=mode,
                symmetric=symmetric,
            )
            pattern_ = pattern
    elif derivative_kind == "hessian":
        if mode is not None and mode not in (
            "fwd_over_rev",
            "rev_over_fwd",
            "rev_over_rev",
        ):
            raise ValueError(
                "Hessian mode must be 'fwd_over_rev', 'rev_over_fwd', "
                "'rev_over_rev', or None."
            )
        if pattern is None:
            colored = asdex.hessian_coloring(
                bound_function,
                coordinates,
                argnums=0,
                mode=mode,
                symmetric=True,
            )
            pattern_ = _normalize_pattern(
                colored.sparsity,
                expected_shape=(source.size, source.size),
                symmetric=True,
            )
        else:
            sparsity = asdex.SparsityPattern.from_coo(
                np.asarray(pattern.rows),
                np.asarray(pattern.cols),
                pattern.shape,
            )
            colored = asdex.hessian_coloring_from_sparsity(
                sparsity,
                mode=mode,
                symmetric=True,
            )
            pattern_ = pattern
    else:
        raise ValueError(f"Unknown sparse derivative kind {derivative_kind!r}.")

    return _normalize_coloring(pattern_, colored)


def _normalize_pattern(
    sparsity: Any,
    /,
    *,
    expected_shape: tuple[int, int],
    symmetric: bool,
) -> SparsePattern:
    shape = (int(sparsity.shape[0]), int(sparsity.shape[1]))
    if shape != expected_shape:
        raise ValueError(
            f"ASDEX produced sparse shape {shape}; expected {expected_shape}."
        )
    return SparsePattern.from_coo(
        np.asarray(sparsity.rows),
        np.asarray(sparsity.cols),
        shape,
        symmetric=symmetric,
        origin="asdex",
    )


def _normalize_coloring(pattern: SparsePattern, colored: Any, /) -> SparseColoring:
    colors = np.asarray(colored.colors, dtype=np.int32)
    rows = np.asarray(pattern.rows, dtype=np.int64)
    cols = np.asarray(pattern.cols, dtype=np.int64)
    mode = colored.mode
    if pattern.nnz == 0:
        gather_colors = np.empty((0,), dtype=np.int32)
        gather_elements = np.empty((0,), dtype=np.int32)
    elif bool(colored.symmetric):
        gather_colors, gather_elements = _symmetric_extraction(
            rows,
            cols,
            colors,
            colored.star_set,
            pattern.source_size,
        )
    elif mode == "rev":
        gather_colors = colors[rows]
        gather_elements = cols
    else:
        gather_colors = colors[cols]
        gather_elements = rows
    return SparseColoring(
        pattern,
        colors,
        gather_colors,
        gather_elements,
        mode=mode,
        symmetric=bool(colored.symmetric),
        compiler="asdex",
        num_colors=int(colored.num_colors),
    )


def _symmetric_extraction(
    rows: np.ndarray,
    cols: np.ndarray,
    colors: np.ndarray,
    star_set: Any,
    dimension: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if star_set is None:
        raise ValueError("ASDEX symmetric coloring did not provide a star set.")
    gather_colors = np.empty(rows.shape, dtype=np.int32)
    gather_elements = np.empty(rows.shape, dtype=np.int32)
    diagonal = rows == cols
    gather_colors[diagonal] = colors[rows[diagonal]]
    gather_elements[diagonal] = rows[diagonal]

    off_diagonal = ~diagonal
    row = rows[off_diagonal]
    column = cols[off_diagonal]
    if row.size == 0:
        return gather_colors, gather_elements
    keys = np.minimum(row, column) * np.int64(dimension) + np.maximum(row, column)
    edge_keys = np.asarray(star_set.edge_lo, dtype=np.int64) * np.int64(
        dimension
    ) + np.asarray(star_set.edge_hi, dtype=np.int64)
    positions = np.searchsorted(edge_keys, keys)
    if np.any(positions >= edge_keys.size) or not np.array_equal(
        edge_keys[positions], keys
    ):
        raise ValueError("ASDEX star coloring omits a sparse pattern edge.")
    star = np.asarray(star_set.star)
    hub = np.asarray(star_set.hub)
    edge_positions = np.asarray(star_set.edge_pos)
    hubs = hub[star[edge_positions[positions]]].astype(np.int64)
    hubs = np.where(hubs < 0, -hubs - 1, hubs)
    gather_colors[off_diagonal] = colors[hubs]
    gather_elements[off_diagonal] = np.where(hubs == column, row, column)
    if np.any(gather_colors < 0):
        raise ValueError("ASDEX star coloring produced a neutral extraction color.")
    return gather_colors, gather_elements


__all__ = ["compile_asdex_coloring"]
