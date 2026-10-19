from __future__ import annotations

import math

import numpy as np


def row_order(rows: np.ndarray, /) -> np.ndarray:
    """Return stable lexicographic order using a mixed-radix key when bounded."""
    if rows.size == 0:
        return np.arange(rows.shape[0], dtype=np.int64)
    lows = np.min(rows, axis=0).astype(np.int64)
    spans = np.max(rows, axis=0).astype(np.int64) - lows + 1
    if math.prod(int(span) for span in spans) > np.iinfo(np.int64).max:
        return np.lexsort(rows.T[::-1])
    key = np.zeros((rows.shape[0],), dtype=np.int64)
    for column, low, span in zip(rows.T, lows, spans, strict=True):
        key *= span
        key += column.astype(np.int64) - low
    return np.argsort(key, kind="stable")


def row_run_starts(ordered: np.ndarray, /) -> np.ndarray:
    starts = np.ones((ordered.shape[0],), dtype=np.bool_)
    np.any(ordered[1:] != ordered[:-1], axis=1, out=starts[1:])
    return starts


def has_duplicate_rows(rows: np.ndarray, /) -> bool:
    """Return whether any two rows of a rank-two integer array are equal."""
    return not np.all(row_run_starts(rows[row_order(rows)]))


def facet_columns(width: int, /) -> np.ndarray:
    """Return vertex-column rows for simplex facets opposite each local vertex."""
    return np.asarray(
        [[j for j in range(width) if j != i] for i in range(width)], dtype=np.int32
    )
