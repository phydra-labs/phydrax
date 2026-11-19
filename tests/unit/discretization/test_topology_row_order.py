"""Packed-key row ordering against NumPy's lexicographic references."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization._topology import (
    _has_duplicate_rows,
    _row_order,
    sorted_unique_rows,
    unique_rows_inverse,
)


# Rows whose mixed-radix key fits int64 with (narrow) and without (wide) the
# row index beside it in one uint64, and rows whose key exceeds int64
# (lexicographic fallback); each draws repeated rows and signed values.
_CASES = {
    "narrow": (4096, ((-3, 40), (0, 7), (5, 900))),
    "uint64-index": (3000, ((0, 2**20), (0, 2**20), (0, 2**12))),
    "stable-argsort": (3000, ((0, 2**26), (0, 2**26), (0, 2**8))),
    "lexicographic": (2000, ((0, 2**40), (0, 2**30), (0, 4))),
}


def _rows(name: str, /) -> np.ndarray:
    count, bounds = _CASES[name]
    rng = np.random.default_rng(sum(map(ord, name)))
    columns = [rng.integers(low, high, size=count // 4) for low, high in bounds]
    unique = np.stack(columns, axis=1).astype(np.int64)
    # Every row is drawn once or more so equal rows exercise stability.
    return unique[rng.integers(0, unique.shape[0], size=count)]


@pytest.mark.parametrize("name", tuple(_CASES))
def test_row_order_is_the_stable_lexicographic_order(name: str) -> None:
    rows = _rows(name)
    np.testing.assert_array_equal(_row_order(rows), np.lexsort(rows.T[::-1]))


@pytest.mark.parametrize("name", tuple(_CASES))
def test_unique_rows_match_numpy(name: str) -> None:
    rows = _rows(name)
    expected, inverse, counts = np.unique(
        rows, axis=0, return_inverse=True, return_counts=True
    )
    np.testing.assert_array_equal(sorted_unique_rows(rows), expected)
    observed = unique_rows_inverse(rows)
    np.testing.assert_array_equal(observed[0], expected)
    np.testing.assert_array_equal(observed[1], inverse.reshape(-1))
    np.testing.assert_array_equal(observed[2], counts)
    assert sorted_unique_rows(rows).dtype == rows.dtype
    assert _has_duplicate_rows(rows)
    assert not _has_duplicate_rows(expected)
