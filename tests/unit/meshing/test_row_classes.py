# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Membership class ranks equal NumPy's lexicographic row inverse."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.meshing._host_banks import row_classes


def _reference(rows: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    return inverse.reshape((-1,)).astype(np.int64)


@pytest.mark.parametrize(
    "dtype",
    (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32, np.uint64),
)
@pytest.mark.parametrize("shape", ((1, 3), (9, 2), (64, 5), (257, 131)))
def test_row_classes_equal_lexicographic_inverse(
    dtype: type[np.generic], shape: tuple[int, int]
) -> None:
    rng = np.random.default_rng(sum(shape))
    if dtype is np.bool_:
        rows = rng.integers(0, 2, size=shape).astype(np.bool_)
    else:
        info = np.iinfo(np.dtype(dtype).str)
        rows = rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)
        # Small ranges pack many columns per key; extreme ranges do not.
        rows[:, ::2] = rng.integers(
            max(int(info.min), -2), 3, size=rows[:, ::2].shape, dtype=dtype
        )
    rows[shape[0] // 2 :] = rows[: shape[0] - shape[0] // 2]
    result = row_classes(rows)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, _reference(rows))


def test_row_classes_keep_extreme_and_wide_membership_order() -> None:
    extreme = np.asarray(
        (
            (np.iinfo(np.int64).min, 0),
            (np.iinfo(np.int64).max, -1),
            (np.iinfo(np.int64).min, 0),
            (0, np.iinfo(np.int64).max),
        ),
        dtype=np.int64,
    )
    np.testing.assert_array_equal(row_classes(extreme), _reference(extreme))
    unsigned = np.asarray(
        ((2**64 - 1, 0, 5), (0, 2**64 - 1, 5), (2**64 - 1, 0, 5)), dtype=np.uint64
    )
    np.testing.assert_array_equal(row_classes(unsigned), _reference(unsigned))
    rng = np.random.default_rng(7)
    membership = (rng.random((2048, 1053)) < 0.004).astype(np.int64)
    membership[1024:1536] = membership[:512]
    np.testing.assert_array_equal(row_classes(membership), _reference(membership))
    empty = np.zeros((5, 0), dtype=np.bool_)
    np.testing.assert_array_equal(row_classes(empty), _reference(empty))
