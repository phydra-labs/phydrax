#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations


def _compositions(total: int, dimension: int, /):
    if dimension == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for remainder in _compositions(total - first, dimension - 1):
            yield (first, *remainder)


def total_degree_multiindices(
    dimension: int,
    degree: int,
    /,
    *,
    include_constant: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Return stable total-degree multiindices through one degree."""

    dimension_ = int(dimension)
    degree_ = int(degree)
    if dimension_ <= 0 or degree_ < 0:
        raise ValueError("Polynomial dimension must be positive and degree nonnegative.")
    start = 0 if include_constant else 1
    return tuple(
        value
        for total in range(start, degree_ + 1)
        for value in _compositions(total, dimension_)
    )


__all__ = ["total_degree_multiindices"]
