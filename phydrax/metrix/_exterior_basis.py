#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations


def exterior_indices(
    dimension: int,
    degree: int,
    /,
) -> tuple[tuple[int, ...], ...]:
    """Return the canonical increasing basis indices for one exterior grade."""
    dimension_ = int(dimension)
    degree_ = int(degree)
    if dimension_ < 0:
        raise ValueError("Exterior dimension must be nonnegative.")
    if degree_ < 0 or degree_ > dimension_:
        raise ValueError(f"Exterior degree must lie in [0, {dimension_}]; got {degree_}.")
    return tuple(combinations(range(dimension_), degree_))


def wedge_sign(left: tuple[int, ...], right: tuple[int, ...], /) -> int:
    """Return the canonical reordering sign for two disjoint basis blades."""
    if set(left).intersection(right):
        return 0
    inversions = sum(left_axis > right_axis for left_axis in left for right_axis in right)
    return -1 if inversions % 2 else 1


def axes_bitmap(axes: tuple[int, ...], dimension: int, /) -> int:
    """Encode one increasing exterior basis index as a Python integer bitmap."""
    dimension_ = int(dimension)
    if dimension_ < 0:
        raise ValueError("Exterior dimension must be nonnegative.")
    if tuple(sorted(axes)) != axes or len(set(axes)) != len(axes):
        raise ValueError("Exterior basis axes must be unique and increasing.")
    if any(axis < 0 or axis >= dimension_ for axis in axes):
        raise ValueError("Exterior basis axis lies outside the declared dimension.")
    bitmap = 0
    for axis in axes:
        bitmap |= 1 << axis
    return bitmap


def bitmap_axes(bitmap: int, dimension: int, /) -> tuple[int, ...]:
    """Decode a nonnegative blade bitmap within the declared dimension."""
    bitmap_ = int(bitmap)
    dimension_ = int(dimension)
    if bitmap_ < 0:
        raise ValueError("Blade bitmap must be nonnegative.")
    if dimension_ < 0:
        raise ValueError("Exterior dimension must be nonnegative.")
    if bitmap_ >> dimension_:
        raise ValueError("Blade bitmap exceeds the declared dimension.")
    return tuple(axis for axis in range(dimension_) if bitmap_ & (1 << axis))


__all__ = ["axes_bitmap", "bitmap_axes", "exterior_indices", "wedge_sign"]
