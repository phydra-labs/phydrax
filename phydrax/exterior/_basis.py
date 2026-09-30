"""Cached immutable host tables for lexicographic exterior coordinates."""

from __future__ import annotations

from functools import cache
from itertools import combinations

from .._validation import nonnegative_integer


def exterior_indices(dimension: int, degree: int, /) -> tuple[tuple[int, ...], ...]:
    n = nonnegative_integer(dimension, "dimension")
    k = nonnegative_integer(degree, "degree")
    if k > n:
        raise ValueError("Exterior degree exceeds dimension.")
    return tuple(combinations(range(n), k))


def wedge_sign(left: tuple[int, ...], right: tuple[int, ...], /) -> int:
    if set(left).intersection(right):
        return 0
    return -1 if sum(a > b for a in left for b in right) % 2 else 1


def axes_bitmap(axes: tuple[int, ...], dimension: int, /) -> int:
    n = nonnegative_integer(dimension, "dimension")
    if tuple(sorted(set(axes))) != axes or any(a < 0 or a >= n for a in axes):
        raise ValueError("Blade axes must be unique, increasing, and in range.")
    return sum(1 << a for a in axes)


def bitmap_axes(bitmap: int, dimension: int, /) -> tuple[int, ...]:
    value = nonnegative_integer(bitmap, "bitmap")
    n = nonnegative_integer(dimension, "dimension")
    if value >> n:
        raise ValueError("Blade bitmap exceeds dimension.")
    return tuple(a for a in range(n) if value & (1 << a))


type ExteriorTable = tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]


@cache
def wedge_table(n: int, k: int, l: int, /) -> ExteriorTable:
    left, right = exterior_indices(n, k), exterior_indices(n, l)
    output = {index: i for i, index in enumerate(exterior_indices(n, k + l))}
    rows = [
        (a, b, output[tuple(sorted(i + j))], wedge_sign(i, j))
        for a, i in enumerate(left)
        for b, j in enumerate(right)
        if set(i).isdisjoint(j)
    ]
    return _columns(rows)


@cache
def derivative_table(n: int, k: int, /) -> ExteriorTable:
    source = {index: i for i, index in enumerate(exterior_indices(n, k))}
    rows = [
        (source[index[:p] + index[p + 1 :]], axis, out, (-1) ** p)
        for out, index in enumerate(exterior_indices(n, k + 1))
        for p, axis in enumerate(index)
    ]
    return _columns(rows)


@cache
def interior_table(n: int, k: int, /) -> ExteriorTable:
    source = {index: i for i, index in enumerate(exterior_indices(n, k))}
    rows = []
    for out, index in enumerate(exterior_indices(n, k - 1)):
        for axis in range(n):
            if axis not in index:
                full = tuple(sorted((axis, *index)))
                rows.append((axis, source[full], out, (-1) ** full.index(axis)))
    return _columns(rows)


@cache
def complement_table(n: int, k: int, /) -> tuple[tuple[int, ...], tuple[int, ...]]:
    output = {index: i for i, index in enumerate(exterior_indices(n, n - k))}
    complements = [
        tuple(a for a in range(n) if a not in index) for index in exterior_indices(n, k)
    ]
    return (
        tuple(output[index] for index in complements),
        tuple(
            wedge_sign(source, target)
            for source, target in zip(exterior_indices(n, k), complements, strict=True)
        ),
    )


def _columns(rows: list[tuple[int, int, int, int]], /) -> ExteriorTable:
    return (
        tuple(row[0] for row in rows),
        tuple(row[1] for row in rows),
        tuple(row[2] for row in rows),
        tuple(row[3] for row in rows),
    )


__all__ = [
    "axes_bitmap",
    "bitmap_axes",
    "exterior_indices",
    "wedge_sign",
    "wedge_table",
    "derivative_table",
    "interior_table",
    "complement_table",
]
