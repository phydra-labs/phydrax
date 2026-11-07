#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Packed, breadth-first traversal of the adaptive implicit cover."""

from __future__ import annotations

import math
import mmap
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import overload, Protocol

import numpy as np
from numpy.typing import DTypeLike

from ._policy import AdaptiveImplicitBoxIssue, AdaptiveImplicitSurfaceStatus


_LeafSink = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    None,
]


class _BoxEvidence(Protocol):
    @property
    def value_lower(self) -> np.ndarray: ...
    @property
    def value_upper(self) -> np.ndarray: ...
    @property
    def gradient_lower(self) -> np.ndarray: ...
    @property
    def gradient_upper(self) -> np.ndarray: ...


class _Bounds(Protocol):
    def boxes(self, lower: np.ndarray, upper: np.ndarray) -> _BoxEvidence: ...


# This is also the numerical batch size of the original traversal.  In
# particular, invalid boxes precede valid boxes *within each such batch*.
_BATCH = 64
_BLOCK = 16384
# Boxes enclosed per call to the field bounds: a whole number of batches.
_EVALUATION_BLOCK = 32 * _BATCH
_CHILDREN = np.asarray(
    [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
    dtype=np.int64,
)


@dataclass(slots=True)
class _LeafAuxiliary:
    level: np.ndarray | None
    index: np.ndarray | None
    issue: np.ndarray | None
    surface_rows: np.ndarray | None

    def get(self, name: str) -> np.ndarray:
        value = getattr(self, name)
        if value is None:
            raise RuntimeError("Packed leaf auxiliary evidence has been consumed.")
        return value

    def release(self) -> None:
        if self.level is None:
            raise RuntimeError(
                "Packed leaf auxiliary evidence has already been consumed."
            )
        self.level = self.index = self.issue = self.surface_rows = None


@dataclass(frozen=True, slots=True)
class PackedLeaves:
    """Immutable traversal evidence.

    ``index`` is int64 through level 62; traversals reaching level 63 use
    Python integers in an object array to preserve the legacy unbounded grid
    arithmetic. This internal representation does not change the final
    scaffold's int64 conversion or its overflow behavior.
    """

    level: np.ndarray
    index: np.ndarray
    boxes: np.ndarray
    value_lower: np.ndarray
    value_upper: np.ndarray
    gradient_lower: np.ndarray
    gradient_upper: np.ndarray
    issue: np.ndarray
    surface_rows: np.ndarray
    deepest_level: int
    status: int


@dataclass(slots=True)
class PackedPolygonLeaves:
    """Private consumable view of a leaf record, retaining no source record."""

    _auxiliary: _LeafAuxiliary
    boxes: np.ndarray
    value_lower: np.ndarray
    value_upper: np.ndarray
    gradient_lower: np.ndarray
    gradient_upper: np.ndarray
    deepest_level: int
    status: int

    @classmethod
    def from_leaves(cls, leaves: PackedLeaves) -> PackedPolygonLeaves:
        return cls(
            _LeafAuxiliary(leaves.level, leaves.index, leaves.issue, leaves.surface_rows),
            leaves.boxes,
            leaves.value_lower,
            leaves.value_upper,
            leaves.gradient_lower,
            leaves.gradient_upper,
            leaves.deepest_level,
            leaves.status,
        )

    @property
    def level(self) -> np.ndarray:
        return self._auxiliary.get("level")

    @property
    def index(self) -> np.ndarray:
        return self._auxiliary.get("index")

    @property
    def issue(self) -> np.ndarray:
        return self._auxiliary.get("issue")

    @property
    def surface_rows(self) -> np.ndarray:
        return self._auxiliary.get("surface_rows")

    def _release_auxiliary(self) -> None:
        """Consume traversal-only columns once their owner has finished with them."""
        self._auxiliary.release()


class _Columns:
    """Append bounded batches without geometric full-buffer copies."""

    def __init__(
        self, shapes: tuple[tuple[int, ...], ...], dtypes: tuple[DTypeLike, ...]
    ) -> None:
        self.shapes = shapes
        self.dtypes = dtypes
        self.blocks: list[list[np.ndarray]] = [[] for _ in shapes]
        self.size = 0

    def append(self, *columns: np.ndarray) -> None:
        count = len(columns[0])
        offset = 0
        while offset < count:
            position = self.size % _BLOCK
            if position == 0:
                for blocks, shape, dtype in zip(self.blocks, self.shapes, self.dtypes):
                    dtype_ = np.dtype(dtype)
                    if dtype_.hasobject:
                        block = np.empty((_BLOCK, *shape), dtype=dtype_)
                    else:
                        size = _BLOCK * math.prod(shape) * dtype_.itemsize
                        mapping = mmap.mmap(-1, size)
                        block = np.ndarray((_BLOCK, *shape), dtype=dtype_, buffer=mapping)
                    blocks.append(block)
                del block
            n = min(count - offset, _BLOCK - position)
            for blocks, column in zip(self.blocks, columns):
                blocks[-1][position : position + n] = column[offset : offset + n]
            self.size += n
            offset += n

    def promote_index_to_objects(self) -> None:
        """Preserve exact indices when a traversal exceeds signed int64."""
        for i, block in enumerate(self.blocks[1]):
            self.blocks[1][i] = block.astype(object)
            mapping = block.base
            del block
            if isinstance(mapping, mmap.mmap):
                mapping.close()
        dtypes = list(self.dtypes)
        dtypes[1] = object
        self.dtypes = tuple(dtypes)

    def finish(self) -> tuple[np.ndarray, ...]:
        result = []
        for blocks, shape, dtype in zip(self.blocks, self.shapes, self.dtypes):
            blocks.reverse()
            array = np.empty((self.size, *shape), dtype=dtype)
            offset = 0
            while blocks:
                block = blocks.pop()
                n = min(_BLOCK, self.size - offset)
                array[offset : offset + n] = block[:n]
                mapping = block.base
                del block
                if isinstance(mapping, mmap.mmap):
                    mapping.close()
                offset += n
            array.flags.writeable = False
            result.append(array)
        return tuple(result)


class _Coordinates:
    """Memoize exact dyadic coordinates by axis, not by octree cell."""

    def __init__(self, domain: np.ndarray, level: int, frontier_size: int) -> None:
        self.denominator = 1 << level
        # A dense coordinate table is beneficial only when its axis extent is
        # bounded by the active frontier. Sparse high-depth traversals must not
        # allocate in proportion to the entire implicit grid.
        self.dense = (
            np.full((3, self.denominator + 1, 3), np.nan, dtype=np.float64)
            if self.denominator + 1 <= frontier_size
            else None
        )
        self.cache: tuple[dict[int, tuple[float, float, float]], ...] = ({}, {}, {})
        self.ends = tuple(
            (Fraction(float(domain[0, axis])), Fraction(float(domain[1, axis])))
            for axis in range(3)
        )

    def _coordinate(self, index: int, axis: int) -> tuple[float, float, float]:
        lower, upper = self.ends[axis]
        if index <= 0:
            exact = lower
        elif index >= self.denominator:
            exact = upper
        else:
            position = Fraction(index, self.denominator)
            exact = (1 - position) * lower + position * upper
        nearest = float(exact)
        rounded = Fraction(nearest)
        outward_lower = (
            float(np.nextafter(nearest, -np.inf)) if rounded > exact else nearest
        )
        outward_upper = (
            float(np.nextafter(nearest, np.inf)) if rounded < exact else nearest
        )
        return outward_lower, outward_upper, nearest

    def values(self, indices: np.ndarray, kind: int) -> np.ndarray:
        result = np.empty(indices.shape, dtype=np.float64)
        for axis in range(3):
            axis_indices = indices[:, axis]
            if self.dense is not None:
                table = self.dense[axis]
                missing = np.isnan(table[axis_indices, 0])
                if np.any(missing):
                    for index_ in np.unique(axis_indices[missing]):
                        index = int(index_)
                        table[index] = self._coordinate(index, axis)
                result[:, axis] = table[axis_indices, kind]
                continue
            unique, inverse = np.unique(axis_indices, return_inverse=True)
            values = np.empty(len(unique), dtype=np.float64)
            cache = self.cache[axis]
            for row, index_ in enumerate(unique):
                index = int(index_)
                value = cache.get(index)
                if value is None:
                    value = self._coordinate(index, axis)
                    cache[index] = value
                values[row] = value[kind]
            result[:, axis] = values[inverse]
        return result


@overload
def discover_leaves(
    bounds: _Bounds, domain: np.ndarray, target_level: int, *, sink: None = None
) -> PackedLeaves: ...


@overload
def discover_leaves(
    bounds: _Bounds,
    domain: np.ndarray,
    target_level: int,
    *,
    sink: _LeafSink,
) -> tuple[int, int]: ...


def discover_leaves(
    bounds: _Bounds,
    domain: np.ndarray,
    target_level: int,
    *,
    sink: _LeafSink | None = None,
) -> PackedLeaves | tuple[int, int]:
    """Classify leaves; sink mode streams columns and returns deepest/status."""
    domain = np.asarray(domain, dtype=np.float64)
    leaves = _Columns(
        ((), (3,), (2, 3), (), (), (3,), (3,), ()),
        (
            np.int64,
            np.int64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.int64,
        ),
    )
    surfaces = _Columns(((),), (np.int64,))
    frontier = np.zeros((1, 3), dtype=np.int64)
    level = 0
    deepest = 0
    status = 0

    def emit_rows(*columns: np.ndarray, surface: np.ndarray) -> None:
        if sink is not None:
            sink(
                columns[0],
                columns[1],
                columns[2],
                columns[3],
                columns[4],
                columns[5],
                columns[6],
                columns[7],
                surface,
            )
        else:
            surfaces.append(leaves.size + np.flatnonzero(surface))
            leaves.append(*columns)

    representability = int(AdaptiveImplicitBoxIssue.REPRESENTABILITY)

    while len(frontier):
        deepest = level
        if level == 63 and sink is None:
            leaves.promote_index_to_objects()
        next_dtype = object if level >= 62 else np.int64
        next_frontier = _Columns(((3,),), (next_dtype,))
        coords = _Coordinates(domain, level, len(frontier))
        middle_coords = (
            _Coordinates(domain, level + 1, len(frontier))
            if level < target_level
            else None
        )
        block_lower = block_upper = np.empty((0, 3), dtype=np.float64)
        block_valid = np.empty((0,), dtype=np.bool_)
        block_value_lower = block_value_upper = np.empty((0,), dtype=np.float64)
        block_gradient_lower = block_gradient_upper = np.empty((0, 3), dtype=np.float64)
        for start in range(0, len(frontier), _BATCH):
            offset = start % _EVALUATION_BLOCK
            if offset == 0:
                # Enclose a whole block in one call; each box's enclosure is
                # independent of its neighbours, so the batches below see the
                # same values as when they were evaluated one at a time.
                block = frontier[start : start + _EVALUATION_BLOCK]
                block_lower = coords.values(block, 0)
                block_upper = coords.values(block + 1, 1)
                block_valid = ~np.any(block_lower >= block_upper, axis=1)
                block_value_lower = np.full(len(block), np.nan)
                block_value_upper = np.full(len(block), np.nan)
                block_gradient_lower = np.full((len(block), 3), np.nan)
                block_gradient_upper = np.full((len(block), 3), np.nan)
                if np.any(block_valid):
                    enclosed = bounds.boxes(
                        block_lower[block_valid], block_upper[block_valid]
                    )
                    block_value_lower[block_valid] = enclosed.value_lower
                    block_value_upper[block_valid] = enclosed.value_upper
                    block_gradient_lower[block_valid] = enclosed.gradient_lower
                    block_gradient_upper[block_valid] = enclosed.gradient_upper
            prefix = frontier[start : start + _BATCH]
            rows_in_block = slice(offset, offset + len(prefix))
            lower = block_lower[rows_in_block]
            upper = block_upper[rows_in_block]
            invalid = ~block_valid[rows_in_block]
            if np.any(invalid):
                p, lo, hi = prefix[invalid], lower[invalid], upper[invalid]
                n = len(p)
                emit_rows(
                    np.full(n, level, dtype=np.int64),
                    p,
                    np.stack((lo, hi), axis=1),
                    np.full(n, -np.inf),
                    np.full(n, np.inf),
                    np.full((n, 3), -np.inf),
                    np.full((n, 3), np.inf),
                    np.full(n, representability, dtype=np.int64),
                    surface=np.zeros(n, dtype=bool),
                )
            valid = ~invalid
            if not np.any(valid):
                continue
            p, lo, hi = prefix[valid], lower[valid], upper[valid]
            value_lower = block_value_lower[rows_in_block][valid]
            value_upper = block_value_upper[rows_in_block][valid]
            gradient_lower = block_gradient_lower[rows_in_block][valid]
            gradient_upper = block_gradient_upper[rows_in_block][valid]
            band = ~((value_upper < 0.0) | (value_lower > 0.0))
            issue = np.zeros(len(p), dtype=np.int64)
            if level < target_level:
                assert middle_coords is not None
                lower_sample = coords.values(p, 2)
                upper_sample = coords.values(p + 1, 2)
                middle_sample = middle_coords.values(2 * p + 1, 2)
                stalled = band & np.any(
                    (middle_sample <= lower_sample) | (middle_sample >= upper_sample),
                    axis=1,
                )
                issue[stalled] = representability
                if np.any(stalled):
                    status |= int(
                        AdaptiveImplicitSurfaceStatus.REPRESENTABILITY_LIMIT_REACHED
                    )
                subdivide = band & ~stalled
                parents = p[subdivide].astype(next_dtype, copy=False)
                children = (2 * parents[:, None, :] + _CHILDREN[None, :, :]).reshape(
                    -1, 3
                )
                next_frontier.append(children)
                emit = ~subdivide
                surface = stalled
            else:
                emit = np.ones(len(p), dtype=bool)
                surface = band
            rows = np.flatnonzero(emit)
            if len(rows):
                n = len(rows)
                emit_rows(
                    np.full(n, level, dtype=np.int64),
                    p[rows],
                    np.stack((lo[rows], hi[rows]), axis=1),
                    value_lower[rows],
                    value_upper[rows],
                    gradient_lower[rows],
                    gradient_upper[rows],
                    issue[rows],
                    surface=surface[rows],
                )
        # The final batch and block are views into the old frontier. Drop every
        # reference before finalizing the next frontier or the cover columns.
        del prefix, block, frontier
        frontier = next_frontier.finish()[0]
        level += 1
    if sink is not None:
        return deepest, status
    columns = leaves.finish()
    surface_rows = surfaces.finish()[0]
    return PackedLeaves(
        columns[0],
        columns[1],
        columns[2],
        columns[3],
        columns[4],
        columns[5],
        columns[6],
        columns[7],
        surface_rows,
        deepest,
        status,
    )
