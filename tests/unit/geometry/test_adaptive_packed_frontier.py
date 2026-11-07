#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""The packed traversal agrees with the original scalar BFS, not its wiring."""

import itertools
import mmap
import weakref
from typing import Protocol

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry._contracts import GeometrySource
from phydrax.geometry.implicit import _adaptive_packed_frontier as packed_frontier
from phydrax.geometry.implicit._adaptive_discovery import (
    _BoxBounds,
    _cell_box,
    _FieldEnclosures,
    _grid_coordinate,
    _grid_coordinate_fraction,
    _outward_coordinate,
)
from phydrax.geometry.implicit._adaptive_packed_frontier import (
    _Coordinates,
    discover_leaves,
    PackedPolygonLeaves,
)
from phydrax.geometry.implicit._policy import (
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitSurfaceStatus,
)


class _Bounds(Protocol):
    def boxes(self, lower: np.ndarray, upper: np.ndarray) -> _BoxBounds: ...


_ScalarLeaf = tuple[
    int, tuple[int, ...], np.ndarray, float, float, np.ndarray, np.ndarray, int
]


def _scalar_traversal(
    bounds: _Bounds, domain: np.ndarray, target: int
) -> tuple[list[_ScalarLeaf], np.ndarray, int, int]:
    frontier = [(0, 0, 0)]
    leaves, surfaces = [], []
    level = deepest = status = 0
    issue = int(AdaptiveImplicitBoxIssue.REPRESENTABILITY)
    while frontier:
        deepest = level
        children = []
        for start in range(0, len(frontier), 64):
            valid = []
            for prefix in frontier[start : start + 64]:
                lower, upper = _cell_box(prefix, level, domain)
                if np.any(lower >= upper):
                    leaves.append(
                        (
                            level,
                            prefix,
                            np.stack((lower, upper)),
                            -np.inf,
                            np.inf,
                            np.full(3, -np.inf),
                            np.full(3, np.inf),
                            issue,
                        )
                    )
                else:
                    valid.append((prefix, lower, upper))
            if not valid:
                continue
            evidence = bounds.boxes(
                np.stack([x[1] for x in valid]), np.stack([x[2] for x in valid])
            )
            for row, (prefix, lower, upper) in enumerate(valid):
                lo, hi = (
                    float(evidence.value_lower[row]),
                    float(evidence.value_upper[row]),
                )
                cell = (
                    level,
                    prefix,
                    np.stack((lower, upper)),
                    lo,
                    hi,
                    evidence.gradient_lower[row],
                    evidence.gradient_upper[row],
                    0,
                )
                if hi < 0.0 or lo > 0.0:
                    leaves.append(cell)
                elif level < target:
                    low = [
                        _grid_coordinate(prefix[a], level, *domain[:, a])
                        for a in range(3)
                    ]
                    high = [
                        _grid_coordinate(prefix[a] + 1, level, *domain[:, a])
                        for a in range(3)
                    ]
                    middle = [
                        _grid_coordinate(2 * prefix[a] + 1, level + 1, *domain[:, a])
                        for a in range(3)
                    ]
                    if any(m <= l or m >= h for l, m, h in zip(low, middle, high)):
                        surfaces.append(len(leaves))
                        leaves.append((*cell[:-1], issue))
                        status |= int(
                            AdaptiveImplicitSurfaceStatus.REPRESENTABILITY_LIMIT_REACHED
                        )
                    else:
                        children.extend(
                            tuple(2 * prefix[a] + bits[a] for a in range(3))
                            for bits in itertools.product((0, 1), repeat=3)
                        )
                else:
                    surfaces.append(len(leaves))
                    leaves.append(cell)
        frontier = children
        level += 1
    return leaves, np.asarray(surfaces, dtype=np.int64), deepest, status


@pytest.mark.parametrize(
    ("geometry", "domain", "target"),
    [
        (
            phx.geometry.Sphere((0.13, -0.1, 0.05), 0.6),
            np.asarray(((-1.03, -0.91, -1.07), (1.01, 1.13, 0.97))),
            3,
        ),
        (
            phx.geometry.Torus((0.01, -0.02, 0.03), 0.3, 0.9),
            np.asarray(((-1.13, -1.07, -0.81), (1.17, 1.03, 0.79))),
            3,
        ),
        (
            phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0),
            np.asarray(
                ((1.0, -1e-16, -1e-16), (np.nextafter(1.0, np.inf), 1e-16, 1e-16))
            ),
            3,
        ),
    ],
)
def test_packed_traversal_matches_independent_scalar_bfs(
    geometry: GeometrySource, domain: np.ndarray, target: int
) -> None:
    bounds = _FieldEnclosures(geometry.compile(), "interval")
    expected, surface_rows, deepest, status = _scalar_traversal(bounds, domain, target)
    actual = discover_leaves(bounds, domain, target)
    for column, name in enumerate(
        (
            "level",
            "index",
            "boxes",
            "value_lower",
            "value_upper",
            "gradient_lower",
            "gradient_upper",
            "issue",
        )
    ):
        array = getattr(actual, name)
        np.testing.assert_array_equal(
            array, np.asarray([row[column] for row in expected])
        )
        assert array.flags.c_contiguous and not array.flags.writeable
    np.testing.assert_array_equal(actual.surface_rows, surface_rows)
    assert (actual.deepest_level, actual.status) == (deepest, status)
    batches = []
    replay = discover_leaves(
        bounds,
        domain,
        target,
        sink=lambda *columns: batches.append(tuple(x.copy() for x in columns)),
    )
    assert replay == (deepest, status)
    for column, name in enumerate(
        (
            "level",
            "index",
            "boxes",
            "value_lower",
            "value_upper",
            "gradient_lower",
            "gradient_upper",
            "issue",
        )
    ):
        np.testing.assert_array_equal(
            np.concatenate([batch[column] for batch in batches]), getattr(actual, name)
        )
    mask = np.concatenate([batch[-1] for batch in batches])
    np.testing.assert_array_equal(np.flatnonzero(mask), actual.surface_rows)
    if domain[0, 0] == 1.0:
        assert status & int(AdaptiveImplicitSurfaceStatus.REPRESENTABILITY_LIMIT_REACHED)


def test_traversal_retains_indices_beyond_int64_without_depth_cap() -> None:
    class OriginBounds:
        def boxes(self, lower: np.ndarray, upper: np.ndarray) -> _BoxBounds:
            contains = np.all((lower <= 0.0) & (upper >= 0.0), axis=1)
            values = np.where(contains, -1.0, 1.0)
            return _BoxBounds(
                value_lower=values,
                value_upper=np.ones(len(lower)),
                gradient_lower=np.zeros_like(lower),
                gradient_upper=np.zeros_like(lower),
            )

    domain = np.asarray(((-1.0, -1.0, -1.0), (0.5, 0.5, 0.5)))
    bounds = OriginBounds()
    expected, surface_rows, deepest, status = _scalar_traversal(bounds, domain, 65)
    actual = discover_leaves(bounds, domain, 65)
    assert actual.index.dtype == object
    assert (
        max(int(index) for row in actual.index for index in row) > np.iinfo(np.int64).max
    )
    for column, name in enumerate(
        (
            "level",
            "index",
            "boxes",
            "value_lower",
            "value_upper",
            "gradient_lower",
            "gradient_upper",
            "issue",
        )
    ):
        np.testing.assert_array_equal(
            getattr(actual, name),
            np.asarray(
                [row[column] for row in expected],
                dtype=object if name == "index" else None,
            ),
        )
    np.testing.assert_array_equal(actual.surface_rows, surface_rows)
    assert (actual.deepest_level, actual.status) == (deepest, status)


@pytest.mark.parametrize("level", (3, 10, 65))
def test_coordinate_cache_matches_exact_scalar_rounding(level: int) -> None:
    domain = np.asarray(((-1.03, -0.91, -1e-200), (1.01, 1.13, 1e-200)))
    denominator = 1 << level
    indices = np.asarray(
        [
            (0, 1, denominator // 3),
            (denominator // 2, denominator - 1, denominator),
            (denominator // 3, 1, denominator // 2),
        ],
        dtype=object if level > 62 else np.int64,
    )
    # Force both representations when feasible; extreme depth stays sparse.
    sizes = (1, denominator + 1) if level < 20 else (1,)
    for frontier_size in sizes:
        coordinates = _Coordinates(domain, level, frontier_size)
        for kind in range(3):
            actual = coordinates.values(indices, kind)
            expected = np.empty_like(actual)
            for row in range(len(indices)):
                for axis in range(3):
                    exact = _grid_coordinate_fraction(
                        int(indices[row, axis]), level, *domain[:, axis]
                    )
                    expected[row, axis] = (
                        float(exact)
                        if kind == 2
                        else _outward_coordinate(exact, lower=kind == 0)
                    )
            np.testing.assert_array_equal(actual, expected)


def test_consuming_auxiliary_releases_columns_but_not_cover() -> None:
    geometry = phx.geometry.Sphere((0.13, -0.1, 0.05), 0.6).compile()
    domain = np.asarray(((-1.03, -0.91, -1.07), (1.01, 1.13, 0.97)))
    leaves = discover_leaves(_FieldEnclosures(geometry, "interval"), domain, 2)
    names = ("level", "index", "issue", "surface_rows")
    auxiliary_refs = [weakref.ref(getattr(leaves, name)) for name in names]
    cover_names = (
        "boxes",
        "value_lower",
        "value_upper",
        "gradient_lower",
        "gradient_upper",
    )
    cover = [getattr(leaves, name) for name in cover_names]
    deepest, status = leaves.deepest_level, leaves.status
    polygon_leaves = PackedPolygonLeaves.from_leaves(leaves)
    del leaves
    polygon_leaves._release_auxiliary()
    assert all(ref() is None for ref in auxiliary_refs)
    for name in names:
        with pytest.raises(RuntimeError, match="consumed"):
            getattr(polygon_leaves, name)
    with pytest.raises(RuntimeError, match="consumed"):
        polygon_leaves._release_auxiliary()
    for name, array in zip(cover_names, cover):
        assert getattr(polygon_leaves, name) is array
        assert not array.flags.writeable
    assert (polygon_leaves.deepest_level, polygon_leaves.status) == (deepest, status)


def test_previous_frontier_is_released_before_next_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BandBounds:
        def boxes(self, lower: np.ndarray, upper: np.ndarray) -> _BoxBounds:
            return _BoxBounds(
                value_lower=np.full(len(lower), -1.0),
                value_upper=np.ones(len(lower)),
                gradient_lower=np.zeros_like(lower),
                gradient_upper=np.zeros_like(lower),
            )

    original_finish = packed_frontier._Columns.finish
    previous = None
    checked = 0

    def checked_finish(columns: packed_frontier._Columns) -> tuple[np.ndarray, ...]:
        nonlocal previous, checked
        if columns.shapes == ((3,),):
            if previous is not None:
                assert previous() is None
                checked += 1
            result = original_finish(columns)
            previous = weakref.ref(result[0])
            return result
        return original_finish(columns)

    monkeypatch.setattr(packed_frontier._Columns, "finish", checked_finish)
    domain = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    leaves = discover_leaves(BandBounds(), domain, 3)
    assert checked == 3
    assert len(leaves.surface_rows) == 8**3


def test_numeric_blocks_unmap_on_finish_and_object_blocks_keep_python_ownership() -> None:
    size = packed_frontier._BLOCK + 3
    source = np.arange(size * 3, dtype=np.int64).reshape(size, 3)
    columns = packed_frontier._Columns(((3,),), (np.int64,))
    columns.append(source)
    mappings = [block.base for block in columns.blocks[0]]
    assert all(isinstance(mapping, mmap.mmap) for mapping in mappings)
    (result,) = columns.finish()
    assert all(isinstance(mapping, mmap.mmap) and mapping.closed for mapping in mappings)
    np.testing.assert_array_equal(result, source)
    assert result.flags.owndata and not result.flags.writeable

    objects = packed_frontier._Columns(((3,),), (object,))
    large = np.asarray([[1 << 80, 1 << 81, 1 << 82]], dtype=object)
    objects.append(large)
    assert objects.blocks[0][0].flags.owndata
    (result_objects,) = objects.finish()
    np.testing.assert_array_equal(result_objects, large)
