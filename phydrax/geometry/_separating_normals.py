#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact candidate separating-plane normals with rigorous binary64 shadows.

Contact proofs try the planes spanned by cell corners in one canonical order:
coordinate axes, cross products of every pair of corner-difference directions
(planar: perpendiculars of the directions), then corner-to-corner directions.
Each plane is proved once, by its exactly normalized representative.

Forming every exact normal is the dominant cost of a pair proof, while the
exact proof only accepts a handful. A `NormalTable` therefore holds binary64
shadows of all candidates together with a componentwise envelope ``M`` with
``|exact - shadow| <= 64 u M`` (``u`` the unit roundoff), formed from the
correctly rounded binary64 corners. A screen may discard a candidate only when
its shadow proves, beyond ``1e-9 M``, that the exact proof of the plane fails.
The proof outcome of a plane depends only on the plane, not on the scale of
its normal, so walking the surviving candidates in canonical order and proving
each plane at its first surviving occurrence accepts exactly the plane the
unscreened walk accepts first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations, islice

import numpy as np
from numpy.typing import NDArray

from ..discretization._coordinate_enclosure import (
    bernstein_coefficients,
    bernstein_degree,
)
from ..discretization._exact_algebraic_geometry import ExactScalar
from ._embedding_cells import EmbeddingCell
from ._exact_polyhedral_geometry import _sign


# Far above the accumulated binary64 rounding error (tens of unit roundoffs of
# the envelope); the absolute term covers underflow.
_SHADOW_RELATIVE = 1e-9
_SHADOW_ABSOLUTE = 1e-300

_AXIS = 0
_CROSS = 1
_DIRECTION = 2
_PERPENDICULAR = 3

type NormalScreen = Callable[
    [NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]
]


def shadow_bound(
    magnitude: NDArray[np.float64], absolute: NDArray[np.float64], /
) -> NDArray[np.float64]:
    """Bound the rounding of ``shadow @ controls`` from their envelopes."""
    return _SHADOW_RELATIVE * (magnitude @ absolute) + _SHADOW_ABSOLUTE


@dataclass(frozen=True, slots=True)
class CellShadow:
    """Binary64 shadows of one cell's exact rational corners and image controls.

    ``exact_controls`` (rows: ambient components) are the exact Bernstein
    controls of the polynomial proof coordinates, all elevated to one common
    degree, and ``controls`` their correctly rounded shadows. The controls of
    ``n . x`` at that degree are ``n @ exact_controls``; they are convex
    combinations of its least-degree controls, so their range lies inside the
    least-degree range every outward Bernstein bound encloses.
    """

    corners: NDArray[np.float64]
    corner_magnitude: NDArray[np.float64]
    exact_controls: tuple[tuple[ExactScalar, ...], ...] | None
    controls: NDArray[np.float64] | None
    control_magnitude: NDArray[np.float64] | None


def _image_controls(cell: EmbeddingCell, /) -> tuple[tuple[ExactScalar, ...], ...] | None:
    components = tuple(
        component for component in cell.proof_coordinates if isinstance(component, dict)
    )
    if not components or len(components) != len(cell.proof_coordinates):
        return None
    domain = cell.proof_domain
    try:
        degrees = tuple(
            bernstein_degree(component, domain, cell.dimension)
            for component in components
        )
        degree = tuple(max(values) for values in zip(*degrees, strict=True))
        return tuple(
            bernstein_coefficients(component, domain, cell.dimension, degree=degree)
            for component in components
        )
    except (ArithmeticError, TypeError, ValueError):
        return None


def _rational_rows(
    rows: tuple[tuple[ExactScalar, ...], ...], /
) -> NDArray[np.float64] | None:
    if not rows or not all(
        isinstance(value, Fraction | int) for row in rows for value in row
    ):
        return None
    try:
        shadows = np.asarray(
            tuple(tuple(float(value) for value in row) for row in rows),
            dtype=np.float64,
        )
    except OverflowError:
        return None
    if shadows.ndim != 2 or shadows.shape[1] == 0 or not np.all(np.isfinite(shadows)):
        return None
    return shadows


def cell_shadow(cell: EmbeddingCell, /) -> CellShadow | None:
    """Return correctly rounded shadows of a cell with rational corners."""
    corners = _rational_rows(cell.corner_images)
    if corners is None:
        return None
    exact_controls = _image_controls(cell)
    controls = None if exact_controls is None else _rational_rows(exact_controls)
    return CellShadow(
        corners,
        np.abs(corners),
        exact_controls,
        controls,
        None if controls is None else np.abs(controls),
    )


def _exact_projection(
    normal: tuple[ExactScalar, ...], controls: tuple[tuple[ExactScalar, ...], ...], /
) -> tuple[ExactScalar, ...]:
    return tuple(
        sum(
            (
                value * row[index]
                for value, row in zip(normal, controls, strict=True)
                if _sign(value)
            ),
            Fraction(0),
        )
        for index in range(len(controls[0]))
    )


@dataclass(frozen=True, slots=True)
class _TableLayout:
    """Index structure of the candidate normals of two cells' corner counts.

    Corners are stacked first cell, then second cell. Directions are the
    stacked corner pairs ``(edge_start, edge_stop)``: first-cell combinations,
    then second-cell combinations. Row ``i`` is built by ``kinds[i]`` from
    ``left[i]``/``right[i]``: an axis, the cross product of two directions,
    the planar perpendicular of one direction, or the difference of two
    stacked corners (first-cell corner to second-cell corner).
    """

    kinds: NDArray[np.int8]
    left: NDArray[np.int64]
    right: NDArray[np.int64]
    edge_start: NDArray[np.int64]
    edge_stop: NDArray[np.int64]
    cross_left: NDArray[np.int64]
    cross_right: NDArray[np.int64]
    corner_left: NDArray[np.int64]
    corner_right: NDArray[np.int64]


def _pairs(count: int, /) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    start, stop = np.triu_indices(count, k=1)
    return start.astype(np.int64), stop.astype(np.int64)


@cache
def _table_layout(first_count: int, second_count: int, ambient: int, /) -> _TableLayout:
    first_start, first_stop = _pairs(first_count)
    second_start, second_stop = _pairs(second_count)
    edge_start = np.concatenate((first_start, second_start + first_count))
    edge_stop = np.concatenate((first_stop, second_stop + first_count))
    axis = np.arange(ambient, dtype=np.int64)
    empty = np.zeros(0, dtype=np.int64)
    if ambient == 2:
        direction = np.arange(edge_start.shape[0], dtype=np.int64)
        cross_left = cross_right = corner_left = corner_right = empty
        kinds = np.concatenate(
            (
                np.full(ambient, _AXIS, dtype=np.int8),
                np.full(direction.shape, _PERPENDICULAR, dtype=np.int8),
            )
        )
        left = right = np.concatenate((axis, direction))
    else:
        cross_left, cross_right = _pairs(edge_start.shape[0])
        corner_left = np.repeat(np.arange(first_count, dtype=np.int64), second_count)
        corner_right = first_count + np.tile(
            np.arange(second_count, dtype=np.int64), first_count
        )
        kinds = np.concatenate(
            (
                np.full(ambient, _AXIS, dtype=np.int8),
                np.full(cross_left.shape, _CROSS, dtype=np.int8),
                np.full(corner_left.shape, _DIRECTION, dtype=np.int8),
            )
        )
        left = np.concatenate((axis, cross_left, corner_left))
        right = np.concatenate((axis, cross_right, corner_right))
    for value in (
        kinds,
        left,
        right,
        edge_start,
        edge_stop,
        cross_left,
        cross_right,
        corner_left,
        corner_right,
    ):
        value.flags.writeable = False
    return _TableLayout(
        kinds,
        left,
        right,
        edge_start,
        edge_stop,
        cross_left,
        cross_right,
        corner_left,
        corner_right,
    )


@dataclass(frozen=True, slots=True)
class NormalTable:
    """Shadows of one cell pair's candidate normals in canonical order."""

    first: EmbeddingCell
    second: EmbeddingCell
    layout: _TableLayout
    values: NDArray[np.float64]
    magnitude: NDArray[np.float64]

    def _corner(self, index: int, /) -> tuple[ExactScalar, ...]:
        count = len(self.first.corner_images)
        if index < count:
            return self.first.corner_images[index]
        return self.second.corner_images[index - count]

    def _edge(self, index: int, /) -> tuple[ExactScalar, ...]:
        start = self._corner(int(self.layout.edge_start[index]))
        stop = self._corner(int(self.layout.edge_stop[index]))
        return tuple(right - left for left, right in zip(start, stop, strict=True))

    def exact(self, index: int, /) -> tuple[ExactScalar, ...]:
        kind = int(self.layout.kinds[index])
        left, right = int(self.layout.left[index]), int(self.layout.right[index])
        if kind == _AXIS:
            ambient = self.values.shape[1]
            return tuple(Fraction(int(column == left)) for column in range(ambient))
        if kind == _CROSS:
            a, b = self._edge(left), self._edge(right)
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
        if kind == _PERPENDICULAR:
            direction = self._edge(left)
            return -direction[1], direction[0]
        start, stop = self._corner(left), self._corner(right)
        return tuple(b - a for a, b in zip(start, stop, strict=True))


def normal_table(
    first: EmbeddingCell,
    second: EmbeddingCell,
    first_shadow: CellShadow | None,
    second_shadow: CellShadow | None,
    /,
) -> NormalTable | None:
    """Tabulate the canonical candidate normals of two rational-corner cells."""
    if first_shadow is None or second_shadow is None:
        return None
    first_points, second_points = first_shadow.corners, second_shadow.corners
    ambient = first_points.shape[1]
    if ambient not in (2, 3) or second_points.shape[1] != ambient:
        return None
    layout = _table_layout(first_points.shape[0], second_points.shape[0], ambient)
    points = np.concatenate((first_points, second_points))
    sizes = np.concatenate(
        (first_shadow.corner_magnitude, second_shadow.corner_magnitude)
    )
    directions = points[layout.edge_stop] - points[layout.edge_start]
    direction_sizes = sizes[layout.edge_stop] + sizes[layout.edge_start]
    count = layout.kinds.shape[0]
    values = np.empty((count, ambient), dtype=np.float64)
    magnitude = np.empty((count, ambient), dtype=np.float64)
    values[:ambient] = np.eye(ambient, dtype=np.float64)
    magnitude[:ambient] = values[:ambient]
    if ambient == 2:
        values[ambient:, 0] = -directions[:, 1]
        values[ambient:, 1] = directions[:, 0]
        magnitude[ambient:] = direction_sizes[:, ::-1]
    else:
        stop = ambient + layout.cross_left.shape[0]
        a, b = directions[layout.cross_left], directions[layout.cross_right]
        a_size = direction_sizes[layout.cross_left]
        b_size = direction_sizes[layout.cross_right]
        for row, (first_axis, second_axis) in enumerate(((1, 2), (2, 0), (0, 1))):
            values[ambient:stop, row] = (
                a[:, first_axis] * b[:, second_axis]
                - a[:, second_axis] * b[:, first_axis]
            )
            magnitude[ambient:stop, row] = (
                a_size[:, first_axis] * b_size[:, second_axis]
                + a_size[:, second_axis] * b_size[:, first_axis]
            )
        values[stop:] = points[layout.corner_right] - points[layout.corner_left]
        magnitude[stop:] = sizes[layout.corner_right] + sizes[layout.corner_left]
    return NormalTable(first, second, layout, values, magnitude)


def candidate_normals(
    first: EmbeddingCell, second: EmbeddingCell, /
) -> Iterator[tuple[ExactScalar, ...]]:
    """Yield the exact candidate normals of two cells in canonical order."""
    ambient = len(first.corner_images[0])
    for axis in range(ambient):
        yield tuple(Fraction(int(index == axis)) for index in range(ambient))
    if ambient not in (2, 3):
        return
    first_points = first.corner_images
    second_points = second.corner_images
    edge_directions = tuple(
        tuple(right[axis] - left[axis] for axis in range(ambient))
        for points in (first_points, second_points)
        for left, right in combinations(points, 2)
    )
    if ambient == 2:
        for direction in edge_directions:
            yield -direction[1], direction[0]
        return
    for left, right in combinations(edge_directions, 2):
        normal = (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )
        if any(_sign(value) for value in normal):
            yield normal
    for left in first_points:
        for right in second_points:
            direction = tuple(right[axis] - left[axis] for axis in range(ambient))
            if any(_sign(value) for value in direction):
                yield direction


def unique_projective_normals(
    normals: Iterable[tuple[ExactScalar, ...]],
    /,
    seen: set[tuple[ExactScalar, ...]] | None = None,
) -> Iterator[tuple[ExactScalar, ...]]:
    """Yield one exactly normalized representative of each candidate plane.

    Exact scalars hash consistently with equality, so the seen set gives the
    same representatives as a pairwise scan in linear rather than quadratic time.
    """
    planes: set[tuple[ExactScalar, ...]] = set() if seen is None else seen
    for normal in normals:
        leading = next((value for value in normal if _sign(value)), None)
        if leading is None:
            continue
        normalized = tuple(value / leading for value in normal)
        if normalized in planes:
            continue
        planes.add(normalized)
        yield normalized


def _rational_shadow(normal: tuple[ExactScalar, ...], /) -> tuple[float, ...] | None:
    if not all(isinstance(value, Fraction | int) for value in normal):
        return None
    try:
        row = tuple(float(value) for value in normal)
    except OverflowError:
        return None
    return row if all(np.isfinite(row)) else None


def _screened_exact(
    normals: Iterator[tuple[ExactScalar, ...]], screen: NormalScreen, /
) -> Iterator[tuple[ExactScalar, ...]]:
    # Rational normals convert with relative error ``u``, so ``|shadow|`` is a
    # valid envelope; any other normal is admitted unscreened.
    while chunk := tuple(islice(normals, 256)):
        rows = [_rational_shadow(normal) for normal in chunk]
        convertible = [index for index, row in enumerate(rows) if row is not None]
        admitted = [True] * len(chunk)
        if convertible:
            values = np.asarray([rows[index] for index in convertible], dtype=np.float64)
            for index, keep in zip(
                convertible, screen(values, np.abs(values)).tolist(), strict=True
            ):
                admitted[index] = bool(keep)
        for normal, keep in zip(chunk, admitted, strict=True):
            if keep:
                yield normal


def screened_candidate_planes(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shadows: dict[int, CellShadow | None],
    screen: NormalScreen,
    /,
    seen: set[tuple[ExactScalar, ...]] | None = None,
) -> Iterator[tuple[ExactScalar, ...]]:
    """Yield normalized candidate planes not excluded by ``screen``, in order.

    ``screen(values, magnitude)`` returns, per shadow row, ``False`` only when
    the exact proof of that row's plane provably fails; ``shadows`` caches
    `cell_shadow` by cell identity for one proof.
    """
    table = normal_table(
        first, second, cached_shadow(first, shadows), cached_shadow(second, shadows)
    )
    if table is None:
        yield from unique_projective_normals(
            _screened_exact(candidate_normals(first, second), screen), seen
        )
        return
    admitted = screen(table.values, table.magnitude)
    finite = np.all(np.isfinite(table.values), axis=1) & np.all(
        np.isfinite(table.magnitude), axis=1
    )
    rows = np.flatnonzero(admitted | ~finite).tolist()
    yield from unique_projective_normals((table.exact(row) for row in rows), seen)


def cached_shadow(
    cell: EmbeddingCell, shadows: dict[int, CellShadow | None], /
) -> CellShadow | None:
    """Return the per-proof shadow of one cell, keyed by cell identity."""
    key = cell.global_id
    if key not in shadows:
        shadows[key] = cell_shadow(cell)
    return shadows[key]


def orthogonal_screen(directions: tuple[tuple[ExactScalar, ...], ...], /) -> NormalScreen:
    """Screen planes that cannot contain every exact ``direction``."""
    rows = [_rational_shadow(direction) for direction in directions]
    if any(row is None for row in rows) or not rows:
        return admit_all
    shadow = np.asarray(rows, dtype=np.float64).T

    def screen(
        values: NDArray[np.float64], magnitude: NDArray[np.float64], /
    ) -> NDArray[np.bool_]:
        products = values @ shadow
        bound = shadow_bound(magnitude, np.abs(shadow))
        undecided = ~(np.isfinite(products) & np.isfinite(bound))
        return np.all(undecided | (np.abs(products) <= bound), axis=1)

    return screen


def admit_all(
    values: NDArray[np.float64], magnitude: NDArray[np.float64], /
) -> NDArray[np.bool_]:
    """Screen nothing."""
    del magnitude
    return np.ones(values.shape[0], dtype=np.bool_)


def _strict_separation_margins(
    first: CellShadow | None, second: CellShadow | None, /
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None:
    """Return rigorous per-plane gaps between two cells' image projections.

    For each shadow row the margin is the larger of the two strict orderings'
    gaps ``min(second) - max(first)`` and ``min(first) - max(second)``, each
    widened by the rounding bound; a nonpositive margin proves that the exact
    common-degree control ranges overlap. Undecided rows are NaN.
    """
    if (
        first is None
        or second is None
        or first.controls is None
        or second.controls is None
        or first.control_magnitude is None
        or second.control_magnitude is None
    ):
        return None
    first_controls, first_absolute = first.controls, first.control_magnitude
    second_controls, second_absolute = second.controls, second.control_magnitude

    def margins(
        values: NDArray[np.float64], magnitude: NDArray[np.float64], /
    ) -> NDArray[np.float64]:
        first_values = values @ first_controls
        first_bound = shadow_bound(magnitude, first_absolute)
        second_values = values @ second_controls
        second_bound = shadow_bound(magnitude, second_absolute)
        with np.errstate(invalid="ignore", over="ignore"):
            first_max_low = np.max(first_values - first_bound, axis=1)
            first_min_high = np.min(first_values + first_bound, axis=1)
            second_max_low = np.max(second_values - second_bound, axis=1)
            second_min_high = np.min(second_values + second_bound, axis=1)
            result = np.maximum(
                second_min_high - first_max_low, first_min_high - second_max_low
            )
        finite = (
            np.all(np.isfinite(first_values), axis=1)
            & np.all(np.isfinite(first_bound), axis=1)
            & np.all(np.isfinite(second_values), axis=1)
            & np.all(np.isfinite(second_bound), axis=1)
            & np.isfinite(result)
        )
        return np.where(finite, result, np.nan)

    return margins


def _common_degree_ranges_separate(
    normal: tuple[ExactScalar, ...],
    first: CellShadow | None,
    second: CellShadow | None,
    /,
) -> bool:
    """Return whether exact common-degree ranges of ``n . x`` may be ordered."""
    if (
        first is None
        or second is None
        or first.exact_controls is None
        or second.exact_controls is None
        or len(normal) != len(first.exact_controls)
        or len(normal) != len(second.exact_controls)
    ):
        return True
    first_values = _exact_projection(normal, first.exact_controls)
    second_values = _exact_projection(normal, second.exact_controls)
    if not all(isinstance(value, Fraction) for value in (*first_values, *second_values)):
        return True
    return max(first_values) < min(second_values) or max(second_values) < min(
        first_values
    )


def strict_separation_planes(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shadows: dict[int, CellShadow | None],
    /,
) -> Iterator[tuple[ExactScalar, ...]]:
    """Yield normalized planes whose outward image projections may separate.

    An outward enclosure of ``n . x`` contains its least-degree control range,
    which contains the common-degree control range, so strictly ordered outward
    enclosures need strictly ordered exact common-degree ranges. Planes whose
    shadow margin proves overlap are skipped, then planes whose exact
    common-degree ranges overlap. Separation is an existence question, so
    surviving planes come widest normalized margin first.
    """
    first_shadow = cached_shadow(first, shadows)
    second_shadow = cached_shadow(second, shadows)
    margins = _strict_separation_margins(first_shadow, second_shadow)
    table = normal_table(first, second, first_shadow, second_shadow)
    if margins is None:
        planes = screened_candidate_planes(first, second, shadows, admit_all)
    elif table is None:

        def screen(
            values: NDArray[np.float64], magnitude: NDArray[np.float64], /
        ) -> NDArray[np.bool_]:
            gap = margins(values, magnitude)
            return np.isnan(gap) | (gap > 0.0)

        planes = screened_candidate_planes(first, second, shadows, screen)
    else:
        gap = margins(table.values, table.magnitude)
        undecided = np.isnan(gap) | ~np.all(np.isfinite(table.magnitude), axis=1)
        rows = np.flatnonzero(undecided | (gap > 0.0))
        # Shadows dominated by cancellation (zero or near-parallel normals)
        # carry no direction, so they are tried last.
        length = np.linalg.norm(table.values[rows], axis=1)
        envelope = np.linalg.norm(table.magnitude[rows], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = gap[rows] / length
        score = np.where(
            undecided[rows] | ~np.isfinite(score) | (length <= 1e-6 * envelope),
            -np.inf,
            score,
        )
        order = rows[np.argsort(-score, kind="stable")]
        planes = unique_projective_normals(table.exact(int(row)) for row in order)
    for normal in planes:
        if _common_degree_ranges_separate(normal, first_shadow, second_shadow):
            yield normal


__all__ = [
    "admit_all",
    "candidate_normals",
    "cached_shadow",
    "cell_shadow",
    "CellShadow",
    "normal_table",
    "NormalScreen",
    "NormalTable",
    "orthogonal_screen",
    "screened_candidate_planes",
    "shadow_bound",
    "strict_separation_planes",
    "unique_projective_normals",
]
