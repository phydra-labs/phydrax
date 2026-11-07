# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Source-bound arrays produced by native trimmed-surface construction."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from fractions import Fraction
from typing import final, NamedTuple, Protocol

import equinox as eqx
import numpy as np
from jax import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState


class _RootChart(NamedTuple):
    lower_u: Fraction
    upper_u: Fraction
    lower_v: Fraction
    upper_v: Fraction
    cell_id: int
    source_piece: tuple[int, ...]
    corners: tuple[tuple[Fraction, Fraction], ...]
    orientation: int


class SurfaceSourceCellAtlas(Protocol):
    """Validated source-piece chart partition consumed by native construction."""

    root_cell_ids: tuple[int, ...]
    root_source_cell_ids: tuple[tuple[int, ...], ...]
    root_parameter_bounds: tuple[tuple[tuple[Fraction, Fraction], ...], ...]

    @property
    def root_patches(self) -> Array | np.ndarray: ...
    @property
    def root_parameters(self) -> Array | np.ndarray: ...
    def require_current(self) -> None: ...


def _contains_point(chart: _RootChart, point: tuple[Fraction, Fraction], /) -> bool:
    """Test exact represented-chart inclusion in one convex source-piece cell."""
    u, v = point
    if not (chart.lower_u <= u <= chart.upper_u and chart.lower_v <= v <= chart.upper_v):
        return False
    for first, last in zip(
        chart.corners, (*chart.corners[1:], chart.corners[0]), strict=True
    ):
        cross = (last[0] - first[0]) * (v - first[1]) - (last[1] - first[1]) * (
            u - first[0]
        )
        if chart.orientation * cross < 0:
            return False
    return True


def source_cell_ancestry(
    atlas: SurfaceSourceCellAtlas,
    triangle_patches: np.ndarray,
    triangle_parameters: np.ndarray,
    exact_triangle_parameters: Sequence[
        tuple[
            tuple[Fraction, Fraction],
            tuple[Fraction, Fraction],
            tuple[Fraction, Fraction],
        ]
    ],
    /,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Bind each triangle to one source-piece cell using exact rational UVs.

    The exact chart bank must lower row-for-row to the numerical construction
    parameters. Root-cell containment uses the retained exact source bounds and
    exact triangles; neither physical coordinates nor binary64 reconstructions
    determine the source-cell identity.
    """
    atlas.require_current()
    patches = np.asarray(triangle_patches)
    parameters = np.asarray(triangle_parameters, dtype=np.float64)
    if (
        patches.ndim != 1
        or patches.dtype.kind not in "iu"
        or np.any(patches < 0)
        or parameters.shape != (patches.size, 3, 2)
        or not np.all(np.isfinite(parameters))
        or len(exact_triangle_parameters) != patches.size
    ):
        raise ValueError(
            "SurfaceConstruction exact and numeric triangle parameters must align as three finite source-chart corners per nonnegative integer patch."
        )

    root_ids = tuple(atlas.root_cell_ids)
    source_pieces = tuple(atlas.root_source_cell_ids)
    root_bounds = tuple(atlas.root_parameter_bounds)
    root_patches = np.asarray(atlas.root_patches)
    root_parameters = np.asarray(atlas.root_parameters, dtype=np.float64)
    if (
        not root_ids
        or len(set(root_ids)) != len(root_ids)
        or any(cell < 0 for cell in root_ids)
        or len(source_pieces) != len(root_ids)
        or len(root_bounds) != len(root_ids)
        or root_patches.shape != (len(root_ids),)
        or root_patches.dtype.kind not in "iu"
        or root_parameters.shape != (len(root_ids), 4, 2)
        or not np.all(np.isfinite(root_parameters))
    ):
        raise ValueError(
            "SurfaceSourceRootAtlas root IDs, source-piece ancestry, and chart corners must align; got malformed root table."
        )

    by_patch: dict[int, list[_RootChart]] = {}
    for row, (cell_id, piece, bounds) in enumerate(
        zip(root_ids, source_pieces, root_bounds, strict=True)
    ):
        if (
            len(bounds) != 4
            or not piece
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in piece
            )
        ):
            raise ValueError(
                "SurfaceSourceRootAtlas source-piece ancestry must retain a nonempty integer key and four exact chart corners; got malformed root."
            )
        exact_corners = tuple(
            (Fraction(point[0]), Fraction(point[1])) for point in bounds
        )
        expected = np.asarray(
            [[float(u), float(v)] for u, v in exact_corners], dtype=np.float64
        )
        if not np.array_equal(root_parameters[row], expected):
            raise ValueError(
                "SurfaceSourceRootAtlas root parameters must realize exact parameter bounds; got stale chart corners."
            )
        corners = exact_corners
        turns = tuple(
            (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0])
            for first, second, third in zip(
                corners,
                (*corners[1:], corners[0]),
                (*corners[2:], *corners[:2]),
                strict=True,
            )
        )
        turn_signs = {1 if value > 0 else -1 for value in turns if value != 0}
        if len(turn_signs) != 1 or any(value == 0 for value in turns):
            raise ValueError(
                "SurfaceSourceRootAtlas represented root chart must be a strictly convex cell; got degenerate or nonconvex bounds."
            )
        orientation = turn_signs.pop()
        lower_u = min(point[0] for point in corners)
        upper_u = max(point[0] for point in corners)
        lower_v = min(point[1] for point in corners)
        upper_v = max(point[1] for point in corners)
        if lower_u == upper_u or lower_v == upper_v:
            raise ValueError(
                "SurfaceSourceRootAtlas represented root chart must have positive extent; got collapsed cell."
            )
        chart = _RootChart(
            lower_u,
            upper_u,
            lower_v,
            upper_v,
            cell_id,
            tuple(piece),
            corners,
            orientation,
        )
        by_patch.setdefault(int(root_patches[row]), []).append(chart)

    lower_us: dict[int, list[Fraction]] = {}
    for patch, charts in by_patch.items():
        charts.sort(key=lambda chart: (chart.lower_u, chart.upper_u, chart.cell_id))
        lower_us[patch] = [chart.lower_u for chart in charts]

    ancestry = np.empty((patches.size,), dtype=np.int64)
    pieces: list[tuple[int, ...]] = []
    for patch_value, triangle, exact_triangle in zip(
        patches.tolist(),
        parameters.tolist(),
        exact_triangle_parameters,
        strict=True,
    ):
        patch = patch_value
        patch_charts = by_patch.get(patch, [])
        patch_lower_us = lower_us.get(patch, [])
        if not patch_charts:
            raise ValueError(
                "SurfaceConstruction triangle patch has no source-piece root chart."
            )
        if len(exact_triangle) != 3 or any(
            len(point) != 2 or any(not isinstance(value, Fraction) for value in point)
            for point in exact_triangle
        ):
            raise TypeError(
                "SurfaceConstruction exact triangle parameters must contain three 2D Fraction points."
            )
        points = exact_triangle
        exact_realization = np.asarray(
            [[float(value) for value in point] for point in points],
            dtype=np.float64,
        )
        if not np.array_equal(exact_realization, np.asarray(triangle, dtype=np.float64)):
            raise ValueError(
                "SurfaceConstruction exact triangle parameters must lower exactly to the numeric parameter bank."
            )
        lower_u = min(point[0] for point in points)
        upper_u = max(point[0] for point in points)
        lower_v = min(point[1] for point in points)
        upper_v = max(point[1] for point in points)
        matches = [
            chart
            for chart in patch_charts[: bisect_right(patch_lower_us, upper_u)]
            if chart.upper_u >= lower_u
            and chart.lower_v <= upper_v
            and chart.upper_v >= lower_v
            and all(_contains_point(chart, point) for point in points)
        ]
        if len(matches) != 1:
            raise ValueError(
                "SurfaceConstruction.triangle_parameters must lie wholly in exactly one source-piece cell; got a cross-cell or ambiguous triangle."
            )
        ancestry[len(pieces)] = matches[0].cell_id
        pieces.append(matches[0].source_piece)
    return ancestry, tuple(pieces)


def curve_vertex_chains(
    curve_parameters: tuple[tuple[int, np.ndarray], ...],
    curve_edges: np.ndarray,
    curve_edge_curves: np.ndarray,
    /,
) -> tuple[tuple[int, np.ndarray], ...]:
    """Recover each canonical source curve's single ordered vertex chain.

    Every curve is represented once in source-parameter order. Its edge rows
    form one connected chain, and no vertex is repeated except the shared first
    and last row of a closed source curve.
    """
    edges = np.asarray(curve_edges)
    owners = np.asarray(curve_edge_curves)
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or edges.dtype.kind not in "iu"
        or owners.shape != (edges.shape[0],)
        or owners.dtype.kind not in "iu"
    ):
        raise ValueError(
            "SurfaceConstruction.curve_edges and curve_edge_curves must be aligned integer segments; got invalid curve incidence."
        )
    identifiers = tuple(curve for curve, _ in curve_parameters)
    if len(set(identifiers)) != len(identifiers) or any(
        isinstance(curve, bool) or not isinstance(curve, int) or curve < 0
        for curve in identifiers
    ):
        raise ValueError(
            "SurfaceConstruction.curve_parameters must retain each canonical nonnegative source-curve ID once; got duplicate identity."
        )
    if set(owners.tolist()) != set(identifiers):
        raise ValueError(
            "SurfaceConstruction.curve_edge_curves must cover exactly the discretized source curves; got missing or foreign curve rows."
        )
    chains: list[tuple[int, np.ndarray]] = []
    for curve, values in curve_parameters:
        parameters = np.asarray(values, dtype=np.float64)
        rows = edges[owners == curve]
        if (
            parameters.ndim != 1
            or parameters.size < 2
            or not np.all(np.isfinite(parameters))
            or np.any(parameters[1:] <= parameters[:-1])
            or rows.shape[0] != parameters.size - 1
            or np.any(rows < 0)
            or np.any(rows[:, 0] == rows[:, 1])
            or np.unique(np.sort(rows, axis=1), axis=0).shape[0] != rows.shape[0]
            or np.any(rows[:-1, 1] != rows[1:, 0])
        ):
            raise ValueError(
                "SurfaceConstruction.curve_parameters and curve_edges must form one strictly ordered source-curve chain; got disconnected or duplicated discretization."
            )
        chain = np.concatenate((rows[:1, 0], rows[:, 1]))
        open_chain = chain[:-1] if chain[0] == chain[-1] else chain
        if np.unique(open_chain).size != open_chain.size:
            raise ValueError(
                "SurfaceConstruction.curve_edges must retain distinct source-parameter vertices, except for one authored closed-curve endpoint; got a repeated junction."
            )
        chains.append((curve, chain))
    return tuple(chains)


type SourceParameterInterval = tuple[Fraction, Fraction]


def curve_edge_parameter_intervals(
    curve_parameters: tuple[tuple[int, np.ndarray], ...],
    curve_edge_curves: np.ndarray,
    /,
) -> tuple[SourceParameterInterval, ...]:
    """Retain exact represented source-parameter intervals for physical edges."""
    identifiers = tuple(curve for curve, _ in curve_parameters)
    owners = np.asarray(curve_edge_curves)
    if (
        len(set(identifiers)) != len(identifiers)
        or any(
            isinstance(curve, bool) or not isinstance(curve, int) or curve < 0
            for curve in identifiers
        )
        or owners.ndim != 1
        or owners.dtype.kind not in "iu"
        or set(owners.tolist()) != set(identifiers)
    ):
        raise ValueError(
            "SurfaceConstruction.curve_edge_curves must retain each canonical source curve exactly once per edge."
        )
    parameter_rows: dict[int, np.ndarray] = {}
    for curve, values in curve_parameters:
        parameters = np.asarray(values, dtype=np.float64)
        if (
            parameters.ndim != 1
            or parameters.size < 2
            or not np.all(np.isfinite(parameters))
            or np.any(parameters[1:] <= parameters[:-1])
        ):
            raise ValueError(
                "SurfaceConstruction.curve_parameters must be finite and strictly increasing for each source curve."
            )
        parameter_rows[curve] = parameters
    segment_rows = dict.fromkeys(parameter_rows, 0)
    intervals: list[SourceParameterInterval] = []
    for curve in owners.tolist():
        index = segment_rows[curve]
        values = parameter_rows[curve]
        if index + 1 >= values.size:
            raise ValueError(
                "SurfaceConstruction.curve_edge_curves must retain each source parameter interval exactly once."
            )
        intervals.append(
            (Fraction(float(values[index])), Fraction(float(values[index + 1])))
        )
        segment_rows[curve] = index + 1
    if any(
        segment_rows[curve] != values.size - 1 for curve, values in parameter_rows.items()
    ):
        raise ValueError(
            "SurfaceConstruction.curve_edge_curves must retain every source-parameter interval exactly once."
        )
    return tuple(intervals)


@final
class SurfaceConstruction(StrictModule, NonTrainableState):
    """Host construction arrays retaining source chart and physical-edge ancestry.

    ``triangle_parameters`` are original patch-UV corners, not extraction-cell
    coordinates. ``triangle_source_cells`` and ``triangle_source_pieces`` bind
    each triangle to one validated source-piece root. ``curve_edges`` contain
    only the once-discretized, canonical physical source curves; source-piece
    partition lines are never encoded as curve edges.
    """

    vertices: np.ndarray
    vertex_source_dimensions: np.ndarray
    vertex_source_indices: np.ndarray
    vertex_parameters: np.ndarray
    chart_restriction_vertices: np.ndarray
    chart_restriction_edges: np.ndarray
    chart_restriction_endpoint_parameters: np.ndarray
    chart_restriction_parameters: np.ndarray
    triangles: np.ndarray
    triangle_patches: np.ndarray
    triangle_charts: np.ndarray
    triangle_parameters: np.ndarray
    triangle_source_cells: np.ndarray
    triangle_source_pieces: tuple[tuple[int, ...], ...]
    triangle_deviations: np.ndarray
    triangle_deviation_bounds: np.ndarray
    triangle_normal_bounds: np.ndarray
    curve_edges: np.ndarray
    curve_edge_curves: np.ndarray
    curve_edge_deviations: np.ndarray
    curve_edge_deviation_bounds: np.ndarray
    unresolved: np.ndarray
    boundary_occurrences: np.ndarray
    chart_triangulations: tuple[tuple[int, np.ndarray], ...]
    curve_parameters: tuple[tuple[int, np.ndarray], ...]
    curve_intervals_exact: tuple[SourceParameterInterval, ...] = eqx.field(static=True)
    minimum_angle: float = eqx.field(static=True)

    def curve_vertex_chains(self) -> tuple[tuple[int, np.ndarray], ...]:
        """Return the shared source-curve vertex chains retained by this record."""
        return curve_vertex_chains(
            self.curve_parameters, self.curve_edges, self.curve_edge_curves
        )

    def curve_edge_parameter_intervals(self) -> tuple[SourceParameterInterval, ...]:
        """Return exact represented source-parameter intervals per curve edge."""
        self.curve_vertex_chains()
        projected = curve_edge_parameter_intervals(
            self.curve_parameters, self.curve_edge_curves
        )
        if len(projected) != len(self.curve_intervals_exact) or any(
            float(first) != float(projected_first)
            or float(last) != float(projected_last)
            or first >= last
            for (first, last), (projected_first, projected_last) in zip(
                self.curve_intervals_exact, projected, strict=True
            )
        ):
            raise ValueError("Exact curve intervals must project to the construction chain.")
        return self.curve_intervals_exact


__all__ = [
    "SurfaceConstruction",
    "SurfaceSourceCellAtlas",
    "SourceParameterInterval",
    "curve_edge_parameter_intervals",
    "curve_vertex_chains",
    "source_cell_ancestry",
]
