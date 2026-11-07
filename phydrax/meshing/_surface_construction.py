# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Source-bound arrays produced by native trimmed-surface construction.

Also the boundary construction of finite lattice specimens: implicit faces and
trimmed analytic caps joined along certified creases.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import assert_never, final, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._contracts import CompiledGeometry
from ..geometry._interval_enclosure import prepare_interval_function
from ..geometry.analytic._operations import SharpCSG
from ..geometry.analytic._specimen_envelope import SpecimenEnvelope
from ..geometry.implicit._clipped_discovery import ClippedImplicitSurface
from ..geometry.implicit._curve_discovery import ParametricChartCurve
from ..geometry.implicit._policy import AdaptiveImplicitSurfacePolicy
from ..typing import checked


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
            raise ValueError(
                "Exact curve intervals must project to the construction chain."
            )
        return self.curve_intervals_exact


class LatticeSpecimenEventError(ValueError):
    """A specimen boundary is a certified unresolved event and is not meshed.

    ``discovery`` carries the decided outcome: a face chart event (a crease
    tangent to the envelope, a chart lattice vertex on the crease, binary64
    resolution) with its event boxes, or unresolved volume cells.
    """

    def __init__(self, discovery: ClippedImplicitSurface, /) -> None:
        events = [
            f"{chart.axis}{'-+'[chart.side]}:{chart.outcome}"
            for chart in discovery.charts
            if not chart.resolved
        ]
        super().__init__(
            "The specimen boundary is a certified unresolved event "
            f"({discovery.outcome}; {', '.join(events) or 'volume cells'})."
        )
        self.discovery = discovery


@final
class LatticeSpecimenSurface(StrictModule, NonTrainableState):
    """Closed oriented boundary of a finite specimen ``Intersection(region, envelope)``.

    The boundary is built from one clipped discovery on the envelope's
    parameter box: ``triangle_facets`` ``0`` rows are implicit faces (the
    discovery's open contour), and rows ``k >= 1`` are trimmed caps on the
    envelope face ``facet_names[k]``, tiled by the target-level face cells
    on the region side and cut by the crease segments. Every vertex is the
    envelope map of a parameter point: crease vertices are the face charts'
    roots (on the envelope surface and on the region's zero set within
    ``crease_root_distance``), cap vertices lie on the envelope surface.
    ``crease_segments`` and ``envelope_edge_segments`` are the protected
    feature curves (vertex pairs). The boundary is closed and consistently
    oriented outward (checked on construction).

    Fidelity bounds, in Cartesian length: ``implicit_face_distance`` bounds
    the distance of every implicit-face point to the region's zero set
    (interval Hessian and gradient bounds of the region field on each
    triangle; see ``_implicit_face_distances``);
    ``cap_deviation`` bounds the distance of each flat cap triangle to its
    envelope surface; ``crease_root_distance`` bounds each crease vertex's
    distance to a root of the field along its chart edge.
    """

    discovery: ClippedImplicitSurface
    envelope: SpecimenEnvelope
    vertices: np.ndarray
    parameter_points: np.ndarray
    triangles: np.ndarray
    triangle_facets: np.ndarray
    crease_segments: np.ndarray
    envelope_edge_segments: np.ndarray
    facet_names: tuple[str, ...] = eqx.field(static=True)
    implicit_face_distance: float = eqx.field(static=True)
    cap_deviation: float = eqx.field(static=True)
    crease_root_distance: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        discovery: ClippedImplicitSurface,
        envelope: SpecimenEnvelope,
        /,
        *,
        arrays: dict[str, np.ndarray],
        facet_names: tuple[str, ...],
        implicit_face_distance: float,
        cap_deviation: float,
        crease_root_distance: float,
        source_id: str,
    ) -> None:
        def frozen(name: str, dtype: type, shape: tuple[int, ...]) -> np.ndarray:
            value = np.array(arrays[name], dtype=dtype, copy=True).reshape(shape)
            value.setflags(write=False)
            return value

        self.discovery = discovery
        self.envelope = envelope
        self.vertices = frozen("vertices", np.float64, (-1, 3))
        self.parameter_points = frozen("parameter_points", np.float64, (-1, 3))
        self.triangles = frozen("triangles", np.int64, (-1, 3))
        self.triangle_facets = frozen("triangle_facets", np.int64, (-1,))
        self.crease_segments = frozen("crease_segments", np.int64, (-1, 2))
        self.envelope_edge_segments = frozen("envelope_edge_segments", np.int64, (-1, 2))
        self.facet_names = facet_names
        self.implicit_face_distance = float(implicit_face_distance)
        self.cap_deviation = float(cap_deviation)
        self.crease_root_distance = float(crease_root_distance)
        self.source_id = source_id
        _require_closed_oriented(self.triangles)

    @property
    def implicit_triangle_count(self) -> int:
        return int(np.count_nonzero(self.triangle_facets == 0))

    @property
    def cap_triangle_count(self) -> int:
        return int(np.count_nonzero(self.triangle_facets > 0))

    def enclosed_volume(self) -> float:
        """Volume enclosed by the oriented boundary (divergence theorem)."""
        corners = self.vertices[self.triangles]
        return float(
            np.sum(
                np.einsum(
                    "ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])
                )
            )
            / 6.0
        )


def _require_closed_oriented(triangles: np.ndarray, /) -> None:
    """Every directed edge occurs once and is matched by its reverse."""
    directed = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])
    )
    if np.any(directed[:, 0] == directed[:, 1]):
        raise ValueError("A specimen boundary triangle repeats a vertex.")
    forward = np.unique(directed, axis=0)
    reverse = np.unique(directed[:, ::-1], axis=0)
    if forward.shape[0] != directed.shape[0] or not np.array_equal(forward, reverse):
        raise ValueError("The specimen boundary is not a closed oriented 2-manifold.")


type _VertexKey = tuple[str, tuple[int, ...]]


class _VertexBank:
    """Parameter points keyed by grid corner or root-edge identity."""

    def __init__(self) -> None:
        self.points: list[np.ndarray] = []
        self.index: dict[_VertexKey, int] = {}

    def add(self, key: _VertexKey, point: np.ndarray, /) -> int:
        row = self.index.get(key)
        if row is None:
            row = len(self.points)
            self.index[key] = row
            self.points.append(np.asarray(point, dtype=np.float64))
        return row


def _root_key(edge: np.ndarray, /) -> _VertexKey:
    return ("root", tuple(int(value) for value in np.asarray(edge).reshape(-1)))


def _leaf_corner_keys(chart: ParametricChartCurve, /) -> set[tuple[int, int, int]]:
    """Three-dimensional target-level ids of every chart-leaf corner on the face."""
    level = chart.target_level
    span = 1 << level
    first, second = chart.chart_axes
    keys: set[tuple[int, int, int]] = set()
    for leaf_level, (i, j) in zip(
        chart.leaf_levels.tolist(), chart.leaf_indices.tolist(), strict=True
    ):
        size = 1 << (level - leaf_level)
        for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
            key = [0, 0, 0]
            key[chart.axis] = 0 if chart.side == 0 else span
            key[first], key[second] = a * size, b * size
            keys.add((key[0], key[1], key[2]))
    return keys


def _face_cap(
    chart: ParametricChartCurve,
    bank: _VertexBank,
    grid_point: Callable[[np.ndarray], np.ndarray],
    used: set[tuple[int, int, int]],
    /,
) -> list[tuple[int, int, int]]:
    """Region-side cap triangles of one face, counter-clockwise in its chart axes.

    Each chart leaf inside the region is one cap polygon: its corners and every
    leaf corner of this or an adjacent face lying on its sides (``used``), so
    the cap is conforming within the face and across parameter-box edges. A
    polygon with hanging vertices is fanned from its centre. A crease cell
    (target level) keeps the convex part on the region side of its segment.
    """
    from ._periodic_implicit_volume import _triangulate_convex

    level = chart.target_level
    span = 1 << level
    first, second = chart.chart_axes
    sign_of = {
        tuple(int(value) for value in key): int(sign)
        for key, sign in zip(
            chart.corner_ids.tolist(), chart.corner_signs.tolist(), strict=True
        )
    }
    crease_of = {
        (int(cell[0]), int(cell[1])): (int(pair[0]), int(pair[1]))
        for cell, pair in zip(
            chart.segment_cells.tolist(), chart.segments.tolist(), strict=True
        )
    }

    def corner(i: int, j: int, /) -> tuple[int, int, int]:
        key = [0, 0, 0]
        key[chart.axis] = 0 if chart.side == 0 else span
        key[first], key[second] = i, j
        return (key[0], key[1], key[2])

    def corner_vertex(key: tuple[int, int, int], /) -> int:
        return bank.add(("corner", key), grid_point(np.asarray(key)))

    def plane_of(loop: list[int], /) -> dict[int, np.ndarray]:
        return {
            vertex: np.asarray((bank.points[vertex][first], bank.points[vertex][second]))
            for vertex in loop
        }

    triangles: list[tuple[int, int, int]] = []
    for leaf_level, (i, j), value in zip(
        chart.leaf_levels.tolist(),
        chart.leaf_indices.tolist(),
        chart.leaf_classes.tolist(),
        strict=True,
    ):
        if value > 0:
            continue
        size = 1 << (level - leaf_level)
        a, b = i * size, j * size
        corners = (
            corner(a, b),
            corner(a + size, b),
            corner(a + size, b + size),
            corner(a, b + size),
        )
        if value < 0:
            perimeter = (
                [corner(a + k, b) for k in range(size)]
                + [corner(a + size, b + k) for k in range(size)]
                + [corner(a + size - k, b + size) for k in range(size)]
                + [corner(a, b + size - k) for k in range(size)]
            )
            loop = [corner_vertex(key) for key in perimeter if key in used]
            if len(loop) == 4:
                triangles.extend(_triangulate_convex(loop, plane_of(loop)))
            else:
                center = corner_vertex(corner(a + size // 2, b + size // 2))
                triangles.extend(
                    (center, loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop))
                )
            continue
        crease = crease_of.get((i, j))
        if crease is None:
            signs = {sign_of[key] for key in corners}
            if signs == {1}:
                continue
            if signs != {-1}:
                raise ValueError("An uncut crease cell must have one corner sign.")
            loop = [corner_vertex(key) for key in corners]
        else:
            roots = {
                tuple(sorted((tuple(edge[0]), tuple(edge[1])))): row
                for row, edge in (
                    (row, chart.root_edge_ids[row].tolist()) for row in crease
                )
            }
            loop = []
            for position, key in enumerate(corners):
                following = corners[(position + 1) % 4]
                if sign_of[key] < 0:
                    loop.append(corner_vertex(key))
                root = roots.get(tuple(sorted((key, following))))
                if root is not None:
                    loop.append(
                        bank.add(
                            _root_key(chart.root_edge_ids[root]),
                            chart.root_points[root],
                        )
                    )
        triangles.extend(_triangulate_convex(loop, plane_of(loop)))
    # Counter-clockwise in (first, second) has normal +e_axis for axes 0 and 2
    # and -e_axis for axis 1; the cap's outward normal is -e_axis on side 0.
    positive = chart.axis != 1
    outward_positive = chart.side == 1
    if positive != outward_positive:
        triangles = [(a, c, b) for a, b, c in triangles]
    return triangles


class _HessianEnclosures:
    """Outward interval enclosure of the six Hessian entries of a region field."""

    _PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    def __init__(self, geometry: CompiledGeometry, /) -> None:
        kernel, state = geometry.kernel, geometry.state

        def scalar(point: Array) -> Array:
            return kernel.boundary_field(state, point[None, :])[0]

        def mixed(point: Array, first: Array, second: Array) -> Array:
            def slope(at: Array) -> Array:
                return jax.jvp(scalar, (at,), (first,))[1]

            return jax.jvp(slope, (point,), (second,))[1]

        units = np.eye(3)

        def entries(point: Array) -> Array:
            return jnp.stack(
                [
                    mixed(point, jnp.asarray(units[i]), jnp.asarray(units[j]))
                    for i, j in self._PAIRS
                ]
            )

        self._program = prepare_interval_function(entries, 3)

    def frobenius_upper(self, lower: np.ndarray, upper: np.ndarray, /) -> np.ndarray:
        """Upper bound of the Hessian Frobenius norm (dominating its spectral norm)."""
        low, high = self._program.evaluate(lower, upper)
        magnitude = np.maximum(np.abs(low), np.abs(high))
        squares = np.nextafter(magnitude * magnitude, np.inf)
        weights = np.asarray((1.0, 1.0, 1.0, 2.0, 2.0, 2.0))
        total = np.zeros(squares.shape[0], dtype=np.float64)
        for column in range(6):
            total = np.nextafter(total + weights[column] * squares[:, column], np.inf)
        return np.nextafter(np.sqrt(total), np.inf)


def _implicit_face_distances(
    geometry: CompiledGeometry, triangles: np.ndarray, groups: np.ndarray, /
) -> np.ndarray:
    """Certified distance from each flat implicit-face triangle to the zero set.

    For ``p = sum_j l_j v_j`` in a triangle, ``g(p) - sum_j l_j g(v_j)`` is
    ``-sum_j l_j (g(v_j) - g(p) - grad g(p) . (v_j - p))``, so
    ``|g(p)| <= U = max_j |g(v_j)| + M R^2 / 2`` with ``M`` an interval bound
    of the Hessian norm on a box containing the triangle and ``R`` the radius
    of its smallest enclosing disc, ``R^2 <= e^2 / 3`` for the longest edge
    ``e`` (Jung's theorem); vertex values are enclosed at the vertex points.
    If ``||grad g|| >= m > 0`` on that box grown by ``delta`` and
    ``U / m <= delta``, the gradient flow from ``p`` reaches the zero set
    within ``U / m``. Enclosures are shared by the triangles of one
    ``groups`` label (one discovery cell): the box is their common bounding
    box. ``delta`` starts at their longest edge; a group whose premise fails
    with ``m > 0`` retries with ``delta`` twice its largest ``U / m``, so
    ``delta`` at least doubles per retry and the loop ends once the premise
    holds or ``m`` vanishes on the grown box (an infinite, uncertified bound).
    """
    from ..geometry.implicit._adaptive_discovery import _FieldEnclosures

    points = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if not points.shape[0]:
        return np.empty((0,), dtype=np.float64)
    bounds = _FieldEnclosures(geometry, "interval")
    if not bounds.gradient_rigorous:
        return np.full(points.shape[0], np.inf)
    vertex_low, vertex_high = bounds.point_values(points.reshape(-1, 3))
    vertex_value = np.max(
        np.maximum(np.abs(vertex_low), np.abs(vertex_high)).reshape(-1, 3), axis=1
    )
    edges = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 1],
            points[:, 0] - points[:, 2],
        ),
        axis=1,
    )
    longest = np.nextafter(np.max(np.linalg.norm(edges, axis=2), axis=1), np.inf)
    labels, group = np.unique(np.asarray(groups), return_inverse=True)
    group = group.reshape(-1)
    count = labels.shape[0]
    lower = np.full((count, 3), np.inf)
    upper = np.full((count, 3), -np.inf)
    reach = np.zeros(count)
    np.minimum.at(lower, group, np.min(points, axis=1))
    np.maximum.at(upper, group, np.max(points, axis=1))
    np.maximum.at(reach, group, longest)
    hessian = _HessianEnclosures(geometry).frobenius_upper(lower, upper)[group]
    squared_radius = np.nextafter(
        np.nextafter(longest * longest, np.inf) / np.nextafter(3.0, -np.inf), np.inf
    )
    excess = np.nextafter(
        vertex_value + np.nextafter(0.5 * hessian * squared_radius, np.inf), np.inf
    )
    distance = np.full(points.shape[0], np.inf)
    pending = np.arange(count)
    while pending.size:
        grown = bounds.boxes(
            np.nextafter(lower[pending] - reach[pending, None], -np.inf),
            np.nextafter(upper[pending] + reach[pending, None], np.inf),
        )
        gradient_low = np.asarray(grown.gradient_lower, dtype=np.float64)
        gradient_high = np.asarray(grown.gradient_upper, dtype=np.float64)
        smallest = np.where(
            (gradient_low <= 0.0) & (gradient_high >= 0.0),
            0.0,
            np.minimum(np.abs(gradient_low), np.abs(gradient_high)),
        )
        norm = np.full(count, 0.0)
        norm[pending] = np.nextafter(
            np.sqrt(
                np.maximum(
                    np.nextafter(np.sum(smallest * smallest, axis=1), -np.inf), 0.0
                )
            ),
            -np.inf,
        )
        rows = np.flatnonzero(np.isin(group, pending) & (norm[group] > 0.0))
        bound = np.nextafter(excess[rows] / norm[group[rows]], np.inf)
        certified = bound <= reach[group[rows]]
        distance[rows[certified]] = bound[certified]
        needed = np.zeros(count)
        np.maximum.at(needed, group[rows[~certified]], bound[~certified])
        retry = np.flatnonzero(needed > 0.0)
        distance[np.isin(group, retry)] = np.inf
        reach[retry] = np.nextafter(2.0 * needed[retry], np.inf)
        pending = retry[np.isfinite(reach[retry])]
    return distance


def _map_lipschitz(envelope: SpecimenEnvelope, /) -> float:
    """Upper bound of the envelope map's Lipschitz constant (Frobenius norm)."""
    a, b, c = envelope.scales
    match envelope.kind:
        case "box":
            bound = math.sqrt(a * a + b * b + c * c)
        case "cylinder":
            # |du/ds|, |dv/dt| <= 1 and |du/dt|, |dv/ds| <= 1/sqrt(2).
            bound = math.sqrt(3.0 * a * a + c * c)
        case "sphere":
            # Diagonal entries <= 1, off-diagonal entries <= 5/(6 sqrt(1/3)) < 1.45.
            bound = a * math.sqrt(3.0 + 6.0 * 1.45 * 1.45)
        case _:
            assert_never(envelope.kind)
    return float(np.nextafter(bound * (1.0 + 1.0e-12), np.inf))


def _expand_rows(start: np.ndarray, keys: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Every ``(query, slot)`` pair of the CSR ranges ``start[k] : start[k + 1]``."""
    sizes = start[keys + 1] - start[keys]
    owner = np.repeat(np.arange(keys.shape[0]), sizes)
    offsets = np.arange(owner.shape[0]) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return owner, start[keys][owner] + offsets


def _triangle_quality(points: np.ndarray, triangles: np.ndarray, /) -> np.ndarray:
    """``4 sqrt(3) area / sum of squared edges``: one for equilateral, zero when flat."""
    corners = points[triangles]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    third = corners[:, 2] - corners[:, 1]
    area = 0.5 * np.linalg.norm(np.cross(first, second), axis=1)
    squares = np.sum(first * first + second * second + third * third, axis=1)
    return 4.0 * math.sqrt(3.0) * area / np.where(squares > 0.0, squares, 1.0)


def _collapse_short_edges(
    points: np.ndarray, triangles: np.ndarray, fixed: np.ndarray, ratio: float, /
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse short edges of a closed oriented surface onto existing vertices.

    A candidate moves a free vertex ``v`` onto a neighbour ``u`` when the
    edge is no longer than ``ratio`` times the longest edge of the triangles
    around ``v``. It is admissible when the link condition holds (``u`` and
    ``v`` share exactly the two opposite vertices, so the surface stays a
    closed 2-manifold), every moved triangle keeps its orientation and grows
    no edge beyond that longest edge, and the worst triangle quality around
    ``v`` does not drop. Each round collapses an
    independent set of admissible candidates (disjoint closed stars of their
    ``v``, shortest first), so every round removes at least one vertex and the
    loop ends when no admissible candidate is left. Returns the triangles and
    the indices of the surviving input triangles.
    """
    rows = np.array(triangles, dtype=np.int64, copy=True)
    alive = np.arange(rows.shape[0], dtype=np.int64)
    count = points.shape[0]
    while True:
        current = rows[alive]
        pairs = np.sort(
            np.stack(
                (current.reshape(-1), np.roll(current, -1, axis=1).reshape(-1)), axis=1
            ),
            axis=1,
        )
        edges, edge_of = np.unique(pairs, axis=0, return_inverse=True)
        lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        longest = np.max(lengths[edge_of.reshape(-1, 3)], axis=1)
        reach = np.zeros(count, dtype=np.float64)
        np.maximum.at(reach, current.reshape(-1), np.repeat(longest, 3))
        moved = np.concatenate((edges[:, 0], edges[:, 1]))
        target = np.concatenate((edges[:, 1], edges[:, 0]))
        length = np.concatenate((lengths, lengths))
        candidate = ~fixed[moved] & (length <= ratio * reach[moved])
        moved, target, length = moved[candidate], target[candidate], length[candidate]
        if not moved.size:
            break
        # Neighbour lists (CSR over sorted directed edges) and triangle stars.
        directed = np.concatenate((edges, edges[:, ::-1]))
        directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
        neighbour_start = np.searchsorted(directed[:, 0], np.arange(count + 1))
        corner_owner = np.repeat(np.arange(current.shape[0]), 3)
        order = np.argsort(current.reshape(-1), kind="stable")
        star_vertex = current.reshape(-1)[order]
        star_triangle = corner_owner[order]
        star_start = np.searchsorted(star_vertex, np.arange(count + 1))

        # Link condition: exactly two common neighbours.
        owner_v, slot_v = _expand_rows(neighbour_start, moved)
        owner_u, slot_u = _expand_rows(neighbour_start, target)
        keys = np.concatenate(
            (
                owner_v * count + directed[slot_v, 1],
                owner_u * count + directed[slot_u, 1],
            )
        )
        shared_keys, multiplicity = np.unique(keys, return_counts=True)
        common = np.bincount(
            shared_keys[multiplicity == 2] // count, minlength=moved.shape[0]
        )
        admissible = common == 2
        # Orientation and worst quality over the star of the moved vertex.
        owner_t, slot_t = _expand_rows(star_start, moved)
        star = star_triangle[slot_t]
        before = current[star]
        after = np.where(before == moved[owner_t, None], target[owner_t, None], before)
        removed = np.any(before == target[owner_t, None], axis=1)
        old_normal = np.cross(
            points[before[:, 1]] - points[before[:, 0]],
            points[before[:, 2]] - points[before[:, 0]],
        )
        new_normal = np.cross(
            points[after[:, 1]] - points[after[:, 0]],
            points[after[:, 2]] - points[after[:, 0]],
        )
        flipped = ~removed & (np.einsum("ij,ij->i", old_normal, new_normal) <= 0.0)
        after_corners = points[after]
        after_longest = np.max(
            np.linalg.norm(after_corners - np.roll(after_corners, 1, axis=1), axis=2),
            axis=1,
        )
        # No moved triangle may outgrow the longest edge already around v.
        flipped |= ~removed & (after_longest > reach[moved[owner_t]])
        old_quality = _triangle_quality(points, before)
        new_quality = np.where(removed, np.inf, _triangle_quality(points, after))
        worst_old = np.full(moved.shape[0], np.inf)
        worst_new = np.full(moved.shape[0], np.inf)
        np.minimum.at(worst_old, owner_t, old_quality)
        np.minimum.at(worst_new, owner_t, new_quality)
        bad = np.zeros(moved.shape[0], dtype=np.bool_)
        np.logical_or.at(bad, owner_t, flipped)
        admissible &= ~bad & (worst_new >= worst_old)
        chosen = np.flatnonzero(admissible)
        if not chosen.size:
            break
        # Independent set: a candidate wins every vertex of its star.
        rank = np.empty(moved.shape[0], dtype=np.int64)
        rank[np.lexsort((target, moved, length))] = np.arange(moved.shape[0])
        in_chosen = admissible[owner_t]
        claim_owner = np.repeat(owner_t[in_chosen], 3)
        claim_vertex = before[in_chosen].reshape(-1)
        best = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(best, claim_vertex, rank[claim_owner])
        lost = np.zeros(moved.shape[0], dtype=np.bool_)
        np.logical_or.at(lost, claim_owner, best[claim_vertex] != rank[claim_owner])
        winners = chosen[~lost[chosen]]
        mapping = np.arange(count, dtype=np.int64)
        mapping[moved[winners]] = target[winners]
        rows[alive] = mapping[current]
        collapsed = rows[alive]
        degenerate = (
            (collapsed[:, 0] == collapsed[:, 1])
            | (collapsed[:, 1] == collapsed[:, 2])
            | (collapsed[:, 2] == collapsed[:, 0])
        )
        alive = alive[~degenerate]
    return rows[alive], alive


@checked
def construct_lattice_specimen_surface(
    specimen: SharpCSG,
    /,
    *,
    policy: AdaptiveImplicitSurfacePolicy = AdaptiveImplicitSurfacePolicy(),
    short_edge_ratio: float = 0.25,
    source_id: str,
) -> LatticeSpecimenSurface:
    """Build the certified closed boundary of a finite lattice specimen.

    ``specimen`` is ``Intersection(region, envelope)`` with a smooth reliable
    three-dimensional ``region`` (for example a ``TrigonometricLevelSet``)
    and a ``Box``, full ``Cylinder`` or ``Sphere`` envelope. The region field
    composed with the envelope map is discovered on the parameter box in
    clipped mode (``discover_clipped_implicit_surface``): the face charts
    certify the creases, the volume discovery gives the implicit faces with
    exactly those creases as boundary, and the caps are the face cells on
    the region side trimmed by the creases. A certified unresolved event
    raises ``LatticeSpecimenEventError``; nothing is meshed.

    Polygonization roots near lattice corners leave implicit-face edges far
    shorter than their neighbours, which pin flat tetrahedra in a fixed
    boundary fill. Each implicit-face edge no longer than ``short_edge_ratio``
    times the longest edge of the triangles around its free end is collapsed
    onto its other end (``_collapse_short_edges``): crease, rim and cap
    vertices never move, no vertex is created, and the implicit-face fidelity
    bound is certified on the collapsed triangles. ``0`` keeps the discovery
    contour unchanged.
    """
    from ..geometry.analytic._specimen_envelope import (
        specimen_parts,
        SpecimenParameterField,
    )
    from ..geometry.implicit._adaptive_packed_polygon import _grid_points
    from ..geometry.implicit._clipped_discovery import (
        discover_clipped_implicit_surface,
    )

    region, envelope = specimen_parts(specimen)
    field = SpecimenParameterField(region, envelope, feature_id=source_id).compile()
    domain = envelope.parameter_domain
    discovery = discover_clipped_implicit_surface(
        field, domain=domain, policy=policy, source_id=source_id
    )
    ratio = float(short_edge_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
        raise ValueError("short_edge_ratio must lie in [0, 1).")
    scaffold = discovery.scaffold
    if not discovery.resolved or scaffold is None:
        raise LatticeSpecimenEventError(discovery)
    level = discovery.target_level
    bank = _VertexBank()
    scaffold_edges = np.asarray(scaffold.root_edge_ids, dtype=np.int64)
    contour_vertices = [
        bank.add(_root_key(scaffold_edges[row]), point)
        for row, point in zip(
            discovery.vertex_root_rows.tolist(), discovery.points, strict=True
        )
    ]
    rows = np.asarray(contour_vertices, dtype=np.int64)
    triangles: list[np.ndarray] = [rows[np.asarray(discovery.faces, dtype=np.int64)]]
    facets: list[np.ndarray] = [np.zeros(discovery.faces.shape[0], dtype=np.int64)]
    names: list[str] = ["lattice"]
    cap_faces: list[np.ndarray] = []

    def grid_point(key: np.ndarray, /) -> np.ndarray:
        return _grid_points(np.asarray(key).reshape(1, 3), level, domain)[0]

    crease_pairs: set[tuple[int, int]] = set()
    used: set[tuple[int, int, int]] = set()
    for chart in discovery.charts:
        used |= _leaf_corner_keys(chart)
    for chart in discovery.charts:
        cap = _face_cap(chart, bank, grid_point, used)
        name = envelope.face_name(chart.axis, chart.side)
        if name not in names:
            names.append(name)
        if cap:
            triangles.append(np.asarray(cap, dtype=np.int64))
            facets.append(np.full(len(cap), names.index(name), dtype=np.int64))
            cap_faces.append(np.tile((chart.axis, chart.side), (len(cap), 1)))
        for a, b in chart.segments.tolist():
            first = bank.index[_root_key(chart.root_edge_ids[a])]
            second = bank.index[_root_key(chart.root_edge_ids[b])]
            crease_pairs.add((min(first, second), max(first, second)))
    triangle_rows = np.concatenate(triangles)
    facet_rows = np.concatenate(facets)
    cap_axes = np.concatenate(cap_faces) if cap_faces else np.empty((0, 2), np.int64)
    crease_segments = np.asarray(sorted(crease_pairs), dtype=np.int64).reshape(-1, 2)
    parameter_points = np.asarray(bank.points, dtype=np.float64).reshape(-1, 3)
    vertices = envelope.map_host(parameter_points)
    implicit = facet_rows == 0
    cell_rows = np.full(facet_rows.shape[0], -1, dtype=np.int64)
    cell_rows[implicit] = np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)
    if ratio > 0.0:
        fixed = np.zeros(vertices.shape[0], dtype=np.bool_)
        fixed[triangle_rows[~implicit].reshape(-1)] = True
        fixed[crease_segments.reshape(-1)] = True
        triangle_rows, kept = _collapse_short_edges(vertices, triangle_rows, fixed, ratio)
        facet_rows, cell_rows = facet_rows[kept], cell_rows[kept]
        survivors = np.unique(triangle_rows)
        remap = np.full(vertices.shape[0], -1, dtype=np.int64)
        remap[survivors] = np.arange(survivors.shape[0], dtype=np.int64)
        vertices, parameter_points = vertices[survivors], parameter_points[survivors]
        triangle_rows, crease_segments = remap[triangle_rows], remap[crease_segments]

    # Envelope edges: cap triangle edges on a sharp parameter-box edge.
    cap_rows = triangle_rows[facet_rows > 0]
    cap_edges = np.sort(
        np.concatenate((cap_rows[:, (0, 1)], cap_rows[:, (1, 2)], cap_rows[:, (2, 0)])),
        axis=1,
    )
    on_edge = np.zeros(cap_edges.shape[0], dtype=np.bool_)
    for axis, side_a, side_b in envelope.physical_edges:
        others = [other for other in range(3) if other != axis]
        fixed_edge = np.ones(parameter_points.shape[0], dtype=np.bool_)
        for other, side in zip(others, (side_a, side_b), strict=True):
            fixed_edge &= parameter_points[:, other] == domain[side, other]
        on_edge |= fixed_edge[cap_edges[:, 0]] & fixed_edge[cap_edges[:, 1]]
    edge_segments = np.unique(cap_edges[on_edge], axis=0).reshape(-1, 2)
    cap_triangles = vertices[cap_rows]
    cap_bound = (
        float(np.max(envelope.surface_deviation_bound(cap_triangles, cap_axes)))
        if cap_axes.shape[0]
        else 0.0
    )
    lipschitz = _map_lipschitz(envelope)
    implicit = facet_rows == 0
    implicit_bounds = _implicit_face_distances(
        region.compile(), vertices[triangle_rows[implicit]], cell_rows[implicit]
    )
    implicit_distance = float(np.max(implicit_bounds, initial=0.0))
    crease_errors = np.concatenate([chart.root_errors for chart in discovery.charts])
    crease_distance = float(
        np.nextafter(lipschitz * float(np.max(crease_errors, initial=0.0)), np.inf)
    )
    return LatticeSpecimenSurface(
        discovery,
        envelope,
        arrays={
            "vertices": vertices,
            "parameter_points": parameter_points,
            "triangles": triangle_rows,
            "triangle_facets": facet_rows,
            "crease_segments": crease_segments,
            "envelope_edge_segments": edge_segments,
        },
        facet_names=tuple(names),
        implicit_face_distance=implicit_distance,
        cap_deviation=cap_bound,
        crease_root_distance=crease_distance,
        source_id=source_id,
    )


__all__ = [
    "construct_lattice_specimen_surface",
    "LatticeSpecimenEventError",
    "LatticeSpecimenSurface",
    "SurfaceConstruction",
    "SurfaceSourceCellAtlas",
    "SourceParameterInterval",
    "curve_edge_parameter_intervals",
    "curve_vertex_chains",
    "source_cell_ancestry",
]
