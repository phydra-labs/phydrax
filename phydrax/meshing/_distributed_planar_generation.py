#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local planar constrained Delaunay generation on a fixed tile complex.

The authored planar source and request are replicated. A lattice of
axis-aligned tiles, whose span is a declared multiple of the resolved region
size, its seams (interior lattice lines inside the region), every source/seam
crossing and every initial constraint point are fixed from the source, request
and schedule before any target work. Each process refines only the tiles it
owns with the native constrained Delaunay mesher. A Steiner point that
refinement inserts on a seam is sent only to the owner of the other incident
tile, and synchronous rounds repeat until no seam changes. Placement selects
which process computes a tile; it never changes a coordinate, a constraint or
an identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from time import monotonic
from typing import final, Self

import equinox as eqx
import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from numpy.typing import NDArray

import phydrax.geometry as triangulation_owner

from .._execution_runtime import owner_execution_mesh, process_owner_devices
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import exact_orient2d, MeshcoreError, MeshcoreStatus, NativeExecutionScope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._collective_failure import collective_failure_barrier
from ._contracts import MeshingLimits, SurfaceMeshingSpec
from ._device_generation import DeviceGenerationLayout
from ._failures import MeshingFailure, MeshingFailureCategory
from ._measurements import NativeExecutionRecord
from ._native_preparation import native_preparation_ledger
from ..geometry._domain_coverage import PiecewiseLinearDomain
from ._trace import MeshingStageKind
from .providers import _native_sources as sources_owner
from .providers._native_planar_domain import (
    _capacity_failure,
    _field_sizes,
    PreparedPlanarDomain,
    source_arrays,
    split_constraints,
)
from .providers._native_route_checks import check_deadline


# Seam packet rows: seam piece row followed by the two binary64 coordinates.
_PACKET_WIDTH = 3
_STAGE = "distributed-planar-initial"
_REMOTE_REFUSAL = (
    "An owner rejected owner-local planar construction; no generated tile was published."
)


@final
class PlanarTileSchedule(StrictModule, NonTrainableState):
    """Declared tiling and seam agreement of owner-local planar generation.

    ``tile_span_sizes`` is the tile edge in units of the resolved region size.
    It is a request parameter, never derived from a process count.
    ``maximum_seam_rounds`` bounds the synchronous seam-agreement rounds; an
    unconverged seam refuses rather than publishing nonconforming tiles.
    """

    tile_span_sizes: int = eqx.field(static=True)
    maximum_seam_rounds: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, *, tile_span_sizes: int = 2, maximum_seam_rounds: int = 16) -> None:
        for name, value in (
            ("tile_span_sizes", tile_span_sizes),
            ("maximum_seam_rounds", maximum_seam_rounds),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"PlanarTileSchedule.{name} must be a positive integer; got {value!r}"
                )
        self.tile_span_sizes = tile_span_sizes
        self.maximum_seam_rounds = maximum_seam_rounds
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "planar-tile-schedule",
                "tile_span_sizes": tile_span_sizes,
                "maximum_seam_rounds": maximum_seam_rounds,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: PlanarTileSchedule, /) -> Self:
        """Revalidate a stored schedule through its constructor."""
        rebuilt = cls(
            tile_span_sizes=value.tile_span_sizes,
            maximum_seam_rounds=value.maximum_seam_rounds,
        )
        if rebuilt.schedule_id != value.schedule_id:
            raise ValueError(
                "PlanarTileSchedule.schedule_id must match its declared controls; got changed stored content"
            )
        return rebuilt


def _morton(columns: np.ndarray, rows: np.ndarray, /) -> np.ndarray:
    """Interleave nonnegative 31-bit lattice coordinates."""
    code = np.zeros(columns.shape, dtype=np.uint64)
    for bit in range(31):
        code |= ((columns.astype(np.uint64) >> np.uint64(bit)) & np.uint64(1)) << np.uint64(
            2 * bit
        )
        code |= ((rows.astype(np.uint64) >> np.uint64(bit)) & np.uint64(1)) << np.uint64(
            2 * bit + 1
        )
    return code


def _inside(points: np.ndarray, a: np.ndarray, b: np.ndarray, /) -> np.ndarray:
    """Exact-orientation crossing parity of points against closed loop edges."""
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.bool_)
    signs = exact_orient2d(a[None, :, :], b[None, :, :], points[:, None, :])
    rising = (a[None, :, 1] <= points[:, None, 1]) & (points[:, None, 1] < b[None, :, 1])
    falling = (b[None, :, 1] <= points[:, None, 1]) & (points[:, None, 1] < a[None, :, 1])
    return (
        np.count_nonzero((rising & (signs > 0)) | (falling & (signs < 0)), axis=1) % 2
        == 1
    )


class _SkeletonPoints:
    """Exact-coordinate deduplicated constraint points with source provenance.

    Provenance precedence is source vertex, then source edge, then interior.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[float, float], int] = {}
        self.coordinates: list[tuple[float, float]] = []
        self.dimensions: list[int] = []
        self.indices: list[int] = []
        self.parameters: list[float] = []

    def add(self, point: np.ndarray, dimension: int, index: int, parameter: float) -> int:
        key = (float(point[0]), float(point[1]))
        row = self.rows.get(key)
        if row is None:
            row = len(self.coordinates)
            self.rows[key] = row
            self.coordinates.append(key)
            self.dimensions.append(dimension)
            self.indices.append(index)
            self.parameters.append(parameter)
        elif dimension < self.dimensions[row]:
            self.dimensions[row] = dimension
            self.indices[row] = index
            self.parameters[row] = parameter
        return row


@final
class PlanarTileComplex(StrictModule, NonTrainableState):
    """Source/request/schedule-determined tiles, seams and constraint points.

    ``points`` hold every shared constraint point: source vertices, source/seam
    crossings, lattice corners and the canonical splits of source and seam
    pieces (equal pieces at the canonical lattice spacing for an admitted
    uniform median, size-field splits otherwise). ``point_dimensions``/
    ``point_indices``/``point_parameters`` retain source-vertex (0),
    source-edge (1) or region (2) provenance and the binary64 parameter along a
    source edge. ``segments`` are the split constraint pieces;
    ``segment_sources`` name their source edge (or -1),
    ``segment_seams`` their seam piece (or -1) and ``segment_tiles`` the one
    (boundary) or two (seam) incident tile rows. Seam piece ``s`` lies on the
    lattice line ``seam_axes[s]`` (0: constant x) at ``seam_coordinates[s]``
    between ``seam_tiles[s]`` (lower side first). Tiles are listed in Morton
    order of their lattice coordinates; ``tile_weights`` are their region
    areas. Nothing here depends on a process count.
    """

    source: sources_owner.NativePlanarSource
    specification: SurfaceMeshingSpec
    schedule: PlanarTileSchedule
    origin: NDArray[np.float64]
    tiles: NDArray[np.int64]
    tile_weights: NDArray[np.float64]
    points: NDArray[np.float64]
    point_dimensions: NDArray[np.int64]
    point_indices: NDArray[np.int64]
    point_parameters: NDArray[np.float64]
    point_tiles: NDArray[np.int64]
    segments: NDArray[np.int64]
    segment_sources: NDArray[np.int64]
    segment_seams: NDArray[np.int64]
    segment_tiles: NDArray[np.int64]
    seam_axes: NDArray[np.int64]
    seam_coordinates: NDArray[np.float64]
    seam_sources: NDArray[np.int64]
    seam_tiles: NDArray[np.int64]
    holes: NDArray[np.float64]
    source_bounds: NDArray[np.float64]
    patch_segments: NDArray[np.float64]
    patch_sizes: NDArray[np.float64]
    span: float = eqx.field(static=True)
    lattice_shape: tuple[int, int] = eqx.field(static=True)
    region_size: float = eqx.field(static=True)
    gradation_slope: float = eqx.field(static=True)
    maximum_area: float = eqx.field(static=True)
    minimum_angle_degrees: float = eqx.field(static=True)
    maximum_steiner: int = eqx.field(static=True)
    maximum_triangles: int = eqx.field(static=True)
    construction_size: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    domain_area: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    compiled_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlanarSource,
        specification: SurfaceMeshingSpec,
        schedule: PlanarTileSchedule,
        /,
    ) -> None:
        limits = specification.limits
        # The canonical planar preparation owns the size field, hole seeds,
        # refinement controls and the uniform-median lattice admission; every
        # tile seeds the same canonical spacing natively.
        prepared = PreparedPlanarDomain(source, specification, lattice=True)
        vertices, edges, loops = source_arrays(source)
        lower = np.min(vertices, axis=0)
        upper = np.max(vertices, axis=0)
        span = schedule.tile_span_sizes * prepared.region_size
        if not math.isfinite(span) or span <= 0.0:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                f"planar_tiles.span must be finite and positive; got {span!r}",
                stage=_STAGE,
            )
        shape = tuple(
            max(1, math.ceil(float(upper[axis] - lower[axis]) / span)) for axis in (0, 1)
        )
        if shape[0] * shape[1] > limits.maximum_cells:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "planar_tiles.lattice must not exceed limits.maximum_cells tiles; got excess tiles",
                stage=_STAGE,
                requested=(("maximum_cells", limits.maximum_cells),),
                achieved=(("lattice_tiles", shape[0] * shape[1]),),
            )
        lines = tuple(
            lower[axis] + span * np.arange(1, shape[axis], dtype=np.float64)
            for axis in (0, 1)
        )
        boundary = loops >= 0
        # Interior lies left of every normalized region edge.
        wanted = np.where(loops == 0, 1, -1)
        forward = np.where(
            boundary, prepared.loop_signs[np.maximum(loops, 0)] == wanted, True
        )
        skeleton = _SkeletonPoints()
        for row, point in enumerate(vertices):
            skeleton.add(point, 0, row, math.nan)
        # Crossings: per source edge, (parameter, skeleton row) cut points.
        cuts: list[list[tuple[float, int]]] = [[] for _ in range(edges.shape[0])]
        seam_intervals: list[tuple[int, float, int, int, int, int]] = []
        for axis in (0, 1):
            across = 1 - axis
            for line, value in enumerate(lines[axis].tolist()):
                intervals = self._line_crossings(
                    vertices,
                    edges,
                    loops,
                    axis,
                    value,
                    lines[across],
                    skeleton,
                    cuts,
                )
                seam_intervals.extend(
                    (axis, value, line, first, last, source_edge)
                    for first, last, source_edge in intervals
                )
        # Seam pieces: inside intervals cut at the perpendicular lattice lines.
        seam_axes: list[int] = []
        seam_values: list[float] = []
        seam_sources: list[int] = []
        seam_tiles: list[tuple[tuple[int, int], tuple[int, int]]] = []
        pieces: list[tuple[int, int]] = []
        piece_sources: list[int] = []
        piece_seams: list[int] = []
        piece_tiles: list[tuple[tuple[int, int], tuple[int, int] | None]] = []
        for axis, value, line, first, last, source_edge in seam_intervals:
            across = 1 - axis
            start = np.asarray(skeleton.coordinates[first])
            end = np.asarray(skeleton.coordinates[last])
            low, high = sorted((float(start[across]), float(end[across])))
            corners = lines[across][(lines[across] > low) & (lines[across] < high)]
            stops = [first]
            for corner in corners.tolist():
                point = np.empty((2,), dtype=np.float64)
                point[axis], point[across] = value, corner
                if source_edge >= 0:
                    a, b = vertices[edges[source_edge]]
                    direction = b - a
                    parameter = float(
                        np.dot(point - a, direction) / np.dot(direction, direction)
                    )
                    stops.append(skeleton.add(point, 1, source_edge, parameter))
                else:
                    stops.append(skeleton.add(point, 2, 0, math.nan))
            stops.append(last)
            for begin, finish in zip(stops[:-1], stops[1:], strict=True):
                middle = 0.5 * (
                    np.asarray(skeleton.coordinates[begin])
                    + np.asarray(skeleton.coordinates[finish])
                )
                index = min(
                    max(int(math.floor((middle[across] - lower[across]) / span)), 0),
                    shape[across] - 1,
                )
                below = [0, 0]
                below[axis], below[across] = line, index
                above = [0, 0]
                above[axis], above[across] = line + 1, index
                seam = len(seam_axes)
                seam_axes.append(axis)
                seam_values.append(value)
                seam_sources.append(source_edge)
                seam_tiles.append(((below[0], below[1]), (above[0], above[1])))
                pieces.append((begin, finish))
                piece_sources.append(source_edge)
                piece_seams.append(seam)
                piece_tiles.append(((below[0], below[1]), (above[0], above[1])))
        # Source sub-pieces between consecutive crossings lie in one tile; an
        # embedded piece on a lattice line is a seam above.
        for edge, (first, last) in enumerate(edges.tolist()):
            stops = sorted(set(cuts[edge] + [(0.0, first), (1.0, last)]))
            rows = [stops[0][1]]
            for parameter, row in stops[1:]:
                if row != rows[-1]:
                    rows.append(row)
            for begin, finish in zip(rows[:-1], rows[1:], strict=True):
                a = np.asarray(skeleton.coordinates[begin])
                b = np.asarray(skeleton.coordinates[finish])
                middle = 0.5 * (a + b)
                tile = [0, 0]
                on_line = False
                for axis in (0, 1):
                    collinear = a[axis] == b[axis] and np.any(lines[axis] == a[axis])
                    if collinear:
                        on_line = True
                        line = int(np.flatnonzero(lines[axis] == a[axis])[0])
                        along = (b - a) if forward[edge] else (a - b)
                        # Left normal of the oriented piece points to the interior.
                        normal = along[1 - axis] * (-1 if axis == 0 else 1)
                        tile[axis] = line if normal < 0 else line + 1
                    else:
                        tile[axis] = min(
                            max(int(math.floor((middle[axis] - lower[axis]) / span)), 0),
                            shape[axis] - 1,
                        )
                if on_line and not boundary[edge]:
                    continue
                pieces.append((begin, finish) if forward[edge] else (finish, begin))
                piece_sources.append(edge)
                piece_seams.append(-1)
                piece_tiles.append(((tile[0], tile[1]), None))
        tile_keys = sorted(
            {key for pair in piece_tiles for key in pair if key is not None}
        )
        if not tile_keys:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "planar_tiles.region must contain constraint pieces; got an empty tile complex",
                stage=_STAGE,
            )
        keys = np.asarray(tile_keys, dtype=np.int64).reshape((-1, 2))
        order = np.argsort(_morton(keys[:, 0], keys[:, 1]), kind="stable")
        keys = keys[order]
        tile_row = {(int(i), int(j)): row for row, (i, j) in enumerate(keys.tolist())}
        coordinates = np.asarray(skeleton.coordinates, dtype=np.float64).reshape((-1, 2))
        piece_rows = np.asarray(pieces, dtype=np.int64).reshape((-1, 2))
        piece_tile_rows = np.asarray(
            [
                (tile_row[first], -1 if second is None else tile_row[second])
                for first, second in piece_tiles
            ],
            dtype=np.int64,
        ).reshape((-1, 2))
        # Region area per tile from its oriented boundary: source pieces keep
        # the interior on their left and seams count once on either side.
        weights = np.zeros((keys.shape[0],), dtype=np.float64)
        start = coordinates[piece_rows[:, 0]]
        end = coordinates[piece_rows[:, 1]]
        cross = start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0]
        seam_rows = np.asarray(piece_seams, dtype=np.int64)
        source_rows = np.asarray(piece_sources, dtype=np.int64)
        region_piece = (seam_rows < 0) & (loops[np.maximum(source_rows, 0)] >= 0)
        np.add.at(weights, piece_tile_rows[region_piece, 0], 0.5 * cross[region_piece])
        seam_piece = seam_rows >= 0
        axes = np.asarray(seam_axes, dtype=np.int64)
        # Increasing coordinate along a constant-x seam leaves the lower tile on
        # the left; along a constant-y seam it leaves the upper tile there.
        increasing = np.where(
            axes[seam_rows[seam_piece]] == 0,
            start[seam_piece, 1] < end[seam_piece, 1],
            start[seam_piece, 0] < end[seam_piece, 0],
        )
        left_is_lower = increasing == (axes[seam_rows[seam_piece]] == 0)
        lower_sign = np.where(left_is_lower, 1.0, -1.0)
        np.add.at(
            weights, piece_tile_rows[seam_piece, 0], 0.5 * lower_sign * cross[seam_piece]
        )
        np.add.at(
            weights, piece_tile_rows[seam_piece, 1], -0.5 * lower_sign * cross[seam_piece]
        )
        region_area = 0.0
        region_edges = edges[boundary]
        a, b = vertices[region_edges[:, 0]], vertices[region_edges[:, 1]]
        oriented = np.where(forward[boundary][:, None], 1.0, -1.0)
        region_area = float(
            0.5
            * np.sum(oriented[:, 0] * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0, float(np.max(np.abs(vertices))) ** 2
        ) * max(1, piece_rows.shape[0])
        if (
            np.any(weights <= 0.0)
            or not math.isfinite(region_area)
            or region_area <= 0.0
            or abs(float(np.sum(weights)) - region_area) > tolerance
        ):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "planar_tiles.weights must partition the authored region area; got an invalid tile cover",
                stage=_STAGE,
                achieved=(
                    ("tile_area_sum", float(np.sum(weights))),
                    ("region_area", region_area),
                    ("minimum_tile_area", float(np.min(weights))),
                ),
            )
        # Canonical splitting of every source and seam piece.
        points, segments, owner, point_sources, point_parameters = split_constraints(
            coordinates,
            piece_rows,
            prepared.construction_size if prepared.lattice_spacing else prepared.region_size,
            prepared.patch_segments,
            prepared.patch_sizes,
            prepared.gradation_slope,
            limits,
            lattice_spacing=prepared.lattice_spacing,
        )
        base = coordinates.shape[0]
        added = point_sources >= 0
        dimensions = np.concatenate(
            (
                np.asarray(skeleton.dimensions, dtype=np.int64),
                np.where(source_rows[point_sources[added]] >= 0, 1, 2),
            )
        )
        indices = np.concatenate(
            (
                np.asarray(skeleton.indices, dtype=np.int64),
                np.maximum(source_rows[point_sources[added]], 0),
            )
        )
        parameters = np.concatenate(
            (
                np.asarray(skeleton.parameters, dtype=np.float64),
                np.full((int(np.sum(added)),), math.nan, dtype=np.float64),
            )
        )
        split_rows = np.flatnonzero(added)
        for offset, row in enumerate(split_rows.tolist()):
            piece = int(point_sources[row])
            edge = int(source_rows[piece])
            if edge < 0:
                continue
            first, last = piece_rows[piece]
            t0 = _edge_parameter(edges[edge], first, dimensions, indices, parameters, edge)
            t1 = _edge_parameter(edges[edge], last, dimensions, indices, parameters, edge)
            parameters[base + offset] = t0 + float(point_parameters[row]) * (t1 - t0)
        segment_tiles = piece_tile_rows[owner]
        point_tiles = np.full((points.shape[0],), np.iinfo(np.int64).max, dtype=np.int64)
        for column in (0, 1):
            valid = segment_tiles[:, column] >= 0
            for end_column in (0, 1):
                np.minimum.at(
                    point_tiles,
                    segments[valid, end_column],
                    segment_tiles[valid, column],
                )
        if np.any(point_tiles == np.iinfo(np.int64).max):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "planar_tiles.points must each bound a tile constraint; got unreferenced constraint points",
                stage=_STAGE,
            )
        seam_tile_rows = np.asarray(
            [(tile_row[first], tile_row[second]) for first, second in seam_tiles],
            dtype=np.int64,
        ).reshape((-1, 2))
        self.source = source
        self.specification = specification
        self.schedule = schedule
        self.origin = lower.astype(np.float64)
        self.tiles = keys
        self.tile_weights = weights
        self.points = points
        self.point_dimensions = dimensions
        self.point_indices = indices
        self.point_parameters = parameters
        self.point_tiles = point_tiles
        self.segments = segments.astype(np.int64)
        self.segment_sources = source_rows[owner]
        self.segment_seams = seam_rows[owner]
        self.segment_tiles = segment_tiles
        self.seam_axes = axes
        self.seam_coordinates = np.asarray(seam_values, dtype=np.float64)
        self.seam_sources = np.asarray(seam_sources, dtype=np.int64)
        self.seam_tiles = seam_tile_rows
        self.holes = np.asarray(prepared.holes, dtype=np.float64).reshape((-1, 2))
        self.source_bounds = np.asarray(prepared.source_bounds, dtype=np.float64)
        self.patch_segments = np.asarray(prepared.patch_segments, dtype=np.float64)
        self.patch_sizes = np.asarray(prepared.patch_sizes, dtype=np.float64)
        self.span = float(span)
        self.lattice_shape = (int(shape[0]), int(shape[1]))
        self.region_size = float(prepared.region_size)
        self.gradation_slope = float(prepared.gradation_slope)
        self.maximum_area = float(prepared.maximum_area)
        self.minimum_angle_degrees = float(prepared.minimum_angle_degrees)
        self.maximum_steiner = int(prepared.maximum_steiner)
        self.maximum_triangles = int(prepared.maximum_triangles)
        self.construction_size = float(prepared.construction_size)
        self.lattice_spacing = float(prepared.lattice_spacing)
        self.domain_area = region_area
        self.prepared_id = prepared.prepared_id
        self.compiled_id = canonical_fingerprint(
            {
                "kind": "planar-tile-complex",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
                "prepared": prepared.prepared_id,
                "arrays": array_tree_fingerprint(
                    (
                        self.origin,
                        keys,
                        weights,
                        points,
                        dimensions,
                        indices,
                        parameters,
                        point_tiles,
                        self.segments,
                        self.segment_sources,
                        self.segment_seams,
                        segment_tiles,
                        axes,
                        self.seam_coordinates,
                        self.seam_sources,
                        seam_tile_rows,
                        self.holes,
                        self.source_bounds,
                    )
                ),
                "span": self.span,
                "lattice_shape": self.lattice_shape,
                "construction_size": self.construction_size,
                "lattice_spacing": self.lattice_spacing,
            }
        )

    @staticmethod
    def _line_crossings(
        vertices: np.ndarray,
        edges: np.ndarray,
        loops: np.ndarray,
        axis: int,
        value: float,
        perpendicular: np.ndarray,
        skeleton: _SkeletonPoints,
        cuts: list[list[tuple[float, int]]],
        /,
    ) -> list[tuple[int, int, int]]:
        """Crossings of one lattice line and its inside intervals.

        A crossing exactly on a lattice corner adopts the corner coordinates so
        both lattice lines construct the same point. Intervals on a region edge
        are boundary, never seam; intervals on an embedded edge are seams that
        retain the embedded source edge.
        """
        across = 1 - axis
        a = vertices[edges[:, 0]]
        b = vertices[edges[:, 1]]
        stops: dict[float, int] = {}
        collinear: list[tuple[float, float, int]] = []
        for edge in range(edges.shape[0]):
            pa, pb = a[edge], b[edge]
            da, db = pa[axis] - value, pb[axis] - value
            if da == 0.0 and db == 0.0:
                low, high = sorted((float(pa[across]), float(pb[across])))
                collinear.append((low, high, edge))
                for vertex in edges[edge].tolist():
                    stops.setdefault(float(vertices[vertex, across]), vertex)
                continue
            if da == 0.0:
                stops.setdefault(float(pa[across]), int(edges[edge, 0]))
                continue
            if db == 0.0:
                stops.setdefault(float(pb[across]), int(edges[edge, 1]))
                continue
            if (da < 0.0) == (db < 0.0):
                continue
            point = np.empty((2,), dtype=np.float64)
            point[axis] = value
            parameter = float((value - pa[axis]) / (pb[axis] - pa[axis]))
            point[across] = pa[across] + parameter * (pb[across] - pa[across])
            corners = np.empty((perpendicular.size, 2), dtype=np.float64)
            corners[:, axis] = value
            corners[:, across] = perpendicular
            if corners.shape[0]:
                exact = exact_orient2d(pa[None, :], pb[None, :], corners) == 0
                low, high = sorted((float(pa[across]), float(pb[across])))
                exact &= (perpendicular > low) & (perpendicular < high)
                if np.any(exact):
                    point = corners[int(np.flatnonzero(exact)[0])]
                    direction = pb - pa
                    parameter = float(
                        np.dot(point - pa, direction) / np.dot(direction, direction)
                    )
            row = skeleton.add(point, 1, edge, parameter)
            cuts[edge].append((parameter, row))
            stops.setdefault(float(point[across]), row)
        values = sorted(stops)
        intervals: list[tuple[int, int, int]] = []
        region_a = a[loops >= 0]
        region_b = b[loops >= 0]
        for low, high in zip(values[:-1], values[1:], strict=True):
            covering = [
                edge for start, stop, edge in collinear if start <= low and high <= stop
            ]
            if covering:
                embedded = [edge for edge in covering if loops[edge] < 0]
                if len(embedded) == len(covering):
                    intervals.append((stops[low], stops[high], embedded[0]))
                continue
            middle = np.empty((1, 2), dtype=np.float64)
            middle[0, axis] = value
            middle[0, across] = 0.5 * (low + high)
            if bool(_inside(middle, region_a, region_b)[0]):
                intervals.append((stops[low], stops[high], -1))
        return intervals

    def tile_box(self, row: int, /) -> tuple[np.ndarray, np.ndarray]:
        """Half-open lattice box ``[low, high)`` of one tile row."""
        low = self.origin + self.span * self.tiles[row].astype(np.float64)
        return low, low + self.span

    def region_domain(self) -> PiecewiseLinearDomain:
        """Authored region with outward source loops from the canonical preparation."""
        return PreparedPlanarDomain(self.source, self.specification).domain(self.source)

    def entity_set_id(self, degree: int, /) -> str:
        """Source entity set of planar vertices (0), edges (1) or the region (2)."""
        if degree not in (0, 1, 2):
            raise ValueError(
                f"PlanarTileComplex.degree must name a planar source stratum; got {degree!r}"
            )
        return canonical_fingerprint(
            {
                "kind": "native-planar-source-entity-set",
                "source": self.source.binding_id,
                "dimension": degree,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: PlanarTileComplex, /) -> Self:
        """Rebuild the complex from its stored source, request and schedule."""
        rebuilt = cls(value.source, value.specification, value.schedule)
        if rebuilt.compiled_id != value.compiled_id:
            raise ValueError(
                "PlanarTileComplex.compiled_id must match its source, request and schedule; got changed stored content"
            )
        return rebuilt


def _edge_parameter(
    edge: np.ndarray,
    row: int,
    dimensions: np.ndarray,
    indices: np.ndarray,
    parameters: np.ndarray,
    edge_row: int,
    /,
) -> float:
    """Binary64 parameter of one constraint point along one source edge."""
    if dimensions[row] == 0:
        if indices[row] == edge[0]:
            return 0.0
        if indices[row] == edge[1]:
            return 1.0
    if dimensions[row] == 1 and indices[row] == edge_row:
        return float(parameters[row])
    raise ValueError(
        "planar_tiles.piece endpoints must lie on their source edge; got foreign provenance"
    )


def _balanced_owners(weights: np.ndarray, partition_count: int, /) -> NDArray[np.int32]:
    """Contiguous Morton blocks of tiles balanced by region area."""
    total = float(np.sum(weights))
    centers = np.cumsum(weights) - 0.5 * weights
    owners = np.minimum(
        np.floor(partition_count * centers / total).astype(np.int64), partition_count - 1
    ).astype(np.int32)
    owners.flags.writeable = False
    return owners


@dataclass(frozen=True, slots=True)
class PreparedDistributedPlanarGeneration:
    """One tile complex bound to the owners of the current process group.

    ``preparation_evidence`` is the ended native record of building the tile
    complex under the request allowance; it is owned by the authored source,
    so a plan imports it once as source preparation instead of recharging it.
    """

    tiles: PlanarTileComplex
    layout: DeviceGenerationLayout
    tile_owners: NDArray[np.int32]
    partition_count: int
    prepared_id: str
    preparation_evidence: NativeExecutionRecord

    def __post_init__(self) -> None:
        if not isinstance(self.tiles, PlanarTileComplex) or not isinstance(
            self.layout, DeviceGenerationLayout
        ):
            raise TypeError(
                "planar_generation.tiles must pair PlanarTileComplex with DeviceGenerationLayout; got incompatible inputs"
            )
        if (
            not isinstance(self.tile_owners, np.ndarray)
            or self.tile_owners.dtype != np.int32
            or self.tile_owners.shape != (self.tiles.tiles.shape[0],)
            or self.tile_owners.flags.writeable
        ):
            raise ValueError(
                "planar_generation.tile_owners must be immutable int32 rows aligned with tiles; got incompatible owners"
            )
        if (
            isinstance(self.partition_count, bool)
            or not isinstance(self.partition_count, int)
            or self.partition_count != jax.process_count()
        ):
            raise ValueError(
                "planar_generation.partition_count must match this process group; got incompatible owner count"
            )
        if np.any(self.tile_owners < 0) or np.any(self.tile_owners >= self.partition_count):
            raise ValueError(
                "planar_generation.tile_owners must lie in [0, partition_count); got foreign owners"
            )
        if (
            not isinstance(self.preparation_evidence, NativeExecutionRecord)
            or self.preparation_evidence.owner_id != self.tiles.source.source_id
        ):
            raise ValueError(
                "planar_generation.preparation_evidence must be the source-owned tile preparation record; got a foreign or missing record"
            )
        self.preparation_evidence.require_valid()

    @property
    def source(self) -> sources_owner.NativePlanarSource:
        return self.tiles.source

    @property
    def specification(self) -> SurfaceMeshingSpec:
        return self.tiles.specification

    @property
    def schedule(self) -> PlanarTileSchedule:
        return self.tiles.schedule

    def require_binding(
        self,
        source: sources_owner.NativePlanarSource,
        specification: SurfaceMeshingSpec,
        /,
    ) -> None:
        """Refuse a plan that rebinds this preparation to another source or request."""
        if (
            source.binding_id != self.tiles.source.binding_id
            or specification.specification_id != self.tiles.specification.specification_id
        ):
            raise ValueError(
                "planar_generation.binding must match source and specification; got a different binding"
            )


@checked
def prepare_distributed_planar_generation(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    schedule: PlanarTileSchedule,
    layout: DeviceGenerationLayout,
    /,
) -> PreparedDistributedPlanarGeneration:
    """Fix the tile complex from source/request/schedule, then assign owners.

    The complex is built inside its own request-limited native scope; the
    ended record is retained as source-owned preparation evidence and, inside
    a caller's scope, remembered by that scope's canonical preparation ledger.
    """
    scope = NativeExecutionScope(specification.limits.native_limits())
    with scope, scope.host_workspace() as storage:
        storage.retain_owner(source)
        storage.retain_owner(specification)
        tiles = PlanarTileComplex(source, specification, schedule)
        storage.retain_owner(tiles)
    if scope.evidence is None:
        raise RuntimeError(
            "planar_generation.preparation must end with native execution evidence; got None"
        )
    receipt = NativeExecutionRecord(scope.evidence, owner_id=source.source_id)
    parent = scope.parent
    if parent is not None:
        with parent.host_workspace() as storage:
            ledger = native_preparation_ledger(parent)
            ledger.remember_receipt(parent, receipt)
            ledger.remember_source(parent, source.source_id)
            storage.retain_owner(tiles)
    partition_count = jax.process_count()
    owners = _balanced_owners(tiles.tile_weights, partition_count)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-distributed-planar-generation",
            "tiles": tiles.compiled_id,
            "layout": (
                layout.vertex_capacity,
                layout.cell_capacity,
                layout.candidate_capacity,
                layout.maximum_work_units,
                layout.neighborhood_depth,
            ),
            "owners": array_tree_fingerprint(owners),
            "partition_count": partition_count,
        }
    )
    return PreparedDistributedPlanarGeneration(
        tiles, layout, owners, partition_count, prepared_id, receipt
    )


@dataclass(frozen=True, slots=True)
class DistributedPlanarConstruction:
    """Owner-local generated rows; acceptance belongs to the collective root.

    Vertex rows cover every vertex of this process's cells (shared seam and
    crossing vertices appear on each incident owner with identical
    coordinates, identity and owner). ``vertex_dimensions``/``vertex_indices``
    and ``vertex_parameters`` retain source-vertex, source-edge or region
    provenance. ``edge_vertices`` are local rows of constrained edges with
    their source edge (or -1) and seam piece (or -1). Neighbor bytes count
    sent plus received fixed-capacity seam packets.
    """

    coordinates: NDArray[np.float64]
    vertex_ids: NDArray[np.int64]
    vertex_owners: NDArray[np.int32]
    vertex_dimensions: NDArray[np.int64]
    vertex_indices: NDArray[np.int64]
    vertex_parameters: NDArray[np.float64]
    triangles: NDArray[np.int64]
    cell_ids: NDArray[np.int64]
    cell_tiles: NDArray[np.int64]
    edge_vertices: NDArray[np.int64]
    edge_sources: NDArray[np.int64]
    edge_seams: NDArray[np.int64]
    local_tiles: NDArray[np.int64]
    global_vertex_count: int
    global_cell_count: int
    minimum_angle: float
    steiner_count: int
    refinement_complete: bool
    seam_rounds: int
    neighbor_packet_count: int
    neighbor_transferred_bytes: int
    metadata_bytes: int
    work_units: int
    elapsed_seconds: float
    source_id: str
    source_revision: str
    specification_id: str
    compiled_id: str


@dataclass(frozen=True, slots=True)
class _TileMesh:
    """One refined tile: mesh points, triangles and point provenance kinds."""

    points: np.ndarray
    triangles: np.ndarray
    kinds: np.ndarray
    references: np.ndarray
    dimensions: np.ndarray
    indices: np.ndarray
    parameters: np.ndarray
    edge_vertices: np.ndarray
    edge_sources: np.ndarray
    edge_seams: np.ndarray
    seam_points: dict[int, np.ndarray]
    minimum_angle: float
    steiner_count: int
    complete: bool


# Point kinds of one tile mesh: skeleton row, seam point, or tile-local point.
_SKELETON, _SEAM, _LOCAL = 0, 1, 2


def _seam_points_on_line(
    tiles: PlanarTileComplex, seam: int, extra: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Ordered skeleton rows and extra points along one seam piece."""
    rows = np.unique(tiles.segments[tiles.segment_seams == seam])
    along = 1 - int(tiles.seam_axes[seam])
    values = np.concatenate((tiles.points[rows, along], extra[:, along]))
    return rows, np.argsort(values, kind="stable")


def _seam_provenance(
    tiles: PlanarTileComplex, seam: int, point: np.ndarray, /
) -> tuple[int, int, float]:
    """Seam points are region points unless their seam lies on an embedded edge."""
    edge = int(tiles.seam_sources[seam])
    if edge < 0:
        return 2, 0, math.nan
    vertices, edges, _ = source_arrays(tiles.source)
    a, b = vertices[edges[edge]]
    direction = b - a
    return 1, edge, float(np.dot(point - a, direction) / np.dot(direction, direction))


def _triangulate_tile(
    tiles: PlanarTileComplex,
    row: int,
    seam_points: dict[int, np.ndarray],
    limits: MeshingLimits,
    started: float,
    /,
) -> _TileMesh:
    """Refine one tile under its fixed constraints and the current seam points."""
    incident = (tiles.segment_tiles[:, 0] == row) | (tiles.segment_tiles[:, 1] == row)
    rows = np.flatnonzero(incident)
    boundary = rows[tiles.segment_seams[rows] < 0]
    seams = np.unique(tiles.segment_seams[rows][tiles.segment_seams[rows] >= 0])
    used = np.unique(tiles.segments[rows])
    coordinates = [tiles.points[used]]
    kinds = [np.full((used.size,), _SKELETON, dtype=np.int64)]
    references = [used]
    seam_of_point = [np.full((used.size,), -1, dtype=np.int64)]
    count = used.size
    segment_rows: list[np.ndarray] = [np.searchsorted(used, tiles.segments[boundary])]
    segment_sources = [tiles.segment_sources[boundary]]
    segment_seams = [np.full((boundary.size,), -1, dtype=np.int64)]
    for seam in seams.tolist():
        extra = seam_points.get(seam, np.empty((0, 2), dtype=np.float64))
        skeleton_rows, order = _seam_points_on_line(tiles, seam, extra)
        local = np.concatenate(
            (
                np.searchsorted(used, skeleton_rows),
                count + np.arange(extra.shape[0], dtype=np.int64),
            )
        )[order]
        coordinates.append(extra)
        kinds.append(np.full((extra.shape[0],), _SEAM, dtype=np.int64))
        references.append(np.arange(extra.shape[0], dtype=np.int64))
        seam_of_point.append(np.full((extra.shape[0],), seam, dtype=np.int64))
        count += extra.shape[0]
        pairs = np.stack((local[:-1], local[1:]), axis=1)
        segment_rows.append(pairs)
        segment_sources.append(
            np.full((pairs.shape[0],), int(tiles.seam_sources[seam]), dtype=np.int64)
        )
        segment_seams.append(np.full((pairs.shape[0],), seam, dtype=np.int64))
    segments = np.concatenate(segment_rows).astype(np.int64)
    sources = np.concatenate(segment_sources)
    seam_rows = np.concatenate(segment_seams)
    inputs = np.concatenate(coordinates)
    input_kinds = np.concatenate(kinds)
    input_references = np.concatenate(references)
    input_seams = np.concatenate(seam_of_point)
    low, high = tiles.tile_box(row)
    holes = tiles.holes[np.all((tiles.holes >= low) & (tiles.holes < high), axis=1)]
    extra = np.empty((0, 2), dtype=np.float64)
    uniform = tiles.patch_sizes.size == 0
    while True:
        try:
            triangulation = triangulation_owner.ConstrainedDelaunayTriangulation(
                np.concatenate((inputs, extra), axis=0),
                segments,
                holes=holes if holes.shape[0] else None,
                keep_convex_hull=False,
                min_angle=tiles.minimum_angle_degrees,
                max_area=tiles.maximum_area,
                max_edge_length=tiles.construction_size if uniform else math.inf,
                lattice_spacing=tiles.lattice_spacing,
                max_steiner=max(tiles.maximum_steiner - extra.shape[0], 0),
                max_triangles=tiles.maximum_triangles,
            )
        except MeshcoreError as error:
            if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
                raise
            raise _capacity_failure(error, limits) from error
        check_deadline(started, limits, MeshingStageKind.SURFACE_MESHING)
        points = np.asarray(triangulation.points, dtype=np.float64)
        triangles = np.asarray(triangulation.triangles, dtype=np.int64)
        segment_ids = np.asarray(triangulation.segment_ids, dtype=np.int64)
        if uniform:
            break
        pairs = np.sort(
            triangles[:, np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)].reshape(
                (-1, 2)
            ),
            axis=1,
        )
        opposite = segment_ids[:, np.asarray((2, 0, 1), dtype=np.int64)].reshape(-1)
        free = opposite < 0
        a, b = points[pairs[:, 0]], points[pairs[:, 1]]
        lengths = np.linalg.norm(b - a, axis=1)
        local = np.min(
            _field_sizes(
                np.concatenate((a, b, 0.5 * (a + b))),
                tiles.region_size,
                tiles.patch_segments,
                tiles.patch_sizes,
                tiles.gradation_slope,
            ).reshape(3, -1),
            axis=0,
        )
        long = free & (lengths > local)
        if not np.any(long):
            break
        midpoints = np.unique(0.5 * (a[long] + b[long]), axis=0)
        if extra.shape[0] + midpoints.shape[0] > tiles.maximum_steiner:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "planar_tiles.refinement must respect the refinement point budget; got excess edge-length work",
                stage=MeshingStageKind.SURFACE_MESHING.value,
                requested=(("maximum_steiner_points", tiles.maximum_steiner),),
                achieved=(("steiner_points", extra.shape[0] + midpoints.shape[0]),),
            )
        extra = np.concatenate((extra, midpoints), axis=0)
    total_inputs = inputs.shape[0] + extra.shape[0]
    parents = np.asarray(triangulation.constraint_parents, dtype=np.int64)
    witness_segments = np.asarray(triangulation.constraint_sources, dtype=np.int64)
    ratios = np.asarray(triangulation.constraint_ratios, dtype=np.float64)
    vertices, edges, _ = source_arrays(tiles.source)
    count = points.shape[0]
    kinds_out = np.full((count,), _LOCAL, dtype=np.int64)
    references_out = np.full((count,), -1, dtype=np.int64)
    dimensions = np.full((count,), 2, dtype=np.int64)
    indices = np.zeros((count,), dtype=np.int64)
    parameters = np.full((count,), math.nan, dtype=np.float64)
    kinds_out[: inputs.shape[0]] = input_kinds
    references_out[: inputs.shape[0]] = input_references
    skeleton = input_kinds == _SKELETON
    skeleton_rows = input_references[skeleton]
    dimensions[: inputs.shape[0]][skeleton] = tiles.point_dimensions[skeleton_rows]
    indices[: inputs.shape[0]][skeleton] = tiles.point_indices[skeleton_rows]
    parameters[: inputs.shape[0]][skeleton] = tiles.point_parameters[skeleton_rows]
    point_seams = np.full((count,), -1, dtype=np.int64)
    point_seams[: inputs.shape[0]] = input_seams
    for local_row in np.flatnonzero(input_kinds == _SEAM).tolist():
        dimension, index, parameter = _seam_provenance(
            tiles, int(input_seams[local_row]), inputs[local_row]
        )
        dimensions[local_row], indices[local_row] = dimension, index
        parameters[local_row] = parameter
    new_seam: dict[int, list[np.ndarray]] = {}
    for vertex in range(total_inputs, count):
        segment = int(witness_segments[vertex])
        if segment == -2:
            continue
        if segment == -1:
            continue
        seam = int(seam_rows[segment])
        if seam >= 0:
            kinds_out[vertex] = _SEAM
            point_seams[vertex] = seam
            new_seam.setdefault(seam, []).append(points[vertex])
            dimension, index, parameter = _seam_provenance(tiles, seam, points[vertex])
            dimensions[vertex], indices[vertex] = dimension, index
            parameters[vertex] = parameter
            continue
        edge = int(sources[segment])
        first, second = parents[vertex].tolist()
        t0 = _point_edge_parameter(first, edge, edges, dimensions, indices, parameters)
        t1 = _point_edge_parameter(second, edge, edges, dimensions, indices, parameters)
        dimensions[vertex], indices[vertex] = 1, edge
        parameters[vertex] = t0 + float(ratios[vertex]) * (t1 - t0)
    referenced = np.zeros((count,), dtype=np.bool_)
    referenced[triangles.reshape(-1)] = True
    if np.any(~referenced[: inputs.shape[0]][input_kinds != _LOCAL]):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "planar_tiles.constraint_points must all bound tile triangles; got a dropped constraint point",
            stage=MeshingStageKind.SURFACE_MESHING.value,
        )
    if np.any(referenced & (witness_segments == -2)):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "planar_tiles.triangles must not retain hull-only splits; got an open tile constraint loop",
            stage=MeshingStageKind.SURFACE_MESHING.value,
        )
    corners = points[triangles]
    orientation = exact_orient2d(corners[:, 0], corners[:, 1], corners[:, 2])
    if np.any(orientation == 0):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "planar_tiles.triangles must have nonzero exact orientation; got a degenerate triangle",
            stage=MeshingStageKind.SURFACE_MESHING.value,
        )
    triangles = np.where(orientation[:, None] < 0, triangles[:, ::-1], triangles)
    segment_ids = np.where(orientation[:, None] < 0, segment_ids[:, ::-1], segment_ids)
    carried = segment_ids >= 0
    cell_rows, opposite = np.nonzero(carried)
    first = triangles[cell_rows, (opposite + 1) % 3]
    second = triangles[cell_rows, (opposite + 2) % 3]
    edge_pairs = np.sort(np.stack((first, second), axis=1), axis=1)
    carried_segments = segment_ids[cell_rows, opposite]
    edge_pairs, unique_rows = np.unique(edge_pairs, axis=0, return_index=True)
    carried_segments = carried_segments[unique_rows]
    return _TileMesh(
        points=points,
        triangles=triangles,
        kinds=kinds_out,
        references=np.where(kinds_out == _SEAM, point_seams, references_out),
        dimensions=dimensions,
        indices=indices,
        parameters=parameters,
        edge_vertices=edge_pairs.astype(np.int64).reshape((-1, 2)),
        edge_sources=sources[carried_segments],
        edge_seams=seam_rows[carried_segments],
        seam_points={
            seam: np.asarray(values, dtype=np.float64).reshape((-1, 2))
            for seam, values in new_seam.items()
        },
        minimum_angle=math.radians(triangulation.evidence.minimum_angle_degrees),
        steiner_count=count - inputs.shape[0],
        complete=triangulation.evidence.status == "ok",
    )


def _point_edge_parameter(
    row: int,
    edge: int,
    edges: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    parameters: np.ndarray,
    /,
) -> float:
    """Parameter of one tile point along one source edge, from its provenance."""
    return _edge_parameter(edges[edge], row, dimensions, indices, parameters, edge)


def _process_placement() -> NamedSharding:
    mesh = owner_execution_mesh(process_owner_devices())
    return NamedSharding(mesh, PartitionSpec(mesh.axis_names[0]))


def _shift(data: jax.Array, mesh: Mesh, permutation: tuple[tuple[int, int], ...], /) -> jax.Array:
    axis = mesh.axis_names[0]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=PartitionSpec(axis),
        out_specs=PartitionSpec(axis),
        check_vma=False,
    )
    def send(value: jax.Array) -> jax.Array:
        return jax.lax.ppermute(value, axis, permutation)

    return jax.jit(send)(data)


def _neighbor_exchange(
    outgoing: dict[int, np.ndarray], sharding: NamedSharding, /
) -> tuple[dict[int, np.ndarray], int, int]:
    """Send fixed-capacity seam packets only along process shifts that carry one.

    Every process learns which owner pairs communicate from a P-by-P flag
    table and packet capacities from a P-by-P count table; payload travels by
    point-to-point permutation, never through a gather.
    """
    parts, me = jax.process_count(), jax.process_index()
    mesh = sharding.mesh
    if not isinstance(mesh, Mesh):
        raise TypeError(
            "planar_neighbor_exchange.sharding must bind concrete owner devices; got an abstract mesh"
        )
    wanted = np.zeros((parts,), dtype=np.bool_)
    sizes = np.zeros((parts,), dtype=np.int64)
    for destination, rows in outgoing.items():
        shift = (destination - me) % parts
        wanted[shift] = rows.shape[0] > 0
        sizes[shift] = rows.shape[0]
    flags = np.asarray(multihost_utils.process_allgather(wanted, tiled=False)).reshape(
        (parts, parts)
    )
    counts = np.asarray(multihost_utils.process_allgather(sizes, tiled=False)).reshape(
        (parts, parts)
    )
    received: dict[int, np.ndarray] = {}
    packets, transferred = 0, 0
    for shift in np.flatnonzero(np.any(flags, axis=0)).tolist():
        if shift == 0:
            continue
        capacity = int(np.max(counts[:, shift]))
        destination = (me + shift) % parts
        local = outgoing.get(destination, np.empty((0, _PACKET_WIDTH), dtype=np.float64))
        packet = np.zeros((capacity + 1, _PACKET_WIDTH), dtype=np.float64)
        packet[0, 0] = local.shape[0]
        packet[1 : 1 + local.shape[0]] = local
        data = jax.make_array_from_process_local_data(
            sharding, packet[None], (parts, capacity + 1, _PACKET_WIDTH)
        )
        permutation = tuple((rank, (rank + shift) % parts) for rank in range(parts))
        moved = _shift(data, mesh, permutation)
        answer = np.asarray(jax.device_get(moved.addressable_shards[0].data))[0]
        count = int(answer[0, 0])
        if count:
            received[(me - shift) % parts] = answer[1 : 1 + count]
        packets += 1
        transferred += 2 * packet.nbytes
    return received, packets, transferred


def _merge_seam(current: np.ndarray, added: np.ndarray, axis: int, /) -> np.ndarray:
    """Union of exact seam coordinates ordered along the seam."""
    union = np.unique(np.concatenate((current, added), axis=0), axis=0)
    return union[np.argsort(union[:, 1 - axis], kind="stable")]


def _owner_failure(error: Exception, /) -> MeshingFailure:
    if isinstance(error, MeshingFailure):
        return error
    if isinstance(error, MeshcoreError):
        resource = error.status in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT)
    else:
        resource = False
    failure = MeshingFailure(
        MeshingFailureCategory.RESOURCE_EXHAUSTED
        if resource
        else MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
        str(error),
        stage=_STAGE,
    )
    failure.__cause__ = error
    return failure


def _agree_inputs(prepared: PreparedDistributedPlanarGeneration, /) -> int:
    binding = canonical_fingerprint(
        (
            prepared.prepared_id,
            prepared.tiles.compiled_id,
            prepared.schedule.schedule_id,
        )
    )
    bindings = np.asarray(
        multihost_utils.process_allgather(
            np.frombuffer(bytes.fromhex(binding), dtype=np.uint8), tiled=False
        )
    )
    if np.any(bindings != bindings[0]):
        raise ValueError(
            "planar_generation.inputs must agree across processes; got different tile complexes or owners"
        )
    return int(bindings.nbytes)


@checked
def execute_distributed_planar_initial(
    prepared: PreparedDistributedPlanarGeneration, /
) -> DistributedPlanarConstruction:
    """Refine owned tiles, agree on seams with neighbors, and number every row.

    Any owner's refusal is raised on every process before rows are returned.
    """
    tiles = prepared.tiles
    limits = prepared.specification.limits
    sharding = _process_placement()
    me = jax.process_index()
    metadata = _agree_inputs(prepared)
    local = np.flatnonzero(prepared.tile_owners == me)
    seam_points: dict[int, np.ndarray] = {}
    results: dict[int, _TileMesh] = {}
    pending = set(local.tolist())
    packets = transferred = 0
    rounds = 0
    begun = monotonic()
    with NativeExecutionScope(limits.native_limits()) as scope:
        started = scope.operation_started
        while True:
            failure = None
            try:
                for row in sorted(pending):
                    results[row] = _triangulate_tile(
                        tiles, row, seam_points, limits, started
                    )
            except (MeshingFailure, MeshcoreError, ValueError) as error:
                failure = _owner_failure(error)
            collective_failure_barrier(
                failure, message=_REMOTE_REFUSAL, stage=f"{_STAGE}-construction"
            )
            rounds += 1
            outgoing: dict[int, list[np.ndarray]] = {}
            changed: set[int] = set()
            for row in sorted(pending):
                for seam, values in results[row].seam_points.items():
                    axis = int(tiles.seam_axes[seam])
                    current = seam_points.get(seam, np.empty((0, 2), dtype=np.float64))
                    merged = _merge_seam(current, values, axis)
                    if merged.shape[0] != current.shape[0]:
                        seam_points[seam] = merged
                        changed.add(seam)
                    for tile in tiles.seam_tiles[seam].tolist():
                        owner = int(prepared.tile_owners[tile])
                        if owner != me:
                            outgoing.setdefault(owner, []).append(
                                np.concatenate(
                                    (np.full((values.shape[0], 1), seam, dtype=np.float64), values),
                                    axis=1,
                                )
                            )
            received, sent, moved = _neighbor_exchange(
                {
                    owner: np.concatenate(rows, axis=0)
                    for owner, rows in outgoing.items()
                },
                sharding,
            )
            packets += sent
            transferred += moved
            for rows in received.values():
                for seam_value in np.unique(rows[:, 0]).tolist():
                    seam = int(seam_value)
                    values = rows[rows[:, 0] == seam_value, 1:]
                    axis = int(tiles.seam_axes[seam])
                    current = seam_points.get(seam, np.empty((0, 2), dtype=np.float64))
                    merged = _merge_seam(current, values, axis)
                    if merged.shape[0] != current.shape[0]:
                        seam_points[seam] = merged
                        changed.add(seam)
            flags = np.asarray(
                multihost_utils.process_allgather(np.bool_(bool(changed)), tiled=False)
            )
            if not np.any(flags):
                break
            if rounds >= prepared.schedule.maximum_seam_rounds:
                collective_failure_barrier(
                    MeshingFailure(
                        MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        "planar_tiles.seams must agree within schedule.maximum_seam_rounds; got unconverged seam refinement",
                        stage=f"{_STAGE}-seams",
                        requested=(
                            ("maximum_seam_rounds", prepared.schedule.maximum_seam_rounds),
                        ),
                        achieved=(("seam_rounds", rounds),),
                    ),
                    message=_REMOTE_REFUSAL,
                    stage=f"{_STAGE}-seams",
                )
            pending = {
                int(row)
                for row in local.tolist()
                if np.any(
                    np.isin(
                        tiles.segment_seams[
                            (tiles.segment_tiles[:, 0] == row)
                            | (tiles.segment_tiles[:, 1] == row)
                        ],
                        np.asarray(sorted(changed), dtype=np.int64),
                    )
                )
            }
        # The owner scope starts from the request allowance; its unreserved
        # remainder therefore measures this owner's charged work.
        work = limits.maximum_work_units - scope.remaining().remaining_work_units
    return _number_rows(
        prepared,
        results,
        seam_points,
        local,
        rounds=rounds,
        packets=packets,
        transferred=transferred,
        metadata=metadata,
        work=work,
        elapsed=monotonic() - begun,
    )


def _number_rows(
    prepared: PreparedDistributedPlanarGeneration,
    results: dict[int, _TileMesh],
    seam_points: dict[int, np.ndarray],
    local: np.ndarray,
    /,
    *,
    rounds: int,
    packets: int,
    transferred: int,
    metadata: int,
    work: int,
    elapsed: float,
) -> DistributedPlanarConstruction:
    """Assign skeleton, seam and per-tile identities from canonical counts."""
    tiles = prepared.tiles
    me = jax.process_index()
    tile_count = tiles.tiles.shape[0]
    seam_count = tiles.seam_axes.shape[0]
    counts = np.zeros((2 * tile_count + seam_count,), dtype=np.int64)
    tile_locals: dict[int, np.ndarray] = {}
    for row in local.tolist():
        mesh = results[row]
        referenced = np.zeros((mesh.points.shape[0],), dtype=np.bool_)
        referenced[mesh.triangles.reshape(-1)] = True
        tile_locals[row] = np.flatnonzero((mesh.kinds == _LOCAL) & referenced)
        counts[row] = tile_locals[row].size
        counts[tile_count + row] = mesh.triangles.shape[0]
    for seam in range(seam_count):
        lower_tile = int(np.min(tiles.seam_tiles[seam]))
        if int(prepared.tile_owners[lower_tile]) == me:
            counts[2 * tile_count + seam] = seam_points.get(
                seam, np.empty((0, 2), dtype=np.float64)
            ).shape[0]
    gathered = np.asarray(multihost_utils.process_allgather(counts, tiled=False)).reshape(
        (jax.process_count(), -1)
    )
    totals = np.sum(gathered, axis=0)
    local_counts = totals[:tile_count]
    cell_counts = totals[tile_count : 2 * tile_count]
    seam_counts = totals[2 * tile_count :]
    skeleton_count = tiles.points.shape[0]
    seam_offsets = skeleton_count + np.concatenate(([0], np.cumsum(seam_counts)[:-1]))
    seam_total = int(np.sum(seam_counts))
    local_offsets = (
        skeleton_count + seam_total + np.concatenate(([0], np.cumsum(local_counts)[:-1]))
    )
    cell_offsets = np.concatenate(([0], np.cumsum(cell_counts)[:-1]))
    vertex_total = skeleton_count + seam_total + int(np.sum(local_counts))
    cell_total = int(np.sum(cell_counts))
    identifiers: dict[int, tuple[np.ndarray, int, int, int, float]] = {}
    triangles, cell_ids, cell_tiles = [], [], []
    edge_rows, edge_sources, edge_seams = [], [], []
    minimum_angle = math.inf
    steiner = 0
    complete = True
    failure = None
    try:
        for row in local.tolist():
            mesh = results[row]
            ids = np.full((mesh.points.shape[0],), -1, dtype=np.int64)
            owners = np.full((mesh.points.shape[0],), -1, dtype=np.int32)
            skeleton = mesh.kinds == _SKELETON
            ids[skeleton] = mesh.references[skeleton]
            owners[skeleton] = prepared.tile_owners[
                tiles.point_tiles[mesh.references[skeleton]]
            ]
            for seam in np.unique(mesh.references[mesh.kinds == _SEAM]).tolist():
                selected = np.flatnonzero(
                    (mesh.kinds == _SEAM) & (mesh.references == seam)
                )
                ordered = seam_points[seam]
                along = 1 - int(tiles.seam_axes[seam])
                positions = np.minimum(
                    np.searchsorted(ordered[:, along], mesh.points[selected, along]),
                    ordered.shape[0] - 1,
                )
                if np.any(ordered[positions] != mesh.points[selected]):
                    raise ValueError(
                        "planar_tiles.seam_points must contain every tile seam vertex; got an unagreed seam vertex"
                    )
                ids[selected] = seam_offsets[seam] + positions
                owners[selected] = prepared.tile_owners[
                    int(np.min(tiles.seam_tiles[seam]))
                ]
            tile_local = tile_locals[row]
            ids[tile_local] = local_offsets[row] + np.arange(
                tile_local.size, dtype=np.int64
            )
            owners[tile_local] = prepared.tile_owners[row]
            for vertex in np.unique(mesh.triangles).tolist():
                identifier = int(ids[vertex])
                value = (
                    mesh.points[vertex],
                    int(owners[vertex]),
                    int(mesh.dimensions[vertex]),
                    int(mesh.indices[vertex]),
                    float(mesh.parameters[vertex]),
                )
                previous = identifiers.get(identifier)
                if identifier < 0 or (
                    previous is not None
                    and (
                        not np.array_equal(previous[0], value[0])
                        or previous[1:4] != value[1:4]
                    )
                ):
                    raise ValueError(
                        "planar_tiles.shared_vertices must agree on coordinates, owner and provenance; got conflicting shared rows"
                    )
                identifiers[identifier] = value
            triangles.append(ids[mesh.triangles])
            cell_ids.append(
                cell_offsets[row] + np.arange(mesh.triangles.shape[0], dtype=np.int64)
            )
            cell_tiles.append(np.full((mesh.triangles.shape[0],), row, dtype=np.int64))
            edge_rows.append(ids[mesh.edge_vertices])
            edge_sources.append(mesh.edge_sources)
            edge_seams.append(mesh.edge_seams)
            minimum_angle = min(minimum_angle, mesh.minimum_angle)
            steiner += mesh.steiner_count
            complete &= mesh.complete
    except ValueError as error:
        failure = _owner_failure(error)
    collective_failure_barrier(failure, message=_REMOTE_REFUSAL, stage=f"{_STAGE}-numbering")
    vertex_ids = np.asarray(sorted(identifiers), dtype=np.int64)
    rows = [identifiers[int(value)] for value in vertex_ids.tolist()]
    coordinates = np.asarray([row[0] for row in rows], dtype=np.float64).reshape((-1, 2))
    vertex_owners = np.asarray([row[1] for row in rows], dtype=np.int32)
    dimensions = np.asarray([row[2] for row in rows], dtype=np.int64)
    indices = np.asarray([row[3] for row in rows], dtype=np.int64)
    parameters = np.asarray([row[4] for row in rows], dtype=np.float64)
    corners = (
        np.concatenate(triangles) if triangles else np.empty((0, 3), dtype=np.int64)
    )
    pairs = np.concatenate(edge_rows) if edge_rows else np.empty((0, 2), dtype=np.int64)
    local_triangles = np.searchsorted(vertex_ids, corners).astype(np.int64)
    local_edges = np.sort(np.searchsorted(vertex_ids, pairs), axis=1).astype(np.int64)
    edge_keys, edge_unique = (
        np.unique(local_edges, axis=0, return_index=True)
        if local_edges.shape[0]
        else (local_edges, np.empty((0,), dtype=np.int64))
    )
    sources = (
        np.concatenate(edge_sources)[edge_unique] if edge_sources else np.empty((0,), np.int64)
    )
    seams = np.concatenate(edge_seams)[edge_unique] if edge_seams else np.empty((0,), np.int64)
    resources = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((work, steiner), dtype=np.int64), tiled=False
        )
    ).reshape((-1, 2))
    limits = prepared.specification.limits
    exhausted = (
        vertex_total > limits.maximum_vertices
        or cell_total > limits.maximum_faces
        or cell_total > limits.maximum_cells
        or int(np.sum(resources[:, 0])) > limits.maximum_work_units
    )
    collective_failure_barrier(
        MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "planar_generation.resources must fit global vertex, cell and work limits; got collective counts exceeding a request limit",
            stage=f"{_STAGE}-resources",
            requested=(
                ("maximum_vertices", limits.maximum_vertices),
                ("maximum_cells", limits.maximum_cells),
                ("maximum_work_units", limits.maximum_work_units),
            ),
            achieved=(
                ("vertices", vertex_total),
                ("cells", cell_total),
                ("work_units", int(np.sum(resources[:, 0]))),
            ),
        )
        if exhausted
        else None,
        message=_REMOTE_REFUSAL,
        stage=f"{_STAGE}-resources",
    )
    source = prepared.source
    return DistributedPlanarConstruction(
        coordinates=coordinates,
        vertex_ids=vertex_ids,
        vertex_owners=vertex_owners,
        vertex_dimensions=dimensions,
        vertex_indices=indices,
        vertex_parameters=parameters,
        triangles=local_triangles,
        cell_ids=np.concatenate(cell_ids) if cell_ids else np.empty((0,), np.int64),
        cell_tiles=np.concatenate(cell_tiles) if cell_tiles else np.empty((0,), np.int64),
        edge_vertices=edge_keys.reshape((-1, 2)),
        edge_sources=sources,
        edge_seams=seams,
        local_tiles=local.astype(np.int64),
        global_vertex_count=vertex_total,
        global_cell_count=cell_total,
        minimum_angle=minimum_angle,
        steiner_count=steiner,
        refinement_complete=bool(complete),
        seam_rounds=rounds,
        neighbor_packet_count=packets,
        neighbor_transferred_bytes=transferred,
        metadata_bytes=metadata + int(gathered.nbytes) + int(resources.nbytes),
        work_units=int(np.sum(resources[:, 0])),
        elapsed_seconds=float(elapsed),
        source_id=source.source_id,
        source_revision=source.source_revision,
        specification_id=prepared.specification.specification_id,
        compiled_id=tiles.compiled_id,
    )


__all__ = [
    "DistributedPlanarConstruction",
    "execute_distributed_planar_initial",
    "PlanarTileComplex",
    "PlanarTileSchedule",
    "prepare_distributed_planar_generation",
    "PreparedDistributedPlanarGeneration",
]
