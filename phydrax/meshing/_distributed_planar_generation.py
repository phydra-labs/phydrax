#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local planar constrained generation with canonical tile/seam identity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

import equinox as eqx
import jax
import numpy as np
from numpy.typing import NDArray

from .._execution_runtime import ExecutionGroup
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import exact_orient2d
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry import ConstrainedDelaunayTriangulation
from ..graph._distributed_partition import (
    DistributedGraphPartitionResult,
    DistributedWeightedCSRGraph,
    partition_graph_distributed,
)
from ..graph._partition import GraphPartitionPlan
from ..typing import checked
from ._contracts import MeshingFailure, MeshingFailureCategory, SurfaceMeshingSpec
from ._distribution import _owner_exchange_message_banks, _owner_group_allgather
from .providers import _native_sources as sources_owner
from .providers._native_planar import planar_support_issues
from .providers._native_planar_domain import (
    constrained_edges,
    PreparedPlanarDomain,
    split_constraints,
)


_STAGE = "distributed-planar-initial"
_REMOTE_REFUSAL = (
    "An execution-group owner rejected planar generation; no tile was published."
)
_SKELETON, _SEAM, _LOCAL = 0, 1, 2


def _frozen(value: np.ndarray, /) -> np.ndarray:
    array = np.asarray(value)
    array.setflags(write=False)
    return array


@final
class PlanarTileSchedule(StrictModule, NonTrainableState):
    """Source-independent tile spacing, expressed in resolved region sizes."""

    tile_span_sizes: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, *, tile_span_sizes: int = 2) -> None:
        if isinstance(tile_span_sizes, bool) or not isinstance(tile_span_sizes, int):
            raise TypeError("tile_span_sizes must be an integer.")
        if tile_span_sizes <= 0:
            raise ValueError("tile_span_sizes must be positive.")
        self.tile_span_sizes = tile_span_sizes
        self.schedule_id = canonical_fingerprint(
            {"kind": "planar-tile-schedule", "tile_span_sizes": tile_span_sizes}
        )


class _SkeletonPoints:
    """Exact-coordinate point bank with source-stratum provenance precedence."""

    def __init__(self) -> None:
        self.rows: dict[tuple[float, float], int] = {}
        self.coordinates: list[tuple[float, float]] = []
        self.dimensions: list[int] = []
        self.indices: list[int] = []
        self.parameters: list[float] = []

    def add(
        self, point: NDArray[np.float64], dimension: int, index: int, parameter: float, /
    ) -> int:
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


@dataclass(frozen=True, slots=True)
class _TileMesh:
    points: NDArray[np.float64]
    triangles: NDArray[np.int64]
    kinds: NDArray[np.int64]
    references: NDArray[np.int64]
    dimensions: NDArray[np.int64]
    indices: NDArray[np.int64]
    parameters: NDArray[np.float64]
    edge_vertices: NDArray[np.int64]
    edge_sources: NDArray[np.int64]
    edge_seams: NDArray[np.int64]
    seam_points: dict[int, NDArray[np.float64]]
    minimum_angle_degrees: float


@dataclass(frozen=True, slots=True)
class _TilePieceSkeleton:
    origin: NDArray[np.float64]
    span: float
    lattice_shape: tuple[int, int]
    loop_areas: NDArray[np.float64]
    source_edge_loops: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    dimensions: NDArray[np.int64]
    indices: NDArray[np.int64]
    parameters: NDArray[np.float64]
    seam_axes: NDArray[np.int64]
    seam_values: NDArray[np.float64]
    seam_sources: NDArray[np.int64]
    seam_tile_keys: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    piece_rows: NDArray[np.int64]
    piece_sources: NDArray[np.int64]
    piece_seams: NDArray[np.int64]
    piece_tile_keys: tuple[tuple[tuple[int, int], tuple[int, int] | None], ...]


@dataclass(frozen=True, slots=True)
class _TileInput:
    points: NDArray[np.float64]
    kinds: NDArray[np.int64]
    references: NDArray[np.int64]
    seam_rows: NDArray[np.int64]
    segments: NDArray[np.int64]
    sources: NDArray[np.int64]
    segment_seams: NDArray[np.int64]
    holes: NDArray[np.float64]
    minimum_angle_degrees: float


@dataclass(frozen=True, slots=True)
class _NumberedTileRows:
    identifiers: dict[int, tuple[NDArray[np.float64], int, int, int, float]]
    triangles: list[NDArray[np.int64]]
    cell_ids: list[NDArray[np.int64]]
    cell_tiles: list[NDArray[np.int64]]
    edge_records: dict[tuple[int, int], tuple[int, int]]
    minimum_angle_degrees: float


@final
class PlanarTileComplex(StrictModule, NonTrainableState):
    """Canonical constraint skeleton, tile graph, and shared seam complex."""

    source: sources_owner.NativePlanarSource
    specification: SurfaceMeshingSpec
    schedule: PlanarTileSchedule
    prepared: PreparedPlanarDomain
    origin: NDArray[np.float64]
    tiles: NDArray[np.int64]
    tile_ids: NDArray[np.int64]
    tile_weights: NDArray[np.float64]
    points: NDArray[np.float64]
    point_dimensions: NDArray[np.int64]
    point_indices: NDArray[np.int64]
    point_ids: NDArray[np.int64]
    point_parameters: NDArray[np.float64]
    point_tiles: NDArray[np.int64]
    segments: NDArray[np.int64]
    segment_sources: NDArray[np.int64]
    segment_seams: NDArray[np.int64]
    segment_tiles: NDArray[np.int64]
    seam_axes: NDArray[np.int64]
    seam_coordinates: NDArray[np.float64]
    seam_ids: NDArray[np.int64]
    seam_sources: NDArray[np.int64]
    seam_tiles: NDArray[np.int64]
    holes: NDArray[np.float64]
    span: float = eqx.field(static=True)
    lattice_shape: tuple[int, int] = eqx.field(static=True)
    compiled_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlanarSource,
        specification: SurfaceMeshingSpec,
        schedule: PlanarTileSchedule,
        /,
    ) -> None:
        unsupported = planar_support_issues(source, specification)
        if unsupported:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(unsupported),
                provider_code="native_planar_preflight",
                stage="source_inspection",
            )
        domain = PreparedPlanarDomain(source, specification)
        arrays = _build_tile_arrays(schedule, domain)
        (
            origin,
            tile_keys,
            weights,
            points,
            dimensions,
            indices,
            parameters,
            point_tiles,
            segments,
            segment_sources,
            segment_seams,
            segment_tiles,
            seam_axes,
            seam_coordinates,
            seam_sources,
            seam_tiles,
            holes,
            span,
            lattice_shape,
        ) = arrays
        self.source = source
        self.specification = specification
        self.schedule = schedule
        self.prepared = domain
        self.origin = _frozen(origin)
        self.tiles = _frozen(tile_keys)
        self.tile_ids = _frozen(
            _canonical_tile_ids(source, specification, schedule, origin, span, tile_keys)
        )
        self.tile_weights = _frozen(weights)
        self.points = _frozen(points)
        self.point_dimensions = _frozen(dimensions)
        self.point_indices = _frozen(indices)
        self.point_parameters = _frozen(parameters)
        self.point_ids = _frozen(
            _canonical_point_ids(domain, points, dimensions, indices, parameters)
        )
        self.point_tiles = _frozen(point_tiles)
        self.segments = _frozen(segments)
        self.segment_sources = _frozen(segment_sources)
        self.segment_seams = _frozen(segment_seams)
        self.segment_tiles = _frozen(segment_tiles)
        self.seam_axes = _frozen(seam_axes)
        self.seam_coordinates = _frozen(seam_coordinates)
        self.seam_ids = _frozen(
            _canonical_seam_ids(
                source,
                specification,
                domain,
                tile_keys,
                points,
                segments,
                segment_seams,
                seam_axes,
                seam_coordinates,
                seam_sources,
                seam_tiles,
            )
        )
        self.seam_sources = _frozen(seam_sources)
        self.seam_tiles = _frozen(seam_tiles)
        self.holes = _frozen(holes)
        self.span = span
        self.lattice_shape = lattice_shape
        self.compiled_id = canonical_fingerprint(
            {
                "kind": "native-planar-tile-complex",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
                "prepared": domain.prepared_id,
                "arrays": array_tree_fingerprint(
                    (
                        self.origin,
                        self.tiles,
                        self.tile_ids,
                        self.tile_weights,
                        self.points,
                        self.point_dimensions,
                        self.point_indices,
                        self.point_parameters,
                        self.point_ids,
                        self.point_tiles,
                        self.segments,
                        self.segment_sources,
                        self.segment_seams,
                        self.segment_tiles,
                        self.seam_axes,
                        self.seam_coordinates,
                        self.seam_ids,
                        self.seam_sources,
                        self.seam_tiles,
                        self.holes,
                    )
                ),
                "span": span,
                "lattice_shape": lattice_shape,
            }
        )

    def tile_box(self, row: int, /) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        low = self.origin + self.span * self.tiles[row].astype(np.float64)
        return low, low + self.span


@dataclass(frozen=True, slots=True)
class PreparedDistributedPlanarGeneration:
    """Prepared source-bound tiles and the local owner/neighbor directory."""

    tiles: PlanarTileComplex
    execution_group: ExecutionGroup
    partition_plan: GraphPartitionPlan
    partition_result: DistributedGraphPartitionResult
    local_tile_rows: NDArray[np.int64]
    known_tile_ids: NDArray[np.int64]
    known_tile_owners: NDArray[np.int32]
    prepared_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.tiles, PlanarTileComplex):
            raise TypeError("tiles must be PlanarTileComplex.")
        if not isinstance(self.execution_group, ExecutionGroup):
            raise TypeError("execution_group must be ExecutionGroup.")
        if not isinstance(self.partition_plan, GraphPartitionPlan) or not isinstance(
            self.partition_result, DistributedGraphPartitionResult
        ):
            raise TypeError(
                "partition evidence must use the native distributed graph API."
            )
        local = np.asarray(self.local_tile_rows)
        identifiers = np.asarray(self.known_tile_ids)
        owners = np.asarray(self.known_tile_owners)
        if (
            local.dtype != np.int64
            or local.ndim != 1
            or np.any((local < 0) | (local >= self.tiles.tiles.shape[0]))
        ):
            raise ValueError(
                "local_tile_rows must identify rows of the canonical tile complex."
            )
        if (
            identifiers.dtype != np.int64
            or identifiers.ndim != 1
            or owners.dtype != np.int32
            or owners.shape != identifiers.shape
            or np.unique(identifiers).size != identifiers.size
            or not np.array_equal(identifiers, np.sort(identifiers))
        ):
            raise ValueError(
                "known tile ownership must be unique, sorted, aligned metadata."
            )
        if np.any((owners < 0) | (owners >= self.partition_plan.part_count)):
            raise ValueError("known tile owners must be execution-group slots.")
        if local.flags.writeable or identifiers.flags.writeable or owners.flags.writeable:
            raise ValueError("prepared owner metadata must be immutable.")

    @property
    def source(self) -> sources_owner.NativePlanarSource:
        return self.tiles.source

    @property
    def specification(self) -> SurfaceMeshingSpec:
        return self.tiles.specification

    @property
    def schedule(self) -> PlanarTileSchedule:
        return self.tiles.schedule


@dataclass(frozen=True, slots=True)
class DistributedPlanarConstruction:
    """Owner-local triangles, constraints, canonical IDs, and source ancestry."""

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
    minimum_angle_degrees: float
    seam_rounds: int
    refinement_complete: bool
    source_id: str
    source_revision: str
    specification_id: str
    compiled_id: str
    partition_id: str


def _signed_loop_area(points: NDArray[np.float64], edges: NDArray[np.int64]) -> float:
    endpoints = points[edges]
    return float(
        0.5
        * np.sum(
            endpoints[:, 0, 0] * endpoints[:, 1, 1]
            - endpoints[:, 0, 1] * endpoints[:, 1, 0]
        )
    )


def _canonical_tile_ids(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    schedule: PlanarTileSchedule,
    origin: NDArray[np.float64],
    span: float,
    tile_keys: NDArray[np.int64],
    /,
) -> NDArray[np.int64]:
    values = np.asarray(
        [
            int(
                canonical_fingerprint(
                    {
                        "kind": "native-planar-tile",
                        "source": source.binding_id,
                        "specification": specification.specification_id,
                        "schedule": schedule.schedule_id,
                        "origin": origin.tolist(),
                        "span": span,
                        "lattice_key": tuple(int(value) for value in key),
                    }
                )[:16],
                16,
            )
            & ((1 << 63) - 1)
            for key in tile_keys.tolist()
        ],
        dtype=np.int64,
    )
    if np.unique(values).size != values.size:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Canonical planar tile identifiers collided.",
            provider_code="native_planar_tiles",
            stage=f"{_STAGE}-identity",
        )
    return values


def _canonical_seam_ids(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    tile_keys: NDArray[np.int64],
    points: NDArray[np.float64],
    segments: NDArray[np.int64],
    segment_seams: NDArray[np.int64],
    seam_axes: NDArray[np.int64],
    seam_coordinates: NDArray[np.float64],
    seam_sources: NDArray[np.int64],
    seam_tiles: NDArray[np.int64],
    /,
) -> NDArray[np.int64]:
    values: list[int] = []
    for seam in range(seam_axes.size):
        seam_segments = segments[segment_seams == seam]
        seam_points = np.unique(seam_segments)
        if seam_points.size < 2:
            raise ValueError("Canonical seam identity requires two endpoint points.")
        axis = int(seam_axes[seam])
        along = 1 - axis
        endpoints = points[seam_points, along]
        source_edge = int(seam_sources[seam])
        source_id = -1 if source_edge < 0 else int(prepared.source_edge_ids[source_edge])
        incident_tiles = tuple(
            sorted(
                tuple(int(value) for value in tile_keys[tile])
                for tile in seam_tiles[seam].tolist()
            )
        )
        key = (
            axis,
            float(seam_coordinates[seam]),
            float(np.min(endpoints)),
            float(np.max(endpoints)),
            source_id,
            incident_tiles,
        )
        digest = canonical_fingerprint(
            {
                "kind": "native-planar-seam",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "key": key,
            }
        )
        values.append(int(digest[:16], 16) & ((1 << 63) - 1))
    identifiers = np.asarray(values, dtype=np.int64)
    if np.unique(identifiers).size != identifiers.size:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Canonical planar seam identifiers collided.",
            provider_code="native_planar_tiles",
            stage=f"{_STAGE}-identity",
        )
    return identifiers


def _canonical_point_ids(
    prepared: PreparedPlanarDomain,
    coordinates: NDArray[np.float64],
    dimensions: NDArray[np.int64],
    indices: NDArray[np.int64],
    parameters: NDArray[np.float64],
    /,
) -> NDArray[np.int64]:
    region_id = int(prepared.source.region_scope_entity_ids[0])
    keys: list[tuple[int, int, float, float, float]] = []
    for row, dimension_value in enumerate(dimensions.tolist()):
        dimension = int(dimension_value)
        index = int(indices[row])
        if dimension == 0:
            entity = int(prepared.source_vertex_ids[index])
            parameter = 0.0
        elif dimension == 1:
            entity = int(prepared.source_edge_ids[index])
            parameter = float(parameters[row])
            if not math.isfinite(parameter):
                raise ValueError(
                    "Source-edge point identity requires a finite parameter."
                )
        elif dimension == 2:
            entity = region_id
            parameter = 0.0
        else:
            raise ValueError("Planar point identity names an unknown source dimension.")
        keys.append(
            (
                dimension,
                entity,
                parameter,
                float(coordinates[row, 0]),
                float(coordinates[row, 1]),
            )
        )
    order = sorted(range(len(keys)), key=keys.__getitem__)
    if any(keys[first] == keys[second] for first, second in zip(order[:-1], order[1:])):
        raise ValueError("Distinct planar points share one canonical semantic identity.")
    identifiers = np.empty((len(keys),), dtype=np.int64)
    identifiers[np.asarray(order, dtype=np.int64)] = np.arange(len(keys), dtype=np.int64)
    return identifiers


def _semantic_source_index(
    prepared: PreparedPlanarDomain, dimension: int, index: int, /
) -> int:
    if dimension == 0:
        return int(prepared.source_vertex_ids[index])
    if dimension == 1:
        return int(prepared.source_edge_ids[index])
    if dimension == 2:
        return int(prepared.source.region_scope_entity_ids[0])
    raise ValueError("Planar point provenance has an unsupported source dimension.")


def _source_edge_parameter(
    prepared: PreparedPlanarDomain,
    point_row: int,
    edge_row: int,
    /,
) -> float:
    source = int(prepared.point_sources[point_row])
    if source >= 0:
        if source != edge_row:
            raise ValueError("Planar constraint point has foreign source-edge ancestry.")
        return float(prepared.point_parameters[point_row])
    first, last = prepared.source_edges[edge_row]
    if point_row == first:
        return 0.0
    if point_row == last:
        return 1.0
    raise ValueError("Planar constraint point does not lie on its source edge.")


def _inside(
    points: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    /,
) -> NDArray[np.bool_]:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.bool_)
    signs = exact_orient2d(first[None, :, :], second[None, :, :], points[:, None, :])
    rising = (first[None, :, 1] <= points[:, None, 1]) & (
        points[:, None, 1] < second[None, :, 1]
    )
    falling = (second[None, :, 1] <= points[:, None, 1]) & (
        points[:, None, 1] < first[None, :, 1]
    )
    crossings = np.count_nonzero((rising & (signs > 0)) | (falling & (signs < 0)), axis=1)
    return crossings % 2 == 1


def _line_crossings(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int64],
    prepared_rows: NDArray[np.int64],
    edge_sources: NDArray[np.int64],
    loops: NDArray[np.int64],
    axis: int,
    value: float,
    perpendicular: NDArray[np.float64],
    skeleton: _SkeletonPoints,
    cuts: list[list[tuple[float, int]]],
    prepared: PreparedPlanarDomain,
    /,
) -> list[tuple[int, int, int]]:
    across = 1 - axis
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    stops: dict[float, int] = {}
    collinear: list[tuple[float, float, int]] = []
    for edge in range(edges.shape[0]):
        pa, pb = a[edge], b[edge]
        da, db = pa[axis] - value, pb[axis] - value
        if da == 0.0 and db == 0.0:
            low, high = sorted((float(pa[across]), float(pb[across])))
            collinear.append((low, high, edge))
            for point_row in edges[edge]:
                stops.setdefault(
                    float(vertices[point_row, across]), int(prepared_rows[point_row])
                )
            continue
        if da == 0.0:
            stops.setdefault(float(pa[across]), int(prepared_rows[edges[edge, 0]]))
            continue
        if db == 0.0:
            stops.setdefault(float(pb[across]), int(prepared_rows[edges[edge, 1]]))
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
        source_edge = int(edge_sources[edge])
        if source_edge < 0:
            dimension, index, source_parameter = 2, 0, math.nan
        else:
            start = _source_edge_parameter(prepared, int(edges[edge, 0]), source_edge)
            finish = _source_edge_parameter(prepared, int(edges[edge, 1]), source_edge)
            dimension, index = 1, source_edge
            source_parameter = start + parameter * (finish - start)
        row = skeleton.add(point, dimension, index, source_parameter)
        cuts[edge].append((parameter, row))
        stops.setdefault(float(point[across]), row)
    values = sorted(stops)
    intervals: list[tuple[int, int, int]] = []
    region_mask = loops >= 0
    region_a, region_b = a[region_mask], b[region_mask]
    for low, high in zip(values[:-1], values[1:], strict=True):
        covering = [
            edge for start, stop, edge in collinear if start <= low and high <= stop
        ]
        if covering:
            embedded = [edge for edge in covering if loops[edge] < 0]
            if embedded:
                intervals.append(
                    (stops[low], stops[high], int(edge_sources[embedded[0]]))
                )
            continue
        midpoint = np.empty((1, 2), dtype=np.float64)
        midpoint[0, axis] = value
        midpoint[0, across] = 0.5 * (low + high)
        if bool(_inside(midpoint, region_a, region_b)[0]):
            intervals.append((stops[low], stops[high], -1))
    return intervals


def _edge_parameter_from_metadata(
    prepared: PreparedPlanarDomain,
    edge: int,
    point: int,
    dimensions: NDArray[np.int64],
    indices: NDArray[np.int64],
    parameters: NDArray[np.float64],
    /,
) -> float:
    if dimensions[point] == 0:
        first, last = prepared.source_edges[edge]
        if indices[point] == first:
            return 0.0
        if indices[point] == last:
            return 1.0
    if dimensions[point] == 1 and indices[point] == edge:
        return float(parameters[point])
    raise ValueError("Tile constraint endpoint has foreign source-edge provenance.")


def _piece_patch_controls(
    prepared: PreparedPlanarDomain,
    source_rows: NDArray[np.int64],
    /,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    pieces_by_source: dict[int, list[int]] = {}
    for piece, source in enumerate(source_rows.tolist()):
        if source >= 0:
            pieces_by_source.setdefault(source, []).append(piece)
    segments: list[NDArray[np.float64]] = []
    sizes: list[float] = []
    rows: list[int] = []
    for patch_row, source in enumerate(prepared.patch_source_rows.tolist()):
        for piece in pieces_by_source.get(source, ()):
            segments.append(prepared.patch_segments[patch_row])
            sizes.append(float(prepared.patch_sizes[patch_row]))
            rows.append(piece)
    patch_segments = (
        np.asarray(segments, dtype=np.float64).reshape((-1, 2, 2))
        if segments
        else np.empty((0, 2, 2), dtype=np.float64)
    )
    return (
        patch_segments,
        np.asarray(sizes, dtype=np.float64),
        np.asarray(rows, dtype=np.int64),
    )


def _canonical_tile_piece_skeleton(
    source: sources_owner.NativePlanarSource,
    schedule: PlanarTileSchedule,
    prepared: PreparedPlanarDomain,
    /,
) -> _TilePieceSkeleton:
    vertices = np.asarray(prepared.points, dtype=np.float64)
    edges = np.asarray(prepared.segments, dtype=np.int64)
    sources = np.asarray(prepared.segment_sources, dtype=np.int64)
    source_edge_loops = np.asarray(prepared.source_loops, dtype=np.int64)
    loops = np.where(sources >= 0, source_edge_loops[np.maximum(sources, 0)], -1)
    lower = np.min(np.asarray(prepared.source_points, dtype=np.float64), axis=0)
    upper = np.max(np.asarray(prepared.source_points, dtype=np.float64), axis=0)
    span = float(schedule.tile_span_sizes * prepared.region_size)
    if not math.isfinite(span) or span <= 0.0:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Planar tile spacing must be finite and positive.",
            provider_code="native_planar_tiles",
            stage=_STAGE,
        )
    shape = tuple(
        max(1, math.ceil(float(upper[axis] - lower[axis]) / span)) for axis in (0, 1)
    )
    lines = tuple(
        lower[axis] + span * np.arange(1, shape[axis], dtype=np.float64)
        for axis in (0, 1)
    )
    region_mask = loops >= 0
    wanted_orientation = np.where(loops == 0, 1, -1)
    offsets = np.asarray(source.region.loop_offsets, dtype=np.int64)
    region_edges = np.asarray(source.region.edges, dtype=np.int64)
    loop_areas = np.asarray(
        [
            _signed_loop_area(
                np.asarray(source.region.vertices, dtype=np.float64),
                region_edges[offsets[loop] : offsets[loop + 1]],
            )
            for loop in range(offsets.size - 1)
        ],
        dtype=np.float64,
    )
    loop_signs = np.where(loop_areas >= 0.0, 1, -1).astype(np.int64)
    forward = np.where(
        region_mask, loop_signs[np.maximum(loops, 0)] == wanted_orientation, True
    )
    skeleton = _SkeletonPoints()
    prepared_rows = np.empty((vertices.shape[0],), dtype=np.int64)
    for row in range(vertices.shape[0]):
        point = np.asarray(
            (
                np.float64(vertices[row, 0]),
                np.float64(vertices[row, 1]),
            ),
            dtype=np.float64,
        )
        if row < prepared.source_points.shape[0]:
            dimension, index, parameter = 0, row, math.nan
        else:
            dimension = 1
            index = int(prepared.point_sources[row])
            parameter = float(prepared.point_parameters[row])
        prepared_rows[row] = skeleton.add(point, dimension, index, parameter)
    cuts: list[list[tuple[float, int]]] = [[] for _ in range(edges.shape[0])]
    seam_intervals: list[tuple[int, float, int, int, int, int]] = []
    for axis in (0, 1):
        for line, value in enumerate(lines[axis]):
            intervals = _line_crossings(
                vertices,
                edges,
                prepared_rows,
                sources,
                loops,
                axis,
                float(value),
                lines[1 - axis],
                skeleton,
                cuts,
                prepared,
            )
            seam_intervals.extend(
                (axis, float(value), line, first, last, edge)
                for first, last, edge in intervals
            )
    seam_axes: list[int] = []
    seam_values: list[float] = []
    seam_sources: list[int] = []
    seam_tile_keys: list[tuple[tuple[int, int], tuple[int, int]]] = []
    piece_rows: list[tuple[int, int]] = []
    piece_sources: list[int] = []
    piece_seams: list[int] = []
    piece_tile_keys: list[tuple[tuple[int, int], tuple[int, int] | None]] = []
    for axis, value, line, first, last, source_edge in seam_intervals:
        across = 1 - axis
        start = np.asarray(skeleton.coordinates[first])
        finish = np.asarray(skeleton.coordinates[last])
        low, high = sorted((float(start[across]), float(finish[across])))
        corners = lines[across][(lines[across] > low) & (lines[across] < high)]
        stops = [first]
        for corner in corners:
            point = np.empty((2,), dtype=np.float64)
            point[axis], point[across] = value, corner
            if source_edge >= 0:
                a, b = prepared.source_points[prepared.source_edges[source_edge]]
                direction = b - a
                parameter = float(
                    np.dot(point - a, direction) / np.dot(direction, direction)
                )
                stops.append(skeleton.add(point, 1, source_edge, parameter))
            else:
                stops.append(skeleton.add(point, 2, 0, math.nan))
        stops.append(last)
        for begin, end in zip(stops[:-1], stops[1:], strict=True):
            first_point = np.asarray(skeleton.coordinates[begin])
            last_point = np.asarray(skeleton.coordinates[end])
            middle = 0.5 * (first_point + last_point)
            index = min(
                max(int(math.floor((middle[across] - lower[across]) / span)), 0),
                shape[across] - 1,
            )
            below = [0, 0]
            above = [0, 0]
            below[axis], below[across] = line, index
            above[axis], above[across] = line + 1, index
            seam = len(seam_axes)
            seam_axes.append(axis)
            seam_values.append(value)
            seam_sources.append(source_edge)
            seam_tile_keys.append(((below[0], below[1]), (above[0], above[1])))
            piece_rows.append((begin, end))
            piece_sources.append(source_edge)
            piece_seams.append(seam)
            piece_tile_keys.append(((below[0], below[1]), (above[0], above[1])))
    for segment, (first, last) in enumerate(edges.tolist()):
        starts = sorted(
            set(
                cuts[segment]
                + [(0.0, int(prepared_rows[first])), (1.0, int(prepared_rows[last]))]
            )
        )
        rows = [starts[0][1]]
        for _, point_row in starts[1:]:
            if point_row != rows[-1]:
                rows.append(point_row)
        for begin, end in zip(rows[:-1], rows[1:], strict=True):
            first_point = np.asarray(skeleton.coordinates[begin])
            last_point = np.asarray(skeleton.coordinates[end])
            middle = 0.5 * (first_point + last_point)
            tile = [0, 0]
            on_line = False
            for axis in (0, 1):
                collinear = first_point[axis] == last_point[axis] and np.any(
                    lines[axis] == first_point[axis]
                )
                if collinear:
                    on_line = True
                    line = int(np.flatnonzero(lines[axis] == first_point[axis])[0])
                    direction = (
                        last_point - first_point
                        if forward[segment]
                        else first_point - last_point
                    )
                    normal = direction[1 - axis] * (-1 if axis == 0 else 1)
                    tile[axis] = line if normal < 0 else line + 1
                else:
                    tile[axis] = min(
                        max(int(math.floor((middle[axis] - lower[axis]) / span)), 0),
                        shape[axis] - 1,
                    )
            if on_line and not region_mask[segment]:
                continue
            piece_rows.append((begin, end) if forward[segment] else (end, begin))
            piece_sources.append(int(sources[segment]))
            piece_seams.append(-1)
            piece_tile_keys.append(((tile[0], tile[1]), None))
    return _TilePieceSkeleton(
        origin=lower.astype(np.float64),
        span=span,
        lattice_shape=(int(shape[0]), int(shape[1])),
        loop_areas=loop_areas,
        source_edge_loops=source_edge_loops,
        coordinates=np.asarray(skeleton.coordinates, dtype=np.float64).reshape((-1, 2)),
        dimensions=np.asarray(skeleton.dimensions, dtype=np.int64),
        indices=np.asarray(skeleton.indices, dtype=np.int64),
        parameters=np.asarray(skeleton.parameters, dtype=np.float64),
        seam_axes=np.asarray(seam_axes, dtype=np.int64),
        seam_values=np.asarray(seam_values, dtype=np.float64),
        seam_sources=np.asarray(seam_sources, dtype=np.int64),
        seam_tile_keys=tuple(seam_tile_keys),
        piece_rows=np.asarray(piece_rows, dtype=np.int64).reshape((-1, 2)),
        piece_sources=np.asarray(piece_sources, dtype=np.int64),
        piece_seams=np.asarray(piece_seams, dtype=np.int64),
        piece_tile_keys=tuple(piece_tile_keys),
    )


def _build_tile_arrays(
    schedule: PlanarTileSchedule,
    prepared: PreparedPlanarDomain,
    /,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    float,
    tuple[int, int],
]:
    pieces = _canonical_tile_piece_skeleton(prepared.source, schedule, prepared)
    lower = pieces.origin
    span = pieces.span
    shape = pieces.lattice_shape
    loop_areas = pieces.loop_areas
    source_edge_loops = pieces.source_edge_loops
    coordinates = pieces.coordinates
    piece_rows = pieces.piece_rows
    piece_sources = pieces.piece_sources
    piece_seams = pieces.piece_seams
    piece_tile_keys = pieces.piece_tile_keys
    seam_axes = pieces.seam_axes
    seam_values = pieces.seam_values
    seam_sources = pieces.seam_sources
    seam_tile_keys = pieces.seam_tile_keys

    tile_key_set = {
        key
        for first, second in piece_tile_keys
        for key in (first, second)
        if key is not None
    }
    if not tile_key_set:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The planar tile complex contains no source constraints.",
            provider_code="native_planar_tiles",
            stage=_STAGE,
        )
    tile_keys = np.asarray(
        sorted(tile_key_set, key=lambda key: (key[1], key[0])), dtype=np.int64
    )
    tile_row = {tuple(map(int, key)): row for row, key in enumerate(tile_keys.tolist())}
    coordinates = pieces.coordinates
    piece_array = np.asarray(piece_rows, dtype=np.int64).reshape((-1, 2))
    piece_tile_rows = np.asarray(
        [
            (tile_row[first], -1 if second is None else tile_row[second])
            for first, second in piece_tile_keys
        ],
        dtype=np.int64,
    ).reshape((-1, 2))
    seam_rows = np.asarray(piece_seams, dtype=np.int64)
    source_rows = np.asarray(piece_sources, dtype=np.int64)
    weights = np.zeros((tile_keys.shape[0],), dtype=np.float64)
    endpoints = coordinates[piece_array]
    crosses = (
        endpoints[:, 0, 0] * endpoints[:, 1, 1] - endpoints[:, 0, 1] * endpoints[:, 1, 0]
    )
    region_piece = (seam_rows < 0) & (source_edge_loops[np.maximum(source_rows, 0)] >= 0)
    np.add.at(weights, piece_tile_rows[region_piece, 0], 0.5 * crosses[region_piece])
    seam_piece = seam_rows >= 0
    axes = np.asarray(seam_axes, dtype=np.int64)
    seam_increasing = np.where(
        axes[seam_rows[seam_piece]] == 0,
        endpoints[seam_piece, 0, 1] < endpoints[seam_piece, 1, 1],
        endpoints[seam_piece, 0, 0] < endpoints[seam_piece, 1, 0],
    )
    left_is_lower = seam_increasing == (axes[seam_rows[seam_piece]] == 0)
    lower_sign = np.where(left_is_lower, 1.0, -1.0)
    np.add.at(
        weights, piece_tile_rows[seam_piece, 0], 0.5 * lower_sign * crosses[seam_piece]
    )
    np.add.at(
        weights, piece_tile_rows[seam_piece, 1], -0.5 * lower_sign * crosses[seam_piece]
    )
    region_area = float(abs(loop_areas[0]) - np.sum(np.abs(loop_areas[1:])))
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            float(np.max(np.abs(prepared.source_points), initial=0.0)) ** 2,
        )
        * max(1, piece_array.shape[0])
    )
    if (
        np.any(weights <= 0.0)
        or not math.isfinite(region_area)
        or region_area <= 0.0
        or abs(float(np.sum(weights)) - region_area) > tolerance
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Planar tile areas do not form a positive partition of the source region.",
            provider_code="native_planar_tiles",
            stage=_STAGE,
        )

    piece_entity_ids = np.where(
        source_rows >= 0,
        prepared.source_edge_ids[np.maximum(source_rows, 0)],
        np.int64(-1),
    )
    piece_patch_segments, piece_patch_sizes, piece_patch_rows = _piece_patch_controls(
        prepared, source_rows
    )
    points, segments, owner, split_sources, split_parameters = split_constraints(
        coordinates,
        piece_array,
        piece_entity_ids,
        prepared.region_size if piece_patch_sizes.size else prepared.construction_size,
        piece_patch_segments,
        piece_patch_sizes,
        piece_patch_rows,
        prepared.gradation_slope,
    )
    dimensions = pieces.dimensions
    indices = pieces.indices
    parameters = pieces.parameters
    base = coordinates.shape[0]
    added = split_sources >= 0
    dimensions = np.concatenate(
        (dimensions, np.where(source_rows[split_sources[added]] >= 0, 1, 2))
    )
    indices = np.concatenate((indices, np.maximum(source_rows[split_sources[added]], 0)))
    parameters = np.concatenate(
        (parameters, np.full((np.count_nonzero(added),), math.nan, dtype=np.float64))
    )
    for offset, row in enumerate(np.flatnonzero(added).tolist()):
        piece = int(split_sources[row])
        edge = int(source_rows[piece])
        if edge < 0:
            continue
        first, last = piece_array[piece]
        low = _edge_parameter_from_metadata(
            prepared, edge, int(first), dimensions, indices, parameters
        )
        high = _edge_parameter_from_metadata(
            prepared, edge, int(last), dimensions, indices, parameters
        )
        parameters[base + offset] = low + float(split_parameters[row]) * (high - low)
    segment_tiles = piece_tile_rows[owner]
    point_tiles = np.full((points.shape[0],), np.iinfo(np.int64).max, dtype=np.int64)
    for column in (0, 1):
        for tile_column in (0, 1):
            valid = segment_tiles[:, tile_column] >= 0
            np.minimum.at(
                point_tiles,
                segments[valid, column],
                segment_tiles[valid, tile_column],
            )
    if np.any(point_tiles == np.iinfo(np.int64).max):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "A canonical planar constraint point is not incident to a tile.",
            provider_code="native_planar_tiles",
            stage=_STAGE,
        )
    seam_tile_rows = np.asarray(
        [(tile_row[first], tile_row[last]) for first, last in seam_tile_keys],
        dtype=np.int64,
    ).reshape((-1, 2))
    return (
        lower.astype(np.float64),
        tile_keys,
        weights,
        np.asarray(points, dtype=np.float64),
        dimensions,
        indices,
        parameters,
        point_tiles,
        np.asarray(segments, dtype=np.int64),
        source_rows[owner],
        seam_rows[owner],
        segment_tiles,
        axes,
        np.asarray(seam_values, dtype=np.float64),
        np.asarray(seam_sources, dtype=np.int64),
        seam_tile_rows,
        np.asarray(prepared.holes, dtype=np.float64).reshape((-1, 2)),
        span,
        (int(shape[0]), int(shape[1])),
    )


def _point_edge_parameter(
    prepared: PreparedPlanarDomain,
    edge: int,
    row: int,
    dimensions: NDArray[np.int64],
    indices: NDArray[np.int64],
    parameters: NDArray[np.float64],
    /,
) -> float:
    return _edge_parameter_from_metadata(
        prepared, edge, row, dimensions, indices, parameters
    )


def _seam_provenance(
    tiles: PlanarTileComplex, seam: int, point: NDArray[np.float64], /
) -> tuple[int, int, float]:
    edge = int(tiles.seam_sources[seam])
    if edge < 0:
        return 2, 0, math.nan
    first, last = tiles.prepared.source_points[tiles.prepared.source_edges[edge]]
    direction = last - first
    parameter = float(np.dot(point - first, direction) / np.dot(direction, direction))
    return 1, edge, parameter


def _seam_points_on_line(
    tiles: PlanarTileComplex, seam: int, extra: NDArray[np.float64], /
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rows = np.unique(tiles.segments[tiles.segment_seams == seam])
    axis = 1 - int(tiles.seam_axes[seam])
    values = np.concatenate((tiles.points[rows, axis], extra[:, axis]))
    return rows, np.argsort(values, kind="stable")


def _unique_edges(triangles: NDArray[np.int64], /) -> NDArray[np.int64]:
    sides = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]), axis=0
    )
    return np.unique(np.sort(sides, axis=1), axis=0)


def _triangulate_tile(
    tiles: PlanarTileComplex,
    row: int,
    seam_points: dict[int, NDArray[np.float64]],
    /,
) -> _TileMesh:
    tile_input = _build_tile_input(tiles, row, seam_points)
    triangulation, input_count = _refine_tile_constraints(tiles, tile_input)
    return _finalize_tile_mesh(tiles, tile_input, triangulation, input_count)


def _build_tile_input(
    tiles: PlanarTileComplex,
    row: int,
    seam_points: dict[int, NDArray[np.float64]],
    /,
) -> _TileInput:
    incident = (tiles.segment_tiles[:, 0] == row) | (tiles.segment_tiles[:, 1] == row)
    segment_rows = np.flatnonzero(incident)
    boundary = segment_rows[tiles.segment_seams[segment_rows] < 0]
    seams = np.unique(
        tiles.segment_seams[segment_rows][tiles.segment_seams[segment_rows] >= 0]
    )
    used = np.unique(tiles.segments[segment_rows])
    coordinates = [tiles.points[used]]
    kinds = [np.full((used.size,), _SKELETON, dtype=np.int64)]
    references = [used]
    input_seams = [np.full((used.size,), -1, dtype=np.int64)]
    local_segments = [np.searchsorted(used, tiles.segments[boundary])]
    segment_sources = [tiles.segment_sources[boundary]]
    segment_seams = [np.full((boundary.size,), -1, dtype=np.int64)]
    count = used.size
    for seam_value in seams.tolist():
        seam = int(seam_value)
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
        references.append(np.full((extra.shape[0],), seam, dtype=np.int64))
        input_seams.append(np.full((extra.shape[0],), seam, dtype=np.int64))
        count += extra.shape[0]
        pairs = np.stack((local[:-1], local[1:]), axis=1)
        local_segments.append(pairs)
        segment_sources.append(
            np.full((pairs.shape[0],), tiles.seam_sources[seam], dtype=np.int64)
        )
        segment_seams.append(np.full((pairs.shape[0],), seam, dtype=np.int64))
    low, high = tiles.tile_box(row)
    holes = tiles.holes[np.all((tiles.holes >= low) & (tiles.holes < high), axis=1)]
    quality = tiles.specification.quality_target
    minimum_angle = (
        0.0
        if quality is None or not quality.hard
        else math.degrees(quality.minimum_angle)
    )
    return _TileInput(
        points=np.concatenate(coordinates, axis=0),
        kinds=np.concatenate(kinds),
        references=np.concatenate(references),
        seam_rows=np.concatenate(input_seams),
        segments=np.concatenate(local_segments).astype(np.int64, copy=False),
        sources=np.concatenate(segment_sources).astype(np.int64, copy=False),
        segment_seams=np.concatenate(segment_seams).astype(np.int64, copy=False),
        holes=holes,
        minimum_angle_degrees=minimum_angle,
    )


def _refine_tile_constraints(
    tiles: PlanarTileComplex, tile_input: _TileInput, /
) -> tuple[ConstrainedDelaunayTriangulation, int]:
    extra_points = np.empty((0, 2), dtype=np.float64)
    while True:
        triangulation = ConstrainedDelaunayTriangulation(
            np.concatenate((tile_input.points, extra_points), axis=0),
            tile_input.segments,
            holes=tile_input.holes if tile_input.holes.shape[0] else None,
            keep_convex_hull=False,
            min_angle=tile_input.minimum_angle_degrees,
        )
        if str(triangulation.evidence.status) != "ok":
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Native constrained triangulation returned {triangulation.evidence.status!s}.",
                provider_code="native_planar_tiles",
                stage=f"{_STAGE}-triangulation",
            )
        points = np.asarray(triangulation.points, dtype=np.float64)
        triangles = np.asarray(triangulation.triangles, dtype=np.int64)
        if triangles.shape[0] == 0:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "A source-owned planar tile produced no triangles.",
                provider_code="native_planar_tiles",
                stage=f"{_STAGE}-triangulation",
            )
        edges = _unique_edges(triangles)
        constrained, _ = constrained_edges(
            triangles,
            np.asarray(triangulation.segment_ids, dtype=np.int64),
            tile_input.sources,
        )
        constrained_set = {tuple(pair) for pair in constrained.tolist()}
        free = np.fromiter(
            (tuple(pair) not in constrained_set for pair in edges.tolist()),
            dtype=np.bool_,
            count=edges.shape[0],
        )
        endpoints = points[edges]
        lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
        midpoints = 0.5 * (endpoints[:, 0] + endpoints[:, 1])
        samples = tiles.prepared.sizes(
            np.concatenate((endpoints[:, 0], endpoints[:, 1], midpoints))
        )
        local_sizes = np.min(samples.reshape((3, edges.shape[0])), axis=0)
        long = free & (lengths > local_sizes)
        if not np.any(long):
            return triangulation, tile_input.points.shape[0] + extra_points.shape[0]
        selected = edges[long]
        new_points = np.unique(
            0.5 * (points[selected[:, 0]] + points[selected[:, 1]]), axis=0
        )
        existing_points = {tuple(point) for point in points.tolist()}
        if any(tuple(point) in existing_points for point in new_points.tolist()):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Planar size refinement reached coordinate representability.",
                provider_code="native_planar_tiles",
                stage=f"{_STAGE}-refinement",
            )
        extra_points = np.concatenate((extra_points, new_points), axis=0)


def _finalize_tile_mesh(
    tiles: PlanarTileComplex,
    tile_input: _TileInput,
    triangulation: ConstrainedDelaunayTriangulation,
    input_count: int,
    /,
) -> _TileMesh:
    points = np.asarray(triangulation.points, dtype=np.float64)
    triangles = np.asarray(triangulation.triangles, dtype=np.int64)
    count = points.shape[0]
    constraint_sources = np.asarray(triangulation.constraint_sources, dtype=np.int64)
    parents = np.asarray(triangulation.constraint_parents, dtype=np.int64)
    ratios = np.asarray(triangulation.constraint_ratios, dtype=np.float64)
    out_kinds = np.full((count,), _LOCAL, dtype=np.int64)
    out_references = np.full((count,), -1, dtype=np.int64)
    dimensions = np.full((count,), 2, dtype=np.int64)
    indices = np.zeros((count,), dtype=np.int64)
    parameters = np.full((count,), math.nan, dtype=np.float64)
    initial_count = tile_input.points.shape[0]
    out_kinds[:initial_count] = tile_input.kinds
    out_references[:initial_count] = tile_input.references
    skeleton = tile_input.kinds == _SKELETON
    skeleton_rows = tile_input.references[skeleton]
    dimensions[:initial_count][skeleton] = tiles.point_dimensions[skeleton_rows]
    indices[:initial_count][skeleton] = tiles.point_indices[skeleton_rows]
    parameters[:initial_count][skeleton] = tiles.point_parameters[skeleton_rows]
    point_seams = np.full((count,), -1, dtype=np.int64)
    point_seams[:initial_count] = tile_input.seam_rows
    for vertex in np.flatnonzero(tile_input.kinds == _SEAM).tolist():
        dimension, index, parameter = _seam_provenance(
            tiles, int(tile_input.seam_rows[vertex]), tile_input.points[vertex]
        )
        dimensions[vertex], indices[vertex], parameters[vertex] = (
            dimension,
            index,
            parameter,
        )
    new_seam: dict[int, list[NDArray[np.float64]]] = {}
    for vertex in range(input_count, count):
        segment = int(constraint_sources[vertex])
        if segment in (-2, -1):
            continue
        seam = int(tile_input.segment_seams[segment])
        if seam >= 0:
            out_kinds[vertex] = _SEAM
            point_seams[vertex] = seam
            new_seam.setdefault(seam, []).append(points[vertex])
            dimension, index, parameter = _seam_provenance(tiles, seam, points[vertex])
            dimensions[vertex], indices[vertex], parameters[vertex] = (
                dimension,
                index,
                parameter,
            )
            continue
        edge = int(tile_input.sources[segment])
        if edge < 0:
            continue
        first, last = parents[vertex]
        t0 = _point_edge_parameter(
            tiles.prepared, edge, int(first), dimensions, indices, parameters
        )
        t1 = _point_edge_parameter(
            tiles.prepared, edge, int(last), dimensions, indices, parameters
        )
        dimensions[vertex], indices[vertex] = 1, edge
        parameters[vertex] = t0 + float(ratios[vertex]) * (t1 - t0)
    referenced = np.zeros((count,), dtype=np.bool_)
    referenced[triangles.reshape(-1)] = True
    if np.any(referenced & (constraint_sources == -2)):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A tile triangulation retained a split from an open constraint hull.",
            provider_code="native_planar_tiles",
            stage=f"{_STAGE}-triangulation",
        )
    if np.any(~referenced[:initial_count][tile_input.kinds != _LOCAL]):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A tile triangulation dropped a canonical constraint point.",
            provider_code="native_planar_tiles",
            stage=f"{_STAGE}-triangulation",
        )
    corners = points[triangles]
    orientation = exact_orient2d(corners[:, 0], corners[:, 1], corners[:, 2])
    if np.any(orientation == 0):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A tile triangulation contains a degenerate triangle.",
            provider_code="native_planar_tiles",
            stage=f"{_STAGE}-triangulation",
        )
    triangles = np.where(orientation[:, None] < 0, triangles[:, ::-1], triangles)
    segment_ids = np.where(
        orientation[:, None] < 0,
        np.asarray(triangulation.segment_ids)[:, ::-1],
        np.asarray(triangulation.segment_ids),
    )
    carried = segment_ids >= 0
    triangle_rows, opposite = np.nonzero(carried)
    first = triangles[triangle_rows, (opposite + 1) % 3]
    last = triangles[triangle_rows, (opposite + 2) % 3]
    edge_pairs = np.sort(np.stack((first, last), axis=1), axis=1)
    carried_segments = segment_ids[triangle_rows, opposite]
    edge_pairs, unique_rows = np.unique(edge_pairs, axis=0, return_index=True)
    carried_segments = carried_segments[unique_rows]
    return _TileMesh(
        points=points,
        triangles=triangles,
        kinds=out_kinds,
        references=np.where(out_kinds == _SEAM, point_seams, out_references),
        dimensions=dimensions,
        indices=indices,
        parameters=parameters,
        edge_vertices=edge_pairs.astype(np.int64, copy=False).reshape((-1, 2)),
        edge_sources=tile_input.sources[carried_segments],
        edge_seams=tile_input.segment_seams[carried_segments],
        seam_points={
            seam: np.asarray(values, dtype=np.float64).reshape((-1, 2))
            for seam, values in new_seam.items()
        },
        minimum_angle_degrees=float(triangulation.evidence.minimum_angle_degrees),
    )


def _merge_seam(
    current: NDArray[np.float64], added: NDArray[np.float64], axis: int, /
) -> NDArray[np.float64]:
    if added.shape[0] == 0:
        return current
    union = np.unique(np.concatenate((current, added), axis=0), axis=0)
    return union[np.argsort(union[:, 1 - axis], kind="stable")]


def _collective_failure_barrier(
    error: Exception | None,
    execution_group: ExecutionGroup,
    stage: str,
    /,
) -> None:
    failures = _owner_group_allgather(
        np.asarray((int(error is not None),), dtype=np.int32),
        execution_group,
        f"planar_{stage.replace('-', '_')}",
    )
    if np.any(failures):
        if error is not None:
            raise error
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            _REMOTE_REFUSAL,
            provider_code="native_planar_distributed",
            stage=stage,
        )


def _agree_identity(
    identity: str, execution_group: ExecutionGroup, stage: str, /
) -> None:
    digest = np.frombuffer(bytes.fromhex(identity), dtype=np.uint8)
    gathered = _owner_group_allgather(
        digest, execution_group, f"planar_{stage.replace('-', '_')}_identity"
    )
    if not np.all(gathered == gathered[0]):
        raise ValueError(f"Every execution-group process must use the same {stage}.")


def _group_rank(execution_group: ExecutionGroup, /) -> int:
    processes = execution_group.spec.process_indices
    if tuple(device.process_index for device in execution_group.devices) != processes:
        raise ValueError(
            "Planar owner ranks require one group device per process in process-index order."
        )
    process = jax.process_index()
    if process not in processes:
        raise ValueError("The current process is not a member of execution_group.")
    return processes.index(process)


def _tile_point_incidence(tiles: PlanarTileComplex, /) -> tuple[NDArray[np.int64], ...]:
    incident: list[set[int]] = [set() for _ in range(tiles.points.shape[0])]
    for segment in range(tiles.segments.shape[0]):
        for tile_column in range(tiles.segment_tiles.shape[1]):
            tile = int(tiles.segment_tiles[segment, tile_column])
            if tile < 0:
                continue
            for point_column in range(tiles.segments.shape[1]):
                point = int(tiles.segments[segment, point_column])
                incident[point].add(tile)
    return tuple(np.asarray(sorted(rows), dtype=np.int64) for rows in incident)


def _tile_point_rows(
    point_incidence: tuple[NDArray[np.int64], ...], tile_count: int, /
) -> tuple[NDArray[np.int64], ...]:
    rows: list[list[int]] = [[] for _ in range(tile_count)]
    for point, incident in enumerate(point_incidence):
        for tile in incident.tolist():
            rows[tile].append(point)
    return tuple(np.asarray(values, dtype=np.int64) for values in rows)


def _partition_tiles(
    tiles: PlanarTileComplex,
    execution_group: ExecutionGroup,
    partition_plan: GraphPartitionPlan,
    /,
) -> tuple[
    DistributedGraphPartitionResult,
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int32],
]:
    rank = _group_rank(execution_group)
    part_count = len(execution_group.spec.process_indices)
    if partition_plan.part_count != part_count:
        raise ValueError(
            "partition_plan.part_count must equal execution_group process count."
        )
    if len(execution_group.devices) != part_count:
        raise ValueError(
            "Planar graph partitioning requires one group device per process."
        )
    tile_count = tiles.tiles.shape[0]
    initial_owners = np.arange(tile_count, dtype=np.int64) % part_count
    adjacency: list[set[int]] = [set() for _ in range(tile_count)]
    for first, second in tiles.seam_tiles.tolist():
        adjacency[first].add(second)
        adjacency[second].add(first)
    local_rows = np.flatnonzero(initial_owners == rank)
    offsets = np.zeros((local_rows.size + 1,), dtype=np.int64)
    neighbor_ids: list[int] = []
    neighbor_owners: list[int] = []
    for local_row, tile_row in enumerate(local_rows.tolist()):
        for neighbor in sorted(adjacency[tile_row]):
            neighbor_ids.append(int(tiles.tile_ids[neighbor]))
            neighbor_owners.append(int(initial_owners[neighbor]))
        offsets[local_row + 1] = len(neighbor_ids)
    graph: DistributedWeightedCSRGraph | None = None
    local_error: Exception | None = None
    try:
        graph = DistributedWeightedCSRGraph(
            offsets,
            np.asarray(neighbor_ids, dtype=np.int64),
            entity_ids=tiles.tile_ids[local_rows],
            owner_ranks=initial_owners[local_rows].astype(np.int32),
            neighbor_owner_ranks=np.asarray(neighbor_owners, dtype=np.int32),
            # Fixed-span tiles use exact count weights; physical area remains
            # a separate source-coverage invariant.
            vertex_weights=np.ones((local_rows.size,), dtype=np.int64),
        )
    except Exception as error:
        local_error = error
    _collective_failure_barrier(
        local_error, execution_group, f"{_STAGE}-graph-preparation"
    )
    if graph is None:
        raise RuntimeError("Collective tile-graph preparation produced no local graph.")
    partition = partition_graph_distributed(
        graph, partition_plan, execution_group, axis_name="planar_graph"
    )
    point_incidence = _tile_point_incidence(tiles)
    tile_points = _tile_point_rows(point_incidence, tile_count)
    local_error: Exception | None = None
    local_assignment: dict[int, int] = {}
    try:
        graph_ids = np.asarray(partition.entity_ids, dtype=np.int64)
        graph_owners = np.asarray(partition.owners, dtype=np.int32)
        local_assignment = {
            int(tile): int(owner)
            for tile, owner in zip(graph_ids, graph_owners, strict=True)
        }
        expected_ids = set(tiles.tile_ids[local_rows].tolist())
        if set(local_assignment) != expected_ids:
            raise ValueError(
                "Graph partition labels differ from the locally owned tile IDs."
            )
    except Exception as error:
        local_error = error
    _collective_failure_barrier(local_error, execution_group, f"{_STAGE}-graph-labels")

    outgoing: list[list[tuple[int, int]]] = [[] for _ in range(part_count)]
    local_error = None
    try:
        for tile_row in local_rows.tolist():
            tile_id = int(tiles.tile_ids[tile_row])
            owner = local_assignment[tile_id]
            destinations = {owner}
            for point in tile_points[tile_row].tolist():
                destinations.update(
                    int(initial_owners[incident_tile])
                    for incident_tile in point_incidence[point].tolist()
                )
            for destination in destinations:
                if destination != rank:
                    outgoing[destination].append((tile_id, owner))
    except Exception as error:
        local_error = error
    _collective_failure_barrier(local_error, execution_group, f"{_STAGE}-owner-routing")
    owner_banks = tuple(
        np.asarray(sorted(set(rows)), dtype=np.int64).reshape((-1, 2))
        for rows in outgoing
    )
    incoming = _owner_exchange_message_banks(
        owner_banks, rank, execution_group, "planar_tile_owners"
    )
    known: dict[int, int] = dict(local_assignment)
    next_outgoing: list[set[tuple[int, int]]] = [set() for _ in range(part_count)]
    local_error = None
    try:
        for records in incoming.values():
            for tile_value, owner_value in np.asarray(records, dtype=np.int64):
                tile, owner = int(tile_value), int(owner_value)
                prior = known.get(tile)
                if prior is not None and prior != owner:
                    raise ValueError(
                        "Conflicting graph-partition owner labels for a tile."
                    )
                known[tile] = owner
        for tile_row in local_rows.tolist():
            tile_id = int(tiles.tile_ids[tile_row])
            final_owner = local_assignment[tile_id]
            if final_owner == rank:
                continue
            related: set[int] = {tile_id}
            for point in tile_points[tile_row].tolist():
                related.update(
                    int(tiles.tile_ids[value])
                    for value in point_incidence[point].tolist()
                )
            for related_tile in related:
                if related_tile not in known:
                    raise ValueError(
                        "Owner directory is missing a tile sharing a constraint point."
                    )
                next_outgoing[final_owner].add((related_tile, known[related_tile]))
    except Exception as error:
        local_error = error
    _collective_failure_barrier(
        local_error, execution_group, f"{_STAGE}-owner-forwarding"
    )
    forwarded = tuple(
        np.asarray(sorted(rows), dtype=np.int64).reshape((-1, 2))
        for rows in next_outgoing
    )
    final_incoming = _owner_exchange_message_banks(
        forwarded, rank, execution_group, "planar_tile_owner_directory"
    )
    local_error = None
    try:
        for records in final_incoming.values():
            for tile_value, owner_value in np.asarray(records, dtype=np.int64):
                tile, owner = int(tile_value), int(owner_value)
                prior = known.get(tile)
                if prior is not None and prior != owner:
                    raise ValueError(
                        "Forwarded tile owner directory disagrees with graph labels."
                    )
                known[tile] = owner
    except Exception as error:
        local_error = error
    _collective_failure_barrier(local_error, execution_group, f"{_STAGE}-owner-directory")
    known_ids = np.asarray(sorted(known), dtype=np.int64)
    known_owners = np.asarray([known[int(tile)] for tile in known_ids], dtype=np.int32)
    local_final = np.asarray(
        [
            row
            for row, tile_id in enumerate(tiles.tile_ids.tolist())
            if known.get(int(tile_id)) == rank
        ],
        dtype=np.int64,
    )
    return partition, local_final, known_ids, known_owners


@checked
def prepare_distributed_planar_generation(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    schedule: PlanarTileSchedule,
    execution_group: ExecutionGroup,
    /,
    *,
    partition_plan: GraphPartitionPlan | None = None,
) -> PreparedDistributedPlanarGeneration:
    """Prepare canonical tiles and assign them with native owner-local graph partitioning."""
    _group_rank(execution_group)
    plan = (
        GraphPartitionPlan(
            len(execution_group.spec.process_indices), require_nonempty=False
        )
        if partition_plan is None
        else partition_plan
    )
    local_error: Exception | None = None
    tiles: PlanarTileComplex | None = None
    try:
        tiles = PlanarTileComplex(source, specification, schedule)
        if plan.part_count != len(execution_group.spec.process_indices):
            raise ValueError(
                "partition_plan.part_count must equal execution_group process count."
            )
        if len(execution_group.devices) != len(execution_group.spec.process_indices):
            raise ValueError(
                "Planar graph partitioning requires one group device per process."
            )
    except Exception as error:
        local_error = error
    _collective_failure_barrier(local_error, execution_group, f"{_STAGE}-preparation")
    if tiles is None:
        raise RuntimeError(
            "Collective planar preparation completed without a tile complex."
        )
    binding = canonical_fingerprint(
        (
            source.binding_id,
            specification.specification_id,
            schedule.schedule_id,
            tiles.compiled_id,
            plan.plan_id,
        )
    )
    _agree_identity(binding, execution_group, f"{_STAGE}-inputs")
    partition_result, local_rows, known_ids, known_owners = _partition_tiles(
        tiles, execution_group, plan
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-distributed-planar-generation",
            "tiles": tiles.compiled_id,
            "execution_group": execution_group.spec.group_id,
            "partition": partition_result.result_id,
        }
    )
    return PreparedDistributedPlanarGeneration(
        tiles,
        execution_group,
        plan,
        partition_result,
        _frozen(local_rows),
        _frozen(known_ids),
        _frozen(known_owners),
        prepared_id,
    )


def _owner_lookup(prepared: PreparedDistributedPlanarGeneration, /) -> dict[int, int]:
    return {
        int(tile): int(owner)
        for tile, owner in zip(
            prepared.known_tile_ids, prepared.known_tile_owners, strict=True
        )
    }


def _tile_owner_by_row(
    prepared: PreparedDistributedPlanarGeneration, /
) -> dict[int, int]:
    owner_by_id = _owner_lookup(prepared)
    return {
        row: owner_by_id[int(tile_id)]
        for row, tile_id in enumerate(prepared.tiles.tile_ids.tolist())
        if int(tile_id) in owner_by_id
    }


def _exchange_seam_points(
    outgoing: list[list[tuple[int, NDArray[np.float64]]]],
    rank: int,
    execution_group: ExecutionGroup,
    /,
) -> dict[int, NDArray[np.float64]]:
    banks: list[NDArray[np.int64]] = []
    for owner, rows in enumerate(outgoing):
        if owner == rank:
            banks.append(np.empty((0, 3), dtype=np.int64))
            continue
        if rows:
            points = np.asarray([row[1] for row in rows], dtype=np.float64).reshape(
                (-1, 2)
            )
            packet = np.column_stack(
                (
                    np.asarray([row[0] for row in rows], dtype=np.int64),
                    points.view(np.int64),
                )
            )
            packet = np.unique(packet, axis=0)
        else:
            packet = np.empty((0, 3), dtype=np.int64)
        banks.append(packet)
    incoming = _owner_exchange_message_banks(
        tuple(banks), rank, execution_group, "planar_seam_points"
    )
    merged: dict[int, list[NDArray[np.float64]]] = {}
    for records in incoming.values():
        packed = np.asarray(records, dtype=np.int64)
        for seam in np.unique(packed[:, 0]).tolist():
            merged.setdefault(int(seam), []).append(
                packed[packed[:, 0] == seam, 1:].copy().view(np.float64)
            )
    return {
        seam: np.concatenate(values, axis=0).reshape((-1, 2))
        for seam, values in merged.items()
    }


def _seam_owner(
    tiles: PlanarTileComplex, seam: int, owner_by_tile: dict[int, int], /
) -> int:
    return owner_by_tile[int(np.min(tiles.seam_tiles[seam]))]


def _cell_order(
    triangles: NDArray[np.int64], vertex_ids: NDArray[np.int64], /
) -> NDArray[np.int64]:
    keys = np.sort(vertex_ids[triangles], axis=1)
    return np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))


def _local_numbering_counts(
    prepared: PreparedDistributedPlanarGeneration,
    results: dict[int, _TileMesh],
    seam_points: dict[int, NDArray[np.float64]],
    owners: dict[int, int],
    rank: int,
    /,
) -> tuple[dict[int, NDArray[np.int64]], NDArray[np.int64]]:
    tiles = prepared.tiles
    tile_count = tiles.tiles.shape[0]
    seam_count = tiles.seam_axes.shape[0]
    counts = np.zeros((2 * tile_count + seam_count,), dtype=np.int64)
    tile_locals: dict[int, NDArray[np.int64]] = {}
    for row in prepared.local_tile_rows.tolist():
        mesh = results[row]
        referenced = np.zeros((mesh.points.shape[0],), dtype=np.bool_)
        referenced[mesh.triangles.reshape(-1)] = True
        candidates = np.flatnonzero((mesh.kinds == _LOCAL) & referenced)
        order = np.lexsort((mesh.points[candidates, 1], mesh.points[candidates, 0]))
        tile_locals[row] = candidates[order]
        counts[row] = candidates.size
        counts[tile_count + row] = mesh.triangles.shape[0]
    for seam in range(seam_count):
        owner_tile = int(np.min(tiles.seam_tiles[seam]))
        if owner_tile in owners and owners[owner_tile] == rank:
            counts[2 * tile_count + seam] = seam_points.get(
                seam, np.empty((0, 2), dtype=np.float64)
            ).shape[0]
    return tile_locals, counts


def _number_owner_rows(
    prepared: PreparedDistributedPlanarGeneration,
    results: dict[int, _TileMesh],
    seam_points: dict[int, NDArray[np.float64]],
    local: NDArray[np.int64],
    tile_locals: dict[int, NDArray[np.int64]],
    owners: dict[int, int],
    seam_offsets: NDArray[np.int64],
    local_offsets: NDArray[np.int64],
    cell_offsets: NDArray[np.int64],
    /,
) -> _NumberedTileRows:
    tiles = prepared.tiles
    identifiers: dict[int, tuple[NDArray[np.float64], int, int, int, float]] = {}
    all_triangles: list[NDArray[np.int64]] = []
    all_cell_ids: list[NDArray[np.int64]] = []
    all_cell_tiles: list[NDArray[np.int64]] = []
    edge_records: dict[tuple[int, int], tuple[int, int]] = {}
    minimum_angle = math.inf
    for row in local.tolist():
        mesh = results[row]
        vertex_ids = np.full((mesh.points.shape[0],), -1, dtype=np.int64)
        vertex_owners = np.full((mesh.points.shape[0],), -1, dtype=np.int32)
        skeleton = mesh.kinds == _SKELETON
        skeleton_rows = mesh.references[skeleton]
        vertex_ids[skeleton] = tiles.point_ids[skeleton_rows]
        for local_point, skeleton_row in zip(
            np.flatnonzero(skeleton).tolist(), skeleton_rows.tolist(), strict=True
        ):
            owner_tile = int(tiles.point_tiles[skeleton_row])
            if owner_tile not in owners:
                raise ValueError(
                    "Owner directory omitted a canonical skeleton-point owner."
                )
            vertex_owners[local_point] = owners[owner_tile]
        for seam_value in np.unique(mesh.references[mesh.kinds == _SEAM]).tolist():
            seam = int(seam_value)
            selected = np.flatnonzero((mesh.kinds == _SEAM) & (mesh.references == seam))
            ordered = seam_points.get(seam, np.empty((0, 2), dtype=np.float64))
            axis = 1 - int(tiles.seam_axes[seam])
            if ordered.shape[0] == 0:
                raise ValueError(
                    "A tile seam vertex is absent from the converged seam set."
                )
            positions = np.minimum(
                np.searchsorted(ordered[:, axis], mesh.points[selected, axis]),
                ordered.shape[0] - 1,
            )
            if not np.array_equal(ordered[positions], mesh.points[selected]):
                raise ValueError(
                    "A tile seam vertex is absent from the converged seam set."
                )
            vertex_ids[selected] = seam_offsets[seam] + positions
            vertex_owners[selected] = _seam_owner(tiles, seam, owners)
        tile_local = tile_locals[row]
        vertex_ids[tile_local] = local_offsets[row] + np.arange(
            tile_local.size, dtype=np.int64
        )
        vertex_owners[tile_local] = owners[row]
        for vertex in np.unique(mesh.triangles).tolist():
            identifier = int(vertex_ids[vertex])
            dimension = int(mesh.dimensions[vertex])
            value = (
                mesh.points[vertex],
                int(vertex_owners[vertex]),
                dimension,
                _semantic_source_index(
                    tiles.prepared, dimension, int(mesh.indices[vertex])
                ),
                float(mesh.parameters[vertex]),
            )
            previous = identifiers.get(identifier)
            same_parameter = previous is not None and (
                previous[4] == value[4]
                or math.isnan(previous[4])
                and math.isnan(value[4])
            )
            if identifier < 0 or (
                previous is not None
                and (
                    not np.array_equal(previous[0], value[0])
                    or previous[1:4] != value[1:4]
                    or not same_parameter
                )
            ):
                raise ValueError(
                    "Shared planar vertex identity/provenance is inconsistent."
                )
            identifiers[identifier] = value
        order = _cell_order(mesh.triangles, vertex_ids)
        ordered_triangles = mesh.triangles[order].copy()
        ordered_ids = vertex_ids[ordered_triangles]
        corners = mesh.points[ordered_triangles]
        orientation = exact_orient2d(corners[:, 0], corners[:, 1], corners[:, 2])
        ordered_triangles[orientation < 0] = ordered_triangles[orientation < 0, ::-1]
        ordered_ids[orientation < 0] = ordered_ids[orientation < 0, ::-1]
        for cell in range(ordered_triangles.shape[0]):
            pivot = int(np.argmin(ordered_ids[cell]))
            ordered_triangles[cell] = np.roll(ordered_triangles[cell], -pivot)
        all_triangles.append(vertex_ids[ordered_triangles])
        all_cell_ids.append(
            cell_offsets[row] + np.arange(ordered_triangles.shape[0], dtype=np.int64)
        )
        all_cell_tiles.append(np.full((ordered_triangles.shape[0],), row, dtype=np.int64))
        for edge_row in range(mesh.edge_vertices.shape[0]):
            first_local = int(mesh.edge_vertices[edge_row, 0])
            last_local = int(mesh.edge_vertices[edge_row, 1])
            first_id = int(vertex_ids[first_local])
            last_id = int(vertex_ids[last_local])
            pair_ids = (min(first_id, last_id), max(first_id, last_id))
            source_row = int(mesh.edge_sources[edge_row])
            seam = int(mesh.edge_seams[edge_row])
            source_id = (
                -1 if source_row < 0 else int(tiles.prepared.source_edge_ids[source_row])
            )
            value = (
                source_id,
                -1 if seam < 0 else int(tiles.seam_ids[seam]),
            )
            previous = edge_records.get(pair_ids)
            if previous is not None and previous != value:
                raise ValueError(
                    "A constrained edge has conflicting source/seam identity."
                )
            edge_records[pair_ids] = value
        minimum_angle = min(minimum_angle, mesh.minimum_angle_degrees)
    return _NumberedTileRows(
        identifiers,
        all_triangles,
        all_cell_ids,
        all_cell_tiles,
        edge_records,
        minimum_angle,
    )


def _number_rows(
    prepared: PreparedDistributedPlanarGeneration,
    results: dict[int, _TileMesh],
    seam_points: dict[int, NDArray[np.float64]],
    rounds: int,
    /,
) -> DistributedPlanarConstruction:
    tiles = prepared.tiles
    group = prepared.execution_group
    rank = _group_rank(group)
    owners = _tile_owner_by_row(prepared)
    local = np.asarray(prepared.local_tile_rows, dtype=np.int64)
    local_error: Exception | None = None
    tile_locals: dict[int, NDArray[np.int64]] = {}
    counts = np.empty((0,), dtype=np.int64)
    try:
        tile_locals, counts = _local_numbering_counts(
            prepared, results, seam_points, owners, rank
        )
    except Exception as error:
        local_error = error
    _collective_failure_barrier(local_error, group, f"{_STAGE}-numbering-counts")
    tile_count = tiles.tiles.shape[0]
    seam_count = tiles.seam_axes.shape[0]
    gathered = _owner_group_allgather(counts, group, "planar_numbering_counts")
    totals = np.sum(np.asarray(gathered, dtype=np.int64), axis=0)
    local_counts = totals[:tile_count]
    cell_counts = totals[tile_count : 2 * tile_count]
    seam_counts = totals[2 * tile_count :]
    skeleton_count = tiles.point_ids.size
    seam_offsets = np.empty((seam_count,), dtype=np.int64)
    seam_cursor = skeleton_count
    for seam in np.argsort(tiles.seam_ids).tolist():
        seam_offsets[seam] = seam_cursor
        seam_cursor += int(seam_counts[seam])
    seam_total = seam_cursor - skeleton_count
    local_offsets = (
        skeleton_count
        + seam_total
        + np.concatenate(
            (
                np.asarray((0,), dtype=np.int64),
                np.cumsum(local_counts[:-1], dtype=np.int64),
            )
        )
    )
    cell_offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum(cell_counts[:-1], dtype=np.int64))
    )
    vertex_total = skeleton_count + seam_total + int(np.sum(local_counts))
    cell_total = int(np.sum(cell_counts))
    local_error = None
    numbered: _NumberedTileRows | None = None
    try:
        numbered = _number_owner_rows(
            prepared,
            results,
            seam_points,
            local,
            tile_locals,
            owners,
            seam_offsets,
            local_offsets,
            cell_offsets,
        )
    except Exception as error:
        local_error = error
    _collective_failure_barrier(local_error, group, f"{_STAGE}-numbering-rows")
    if numbered is None:
        raise RuntimeError("Collective planar numbering produced no local rows.")

    local_error = None
    topology_digest: str | None = None
    try:
        vertex_ids = np.asarray(sorted(numbered.identifiers), dtype=np.int64)
        vertices = [
            numbered.identifiers[int(identifier)] for identifier in vertex_ids.tolist()
        ]
        coordinates = np.asarray(
            [value[0] for value in vertices], dtype=np.float64
        ).reshape((-1, 2))
        vertex_owners = np.asarray([value[1] for value in vertices], dtype=np.int32)
        dimensions = np.asarray([value[2] for value in vertices], dtype=np.int64)
        indices = np.asarray([value[3] for value in vertices], dtype=np.int64)
        parameters = np.asarray([value[4] for value in vertices], dtype=np.float64)
        triangle_ids = (
            np.concatenate(numbered.triangles, axis=0)
            if numbered.triangles
            else np.empty((0, 3), dtype=np.int64)
        )
        cell_ids = (
            np.concatenate(numbered.cell_ids)
            if numbered.cell_ids
            else np.empty((0,), dtype=np.int64)
        )
        cell_tiles = (
            tiles.tile_ids[np.concatenate(numbered.cell_tiles)]
            if numbered.cell_tiles
            else np.empty((0,), dtype=np.int64)
        )
        local_triangles = np.searchsorted(vertex_ids, triangle_ids).astype(np.int64)
        edge_id_pairs = np.asarray(sorted(numbered.edge_records), dtype=np.int64).reshape(
            (-1, 2)
        )
        edge_vertices = np.searchsorted(vertex_ids, edge_id_pairs).astype(np.int64)
        edge_sources = np.asarray(
            [
                numbered.edge_records[
                    (int(edge_id_pairs[row, 0]), int(edge_id_pairs[row, 1]))
                ][0]
                for row in range(edge_id_pairs.shape[0])
            ],
            dtype=np.int64,
        )
        edge_seams = np.asarray(
            [
                numbered.edge_records[
                    (int(edge_id_pairs[row, 0]), int(edge_id_pairs[row, 1]))
                ][1]
                for row in range(edge_id_pairs.shape[0])
            ],
            dtype=np.int64,
        )
        topology_digest = canonical_fingerprint(
            array_tree_fingerprint(
                (
                    coordinates,
                    vertex_ids,
                    vertex_owners,
                    dimensions,
                    indices,
                    parameters,
                    local_triangles,
                    cell_ids,
                    cell_tiles,
                    edge_vertices,
                    edge_sources,
                    edge_seams,
                )
            )
        )
    except Exception as error:
        local_error = error
    _collective_failure_barrier(local_error, group, f"{_STAGE}-numbering-identity")
    if topology_digest is None:
        raise RuntimeError("Collective planar identity preparation produced no digest.")
    global_digests = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(topology_digest), dtype=np.uint8),
        group,
        "planar_topology_identity",
    )
    minimum_angle = float(
        np.min(
            _owner_group_allgather(
                np.asarray((numbered.minimum_angle_degrees,), dtype=np.float64),
                group,
                "planar_minimum_angle",
            )
        )
    )
    construction_id = canonical_fingerprint(
        {
            "kind": "distributed-planar-construction",
            "prepared": prepared.prepared_id,
            "partition": prepared.partition_result.result_id,
            "counts": array_tree_fingerprint(totals),
            "owner_topologies": np.asarray(global_digests, dtype=np.uint8).tolist(),
            "seam_rounds": rounds,
        }
    )
    return DistributedPlanarConstruction(
        coordinates=coordinates,
        vertex_ids=vertex_ids,
        vertex_owners=vertex_owners,
        vertex_dimensions=dimensions,
        vertex_indices=indices,
        vertex_parameters=parameters,
        triangles=local_triangles,
        cell_ids=cell_ids,
        cell_tiles=cell_tiles,
        edge_vertices=edge_vertices,
        edge_sources=edge_sources,
        edge_seams=edge_seams,
        local_tiles=tiles.tile_ids[local],
        global_vertex_count=vertex_total,
        global_cell_count=cell_total,
        minimum_angle_degrees=minimum_angle,
        seam_rounds=rounds,
        refinement_complete=True,
        source_id=prepared.source.source_id,
        source_revision=prepared.source.source_revision,
        specification_id=prepared.specification.specification_id,
        compiled_id=construction_id,
        partition_id=prepared.partition_result.result_id,
    )


@checked
def execute_distributed_planar_initial(
    prepared: PreparedDistributedPlanarGeneration, /
) -> DistributedPlanarConstruction:
    """Triangulate owned tiles and exchange seam constraints to a fixed point."""
    tiles = prepared.tiles
    group = prepared.execution_group
    rank = _group_rank(group)
    owner_by_tile = _tile_owner_by_row(prepared)
    local = np.asarray(prepared.local_tile_rows, dtype=np.int64)
    seam_points: dict[int, NDArray[np.float64]] = {}
    results: dict[int, _TileMesh] = {}
    pending = set(local.tolist())
    rounds = 0
    while True:
        local_error: Exception | None = None
        try:
            for tile_row in sorted(pending):
                results[tile_row] = _triangulate_tile(tiles, tile_row, seam_points)
        except Exception as error:
            local_error = error
        _collective_failure_barrier(local_error, group, f"{_STAGE}-triangulation")
        outgoing: list[list[tuple[int, NDArray[np.float64]]]] = [
            [] for _ in range(prepared.partition_plan.part_count)
        ]
        changed: set[int] = set()
        local_error = None
        try:
            for tile_row in sorted(pending):
                for seam, values in results[tile_row].seam_points.items():
                    axis = int(tiles.seam_axes[seam])
                    current = seam_points.get(seam, np.empty((0, 2), dtype=np.float64))
                    merged = _merge_seam(current, values, axis)
                    if merged.shape[0] != current.shape[0]:
                        seam_points[seam] = merged
                        changed.add(seam)
                    for tile in tiles.seam_tiles[seam].tolist():
                        destination = owner_by_tile[int(tile)]
                        if destination != rank:
                            outgoing[destination].extend(
                                (seam, point) for point in values
                            )
        except Exception as error:
            local_error = error
        _collective_failure_barrier(local_error, group, f"{_STAGE}-seam-routing")
        incoming = _exchange_seam_points(outgoing, rank, group)
        local_error = None
        try:
            for seam, values in incoming.items():
                if not 0 <= seam < tiles.seam_axes.shape[0]:
                    raise ValueError("Seam exchange named an unknown canonical seam.")
                axis = int(tiles.seam_axes[seam])
                current = seam_points.get(seam, np.empty((0, 2), dtype=np.float64))
                merged = _merge_seam(current, values, axis)
                if merged.shape[0] != current.shape[0]:
                    seam_points[seam] = merged
                    changed.add(seam)
        except Exception as error:
            local_error = error
        _collective_failure_barrier(local_error, group, f"{_STAGE}-seam-merge")
        changed_flags = _owner_group_allgather(
            np.asarray((int(bool(changed)),), dtype=np.int32),
            group,
            "planar_seam_convergence",
        )
        rounds += 1
        if not np.any(changed_flags):
            break
        changed_array = np.asarray(sorted(changed), dtype=np.int64)
        pending = {
            int(tile_row)
            for tile_row in local.tolist()
            if np.any(
                np.isin(
                    tiles.segment_seams[
                        (tiles.segment_tiles[:, 0] == tile_row)
                        | (tiles.segment_tiles[:, 1] == tile_row)
                    ],
                    changed_array,
                )
            )
        }
    return _number_rows(prepared, results, seam_points, rounds)


__all__ = [
    "DistributedPlanarConstruction",
    "execute_distributed_planar_initial",
    "PlanarTileComplex",
    "PlanarTileSchedule",
    "prepare_distributed_planar_generation",
    "PreparedDistributedPlanarGeneration",
]
