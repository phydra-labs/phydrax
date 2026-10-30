#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Oriented piecewise-linear source contracts for volume construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, Literal

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import normalized_identifier
from ..discretization import CellGeometrySpec
from ..typing import (
    checked,
    ConvertibleToArray,
    Dim,
    HostFloat64,
    HostInt64,
    parse,
    Scope,
)


def _occurrence_path_bank(
    values: tuple[tuple[str, ...], ...], count: int, name: str, /
) -> tuple[tuple[str, ...], ...]:
    if len(values) != count:
        raise ValueError(f"{name} must contain one authored path per source entity.")
    paths = tuple(tuple(value) for value in values)
    if any(
        any(
            not isinstance(component, str)
            or not component
            or component != component.strip()
            for component in path
        )
        for path in paths
    ):
        raise ValueError(f"{name} components must be non-empty normalized identifiers.")
    return paths


def _identifier_bank(
    values: tuple[str, ...],
    occurrence_paths: tuple[tuple[str, ...], ...],
    count: int,
    name: str,
    /,
) -> tuple[str, ...]:
    identifiers = tuple(normalized_identifier(value, name) for value in values)
    if len(identifiers) != count:
        raise ValueError(f"{name} must contain exactly {count} authored identifiers.")
    if len(set(zip(identifiers, occurrence_paths, strict=True))) != count:
        raise ValueError(f"{name} and occurrence paths must uniquely identify each source entity.")
    return identifiers


def _numeric_entity_ids(
    kind: str,
    values: tuple[str, ...],
    occurrence_paths: tuple[tuple[str, ...], ...],
    /,
) -> tuple[int, ...]:
    identifiers = tuple(
        int(
            canonical_fingerprint(
                {"kind": kind, "value": (value, path)}
            )[:16],
            16,
        )
        & ((1 << 63) - 1)
        for value, path in zip(values, occurrence_paths, strict=True)
    )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            f"Authored {kind} identity/occurrence pairs collide in the int64 scope projection."
        )
    return identifiers


def _plc_edge_records(
    offsets: np.ndarray,
    polygon_vertices: np.ndarray,
    polygon_facets: np.ndarray,
    segments: np.ndarray,
    vertex_ids: tuple[str, ...],
    vertex_paths: tuple[tuple[str, ...], ...],
    facet_ids: tuple[str, ...],
    facet_paths: tuple[tuple[str, ...], ...],
    segment_ids: tuple[str, ...],
    segment_paths: tuple[tuple[str, ...], ...],
    /,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, ...], ...],
    tuple[str, ...],
    tuple[tuple[str, ...], ...],
    tuple[int, ...],
]:
    boundary: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for polygon, facet in enumerate(polygon_facets.tolist()):
        start, stop = offsets[polygon : polygon + 2]
        loop = polygon_vertices[start:stop]
        for offset, first in enumerate(loop.tolist()):
            second = int(loop[(offset + 1) % loop.size])
            first_vertex = int(first)
            key = (min(first_vertex, second), max(first_vertex, second))
            boundary.setdefault(key, []).append((facet, int(first), second))

    records: list[tuple[tuple[int, int], tuple[int, ...], str, tuple[str, ...]]] = []
    explicit_keys: set[tuple[int, int]] = set()
    for row, segment in enumerate(np.asarray(segments, dtype=np.int64).tolist()):
        first, second = int(segment[0]), int(segment[1])
        key = (min(first, second), max(first, second))
        if key in explicit_keys:
            raise ValueError("Authored PLC segments cannot duplicate one geometric edge.")
        explicit_keys.add(key)
        facets = tuple(sorted({entry[0] for entry in boundary.get(key, ())}))
        facet_paths_on_edge = {
            facet_paths[facet] for facet in facets
        }
        if facet_paths_on_edge and (
            len(facet_paths_on_edge) != 1
            or segment_paths[row] not in facet_paths_on_edge
        ):
            raise ValueError(
                "An explicit PLC segment must preserve one unique incident facet occurrence path."
            )
        records.append(( (first, second), facets, segment_ids[row], segment_paths[row] ))

    boundary_keys = set(boundary) - explicit_keys
    ordered_boundary_keys = sorted(
        boundary_keys,
        key=lambda key: (
            tuple(
                sorted(
                    (
                        vertex_ids[vertex],
                        vertex_paths[vertex],
                    )
                    for vertex in key
                )
            ),
            tuple(
                sorted(
                    (
                        facet_ids[facet],
                        facet_paths[facet],
                    )
                    for facet in {entry[0] for entry in boundary[key]}
                )
            ),
        ),
    )
    for key in ordered_boundary_keys:
        facets = tuple(
            sorted(
                {entry[0] for entry in boundary[key]},
                key=lambda facet: (facet_ids[facet], facet_paths[facet]),
            )
        )
        paths = {facet_paths[facet] for facet in facets}
        if len(paths) != 1:
            raise ValueError(
                "A PLC edge joining distinct source occurrences requires an authored segment identity and path."
            )
        path = next(iter(paths))
        directed_by_facet = {
            facet: tuple(
                (first, second)
                for entry_facet, first, second in boundary[key]
                if entry_facet == facet
            )
            for facet in facets
        }
        if len(facets) == 1:
            directed = directed_by_facet[facets[0]]
            if len(directed) == 1:
                first, second = directed[0]
            elif len(directed) == 2 and directed[0] == tuple(reversed(directed[1])):
                first, second = sorted(
                    key,
                    key=lambda vertex: (vertex_ids[vertex], vertex_paths[vertex]),
                )
            else:
                raise ValueError("A PLC facet must orient each boundary edge exactly once.")
        else:
            if any(len(directed_by_facet[facet]) != 1 for facet in facets):
                raise ValueError("Each incident PLC facet must orient a shared boundary edge once.")
            first, second = directed_by_facet[facets[0]][0]
        identity = canonical_fingerprint(
            {
                "kind": "plc-facet-edge",
                "vertices": tuple(
                    sorted(
                        (vertex_ids[vertex], vertex_paths[vertex])
                        for vertex in key
                    )
                ),
                "facets": tuple(
                    sorted((facet_ids[facet], facet_paths[facet]) for facet in facets)
                ),
                "occurrence_path": path,
            }
        )
        records.append(((first, second), facets, identity, path))

    vertices = tuple(record[0] for record in records)
    facets = tuple(record[1] for record in records)
    identities = tuple(record[2] for record in records)
    paths = tuple(record[3] for record in records)
    entity_ids = _numeric_entity_ids("plc-edge", identities, paths)
    return vertices, facets, identities, paths, entity_ids


type PlcBoundaryPolicy = Literal["fixed", "conforming"]


class PlcVertexDim(Dim, minimum=4):
    """Vertices of one three-dimensional piecewise-linear complex."""


class PlcPolygonDim(Dim, minimum=1):
    """Oriented polygons of one piecewise-linear complex."""


class PlcPolygonOffsetDim(Dim, minimum=2):
    """Loop offsets of the polygons of one complex (polygons plus one)."""


class PlcLoopDim(Dim, minimum=3):
    """Concatenated polygon loop vertices of one complex."""


class PlcFacetDim(Dim, minimum=1):
    """Facets (region-incidence groups of polygons) of one complex."""


class PlcSegmentDim(Dim):
    """Explicit internal curve segments of one complex."""


def _integer_array(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer indices.")
    if array.size and (
        np.any(array < np.iinfo(np.int64).min)
        or np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} indices must be representable as signed int64.")
    return np.asarray(array, dtype=np.int64)


@final
class PiecewiseLinearComplex(StrictModule, NonTrainableState):
    """Oriented three-dimensional piecewise-linear source for solid filling.

    ``polygon_facets`` groups polygon loops into source facets. Each row of
    ``facet_regions`` is ``(positive, negative)``: the first region lies on
    the right-hand-normal side of the oriented facet, and the second on the
    opposite side. ``-1`` denotes void; equal region indices declare an
    internal sheet. Every declared region must occur in the incidence table.

    ``boundary="fixed"`` preserves every source vertex and facet exactly;
    ``boundary="conforming"`` permits constrained subdivision while retaining
    its source ancestry. ``geometry`` may retain a source coordinate map whose
    rows correspond to ``vertices``; its rounded execution coordinates do not
    strengthen the represented source's geometric claims.

    Each source vertex, facet and explicit segment carries an authored stable
    identity. Polygon-loop edges derive identity from their source endpoint and
    facet keys, including seams between polygons of the same facet. Recovery-only
    triangulation diagonals remain facet-supported carrier edges, not fabricated
    source entities. Missing or ambiguous source identity is refused; coordinates
    and array positions never establish entity identity.

    Occurrence paths are explicit source-entity identity components. A repeated
    entity ID is valid only when its authored occurrence path distinguishes it.
    """

    __strict_contract__ = True

    vertices: HostFloat64[PlcVertexDim, Literal[3]]
    polygon_offsets: HostInt64[PlcPolygonOffsetDim]
    polygon_vertices: HostInt64[PlcLoopDim]
    polygon_facets: HostInt64[PlcPolygonDim]
    facet_regions: HostInt64[PlcFacetDim, Literal[2]]
    segments: HostInt64[PlcSegmentDim, Literal[2]]
    geometry: CellGeometrySpec | None
    vertex_ids: tuple[str, ...] = eqx.field(static=True)
    vertex_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    facet_ids: tuple[str, ...] = eqx.field(static=True)
    facet_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    segment_ids: tuple[str, ...] = eqx.field(static=True)
    segment_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    vertex_entity_ids: tuple[int, ...] = eqx.field(static=True)
    facet_entity_ids: tuple[int, ...] = eqx.field(static=True)
    segment_entity_ids: tuple[int, ...] = eqx.field(static=True)
    plc_edge_vertices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    plc_edge_facets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    plc_edge_ids: tuple[str, ...] = eqx.field(static=True)
    plc_edge_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    plc_edge_entity_ids: tuple[int, ...] = eqx.field(static=True)
    region_ids: tuple[str, ...] = eqx.field(static=True)
    region_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    region_entity_ids: tuple[int, ...] = eqx.field(static=True)
    boundary: PlcBoundaryPolicy = eqx.field(static=True)
    complex_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        vertices: ConvertibleToArray,
        polygons: Sequence[ConvertibleToArray] | np.ndarray,
        polygon_facets: ConvertibleToArray,
        facet_regions: ConvertibleToArray,
        region_ids: tuple[str, ...],
        /,
        *,
        segments: ConvertibleToArray | None = None,
        vertex_ids: tuple[str, ...],
        vertex_occurrence_paths: tuple[tuple[str, ...], ...],
        facet_ids: tuple[str, ...],
        facet_occurrence_paths: tuple[tuple[str, ...], ...],
        segment_ids: tuple[str, ...],
        segment_occurrence_paths: tuple[tuple[str, ...], ...],
        region_occurrence_paths: tuple[tuple[str, ...], ...],
        boundary: PlcBoundaryPolicy = "conforming",
        geometry: CellGeometrySpec | None = None,
    ) -> None:
        if isinstance(polygons, np.ndarray) and polygons.ndim != 2:
            raise TypeError("polygons must be a sequence of one-dimensional index loops.")
        loops = tuple(_integer_array(loop, "polygons") for loop in polygons)
        if not loops:
            raise ValueError("polygons must contain at least one oriented loop.")
        if any(loop.ndim != 1 or loop.shape[0] < 3 for loop in loops):
            raise ValueError("Every polygon loop must be one-dimensional with at least three vertices.")
        total_loop_vertices = sum(loop.shape[0] for loop in loops)
        if total_loop_vertices > np.iinfo(np.int64).max:
            raise OverflowError("The packed polygon incidence extent exceeds int64.")

        scope = Scope()
        points = parse(
            np.array(vertices, dtype=np.float64, copy=True, order="C"),
            HostFloat64[PlcVertexDim, Literal[3]],
            "vertices",
            scope=scope,
        )
        offsets_raw = np.empty((len(loops) + 1,), dtype=np.int64)
        offsets_raw[0] = 0
        for row, loop in enumerate(loops, start=1):
            offsets_raw[row] = offsets_raw[row - 1] + loop.shape[0]
        offsets = parse(
            offsets_raw,
            HostInt64[PlcPolygonOffsetDim],
            "polygon_offsets",
            scope=scope,
        )
        flat = parse(
            np.concatenate(loops),
            HostInt64[PlcLoopDim],
            "polygon_vertices",
            scope=scope,
        )
        owners = parse(
            _integer_array(polygon_facets, "polygon_facets"),
            HostInt64[PlcPolygonDim],
            "polygon_facets",
            scope=scope,
        )
        incidence = parse(
            _integer_array(facet_regions, "facet_regions"),
            HostInt64[PlcFacetDim, Literal[2]],
            "facet_regions",
            scope=scope,
        )
        curves = parse(
            np.zeros((0, 2), dtype=np.int64)
            if segments is None
            else _integer_array(segments, "segments"),
            HostInt64[PlcSegmentDim, Literal[2]],
            "segments",
            scope=scope,
        )
        region_names = tuple(
            normalized_identifier(identifier, "region_id")
            for identifier in region_ids
        )
        if not region_names:
            raise ValueError("region_ids must contain at least one authored source region.")
        policy = parse(boundary, PlcBoundaryPolicy, "boundary")

        if owners.shape[0] != len(loops):
            raise ValueError("polygon_facets must name one facet for each polygon loop.")
        if not np.all(np.isfinite(points)):
            raise ValueError("vertices must contain only finite coordinates.")
        vertex_count = points.shape[0]
        if np.any((flat < 0) | (flat >= vertex_count)) or np.any(
            (curves < 0) | (curves >= vertex_count)
        ):
            raise ValueError("Polygon and segment incidence must index declared vertices.")
        facet_count = incidence.shape[0]
        if np.any((owners < 0) | (owners >= facet_count)):
            raise ValueError("polygon_facets must index declared facets.")
        if np.setdiff1d(np.arange(facet_count), owners).size:
            raise ValueError("Every declared facet must own at least one polygon.")
        if np.any((incidence < -1) | (incidence >= len(region_names))) or np.any(
            np.all(incidence < 0, axis=1)
        ):
            raise ValueError("Each facet must separate at least one declared region from a region or void.")
        if np.setdiff1d(np.arange(len(region_names)), incidence).size:
            raise ValueError("Every declared region must occur in the facet incidence table.")
        if geometry is not None and geometry.coordinates.shape != points.shape:
            raise ValueError("geometry must bind one source coordinate row per PLC vertex.")
        vertex_paths = _occurrence_path_bank(
            vertex_occurrence_paths, vertex_count, "vertex_occurrence_paths"
        )
        facet_paths = _occurrence_path_bank(
            facet_occurrence_paths, facet_count, "facet_occurrence_paths"
        )
        segment_paths = _occurrence_path_bank(
            segment_occurrence_paths, curves.shape[0], "segment_occurrence_paths"
        )
        region_paths = _occurrence_path_bank(
            region_occurrence_paths, len(region_names), "region_occurrence_paths"
        )
        if len(set(zip(region_names, region_paths, strict=True))) != len(region_names):
            raise ValueError("Region IDs and occurrence paths must uniquely identify each source region.")
        vertex_names = _identifier_bank(
            vertex_ids, vertex_paths, vertex_count, "vertex_ids"
        )
        facet_names = _identifier_bank(
            facet_ids, facet_paths, facet_count, "facet_ids"
        )
        segment_names = _identifier_bank(
            segment_ids, segment_paths, curves.shape[0], "segment_ids"
        )
        vertex_entities = _numeric_entity_ids(
            "plc-vertex", vertex_names, vertex_paths
        )
        facet_entities = _numeric_entity_ids(
            "plc-facet", facet_names, facet_paths
        )
        segment_entities = _numeric_entity_ids(
            "plc-segment", segment_names, segment_paths
        )
        region_entities = _numeric_entity_ids(
            "plc-region", region_names, region_paths
        )
        (
            plc_edge_vertices,
            plc_edge_facets,
            plc_edge_ids,
            plc_edge_paths,
            plc_edge_entities,
        ) = _plc_edge_records(
            offsets,
            flat,
            owners,
            curves,
            vertex_names,
            vertex_paths,
            facet_names,
            facet_paths,
            segment_names,
            segment_paths,
        )

        for array in (points, offsets, flat, owners, incidence, curves):
            array.setflags(write=False)
        self.vertices = points
        self.polygon_offsets = offsets
        self.polygon_vertices = flat
        self.polygon_facets = owners
        self.facet_regions = incidence
        self.segments = curves
        self.geometry = geometry
        self.vertex_ids = vertex_names
        self.vertex_occurrence_paths = vertex_paths
        self.facet_ids = facet_names
        self.facet_occurrence_paths = facet_paths
        self.segment_ids = segment_names
        self.segment_occurrence_paths = segment_paths
        self.vertex_entity_ids = vertex_entities
        self.facet_entity_ids = facet_entities
        self.segment_entity_ids = segment_entities
        self.plc_edge_vertices = plc_edge_vertices
        self.plc_edge_facets = plc_edge_facets
        self.plc_edge_ids = plc_edge_ids
        self.plc_edge_occurrence_paths = plc_edge_paths
        self.plc_edge_entity_ids = plc_edge_entities
        self.region_occurrence_paths = region_paths
        self.region_ids = region_names
        self.region_entity_ids = region_entities
        self.boundary = policy
        self.complex_id = canonical_fingerprint(
            {
                "kind": "piecewise-linear-complex",
                "vertices": array_tree_fingerprint(points),
                "polygon_offsets": array_tree_fingerprint(offsets),
                "polygon_vertices": array_tree_fingerprint(flat),
                "polygon_facets": array_tree_fingerprint(owners),
                "facet_regions": array_tree_fingerprint(incidence),
                "segments": array_tree_fingerprint(curves),
                "vertex_ids": vertex_names,
                "vertex_occurrence_paths": vertex_paths,
                "facet_ids": facet_names,
                "facet_occurrence_paths": facet_paths,
                "segment_ids": segment_names,
                "segment_occurrence_paths": segment_paths,
                "plc_edge_vertices": plc_edge_vertices,
                "plc_edge_ids": plc_edge_ids,
                "plc_edge_occurrence_paths": plc_edge_paths,
                "region_ids": region_names,
                "region_occurrence_paths": region_paths,
                "boundary": policy,
                **(
                    {"geometry": array_tree_fingerprint(geometry)}
                    if geometry is not None
                    else {}
                ),
            }
        )


    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PiecewiseLinearComplex], value: PiecewiseLinearComplex, /
    ) -> PiecewiseLinearComplex:
        """Rebuild packed loops and verify their retained scientific identity."""
        if type(value) is not cls:
            raise TypeError("Stored PLC fields must have the registered PiecewiseLinearComplex type.")
        offsets = np.asarray(value.polygon_offsets, dtype=np.int64)
        vertices = np.asarray(value.polygon_vertices, dtype=np.int64)
        if (
            offsets.ndim != 1
            or offsets.shape[0] < 2
            or offsets.shape[0] != value.polygon_facets.shape[0] + 1
            or offsets[0] != 0
            or offsets[-1] != vertices.size
            or np.any(offsets[1:] - offsets[:-1] < 3)
        ):
            raise ValueError("Stored polygon offsets must partition loops of at least three indices.")
        loops = tuple(
            vertices[start:end]
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        )
        fresh = cls(
            value.vertices,
            loops,
            value.polygon_facets,
            value.facet_regions,
            value.region_ids,
            segments=value.segments,
            vertex_ids=value.vertex_ids,
            vertex_occurrence_paths=value.vertex_occurrence_paths,
            facet_ids=value.facet_ids,
            segment_ids=value.segment_ids,
            facet_occurrence_paths=value.facet_occurrence_paths,
            boundary=value.boundary,
            geometry=value.geometry,
            segment_occurrence_paths=value.segment_occurrence_paths,
            region_occurrence_paths=value.region_occurrence_paths,
        )
        if fresh.complex_id != value.complex_id:
            raise ValueError("Stored PLC fields do not match their complex identity.")
        return fresh

    @property
    def facet_count(self) -> int:
        return self.facet_regions.shape[0]

    @property
    def region_count(self) -> int:
        return len(self.region_ids)


__all__ = ["PiecewiseLinearComplex", "PlcBoundaryPolicy"]
