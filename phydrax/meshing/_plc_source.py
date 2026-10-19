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
from .._meshcore import TetMeshBoundaryPolicy
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import unique_identifiers
from ..typing import (
    checked,
    ConvertibleToArray,
    Dim,
    HostFloat64,
    HostInt64,
    parse,
    Scope,
)


class PlcVertexDim(Dim, minimum=4):
    """Vertices of one piecewise-linear complex."""


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


@final
class PiecewiseLinearComplex(StrictModule, NonTrainableState):
    """Oriented 3D piecewise-linear complex: the source of a PLC volume mesh.

    ``polygons`` are loops of distinct, exactly coplanar vertex indices;
    ``polygon_facets`` names each polygon's facet. ``facet_regions[f]`` is the
    pair (positive, negative): the index into ``region_ids`` of the region on
    the side of the loops' right-hand normal and on the opposite side; ``-1``
    is void (exterior or a hole) and equal entries declare an internal sheet.
    Entities meet only at shared vertices and edges, and every region's
    oriented facet chain must be closed; the native recovery certifies both
    exactly. ``segments`` are internal curves inside regions. ``boundary``
    ``"fixed"`` declares the supplied boundary immutable (no Steiner point on
    any facet or edge; triangle polygons are kept exactly); ``"conforming"``
    admits rational boundary subdivision with recorded ancestry.
    """

    __strict_contract__ = True

    vertices: HostFloat64[PlcVertexDim, Literal[3]]
    polygon_offsets: HostInt64[PlcPolygonOffsetDim]
    polygon_vertices: HostInt64[PlcLoopDim]
    polygon_facets: HostInt64[PlcPolygonDim]
    facet_regions: HostInt64[PlcFacetDim, Literal[2]]
    segments: HostInt64[PlcSegmentDim, Literal[2]]
    region_ids: tuple[str, ...] = eqx.field(static=True)
    boundary: TetMeshBoundaryPolicy = eqx.field(static=True)
    complex_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ConvertibleToArray,
        polygons: Sequence[ConvertibleToArray],
        polygon_facets: ConvertibleToArray,
        facet_regions: ConvertibleToArray,
        region_ids: tuple[str, ...],
        /,
        *,
        segments: ConvertibleToArray | None = None,
        boundary: TetMeshBoundaryPolicy = "conforming",
    ) -> None:
        if isinstance(polygons, np.ndarray) and polygons.ndim != 2:
            raise TypeError(
                "PiecewiseLinearComplex.polygons must be a sequence of index loops; got a nonmatrix ndarray"
            )
        loops = [np.asarray(loop) for loop in polygons]
        if not loops:
            raise ValueError(
                "PiecewiseLinearComplex.polygons must contain at least one polygon; got an empty sequence"
            )
        for loop in loops:
            if loop.ndim != 1 or not np.issubdtype(loop.dtype, np.integer):
                raise TypeError(
                    "PiecewiseLinearComplex.polygons must contain one-dimensional integer loops; got an invalid loop"
                )
            if loop.shape[0] < 3:
                raise ValueError(
                    "PiecewiseLinearComplex.polygons must contain at least three vertices per loop; got a short loop"
                )
        scope = Scope()
        points = parse(
            np.asarray(vertices, dtype=np.float64),
            HostFloat64[PlcVertexDim, Literal[3]],
            "vertices",
            scope=scope,
        )
        offsets_raw = np.concatenate(
            ([0], np.cumsum([loop.shape[0] for loop in loops]))
        ).astype(np.int64)
        flat_raw = np.concatenate(loops).astype(np.int64)
        offsets = parse(
            offsets_raw,
            HostInt64[PlcPolygonOffsetDim],
            "polygon_offsets",
            scope=scope,
        )
        flat = parse(
            flat_raw,
            HostInt64[PlcLoopDim],
            "polygon_vertices",
            scope=scope,
        )
        owners = parse(
            np.asarray(polygon_facets, dtype=np.int64),
            HostInt64[PlcPolygonDim],
            "polygon_facets",
            scope=scope,
        )
        incidence = parse(
            np.asarray(facet_regions, dtype=np.int64),
            HostInt64[PlcFacetDim, Literal[2]],
            "facet_regions",
            scope=scope,
        )
        curves = parse(
            np.zeros((0, 2), dtype=np.int64)
            if segments is None
            else np.asarray(segments, dtype=np.int64),
            HostInt64[PlcSegmentDim, Literal[2]],
            "segments",
            scope=scope,
        )
        ids = unique_identifiers(region_ids, "region_ids")
        policy = parse(boundary, TetMeshBoundaryPolicy, "boundary")
        if owners.shape[0] != len(loops):
            raise ValueError(
                "PiecewiseLinearComplex.polygon_facets must name one facet per polygon; got a different count"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError(
                "PiecewiseLinearComplex.vertices must be finite; got nonfinite coordinates"
            )
        vertex_count = points.shape[0]
        if np.any((flat < 0) | (flat >= vertex_count)) or np.any(
            (curves < 0) | (curves >= vertex_count)
        ):
            raise ValueError(
                "PiecewiseLinearComplex.incidence must index declared vertices; got out-of-range polygon or segment IDs"
            )
        facet_count = incidence.shape[0]
        if np.any((owners < 0) | (owners >= facet_count)):
            raise ValueError(
                "PiecewiseLinearComplex.polygon_facets must index declared facets; got out-of-range facet IDs"
            )
        if np.setdiff1d(np.arange(facet_count), owners).size:
            raise ValueError(
                "PiecewiseLinearComplex.polygon_facets must cover every facet; got an unused facet"
            )
        if np.any((incidence < -1) | (incidence >= len(ids))) or np.any(
            np.all(incidence < 0, axis=1)
        ):
            raise ValueError(
                "PiecewiseLinearComplex.facet_regions must index regions or -1 with a nonvoid side; got invalid facet incidence"
            )
        if np.setdiff1d(np.arange(len(ids)), incidence).size:
            raise ValueError(
                "PiecewiseLinearComplex.facet_regions must cover every region; got an unbounded region"
            )
        self.vertices = points
        self.polygon_offsets = offsets
        self.polygon_vertices = flat
        self.polygon_facets = owners
        self.facet_regions = incidence
        self.segments = curves
        self.region_ids = ids
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
                "region_ids": ids,
                "boundary": policy,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PiecewiseLinearComplex], value: PiecewiseLinearComplex, /
    ) -> PiecewiseLinearComplex:
        """Rebuild packed polygon loops and validate their retained source identity."""
        if type(value) is not cls:
            raise TypeError(
                f"PiecewiseLinearComplex.stored_fields must have the registered type; got {type(value).__name__}"
            )
        scope = Scope()
        offsets = parse(
            value.polygon_offsets,
            HostInt64[PlcPolygonOffsetDim],
            "polygon_offsets",
            scope=scope,
        )
        vertices = parse(
            value.polygon_vertices, HostInt64[PlcLoopDim], "polygon_vertices", scope=scope
        )
        if (
            offsets[0] != 0
            or offsets[-1] != vertices.size
            or np.any(offsets[1:] - offsets[:-1] < 3)
        ):
            raise ValueError(
                "PiecewiseLinearComplex.polygon_offsets must partition polygon_vertices into loops of at least three entries; got invalid stored offsets"
            )
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
            boundary=value.boundary,
        )
        if fresh.complex_id != value.complex_id:
            raise ValueError(
                f"PiecewiseLinearComplex.complex_id must match its stored source fields; got {value.complex_id}"
            )
        return fresh

    @property
    def facet_count(self) -> int:
        return self.facet_regions.shape[0]

    @property
    def region_count(self) -> int:
        return len(self.region_ids)


__all__ = ["PiecewiseLinearComplex"]
