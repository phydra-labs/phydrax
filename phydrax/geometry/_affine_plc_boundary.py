#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact global-affine PLC boundary query with original facet ancestry."""

from __future__ import annotations

from fractions import Fraction
from typing import final, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ._affine_plc_domain import AffinePlcDomain
from ._domain_boundary import _point_triangle_squared
from ._rational import fraction_endpoint, rational_vector
from ._source_boundary_records import (
    SourceBoundaryChartCover,
    SourceBoundaryChartWitness,
    SourceBoundaryDistance,
)


if TYPE_CHECKING:
    from ..meshing.providers._native_sources import NativePlcSource


@final
class AffinePlcBoundarySource(StrictModule, NonTrainableState):
    """Complete triangle query retaining each triangle's original facet key."""

    owner: NativePlcSource
    domain: AffinePlcDomain
    facet_rows: np.ndarray
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_query_id: str = eqx.field(static=True)
    face_chart_ids: tuple[str, ...] = eqx.field(static=True)
    face_scope_keys: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    face_keys: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)

    def __init__(
        self, owner: object, domain: AffinePlcDomain, facet_rows: ArrayLike, /
    ) -> None:
        from ..meshing.providers._native_sources import NativePlcSource

        if not isinstance(owner, NativePlcSource) or type(domain) is not AffinePlcDomain:
            raise TypeError(
                "An original native PLC source and affine PLC domain are required."
            )
        if (owner.source_id, owner.source_revision) != (
            domain.source_id,
            domain.source_revision,
        ):
            raise ValueError("Affine boundary and original PLC source identities differ.")
        complex_ = owner.complex
        if domain.authority is not None:
            from ..meshing._compartments import prepare_compartment_plc

            if (
                complex_.complex_id
                != prepare_compartment_plc(domain.authority).complex.complex_id
            ):
                raise ValueError(
                    "PLC source complex differs from the categorical authority."
                )
        reference = domain.reference_domain
        raw = np.asarray(facet_rows)
        if (
            raw.ndim != 1
            or raw.dtype.kind not in "iu"
            or (raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max))
        ):
            raise ValueError("Triangle facet ancestry must be an integer row vector.")
        rows = np.array(raw, dtype=np.int64, copy=True)
        if (
            rows.shape != (len(reference.facets),)
            or np.any(rows < 0)
            or np.any(rows >= len(complex_.facet_regions))
            or not np.array_equal(reference.vertices, complex_.vertices)
            or reference.region_ids != complex_.region_ids
            or reference.region_occurrence_paths != complex_.region_occurrence_paths
            or not np.array_equal(
                reference.facet_regions, np.asarray(complex_.facet_regions)[rows, ::-1]
            )
        ):
            raise ValueError(
                "Affine boundary ancestry changes the original PLC or its material pairs."
            )
        polygon_vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
        offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
        polygon_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
        facet_vertices = [set() for _ in range(len(complex_.facet_regions))]
        for polygon, facet in enumerate(polygon_facets):
            facet_vertices[int(facet)].update(
                polygon_vertices[offsets[polygon] : offsets[polygon + 1]].tolist()
            )
        if any(
            not set(map(int, triangle)) <= facet_vertices[int(facet)]
            for triangle, facet in zip(
                np.asarray(reference.facets).tolist(), rows, strict=True
            )
        ):
            raise ValueError(
                "A source triangle names vertices outside its original PLC facet."
            )
        keys = tuple(
            owner.source_entity_key_for_scope(2, int(complex_.facet_entity_ids[int(row)]))
            for row in rows
        )
        rows.setflags(write=False)
        query_id = canonical_fingerprint(
            {
                "kind": "affine-plc-boundary",
                "domain": domain.domain_id,
                "owner": owner.binding_id,
                "facet_rows": array_tree_fingerprint(rows),
                "keys": keys,
            }
        )
        self.owner = owner
        self.domain = domain
        self.facet_rows = rows
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.source_query_id = query_id
        self.face_keys = keys
        self.face_scope_keys = tuple(sorted(set(keys)))
        self.face_chart_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "affine-plc-face-chart",
                    "query": query_id,
                    "triangle": row,
                    "source": key,
                }
            )
            for row, key in enumerate(keys)
        )

    @property
    def entity_dimension(self) -> int:
        return 2

    @property
    def is_full_scope(self) -> bool:
        return True

    @property
    def chart_ids(self) -> tuple[str, ...]:
        return self.face_chart_ids

    @property
    def source_rows(self) -> np.ndarray:
        return self.facet_rows

    def require_current(self, /) -> None:
        owner = self.owner
        self.domain.require_current()
        owner.validate_restored()
        rebuilt = type(self)(owner, self.domain, self.facet_rows)
        if (
            rebuilt.source_query_id != self.source_query_id
            or rebuilt.face_keys != self.face_keys
            or rebuilt.face_chart_ids != self.face_chart_ids
            or rebuilt.face_scope_keys != self.face_scope_keys
        ):
            raise ValueError(
                "Affine PLC boundary query differs from its source authority."
            )

    def validate_chart_witness(self, witness: SourceBoundaryChartWitness, /) -> None:
        if not isinstance(witness, SourceBoundaryChartWitness):
            raise TypeError("Affine PLC charts require a source boundary chart witness.")
        self.require_current()
        if (
            witness.source_id != self.source_id
            or witness.source_revision != self.source_revision
            or witness.source_query_id != self.source_query_id
            or witness.chart_dimension != 2
            or witness.implicit_cell_keys.shape != (0, 4)
            or witness.implicit_root_edge_keys.shape != (0, 3, 2, 3)
            or set(witness.source_chart_ids) != set(self.face_chart_ids)
        ):
            raise ValueError("Affine PLC chart witness changes its source query.")
        for row, chart in enumerate(witness.source_chart_ids):
            if (
                witness.source_entity_ids[row],
                witness.occurrence_paths[row],
            ) != self.chart_source_key(2, chart):
                raise ValueError(
                    "Affine PLC chart witness changes original facet identity."
                )
            if any(
                u < 0 or v < 0 or u + v > 1
                for u, v in witness.source_parameter_chart(row)
            ):
                raise ValueError(
                    "Affine PLC chart parameters leave their source triangle."
                )

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def query_id(self) -> str:
        return self.source_query_id

    def chart_row(self, dimension: int, chart_id: str, /) -> int:
        if dimension != 2:
            raise ValueError("Affine PLC charts are triangle charts.")
        return self.face_chart_ids.index(chart_id)

    def chart_source_key(
        self, dimension: int, chart_id: str, /
    ) -> tuple[str, tuple[str, ...]]:
        return self.face_keys[self.chart_row(dimension, chart_id)]

    def chart_coordinate_expressions(
        self, dimension: int, chart_id: str, /
    ) -> tuple[dict[tuple[int, ...], Fraction], ...]:
        row = self.chart_row(dimension, chart_id)
        triangle = self.domain.facets[row]
        a, b, c = (self.domain.exact_vertices[int(vertex)] for vertex in triangle)
        return tuple(
            {(0, 0): a[axis], (1, 0): b[axis] - a[axis], (0, 1): c[axis] - a[axis]}
            for axis in range(3)
        )

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or np.any(~np.isfinite(values)):
            raise ValueError("Boundary distance points must be finite three-vectors.")
        self.require_current()
        vertices = tuple(
            (point[0], point[1], point[2]) for point in self.domain.exact_vertices
        )
        triangles = tuple(
            (vertices[int(facet[0])], vertices[int(facet[1])], vertices[int(facet[2])])
            for facet in self.domain.facets
        )
        lower = np.empty(len(values), dtype=np.float64)
        upper = np.empty(len(values), dtype=np.float64)
        for row in range(len(values)):
            point = rational_vector(
                (float(values[row, 0]), float(values[row, 1]), float(values[row, 2]))
            )
            squared = min(
                _point_triangle_squared(point, triangle) for triangle in triangles
            )
            low, high = fraction_sqrt_interval(squared)
            lower[row] = fraction_endpoint(low, upper=False, saturate=True)
            upper[row] = fraction_endpoint(high, upper=True, saturate=True)
        return SourceBoundaryDistance(lower, upper, "certified")

    def boundary_chart_cover(self, /) -> SourceBoundaryChartCover:
        self.require_current()
        vertices = self.domain.vertices
        triangles = vertices[self.domain.facets]
        error = max(
            sum(
                (
                    (x - Fraction(float(y))) ** 2
                    for x, y in zip(exact, rounded, strict=True)
                ),
                Fraction(0),
            )
            for exact, rounded in zip(self.domain.exact_vertices, vertices, strict=True)
        )
        _, high = fraction_sqrt_interval(error)
        bound = fraction_endpoint(high, upper=True, saturate=True)
        return SourceBoundaryChartCover(
            triangles,
            np.full(len(triangles), bound),
            "certified",
            True,
            self.source_id,
            self.source_revision,
            restriction_ids=(self.source_query_id,),
        )


__all__ = ["AffinePlcBoundarySource"]
