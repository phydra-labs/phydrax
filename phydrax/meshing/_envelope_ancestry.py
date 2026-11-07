#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact envelope source charts for canonically recovered PLC vertices."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry._planar_coverage import _source_point, PlcSourcePointWitness


if TYPE_CHECKING:
    from ._surface_envelope import SurfaceEnvelopeSource


type _Point = tuple[Fraction, Fraction, Fraction]


class EnvelopeAncestry:
    """Resolve target-vertex rows to exact source facet and edge parameters.

    The point witness describes exact source positions, not rounded carriers.
    The caller must also validate it against the target mesh and its coverage
    domain; this class verifies that its source rows are the envelope facets.
    """

    def __init__(
        self, source: SurfaceEnvelopeSource, witness: PlcSourcePointWitness, /
    ) -> None:
        if not isinstance(witness, PlcSourcePointWitness):
            raise TypeError("Envelope subdivision requires a PLC source-point witness.")
        domain = source.domain
        vertices = np.asarray(domain.vertices, dtype=np.float64)
        facets = np.asarray(domain.facets, dtype=np.int64)
        faces = np.asarray(witness.faces, dtype=np.int64)
        domain_rows = np.asarray(witness.domain_face_rows, dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or faces.shape != facets.shape
            or domain_rows.shape != (facets.shape[0],)
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
            or np.any(np.asarray(witness.segments) >= len(vertices))
            or sorted(tuple(sorted(face)) for face in faces[domain_rows].tolist())
            != sorted(tuple(sorted(face)) for face in facets.tolist())
        ):
            raise ValueError(
                "PLC source-point witness does not realize the envelope facets."
            )
        self._vertices: tuple[_Point, ...] = tuple(
            (
                Fraction.from_float(float(point[0])),
                Fraction.from_float(float(point[1])),
                Fraction.from_float(float(point[2])),
            )
            for point in vertices
        )
        self._facets = facets
        self._edges = np.asarray(source.edges, dtype=np.int64)
        points = (
            None if int(kind) < 0 else _source_point(witness, row, self._vertices)
            for row, kind in enumerate(witness.vertex_kinds)
        )
        self._points: tuple[_Point | None, ...] = tuple(
            None if point is None else (point[0], point[1], point[2]) for point in points
        )

    def _point(self, row: int, /) -> _Point:
        if row < 0 or row >= len(self._points) or self._points[row] is None:
            raise ValueError("Target vertex has no exact envelope source point.")
        point = self._points[row]
        assert point is not None
        return point

    def facet_parameters(
        self, target_vertex_rows: tuple[int, ...], source_facet_row: int, /
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        if source_facet_row < 0 or source_facet_row >= len(self._facets):
            raise ValueError("Envelope source facet row is out of range.")
        a, b, c = (self._vertices[int(row)] for row in self._facets[source_facet_row])
        ab = tuple(y - x for x, y in zip(a, b, strict=True))
        ac = tuple(y - x for x, y in zip(a, c, strict=True))
        axes = next(
            (
                (i, j)
                for i, j in combinations(range(3), 2)
                if ab[i] * ac[j] != ab[j] * ac[i]
            ),
            None,
        )
        if axes is None:
            raise ValueError("Envelope source facet is degenerate.")
        i, j = axes
        determinant = ab[i] * ac[j] - ab[j] * ac[i]
        parameters = []
        for row in target_vertex_rows:
            point = self._point(int(row))
            delta = tuple(y - x for x, y in zip(a, point, strict=True))
            u = (delta[i] * ac[j] - delta[j] * ac[i]) / determinant
            v = (ab[i] * delta[j] - ab[j] * delta[i]) / determinant
            if (
                u < 0
                or v < 0
                or u + v > 1
                or any(delta[k] != u * ab[k] + v * ac[k] for k in range(3))
            ):
                raise ValueError(
                    "Target source point is not on its declared envelope facet."
                )
            parameters.append((u, v))
        return tuple(parameters)

    def edge_parameters(
        self, target_vertex_rows: tuple[int, ...], source_edge_row: int, /
    ) -> tuple[tuple[Fraction], ...]:
        if source_edge_row < 0 or source_edge_row >= len(self._edges):
            raise ValueError("Envelope source edge row is out of range.")
        a, b = (self._vertices[int(row)] for row in self._edges[source_edge_row])
        direction = tuple(y - x for x, y in zip(a, b, strict=True))
        axis = next((i for i in range(3) if direction[i]), None)
        if axis is None:
            raise ValueError("Envelope source edge is degenerate.")
        parameters = []
        for row in target_vertex_rows:
            point = self._point(int(row))
            t = (point[axis] - a[axis]) / direction[axis]
            if not 0 <= t <= 1 or any(
                point[k] - a[k] != t * direction[k] for k in range(3)
            ):
                raise ValueError(
                    "Target source point is not on its declared envelope edge."
                )
            parameters.append((t,))
        return tuple(parameters)
