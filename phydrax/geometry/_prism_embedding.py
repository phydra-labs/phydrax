#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact affine prism facet routing for oriented volume complexes."""

from __future__ import annotations

from ..discretization._exact_algebraic_geometry import ExactScalar
from ..discretization._reference_cell import reference_cell_topology
from ._embedding_cells import EmbeddingCell, UnsupportedEmbeddingMap
from ._exact_polyhedral_geometry import Point3


def prism_facet_routes(cell: EmbeddingCell, /) -> tuple[tuple[int, ...], ...]:
    """Return the outward-oriented triangular and quadrilateral prism faces."""
    if cell.kind != "prism" or cell.dimension != 3 or not cell.affine:
        raise UnsupportedEmbeddingMap("Prism boundary routing requires an exact affine 3-D prism.")
    topology = reference_cell_topology("prism")
    return tuple(tuple(int(index) for index in row) for row in topology.entities[2])


def _point3(point: tuple[ExactScalar, ...], /) -> Point3:
    if len(point) != 3:
        raise UnsupportedEmbeddingMap(
            "Prism facet images require three-dimensional exact source corners."
        )
    return point[0], point[1], point[2]


def prism_facet_images(
    cell: EmbeddingCell, /
) -> tuple[tuple[tuple[int, ...], tuple[Point3, ...]], ...]:
    """Return each oriented reference face with its exact source-corner image."""
    if cell.kind != "prism" or cell.dimension != 3 or not cell.affine:
        raise UnsupportedEmbeddingMap("Prism boundary routing requires an exact affine 3-D prism.")
    return tuple(
        (
            tuple(cell.vertex_ids[index] for index in route),
            tuple(_point3(cell.corner_images[index]) for index in route),
        )
        for route in prism_facet_routes(cell)
    )


__all__ = ["prism_facet_images", "prism_facet_routes"]
