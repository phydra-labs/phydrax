#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact source-corner identity checks shared by mapped embedding routes."""

from __future__ import annotations

from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_algebraic_geometry import ExactScalar
from ._embedding_cells import prepare_embedding_cells, UnsupportedEmbeddingMap


def mapped_source_corner_coordinates(
    reference_mesh: CellMesh, source_geometry: CellGeometrySpec, /
) -> dict[int, tuple[ExactScalar, ...]]:
    """Return exact images of declared vertex identities, rejecting trace mismatch."""
    result: dict[int, tuple[ExactScalar, ...]] = {}
    for cell in prepare_embedding_cells(reference_mesh, source_geometry):
        for identifier, image in zip(cell.vertex_ids, cell.corner_images, strict=True):
            prior = result.get(identifier)
            if prior is not None and prior != image:
                raise ValueError(
                    f"Exact source-map traces disagree at declared vertex {identifier}."
                )
            result[identifier] = image
    if not result:
        raise UnsupportedEmbeddingMap("No exact source corners were prepared.")
    return result


__all__ = ["mapped_source_corner_coordinates"]
