#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import RowRelation
from .._cell_complex import PolygonalConnectivity
from .._cell_mesh import CellMesh


def validate_conforming_polygon_segments(mesh: CellMesh, /) -> None:
    """Reject geometric T-junctions hidden from exact edge connectivity."""
    if not isinstance(mesh.connectivity, PolygonalConnectivity):
        raise TypeError("Conforming polygon validation requires polygon connectivity.")
    points = np.asarray(mesh.coordinates, dtype=float)
    edges = np.asarray(mesh.connectivity.edges, dtype=np.int32)
    extent = max(float(np.max(np.ptp(points, axis=0))), 1.0)
    tolerance = 512.0 * np.finfo(points.dtype).eps * extent
    for start_index, stop_index in edges:
        start = points[start_index]
        stop = points[stop_index]
        tangent = stop - start
        length2 = float(np.dot(tangent, tangent))
        if length2 <= tolerance * tolerance:
            raise ValueError("Polygon interfaces contain a zero-length edge.")
        relative = points - start
        parameter = relative @ tangent / length2
        projection = start + parameter[:, None] * tangent
        distance = np.sqrt(np.sum((points - projection) ** 2, axis=1))
        interior = (
            (parameter > tolerance)
            & (parameter < 1.0 - tolerance)
            & (distance <= tolerance)
        )
        interior[start_index] = False
        interior[stop_index] = False
        if np.any(interior):
            raise ValueError(
                "Explicit polygon H1 requires matching interface segmentation; "
                "T-junction or hanging-node geometry was detected."
            )


class ExplicitPolygonH1DofMap(StrictModule, NonTrainableState):
    """Padded vertex routes for one conforming explicit polygon field."""

    block_names: tuple[str, ...] = eqx.field(static=True)
    cell_dofs: tuple[Array, ...]
    cell_dof_valid: tuple[Array, ...]
    relations: tuple[RowRelation, ...]
    boundary_dof_mask: Array
    default_dof_points: Array
    global_dof_count: int = eqx.field(static=True)
    local_width: int = eqx.field(static=True)
    dof_map_id: str = eqx.field(static=True)

    def __init__(self, mesh: CellMesh, /):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise TypeError("Explicit polygon H1 requires polygon connectivity.")
        validate_conforming_polygon_segments(mesh)
        vertex_count = int(mesh.coordinates.shape[0])
        used = np.unique(
            np.concatenate(
                tuple(
                    np.asarray(block.vertices, dtype=np.int32).reshape((-1,))
                    for block in mesh.blocks
                )
            )
        )
        if used.size != vertex_count:
            raise ValueError("Explicit polygon H1 requires every mesh vertex to be used.")
        width = max(block.arity for block in mesh.blocks)
        routes = []
        validity = []
        relations = []
        for block in mesh.blocks:
            active = np.asarray(block.vertices, dtype=np.int32)
            padded = np.zeros((block.cell_count, width), dtype=np.int32)
            valid = np.zeros((block.cell_count, width), dtype=bool)
            padded[:, : block.arity] = active
            valid[:, : block.arity] = True
            routes.append(jnp.asarray(padded))
            validity.append(jnp.asarray(valid))
            relations.append(RowRelation(padded, source_size=vertex_count, valid=valid))
        boundary = np.asarray(mesh.connectivity.boundary_vertices, dtype=bool)
        self.block_names = tuple(block.name for block in mesh.blocks)
        self.cell_dofs = tuple(routes)
        self.cell_dof_valid = tuple(validity)
        self.relations = tuple(relations)
        self.boundary_dof_mask = jnp.asarray(boundary)
        self.default_dof_points = jnp.asarray(mesh.coordinates)
        self.global_dof_count = vertex_count
        self.local_width = width
        self.dof_map_id = canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-dof-map",
                "mesh": mesh.topology_id,
                "width": width,
                "routes": [array_tree_fingerprint(value) for value in routes],
                "valid": [array_tree_fingerprint(value) for value in validity],
                "boundary": array_tree_fingerprint(boundary),
            }
        )

    def evaluate_point_coordinates(
        self, coordinates: ArrayLike, mesh: CellMesh, /
    ) -> Array:
        points = jnp.asarray(coordinates)
        if points.shape != mesh.coordinates.shape:
            raise ValueError(
                "Explicit polygon coordinate refresh must preserve coordinate shape."
            )
        return points


__all__ = [
    "ExplicitPolygonH1DofMap",
    "validate_conforming_polygon_segments",
]
