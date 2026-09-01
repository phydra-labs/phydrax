#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cell_complex import (
    polygonal_cell_complex,
    polygonal_connectivity,
    PolygonalConnectivity,
    tetrahedral_cell_complex,
    tetrahedral_connectivity,
    TetrahedralConnectivity,
)
from ._hexahedral import (
    hexahedral_cell_complex,
    hexahedral_connectivity,
    HexahedralConnectivity,
)
from ._support import DiscreteSupport
from ._topology import CellComplexTopology, EntitySet


_CELL_ARITIES = {
    "triangle": 3,
    "quadrilateral": 4,
    "tetrahedron": 4,
    "hexahedron": 8,
}
_CELL_DIMENSIONS = {
    "triangle": 2,
    "quadrilateral": 2,
    "tetrahedron": 3,
    "hexahedron": 3,
    "polygon": 2,
}


class CellBlock(StrictModule, NonTrainableState):
    """One homogeneous, ordered block of top-dimensional cells."""

    name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    vertices: Array
    global_ids: Array
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        cell_kind: str,
        vertices: ArrayLike,
        /,
        *,
        global_ids: ArrayLike | None = None,
    ):
        block_name = str(name)
        kind = str(cell_kind)
        if not block_name:
            raise ValueError("Cell block name must be non-empty.")
        if kind not in (*_CELL_ARITIES, "polygon"):
            raise ValueError(
                "cell_kind must be triangle, quadrilateral, polygon, "
                "tetrahedron, or hexahedron."
            )
        cells = np.asarray(vertices, dtype=np.int32)
        arity = (
            int(cells.shape[1])
            if kind == "polygon" and cells.ndim == 2
            else _CELL_ARITIES.get(kind, -1)
        )
        if (
            cells.ndim != 2
            or cells.shape[0] == 0
            or cells.shape[1] != arity
            or arity < 3
            or (kind == "polygon" and arity < 5)
        ):
            raise ValueError(f"{kind} cell vertices have incompatible arity {arity}.")
        if np.any(cells < 0):
            raise ValueError("Cell vertex indices must be non-negative.")
        if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
            raise ValueError("Each cell must reference distinct vertices.")
        if np.unique(np.sort(cells, axis=1), axis=0).shape[0] != cells.shape[0]:
            raise ValueError("Cell blocks cannot contain duplicate cells.")
        ids = (
            np.arange(cells.shape[0], dtype=np.int64)
            if global_ids is None
            else np.asarray(global_ids, dtype=np.int64)
        )
        if ids.shape != (cells.shape[0],):
            raise ValueError("Cell global_ids must have shape (cell_count,).")
        if np.any(ids < 0) or np.unique(ids).size != ids.size:
            raise ValueError("Cell global_ids must be unique non-negative integers.")
        self.name = block_name
        self.cell_kind = kind
        self.vertices = jnp.asarray(cells)
        self.global_ids = jnp.asarray(ids)
        self.block_id = canonical_fingerprint(
            {
                "kind": "cell-block",
                "name": block_name,
                "cell_kind": kind,
                "vertices": array_tree_fingerprint(cells),
                "global_ids": array_tree_fingerprint(ids),
            }
        )

    @property
    def cell_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def arity(self) -> int:
        return (
            int(self.vertices.shape[1])
            if self.cell_kind == "polygon"
            else _CELL_ARITIES[self.cell_kind]
        )

    @property
    def topological_dimension(self) -> int:
        return _CELL_DIMENSIONS[self.cell_kind]


class CellMesh(StrictModule, NonTrainableState):
    """Canonical computational mesh shared by unstructured discretizations."""

    coordinates: Array
    blocks: tuple[CellBlock, ...]
    vertex_global_ids: Array
    connectivity: PolygonalConnectivity | TetrahedralConnectivity | HexahedralConnectivity
    topology: CellComplexTopology
    support: DiscreteSupport
    topological_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        blocks: Sequence[CellBlock],
        /,
        *,
        vertex_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ):
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("Cell mesh coordinates must have shape (n > 0, d > 0).")
        if not np.all(np.isfinite(points)):
            raise ValueError("Cell mesh coordinates must be finite.")
        normalized_blocks = tuple(blocks)
        if not normalized_blocks:
            raise ValueError("Cell mesh requires at least one cell block.")
        if not all(isinstance(block, CellBlock) for block in normalized_blocks):
            raise TypeError("blocks must contain only CellBlock instances.")
        names = tuple(block.name for block in normalized_blocks)
        if len(set(names)) != len(names):
            raise ValueError("Cell block names must be unique.")
        dimensions = {block.topological_dimension for block in normalized_blocks}
        if len(dimensions) != 1:
            raise ValueError("All cell blocks must share one topological dimension.")
        topological_dimension = dimensions.pop()
        if points.shape[1] < topological_dimension:
            raise ValueError(
                "Cell mesh ambient dimension cannot be smaller than its "
                "topological dimension."
            )
        for block in normalized_blocks:
            if np.any(np.asarray(block.vertices) >= points.shape[0]):
                raise ValueError(
                    f"Cell block {block.name!r} indexes undeclared vertices."
                )
        global_ids = (
            np.arange(points.shape[0], dtype=np.int64)
            if vertex_global_ids is None
            else np.asarray(vertex_global_ids, dtype=np.int64)
        )
        if global_ids.shape != (points.shape[0],):
            raise ValueError("vertex_global_ids must have shape (coordinate_count,).")
        if np.any(global_ids < 0) or np.unique(global_ids).size != global_ids.size:
            raise ValueError("vertex_global_ids must be unique non-negative integers.")
        all_cell_ids = np.concatenate(
            tuple(
                np.asarray(block.global_ids, dtype=np.int64)
                for block in normalized_blocks
            )
        )
        if np.unique(all_cell_ids).size != all_cell_ids.size:
            raise ValueError("Cell global IDs must be unique across mesh blocks.")

        cell_global_ids = np.concatenate(
            tuple(
                np.asarray(block.global_ids, dtype=np.int64)
                for block in normalized_blocks
            )
        )
        if topological_dimension == 2:
            if any(
                block.cell_kind not in ("triangle", "quadrilateral", "polygon")
                for block in normalized_blocks
            ):
                raise ValueError("Two-dimensional meshes support polygonal blocks only.")
            arities = tuple(block.arity for block in normalized_blocks)
            if arities != tuple(sorted(arities)):
                raise ValueError(
                    "Polygonal CellMesh blocks must be ordered by increasing arity."
                )
            triangles = [
                np.asarray(block.vertices, dtype=np.int32)
                for block in normalized_blocks
                if block.cell_kind == "triangle"
            ]
            quadrilaterals = [
                np.asarray(block.vertices, dtype=np.int32)
                for block in normalized_blocks
                if block.cell_kind == "quadrilateral"
            ]
            polygons = tuple(
                np.asarray(block.vertices, dtype=np.int32)
                for block in normalized_blocks
                if block.cell_kind == "polygon"
            )
            triangle_cells = np.concatenate(triangles, axis=0) if triangles else None
            quadrilateral_cells = (
                np.concatenate(quadrilaterals, axis=0) if quadrilaterals else None
            )
            connectivity = polygonal_connectivity(
                triangle_cells,
                quadrilateral_cells,
                points.shape[0],
                polygons=polygons,
            )
            topology = polygonal_cell_complex(
                triangle_cells,
                quadrilateral_cells,
                points.shape[0],
                polygons=polygons,
                vertex_global_ids=global_ids,
                cell_global_ids=cell_global_ids,
            )
        else:
            if len(normalized_blocks) != 1:
                raise ValueError(
                    "Three-dimensional CellMesh requires one homogeneous block."
                )
            block = normalized_blocks[0]
            if block.cell_kind == "tetrahedron":
                tetrahedra = np.asarray(block.vertices, dtype=np.int32)
                connectivity = tetrahedral_connectivity(tetrahedra, points.shape[0])
                topology = tetrahedral_cell_complex(
                    tetrahedra,
                    points.shape[0],
                    vertex_global_ids=global_ids,
                    cell_global_ids=cell_global_ids,
                )
            elif block.cell_kind == "hexahedron":
                hexahedra = np.asarray(block.vertices, dtype=np.int32)
                connectivity = hexahedral_connectivity(hexahedra, points.shape[0])
                topology = hexahedral_cell_complex(
                    hexahedra,
                    points.shape[0],
                    vertex_global_ids=global_ids,
                    cell_global_ids=cell_global_ids,
                )
            else:
                raise ValueError("Unsupported three-dimensional cell block.")

        canonical_blocks = []
        for block in normalized_blocks:
            block_ids = np.asarray(block.global_ids, dtype=np.int64)
            order = np.argsort(block_ids, kind="stable")
            global_vertices = global_ids[np.asarray(block.vertices, dtype=np.int32)]
            canonical_blocks.append(
                {
                    "name": block.name,
                    "cell_kind": block.cell_kind,
                    "global_ids": array_tree_fingerprint(block_ids[order]),
                    "global_vertices": array_tree_fingerprint(global_vertices[order]),
                }
            )

        topology_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-topology",
                "topological_dimension": topological_dimension,
                "vertex_global_ids": array_tree_fingerprint(global_ids),
                "blocks": canonical_blocks,
                "cell_complex": topology.topology_id,
            }
        )
        geometry_layout_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-geometry-layout",
                "topology": topology_id,
                "ambient_dimension": points.shape[1],
                "coordinate_count": points.shape[0],
                "coordinate_dtype": str(points.dtype),
            }
        )
        geometry_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-geometry",
                "layout": geometry_layout_id,
                "coordinates": array_tree_fingerprint(points),
                "numeric_version": str(numeric_version),
            }
        )
        support = DiscreteSupport(topology, points.shape[1], geometry_layout_id)
        self.coordinates = jnp.asarray(points)
        self.blocks = normalized_blocks
        self.vertex_global_ids = jnp.asarray(global_ids)
        self.topology = topology
        self.connectivity = connectivity
        self.support = support
        self.topological_dimension = topological_dimension
        self.ambient_dimension = points.shape[1]
        self.topology_id = topology_id
        self.geometry_layout_id = geometry_layout_id
        self.geometry_id = geometry_id
        self.mesh_id = canonical_fingerprint(
            {
                "kind": "cell-mesh",
                "topology": topology_id,
                "geometry": geometry_id,
            }
        )
        self.numeric_version = str(numeric_version)

    @classmethod
    def from_triangles(
        cls,
        coordinates: ArrayLike,
        triangles: ArrayLike,
        /,
        *,
        block_name: str = "triangles",
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        return cls(
            coordinates,
            (
                CellBlock(
                    block_name,
                    "triangle",
                    triangles,
                    global_ids=cell_global_ids,
                ),
            ),
            vertex_global_ids=vertex_global_ids,
            numeric_version=numeric_version,
        )

    @classmethod
    def from_polygons(
        cls,
        coordinates: ArrayLike,
        polygons: Sequence[ArrayLike],
        /,
        *,
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        """Build a canonical mixed-arity polygon mesh from cyclic vertex loops."""

        points = np.asarray(coordinates, dtype=float)
        loops = tuple(np.asarray(loop, dtype=np.int32) for loop in polygons)
        if not loops:
            raise ValueError("from_polygons requires at least one cell.")
        if any(loop.ndim != 1 or loop.size < 3 for loop in loops):
            raise ValueError(
                "Every polygon must be one rank-1 loop with at least 3 vertices."
            )
        identifiers = (
            np.arange(len(loops), dtype=np.int64)
            if cell_global_ids is None
            else np.asarray(cell_global_ids, dtype=np.int64)
        )
        if identifiers.shape != (len(loops),):
            raise ValueError("cell_global_ids must have one ID per polygon.")
        grouped: dict[int, list[tuple[np.ndarray, int]]] = {}
        for loop, identifier in zip(loops, identifiers, strict=True):
            if np.any(loop < 0) or np.any(loop >= points.shape[0]):
                raise ValueError("Polygon loop indexes an undeclared vertex.")
            cell_points = points[loop]
            area2 = float(
                np.sum(
                    cell_points[:, 0] * np.roll(cell_points[:, 1], -1)
                    - np.roll(cell_points[:, 0], -1) * cell_points[:, 1]
                )
            )
            if not np.isfinite(area2) or area2 == 0.0:
                raise ValueError("Polygon loops require finite nonzero signed area.")
            oriented = loop[::-1] if area2 < 0.0 else loop
            grouped.setdefault(int(loop.size), []).append((oriented, int(identifier)))
        blocks = []
        for arity in sorted(grouped):
            entries = grouped[arity]
            kind = (
                "triangle" if arity == 3 else "quadrilateral" if arity == 4 else "polygon"
            )
            blocks.append(
                CellBlock(
                    f"polygons-{arity}",
                    kind,
                    np.stack(tuple(entry[0] for entry in entries)),
                    global_ids=np.asarray(
                        tuple(entry[1] for entry in entries), dtype=np.int64
                    ),
                )
            )
        return cls(
            points,
            tuple(blocks),
            vertex_global_ids=vertex_global_ids,
            numeric_version=numeric_version,
        )

    @classmethod
    def from_tetrahedra(
        cls,
        coordinates: ArrayLike,
        tetrahedra: ArrayLike,
        /,
        *,
        block_name: str = "tetrahedra",
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        return cls(
            coordinates,
            (
                CellBlock(
                    block_name,
                    "tetrahedron",
                    tetrahedra,
                    global_ids=cell_global_ids,
                ),
            ),
            vertex_global_ids=vertex_global_ids,
            numeric_version=numeric_version,
        )

    def block(self, name: str, /) -> CellBlock:
        requested = str(name)
        for block in self.blocks:
            if block.name == requested:
                return block
        raise KeyError(f"Unknown cell block {requested!r}.")

    def entity_set(self, dimension: int, /) -> EntitySet:
        target = int(dimension)
        for entities in self.topology.entity_sets:
            if entities.intrinsic_dimension == target:
                return entities
        raise KeyError(f"Cell mesh has no entity set of dimension {target}.")

    def with_coordinates(
        self,
        coordinates: ArrayLike,
        /,
        *,
        numeric_version: str,
    ) -> CellMesh:
        points = jnp.asarray(coordinates)
        if points.shape != self.coordinates.shape:
            raise ValueError(
                "Fixed-topology coordinate refresh must preserve coordinate shape."
            )
        return CellMesh(
            points,
            self.blocks,
            vertex_global_ids=self.vertex_global_ids,
            numeric_version=numeric_version,
        )


__all__ = ["CellBlock", "CellMesh"]
