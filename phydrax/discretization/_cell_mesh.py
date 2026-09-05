#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cell_complex import (
    interval_cell_complex,
    interval_connectivity,
    IntervalConnectivity,
    polygonal_cell_complex,
    polygonal_connectivity,
    PolygonalConnectivity,
    polyhedral_cell_complex,
    polyhedral_connectivity as _build_polyhedral_connectivity,
    PolyhedralConnectivity,
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
    "interval": 2,
    "triangle": 3,
    "quadrilateral": 4,
    "tetrahedron": 4,
    "hexahedron": 8,
    "prism": 6,
    "pyramid": 5,
}
_CELL_DIMENSIONS = {
    "interval": 1,
    "triangle": 2,
    "quadrilateral": 2,
    "tetrahedron": 3,
    "hexahedron": 3,
    "prism": 3,
    "pyramid": 3,
    "polygon": 2,
    "polyhedron": 3,
}


class CellBlock(StrictModule, NonTrainableState):
    """One homogeneous, ordered block of top-dimensional cells."""

    name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    vertices: Array
    vertex_valid: Array
    global_ids: Array
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        cell_kind: str,
        vertices: ArrayLike,
        /,
        *,
        vertex_valid: ArrayLike | None = None,
        global_ids: ArrayLike | None = None,
    ):
        block_name = str(name)
        kind = str(cell_kind)
        if not block_name:
            raise ValueError("Cell block name must be non-empty.")
        if kind not in (*_CELL_ARITIES, "polygon", "polyhedron"):
            raise ValueError(
                "cell_kind must be interval, triangle, quadrilateral, polygon, "
                "tetrahedron, hexahedron, prism, pyramid, or polyhedron."
            )
        cells = np.asarray(vertices, dtype=np.int32)
        arity = (
            int(cells.shape[1])
            if kind in ("polygon", "polyhedron") and cells.ndim == 2
            else _CELL_ARITIES.get(kind, -1)
        )
        minimum_arity = 4 if kind == "polyhedron" else 2 if kind == "interval" else 3
        if (
            cells.ndim != 2
            or cells.shape[0] == 0
            or cells.shape[1] != arity
            or arity < minimum_arity
            or (kind == "polygon" and arity < 5)
        ):
            raise ValueError(f"{kind} cell vertices have incompatible arity {arity}.")
        valid = (
            np.ones_like(cells, dtype=bool)
            if vertex_valid is None
            else np.asarray(vertex_valid, dtype=bool)
        )
        if valid.shape != cells.shape:
            raise ValueError("vertex_valid must match cell vertex storage.")
        if kind != "polyhedron" and not np.all(valid):
            raise ValueError("Only polyhedron blocks may contain padded vertices.")
        if np.any(np.sum(valid, axis=1) < minimum_arity):
            raise ValueError(
                f"Each {kind} cell requires at least {minimum_arity} vertices."
            )
        if np.any(cells[valid] < 0):
            raise ValueError("Cell vertex indices must be non-negative.")
        canonical_cells = []
        for row, mask in zip(cells, valid, strict=True):
            active = row[mask]
            if np.unique(active).size != active.size:
                raise ValueError("Each cell must reference distinct active vertices.")
            canonical_cells.append(tuple(sorted(int(value) for value in active)))
        if len(set(canonical_cells)) != len(canonical_cells):
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
        self.vertex_valid = jnp.asarray(valid)
        self.global_ids = jnp.asarray(ids)
        self.block_id = canonical_fingerprint(
            {
                "kind": "cell-block",
                "name": block_name,
                "cell_kind": kind,
                "vertices": array_tree_fingerprint(cells),
                "vertex_valid": array_tree_fingerprint(valid),
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
            if self.cell_kind in ("polygon", "polyhedron")
            else _CELL_ARITIES[self.cell_kind]
        )

    @property
    def topological_dimension(self) -> int:
        return _CELL_DIMENSIONS[self.cell_kind]


class PolyhedralBlock(StrictModule, NonTrainableState):
    """One exact-width block of arbitrary polyhedra grouped by vertex count."""

    name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    vertices: Array
    vertex_valid: Array
    global_ids: Array
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        vertices: ArrayLike,
        /,
        *,
        global_ids: ArrayLike | None = None,
    ):
        block_name = str(name)
        cells = np.asarray(vertices, dtype=np.int32)
        if not block_name:
            raise ValueError("Polyhedral block name must be non-empty.")
        if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] < 4:
            raise ValueError(
                "Polyhedral block vertices must have shape (cells > 0, arity >= 4)."
            )
        if np.any(cells < 0):
            raise ValueError("Polyhedral cell vertex indices must be non-negative.")
        canonical = []
        for row in cells:
            if np.unique(row).size != row.size:
                raise ValueError("Each polyhedral cell must reference distinct vertices.")
            canonical.append(tuple(sorted(int(value) for value in row)))
        if len(set(canonical)) != len(canonical):
            raise ValueError("Polyhedral blocks cannot contain duplicate cells.")
        ids = (
            np.arange(cells.shape[0], dtype=np.int64)
            if global_ids is None
            else np.asarray(global_ids, dtype=np.int64)
        )
        if ids.shape != (cells.shape[0],):
            raise ValueError("Polyhedral global_ids must match the cell count.")
        if np.any(ids < 0) or np.unique(ids).size != ids.size:
            raise ValueError(
                "Polyhedral global_ids must be unique non-negative integers."
            )
        valid = np.ones_like(cells, dtype=bool)
        self.name = block_name
        self.cell_kind = "polyhedron"
        self.vertices = jnp.asarray(cells)
        self.vertex_valid = jnp.asarray(valid)
        self.global_ids = jnp.asarray(ids)
        self.block_id = canonical_fingerprint(
            {
                "kind": "polyhedral-block",
                "name": block_name,
                "vertices": array_tree_fingerprint(cells),
                "global_ids": array_tree_fingerprint(ids),
            }
        )

    @property
    def cell_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def arity(self) -> int:
        return int(self.vertices.shape[1])

    @property
    def topological_dimension(self) -> int:
        return 3


class CellMesh(StrictModule, NonTrainableState):
    """Canonical computational mesh shared by unstructured discretizations."""

    coordinates: Array
    blocks: tuple[CellBlock | PolyhedralBlock, ...]
    vertex_global_ids: Array
    connectivity: (
        IntervalConnectivity
        | PolygonalConnectivity
        | TetrahedralConnectivity
        | HexahedralConnectivity
        | PolyhedralConnectivity
    )
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
        blocks: Sequence[CellBlock | PolyhedralBlock],
        /,
        *,
        vertex_global_ids: ArrayLike | None = None,
        entity_global_ids: Mapping[int, ArrayLike] | None = None,
        polyhedral_connectivity: PolyhedralConnectivity | None = None,
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
        if not all(
            isinstance(block, (CellBlock, PolyhedralBlock)) for block in normalized_blocks
        ):
            raise TypeError(
                "blocks must contain only CellBlock or PolyhedralBlock instances."
            )
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
            vertices_ = np.asarray(block.vertices)
            valid_ = np.asarray(block.vertex_valid, dtype=bool)
            if np.any(vertices_[valid_] >= points.shape[0]):
                raise ValueError(
                    f"Cell block {block.name!r} indexes undeclared vertices."
                )
        entity_ids = (
            {}
            if entity_global_ids is None
            else {
                int(dimension): np.asarray(values, dtype=np.int64)
                for dimension, values in entity_global_ids.items()
            }
        )
        if any(
            dimension < 0 or dimension > topological_dimension for dimension in entity_ids
        ):
            raise ValueError("entity_global_ids contains an undeclared dimension.")
        supplied_vertices = entity_ids.get(0)
        if supplied_vertices is not None and vertex_global_ids is not None:
            if not np.array_equal(
                supplied_vertices, np.asarray(vertex_global_ids, dtype=np.int64)
            ):
                raise ValueError(
                    "vertex_global_ids contradicts entity_global_ids dimension zero."
                )
        global_ids = (
            supplied_vertices
            if supplied_vertices is not None
            else (
                np.arange(points.shape[0], dtype=np.int64)
                if vertex_global_ids is None
                else np.asarray(vertex_global_ids, dtype=np.int64)
            )
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
        supplied_cells = entity_ids.get(topological_dimension)
        if supplied_cells is not None and not np.array_equal(
            supplied_cells, cell_global_ids
        ):
            raise ValueError(
                "Top-dimensional entity_global_ids contradict cell block global IDs."
            )
        if polyhedral_connectivity is not None and topological_dimension != 3:
            raise ValueError(
                "polyhedral_connectivity is valid only for three-dimensional meshes."
            )
        if topological_dimension == 1:
            if any(block.cell_kind != "interval" for block in normalized_blocks):
                raise ValueError("One-dimensional meshes support interval blocks only.")
            intervals = np.concatenate(
                tuple(
                    np.asarray(block.vertices, dtype=np.int32)
                    for block in normalized_blocks
                ),
                axis=0,
            )
            connectivity = interval_connectivity(intervals, points.shape[0])
            topology = interval_cell_complex(
                intervals,
                points.shape[0],
                vertex_global_ids=global_ids,
                cell_global_ids=cell_global_ids,
            )
        elif topological_dimension == 2:
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
                edge_global_ids=entity_ids.get(1),
                cell_global_ids=cell_global_ids,
            )
        else:
            if polyhedral_connectivity is not None:
                connectivity = polyhedral_connectivity
                if (
                    connectivity.vertex_count != points.shape[0]
                    or connectivity.cell_count
                    != sum(block.cell_count for block in normalized_blocks)
                    or not np.array_equal(
                        np.asarray(connectivity.vertex_global_ids), global_ids
                    )
                    or (
                        entity_ids.get(1) is not None
                        and not np.array_equal(
                            entity_ids[1],
                            np.asarray(connectivity.edge_global_ids),
                        )
                    )
                    or (
                        entity_ids.get(2) is not None
                        and not np.array_equal(
                            entity_ids[2],
                            np.asarray(connectivity.face_global_ids),
                        )
                    )
                    or not np.array_equal(
                        np.asarray(connectivity.cell_global_ids), cell_global_ids
                    )
                ):
                    raise ValueError(
                        "PolyhedralConnectivity does not match the CellMesh IDs."
                    )
                offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
                values = np.asarray(connectivity.cell_vertex_values, dtype=np.int32)
                cell_offset = 0
                for block in normalized_blocks:
                    widths = np.diff(
                        offsets[cell_offset : cell_offset + block.cell_count + 1]
                    )
                    if np.any(widths != block.arity):
                        raise ValueError(
                            "PolyhedralConnectivity cell widths do not match blocks."
                        )
                    start = int(offsets[cell_offset])
                    stop = int(offsets[cell_offset + block.cell_count])
                    expected = np.asarray(block.vertices, dtype=np.int32)
                    if not np.array_equal(
                        values[start:stop].reshape(expected.shape),
                        np.sort(expected, axis=1),
                    ):
                        raise ValueError(
                            "PolyhedralConnectivity cell vertices do not match blocks."
                        )
                    cell_offset += block.cell_count
                topology = polyhedral_cell_complex(connectivity)
            elif (
                len(normalized_blocks) == 1
                and normalized_blocks[0].cell_kind == "tetrahedron"
            ):
                block = normalized_blocks[0]
                tetrahedra = np.asarray(block.vertices, dtype=np.int32)
                connectivity = tetrahedral_connectivity(tetrahedra, points.shape[0])
                topology = tetrahedral_cell_complex(
                    tetrahedra,
                    points.shape[0],
                    vertex_global_ids=global_ids,
                    edge_global_ids=entity_ids.get(1),
                    face_global_ids=entity_ids.get(2),
                    cell_global_ids=cell_global_ids,
                )
            elif (
                len(normalized_blocks) == 1
                and normalized_blocks[0].cell_kind == "hexahedron"
            ):
                block = normalized_blocks[0]
                hexahedra = np.asarray(block.vertices, dtype=np.int32)
                connectivity = hexahedral_connectivity(hexahedra, points.shape[0])
                topology = hexahedral_cell_complex(
                    hexahedra,
                    points.shape[0],
                    vertex_global_ids=global_ids,
                    edge_global_ids=entity_ids.get(1),
                    face_global_ids=entity_ids.get(2),
                    cell_global_ids=cell_global_ids,
                )
            elif any(block.cell_kind == "polyhedron" for block in normalized_blocks):
                raise ValueError(
                    "Polyhedron blocks require matching PolyhedralConnectivity."
                )
            else:
                polyhedral_blocks = tuple(
                    (block.cell_kind, np.asarray(block.vertices, dtype=np.int32))
                    for block in normalized_blocks
                )
                connectivity = _build_polyhedral_connectivity(
                    polyhedral_blocks,
                    points.shape[0],
                    vertex_global_ids=global_ids,
                    edge_global_ids=entity_ids.get(1),
                    face_global_ids=entity_ids.get(2),
                    cell_global_ids=cell_global_ids,
                )
                topology = polyhedral_cell_complex(connectivity)

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
                    "vertex_valid": array_tree_fingerprint(
                        np.asarray(block.vertex_valid)[order]
                    ),
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

    @classmethod
    def from_polyhedra(
        cls,
        coordinates: ArrayLike,
        cells: Sequence[Sequence[ArrayLike]],
        /,
        *,
        block_name: str = "polyhedra",
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        """Build a canonical mesh from closed outward-oriented face loops."""

        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "Polyhedral coordinates must have shape (vertex_count, d >= 3)."
            )
        normalized_cells = tuple(tuple(cell) for cell in cells)
        if not normalized_cells:
            raise ValueError("At least one polyhedral cell is required.")
        source_ids = (
            np.arange(len(normalized_cells), dtype=np.int64)
            if cell_global_ids is None
            else np.asarray(cell_global_ids, dtype=np.int64)
        )
        if source_ids.shape != (len(normalized_cells),):
            raise ValueError("cell_global_ids must match the polyhedral cell count.")
        vertex_counts = np.asarray(
            [
                len(
                    {
                        int(vertex)
                        for face in cell
                        for vertex in np.asarray(face, dtype=np.int32)
                    }
                )
                for cell in normalized_cells
            ],
            dtype=np.int32,
        )
        order = np.argsort(vertex_counts, kind="stable")
        ordered_cells = tuple(normalized_cells[int(index)] for index in order)
        ordered_ids = source_ids[order]
        ordered_counts = vertex_counts[order]
        connectivity = _build_polyhedral_connectivity(
            ordered_cells,
            points.shape[0],
            vertex_global_ids=vertex_global_ids,
            cell_global_ids=ordered_ids,
        )
        offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
        values = np.asarray(connectivity.cell_vertex_values, dtype=np.int32)
        blocks = []
        start_cell = 0
        unique_counts = tuple(int(value) for value in np.unique(ordered_counts))
        for vertex_count in unique_counts:
            stop_cell = start_cell + int(np.sum(ordered_counts == vertex_count))
            start = int(offsets[start_cell])
            stop = int(offsets[stop_cell])
            name = (
                str(block_name)
                if len(unique_counts) == 1
                else f"{block_name}-{vertex_count}"
            )
            blocks.append(
                PolyhedralBlock(
                    name,
                    values[start:stop].reshape((-1, vertex_count)),
                    global_ids=ordered_ids[start_cell:stop_cell],
                )
            )
            start_cell = stop_cell
        return cls(
            points,
            tuple(blocks),
            vertex_global_ids=connectivity.vertex_global_ids,
            polyhedral_connectivity=connectivity,
            numeric_version=numeric_version,
        )

    @classmethod
    def from_mixed_3d(
        cls,
        coordinates: ArrayLike,
        blocks: Sequence[CellBlock],
        /,
        *,
        polyhedra: Mapping[str, Sequence[Sequence[ArrayLike]]],
        vertex_global_ids: ArrayLike | None = None,
        polyhedral_cell_global_ids: Mapping[str, ArrayLike] | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        """Build one mixed standard/polyhedral three-dimensional mesh."""

        points = np.asarray(coordinates, dtype=float)
        standard_blocks = tuple(blocks)
        if any(
            block.cell_kind not in ("tetrahedron", "hexahedron", "prism", "pyramid")
            for block in standard_blocks
        ):
            raise ValueError("Mixed 3-D dense blocks use standard volume cell kinds.")
        named_cells = tuple(
            (str(name), tuple(tuple(cell) for cell in cells))
            for name, cells in sorted(polyhedra.items())
        )
        if any(not name or not cells for name, cells in named_cells):
            raise ValueError(
                "Polyhedral block names and cell collections must be non-empty."
            )
        id_mapping = (
            {}
            if polyhedral_cell_global_ids is None
            else {
                str(name): np.asarray(values, dtype=np.int64)
                for name, values in polyhedral_cell_global_ids.items()
            }
        )
        if set(id_mapping) not in (set(), {name for name, _ in named_cells}):
            raise ValueError(
                "polyhedral_cell_global_ids must cover every polyhedral block."
            )
        used_ids = [
            int(value)
            for block in standard_blocks
            for value in np.asarray(block.global_ids, dtype=np.int64)
        ]
        next_id = max(used_ids, default=-1) + 1
        poly_blocks: list[PolyhedralBlock] = []
        explicit_cells: list[tuple[ArrayLike, ...]] = []
        explicit_ids: list[int] = []
        for name, cells in named_cells:
            ids = id_mapping.get(name)
            if ids is None:
                ids = np.arange(next_id, next_id + len(cells), dtype=np.int64)
            if ids.shape != (len(cells),):
                raise ValueError(
                    f"Polyhedral global IDs for {name!r} must match its cell count."
                )
            counts = np.asarray(
                [
                    len(
                        {
                            int(vertex)
                            for face in cell
                            for vertex in np.asarray(face, dtype=np.int32)
                        }
                    )
                    for cell in cells
                ],
                dtype=np.int32,
            )
            order = np.argsort(counts, kind="stable")
            ordered_cells = tuple(cells[int(index)] for index in order)
            ordered_ids = ids[order]
            ordered_counts = counts[order]
            for vertex_count in tuple(int(value) for value in np.unique(ordered_counts)):
                selected = np.flatnonzero(ordered_counts == vertex_count)
                selected_cells = tuple(ordered_cells[int(index)] for index in selected)
                selected_ids = ordered_ids[selected]
                vertices = np.asarray(
                    [
                        sorted(
                            {
                                int(vertex)
                                for face in cell
                                for vertex in np.asarray(face, dtype=np.int32)
                            }
                        )
                        for cell in selected_cells
                    ],
                    dtype=np.int32,
                )
                block_name = (
                    name
                    if len(np.unique(ordered_counts)) == 1
                    else f"{name}-{vertex_count}"
                )
                poly_blocks.append(
                    PolyhedralBlock(
                        block_name,
                        vertices,
                        global_ids=selected_ids,
                    )
                )
                explicit_cells.extend(selected_cells)
                explicit_ids.extend(int(value) for value in selected_ids)
            next_id = max(next_id, int(np.max(ids, initial=next_id - 1)) + 1)
        combined_blocks = (*standard_blocks, *poly_blocks)
        if not combined_blocks:
            raise ValueError("Mixed 3-D mesh requires at least one cell block.")
        entries: list[Sequence[ArrayLike] | tuple[str, ArrayLike]] = [
            (block.cell_kind, np.asarray(block.vertices, dtype=np.int32))
            for block in standard_blocks
        ]
        entries.extend(explicit_cells)
        all_ids = np.concatenate(
            (
                *(
                    np.asarray(block.global_ids, dtype=np.int64)
                    for block in standard_blocks
                ),
                np.asarray(explicit_ids, dtype=np.int64),
            )
        )
        connectivity = _build_polyhedral_connectivity(
            entries,
            points.shape[0],
            vertex_global_ids=vertex_global_ids,
            cell_global_ids=all_ids,
        )
        return cls(
            points,
            combined_blocks,
            vertex_global_ids=connectivity.vertex_global_ids,
            polyhedral_connectivity=connectivity,
            numeric_version=numeric_version,
        )

    def block(self, name: str, /) -> CellBlock | PolyhedralBlock:
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
            entity_global_ids={
                entities.intrinsic_dimension: entities.entity_ids
                for entities in self.topology.entity_sets
            },
            polyhedral_connectivity=(
                self.connectivity
                if any(isinstance(block, PolyhedralBlock) for block in self.blocks)
                else None
            ),
            numeric_version=numeric_version,
        )


__all__ = ["CellBlock", "CellMesh", "PolyhedralBlock"]
