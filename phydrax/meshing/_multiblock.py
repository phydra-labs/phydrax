#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit conforming gluing for structured block constructions."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry import CellGeometryElement
from ..discretization.fem._reference import FiniteElementSpec
from ._controls import BlockInterfaceControl
from ._structured import StructuredConstruction


@dataclass(frozen=True, slots=True)
class MultiblockConstruction:
    """One canonical mesh plus block vertex maps and source-cell ancestry."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    block_vertex_ids: tuple[tuple[str, np.ndarray], ...]
    block_cell_parent_ids: tuple[tuple[str, np.ndarray], ...]
    interface_ids: tuple[str, ...]
    maximum_gluing_residual: float


class _DisjointSet:
    def __init__(self, count: int, /) -> None:
        self.parent = np.arange(count, dtype=np.int64)
        self.rank = np.zeros(count, dtype=np.int8)
        self.minimum = np.arange(count, dtype=np.int64)

    def find(self, item: int, /) -> int:
        root = item
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while item != root:
            next_item = int(self.parent[item])
            self.parent[item] = root
            item = next_item
        return root

    def union(self, first: int, second: int, /) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        self.minimum[left] = min(int(self.minimum[left]), int(self.minimum[right]))
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1

    def inverse(self) -> tuple[np.ndarray, np.ndarray]:
        roots = np.empty(self.parent.size, dtype=np.int64)
        for item in range(self.parent.size):
            root = self.find(item)
            roots[item] = self.minimum[root]
        representatives, inverse = np.unique(roots, return_inverse=True)
        return representatives, inverse.astype(np.int64, copy=False)


def logical_face_vertices(shape: tuple[int, ...], face: int, /) -> np.ndarray:
    """Return C-order node IDs on one declared logical block face."""
    if len(shape) not in (2, 3) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 2 for size in shape
    ):
        raise ValueError("A logical block shape must have two or three nontrivial axes.")
    if (
        isinstance(face, bool)
        or not isinstance(face, int)
        or not 0 <= face < 2 * len(shape)
    ):
        raise ValueError("A logical face index must belong to its block dimension.")
    axis, side = divmod(face, 2)
    strides = tuple(prod(shape[index + 1 :]) for index in range(len(shape)))
    free = tuple(index for index in range(len(shape)) if index != axis)
    ids = np.full(
        tuple(shape[index] for index in free),
        (0 if side == 0 else shape[axis] - 1) * strides[axis],
        dtype=np.int64,
    )
    for position, logical_axis in enumerate(free):
        reshape = [1] * len(free)
        reshape[position] = shape[logical_axis]
        ids += (
            np.arange(shape[logical_axis], dtype=np.int64) * strides[logical_axis]
        ).reshape(reshape)
    return ids


def _face_correspondence(
    first: StructuredConstruction,
    second: StructuredConstruction,
    interface: BlockInterfaceControl,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if len(first.logical_shape) != len(second.logical_shape):
        raise ValueError(
            "Conforming interfaces require blocks of the same topological dimension."
        )
    first_ids = logical_face_vertices(first.logical_shape, interface.first_face)
    second_ids = logical_face_vertices(second.logical_shape, interface.second_face)
    if (
        len(interface.permutation) != first_ids.ndim
        or len(interface.flips) != first_ids.ndim
    ):
        raise ValueError("Interface permutation and flips must match the face dimension.")
    second_ids = second_ids.transpose(interface.permutation)
    for axis, flip in enumerate(interface.flips):
        if flip:
            second_ids = np.flip(second_ids, axis=axis)
    if first_ids.shape != second_ids.shape:
        raise ValueError(
            f"Conforming face node counts differ after the declared permutation: {first_ids.shape} and {second_ids.shape}."
        )
    return first_ids.reshape(-1), second_ids.reshape(-1)


def _reconcile_pairs(
    points: np.ndarray,
    pairs: tuple[tuple[np.ndarray, np.ndarray, float, str], ...],
    /,
) -> tuple[np.ndarray, np.ndarray, float]:
    sets = _DisjointSet(points.shape[0])
    maximum_residual = 0.0
    for first, second, tolerance, identity in pairs:
        distances = np.linalg.norm(points[first] - points[second], axis=-1)
        residual = float(np.max(distances, initial=0.0))
        if residual > tolerance:
            raise ValueError(
                f"Declared interface {identity!r} has coordinate residual {residual}, exceeding {tolerance}."
            )
        maximum_residual = max(maximum_residual, residual)
        for left, right in zip(first, second, strict=True):
            sets.union(int(left), int(right))
    representatives, inverse = sets.inverse()
    for first, second, tolerance, identity in pairs:
        for rows in (first, second):
            final_distances = np.linalg.norm(
                points[rows] - points[representatives[inverse[rows]]], axis=-1
            )
            displacement = float(np.max(final_distances, initial=0.0))
            if displacement > tolerance:
                raise ValueError(
                    f"Explicit interface {identity!r} would move a node by {displacement}, exceeding {tolerance}."
                )
            maximum_residual = max(maximum_residual, displacement)
    return representatives, inverse, maximum_residual


def _geometry_face_shape(
    construction: StructuredConstruction,
    element: CellGeometryElement,
    /,
) -> tuple[int, ...]:
    if not isinstance(element, FiniteElementSpec) or element.degree < 1:
        raise ValueError(
            "Structured coordinate maps require a positive-degree polynomial element."
        )
    return tuple((size - 1) * element.degree + 1 for size in construction.logical_shape)


def construct_glued_structured_blocks(
    blocks: tuple[StructuredConstruction, ...],
    interfaces: tuple[BlockInterfaceControl, ...],
    /,
) -> MultiblockConstruction:
    """Merge only node pairs named by conforming interface controls."""
    if not blocks:
        raise ValueError("At least one structured block is required.")
    ordered = tuple(sorted(blocks, key=lambda block: block.mesh.blocks[0].name))
    names = tuple(block.mesh.blocks[0].name for block in ordered)
    if len(set(names)) != len(names):
        raise ValueError("Structured block names must be unique.")
    if any(len(block.mesh.blocks) != 1 for block in ordered):
        raise ValueError(
            "Each structured construction must contain exactly one cell block."
        )
    dimensions = {
        (block.mesh.topological_dimension, block.mesh.ambient_dimension)
        for block in ordered
    }
    if len(dimensions) != 1:
        raise ValueError(
            "Conforming structured blocks must share topology and ambient dimensions."
        )
    by_name = dict(zip(names, ordered, strict=True))
    occupied: set[tuple[str, int]] = set()
    vertex_offsets: dict[str, int] = {}
    vertex_count = 0
    for name, construction in zip(names, ordered, strict=True):
        if prod(construction.logical_shape) != construction.mesh.coordinates.shape[0]:
            raise ValueError(
                f"Block {name!r} logical shape does not match its mesh vertex count."
            )
        vertex_offsets[name] = vertex_count
        vertex_count += construction.mesh.coordinates.shape[0]
    if vertex_count - 1 > np.iinfo(np.int32).max:
        raise OverflowError(
            "Glued block vertex identities exceed int32 connectivity range."
        )
    local_points = tuple(
        np.asarray(block.mesh.coordinates, dtype=np.float64) for block in ordered
    )
    points = np.concatenate(local_points, axis=0)
    vertex_pairs: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    geometry_pair_records: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    geometry_banks: list[np.ndarray] = []
    geometry_elements: dict[str, CellGeometryElement] = {}
    local_geometry_routes: dict[str, np.ndarray] = {}
    geometry_offsets: dict[str, int] = {}
    geometry_count = 0
    geometry_shapes: dict[str, tuple[int, ...]] = {}
    for construction in ordered:
        name = construction.mesh.blocks[0].name
        elements, routes, coordinates = construction.geometry.resolve(construction.mesh)
        if len(elements) != 1:
            raise ValueError("A structured block must have one coordinate element.")
        element = elements[0]
        geometry_elements[name] = element
        local_geometry_routes[name] = np.asarray(routes[0], dtype=np.int32)
        values = np.asarray(coordinates, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != ordered[0].mesh.ambient_dimension:
            raise ValueError(
                "Structured block coordinate maps must match the mesh ambient dimension."
            )
        geometry_offsets[name] = geometry_count
        geometry_count += values.shape[0]
        geometry_banks.append(values)
        geometry_shapes[name] = _geometry_face_shape(construction, element)
    if geometry_count - 1 > np.iinfo(np.int32).max:
        raise OverflowError("Glued coordinate-map identities exceed int32 route range.")
    geometry_points = np.concatenate(geometry_banks, axis=0)
    for interface in interfaces:
        if not isinstance(interface, BlockInterfaceControl) or not interface.conforming:
            raise ValueError(
                "Only explicitly declared conforming block interfaces can be glued."
            )
        if interface.first_block not in by_name or interface.second_block not in by_name:
            raise ValueError("Every interface must name two constructed blocks.")
        first_endpoint = (interface.first_block, interface.first_face)
        second_endpoint = (interface.second_block, interface.second_face)
        if (
            first_endpoint == second_endpoint
            or first_endpoint in occupied
            or second_endpoint in occupied
        ):
            raise ValueError(
                "Each logical block face may participate in only one gluing."
            )
        occupied.update((first_endpoint, second_endpoint))
        first_block = by_name[interface.first_block]
        second_block = by_name[interface.second_block]
        first_local, second_local = _face_correspondence(
            first_block, second_block, interface
        )
        vertex_pairs.append(
            (
                first_local + vertex_offsets[interface.first_block],
                second_local + vertex_offsets[interface.second_block],
                interface.tolerance,
                interface.control_id,
            )
        )
        first_geometry_shape = geometry_shapes[interface.first_block]
        second_geometry_shape = geometry_shapes[interface.second_block]
        first_geometry = logical_face_vertices(first_geometry_shape, interface.first_face)
        second_geometry = logical_face_vertices(
            second_geometry_shape, interface.second_face
        )
        if len(interface.permutation) != first_geometry.ndim:
            raise ValueError(
                "Interface permutation does not match the high-order trace dimension."
            )
        second_geometry = second_geometry.transpose(interface.permutation)
        for axis, flip in enumerate(interface.flips):
            if flip:
                second_geometry = np.flip(second_geometry, axis=axis)
        if first_geometry.shape != second_geometry.shape:
            raise ValueError(
                "High-order coordinate trace counts differ across an interface."
            )
        geometry_pairs = (
            first_geometry.reshape(-1) + geometry_offsets[interface.first_block],
            second_geometry.reshape(-1) + geometry_offsets[interface.second_block],
            interface.tolerance,
            interface.control_id,
        )
        geometry_pair_records.append(geometry_pairs)
    vertex_representatives, vertex_inverse, vertex_residual = _reconcile_pairs(
        points, tuple(vertex_pairs)
    )
    geometry_representatives, geometry_inverse, geometry_residual = _reconcile_pairs(
        geometry_points, tuple(geometry_pair_records)
    )
    cell_blocks: list[CellBlock] = []
    block_vertex_maps: list[tuple[str, np.ndarray]] = []
    cell_parents: list[tuple[str, np.ndarray]] = []
    output_elements: dict[str, CellGeometryElement] = {}
    output_routes: dict[str, np.ndarray] = {}
    cell_offset = 0
    for construction in ordered:
        source_block = construction.mesh.blocks[0]
        name = source_block.name
        vertex_start = vertex_offsets[name]
        vertex_end = vertex_start + construction.mesh.coordinates.shape[0]
        vertex_map = vertex_inverse[vertex_start:vertex_end].copy()
        block_vertex_maps.append((name, vertex_map))
        connectivity = vertex_map[np.asarray(source_block.vertices, dtype=np.int64)]
        cell_ids = np.arange(
            cell_offset, cell_offset + source_block.cell_count, dtype=np.int64
        )
        cell_blocks.append(
            CellBlock(name, source_block.cell_kind, connectivity, global_ids=cell_ids)
        )
        cell_parents.append(
            (name, np.asarray(source_block.global_ids, dtype=np.int64).copy())
        )
        cell_offset += source_block.cell_count
        element = geometry_elements[name]
        output_elements[name] = element
        output_routes[name] = (
            geometry_inverse[geometry_offsets[name] + local_geometry_routes[name]]
        ).astype(np.int32, copy=False)
    mesh = CellMesh(points[vertex_representatives], tuple(cell_blocks))
    geometry = CellGeometrySpec(
        output_elements,
        output_routes,
        geometry_points[geometry_representatives],
    )
    return MultiblockConstruction(
        mesh,
        geometry,
        tuple(block_vertex_maps),
        tuple(cell_parents),
        tuple(sorted(interface.control_id for interface in interfaces)),
        max(vertex_residual, geometry_residual),
    )


def glue_structured_blocks(
    blocks: tuple[StructuredConstruction, ...],
    interfaces: tuple[BlockInterfaceControl, ...],
    /,
) -> MultiblockConstruction:
    """Glue conforming structured interfaces using only declared face maps."""
    return construct_glued_structured_blocks(blocks, interfaces)


__all__ = [
    "MultiblockConstruction",
    "logical_face_vertices",
    "construct_glued_structured_blocks",
    "glue_structured_blocks",
]
