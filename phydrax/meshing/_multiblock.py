#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact declared logical-face gluing and independent-part coupling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry import (
    CellGeometryElement,
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
)
from ..geometry._mesh_certificates import MeshCertificateLimits
from ._assembly import MeshAssembly, MeshPart
from ._controls import BlockInterfaceControl
from ._coupling import MeshCoupling
from ._quad_generation import _family_host_array
from ._result import CellMeshingResult
from ._structured import certify_constructed_mesh, StructuredConstruction


@dataclass(frozen=True, slots=True)
class MultiblockConstruction:
    mesh: CellMesh
    geometry: CellGeometrySpec
    block_vertex_ids: tuple[tuple[str, np.ndarray], ...]
    interface_ids: tuple[str, ...]
    maximum_gluing_residual: float


def logical_face_vertices(shape: tuple[int, ...], face: int, /) -> np.ndarray:
    if len(shape) not in (2, 3) or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in shape
    ):
        raise ValueError(
            "Logical faces require two or three axes with at least two nodes each."
        )
    if not 0 <= face < 2 * len(shape):
        raise ValueError("A logical face must belong to the block dimension.")
    axis, side = divmod(face, 2)
    selector = tuple(
        (0 if side == 0 else -1) if i == axis else slice(None) for i in range(len(shape))
    )
    return np.arange(np.prod(shape), dtype=np.int64).reshape(shape)[selector]


def _face_correspondence(
    first: StructuredConstruction,
    second: StructuredConstruction,
    interface: BlockInterfaceControl,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if len(first.logical_shape) != len(second.logical_shape):
        raise ValueError("Glued blocks must have the same topological dimension.")
    first_ids = logical_face_vertices(first.logical_shape, interface.first_face)
    second_ids = logical_face_vertices(second.logical_shape, interface.second_face)
    if len(interface.permutation) != first_ids.ndim:
        raise ValueError("Interface permutation has the wrong face dimension.")
    second_ids = second_ids.transpose(interface.permutation)
    for axis, flip in enumerate(interface.flips):
        if flip:
            second_ids = np.flip(second_ids, axis=axis)
    if first_ids.shape != second_ids.shape:
        raise ValueError("Conforming interfaces require exact common edge/face counts.")
    return first_ids.reshape(-1), second_ids.reshape(-1)


def _glued_geometry(
    blocks: tuple[StructuredConstruction, ...],
    vertex_routes: tuple[tuple[str, np.ndarray], ...],
    points: np.ndarray,
    /,
) -> CellGeometrySpec:
    """Retain full source banks; corner-defined maps follow declared node gluing."""
    elements: dict[str, CellGeometryElement] = {}
    routes: dict[str, np.ndarray] = {}
    coordinates = []
    offset = 0
    records = tuple(block.geometry.restriction_source for block in blocks)
    authored = tuple(record for record in records if record is not None)
    if authored and (
        len(authored) != len(records)
        or len(
            {
                (record.source_geometry_id, record.source_topology_id)
                for record in authored
            }
        )
        != 1
    ):
        raise ValueError(
            "Exact mapped block gluing requires one authoritative source geometry and root topology."
        )
    parent_cells = {}
    parent_vertices = {}
    source_blocks: dict[str, str] | None = {}
    for record in authored:
        parent_cells.update(record.block_parent_cell_ids)
        parent_vertices.update(record.block_parent_vertex_ids)
        owners = record.block_source_blocks
        source_blocks = (
            None
            if owners is None or source_blocks is None
            else {**source_blocks, **owners}
        )
    for construction, (_, vertex_route) in zip(blocks, vertex_routes, strict=True):
        source_elements, source_routes, source_points = construction.geometry.resolve(
            construction.mesh
        )
        corner_defined = (
            construction.geometry.restriction_source is None
            and np.array_equal(source_points, construction.mesh.coordinates)
            and all(
                element.element_id
                == coordinate_lagrange_element(block.cell_kind, 1).element_id
                and np.array_equal(route, block.vertices)
                for block, element, route in zip(
                    construction.mesh.blocks, source_elements, source_routes, strict=True
                )
            )
        )
        # The interface tolerance explicitly permits identifying nearby corner
        # nodes. Only a map defined by those very corners follows that snapping;
        # authored higher-order/restricted maps retain every source coefficient.
        coordinates.append(
            points[vertex_route]
            if corner_defined
            else np.asarray(source_points, dtype=np.float64)
        )
        for block, element, route in zip(
            construction.mesh.blocks, source_elements, source_routes, strict=True
        ):
            if block.name in elements:
                raise ValueError("Every constituent cell block requires a unique name.")
            elements[block.name] = element
            routes[block.name] = np.asarray(route, dtype=np.int32) + offset
        offset += source_points.shape[0]
    ancestry = (
        None
        if not authored
        else CellGeometryRestrictionSource(
            authored[0].source_geometry_id,
            authored[0].source_topology_id,
            parent_cells,
            parent_vertices,
            block_source_blocks=source_blocks,
        )
    )
    bank = _family_host_array((offset, coordinates[0].shape[1]), np.float64)
    start = 0
    for values in coordinates:
        bank[start : start + values.shape[0]] = values
        start += values.shape[0]
    return CellGeometrySpec(elements, routes, bank, restriction_source=ancestry)


def glue_structured_blocks(
    blocks: tuple[StructuredConstruction, ...],
    interfaces: tuple[BlockInterfaceControl, ...],
    /,
    *,
    certificate_limits: MeshCertificateLimits | None = None,
) -> MultiblockConstruction:
    """Publish one topology: only declared node pairs receive the same global ID."""
    if not blocks:
        raise ValueError("A multiblock mesh requires blocks.")
    ordered = tuple(sorted(blocks, key=lambda b: b.mesh.blocks[0].name))
    names = tuple(b.mesh.blocks[0].name for b in ordered)
    if len(set(names)) != len(names):
        raise ValueError("Multiblock names must be unique.")
    if any(
        not isinstance(i, BlockInterfaceControl) or not i.conforming for i in interfaces
    ):
        raise ValueError(
            "Nonconforming interfaces require assemble_independent_blocks and explicit couplings."
        )
    if (
        len({(b.mesh.topological_dimension, b.mesh.ambient_dimension) for b in ordered})
        != 1
    ):
        raise ValueError("All glued blocks must use one topology and ambient dimension.")
    by_name = dict(zip(names, ordered, strict=True))
    offsets: dict[str, int] = {}
    total = 0
    for name, block in zip(names, ordered, strict=True):
        offsets[name] = total
        total += block.mesh.coordinates.shape[0]
    points = _family_host_array((total, ordered[0].mesh.ambient_dimension), np.float64)
    for name, block in zip(names, ordered, strict=True):
        start = offsets[name]
        points[start : start + block.mesh.coordinates.shape[0]] = np.asarray(
            block.mesh.coordinates, dtype=np.float64
        )
    parent = _family_host_array((total,), np.int64)
    parent[:] = np.arange(total, dtype=np.int64)

    def root(index: int, /) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    occupied: set[tuple[str, int]] = set()
    residual = 0.0
    declared_pairs = []
    for interface in sorted(interfaces, key=lambda i: i.control_id):
        if interface.first_block not in by_name or interface.second_block not in by_name:
            raise ValueError("Interface references an unknown block.")
        endpoints = (
            (interface.first_block, interface.first_face),
            (interface.second_block, interface.second_face),
        )
        if endpoints[0] == endpoints[1] or any(
            endpoint in occupied for endpoint in endpoints
        ):
            raise ValueError("Each logical face may participate in exactly one gluing.")
        occupied.update(endpoints)
        left, right = _face_correspondence(
            by_name[interface.first_block], by_name[interface.second_block], interface
        )
        left = left + offsets[interface.first_block]
        right = right + offsets[interface.second_block]
        error = float(
            np.max(np.linalg.norm(points[left] - points[right], axis=-1), initial=0.0)
        )
        if error > interface.tolerance:
            raise ValueError(
                f"Declared gluing {interface.control_id} has incompatible coordinates: {error}."
            )
        residual = max(residual, error)
        declared_pairs.append((interface, left, right))
        for a, b in zip(left.tolist(), right.tolist(), strict=True):
            ra, rb = root(a), root(b)
            parent[max(ra, rb)] = min(ra, rb)
    roots = _family_host_array((total,), np.int64)
    for index in range(total):
        roots[index] = root(index)
    # A corner shared by several faces can acquire a representative through a
    # different interface. Pairwise agreement alone does not bound that final
    # displacement: tolerances may not accumulate around a block cycle.
    for interface, left, right in declared_pairs:
        displacement = max(
            float(
                np.max(
                    np.linalg.norm(points[left] - points[roots[left]], axis=-1),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.linalg.norm(points[right] - points[roots[right]], axis=-1),
                    initial=0.0,
                )
            ),
        )
        if displacement > interface.tolerance:
            raise ValueError(
                f"Declared gluing {interface.control_id} exceeds its final node displacement tolerance: {displacement}."
            )
        residual = max(residual, displacement)
    representatives, inverse = np.unique(roots, return_inverse=True)
    cells = []
    cell_offset = 0
    routes = []
    for name, block in zip(names, ordered, strict=True):
        offset = offsets[name]
        route = inverse[offset : offset + block.mesh.coordinates.shape[0]]
        routes.append((name, route.astype(np.int64)))
        for cell_block in block.mesh.blocks:
            rows = route[np.asarray(cell_block.vertices, dtype=np.int64)]
            cells.append(
                CellBlock(
                    cell_block.name,
                    cell_block.cell_kind,
                    rows,
                    global_ids=np.arange(
                        cell_offset, cell_offset + cell_block.cell_count, dtype=np.int64
                    ),
                )
            )
            cell_offset += cell_block.cell_count
    mesh = CellMesh(
        points[representatives],
        tuple(cells),
        vertex_global_ids=np.arange(representatives.size, dtype=np.int64),
    )
    geometry = _glued_geometry(ordered, tuple(routes), np.asarray(mesh.coordinates))
    certify_constructed_mesh(
        mesh, geometry=geometry, certificate_limits=certificate_limits
    )
    return MultiblockConstruction(
        mesh,
        geometry,
        tuple(routes),
        tuple(sorted(i.control_id for i in interfaces)),
        residual,
    )


def assemble_independent_blocks(
    parts: tuple[MeshPart, ...],
    interfaces: tuple[BlockInterfaceControl, ...],
    couplings: Mapping[str, MeshCoupling],
    /,
    *,
    logical_shapes: Mapping[str, tuple[int, ...]],
) -> MeshAssembly:
    """Retain separately certified parts with revision-bound explicit coupling.

    Couplings are keyed by interface control identity. No interpolation or mortar
    map is fabricated from nearby coordinates; the existing coupling owner is
    responsible for the declared transfer and conservation contract.
    """
    names = {part.name for part in parts}
    if len(names) != len(parts):
        raise ValueError("Independent block names must be unique.")
    by_name = {part.name: part for part in parts}
    if set(logical_shapes) != names:
        raise ValueError(
            "Independent parts require exact logical shapes for every block."
        )
    if not interfaces or any(interface.conforming for interface in interfaces):
        raise ValueError("Independent parts require explicitly nonconforming interfaces.")
    required = {interface.control_id for interface in interfaces}
    if set(couplings) != required:
        raise ValueError(
            "Every nonconforming interface requires exactly one explicit coupling."
        )
    occupied: set[tuple[str, int]] = set()
    for interface in interfaces:
        if interface.first_block not in names or interface.second_block not in names:
            raise ValueError("A nonconforming interface names an unknown part.")
        endpoints = (
            (interface.first_block, interface.first_face),
            (interface.second_block, interface.second_face),
        )
        if endpoints[0] == endpoints[1] or any(
            endpoint in occupied for endpoint in endpoints
        ):
            raise ValueError("Each logical face may participate in exactly one coupling.")
        occupied.update(endpoints)
        coupling = couplings[interface.control_id]
        if {coupling.source_scope.source_id, coupling.target_scope.source_id} != {
            interface.first_block,
            interface.second_block,
        }:
            raise ValueError("A coupling must bind the exact declared interface parts.")
        for scope in (coupling.source_scope, coupling.target_scope):
            part = by_name[scope.source_id]
            shape = logical_shapes[part.name]
            if not isinstance(part.carrier, CellMeshingResult):
                raise TypeError(
                    "Independent structured parts require certified cell carriers."
                )
            mesh = part.carrier.mesh
            if np.prod(shape) != mesh.coordinates.shape[0]:
                raise ValueError("The declared logical shape does not bind the part.")
            face = (
                interface.first_face
                if part.name == interface.first_block
                else interface.second_face
            )
            rows = logical_face_vertices(shape, face).reshape(-1)
            if scope.entity_dimension != 0:
                raise ValueError(
                    "Independent block nodal coupling requires zero-dimensional face scopes."
                )
            expected = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[rows]
            if not np.array_equal(np.sort(expected), np.asarray(scope.entity_ids)):
                raise ValueError(
                    "Coupling scopes must cover exactly the declared logical faces."
                )
    return MeshAssembly(
        parts, couplings=tuple(couplings[key] for key in sorted(couplings))
    )


__all__ = [
    "MultiblockConstruction",
    "logical_face_vertices",
    "glue_structured_blocks",
    "assemble_independent_blocks",
]
