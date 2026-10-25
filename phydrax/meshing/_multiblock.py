#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Glue declared logical faces and couple independent parts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

import numpy as np

import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.meshing._assembly as _dependency_phydrax_meshing__assembly
import phydrax.meshing._controls as _dependency_phydrax_meshing__controls
import phydrax.meshing._coupling as _dependency_phydrax_meshing__coupling
import phydrax.meshing._family_execution as family_execution_owner
import phydrax.meshing._host_banks as _dependency_phydrax_meshing__host_banks
import phydrax.meshing._mesh_part as _dependency_phydrax_meshing__mesh_part
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._structured as _dependency_phydrax_meshing__structured
import phydrax.meshing._volume_generation as _dependency_phydrax_meshing__volume_generation
from phydrax.discretization._reference_cell_types import CellGeometryElement

from .._meshcore import NativeExecutionScope
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_mesh import (
    cell_block_construction_bounds,
    cell_block_construction_work,
    cell_mesh_construction_bounds,
    cell_mesh_construction_work,
)
from ..typing import checked
from ._contracts import MeshingLimits


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
            f"multiblock.logical_shape must have two or three axes with at least two nodes each; got {shape}"
        )
    if not 0 <= face < 2 * len(shape):
        raise ValueError(
            f"multiblock.logical_face must belong to the block dimension; got face {face} for {len(shape)} axes"
        )
    axis, side = divmod(face, 2)
    # C-order node IDs of the fixed face, without a whole-block identity range.
    strides = tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
    free = tuple(i for i in range(len(shape)) if i != axis)
    ids = np.full(
        tuple(shape[i] for i in free),
        (0 if side == 0 else shape[axis] - 1) * strides[axis],
        dtype=np.int64,
    )
    for position, i in enumerate(free):
        offsets = [1] * len(free)
        offsets[position] = shape[i]
        ids += (np.arange(shape[i], dtype=np.int64) * strides[i]).reshape(offsets)
    return ids


def _face_correspondence(
    first: _dependency_phydrax_meshing__structured.StructuredConstruction,
    second: _dependency_phydrax_meshing__structured.StructuredConstruction,
    interface: _dependency_phydrax_meshing__controls.BlockInterfaceControl,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if len(first.logical_shape) != len(second.logical_shape):
        raise ValueError(
            f"multiblock.interface.blocks must have the same topological dimension; got {len(first.logical_shape)} and {len(second.logical_shape)}"
        )
    first_ids = logical_face_vertices(first.logical_shape, interface.first_face)
    second_ids = logical_face_vertices(second.logical_shape, interface.second_face)
    if len(interface.permutation) != first_ids.ndim:
        raise ValueError(
            f"multiblock.interface.permutation must match the face dimension; got {interface.permutation} for rank {first_ids.ndim}"
        )
    second_ids = second_ids.transpose(interface.permutation)
    for axis, flip in enumerate(interface.flips):
        if flip:
            second_ids = np.flip(second_ids, axis=axis)
    if first_ids.shape != second_ids.shape:
        raise ValueError(
            f"multiblock.interface.face_counts must match on both sides; got {first_ids.shape} and {second_ids.shape}"
        )
    return first_ids.reshape(-1), second_ids.reshape(-1)


def _glued_geometry(
    blocks: tuple[_dependency_phydrax_meshing__structured.StructuredConstruction, ...],
    vertex_routes: tuple[tuple[str, np.ndarray], ...],
    points: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CellGeometrySpec:
    """Retain full source banks; corner-defined maps follow declared node gluing."""
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        scope.host_workspace() as workspace,
    ):
        elements: dict[str, CellGeometryElement] = {}
        routes: dict[str, np.ndarray] = {}
        coordinates = []
        offset = 0
        retained = 0
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
                "multiblock.geometry.restriction_source must share one source geometry and root topology; got different source bindings"
            )
        # Each record's ancestry maps are merged once; a block-owner merge
        # copies at most every owner entry per record.
        family_execution_owner.count_geometry_queries(
            0,
            work_units=len(records)
            + sum(
                len(record.block_parent_cell_ids)
                + len(record.block_parent_vertex_ids)
                + len(authored) * len(record.block_source_blocks or ())
                for record in authored
            ),
            budget=budget,
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
                construction.mesh, budget=budget
            )
            entries = sum(block.vertices.size for block in construction.mesh.blocks)
            route_entries = sum(np.size(route) for route in source_routes)
            # Coordinate and route transfers with their comparisons, the corner
            # gather or source copy, and the shifted route conversion.
            family_execution_owner.count_geometry_queries(
                0,
                work_units=3 * np.size(source_points)
                + 2 * entries
                + 4 * route_entries
                + max(vertex_route.size * (points.shape[1] + 1), np.size(source_points)),
                budget=budget,
            )
            # Coordinate and route comparisons own two host transfers and one
            # boolean temporary each.
            workspace.set_bound(
                retained + 17 * max(np.size(source_points), entries, route_entries) + 4096
            )
            corner_defined = (
                construction.geometry.restriction_source is None
                and np.array_equal(source_points, construction.mesh.coordinates)
                and all(
                    element.element_id
                    == _dependency_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
                        block.cell_kind, 1
                    ).element_id
                    and np.array_equal(route, block.vertices)
                    for block, element, route in zip(
                        construction.mesh.blocks,
                        source_elements,
                        source_routes,
                        strict=True,
                    )
                )
            )
            # Gathered corner coordinates and shifted routes stay live until the
            # bank is assembled; a route conversion may own one more copy.
            gathered = (
                vertex_route.size * points.shape[1]
                if corner_defined
                else np.size(source_points)
            )
            retained += 8 * gathered + 8 * route_entries
            workspace.set_bound(retained + 4096)
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
                    raise ValueError(
                        f"multiblock.geometry.block_names must be unique; got duplicate name {block.name!r}"
                    )
                elements[block.name] = element
                routes[block.name] = np.asarray(route, dtype=np.int32) + offset
            offset += source_points.shape[0]
        ancestry = (
            None
            if not authored
            else _dependency_phydrax_discretization__cell_geometry.CellGeometryRestrictionSource(
                authored[0].source_geometry_id,
                authored[0].source_topology_id,
                parent_cells,
                parent_vertices,
                block_source_blocks=source_blocks,
            )
        )
        width = coordinates[0].shape[1]
        # One copy per retained coordinate into the pooled bank.
        family_execution_owner.count_geometry_queries(
            0, work_units=offset * width, budget=budget
        )
        bank = _dependency_phydrax_meshing__host_banks.family_host_array(
            (offset, width), np.float64, scope=scope
        )
        start = 0
        for values in coordinates:
            bank[start : start + values.shape[0]] = values
            start += values.shape[0]
        return CellGeometrySpec(
            elements, routes, bank, restriction_source=ancestry, budget=budget
        )


@checked
def construct_glued_structured_blocks(
    blocks: tuple[_dependency_phydrax_meshing__structured.StructuredConstruction, ...],
    interfaces: tuple[_dependency_phydrax_meshing__controls.BlockInterfaceControl, ...],
    /,
    *,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MultiblockConstruction:
    """Publish one topology: only declared node pairs receive the same global ID."""
    if not blocks:
        raise ValueError("multiblock.blocks must be nonempty; got zero blocks")
    ordered = tuple(sorted(blocks, key=lambda b: b.mesh.blocks[0].name))
    names = tuple(b.mesh.blocks[0].name for b in ordered)
    if len(set(names)) != len(names):
        raise ValueError(f"multiblock.block_names must be unique; got {names}")
    for interface in interfaces:
        match interface:
            case _dependency_phydrax_meshing__controls.BlockInterfaceControl(
                conforming=True
            ):
                pass
            case _:
                raise ValueError(
                    "multiblock.interfaces must be conforming controls for shared topology; got a nonconforming or wrong-kind interface"
                )
    if (
        len({(b.mesh.topological_dimension, b.mesh.ambient_dimension) for b in ordered})
        != 1
    ):
        raise ValueError(
            "multiblock.blocks must share topology and ambient dimensions; got mixed block dimensions"
        )
    by_name = dict(zip(names, ordered, strict=True))
    offsets: dict[str, int] = {}
    total = 0
    for name, block in zip(names, ordered, strict=True):
        offsets[name] = total
        total += block.mesh.coordinates.shape[0]
    ambient = ordered[0].mesh.ambient_dimension
    ordered_interfaces = tuple(sorted(interfaces, key=lambda i: i.control_id))
    # Face node counts follow from the logical shapes. An undeclared block or
    # face contributes nothing: its interface refuses before work.
    extents = []
    for interface in ordered_interfaces:
        sides = []
        for name, face in (
            (interface.first_block, interface.first_face),
            (interface.second_block, interface.second_face),
        ):
            shape = by_name[name].logical_shape if name in by_name else ()
            sides.append(
                prod(shape) // shape[face // 2] if 0 <= face < 2 * len(shape) else 0
            )
        extents.append((sides[0], sides[1]))
    pair_count = sum(first for first, _ in extents)
    largest = max((first for first, _ in extents), default=0)
    cell_blocks = tuple(
        cell_block for block in ordered for cell_block in block.mesh.blocks
    )
    mesh_shapes = tuple(
        (cell_block.cell_kind, cell_block.cell_count, cell_block.arity)
        for cell_block in cell_blocks
    )
    entries = sum(cell_block.vertices.size for cell_block in cell_blocks)
    cell_count = sum(cell_block.cell_count for cell_block in cell_blocks)
    largest_block = max(block.mesh.coordinates.shape[0] for block in ordered)
    # Union by size keeps every root path within the bit length of the largest
    # possible glued class, which has at most one node per pair beyond its
    # first. Only nodes named by a pair leave a singleton class.
    height = (pair_count + 1).bit_length()
    # Coordinate transfers and copies; identity, least-member and size banks;
    # per side the face identities, their flattened copy and offset rows; the
    # pair residual gathers and reductions; two bounded root walks and a union
    # per pair; one representative per node; and both final displacement
    # sides with their root gathers.
    scope.charge(
        work=2 * total * ambient
        + 6 * total
        + sum(5 * (first + second) for first, second in extents)
        + pair_count * (15 * ambient + 14 + 8 * height)
    )
    with scope.host_workspace() as workspace:
        # The parent identity range, or one block's coordinate transfer, is the
        # assembly's one unpooled temporary.
        workspace.set_bound(8 * total + 8 * largest_block * ambient + 4096)
        points = _dependency_phydrax_meshing__host_banks.family_host_array(
            (total, ambient), np.float64, scope=scope
        )
        for name, block in zip(names, ordered, strict=True):
            start = offsets[name]
            points[start : start + block.mesh.coordinates.shape[0]] = np.asarray(
                block.mesh.coordinates, dtype=np.float64
            )
        parent = _dependency_phydrax_meshing__host_banks.family_host_array(
            (total,), np.int64, scope=scope
        )
        least = _dependency_phydrax_meshing__host_banks.family_host_array(
            (total,), np.int64, scope=scope
        )
        size = _dependency_phydrax_meshing__host_banks.family_host_array(
            (total,), np.int64, scope=scope
        )
        parent[:] = np.arange(total, dtype=np.int64)
        np.copyto(least, parent)
        size.fill(1)

        def root(index: int, /) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        left_rows = _dependency_phydrax_meshing__host_banks.family_host_array(
            (pair_count,), np.int64, scope=scope
        )
        right_rows = _dependency_phydrax_meshing__host_banks.family_host_array(
            (pair_count,), np.int64, scope=scope
        )
        differences = _dependency_phydrax_meshing__host_banks.family_host_array(
            (largest, ambient), np.float64, scope=scope
        )
        distances = _dependency_phydrax_meshing__host_banks.family_host_array(
            (largest,), np.float64, scope=scope
        )
        occupied: set[tuple[str, int]] = set()
        residual = 0.0
        declared_pairs = []
        start = 0
        for interface, (first_face, second_face) in zip(
            ordered_interfaces, extents, strict=True
        ):
            if (
                interface.first_block not in by_name
                or interface.second_block not in by_name
            ):
                raise ValueError(
                    f"multiblock.interface.blocks must name declared blocks; got {interface.first_block!r} and {interface.second_block!r}"
                )
            endpoints = (
                (interface.first_block, interface.first_face),
                (interface.second_block, interface.second_face),
            )
            if endpoints[0] == endpoints[1] or any(
                endpoint in occupied for endpoint in endpoints
            ):
                raise ValueError(
                    f"multiblock.interface.faces must participate in one gluing each; got reused face endpoints {endpoints}"
                )
            occupied.update(endpoints)
            # Both face identity banks, their flattened copies and the two
            # coordinate gathers of the residual may coexist.
            workspace.set_bound(
                16 * (first_face + second_face) + 16 * first_face * ambient + 4096
            )
            first_ids, second_ids = _face_correspondence(
                by_name[interface.first_block], by_name[interface.second_block], interface
            )
            count = first_ids.size
            left = left_rows[start : start + count]
            right = right_rows[start : start + count]
            start += count
            np.add(first_ids, offsets[interface.first_block], out=left)
            np.add(second_ids, offsets[interface.second_block], out=right)
            difference = differences[:count]
            distance = distances[:count]
            np.subtract(points[left], points[right], out=difference)
            np.square(difference, out=difference)
            np.sum(difference, axis=-1, out=distance)
            np.sqrt(distance, out=distance)
            error = float(np.max(distance, initial=0.0))
            if error > interface.tolerance:
                raise ValueError(
                    f"multiblock.interface.coordinates must agree within tolerance {interface.tolerance}; got incompatible coordinates for {interface.control_id}, residual {error}"
                )
            residual = max(residual, error)
            declared_pairs.append((interface, left, right))
            for first_row, second_row in zip(left, right, strict=True):
                first_root = root(int(first_row))
                second_root = root(int(second_row))
                if first_root != second_root:
                    if size[first_root] < size[second_root]:
                        first_root, second_root = second_root, first_root
                    parent[second_root] = first_root
                    size[first_root] += size[second_root]
                    least[first_root] = min(least[first_root], least[second_root])
        roots = _dependency_phydrax_meshing__host_banks.family_host_array(
            (total,), np.int64, scope=scope
        )
        for index in range(total):
            roots[index] = least[root(index)]
        # A corner shared by several faces can acquire a representative through a
        # different interface. Pairwise agreement alone does not bound that final
        # displacement: tolerances may not accumulate around a block cycle.
        for interface, left, right in declared_pairs:
            count = left.size
            # One root gather and two coordinate gathers coexist per side.
            workspace.set_bound(8 * count + 16 * count * ambient + 4096)
            difference = differences[:count]
            distance = distances[:count]
            extremes = []
            for rows in (left, right):
                np.subtract(points[rows], points[roots[rows]], out=difference)
                np.square(difference, out=difference)
                np.sum(difference, axis=-1, out=distance)
                np.sqrt(distance, out=distance)
                extremes.append(float(np.max(distance, initial=0.0)))
            displacement = max(extremes)
            if displacement > interface.tolerance:
                raise ValueError(
                    f"multiblock.interface.displacement must fit tolerance {interface.tolerance}; got final node displacement {displacement} for {interface.control_id}"
                )
            residual = max(residual, displacement)
        # The representative sort compares at most the bit length of the node
        # count per node, with six further unique/inverse passes.
        scope.charge(work=total * (total.bit_length() + 6))
        # Flattened input, sort permutation, sorted values, boundary masks,
        # unique values, running counts and inverse rows.
        workspace.set_bound(64 * total + 4096)
        representatives, inverse = np.unique(roots, return_inverse=True)
        # Assembly copies routes, converts and gathers every corner, ranges
        # cell IDs and gathers representative coordinates and identities; then
        # the block and mesh constructors' priced passes.
        scope.charge(
            work=total
            + 2 * entries
            + cell_count
            + representatives.size * (ambient + 1)
            + sum(
                cell_block_construction_work(cell_block.cell_count, cell_block.arity)
                for cell_block in cell_blocks
            )
            + cell_mesh_construction_work(representatives.size, ambient, mesh_shapes)
        )
        # The representatives and inverse stay live; each cell block owns a
        # converted row bank, a gathered row bank and an identity range, as do
        # the representative coordinates and vertex identities.
        assembly_bytes = (
            16 * total
            + sum(16 * block.vertices.size + 8 * block.cell_count for block in cell_blocks)
            + 8 * representatives.size * (ambient + 1)
            + 4096
        )
        workspace.set_bound(assembly_bytes)
        cells = []
        cell_offset = 0
        routes = []
        for name, block in zip(names, ordered, strict=True):
            offset = offsets[name]
            count = block.mesh.coordinates.shape[0]
            route = _dependency_phydrax_meshing__host_banks.family_host_array(
                (count,), np.int64, scope=scope
            )
            np.copyto(route, inverse[offset : offset + count])
            routes.append((name, route))
            for cell_block in block.mesh.blocks:
                rows = route[np.asarray(cell_block.vertices, dtype=np.int64)]
                workspace.set_bound(
                    assembly_bytes
                    + cell_block_construction_bounds(
                        cell_block.cell_count, cell_block.arity
                    )
                )
                cells.append(
                    CellBlock(
                        cell_block.name,
                        cell_block.cell_kind,
                        rows,
                        global_ids=np.arange(
                            cell_offset,
                            cell_offset + cell_block.cell_count,
                            dtype=np.int64,
                        ),
                    )
                )
                cell_offset += cell_block.cell_count
        workspace.set_bound(
            assembly_bytes
            + cell_mesh_construction_bounds(
                representatives.size, ambient, mesh_shapes
            )
        )
        mesh = CellMesh(
            points[representatives],
            tuple(cells),
            vertex_global_ids=np.arange(representatives.size, dtype=np.int64),
        )
    glued = mesh.coordinates
    # One host transfer per glued coordinate, live through the geometry phase.
    scope.charge(work=glued.size)
    with scope.host_workspace() as workspace:
        workspace.set_bound(8 * glued.size + 4096)
        geometry = _glued_geometry(
            ordered, tuple(routes), np.asarray(glued), scope=scope, budget=budget
        )
    _dependency_phydrax_meshing__structured.certify_constructed_mesh(
        mesh,
        geometry=geometry,
        certificate_limits=certificate_limits,
        scope=scope,
        budget=budget,
    )
    return MultiblockConstruction(
        mesh,
        geometry,
        tuple(routes),
        tuple(sorted(i.control_id for i in interfaces)),
        residual,
    )


@checked
def glue_structured_blocks(
    blocks: tuple[_dependency_phydrax_meshing__structured.StructuredConstruction, ...],
    interfaces: tuple[_dependency_phydrax_meshing__controls.BlockInterfaceControl, ...],
    /,
    *,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    limits: MeshingLimits = MeshingLimits(),
) -> MultiblockConstruction:
    """Glue and certify declared interfaces within the meshing limits."""
    with _dependency_phydrax_meshing__volume_generation.native_volume_execution_budget(
        limits
    ) as scope:
        ticket = scope.batch(0)
        ticket.settle()
        budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with budget.retained_storage(lambda n: scope.batch(0, bytes_upper=n)):
            return construct_glued_structured_blocks(
                blocks,
                interfaces,
                certificate_limits=certificate_limits,
                scope=scope,
                budget=budget,
            )


def assemble_independent_blocks(
    parts: tuple[_dependency_phydrax_meshing__mesh_part.MeshPart, ...],
    interfaces: tuple[_dependency_phydrax_meshing__controls.BlockInterfaceControl, ...],
    couplings: Mapping[str, _dependency_phydrax_meshing__coupling.MeshCoupling],
    /,
    *,
    logical_shapes: Mapping[str, tuple[int, ...]],
) -> _dependency_phydrax_meshing__assembly.MeshAssembly:
    """Retain separately certified parts with revision-bound explicit coupling.

    Couplings are keyed by interface control identity. No interpolation or mortar
    map is fabricated from nearby coordinates; the existing coupling owner is
    responsible for the declared transfer and conservation contract.
    """
    names = {part.name for part in parts}
    if len(names) != len(parts):
        raise ValueError(
            f"multiblock.parts.names must be unique; got {tuple(part.name for part in parts)}"
        )
    by_name = {part.name: part for part in parts}
    if set(logical_shapes) != names:
        raise ValueError(
            f"multiblock.logical_shapes must cover every part; got {tuple(sorted(logical_shapes))} for {tuple(sorted(names))}"
        )
    if not interfaces or any(interface.conforming for interface in interfaces):
        raise ValueError(
            "multiblock.interfaces must be nonempty and nonconforming; got empty or conforming interfaces"
        )
    required = {interface.control_id for interface in interfaces}
    if set(couplings) != required:
        raise ValueError(
            "multiblock.couplings must cover each declared interface once; got different interface and coupling IDs"
        )
    occupied: set[tuple[str, int]] = set()
    for interface in interfaces:
        if interface.first_block not in names or interface.second_block not in names:
            raise ValueError(
                f"multiblock.interface.parts must name declared parts; got {interface.first_block!r} and {interface.second_block!r}"
            )
        endpoints = (
            (interface.first_block, interface.first_face),
            (interface.second_block, interface.second_face),
        )
        if endpoints[0] == endpoints[1] or any(
            endpoint in occupied for endpoint in endpoints
        ):
            raise ValueError(
                f"multiblock.interface.faces must participate in one coupling each; got reused face endpoints {endpoints}"
            )
        occupied.update(endpoints)
        coupling = couplings[interface.control_id]
        if {coupling.source_scope.source_id, coupling.target_scope.source_id} != {
            interface.first_block,
            interface.second_block,
        }:
            raise ValueError(
                "multiblock.coupling.parts must bind the declared interface parts; got different coupling source IDs"
            )
        for scope in (coupling.source_scope, coupling.target_scope):
            part = by_name[scope.source_id]
            shape = logical_shapes[part.name]
            match part.carrier:
                case _dependency_phydrax_meshing__result.CellMeshingResult() as carrier:
                    mesh = carrier.mesh
                case _:
                    raise TypeError(
                        f"multiblock.part.carrier must be a certified cell carrier; got {type(part.carrier).__name__}"
                    )
            if np.prod(shape) != mesh.coordinates.shape[0]:
                raise ValueError(
                    f"multiblock.part.logical_shape must bind its coordinate count; got {shape} for {mesh.coordinates.shape[0]} coordinates"
                )
            face = (
                interface.first_face
                if part.name == interface.first_block
                else interface.second_face
            )
            rows = logical_face_vertices(shape, face).reshape(-1)
            if scope.entity_dimension != 0:
                raise ValueError(
                    f"multiblock.coupling.scope must have entity dimension zero; got {scope.entity_dimension}"
                )
            expected = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[rows]
            if not np.array_equal(np.sort(expected), np.asarray(scope.entity_ids)):
                raise ValueError(
                    "multiblock.coupling.scope must cover the declared logical face; got different entity IDs"
                )
    return _dependency_phydrax_meshing__assembly.MeshAssembly(
        parts, couplings=tuple(couplings[key] for key in sorted(couplings))
    )


__all__ = [
    "MultiblockConstruction",
    "logical_face_vertices",
    "glue_structured_blocks",
    "assemble_independent_blocks",
]
