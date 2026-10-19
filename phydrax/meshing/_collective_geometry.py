#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from .._strict import StrictModule
from ..discretization import _cell_mesh as cell_mesh_owner
from ..discretization._adaptive_simplex import AdaptiveSimplexState
from ..discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometrySpec,
    coordinate_lagrange_element,
    require_p1_cardinal_source,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    coordinate_corner_images,
    coordinate_source_signature,
    prepared_coordinate_source_bank,
)
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..discretization._periodic_topology import PeriodicMeshTopology
from ..discretization._reference_cell_types import CellGeometryElement
from ..typing import checked
from . import (
    _bisection as bisection_owner,
    _initial_certification as initial_certification_owner,
    _result as result_owner,
)
from ._bisection_kernel_host import uniform_count, uniform_retain_source


@checked
def periodic_source_closure_cells(
    source: cell_mesh_owner.CellMesh,
    cell_ids: ArrayLike,
    /,
    *,
    cell_capacity: int,
    vertex_capacity: int,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Include source orbit members within the declared closure capacities."""
    periodic = source.periodic_topology
    requested = np.unique(np.asarray(cell_ids, dtype=np.int64))
    if periodic is None:
        return requested

    ids = np.concatenate([np.asarray(block.global_ids) for block in source.blocks])
    corners = np.concatenate(
        [
            np.asarray(source.vertex_global_ids)[np.asarray(block.vertices)]
            for block in source.blocks
        ]
    )
    rows = {int(identifier): row for row, identifier in enumerate(ids)}
    if any(int(identifier) not in rows for identifier in requested):
        raise ValueError(
            "periodic_source_closure_cells.cell_ids must identify source cells; got an absent cell"
        )
    selected = np.zeros(ids.size, dtype=np.bool_)
    selected[[rows[int(identifier)] for identifier in requested]] = True
    orbits = np.asarray(periodic.orbits(source.topological_dimension)[0])
    vertices = np.asarray(source.vertex_global_ids)
    representatives = vertices[np.asarray(periodic.vertex_representatives)]
    representative_by_id = dict(
        zip(vertices.tolist(), representatives.tolist(), strict=True)
    )
    while True:
        budget.reserve(corners.size + ids.size, 0)
        selected |= np.isin(orbits, orbits[selected])
        retained = np.unique(corners[selected])
        required = np.unique(
            [representative_by_id[int(identifier)] for identifier in retained]
        )
        missing = required[~np.isin(required, retained)]
        if missing.size:
            for identifier in missing:
                incident = np.flatnonzero(np.any(corners == identifier, axis=1))
                if not incident.size:
                    raise ValueError(
                        "periodic_source_closure_cells.representative must have an incident source cell; got an isolated representative"
                    )
                selected[incident[np.argmin(ids[incident])]] = True
        actual_vertices = np.unique(corners[selected]).size
        if (
            np.count_nonzero(selected) > cell_capacity
            or actual_vertices > vertex_capacity
        ):
            raise ValueError(
                "periodic_source_closure_cells.support must fit closure capacity; got excess source support"
            )
        if not missing.size:
            return np.sort(ids[selected])


@checked
def project_source_periodic_topology(
    source: cell_mesh_owner.CellMesh,
    lifted: cell_mesh_owner.CellMesh,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PeriodicMeshTopology | None:
    """Restrict the source periodic quotient to a lifted closure.

    Missing source representatives or orbit members fail without renumbering
    the quotient or changing authored winding.
    """

    periodic = source.periodic_topology
    if periodic is None:
        return None
    source_ids = np.asarray(source.vertex_global_ids)
    local_ids = np.asarray(lifted.vertex_global_ids)
    source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}
    local_rows = {int(identifier): row for row, identifier in enumerate(local_ids)}
    if any(int(identifier) not in source_rows for identifier in local_ids):
        raise ValueError(
            "project_source_periodic_topology.lifted must use source vertices; got an unauthored vertex"
        )
    selected = np.asarray(
        [source_rows[int(identifier)] for identifier in local_ids], dtype=np.int64
    )
    representatives = source_ids[np.asarray(periodic.vertex_representatives)[selected]]
    if any(int(identifier) not in local_rows for identifier in representatives):
        raise ValueError(
            "project_source_periodic_topology.lifted must retain source orbit representatives; got a missing representative"
        )
    local_representatives = np.asarray(
        [local_rows[int(identifier)] for identifier in representatives],
        dtype=np.int32,
    )
    shifts = np.asarray(periodic.vertex_shifts)[selected]
    actual_geometry = None
    if periodic.actual_geometry is not None:
        parent = periodic.actual_geometry.exact_source
        if not isinstance(
            parent,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        ):
            raise TypeError(
                "project_source_periodic_topology.geometry must bind a power geometry source; got a different source"
            )
        restricted = ExactPowerCellGeometryRestrictionSource(
            parent,
            np.empty((0, 4), dtype=np.float64),
            np.column_stack((selected, np.full(selected.size, -1, dtype=np.int64))),
            np.full(selected.size, -1, dtype=np.int64),
        )
        actual_geometry = CellGeometrySpec.power(lifted, restricted, budget=budget)
    probe = PeriodicMeshTopology(
        lifted,
        periodic.cell,
        local_representatives,
        shifts,
        actual_geometry=actual_geometry,
        budget=budget,
    )
    identifiers = {}
    for degree in range(source.topological_dimension + 1):
        original = dict(
            zip(
                periodic.entity_keys(degree),
                np.asarray(periodic.quotient.entities(degree).entity_ids).tolist(),
                strict=True,
            )
        )
        keys = probe.entity_keys(degree)
        if any(key not in original for key in keys):
            raise ValueError(
                "project_source_periodic_topology.lifted must retain source winding keys; got a changed key"
            )
        expected = np.asarray([original[key] for key in keys], dtype=np.int64)
        if 0 < degree < source.topological_dimension:
            identifiers[degree] = expected
        elif not np.array_equal(
            expected, np.asarray(probe.quotient.entities(degree).entity_ids)
        ):
            raise ValueError(
                "project_source_periodic_topology.lifted must retain source orbit members; got a missing member"
            )
    return PeriodicMeshTopology(
        lifted,
        periodic.cell,
        local_representatives,
        shifts,
        entity_global_ids=identifiers,
        actual_geometry=actual_geometry,
        entity_allocator_next_ids={
            degree: cursor
            for degree, cursor in enumerate(periodic.allocator_next_ids)
            if 0 < degree < source.topological_dimension and cursor >= 0
        },
        budget=budget,
    )


@checked
def original_p1_vertex_images(
    original: result_owner.CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> dict[int, tuple[Fraction, ...]]:
    """Bind physical P1 corner images to scientific vertex IDs."""

    uniform_retain_source(original, budget=budget)
    elements, routes, coordinates = original.geometry.resolve(
        original.mesh, budget=budget
    )
    uniform_count(coordinates.size, coordinates.size * 512, budget=budget)
    bank = prepared_coordinate_source_bank(original.geometry, budget=budget)
    images: dict[int, tuple[Fraction, ...]] = {}
    vertex_ids = np.asarray(original.mesh.vertex_global_ids)
    carrier = np.asarray(original.mesh.coordinates)
    for block, element, route in zip(original.mesh.blocks, elements, routes, strict=True):
        root = element
        while isinstance(root, BarycentricCellGeometryElement):
            root = root.source_element
        require_p1_cardinal_source(root)
        if root.cell_kind != block.cell_kind:
            raise ValueError(
                "original_p1_vertex_images.element must match the cell kind; got a different source basis"
            )
        uniform_count(
            block.vertices.size,
            block.vertices.size * carrier.shape[1] * 512,
            budget=budget,
        )
        for row, coefficients in zip(
            np.asarray(block.vertices), np.asarray(route), strict=True
        ):
            corners = coordinate_corner_images(
                element,
                tuple(bank[int(coefficient)] for coefficient in coefficients),
                budget=budget,
            )
            if corners is None:
                raise ValueError(
                    "original_p1_vertex_images.source must retain its affine P1 coefficient law; got a different law"
                )
            for vertex, image in zip(row, corners, strict=True):
                image = tuple(
                    rational_coefficient(value, role="original_p1_vertex_images.corner")
                    for value in image
                )
                identifier = int(vertex_ids[vertex])
                if identifier in images and images[identifier] != image:
                    raise ValueError(
                        "original_p1_vertex_images.cells must agree on shared physical corners; got different corner images"
                    )
                if not np.array_equal(
                    np.asarray(tuple(float(value) for value in image), dtype=np.float64),
                    carrier[vertex],
                ):
                    raise ValueError(
                        "original_p1_vertex_images.corner must match its RNE carrier; got a different physical image"
                    )
                images[identifier] = image
    if set(images) != set(vertex_ids.tolist()):
        raise ValueError(
            "original_p1_vertex_images.source must bind every carrier vertex; got an unbound vertex"
        )
    return images


@jax.jit
def _collective_tree_depth(parents: Array, cursors: Array, /) -> Array:
    def tree_depth(parent_records: Array, cursor: Array) -> Array:
        def visit(slot: Array, levels: Array) -> Array:
            parent = parent_records[slot] // 2
            level = jnp.where(parent >= 0, levels[jnp.maximum(parent, 0)] + 1, 0)
            return levels.at[slot].set(level)

        levels = jax.lax.fori_loop(
            0,
            cursor.astype(jnp.int32),
            visit,
            jnp.zeros(parent_records.shape, dtype=jnp.int32),
        )
        return jnp.max(levels)

    return jnp.max(jax.vmap(tree_depth)(parents, cursors))


@checked
def collective_action_stacks(
    states: AdaptiveSimplexState,
    uniform: bisection_owner.BisectionUniformRefinement | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Array, Array, Array]:

    uniform_count(states.mesh.cell_ids.size, states.mesh.cell_ids.size * 4, budget=budget)
    depth = (
        max(
            1,
            int(
                jax.device_get(
                    _collective_tree_depth(states.parents, states.cursors[:, 1])
                )
            ),
        )
        + 1
    )
    parts, cells, width = states.mesh.cells.shape
    uniform_count(
        parts * cells * depth * width * width,
        parts * cells * (depth * width * width * 8 + 12),
        budget=budget,
    )
    if uniform is None:
        child_ids = jnp.zeros((0,), dtype=jnp.int64)
        parent_ids = child_ids
        root_actions = jnp.zeros((0, width, width), dtype=jnp.float64)
    else:
        child_ids = uniform.child_ids.reshape(-1)
        parent_ids = jnp.broadcast_to(
            uniform.parent_ids[:, None], uniform.child_ids.shape
        ).reshape(-1)
        root_actions = uniform.barycentric_weights.reshape((-1, width, width))
        order = jnp.argsort(child_ids, stable=True)
        child_ids, parent_ids, root_actions = (
            child_ids[order],
            parent_ids[order],
            root_actions[order],
        )
    return _collective_action_values(states, child_ids, parent_ids, root_actions, depth)


@eqx.filter_jit
def _collective_action_values(
    states: AdaptiveSimplexState,
    child_ids: Array,
    parent_ids: Array,
    root_actions: Array,
    depth: int,
    /,
) -> tuple[Array, Array, Array]:
    _, cells, width = states.mesh.cells.shape

    def part_actions(state: AdaptiveSimplexState) -> tuple[Array, Array, Array]:
        def cell_actions(slot: Array) -> tuple[Array, Array, Array]:
            def step(
                level: Array, carry: tuple[Array, Array, Array]
            ) -> tuple[Array, Array, Array]:
                current, actions, count = carry
                parent = state.parents[current] // 2
                present = parent >= 0
                safe = jnp.maximum(parent, 0)
                corners = state.mesh.vertex_ids[state.mesh.cells[current]]
                original = state.mesh.vertex_ids[state.mesh.cells[safe]]
                midpoint = state.mesh.vertex_ids[
                    jnp.maximum(state.bisection_vertices[safe], 0)
                ]
                ordered = state.mesh.vertex_ids[state.tuples[safe]]
                endpoints = jnp.stack((ordered[0], ordered[state.tags[safe]]))
                action = jnp.where(
                    corners[:, None] == midpoint,
                    0.5
                    * jnp.any(
                        original[None, :, None] == endpoints[None, None, :], axis=2
                    ),
                    corners[:, None] == original[None, :],
                ).astype(jnp.float64)
                return (
                    jnp.where(present, safe, current),
                    actions.at[level].set(jnp.where(present, action, 0.0)),
                    count + present.astype(jnp.int32),
                )

            current, actions, count = jax.lax.fori_loop(
                0,
                depth - 1,
                step,
                (
                    slot,
                    jnp.zeros((depth, width, width), dtype=jnp.float64),
                    jnp.asarray(0, dtype=jnp.int32),
                ),
            )
            root = state.mesh.cell_ids[current]
            if child_ids.shape[0]:
                position = jnp.minimum(
                    jnp.searchsorted(child_ids, root), child_ids.shape[0] - 1
                )
                member = child_ids[position] == root
                actions = actions.at[count].set(
                    jnp.where(member, root_actions[position], 0.0)
                )
                count += member.astype(jnp.int32)
                root = jnp.where(member, parent_ids[position], root)
            active = state.mesh.cell_active[slot]
            return (
                jnp.where(active, root, -1),
                jnp.where(active, count, 0),
                jnp.where(active, actions, 0.0),
            )

        return jax.vmap(cell_actions)(jnp.arange(cells, dtype=jnp.int32))

    return jax.vmap(part_actions)(states)


class _GeometrySourceBlock(StrictModule):
    name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    cell_ids: Array
    vertex_ids: Array
    routes: Array
    element: CellGeometryElement


class _GeometrySource(StrictModule):
    coordinate_count: int = eqx.field(static=True)
    coefficients: Array
    coordinate_ids: Array
    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    blocks: tuple[_GeometrySourceBlock, ...]


def _prepare_geometry_source(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    budget: CoordinateEnclosureBudget,
) -> _GeometrySource:
    match original:
        case initial_certification_owner.InitialCollectiveMeshEvidence():
            original.require_passed()
            source_banks = dict(original.logical_arrays)
            coordinate_count = original.global_entity_counts[0]
            coefficients = source_banks["geometry/coordinates"][:coordinate_count]
            scientific_coordinate_ids = source_banks["geometry/coordinate_ids"][
                :coordinate_count
            ]
            source_geometry_id = original.coordinate_geometry_id
            source_topology_id = original.topology_id

            cells = original.global_entity_counts[-1]
            source_blocks = (
                _GeometrySourceBlock(
                    "surface",
                    "triangle",
                    source_banks["geometry/cell_ids/surface"][:cells],
                    source_banks["cell_vertices"][:cells],
                    source_banks["geometry/routes/surface"][:cells],
                    coordinate_lagrange_element("triangle", 1),
                ),
            )
        case result_owner.CellMeshingResult():
            elements, routes, coefficients = original.geometry.resolve(
                original.mesh, budget=budget
            )
            coordinate_count = coefficients.shape[0]
            # Dense routes explicitly address the authored coefficient axis.
            scientific_coordinate_ids = jnp.arange(coordinate_count, dtype=jnp.int64)
            source_geometry_id = cell_geometry_id(original.geometry)
            source_topology_id = original.mesh.topology_id
            source_blocks = tuple(
                _GeometrySourceBlock(
                    block.name,
                    block.cell_kind,
                    jnp.asarray(block.global_ids, dtype=jnp.int64),
                    original.mesh.vertex_global_ids[block.vertices],
                    scientific_coordinate_ids[route],
                    element,
                )
                for block, element, route in zip(
                    original.mesh.blocks, elements, routes, strict=True
                )
            )
        case _:
            raise TypeError(
                "collective_geometry.original must be initial evidence or a cell meshing result; "
                f"got {type(original).__name__}"
            )
    return _GeometrySource(
        coordinate_count,
        coefficients,
        scientific_coordinate_ids,
        source_geometry_id,
        source_topology_id,
        source_blocks,
    )


class _GeometryActions(StrictModule):
    corners: Array
    counts: Array
    weights: Array


def _prepare_geometry_actions(
    states: AdaptiveSimplexState,
    uniform_refinement: bisection_owner.BisectionUniformRefinement | None,
    target_ids: tuple[Array, ...],
    counts: tuple[int, ...],
    cell_sources: Array,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> _GeometryActions:
    width = dimension + 1
    action_roots, action_counts, action_weights = collective_action_stacks(
        states, uniform_refinement, budget=budget
    )
    flat_cell_ids = states.mesh.cell_ids.reshape(-1)
    flat_active = states.mesh.cell_active.reshape(-1)
    order = jnp.argsort(
        jnp.where(flat_active, flat_cell_ids, jnp.iinfo(jnp.int64).max), stable=True
    )[: target_ids[-1].shape[0]]
    global_corners = jax.vmap(lambda part: part.mesh.vertex_ids[part.mesh.cells])(
        states
    ).reshape((-1, width))[order]
    roots = action_roots.reshape(-1)[order]
    counts_by_cell = action_counts.reshape(-1)[order]
    actions_by_cell = action_weights.reshape((-1, *action_weights.shape[-3:]))[order]
    if not bool(
        jax.device_get(
            jnp.all(
                (jnp.arange(cell_sources.shape[0]) >= counts[-1])
                | (roots == cell_sources)
            )
        )
    ):
        raise ValueError(
            "collective_geometry.action_roots must match scientific cell routes; got different roots"
        )
    return _GeometryActions(global_corners, counts_by_cell, actions_by_cell)


class _GeometryBlockRestriction(StrictModule):
    arrays: tuple[tuple[str, Array], ...]
    shapes: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    block_shape: tuple[str, str, int, str] = eqx.field(static=True)
    coordinate_owners: Array
    restored: Array


@jax.jit
def _geometry_bank_membership(
    original_cells: Array, cell_sources: Array, cell_count: int
) -> tuple[Array, Array, Array, Array, Array]:
    order = jnp.argsort(original_cells)
    sorted_cells = original_cells[order]
    positions = jnp.minimum(
        jnp.searchsorted(sorted_cells, cell_sources), sorted_cells.shape[0] - 1
    )
    member = (jnp.arange(cell_sources.shape[0]) < cell_count) & (
        sorted_cells[positions] == cell_sources
    )
    return order, sorted_cells, positions, member, jnp.sum(member, dtype=jnp.int64)


@partial(jax.jit, static_argnames=("capacity",))
def _geometry_block_restriction_values(
    original_order: Array,
    sorted_cells: Array,
    positions: Array,
    member: Array,
    block_count: Array,
    original_vertices: Array,
    original_routes: Array,
    scientific_coordinate_ids: Array,
    coordinate_owners: Array,
    target_cells: Array,
    target_cell_owners: Array,
    cell_sources: Array,
    global_corners: Array,
    counts_by_cell: Array,
    actions_by_cell: Array,
    restored: Array,
    part_count: int,
    *,
    capacity: int,
) -> tuple[tuple[Array, ...], Array, Array]:
    selected = jnp.nonzero(member, size=capacity, fill_value=0)[0]
    root_rows = original_order[positions[selected]]
    root_vertices = original_vertices[root_rows]
    counts_for_block = counts_by_cell[selected]
    actions_for_block = actions_by_cell[selected]
    coefficient_routes = original_routes[root_rows]
    coefficient_rows = jnp.searchsorted(scientific_coordinate_ids, coefficient_routes)
    prefix = jnp.arange(capacity) < block_count
    coordinate_owners = coordinate_owners.at[coefficient_rows.reshape(-1)].min(
        jnp.broadcast_to(
            jnp.where(prefix, target_cell_owners[selected], part_count)[:, None],
            coefficient_routes.shape,
        ).reshape(-1)
    )
    original_count = sorted_cells.shape[0]
    if capacity < original_count:
        restored = jnp.asarray(False)
    else:
        restored &= (
            (block_count == original_count)
            & jnp.all(target_cells[selected[:original_count]] == sorted_cells)
            & jnp.all(
                global_corners[selected[:original_count]]
                == original_vertices[original_order]
            )
            & jnp.all(counts_for_block[:original_count] == 0)
        )
    return (
        (
            jnp.where(prefix, target_cells[selected], -1),
            jnp.where(prefix[:, None], coefficient_routes, -1),
            jnp.where(prefix, cell_sources[selected], -1),
            jnp.where(prefix[:, None], root_vertices, -1),
            jnp.where(prefix, counts_for_block, 0),
            jnp.where(prefix[:, None, None, None], actions_for_block, 0.0),
        ),
        coordinate_owners,
        restored,
    )


def _prepare_geometry_block_restriction(
    block: _GeometrySourceBlock,
    cell_sources: Array,
    scientific_coordinate_ids: Array,
    coordinate_owners: Array,
    target_cells: Array,
    target_cell_owners: Array,
    logical_cell_count: int,
    part_count: int,
    actions: _GeometryActions,
    restored: Array,
) -> _GeometryBlockRestriction:
    global_corners, counts_by_cell, actions_by_cell = (
        actions.corners,
        actions.counts,
        actions.weights,
    )
    sharding = target_cells.sharding
    arrays: dict[str, Array] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    block_name, cell_kind = block.name, block.cell_kind
    original_cells, original_vertices, original_routes, element = (
        block.cell_ids,
        block.vertex_ids,
        block.routes,
        block.element,
    )
    original_order, sorted_cells, positions, member, count_value = (
        _geometry_bank_membership(original_cells, cell_sources, logical_cell_count)
    )
    block_count = int(jax.device_get(count_value))
    capacity = ((block_count + part_count - 1) // part_count) * part_count
    # Capacity is host-owned; counts and scientific values remain dynamic.

    endpoint = element
    while isinstance(endpoint, BarycentricCellGeometryElement):
        endpoint = endpoint.source_element
    require_p1_cardinal_source(endpoint)
    values, coordinate_owners, restored = _geometry_block_restriction_values(
        original_order,
        sorted_cells,
        positions,
        member,
        count_value,
        original_vertices,
        original_routes,
        scientific_coordinate_ids,
        coordinate_owners,
        target_cells,
        target_cell_owners,
        cell_sources,
        global_corners,
        counts_by_cell,
        actions_by_cell,
        restored,
        part_count,
        capacity=capacity,
    )
    digest = canonical_fingerprint(
        {
            "signature": coordinate_source_signature(element),
            "arrays": array_tree_fingerprint(element),
        }
    )
    basis = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).copy()
    entries = {
        f"geometry/cell_ids/{block_name}": values[0],
        f"geometry/routes/{block_name}": values[1],
        f"geometry/parent_cell_ids/{block_name}": values[2],
        f"geometry/parent_vertex_ids/{block_name}": values[3],
        f"geometry/action_counts/{block_name}": values[4],
        f"geometry/action_weights/{block_name}": values[5],
        f"geometry/source_basis/{block_name}": jnp.broadcast_to(
            jnp.asarray(basis), (capacity, 32)
        ),
    }
    for name, value in entries.items():
        arrays[name] = jax.device_put(value, sharding)
        shapes[name] = (block_count, *value.shape[1:])
    return _GeometryBlockRestriction(
        tuple(arrays.items()),
        tuple(shapes.items()),
        (block_name, cell_kind, block_count, element.element_id),
        coordinate_owners,
        restored,
    )


@checked
def build_collective_geometry_witness(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    states: AdaptiveSimplexState,
    organization_arrays: tuple[tuple[str, Array], ...],
    target_ids: tuple[Array, ...],
    target_owners: tuple[Array, ...],
    counts: tuple[int, ...],
    /,
    *,
    uniform_refinement: bisection_owner.BisectionUniformRefinement | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[str, Array], ...], str]:
    """Publish source coefficients and composed dyadic restriction charts.

    Mesh coordinates remain the FP64 topological carrier. The physical map is
    the source scalar coordinate basis restricted by the explicit charts,
    including when its midpoint image is not representable by that carrier.
    """
    banks = dict(organization_arrays)
    dimension = len(counts) - 1
    cell_sources = banks["organization/cell/source_ids"]
    sharding = target_ids[0].sharding
    part_count = states.mesh.cell_ids.shape[0]

    source = _prepare_geometry_source(original, budget)
    coordinate_count = source.coordinate_count
    coefficients = source.coefficients
    scientific_coordinate_ids = source.coordinate_ids
    source_geometry_id, source_topology_id = source.geometry_id, source.topology_id
    source_blocks = source.blocks
    padded_count = ((coordinate_count + part_count - 1) // part_count) * part_count
    pad = padded_count - coordinate_count
    coordinate_owners = jnp.full((coordinate_count,), part_count, dtype=jnp.int32)
    arrays: dict[str, Array] = {
        "geometry/coordinate_ids": jax.device_put(
            jnp.pad(scientific_coordinate_ids, (0, pad), constant_values=-1), sharding
        ),
        "geometry/coordinates": jax.device_put(
            jnp.pad(coefficients, ((0, pad), (0, 0))), sharding
        ),
    }
    shapes = {
        "geometry/coordinate_ids": (coordinate_count,),
        "geometry/coordinate_owners": (coordinate_count,),
        "geometry/coordinates": coefficients.shape,
    }
    metadata_sharding = (
        NamedSharding(sharding.mesh, PartitionSpec())
        if isinstance(sharding, NamedSharding)
        else sharding
    )

    def identity_bytes(value: str) -> Array:
        return jax.device_put(
            np.frombuffer(
                bytes.fromhex(canonical_fingerprint(value)), dtype=np.uint8
            ).copy(),
            metadata_sharding,
        )

    arrays["geometry/source_geometry_id"] = identity_bytes(source_geometry_id)
    arrays["geometry/source_topology_id"] = identity_bytes(source_topology_id)
    shapes["geometry/source_geometry_id"] = (32,)
    shapes["geometry/source_topology_id"] = (32,)

    actions = _prepare_geometry_actions(
        states, uniform_refinement, target_ids, counts, cell_sources, dimension, budget
    )
    restored = jnp.asarray(True)
    all_shapes = []
    for block in source_blocks:
        restriction = _prepare_geometry_block_restriction(
            block,
            cell_sources,
            scientific_coordinate_ids,
            coordinate_owners,
            target_ids[-1],
            target_owners[-1],
            counts[-1],
            part_count,
            actions,
            restored,
        )
        arrays.update(restriction.arrays)
        shapes.update(restriction.shapes)
        coordinate_owners, restored = restriction.coordinate_owners, restriction.restored
        all_shapes.append(restriction.block_shape)
    if not bool(jax.device_get(jnp.all(coordinate_owners < part_count))):
        raise ValueError(
            "collective_geometry.coordinate_owners must cover every source coefficient; got an unassigned coefficient"
        )
    arrays["geometry/coordinate_owners"] = jax.device_put(
        jnp.pad(coordinate_owners, (0, pad), constant_values=-1),
        sharding,
    )
    digest = logical_array_value_collection_digest(
        {
            name: value
            for name, value in arrays.items()
            if name != "geometry/coordinate_owners"
        },
        logical_shapes={
            name: shape
            for name, shape in shapes.items()
            if name != "geometry/coordinate_owners"
        },
    )
    geometry_id = canonical_fingerprint(
        {
            "kind": "collective-original-source-restriction-geometry",
            "source_geometry": source_geometry_id,
            "source_topology": source_topology_id,
            "blocks": all_shapes,
            "coordinate_count": coordinate_count,
            "arrays": digest,
        }
    )
    if bool(jax.device_get(restored)):
        # Coefficients and source basis were retained unchanged. Equal scientific
        # cell IDs/corners and zero action counts restore the source map itself,
        # so reuse its geometry identity rather than the restriction wrapper ID.
        geometry_id = source_geometry_id
    return tuple(sorted(arrays.items())), geometry_id
