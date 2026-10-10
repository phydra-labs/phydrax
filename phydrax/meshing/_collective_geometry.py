#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

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
from ..discretization._adaptive_simplex import AdaptiveSimplexState
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import coordinate_source_signature


if TYPE_CHECKING:
    from ..discretization._cell_mesh import CellMesh
    from ..discretization._periodic_topology import PeriodicMeshTopology
    from ._bisection import BisectionUniformRefinement
    from ._initial_certification import InitialCollectiveMeshEvidence
    from ._result import CellMeshingResult


def periodic_source_closure_cells(
    source: CellMesh,
    cell_ids: ArrayLike,
    /,
    *,
    cell_capacity: int,
    vertex_capacity: int,
) -> np.ndarray:
    """Include genuine source orbit identity members under original local caps."""
    periodic = source.periodic_topology
    requested = np.unique(np.asarray(cell_ids, dtype=np.int64))
    if periodic is None:
        return requested
    from .._meshcore import charge_native_geometry_queries

    ids = np.concatenate([np.asarray(block.global_ids) for block in source.blocks])
    corners = np.concatenate(
        [
            np.asarray(source.vertex_global_ids)[np.asarray(block.vertices)]
            for block in source.blocks
        ]
    )
    rows = {int(identifier): row for row, identifier in enumerate(ids)}
    if any(int(identifier) not in rows for identifier in requested):
        raise ValueError("Periodic support names an absent immutable source cell.")
    selected = np.zeros(ids.size, dtype=np.bool_)
    selected[[rows[int(identifier)] for identifier in requested]] = True
    orbits = np.asarray(periodic.orbits(source.topological_dimension)[0])
    vertices = np.asarray(source.vertex_global_ids)
    representatives = vertices[np.asarray(periodic.vertex_representatives)]
    representative_by_id = dict(
        zip(vertices.tolist(), representatives.tolist(), strict=True)
    )
    while True:
        charge_native_geometry_queries(0, work_units=int(corners.size + ids.size))
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
                        "An actual source orbit representative has no incident source cell."
                    )
                selected[incident[np.argmin(ids[incident])]] = True
        actual_vertices = np.unique(corners[selected]).size
        if (
            np.count_nonzero(selected) > cell_capacity
            or actual_vertices > vertex_capacity
        ):
            raise ValueError(
                "Actual periodic source support exceeds the original closure capacity."
            )
        if not missing.size:
            return np.sort(ids[selected])


def project_source_periodic_topology(
    source: CellMesh, lifted: CellMesh, /
) -> PeriodicMeshTopology | None:
    """Reconstruct a closure's quotient from the actual immutable source.

    This is a restriction, not a new periodic identification. Missing source
    representatives or orbit members are a producer error, never an invitation
    to renumber the quotient or to change the authored winding.
    """
    from ..discretization._periodic_topology import PeriodicMeshTopology

    periodic = source.periodic_topology
    if periodic is None:
        return None
    source_ids = np.asarray(source.vertex_global_ids)
    local_ids = np.asarray(lifted.vertex_global_ids)
    source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}
    local_rows = {int(identifier): row for row, identifier in enumerate(local_ids)}
    if any(int(identifier) not in source_rows for identifier in local_ids):
        raise ValueError(
            "Periodic source projection contains an unauthored lifted vertex."
        )
    selected = np.asarray(
        [source_rows[int(identifier)] for identifier in local_ids], dtype=np.int64
    )
    representatives = source_ids[np.asarray(periodic.vertex_representatives)[selected]]
    if any(int(identifier) not in local_rows for identifier in representatives):
        raise ValueError("Periodic closure lacks an actual source orbit representative.")
    local_representatives = np.asarray(
        [local_rows[int(identifier)] for identifier in representatives],
        dtype=np.int32,
    )
    shifts = np.asarray(periodic.vertex_shifts)[selected]
    actual_geometry = None
    if periodic.actual_geometry is not None:
        from ..discretization._cell_geometry import CellGeometrySpec
        from ..discretization._exact_power_geometry import (
            ExactPowerCellGeometryLinearActionSource,
            ExactPowerCellGeometryRestrictionSource,
            ExactPowerCellGeometrySource,
        )

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
                "Periodic actual geometry projection requires exact power source authority."
            )
        restricted = ExactPowerCellGeometryRestrictionSource(
            parent,
            np.empty((0, 4), dtype=np.float64),
            np.column_stack((selected, np.full(selected.size, -1, dtype=np.int64))),
            np.full(selected.size, -1, dtype=np.int64),
        )
        actual_geometry = CellGeometrySpec.power(lifted, restricted)
    probe = PeriodicMeshTopology(
        lifted,
        periodic.cell,
        local_representatives,
        shifts,
        actual_geometry=actual_geometry,
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
            raise ValueError("Periodic closure changed an actual source winding key.")
        expected = np.asarray([original[key] for key in keys], dtype=np.int64)
        if 0 < degree < source.topological_dimension:
            identifiers[degree] = expected
        elif not np.array_equal(
            expected, np.asarray(probe.quotient.entities(degree).entity_ids)
        ):
            raise ValueError(
                "Periodic closure lacks the persistent source orbit identity member."
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
    )


def _original_p1_vertex_images(
    original: CellMeshingResult,
    /,
) -> dict[int, tuple[Fraction, ...]]:
    """Bind exact physical corner images to original scientific vertex IDs."""
    from ..discretization._cell_geometry import (
        _require_p1_cardinal_source,
        BarycentricCellGeometryElement,
    )
    from ..discretization._coordinate_enclosure import (
        coordinate_corner_images,
        prepared_coordinate_source_bank,
    )
    from ._bisection import _uniform_charge, _uniform_retain_source

    _uniform_retain_source(original)
    elements, routes, coordinates = original.geometry.resolve(original.mesh)
    _uniform_charge(coordinates.size, coordinates.size * 512)
    bank = prepared_coordinate_source_bank(original.geometry)
    images: dict[int, tuple[Fraction, ...]] = {}
    vertex_ids = np.asarray(original.mesh.vertex_global_ids)
    carrier = np.asarray(original.mesh.coordinates)
    for block, element, route in zip(original.mesh.blocks, elements, routes, strict=True):
        root = element
        while isinstance(root, BarycentricCellGeometryElement):
            root = root.source_element
        _require_p1_cardinal_source(root)
        if root.cell_kind != block.cell_kind:
            raise ValueError(
                "Original physical source basis and scientific cell kind differ."
            )
        _uniform_charge(block.vertices.size, block.vertices.size * carrier.shape[1] * 512)
        for row, coefficients in zip(
            np.asarray(block.vertices), np.asarray(route), strict=True
        ):
            corners = coordinate_corner_images(
                element, tuple(bank[int(coefficient)] for coefficient in coefficients)
            )
            if corners is None:
                raise ValueError(
                    "An original affine P1 source lost its exact full coefficient law."
                )
            for vertex, image in zip(row, corners, strict=True):
                identifier = int(vertex_ids[vertex])
                if identifier in images and images[identifier] != image:
                    raise ValueError(
                        "Incident original P1 cells disagree on their exact physical source corner."
                    )
                if not np.array_equal(
                    np.asarray(tuple(float(value) for value in image), dtype=np.float64),
                    carrier[vertex],
                ):
                    raise ValueError(
                        "Original P1 physical source corner differs from its RNE carrier."
                    )
                images[identifier] = image
    if set(images) != set(vertex_ids.tolist()):
        raise ValueError(
            "Original P1 source does not own every scientific carrier vertex."
        )
    return images


def _collective_action_stacks(
    states: AdaptiveSimplexState,
    uniform: BisectionUniformRefinement | None,
    /,
) -> tuple[Array, Array, Array]:
    from ._bisection import _uniform_charge

    cells = states.mesh.cell_ids.shape[1]

    def tree_depth(state: AdaptiveSimplexState) -> Array:
        def visit(slot: Array, levels: Array) -> Array:
            parent = state.parents[slot] // 2
            level = jnp.where(parent >= 0, levels[jnp.maximum(parent, 0)] + 1, 0)
            return levels.at[slot].set(level)

        levels = jax.lax.fori_loop(
            0,
            state.cursors[1].astype(jnp.int32),
            visit,
            jnp.zeros((cells,), dtype=jnp.int32),
        )
        return jnp.max(levels)

    _uniform_charge(states.mesh.cell_ids.size, states.mesh.cell_ids.size * 4)
    depth = max(1, int(jax.device_get(jnp.max(jax.vmap(tree_depth)(states)))) + 1)
    parts, cells, width = states.mesh.cells.shape
    _uniform_charge(
        parts * cells * depth * width * width,
        parts * cells * (depth * width * width * 8 + 12),
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


def build_collective_geometry_witness(
    original: CellMeshingResult | InitialCollectiveMeshEvidence,
    states: AdaptiveSimplexState,
    organization_arrays: tuple[tuple[str, Array], ...],
    target_ids: tuple[Array, ...],
    target_owners: tuple[Array, ...],
    counts: tuple[int, ...],
    /,
    *,
    uniform_refinement: BisectionUniformRefinement | None = None,
) -> tuple[tuple[tuple[str, Array], ...], str]:
    """Publish exact original coefficients and composed dyadic source charts.

    Mesh coordinates remain the FP64 topological carrier. The physical map is
    the original scalar coordinate basis restricted by the explicit charts,
    including when its midpoint image is not representable by that carrier.
    """
    banks = dict(organization_arrays)
    dimension = len(counts) - 1
    cell_sources = banks["organization/cell/source_ids"]
    sharding = target_ids[0].sharding
    part_count = states.mesh.cell_ids.shape[0]
    from ._initial_certification import InitialCollectiveMeshEvidence

    if isinstance(original, InitialCollectiveMeshEvidence):
        original.require_passed()
        source_banks = dict(original.logical_arrays)
        coordinate_count = original.global_entity_counts[0]
        coefficients = source_banks["geometry/coordinates"][:coordinate_count]
        scientific_coordinate_ids = source_banks["geometry/coordinate_ids"][
            :coordinate_count
        ]
        source_geometry_id = original.coordinate_geometry_id
        source_topology_id = original.topology_id
        from ..discretization._cell_geometry import coordinate_lagrange_element

        cells = original.global_entity_counts[-1]
        source_blocks = (
            (
                "surface",
                "triangle",
                source_banks["geometry/cell_ids/surface"][:cells],
                source_banks["cell_vertices"][:cells],
                source_banks["geometry/routes/surface"][:cells],
                coordinate_lagrange_element("triangle", 1),
            ),
        )
    else:
        elements, routes, coefficients = original.geometry.resolve(original.mesh)
        coordinate_count = coefficients.shape[0]
        # Dense routes explicitly address the authored coefficient axis.
        scientific_coordinate_ids = jnp.arange(coordinate_count, dtype=jnp.int64)
        source_geometry_id = cell_geometry_id(original.geometry)
        source_topology_id = original.mesh.topology_id
        source_blocks = tuple(
            (
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

    width = dimension + 1
    action_roots, action_counts, action_weights = _collective_action_stacks(
        states, uniform_refinement
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
            "Full action paths differ from their explicit original scientific cell routes."
        )
    restored = jnp.asarray(True)
    all_shapes = []
    for (
        block_name,
        cell_kind,
        original_cells,
        original_vertices,
        original_routes,
        element,
    ) in source_blocks:
        original_order = jnp.argsort(original_cells)
        sorted_cells = original_cells[original_order]
        positions = jnp.minimum(
            jnp.searchsorted(sorted_cells, cell_sources), sorted_cells.shape[0] - 1
        )
        member = (jnp.arange(cell_sources.shape[0]) < counts[-1]) & (
            sorted_cells[positions] == cell_sources
        )
        block_count = int(jax.device_get(jnp.sum(member, dtype=jnp.int64)))
        capacity = ((block_count + part_count - 1) // part_count) * part_count
        selected = jnp.nonzero(member, size=capacity, fill_value=0)[0]
        root_rows = original_order[positions[selected]]
        root_vertices = original_vertices[root_rows]
        counts_for_block = counts_by_cell[selected]
        actions_for_block = actions_by_cell[selected]
        from ..discretization._cell_geometry import (
            _require_p1_cardinal_source,
            BarycentricCellGeometryElement,
        )

        endpoint = element
        while isinstance(endpoint, BarycentricCellGeometryElement):
            endpoint = endpoint.source_element
        _require_p1_cardinal_source(endpoint)
        coefficient_routes = original_routes[root_rows]
        coefficient_rows = jnp.searchsorted(scientific_coordinate_ids, coefficient_routes)
        prefix = jnp.arange(capacity) < block_count
        coordinate_owners = coordinate_owners.at[coefficient_rows.reshape(-1)].min(
            jnp.broadcast_to(
                jnp.where(prefix, target_owners[-1][selected], part_count)[:, None],
                coefficient_routes.shape,
            ).reshape(-1)
        )
        if block_count != original_cells.shape[0]:
            restored = jnp.asarray(False)
        else:
            restored &= (
                jnp.all(target_ids[-1][selected[:block_count]] == sorted_cells)
                & jnp.all(
                    global_corners[selected[:block_count]]
                    == original_vertices[original_order]
                )
                & jnp.all(counts_for_block[:block_count] == 0)
            )
        digest = canonical_fingerprint(
            {
                "signature": coordinate_source_signature(element),
                "arrays": array_tree_fingerprint(element),
            }
        )
        basis = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).copy()
        entries = {
            f"geometry/cell_ids/{block_name}": jnp.where(
                prefix, target_ids[-1][selected], -1
            ),
            f"geometry/routes/{block_name}": jnp.where(
                prefix[:, None], coefficient_routes, -1
            ),
            f"geometry/parent_cell_ids/{block_name}": jnp.where(
                prefix, cell_sources[selected], -1
            ),
            f"geometry/parent_vertex_ids/{block_name}": jnp.where(
                prefix[:, None], root_vertices, -1
            ),
            f"geometry/action_counts/{block_name}": jnp.where(
                prefix, counts_for_block, 0
            ),
            f"geometry/action_weights/{block_name}": jnp.where(
                prefix[:, None, None, None], actions_for_block, 0.0
            ),
            f"geometry/source_basis/{block_name}": jnp.broadcast_to(
                jnp.asarray(basis), (capacity, 32)
            ),
        }
        for name, value in entries.items():
            arrays[name] = jax.device_put(value, sharding)
            shapes[name] = (block_count, *value.shape[1:])
        all_shapes.append((block_name, cell_kind, block_count, element.element_id))
    if not bool(jax.device_get(jnp.all(coordinate_owners < part_count))):
        raise ValueError(
            "An original scientific coordinate has no target route owning its occurrence."
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
        # This is numerical equality of the complete scientific coefficient
        # field, routes and charts, not a relabeled acceptance report.
        geometry_id = source_geometry_id
    return tuple(sorted(arrays.items())), geometry_id
