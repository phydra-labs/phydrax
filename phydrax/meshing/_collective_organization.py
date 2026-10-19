#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils

from phydrax.meshing._entity_keys import facet_packets

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._strict import StrictModule
from ..discretization._adaptive_simplex import AdaptiveSimplexState
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import checked
from . import (
    _bisection as bisection_owner,
    _contracts as contracts_owner,
    _domain as domain_owner,
    _initial_certification as initial_certification_owner,
    _organization_records as organization_owner,
    _result as result_owner,
)
from ._entity_keys import entity_keys, key_positions
from ._scope import MeshingEntityKind, MeshingScope, resolve_mesh_scope


type OrganizationDefinitions = tuple[
    tuple[str, tuple[organization_owner.MeshPatch, ...]],
    tuple[str, tuple[organization_owner.MeshZone, ...]],
    tuple[str, tuple[organization_owner.MeshLabel, ...]],
    tuple[str, tuple[organization_owner.MeshAttribute, ...]],
]


@checked
def initial_organization_definitions(
    evidence: initial_certification_owner.InitialCollectiveMeshEvidence, /
) -> OrganizationDefinitions:
    """Return authored source declarations for accepted collective evidence."""
    return _initial_organization_definitions(evidence.compiled, evidence.specification)


def _initial_organization_definitions(
    compiled: domain_owner.CompiledSurfaceDomain,
    specification: contracts_owner.SurfaceMeshingSpec,
    /,
) -> OrganizationDefinitions:
    """Return authored definitions independently of owner-local rows."""
    _owner = organization_owner

    domain = compiled.domain

    def scope(degree: int, indices: Sequence[int]) -> MeshingScope:
        return MeshingScope(
            domain.source_id,
            domain.source_revision,
            MeshingEntityKind.GEOMETRY,
            degree,
            domain.entity_set_id(degree),
            np.asarray(domain.scope_indices(degree), dtype=np.int64)[
                np.asarray(indices, dtype=np.int64)
            ],
        )

    patches = []
    for patch in np.asarray(compiled.patches, dtype=np.int64).tolist():
        controls = tuple(
            control
            for control in specification.patch_controls
            if control.scope.entity_dimension == 2
            and patch
            in domain.resolve_indices(
                2, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
            )
        )
        pair = domain.patch_regions[patch]
        source_pair = (
            None if pair[0] < 0 else domain.entity_id(3, int(pair[0])),
            None if pair[1] < 0 else domain.entity_id(3, int(pair[1])),
        )
        for name in tuple(control.name for control in controls) or (f"surface:{patch}",):
            patches.append(
                _owner.MeshPatch(
                    name,
                    scope(2, (patch,)),
                    source_adjacent_region_ids=None
                    if source_pair == (None, None)
                    else source_pair,
                )
            )
    region_controls = tuple(
        control
        for control in specification.region_controls
        if control.scope.entity_dimension == 2
    )
    zones = tuple(
        _owner.MeshZone(
            control.region_name,
            _owner.MeshZoneRole.REGION,
            control.scope,
            material_id=control.material_id,
            region_role=control.role,
        )
        for control in region_controls
    )
    for control in specification.patch_controls:
        if control.scope.entity_dimension != 1:
            continue
        curves = domain.resolve_indices(
            1, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
        )
        adjacent_patches = {
            patch for curve in curves.tolist() for patch, _ in domain.curve_uses(curve)
        }
        adjacent_zones = tuple(
            zone.zone_id
            for region, zone in zip(region_controls, zones, strict=True)
            if adjacent_patches.intersection(
                domain.resolve_indices(
                    2, np.asarray(region.scope.global_entity_ids, dtype=np.int64)
                ).tolist()
            )
        )
        patches.append(
            _owner.MeshPatch(
                control.name,
                control.scope,
                connected=curves.size == 1,
                adjacent_zone_ids=adjacent_zones,
            )
        )
    labels = [
        _owner.MeshLabel(f"curve:{curve}", scope(1, (curve,)))
        for curve in range(len(domain.curves))
        if any(
            patch in np.asarray(compiled.patches).tolist()
            for patch, _ in domain.curve_uses(curve)
        )
    ]
    for region, source_region in enumerate(domain.regions):
        boundary = tuple(patch for patch, _ in source_region.boundary)
        if not set(boundary).intersection(np.asarray(compiled.patches).tolist()):
            continue
        identifier = domain.entity_id(3, region)
        controls = tuple(
            control
            for control in specification.region_controls
            if (
                control.scope.entity_dimension == 3
                and region
                in domain.resolve_indices(
                    3, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
                )
            )
        )
        control = controls[0] if controls else None
        name = (
            f"region:{identifier}"
            if control is None
            else control.region_name
            if control.scope.global_entity_ids.shape[0] == 1
            else f"{control.region_name}:{identifier}"
        )
        labels.append(_owner.MeshLabel(name, scope(2, boundary)))
    return (
        ("patch", tuple(patches)),
        ("zone", zones),
        ("label", tuple(labels)),
        ("attribute", ()),
    )


@checked
def initial_organization_membership(
    compiled: domain_owner.CompiledSurfaceDomain,
    specification: contracts_owner.SurfaceMeshingSpec,
    banks: Mapping[str, Array],
    counts: tuple[int, ...],
    /,
) -> tuple[tuple[str, Array], ...]:
    """Evaluate source scope memberships before collective evidence construction."""
    result = []
    for family, records in _initial_organization_definitions(compiled, specification):
        for index, record in enumerate(records):
            degree = record.scope.entity_dimension
            selected = compiled.domain.resolve_indices(
                degree, np.asarray(record.scope.global_entity_ids, dtype=np.int64)
            )
            source = banks[
                "initial/cell_patches" if degree == 2 else "initial/edge_curves"
            ]
            member = jnp.isin(source, jnp.asarray(selected, dtype=jnp.int64))
            member &= jnp.arange(member.shape[0]) < counts[degree]
            result.append((f"organization/{family}/{index}/membership", member))
    return tuple(result)


@checked
def original_vertex_ancestry(
    state: AdaptiveSimplexState,
    uniform_refinement: bisection_owner.BisectionUniformRefinement | None,
    /,
) -> tuple[Array, Array, Array]:
    """Compose the retained midpoint DAG, without rounding its source coordinates."""
    uniform = (
        None
        if uniform_refinement is None
        else (
            uniform_refinement.child_vertices,
            uniform_refinement.parent_rows,
            uniform_refinement.barycentric_weights,
        )
    )
    return _original_vertex_ancestry_values(
        state.mesh.vertex_ids,
        state.vertex_parents,
        state.cursors[0],
        state.mesh.cells.shape[1],
        uniform,
    )


@eqx.filter_jit
def _original_vertex_ancestry_values(
    vertex_ids: Array,
    vertex_parents: Array,
    cursor: Array,
    width: int,
    uniform: tuple[Array, Array, Array] | None,
    /,
) -> tuple[Array, Array, Array]:
    vertices = vertex_ids.shape[0]
    identifiers = jnp.full((vertices, width), -1, dtype=jnp.int64)
    coefficients = jnp.zeros((vertices, width), dtype=jnp.float64)
    sentinel = jnp.iinfo(jnp.int64).max
    root_ids = jnp.full((vertices, width), -1, dtype=jnp.int64).at[:, 0].set(vertex_ids)
    root_weights = jnp.zeros((vertices, width), dtype=jnp.float64).at[:, 0].set(1.0)
    if uniform is not None:
        child_vertices, parent_rows, barycentric_weights = uniform
        construction_ids = child_vertices.reshape(-1)
        source_ids = jnp.broadcast_to(
            parent_rows[:, None, None, :],
            barycentric_weights.shape,
        ).reshape((-1, width))
        source_weights = barycentric_weights.reshape((-1, width))
        order = jnp.argsort(construction_ids, stable=True)
        keys = construction_ids[order]
        positions = jnp.minimum(jnp.searchsorted(keys, vertex_ids), keys.shape[0] - 1)
        member = keys[positions] == vertex_ids
        selected = order[positions]
        scientific_order = jnp.argsort(source_ids[selected], axis=1, stable=True)
        packed_ids = jnp.take_along_axis(source_ids[selected], scientific_order, axis=1)
        packed_weights = jnp.take_along_axis(
            source_weights[selected], scientific_order, axis=1
        )
        # Keep the source full construction weights, including all three
        # binary64 thirds. Never reconstruct the first weight as 1-x-y-z.
        root_ids = jnp.where(
            member[:, None], jnp.where(packed_weights != 0, packed_ids, -1), root_ids
        )
        root_weights = jnp.where(member[:, None], packed_weights, root_weights)

    def compose(
        slot: Array, carry: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        ids, weights, passed = carry
        parents = vertex_parents[slot]
        root = jnp.all(parents < 0)
        safe = jnp.clip(parents, 0, vertices - 1)
        keys = jnp.concatenate((ids[safe[0]], ids[safe[1]]))
        values = 0.5 * jnp.concatenate((weights[safe[0]], weights[safe[1]]))
        keys = jnp.where(values != 0, keys, sentinel)
        order = jnp.argsort(keys, stable=True)
        keys, values = keys[order], values[order]
        fresh = (keys != sentinel) & jnp.concatenate(
            (jnp.ones((1,), dtype=jnp.bool_), keys[1:] != keys[:-1])
        )
        rows = jnp.cumsum(fresh, dtype=jnp.int32) - 1
        destination = jnp.where(keys != sentinel, rows, width)
        combined_ids = (
            jnp.full((width,), sentinel, dtype=jnp.int64)
            .at[destination]
            .min(keys, mode="drop")
        )
        combined_weights = (
            jnp.zeros((width,), dtype=jnp.float64)
            .at[destination]
            .add(values, mode="drop")
        )
        combined_ids = jnp.where(combined_weights != 0, combined_ids, -1)
        row_ids = jnp.where(root, root_ids[slot], combined_ids)
        row_weights = jnp.where(root, root_weights[slot], combined_weights)
        valid = root | (
            jnp.all(parents >= 0) & jnp.all(parents < slot) & (jnp.sum(fresh) <= width)
        )
        return (
            ids.at[slot].set(row_ids),
            weights.at[slot].set(row_weights),
            passed & valid,
        )

    return jax.lax.fori_loop(
        0,
        cursor.astype(jnp.int32),
        compose,
        (identifiers, coefficients, jnp.asarray(True)),
    )


class _VertexOccurrences(StrictModule):
    sources: Array
    weights: Array
    rows: Array
    arrays: dict[str, Array]


@jax.jit
def _select_vertex_occurrence_values(
    vertex_ids: Array,
    active: Array,
    support_ids: Array,
    support_weights: Array,
    target_ids: Array,
) -> tuple[Array, Array]:
    vertex_ids = vertex_ids.reshape(-1)
    active = active.reshape(-1)
    sentinel = jnp.iinfo(jnp.int64).max
    order = jnp.argsort(jnp.where(active, vertex_ids, sentinel), stable=True)
    sorted_ids = jnp.where(active[order], vertex_ids[order], sentinel)
    positions = jnp.minimum(
        jnp.searchsorted(sorted_ids, target_ids), sorted_ids.shape[0] - 1
    )
    selected = order[positions]
    width = support_ids.shape[-1]
    return (
        support_ids.reshape((-1, width))[selected],
        support_weights.reshape((-1, width))[selected],
    )


@jax.jit
def _bind_vertex_source_rows(
    source_vertex_ids: Array, vertex_sources: Array, count: Array
) -> tuple[Array, Array]:
    source_vertex_order = jnp.argsort(source_vertex_ids)
    source_vertex_ids = source_vertex_ids[source_vertex_order]
    source_positions = jnp.minimum(
        jnp.searchsorted(source_vertex_ids, vertex_sources),
        source_vertex_ids.shape[0] - 1,
    )
    rows = jnp.where(vertex_sources >= 0, source_vertex_order[source_positions], -1)
    passed = jnp.all(
        (jnp.arange(vertex_sources.shape[0])[:, None] >= count)
        | (vertex_sources < 0)
        | (source_vertex_ids[source_positions] == vertex_sources)
    )
    return rows, passed


def _prepare_vertex_occurrences(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    states: AdaptiveSimplexState,
    target_ids: tuple[Array, ...],
    counts: tuple[int, ...],
    uniform_refinement: bisection_owner.BisectionUniformRefinement | None,
    source_counts: tuple[int, ...],
) -> _VertexOccurrences:
    support_ids, support_weights, valid = jax.vmap(
        lambda state: original_vertex_ancestry(state, uniform_refinement),
    )(states)
    if not bool(jax.device_get(jnp.all(valid))):
        raise ValueError(
            "collective_organization.ancestry must be a valid midpoint DAG; got invalid ancestry"
        )
    vertex_sources, vertex_weights = _select_vertex_occurrence_values(
        states.mesh.vertex_ids,
        states.mesh.vertex_active,
        support_ids,
        support_weights,
        target_ids[0],
    )
    placement = target_ids[0].sharding
    vertex_sources = jax.device_put(vertex_sources, placement)
    vertex_weights = jax.device_put(vertex_weights, placement)
    arrays: dict[str, Array] = {
        "organization/vertex/source_ids": vertex_sources,
        "organization/vertex/source_weights": vertex_weights,
    }
    source_vertex_ids = (
        original.entity_ids[0][: source_counts[0]]
        if isinstance(original, initial_certification_owner.InitialCollectiveMeshEvidence)
        else jnp.asarray(original.mesh.vertex_global_ids, dtype=jnp.int64)
    )
    vertex_source_rows, source_binding = _bind_vertex_source_rows(
        source_vertex_ids, vertex_sources, jnp.asarray(counts[0], dtype=jnp.int64)
    )
    if not bool(jax.device_get(source_binding)):
        raise ValueError(
            "collective_organization.vertex_support must belong to the source; got an absent support"
        )
    arrays["organization/vertex/source_rows"] = jax.device_put(
        vertex_source_rows, placement
    )
    return _VertexOccurrences(vertex_sources, vertex_weights, vertex_source_rows, arrays)


class _CellOccurrences(StrictModule):
    rows: Array
    active: Array
    cell_ids: Array
    order: Array


@jax.jit
def _cell_root_ids(cell_ids: Array, parents: Array, cursors: Array, /) -> Array:
    def roots(ids: Array, parent_records: Array, cursor: Array) -> Array:
        slots = jnp.full(ids.shape, -1, dtype=jnp.int32)

        def ancestor(slot: Array, values: Array) -> Array:
            parent = parent_records[slot] // 2
            root = jnp.where(parent < 0, slot, values[jnp.maximum(parent, 0)])
            return values.at[slot].set(root)

        slots = jax.lax.fori_loop(0, cursor.astype(jnp.int32), ancestor, slots)
        return ids[jnp.maximum(slots, 0)]

    return jax.vmap(roots)(cell_ids, parents, cursors).reshape(-1)


@jax.jit
def _entity_carrier_rows(scientific: Array, supports: Array, /) -> Array:
    def carrier(query: Array) -> Array:
        match = jnp.all(
            (query[None, :] < 0)
            | jnp.any(query[None, :, None] == scientific[:, None, :], axis=2),
            axis=1,
        )
        return jnp.min(
            jnp.where(match, jnp.arange(scientific.shape[0]), scientific.shape[0])
        )

    rows = jax.lax.map(carrier, supports)
    return jnp.where(rows < scientific.shape[0], rows, -1)


@jax.jit
def _translate_uniform_cell_roots(
    root_ids: Array, child_ids: Array, parent_ids: Array
) -> Array:
    siblings = child_ids.reshape(-1)
    parents = jnp.broadcast_to(parent_ids[:, None], child_ids.shape).reshape(-1)
    sibling_order = jnp.argsort(siblings, stable=True)
    table = siblings[sibling_order]
    slots = jnp.minimum(jnp.searchsorted(table, root_ids), table.shape[0] - 1)
    return jnp.where(table[slots] == root_ids, parents[sibling_order[slots]], root_ids)


@jax.jit
def _select_cell_occurrence_values(
    root_ids: Array, active: Array, cell_ids: Array, target_ids: Array
) -> tuple[Array, Array, Array, Array]:
    active = active.reshape(-1)
    cell_ids = cell_ids.reshape(-1)
    order = jnp.argsort(
        jnp.where(active, cell_ids, jnp.iinfo(jnp.int64).max), stable=True
    )
    return root_ids[order[: target_ids.shape[0]]], active, cell_ids, order


@jax.jit
def _bind_cell_source_rows(
    source_cells: Array, cell_sources: Array, count: Array
) -> tuple[Array, Array]:
    source_order = jnp.argsort(source_cells)
    source_sorted = source_cells[source_order]
    positions = jnp.minimum(
        jnp.searchsorted(source_sorted, cell_sources), source_sorted.shape[0] - 1
    )
    return source_order[positions], jnp.all(
        (jnp.arange(cell_sources.shape[0]) >= count)
        | (source_sorted[positions] == cell_sources)
    )


@jax.jit
def _bind_entity_support_rows(
    target_vertex_ids: Array,
    vertex_count: Array,
    keys: Array,
    vertex_sources: Array,
    scientific: Array,
) -> Array:
    logical_vertices = jnp.where(
        jnp.arange(target_vertex_ids.shape[0]) < vertex_count,
        target_vertex_ids,
        jnp.iinfo(jnp.int64).max,
    )
    vertex_rows = jnp.minimum(
        jnp.searchsorted(logical_vertices, keys), target_vertex_ids.shape[0] - 1
    )
    # Only the accepted sorted logical prefix participates in occurrence claims.
    supports = vertex_sources[vertex_rows].reshape((keys.shape[0], -1))
    return _entity_carrier_rows(scientific, supports)


def _prepare_cell_occurrences(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    states: AdaptiveSimplexState,
    target_ids: tuple[Array, ...],
    target_keys: tuple[Array, ...],
    counts: tuple[int, ...],
    uniform_refinement: bisection_owner.BisectionUniformRefinement | None,
    source_counts: tuple[int, ...],
    dimension: int,
    vertex_sources: Array,
    arrays: dict[str, Array],
) -> _CellOccurrences:
    root_ids = _cell_root_ids(states.mesh.cell_ids, states.parents, states.cursors[:, 1])
    if uniform_refinement is not None:
        root_ids = _translate_uniform_cell_roots(
            root_ids, uniform_refinement.child_ids, uniform_refinement.parent_ids
        )
    cell_sources, cell_active, cell_ids, order = _select_cell_occurrence_values(
        root_ids, states.mesh.cell_active, states.mesh.cell_ids, target_ids[-1]
    )
    cell_sources = jax.device_put(cell_sources, target_ids[-1].sharding)
    arrays["organization/cell/source_ids"] = cell_sources
    source_cells = (
        original.entity_ids[-1][: source_counts[-1]]
        if isinstance(original, initial_certification_owner.InitialCollectiveMeshEvidence)
        else jnp.asarray(entity_keys(original.mesh, dimension)[:, 0], dtype=jnp.int64)
    )
    cell_rows, source_binding = _bind_cell_source_rows(
        source_cells, cell_sources, jnp.asarray(counts[-1], dtype=jnp.int64)
    )
    if not bool(jax.device_get(source_binding)):
        raise ValueError(
            "collective_organization.cell_sources must identify source cells; got an absent occurrence"
        )
    arrays[f"organization/entity/{dimension}/source_rows"] = jax.device_put(
        cell_rows, target_ids[-1].sharding
    )

    for degree in range(1, dimension):
        keys = target_keys[degree]
        scientific = (
            original.entity_keys[degree][: source_counts[degree]]
            if isinstance(
                original, initial_certification_owner.InitialCollectiveMeshEvidence
            )
            else jnp.asarray(entity_keys(original.mesh, degree), dtype=jnp.int64)
        )

        rows = _bind_entity_support_rows(
            target_ids[0],
            jnp.asarray(counts[0], dtype=jnp.int64),
            keys,
            vertex_sources,
            scientific,
        )
        arrays[f"organization/entity/{degree}/source_rows"] = jax.device_put(
            rows, target_ids[degree].sharding
        )
    return _CellOccurrences(cell_rows, cell_active, cell_ids, order)


class _RegionOccurrences(StrictModule):
    arrays: tuple[tuple[str, Array], ...]


def _prepare_region_occurrences(
    original: result_owner.CellMeshingResult,
    states: AdaptiveSimplexState,
    target_ids: tuple[Array, ...],
    target_keys: tuple[Array, ...],
    counts: tuple[int, ...],
    dimension: int,
    cells: _CellOccurrences,
) -> _RegionOccurrences:
    arrays: dict[str, Array] = {}
    cell_rows, cell_active, cell_ids, order = (
        cells.rows,
        cells.active,
        cells.cell_ids,
        cells.order,
    )
    evidence = original.region_evidence
    if evidence is None:
        raise ValueError(
            "collective_organization.region_evidence must bind source materials; got missing evidence"
        )
    region_lookup = {name: index for index, name in enumerate(evidence.domain.region_ids)}
    scientific_regions = dict(
        zip(evidence.cell_global_ids, evidence.cell_region_ids, strict=True)
    )
    region_values = jnp.asarray(
        [
            region_lookup[scientific_regions[int(identifier)]]
            for identifier in np.asarray(
                original.mesh.entity_set(dimension).entity_ids, dtype=np.int64
            )
        ],
        dtype=jnp.int32,
    )
    regions = region_values[cell_rows]
    regions = jnp.where(jnp.arange(regions.shape[0]) < counts[-1], regions, -1)
    arrays["organization/region/cell_indices"] = jax.device_put(
        regions, target_ids[-1].sharding
    )
    raw_regions = jnp.full(cell_ids.shape, -1, dtype=jnp.int32).at[order].set(regions)
    facet_keys, facet_signs = jax.vmap(facet_packets)(states)
    facet_keys = facet_keys.reshape((-1, dimension))
    facet_signs = facet_signs.reshape(-1)
    table = jnp.where(
        jnp.arange(target_keys[-2].shape[0])[:, None] < counts[-2],
        target_keys[-2],
        jnp.iinfo(jnp.int64).max,
    )
    positions = key_positions(table, facet_keys)
    active_facets = jnp.repeat(cell_active, dimension + 1)
    destinations = jnp.where(active_facets & (positions >= 0), positions, table.shape[0])
    sides = jnp.where(facet_signs > 0, 0, 1)
    neighbors = (
        jnp.full((table.shape[0], 2), -1, dtype=jnp.int32)
        .at[destinations, sides]
        .set(jnp.repeat(raw_regions, dimension + 1), mode="drop")
    )
    interface_indices = jnp.full((table.shape[0],), -1, dtype=jnp.int32)
    interface_orientations = jnp.zeros((table.shape[0],), dtype=jnp.int8)
    for index, (_, first, second, required) in enumerate(evidence.interface_definitions):
        forward = (neighbors[:, 0] == region_lookup[first]) & (
            neighbors[:, 1] == region_lookup[second]
        )
        backward = (neighbors[:, 0] == region_lookup[second]) & (
            neighbors[:, 1] == region_lookup[first]
        )
        selected = forward | backward
        interface_indices = jnp.where(selected, index, interface_indices)
        interface_orientations = jnp.where(
            selected, jnp.where(forward, 1, -1), interface_orientations
        ).astype(jnp.int8)
        if required and not bool(jax.device_get(jnp.any(selected))):
            raise ValueError(
                "collective_organization.interfaces must retain required global facets; got a missing interface"
            )
    cross_region = jnp.all(neighbors >= 0, axis=1) & (neighbors[:, 0] != neighbors[:, 1])
    if not bool(jax.device_get(jnp.all(~cross_region | (interface_indices >= 0)))):
        raise ValueError(
            "collective_organization.interfaces must have source definitions; got an undefined material interface"
        )
    arrays["organization/region/facet_indices"] = jax.device_put(
        interface_indices, target_ids[-2].sharding
    )
    arrays["organization/region/facet_orientations"] = jax.device_put(
        interface_orientations, target_ids[-2].sharding
    )
    arrays["organization/region/facet_cell_regions"] = jax.device_put(
        neighbors, target_ids[-2].sharding
    )
    return _RegionOccurrences(tuple(arrays.items()))


class _DefinitionOccurrence(StrictModule):
    arrays: tuple[tuple[str, Array], ...]
    definition: tuple[str, int, int, str, str] = eqx.field(static=True)


class _AttributeValues(StrictModule):
    values: Array


def _prepare_attribute_values(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    record: organization_owner.MeshAttribute,
    member: Array,
    vertices: _VertexOccurrences,
    entity_arrays: dict[str, Array],
) -> _AttributeValues:
    degree = record.scope.entity_dimension
    vertex_sources, vertex_source_rows = vertices.sources, vertices.rows
    if isinstance(original, initial_certification_owner.InitialCollectiveMeshEvidence):
        raise ValueError(
            "collective_organization.attribute must come from a generated result; got an initial source"
        )
    scientific_ids = jnp.asarray(
        original.mesh.entity_set(degree).entity_ids, dtype=jnp.int64
    )
    scoped_ids = record.scope.global_entity_ids
    scoped_order = jnp.argsort(scoped_ids)
    scoped_rows = jnp.searchsorted(scoped_ids[scoped_order], scientific_ids)
    scoped_rows = jnp.minimum(scoped_rows, scoped_ids.size - 1)
    source_values = record.global_values[scoped_order[scoped_rows]]
    if degree == 0:
        support_values = source_values[jnp.maximum(vertex_source_rows, 0)]
        values = support_values[:, 0]
        agreement = jnp.all(
            (vertex_sources < 0).reshape(
                (*vertex_sources.shape, *((1,) * len(record.component_shape)))
            )
            | (support_values == values[:, None]),
            axis=tuple(range(1, support_values.ndim)),
        )
        if not bool(jax.device_get(jnp.all(~member | agreement))):
            raise ValueError(
                "collective_organization.attribute_values must agree across merged supports; got different marker values"
            )
    else:
        rows = entity_arrays[f"organization/entity/{degree}/source_rows"]
        values = source_values[jnp.maximum(rows, 0)]
    values = jnp.where(
        member.reshape((member.shape[0], *((1,) * len(record.component_shape)))),
        values,
        jnp.zeros((), dtype=values.dtype),
    )
    return _AttributeValues(values)


@jax.jit
def _vertex_membership_values(
    membership: Array, sources: Array, source_rows: Array, count: int
) -> Array:
    source_members = membership[jnp.maximum(source_rows, 0)]
    member = jnp.all((sources < 0) | source_members, axis=1)
    return member & (jnp.arange(member.shape[0]) < count)


@jax.jit
def _entity_membership_values(membership: Array, source_rows: Array, count: int) -> Array:
    member = (source_rows >= 0) & membership[jnp.maximum(source_rows, 0)]
    return member & (jnp.arange(member.shape[0]) < count)


def _prepare_definition_occurrence(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    record: organization_owner.MeshPatch
    | organization_owner.MeshZone
    | organization_owner.MeshLabel
    | organization_owner.MeshAttribute,
    family: str,
    index: int,
    source_memberships: dict[str, Array],
    source_counts: tuple[int, ...],
    counts: tuple[int, ...],
    target_ids: tuple[Array, ...],
    vertices: _VertexOccurrences,
    entity_arrays: dict[str, Array],
) -> _DefinitionOccurrence:
    _owner = organization_owner
    arrays: dict[str, Array] = {}
    vertex_sources, vertex_source_rows = vertices.sources, vertices.rows
    degree = record.scope.entity_dimension
    membership = (
        source_memberships[f"organization/{family}/{index}/membership"][
            : source_counts[degree]
        ]
        if isinstance(original, initial_certification_owner.InitialCollectiveMeshEvidence)
        else resolve_mesh_scope(original.mesh, record.scope).mask
    )
    if degree == 0:
        member = _vertex_membership_values(
            jnp.asarray(membership), vertex_sources, vertex_source_rows, counts[degree]
        )
    else:
        rows = entity_arrays[f"organization/entity/{degree}/source_rows"]
        member = _entity_membership_values(jnp.asarray(membership), rows, counts[degree])
    arrays[f"organization/{family}/{index}/membership"] = jax.device_put(
        member, target_ids[degree].sharding
    )
    if family == "attribute":
        match record:
            case organization_owner.MeshAttribute():
                values = _prepare_attribute_values(
                    original, record, member, vertices, entity_arrays
                ).values
            case _:
                raise TypeError(
                    "collective_organization.attribute must be MeshAttribute; got a different definition kind"
                )
        arrays[f"organization/attribute/{index}/values"] = jax.device_put(
            values, target_ids[degree].sharding
        )

    match record:
        case _owner.MeshPatch():
            identifier = record.patch_id
        case _owner.MeshZone():
            identifier = record.zone_id
        case _owner.MeshLabel():
            identifier = record.label_id
        case _owner.MeshAttribute():
            identifier = record.attribute_id
        case _:
            raise TypeError(
                "collective_organization.definition must be an organization record; got a different record kind"
            )
    return _DefinitionOccurrence(
        tuple(arrays.items()),
        (family, index, degree, identifier, _owner.organization_definition_id(record)),
    )


@checked
def build_collective_organization_witness(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    states: AdaptiveSimplexState,
    target_keys: tuple[Array, ...],
    target_ids: tuple[Array, ...],
    counts: tuple[int, ...],
    /,
    *,
    uniform_refinement: bisection_owner.BisectionUniformRefinement | None,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[str, Array], ...], str]:
    """Build global scientific occurrence banks and their organization fingerprint.

    Source definitions and occurrences remain immutable. Target membership is
    evaluated from retained cell ancestry and dyadic vertex supports. The JAX
    banks can be lowered to any accepted local view.
    """

    initial = isinstance(
        original, initial_certification_owner.InitialCollectiveMeshEvidence
    )
    source_memberships = (
        dict(
            initial_organization_membership(
                original.compiled,
                original.specification,
                dict(original.logical_arrays),
                original.global_entity_counts,
            )
        )
        if initial
        else {}
    )
    source_counts = original.global_entity_counts if initial else ()
    dimension = len(counts) - 1
    vertices = _prepare_vertex_occurrences(
        original, states, target_ids, counts, uniform_refinement, source_counts
    )
    vertex_sources = vertices.sources
    arrays = vertices.arrays
    cells = _prepare_cell_occurrences(
        original,
        states,
        target_ids,
        target_keys,
        counts,
        uniform_refinement,
        source_counts,
        dimension,
        vertex_sources,
        arrays,
    )
    if not initial and original.region_evidence is not None:
        region = _prepare_region_occurrences(
            original, states, target_ids, target_keys, counts, dimension, cells
        )
        arrays.update(region.arrays)

    definitions = []
    groups = (
        initial_organization_definitions(original)
        if initial
        else (
            ("patch", original.patches),
            ("zone", original.zones),
            ("label", original.labels),
            ("attribute", original.attributes),
        )
    )
    for family, records in groups:
        for index, record in enumerate(records):
            occurrence = _prepare_definition_occurrence(
                original,
                record,
                family,
                index,
                source_memberships,
                source_counts,
                counts,
                target_ids,
                vertices,
                arrays,
            )
            arrays.update(occurrence.arrays)
            definitions.append(occurrence.definition)
    definition_digest = np.frombuffer(
        bytes.fromhex(canonical_fingerprint(definitions)), dtype=np.uint8
    ).copy()
    declarations = np.asarray(
        multihost_utils.process_allgather(definition_digest, tiled=False),
        dtype=np.uint8,
    ).reshape((-1, 32))
    if not np.all(declarations == declarations[0]):
        raise ValueError(
            "collective_organization.definitions must agree across processes; got different declarations"
        )
    result = tuple(sorted(arrays.items()))
    shapes = {}
    for name, value in result:
        if name.startswith("organization/vertex/"):
            count = counts[0]
        elif name == "organization/cell/source_ids":
            count = counts[-1]
        elif name.startswith("organization/entity/"):
            count = counts[int(name.split("/")[2])]
        elif name.startswith("organization/region/"):
            count = counts[-1] if name.endswith("/cell_indices") else counts[-2]
        else:
            family, index = name.split("/")[1:3]
            records = dict(groups)[family]
            count = counts[records[int(index)].scope.entity_dimension]
        shapes[name] = (count, *value.shape[1:])
    digest = logical_array_value_collection_digest(dict(result), logical_shapes=shapes)
    return result, canonical_fingerprint(
        {
            "kind": "collective-scientific-organization-occurrences",
            "original_source": original.evidence_id if initial else original.result_id,
            "definitions": definitions,
            "source_associations": original.global_organization_id
            if initial
            else [record.association_id for record in original.associations],
            "source_regions": None
            if initial or original.region_evidence is None
            else original.region_evidence.evidence_id,
            "source_region_boundaries": []
            if initial
            else [record.evidence_id for record in original.region_boundary_evidence],
            "occurrences": digest,
        }
    )
