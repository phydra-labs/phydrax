#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from ..discretization._adaptive_simplex import AdaptiveSimplexState
from ._organization import (
    _organization_definition_id,
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
)
from ._scope import MeshingScope, resolve_mesh_scope
from ._topology_edit import entity_keys


if TYPE_CHECKING:
    from ._bisection import BisectionUniformRefinement
    from ._contracts import SurfaceMeshingSpec
    from ._domain import CompiledSurfaceDomain
    from ._initial_certification import InitialCollectiveMeshEvidence
    from ._result import CellMeshingResult

type OrganizationDefinitions = tuple[
    tuple[str, tuple[MeshPatch, ...]],
    tuple[str, tuple[MeshZone, ...]],
    tuple[str, tuple[MeshLabel, ...]],
    tuple[str, tuple[MeshAttribute, ...]],
]


def initial_organization_definitions(
    evidence: InitialCollectiveMeshEvidence, /
) -> OrganizationDefinitions:
    """Return the immutable source declarations consumed by an accepted theorem."""
    return _initial_organization_definitions(evidence.compiled, evidence.specification)


def _initial_organization_definitions(
    compiled: CompiledSurfaceDomain,
    specification: SurfaceMeshingSpec,
    /,
) -> OrganizationDefinitions:
    """Return exact authored definitions, independent of any owner's local rows."""
    from ._scope import MeshingEntityKind, MeshingScope

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
                MeshPatch(
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
        MeshZone(
            control.region_name,
            MeshZoneRole.REGION,
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
            MeshPatch(
                control.name,
                control.scope,
                connected=curves.size == 1,
                adjacent_zone_ids=adjacent_zones,
            )
        )
    labels = [
        MeshLabel(f"curve:{curve}", scope(1, (curve,)))
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
        labels.append(MeshLabel(name, scope(2, boundary)))
    return (
        ("patch", tuple(patches)),
        ("zone", zones),
        ("label", tuple(labels)),
        ("attribute", ()),
    )


def initial_organization_membership(
    compiled: CompiledSurfaceDomain,
    specification: SurfaceMeshingSpec,
    banks: Mapping[str, Array],
    counts: tuple[int, ...],
    /,
) -> tuple[tuple[str, Array], ...]:
    """Evaluate actual authored source scopes before theorem content is finalized."""
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


def _original_vertex_ancestry(
    state: AdaptiveSimplexState,
    uniform_refinement: BisectionUniformRefinement | None,
    /,
) -> tuple[Array, Array, Array]:
    """Compose the retained midpoint DAG, without rounding its source coordinates."""
    vertices = state.mesh.vertex_ids.shape[0]
    width = state.mesh.cells.shape[1]
    identifiers = jnp.full((vertices, width), -1, dtype=jnp.int64)
    coefficients = jnp.zeros((vertices, width), dtype=jnp.float64)
    sentinel = jnp.iinfo(jnp.int64).max
    root_ids = (
        jnp.full((vertices, width), -1, dtype=jnp.int64)
        .at[:, 0]
        .set(state.mesh.vertex_ids)
    )
    root_weights = jnp.zeros((vertices, width), dtype=jnp.float64).at[:, 0].set(1.0)
    if uniform_refinement is not None:
        construction_ids = uniform_refinement.child_vertices.reshape(-1)
        source_ids = jnp.broadcast_to(
            uniform_refinement.parent_rows[:, None, None, :],
            uniform_refinement.barycentric_weights.shape,
        ).reshape((-1, width))
        source_weights = uniform_refinement.barycentric_weights.reshape((-1, width))
        order = jnp.argsort(construction_ids, stable=True)
        keys = construction_ids[order]
        positions = jnp.minimum(
            jnp.searchsorted(keys, state.mesh.vertex_ids), keys.shape[0] - 1
        )
        member = keys[positions] == state.mesh.vertex_ids
        selected = order[positions]
        scientific_order = jnp.argsort(source_ids[selected], axis=1, stable=True)
        packed_ids = jnp.take_along_axis(source_ids[selected], scientific_order, axis=1)
        packed_weights = jnp.take_along_axis(
            source_weights[selected], scientific_order, axis=1
        )
        # Keep the original full construction weights, including all three
        # binary64 thirds. Never reconstruct the first weight as 1-x-y-z.
        root_ids = jnp.where(
            member[:, None], jnp.where(packed_weights != 0, packed_ids, -1), root_ids
        )
        root_weights = jnp.where(member[:, None], packed_weights, root_weights)

    def compose(
        slot: Array, carry: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        ids, weights, passed = carry
        parents = state.vertex_parents[slot]
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
        state.cursors[0].astype(jnp.int32),
        compose,
        (identifiers, coefficients, jnp.asarray(True)),
    )


def build_collective_organization_witness(
    original: CellMeshingResult | InitialCollectiveMeshEvidence,
    states: AdaptiveSimplexState,
    target_keys: tuple[Array, ...],
    target_ids: tuple[Array, ...],
    counts: tuple[int, ...],
    /,
    *,
    uniform_refinement: BisectionUniformRefinement | None,
) -> tuple[tuple[tuple[str, Array], ...], str]:
    """Globally logical scientific occurrences, not hashes of local record tuples.

    Original definitions and their source occurrences remain immutable. Target
    membership is evaluated from retained cell ancestry and exact dyadic vertex
    supports. The resulting JAX banks can be lowered to any accepted local view.
    """
    from ._initial_certification import InitialCollectiveMeshEvidence

    initial = isinstance(original, InitialCollectiveMeshEvidence)
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
    support_ids, support_weights, valid = jax.vmap(
        lambda state: _original_vertex_ancestry(state, uniform_refinement),
    )(states)
    if not bool(jax.device_get(jnp.all(valid))):
        raise ValueError("Scientific organization ancestry has an invalid midpoint DAG.")
    vertex_ids = states.mesh.vertex_ids.reshape(-1)
    active = states.mesh.vertex_active.reshape(-1)
    order = jnp.argsort(
        jnp.where(active, vertex_ids, jnp.iinfo(jnp.int64).max), stable=True
    )
    sorted_ids = jnp.where(active[order], vertex_ids[order], jnp.iinfo(jnp.int64).max)
    positions = jnp.minimum(
        jnp.searchsorted(sorted_ids, target_ids[0]), sorted_ids.shape[0] - 1
    )
    selected = order[positions]
    width = states.mesh.cells.shape[-1]
    placement = target_ids[0].sharding
    vertex_sources = jax.device_put(support_ids.reshape((-1, width))[selected], placement)
    vertex_weights = jax.device_put(
        support_weights.reshape((-1, width))[selected], placement
    )
    arrays: dict[str, Array] = {
        "organization/vertex/source_ids": vertex_sources,
        "organization/vertex/source_weights": vertex_weights,
    }
    source_vertex_ids = (
        original.entity_ids[0][: source_counts[0]]
        if initial
        else jnp.asarray(original.mesh.vertex_global_ids, dtype=jnp.int64)
    )
    source_vertex_order = jnp.argsort(source_vertex_ids)
    source_vertex_ids = source_vertex_ids[source_vertex_order]
    source_positions = jnp.minimum(
        jnp.searchsorted(source_vertex_ids, vertex_sources),
        source_vertex_ids.shape[0] - 1,
    )
    vertex_source_rows = jnp.where(
        vertex_sources >= 0, source_vertex_order[source_positions], -1
    )
    source_binding = jnp.all(
        (jnp.arange(vertex_sources.shape[0])[:, None] >= counts[0])
        | (vertex_sources < 0)
        | (source_vertex_ids[source_positions] == vertex_sources)
    )
    if not bool(jax.device_get(source_binding)):
        raise ValueError(
            "A target scientific support is absent from the original source."
        )
    arrays["organization/vertex/source_rows"] = jax.device_put(
        vertex_source_rows, placement
    )

    def roots(part: AdaptiveSimplexState) -> Array:
        slots = jnp.full(part.mesh.cell_ids.shape, -1, dtype=jnp.int32)

        def ancestor(slot: Array, values: Array) -> Array:
            parent = part.parents[slot] // 2
            root = jnp.where(parent < 0, slot, values[jnp.maximum(parent, 0)])
            return values.at[slot].set(root)

        slots = jax.lax.fori_loop(0, part.cursors[1].astype(jnp.int32), ancestor, slots)
        return part.mesh.cell_ids[jnp.maximum(slots, 0)]

    root_ids = jax.vmap(roots)(states).reshape(-1)
    if uniform_refinement is not None:
        siblings = uniform_refinement.child_ids.reshape(-1)
        parents = jnp.broadcast_to(
            uniform_refinement.parent_ids[:, None], uniform_refinement.child_ids.shape
        ).reshape(-1)
        sibling_order = jnp.argsort(siblings, stable=True)
        table = siblings[sibling_order]
        slots = jnp.minimum(jnp.searchsorted(table, root_ids), table.shape[0] - 1)
        root_ids = jnp.where(
            table[slots] == root_ids, parents[sibling_order[slots]], root_ids
        )
    cell_active = states.mesh.cell_active.reshape(-1)
    cell_ids = states.mesh.cell_ids.reshape(-1)
    order = jnp.argsort(
        jnp.where(cell_active, cell_ids, jnp.iinfo(jnp.int64).max), stable=True
    )
    cell_sources = jax.device_put(
        root_ids[order[: target_ids[-1].shape[0]]], target_ids[-1].sharding
    )
    arrays["organization/cell/source_ids"] = cell_sources
    source_cells = (
        original.entity_ids[-1][: source_counts[-1]]
        if initial
        else jnp.asarray(entity_keys(original.mesh, dimension)[:, 0], dtype=jnp.int64)
    )
    source_order = jnp.argsort(source_cells)
    source_sorted = source_cells[source_order]
    positions = jnp.minimum(
        jnp.searchsorted(source_sorted, cell_sources), source_sorted.shape[0] - 1
    )
    cell_rows = source_order[positions]
    if not bool(
        jax.device_get(
            jnp.all(
                (jnp.arange(cell_sources.shape[0]) >= counts[-1])
                | (source_sorted[positions] == cell_sources)
            )
        )
    ):
        raise ValueError("A target cell has no original scientific cell occurrence.")
    arrays[f"organization/entity/{dimension}/source_rows"] = jax.device_put(
        cell_rows, target_ids[-1].sharding
    )

    logical_vertices = jnp.where(
        jnp.arange(target_ids[0].shape[0]) < counts[0],
        target_ids[0],
        jnp.iinfo(jnp.int64).max,
    )
    for degree in range(1, dimension):
        keys = target_keys[degree]
        vertex_rows = jnp.minimum(
            jnp.searchsorted(logical_vertices, keys), target_ids[0].shape[0] - 1
        )
        # Target vertex IDs are an accepted, sorted logical prefix. Padding does
        # not participate in scientific occurrence claims.
        supports = vertex_sources[vertex_rows].reshape((keys.shape[0], -1))
        scientific = (
            original.entity_keys[degree][: source_counts[degree]]
            if initial
            else jnp.asarray(entity_keys(original.mesh, degree), dtype=jnp.int64)
        )

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
        rows = jnp.where(rows < scientific.shape[0], rows, -1)
        arrays[f"organization/entity/{degree}/source_rows"] = jax.device_put(
            rows, target_ids[degree].sharding
        )

    if not initial and original.region_evidence is not None:
        from ._device_adaptation import _key_positions
        from ._distribution import _facet_packets

        evidence = original.region_evidence
        region_lookup = {
            name: index for index, name in enumerate(evidence.domain.region_ids)
        }
        scientific_regions = dict(
            zip(evidence.cell_global_ids, evidence.cell_region_ids, strict=True)
        )
        region_values = jnp.asarray(
            [
                region_lookup[scientific_regions[int(identifier)]]
                for identifier in np.asarray(source_cells)
            ],
            dtype=jnp.int32,
        )
        regions = region_values[cell_rows]
        regions = jnp.where(jnp.arange(regions.shape[0]) < counts[-1], regions, -1)
        arrays["organization/region/cell_indices"] = jax.device_put(
            regions, target_ids[-1].sharding
        )
        raw_regions = jnp.full(cell_ids.shape, -1, dtype=jnp.int32).at[order].set(regions)
        facet_keys, facet_signs = jax.vmap(_facet_packets)(states)
        facet_keys = facet_keys.reshape((-1, dimension))
        facet_signs = facet_signs.reshape(-1)
        table = jnp.where(
            jnp.arange(target_keys[-2].shape[0])[:, None] < counts[-2],
            target_keys[-2],
            jnp.iinfo(jnp.int64).max,
        )
        positions = _key_positions(table, facet_keys)
        active_facets = jnp.repeat(cell_active, dimension + 1)
        destinations = jnp.where(
            active_facets & (positions >= 0), positions, table.shape[0]
        )
        sides = jnp.where(facet_signs > 0, 0, 1)
        neighbors = (
            jnp.full((table.shape[0], 2), -1, dtype=jnp.int32)
            .at[destinations, sides]
            .set(jnp.repeat(raw_regions, dimension + 1), mode="drop")
        )
        interface_indices = jnp.full((table.shape[0],), -1, dtype=jnp.int32)
        interface_orientations = jnp.zeros((table.shape[0],), dtype=jnp.int8)
        for index, (_, first, second, required) in enumerate(
            evidence.interface_definitions
        ):
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
                    "A required scientific material interface lost its global facet occurrence."
                )
        cross_region = jnp.all(neighbors >= 0, axis=1) & (
            neighbors[:, 0] != neighbors[:, 1]
        )
        if not bool(jax.device_get(jnp.all(~cross_region | (interface_indices >= 0)))):
            raise ValueError(
                "A target material interface has no original scientific definition."
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
            degree = record.scope.entity_dimension
            membership = (
                source_memberships[f"organization/{family}/{index}/membership"][
                    : source_counts[degree]
                ]
                if initial
                else np.asarray(
                    resolve_mesh_scope(original.mesh, record.scope).mask, dtype=np.bool_
                )
            )
            if degree == 0:
                source_members = jnp.asarray(membership)[
                    jnp.maximum(vertex_source_rows, 0)
                ]
                member = jnp.all((vertex_sources < 0) | source_members, axis=1)
            else:
                rows = arrays[f"organization/entity/{degree}/source_rows"]
                member = (rows >= 0) & jnp.asarray(membership)[jnp.maximum(rows, 0)]
            member &= jnp.arange(member.shape[0]) < counts[degree]
            arrays[f"organization/{family}/{index}/membership"] = jax.device_put(
                member, target_ids[degree].sharding
            )
            if family == "attribute":
                if not isinstance(record, MeshAttribute):
                    raise TypeError(
                        "Scientific attribute membership requires its canonical attribute definition."
                    )
                if isinstance(original, InitialCollectiveMeshEvidence):
                    raise ValueError(
                        "An initial authored source has no generated scientific attribute definition."
                    )
                scientific_ids = np.asarray(
                    original.mesh.entity_set(degree).entity_ids, dtype=np.int64
                )
                scoped_ids = np.asarray(record.scope.global_entity_ids, dtype=np.int64)
                scoped_order = np.argsort(scoped_ids)
                scoped_rows = np.searchsorted(scoped_ids[scoped_order], scientific_ids)
                scoped_rows = np.minimum(scoped_rows, scoped_ids.size - 1)
                source_values = jnp.asarray(record.global_values)[
                    jnp.asarray(scoped_order[scoped_rows])
                ]
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
                            "Scientific marker ancestry merges different attribute values."
                        )
                else:
                    values = source_values[jnp.maximum(rows, 0)]
                values = jnp.where(
                    member.reshape(
                        (member.shape[0], *((1,) * len(record.component_shape)))
                    ),
                    values,
                    jnp.zeros((), dtype=values.dtype),
                )
                arrays[f"organization/attribute/{index}/values"] = jax.device_put(
                    values, target_ids[degree].sharding
                )

            match record:
                case MeshPatch():
                    identifier = record.patch_id
                case MeshZone():
                    identifier = record.zone_id
                case MeshLabel():
                    identifier = record.label_id
                case MeshAttribute():
                    identifier = record.attribute_id
                case _:
                    raise TypeError(
                        "Scientific definitions require their canonical organization class."
                    )
            definitions.append(
                (family, index, degree, identifier, _organization_definition_id(record))
            )
    definition_digest = np.frombuffer(
        bytes.fromhex(canonical_fingerprint(definitions)), dtype=np.uint8
    ).copy()
    declarations = np.asarray(
        multihost_utils.process_allgather(definition_digest, tiled=False),
        dtype=np.uint8,
    ).reshape((-1, 32))
    if not np.all(declarations == declarations[0]):
        raise ValueError(
            "Scientific organization definitions differ between collective owners."
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
