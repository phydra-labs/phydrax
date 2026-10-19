#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import fields

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.meshing._result as result_owner

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..typing import checked, parse
from . import _lineage_records as lineage_records_owner
from ._organization_records import MeshAttribute, MeshLabel, MeshPatch, MeshZone
from ._scope import mesh_global_ids, MeshingEntityKind, MeshingScope, selected_ids


def _parent_witnesses(
    target: result_owner.CellMeshingResult,
    parents: transition_types_owner.NestedReferenceWitnesses,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Parent IDs and reference corners in target concatenated block order."""

    mesh = target.mesh
    ids = np.concatenate(
        [np.asarray(block.global_ids, np.int64) for block in mesh.blocks]
    )
    dimension = mesh.topological_dimension
    fine = np.asarray(parents.fine_cell_ids, dtype=np.int64)
    vertices = np.asarray(parents.fine_reference_vertices, dtype=np.float64)
    if (
        vertices.ndim != 3
        or vertices.shape[0] != fine.size
        or vertices.shape[2] != dimension
    ):
        raise ValueError(
            f"CellMeshTransition.parents must have finite (C, padded corners, d) reference arrays; got {vertices.shape} for {fine.size} cells and dimension {dimension}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError(
            "CellMeshTransition.parents must contain finite reference coordinates; got nonfinite coordinates"
        )
    order = np.argsort(ids, kind="stable")
    position = np.minimum(np.searchsorted(ids[order], fine), ids.size - 1)
    if np.any(ids[order][position] != fine) or np.unique(fine).size != fine.size:
        raise ValueError(
            "CellMeshTransition.parents must name distinct target cells; got duplicate or absent target IDs"
        )
    rows = order[position]
    parent_ids = np.full(ids.shape, -1, dtype=np.int64)
    parent_ids[rows] = np.asarray(parents.coarse_cell_ids, dtype=np.int64)
    reference = np.zeros((ids.size, vertices.shape[1], dimension), dtype=np.float64)
    reference[rows] = vertices
    return parent_ids, reference


def _coarsening_witnesses(
    target: result_owner.CellMeshingResult,
    lineage: lineage_records_owner.MeshLineage,
    witnesses: transition_types_owner.NestedReferenceWitnesses,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate fine-to-coarse reference charts against accepted cell lineage."""

    fine = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    coarse = np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)
    reference = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    dimension = target.mesh.topological_dimension
    if (
        coarse.shape != fine.shape
        or reference.ndim != 3
        or reference.shape[0] != fine.size
        or reference.shape[2] != dimension
        or np.unique(fine).size != fine.size
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError(
            f"CellMeshTransition.coarsening must name distinct fine cells with finite (C, padded corners, d) records; got shapes {fine.shape}/{coarse.shape}/{reference.shape}"
        )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target.mesh.blocks]
    )
    if not np.all(np.isin(coarse, target_ids)):
        raise ValueError(
            "CellMeshTransition.coarsening must name accepted target cells; got absent coarse IDs"
        )
    cells = lineage.entity_lineage(dimension)
    relations = {
        (int(source), int(successor))
        for source, successor, kind in zip(
            np.asarray(cells.source_global_ids, dtype=np.int64),
            np.asarray(cells.target_global_ids, dtype=np.int64),
            np.asarray(cells.relation_kinds, dtype=np.int32),
            strict=True,
        )
        if kind
        in (
            int(lineage_records_owner.EntityLineageKind.PRESERVED),
            int(lineage_records_owner.EntityLineageKind.COARSENED_INTO),
        )
    }
    if any(
        (int(source), int(successor)) not in relations
        for source, successor in zip(fine, coarse, strict=True)
    ):
        raise ValueError(
            "CellMeshTransition.coarsening must bind preserved or coarsened cell lineage; got an unbound fine/coarse pair"
        )
    return fine, coarse, reference


class CellMeshTransition(StrictModule, NonTrainableState):
    """One accepted topology transition of a certified mesh.

    ``geometry_transition`` records how the target coordinate map was built from
    a non-affine source map (absent when both maps are affine carriers).
    ``parent_cell_ids``/``parent_reference_vertices`` are the nested witnesses in
    target concatenated block order: the source cell containing each target cell
    (``-1`` for a target assembled from several source cells) and the target
    corners in that parent's reference coordinates. They are absent for
    non-nested transitions.
    The ``coarsened_*`` fields retain the ordered fine-to-coarse reference bank,
    including its full padded reference arrays. ``from_stored_fields`` rebuilds
    both banks through the reference-correspondence constructor and authenticates
    every stored field against the retained target, lineage, and geometry transition.
    """

    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target: result_owner.CellMeshingResult
    lineage: lineage_records_owner.MeshLineage
    vertex_stencil: lineage_records_owner.VertexInterpolationStencil | None
    geometry_transition: transition_types_owner.CellGeometryTransition | None
    parent_cell_ids: Array | None
    parent_reference_vertices: Array | None
    coarsened_cell_ids: Array | None
    coarsened_into_ids: Array | None
    coarsened_reference_vertices: Array | None
    transition_kind: lineage_records_owner.MeshTransitionKind = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_mesh_id: str,
        source_topology_id: str,
        target: result_owner.CellMeshingResult,
        lineage: lineage_records_owner.MeshLineage,
        transition_kind: lineage_records_owner.MeshTransitionKind,
        /,
        *,
        vertex_stencil: lineage_records_owner.VertexInterpolationStencil | None = None,
        geometry_transition: transition_types_owner.CellGeometryTransition | None = None,
        parents: transition_types_owner.NestedReferenceWitnesses | None = None,
        coarsening: transition_types_owner.NestedReferenceWitnesses | None = None,
    ) -> None:
        mesh_id = str(source_mesh_id).strip()
        topology_id = str(source_topology_id).strip()
        if not mesh_id or not topology_id:
            raise ValueError(
                f"CellMeshTransition.source_ids must be nonempty; got {mesh_id!r}/{topology_id!r}"
            )
        if (
            lineage.source_topology_id != topology_id
            or lineage.target_topology_id != target.mesh.topology_id
        ):
            raise ValueError(
                f"CellMeshTransition.lineage must match source and target topology; got {lineage.source_topology_id!r}/{lineage.target_topology_id!r}"
            )
        parse(
            transition_kind,
            lineage_records_owner.MeshTransitionKind,
            "CellMeshTransition.transition_kind",
        )
        if geometry_transition is not None:
            if (
                geometry_transition.source_topology_id != topology_id
                or geometry_transition.target_topology_id != target.mesh.topology_id
                or geometry_transition.target_geometry_id
                != cell_geometry_id(target.geometry)
            ):
                raise ValueError(
                    f"CellMeshTransition.geometry_transition must match source topology and target map; got target geometry ID {geometry_transition.target_geometry_id!r}"
                )
        witnesses = None if parents is None else _parent_witnesses(target, parents)
        coarsened = (
            None
            if coarsening is None or np.asarray(coarsening.fine_cell_ids).size == 0
            else _coarsening_witnesses(target, lineage, coarsening)
        )
        self.source_mesh_id = mesh_id
        self.source_topology_id = topology_id
        self.target = target
        self.lineage = lineage
        self.vertex_stencil = vertex_stencil
        self.geometry_transition = geometry_transition
        self.parent_cell_ids = (
            None if witnesses is None else jax.device_put(np.asarray(witnesses[0]))
        )
        self.parent_reference_vertices = (
            None if witnesses is None else jax.device_put(np.asarray(witnesses[1]))
        )
        self.coarsened_cell_ids = (
            None if coarsened is None else jax.device_put(np.asarray(coarsened[0]))
        )
        self.coarsened_into_ids = (
            None if coarsened is None else jax.device_put(np.asarray(coarsened[1]))
        )
        self.coarsened_reference_vertices = (
            None if coarsened is None else jax.device_put(np.asarray(coarsened[2]))
        )
        self.transition_kind = transition_kind
        self.transition_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-transition",
                "source_mesh": mesh_id,
                "source_topology": topology_id,
                "target": target.result_id,
                "lineage": lineage.lineage_id,
                "vertex_stencil": None
                if vertex_stencil is None
                else vertex_stencil.stencil_id,
                "geometry_transition": None
                if geometry_transition is None
                else geometry_transition.transition_id,
                "parents": None
                if witnesses is None
                else [array_tree_fingerprint(value) for value in witnesses],
                "coarsening": None
                if coarsened is None
                else [array_tree_fingerprint(value) for value in coarsened],
                "transition_kind": transition_kind.value,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[CellMeshTransition], value: CellMeshTransition, /
    ) -> CellMeshTransition:
        """Rebuild retained reference banks against their target and lineage owners."""
        if cls is not CellMeshTransition or type(value) is not CellMeshTransition:
            raise TypeError(
                "CellMeshTransition.restore_type must be the registered CellMeshTransition; "
                f"got {type(value).__qualname__}"
            )
        parents = None
        if value.parent_cell_ids is None:
            if value.parent_reference_vertices is not None:
                raise ValueError(
                    "CellMeshTransition.parents must retain both IDs and reference vertices; got reference vertices without parent IDs"
                )
        else:
            if value.parent_reference_vertices is None:
                raise ValueError(
                    "CellMeshTransition.parents must retain both IDs and reference vertices; got parent IDs without reference vertices"
                )
            target_ids = np.concatenate(
                [
                    np.asarray(block.global_ids, dtype=np.int64)
                    for block in value.target.mesh.blocks
                ]
            )
            parent_ids = np.asarray(value.parent_cell_ids, dtype=np.int64)
            reference = np.asarray(value.parent_reference_vertices, dtype=np.float64)
            if (
                parent_ids.shape != target_ids.shape
                or reference.ndim != 3
                or reference.shape[0] != target_ids.size
            ):
                raise ValueError(
                    "CellMeshTransition.parents must align with target concatenated block order; got incompatible parent-bank shapes"
                )
            # -1 rows belong to assembled target cells, not nested witnesses.
            nested = parent_ids >= 0
            parents = transition_types_owner.NestedReferenceWitnesses(
                target_ids[nested], parent_ids[nested], reference[nested]
            )
        coarsening = None
        if value.coarsened_cell_ids is None:
            if (
                value.coarsened_into_ids is not None
                or value.coarsened_reference_vertices is not None
            ):
                raise ValueError(
                    "CellMeshTransition.coarsening must retain fine IDs, coarse IDs, and reference vertices; got coarse IDs or reference vertices without fine IDs"
                )
        else:
            if (
                value.coarsened_into_ids is None
                or value.coarsened_reference_vertices is None
            ):
                raise ValueError(
                    "CellMeshTransition.coarsening must retain fine IDs, coarse IDs, and reference vertices; got missing coarse IDs or reference vertices"
                )
            coarsening = transition_types_owner.NestedReferenceWitnesses(
                np.asarray(value.coarsened_cell_ids, dtype=np.int64),
                np.asarray(value.coarsened_into_ids, dtype=np.int64),
                np.asarray(value.coarsened_reference_vertices, dtype=np.float64),
            )
        rebuilt = cls(
            value.source_mesh_id,
            value.source_topology_id,
            value.target,
            value.lineage,
            value.transition_kind,
            vertex_stencil=value.vertex_stencil,
            geometry_transition=value.geometry_transition,
            parents=parents,
            coarsening=coarsening,
        )
        for member in fields(value):
            if not tree_content_equal(
                getattr(rebuilt, member.name), getattr(value, member.name)
            ):
                raise ValueError(
                    f"stored_fields.CellMeshTransition.{member.name} must match reconstructed retained content; got changed {member.name}"
                )
        return rebuilt


@checked
def identity_lineage(
    source: cell_mesh_owner.CellMesh, target: cell_mesh_owner.CellMesh, /
) -> lineage_records_owner.MeshLineage:
    """PRESERVED lineage of every entity of a same-topology coordinate change."""

    if source.topology_id != target.topology_id:
        raise ValueError(
            f"MeshLineage.identity must use a shared topology; got {source.topology_id!r}/{target.topology_id!r}"
        )
    records = []
    for dimension in range(source.topological_dimension + 1):
        ids = mesh_global_ids(source, dimension)
        records.append(
            lineage_records_owner.EntityLineage(
                dimension,
                source.entity_set(dimension).entity_set_id,
                target.entity_set(dimension).entity_set_id,
                ids,
                ids,
                jnp.full(
                    ids.shape,
                    int(lineage_records_owner.EntityLineageKind.PRESERVED),
                    dtype=jnp.int32,
                ),
                created_target_ids=jnp.zeros((0,), dtype=jnp.int64),
                deleted_source_ids=jnp.zeros((0,), dtype=jnp.int64),
            )
        )
    return lineage_records_owner.MeshLineage(
        source.topology_id, target.topology_id, tuple(records)
    )


# Relations through which organization membership (patches/zones/labels) is inherited.
_INHERITING_KINDS = np.asarray(
    [
        lineage_records_owner.EntityLineageKind.PRESERVED,
        lineage_records_owner.EntityLineageKind.REFINED_FROM,
        lineage_records_owner.EntityLineageKind.COARSENED_INTO,
        lineage_records_owner.EntityLineageKind.MERGED_INTO,
        lineage_records_owner.EntityLineageKind.COLLAPSED_INTO,
        lineage_records_owner.EntityLineageKind.SWAPPED_FROM,
        lineage_records_owner.EntityLineageKind.RELOCATED,
    ],
    dtype=np.int32,
)


def _inherited_ids(
    scope: MeshingScope, record: lineage_records_owner.EntityLineage, /
) -> Array:
    """Target IDs inheriting one organization scope through one lineage record.

    COLLAPSED_INTO sources are non-dominant: a target with any other inheriting
    relation takes its membership from those; otherwise the collapsed sources
    decide. Mixed membership among deciding sources is a route error.
    """

    arrays = (
        record.relation_kinds,
        record.source_global_ids,
        record.target_global_ids,
        scope.global_entity_ids,
    )
    if all(
        not isinstance(value, Tracer)
        and value.is_fully_addressable
        and isinstance(value.sharding, jax.sharding.SingleDeviceSharding)
        for value in arrays
    ):
        # Organization is immutable host preparation. Retain the collective
        # array route below when any scientific inventory is not addressable.
        kinds_host, sources_host, targets_host, members_host = (
            np.asarray(value) for value in jax.device_get(arrays)
        )
        inheriting_host = np.isin(kinds_host, _INHERITING_KINDS)
        targets_host = targets_host[inheriting_host]
        if targets_host.size == 0:
            return jnp.zeros((0,), dtype=jnp.int64)
        order_host = np.argsort(targets_host, kind="stable")
        targets_host = targets_host[order_host]
        members = np.isin(sources_host[inheriting_host][order_host], members_host)
        collapsed = (
            kinds_host[inheriting_host][order_host]
            == lineage_records_owner.EntityLineageKind.COLLAPSED_INTO
        )
        unique, inverse = np.unique(targets_host, return_inverse=True)
        dominant = np.zeros(unique.shape, dtype=np.int64)
        np.add.at(dominant, inverse, (~collapsed).astype(np.int64))
        deciding = ~collapsed | (dominant[inverse] == 0)
        total = np.zeros(unique.shape, dtype=np.int64)
        member = np.zeros(unique.shape, dtype=np.int64)
        np.add.at(total, inverse, deciding.astype(np.int64))
        np.add.at(member, inverse, (deciding & members).astype(np.int64))
        if np.any((member > 0) & (member < total)):
            raise ValueError(
                f"MeshLineage.organization_membership must agree across deciding source entities; got mixed membership for dimension {record.dimension}"
            )
        return jnp.asarray(unique[(member > 0) & (member == total)], dtype=jnp.int64)

    kinds = record.relation_kinds
    inheriting = jnp.isin(kinds, jnp.asarray(_INHERITING_KINDS))
    targets = selected_ids(record.target_global_ids, inheriting)
    sources = selected_ids(record.source_global_ids, inheriting)
    selected_kinds = selected_ids(kinds, inheriting)
    if targets.size == 0:
        return jnp.zeros((0,), dtype=jnp.int64)
    order = jnp.argsort(targets, stable=True)
    targets = targets[order]
    members = jnp.isin(sources[order], scope.global_entity_ids)
    collapsed = (
        selected_kinds[order] == lineage_records_owner.EntityLineageKind.COLLAPSED_INTO
    )
    fresh = jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), targets[1:] != targets[:-1])
    )
    unique = selected_ids(targets, fresh)
    inverse = jnp.cumsum(fresh, dtype=jnp.int32) - 1
    dominant = (
        jnp.zeros(unique.shape, dtype=jnp.int64)
        .at[inverse]
        .add((~collapsed).astype(jnp.int64))
        > 0
    )
    deciding = ~collapsed | ~dominant[inverse]
    total = (
        jnp.zeros(unique.shape, dtype=jnp.int64)
        .at[inverse]
        .add(deciding.astype(jnp.int64))
    )
    member = (
        jnp.zeros(unique.shape, dtype=jnp.int64)
        .at[inverse]
        .add((deciding & members).astype(jnp.int64))
    )
    if bool(jax.device_get(jnp.any((member > 0) & (member < total)))):
        raise ValueError(
            f"MeshLineage.organization_membership must agree across deciding source entities; got mixed membership for dimension {record.dimension}"
        )
    return selected_ids(unique, (member > 0) & (member == total))


@checked
def inherit_scope(
    scope: MeshingScope,
    lineage: lineage_records_owner.MeshLineage,
    target: cell_mesh_owner.CellMesh,
    name: str,
    /,
) -> MeshingScope:
    """Rebind one mesh scope onto ``target`` through ``lineage``; ambiguity refuses."""

    dimension = scope.entity_dimension
    record = lineage.entity_lineage(dimension)
    if (
        scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_set_id != record.source_entity_set_id
        or record.target_entity_set_id != target.entity_set(dimension).entity_set_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            f"MeshLineage.organization_scope must match source and target entity bindings; got {scope.entity_set_id!r} for dimension {dimension}"
        )
    identifiers = _inherited_ids(scope, record)
    if identifiers.size == 0:
        raise ValueError(
            f"MeshLineage.organization_scope must retain at least one entity; got empty membership for {name!r}"
        )
    return MeshingScope(
        target.mesh_id,
        target.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        target.entity_set(dimension).entity_set_id,
        identifiers,
        local_mesh=target,
    )


@checked
def inherit_mesh_attributes(
    source: result_owner.CellMeshingResult,
    target: cell_mesh_owner.CellMesh,
    lineage: lineage_records_owner.MeshLineage,
    /,
    *,
    excluded_names: tuple[str, ...] = (),
) -> tuple[MeshAttribute, ...]:
    """Carry entity data through the same deciding-source organization lineage.

    Refinement copies parent values. Merged deciding sources must agree exactly;
    a numeric dtype alone does not authorize interpolation or averaging. Physical
    finite-element coefficients use their separately declared field transfer.
    Distributed publication consumes its globally prepared attribute banks.
    """
    if source.mesh.storage is not None or target.storage is not None:
        raise ValueError(
            "MeshLineage.attributes must use global publication for distributed carriers; got an owner-local carrier"
        )
    if (
        lineage.source_topology_id != source.mesh.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            f"MeshLineage.attributes must match source and target topology lineage; got {lineage.source_topology_id!r}/{lineage.target_topology_id!r}"
        )
    result = []
    for attribute in source.attributes:
        if attribute.name in excluded_names:
            continue
        scope = inherit_scope(attribute.scope, lineage, target, attribute.name)
        record = lineage.entity_lineage(scope.entity_dimension)
        source_ids = np.asarray(attribute.scope.global_entity_ids, dtype=np.int64)
        values = np.asarray(attribute.global_values)
        route_sources = np.asarray(record.source_global_ids, dtype=np.int64)
        route_targets = np.asarray(record.target_global_ids, dtype=np.int64)
        kinds = np.asarray(record.relation_kinds, dtype=np.int32)
        inheriting = np.isin(kinds, _INHERITING_KINDS)
        target_values = np.empty(
            (scope.global_entity_ids.shape[0], *attribute.component_shape),
            dtype=values.dtype,
        )
        for row, identifier in enumerate(
            np.asarray(scope.global_entity_ids, dtype=np.int64)
        ):
            routes = inheriting & (route_targets == identifier)
            dominant = routes & (
                kinds != lineage_records_owner.EntityLineageKind.COLLAPSED_INTO
            )
            deciding = dominant if np.any(dominant) else routes
            selected = route_sources[deciding]
            positions = np.searchsorted(source_ids, selected)
            if (
                selected.size == 0
                or np.any(positions >= source_ids.size)
                or not np.array_equal(
                    source_ids[positions],
                    selected,
                )
            ):
                raise ValueError(
                    f"MeshLineage.attribute_sources must include deciding entities; got absent source IDs for {attribute.name!r} target {identifier}"
                )
            candidates = values[positions]
            if not np.all(candidates == candidates[0]):
                raise ValueError(
                    f"MeshLineage.attribute_values must agree across deciding source entities; got differing values for {attribute.name!r} target {identifier}"
                )
            target_values[row] = candidates[0]
        result.append(
            MeshAttribute(
                attribute.name,
                attribute.role,
                scope,
                target_values,
                unit=attribute.unit,
            )
        )
    return tuple(result)


@checked
def inherit_mesh_organization(
    source: result_owner.CellMeshingResult,
    target: cell_mesh_owner.CellMesh,
    lineage: lineage_records_owner.MeshLineage,
    /,
) -> tuple[tuple[MeshPatch, ...], tuple[MeshZone, ...], tuple[MeshLabel, ...]]:
    """Rebind every patch, zone, and label of ``source`` onto ``target``.

    Membership follows the lineage relations; region identity (material and
    region role), zone adjacency, and oriented source-region sides are kept.
    Ambiguous membership refuses the whole remap, so no partial region evidence
    is published.
    """

    if lineage.target_topology_id != target.topology_id or (
        lineage.source_topology_id != source.mesh.topology_id
    ):
        raise ValueError(
            f"MeshLineage.organization must join source and target topologies; got {lineage.source_topology_id!r}/{lineage.target_topology_id!r}"
        )
    for record in (*source.patches, *source.zones, *source.labels):
        scope = record.scope
        if (
            scope.source_id != source.mesh.mesh_id
            or scope.source_revision != source.mesh.numeric_version
            or scope.entity_kind is not MeshingEntityKind.MESH
        ):
            raise ValueError(
                f"MeshLineage.organization_scope must match the source mesh revision; got {scope.source_id!r}/{scope.source_revision!r}"
            )
    zone_ids = {}
    zones = []
    for zone in source.zones:
        inherited = MeshZone(
            zone.name,
            zone.role,
            inherit_scope(zone.scope, lineage, target, zone.name),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        zone_ids[zone.zone_id] = inherited.zone_id
        zones.append(inherited)
    patches = tuple(
        MeshPatch(
            patch.name,
            inherit_scope(patch.scope, lineage, target, patch.name),
            connected=patch.connected,
            adjacent_zone_ids=tuple(zone_ids[value] for value in patch.adjacent_zone_ids),
            source_adjacent_region_ids=patch.source_adjacent_region_ids,
        )
        for patch in source.patches
    )
    labels = tuple(
        MeshLabel(label.name, inherit_scope(label.scope, lineage, target, label.name))
        for label in source.labels
    )
    return patches, tuple(zones), labels


__all__ = [
    "CellMeshTransition",
    "identity_lineage",
    "inherit_mesh_organization",
    "inherit_mesh_attributes",
]
