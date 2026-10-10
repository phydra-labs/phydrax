#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import final, Mapping, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization import (
    CellGeometrySpec,
    CellMesh,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..geometry._compartments import (
    _complete_interface_definitions,
    CompartmentComplex,
)
from ..geometry._mesh_certificates import (
    DomainCoverageCertificate,
    PiecewiseLinearDomain,
)
from ..typing import checked
from ..units import UnitDefinition
from ._scope import (
    _local_logical_lookup,
    _selected_ids,
    _unique_ids,
    MeshingEntityKind,
    MeshingScope,
    MeshScopeProjection,
    resolve_mesh_scope,
)


if TYPE_CHECKING:
    from ._initial_certification import InitialCollectiveMeshEvidence
    from ._publication_lowering import PublicationProjection
    from ._result import CellMeshingResult, CollectiveMeshEvidence


class MeshZoneRole(StrEnum):
    BOUNDARY = "boundary"
    MATERIAL = "material"
    REGION = "region"
    PARTITION = "partition"
    USER = "user"


class RegionRole(StrEnum):
    FLUID = "fluid"
    SOLID = "solid"
    VOID = "void"
    POROUS = "porous"
    USER = "user"


class MeshAttributeRole(StrEnum):
    MARKER = "marker"
    MATERIAL = "material"
    GEOMETRY_CLASSIFICATION = "geometry_classification"
    PARTITION = "partition"
    USER = "user"


class MeshPatch(StrictModule, NonTrainableState):
    """One connected same-dimensional mesh subset."""

    name: str = eqx.field(static=True)
    scope: MeshingScope
    connected: bool = eqx.field(static=True)
    adjacent_zone_ids: tuple[str, ...] = eqx.field(static=True)
    source_adjacent_region_ids: tuple[str | None, str | None] | None = eqx.field(
        static=True
    )
    patch_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        name: str,
        scope: MeshingScope,
        /,
        *,
        connected: bool = True,
        adjacent_zone_ids: tuple[str, ...] = (),
        source_adjacent_region_ids: tuple[str | None, str | None] | None = None,
    ) -> None:
        value = str(name).strip()
        if not value:
            raise ValueError("Mesh patch name must be non-empty.")
        if isinstance(adjacent_zone_ids, str):
            raise TypeError("adjacent_zone_ids must be an iterable of zone IDs.")
        adjacent = tuple(
            sorted(str(identifier).strip() for identifier in adjacent_zone_ids)
        )
        if any(not identifier for identifier in adjacent):
            raise ValueError("Adjacent mesh zone IDs must be non-empty.")
        if len(adjacent) > 2 or len(set(adjacent)) != len(adjacent):
            raise ValueError(
                "Mesh patches may name at most two distinct adjacent zone IDs."
            )
        source_pair = source_adjacent_region_ids
        if source_pair is not None:
            if len(source_pair) != 2 or all(
                identifier is None for identifier in source_pair
            ):
                raise ValueError(
                    "Source region sides must explicitly name at least one region."
                )
            if any(
                identifier is not None
                and (not isinstance(identifier, str) or not identifier.strip())
                for identifier in source_pair
            ):
                raise ValueError(
                    "Source region side IDs must be nonempty identifiers or exterior None."
                )
            if source_pair[0] is not None and source_pair[0] == source_pair[1]:
                raise ValueError(
                    "A source interface cannot have the same region on both sides."
                )
        self.name = value
        self.scope = scope
        self.connected = bool(connected)
        self.adjacent_zone_ids = adjacent
        self.source_adjacent_region_ids = source_pair
        self.patch_id = canonical_fingerprint(
            {
                "kind": "mesh-patch",
                "name": value,
                "scope": scope.scope_id,
                "connected": bool(connected),
                "adjacent_zone_ids": adjacent,
                "source_adjacent_region_ids": source_pair,
            }
        )


class MeshZone(StrictModule, NonTrainableState):
    """Named exclusive semantic assignment on one entity set."""

    name: str = eqx.field(static=True)
    role: MeshZoneRole = eqx.field(static=True)
    scope: MeshingScope
    material_id: str | None = eqx.field(static=True)
    region_role: RegionRole | None = eqx.field(static=True)
    zone_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        name: str,
        role: MeshZoneRole,
        scope: MeshingScope,
        /,
        *,
        material_id: str | None = None,
        region_role: RegionRole | None = None,
    ) -> None:
        value = str(name).strip()
        if not value:
            raise ValueError("Mesh zone name must be non-empty.")
        if not isinstance(role, MeshZoneRole):
            raise TypeError("role must be MeshZoneRole.")
        material = None if material_id is None else str(material_id).strip()
        if material == "":
            raise ValueError("Mesh zone material_id must be non-empty when supplied.")
        if (material is None) != (region_role is None):
            raise ValueError(
                "Mesh zone material_id and region_role must be supplied together."
            )
        if region_role is not None and not isinstance(region_role, RegionRole):
            raise TypeError("region_role must be RegionRole or None.")
        if material is not None and role is not MeshZoneRole.REGION:
            raise ValueError(
                "Mesh zone material_id and region_role are valid only for REGION zones."
            )
        self.name = value
        self.role = role
        self.scope = scope
        self.material_id = material
        self.region_role = region_role
        self.zone_id = canonical_fingerprint(
            {
                "kind": "mesh-zone",
                "name": value,
                "role": role.value,
                "scope": scope.scope_id,
                "material_id": material,
                "region_role": None if region_role is None else region_role.value,
            }
        )


class MeshLabel(StrictModule, NonTrainableState):
    """Named overlapping semantic selection."""

    name: str = eqx.field(static=True)
    scope: MeshingScope
    label_id: str = eqx.field(static=True)

    @checked
    def __init__(self, name: str, scope: MeshingScope, /) -> None:
        value = str(name).strip()
        if not value:
            raise ValueError("Mesh label name must be non-empty.")
        self.name = value
        self.scope = scope
        self.label_id = canonical_fingerprint(
            {"kind": "mesh-label", "name": value, "scope": scope.scope_id}
        )


@final
class MeshAttributeProjection(StrictModule, NonTrainableState):
    """Finite globally scoped values and their exact owner-local numerical receipt."""

    scope_projection: MeshScopeProjection
    global_values: Array
    packet_values: Array
    value_name: str = eqx.field(static=True)
    values_content_id: str = eqx.field(static=True)

    def __init__(self, scope_projection: MeshScopeProjection, /) -> None:
        if not isinstance(scope_projection, MeshScopeProjection):
            raise TypeError(
                "Attribute projection requires an accepted scientific scope projection."
            )
        name = scope_projection.membership_name
        if not name.startswith("organization/attribute/") or not name.endswith(
            "/membership"
        ):
            raise ValueError(
                "Attribute projection requires its exact scientific attribute membership."
            )
        value_name = name.removesuffix("/membership") + "/values"
        publication = scope_projection.publication
        values = dict(publication.source_arrays)[value_name]
        count = scope_projection.global_ids.shape[0]
        ids = publication.entity_ids[scope_projection.dimension][:count]
        if (
            values.ndim == 0
            or values.shape[0]
            != publication.entity_ids[scope_projection.dimension].shape[0]
        ):
            raise ValueError(
                "Attribute projection values differ from their accepted scientific entity axis."
            )
        if values.dtype.kind not in "biuf":
            raise TypeError("Attribute projection values must be real numeric arrays.")
        rows = jnp.argsort(ids, stable=True)
        positions = jnp.searchsorted(ids[rows], scope_projection.members)
        scoped_values = values[rows[positions]]
        if scoped_values.dtype.kind == "f" and not bool(
            jax.device_get(jnp.all(jnp.isfinite(scoped_values)))
        ):
            raise ValueError("Scoped scientific attribute values must be finite.")
        packet_values = dict(publication.projected_arrays)[value_name]
        if (
            packet_values.shape != (*scope_projection.packet_ids.shape, *values.shape[1:])
            or packet_values.dtype != values.dtype
        ):
            raise ValueError(
                "Attribute numerical receipts have an incompatible part-leading component axis."
            )
        packet_rows = jnp.minimum(
            jnp.searchsorted(ids[rows], scope_projection.packet_ids), count - 1
        )
        expected = values[rows[packet_rows]]
        valid = scope_projection.packet_valid.reshape(
            (*scope_projection.packet_valid.shape, *((1,) * (values.ndim - 1)))
        )
        if not bool(jax.device_get(jnp.all(~valid | (packet_values == expected)))):
            raise ValueError(
                "Attribute numerical receipts differ from their actual accepted scientific values."
            )
        content_id = logical_array_value_collection_digest({"values": scoped_values})
        self.scope_projection = scope_projection
        self.global_values = scoped_values
        self.packet_values = packet_values
        self.value_name = value_name
        self.values_content_id = content_id

    def local_values(self, scope: MeshingScope, /) -> Array:
        local = scope._scope_projection
        source = self.scope_projection
        if (
            local is None
            or scope.local_partition_index is None
            or (
                scope.scope_id != source.scope_id
                or scope.global_entity_ids is not source.members
                or local.members is not source.members
                or local.evidence_id != source.evidence_id
                or local.source_id != source.source_id
                or local.source_revision != source.source_revision
                or local.dimension != source.dimension
                or local.publication.topology_id != source.publication.topology_id
                or local.publication.organization_id != source.publication.organization_id
                or local.publication.coordinate_geometry_id
                != source.publication.coordinate_geometry_id
                or local.packet_ids is not source.packet_ids
                or local.packet_owners is not source.packet_owners
                or local.packet_valid is not source.packet_valid
                or local.packet_membership is not source.packet_membership
            )
        ):
            raise ValueError(
                "Attribute receipt does not belong to the exact prepared local scientific scope."
            )
        if (
            dict(self.scope_projection.publication.projected_arrays)[self.value_name]
            is not self.packet_values
        ):
            raise ValueError(
                "Attribute numerical receipt changed after its actual scientific proof."
            )
        packets = dict(
            self.scope_projection.publication.addressable_arrays(
                scope.local_partition_index
            )
        )
        degree = scope.entity_dimension
        valid = np.asarray(packets[f"entity/{degree}/valid"], dtype=np.bool_)
        ids = np.asarray(packets[f"entity/{degree}/ids"], dtype=np.int64)[valid]
        values = np.asarray(packets[self.value_name])[valid]
        order = np.argsort(ids, stable=True)
        ids, values = ids[order], values[order]
        requested = np.asarray(scope.entity_ids, dtype=np.int64)
        rows = np.searchsorted(ids, requested)
        if np.any(rows >= ids.size) or not np.array_equal(ids[rows], requested):
            raise ValueError(
                "Attribute receipt is missing an actual local scoped entity."
            )
        return jnp.asarray(values[rows], dtype=self.global_values.dtype)


def prepare_mesh_attribute_projections(
    scopes: tuple[tuple[str, MeshScopeProjection], ...],
    /,
) -> tuple[tuple[str, MeshAttributeProjection], ...]:
    """Prepare exact shared attribute buffers in the all-owner publication phase."""
    return tuple(
        (projection.value_name, projection)
        for name, scope in scopes
        if name.startswith("organization/attribute/")
        for projection in (MeshAttributeProjection(scope),)
    )


class MeshAttribute(StrictModule, NonTrainableState):
    """Typed numeric data associated with one exact mesh scope."""

    name: str = eqx.field(static=True)
    role: MeshAttributeRole = eqx.field(static=True)
    scope: MeshingScope
    global_values: Array
    values: Array
    unit: UnitDefinition | None = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    attribute_id: str = eqx.field(static=True)
    values_content_id: str = eqx.field(static=True)
    _attribute_projection: MeshAttributeProjection | None

    @checked
    def __init__(
        self,
        name: str,
        role: MeshAttributeRole,
        scope: MeshingScope,
        values: ArrayLike,
        /,
        *,
        unit: UnitDefinition | None = None,
        _projection: MeshAttributeProjection | None = None,
    ) -> None:
        value = str(name).strip()
        if not value:
            raise ValueError("Mesh attribute name must be non-empty.")
        if not isinstance(role, MeshAttributeRole):
            raise TypeError("role must be MeshAttributeRole.")
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != scope.global_entity_ids.shape[0]:
            raise ValueError(
                "Mesh attribute values must match its global scoped entity count."
            )
        if _projection is None:
            if array.dtype.kind not in "biuf" or (
                array.dtype.kind == "f"
                and not bool(jax.device_get(jnp.all(jnp.isfinite(array))))
            ):
                raise ValueError(
                    "Mesh attributes must contain finite real numeric values."
                )
        elif array is not _projection.global_values:
            raise ValueError(
                "Mesh attributes require the exact globally prepared scientific value bank."
            )
        if _projection is not None:
            local_values = _projection.local_values(scope)
        elif scope.entity_ids is scope.global_entity_ids:
            local_values = array
        else:
            selected, valid = _local_logical_lookup(
                scope.global_entity_ids, array, scope.entity_ids
            )
            if not np.all(valid):
                raise ValueError(
                    "Local attribute membership is absent from its global scope."
                )
            local_values = jnp.asarray(selected, dtype=array.dtype)
        content_id = (
            logical_array_value_collection_digest({"values": array})
            if _projection is None
            else _projection.values_content_id
        )
        self.name = value
        self.role = role
        self.scope = scope
        self.global_values = array
        self.values = local_values
        self.unit = unit
        self.component_shape = tuple(array.shape[1:])
        self.values_content_id = content_id
        self._attribute_projection = _projection
        self.attribute_id = canonical_fingerprint(
            {
                "kind": "mesh-attribute",
                "name": value,
                "role": role.value,
                "scope": scope.scope_id,
                "unit": None if unit is None else unit.unit_id,
                "values": content_id,
            }
        )


@final
class RegionBoundaryEvidence(StrictModule, NonTrainableState):
    """Authoritative volume-region identity bound to an overlapping surface selection."""

    source_scope: MeshingScope
    source_region_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    material_id: str | None = eqx.field(static=True)
    role: RegionRole | None = eqx.field(static=True)
    boundary_label: MeshLabel
    patch_sides: tuple[tuple[str, int], ...] = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        source_scope: MeshingScope,
        source_region_id: str,
        control_id: str,
        material_id: str | None,
        role: RegionRole | None,
        boundary_label: MeshLabel,
        patch_sides: tuple[tuple[str, int], ...],
        /,
    ) -> None:
        if not isinstance(mesh, CellMesh) or mesh.topological_dimension not in (2, 3):
            raise TypeError(
                "Region boundary evidence requires a surface mesh or a volume mesh with surface facets."
            )
        if not isinstance(source_scope, MeshingScope) or (
            source_scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or source_scope.entity_dimension != 3
            or source_scope.global_entity_ids.shape != (1,)
        ):
            raise ValueError(
                "Region boundary evidence requires one exact source-region scope."
            )
        if not isinstance(boundary_label, MeshLabel):
            raise TypeError("boundary_label must be MeshLabel.")
        target = boundary_label.scope
        if (
            target.source_id != mesh.mesh_id
            or target.source_revision != mesh.numeric_version
            or target.entity_kind is not MeshingEntityKind.MESH
            or target.entity_dimension != 2
            or target.entity_set_id != mesh.entity_set(2).entity_set_id
        ):
            raise ValueError(
                "Region boundary label must bind this exact surface mesh revision."
            )
        resolve_mesh_scope(mesh, target)
        region = canonical_identifier(source_region_id, "source_region_id")
        control = canonical_identifier(control_id, "control_id")
        material = (
            None
            if material_id is None
            else canonical_identifier(material_id, "material_id")
        )
        if (material is None) != (role is None) or (
            role is not None and not isinstance(role, RegionRole)
        ):
            raise ValueError(
                "Region boundary material and role must be supplied together."
            )
        sides = tuple(sorted(patch_sides))
        if (
            not sides
            or len({patch for patch, _ in sides}) != len(sides)
            or any(not patch or side not in (-1, 1) for patch, side in sides)
        ):
            raise ValueError(
                "Region boundary evidence requires distinct oriented source patches."
            )
        self.source_scope = source_scope
        self.source_region_id, self.control_id = region, control
        self.material_id, self.role = material, role
        self.boundary_label = boundary_label
        self.patch_sides = sides
        self.target_mesh_id, self.target_topology_id = mesh.mesh_id, mesh.topology_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "region-boundary-evidence",
                "source": source_scope.scope_id,
                "region": region,
                "control": control,
                "material": material,
                "role": None if role is None else role.value,
                "label": boundary_label.label_id,
                "patch_sides": sides,
                "mesh": mesh.mesh_id,
                "topology": mesh.topology_id,
            }
        )

    def require_current(
        self,
        mesh: CellMesh,
        labels: tuple[MeshLabel, ...],
        patches: tuple[MeshPatch, ...],
        /,
    ) -> None:
        if (
            mesh.mesh_id != self.target_mesh_id
            or mesh.topology_id != self.target_topology_id
        ):
            raise ValueError("Region boundary evidence is stale for this surface mesh.")
        if self.boundary_label.label_id not in {label.label_id for label in labels}:
            raise ValueError("A source-region boundary label is absent from the result.")
        lookup = {patch.patch_id: patch for patch in patches}
        selected = []
        for identifier, side in self.patch_sides:
            if identifier not in lookup:
                raise ValueError(
                    "A source-region boundary patch is absent from the result."
                )
            patch = lookup[identifier]
            pair = patch.source_adjacent_region_ids
            if pair is None or pair[0 if side > 0 else 1] != self.source_region_id:
                raise ValueError(
                    "A surface patch contradicts authoritative source-region orientation."
                )
            selected.append(patch.scope.global_entity_ids)
        union = _unique_ids(jnp.concatenate(selected))
        expected = self.boundary_label.scope.global_entity_ids
        if union.shape != expected.shape or not bool(
            jax.device_get(jnp.all(union == expected))
        ):
            raise ValueError(
                "A source-region boundary label omits or adds an oriented patch."
            )


@final
class RegionMeshingEvidence(StrictModule, NonTrainableState):
    """Authoritative source-material assignment and exact domain coverage.

    Interface orientation is relative to the canonical mesh facet: ``+1``
    points from the definition's first region to its second. Names are never
    used to recover scientific identities. The retained source domain permits
    independent coverage revalidation after topology or coordinate changes.
    """

    source_revision: str = eqx.field(static=True)
    source_complex_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    cell_region_ids: tuple[str, ...] = eqx.field(static=True)
    logical_cell_global_ids: Array
    logical_cell_region_indices: Array
    logical_facet_global_ids: Array
    logical_facet_interface_indices: Array
    logical_facet_orientations: Array
    region_zone_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    interface_patch_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    interface_definitions: tuple[tuple[str, str, str, bool], ...] = eqx.field(static=True)
    interface_facets: tuple[tuple[str, int, int, str, str], ...] = eqx.field(static=True)
    adjacency_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    domain: PiecewiseLinearDomain
    coverage: DomainCoverageCertificate
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source_revision: str,
        source_complex_id: str,
        cell_region_ids: tuple[str, ...],
        region_zone_ids: tuple[tuple[str, str], ...],
        interface_patch_ids: tuple[tuple[str, str], ...],
        interface_definitions: tuple[tuple[str, str, str, bool], ...],
        interface_facets: tuple[tuple[str, int, int, str, str], ...],
        domain: PiecewiseLinearDomain,
        coverage: DomainCoverageCertificate,
        /,
    ) -> None:
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError("Region evidence requires a canonical mesh and geometry.")
        if not isinstance(domain, PiecewiseLinearDomain):
            raise TypeError("Region evidence requires a declared material domain.")
        if not isinstance(coverage, DomainCoverageCertificate):
            raise TypeError("Region evidence requires a domain coverage certificate.")
        coverage.binding.require(mesh, geometry)
        if coverage.status != "certified" or coverage.domain_id != domain.domain_id:
            raise ValueError("Region evidence requires certified coverage of its domain.")
        revision = canonical_identifier(source_revision, "source_revision")
        complex_id = canonical_identifier(source_complex_id, "source_complex_id")
        ids = tuple(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64).tolist())
        regions = tuple(
            canonical_identifier(value, "cell_region_id") for value in cell_region_ids
        )
        if len(regions) != len(ids) or any(
            region not in domain.region_ids for region in regions
        ):
            raise ValueError(
                "Every local cell requires an authoritative source-region assignment."
            )
        zone_ids = tuple(sorted(region_zone_ids))
        patch_ids = tuple(sorted(interface_patch_ids))
        definitions = tuple(sorted(interface_definitions))
        facets = tuple(sorted(interface_facets))
        if tuple(value[0] for value in zone_ids) != tuple(
            sorted(domain.region_ids)
        ) or len({value[1] for value in zone_ids}) != len(zone_ids):
            raise ValueError("Region evidence requires one exclusive zone per region.")
        if len({value[0] for value in definitions}) != len(definitions):
            raise ValueError("Interface scientific identities must be unique.")
        known = {value[0]: value[1:3] for value in definitions}
        if any(
            first not in domain.region_ids
            or second not in domain.region_ids
            or first == second
            for _, first, second, _ in definitions
        ):
            raise ValueError("Interface definitions require distinct known regions.")
        if len({value[1] for value in facets}) != len(facets) or any(
            name not in known or (first, second) != known[name] or sign not in (-1, 1)
            for name, _, sign, first, second in facets
        ):
            raise ValueError(
                "Interface facet assignments must be exclusive and oriented."
            )
        if mesh.storage is None:
            if set(regions) != set(domain.region_ids):
                raise ValueError("Every source region requires a global cell assignment.")
            logical_cells = jnp.asarray(ids, dtype=jnp.int64)
            region_rows = {name: index for index, name in enumerate(domain.region_ids)}
            logical_regions = jnp.asarray(
                tuple(region_rows[name] for name in regions), dtype=jnp.int32
            )
            logical_facets = jnp.asarray(mesh.entity_set(2).entity_ids, dtype=jnp.int64)
            facet_lookup = {
                int(value): index
                for index, value in enumerate(np.asarray(logical_facets))
            }
            interface_rows = {
                name: index for index, (name, _, _, _) in enumerate(definitions)
            }
            interface_indices = np.full(logical_facets.shape, -1, dtype=np.int32)
            orientations = np.zeros(logical_facets.shape, dtype=np.int32)
            for name, facet, sign, _, _ in facets:
                interface_indices[facet_lookup[facet]] = interface_rows[name]
                orientations[facet_lookup[facet]] = sign
            logical_interfaces = jnp.asarray(interface_indices)
            logical_orientations = jnp.asarray(orientations)
            cell_order, facet_order = (
                jnp.argsort(logical_cells, stable=True),
                jnp.argsort(logical_facets, stable=True),
            )
            logical_cells, logical_regions = (
                logical_cells[cell_order],
                logical_regions[cell_order],
            )
            logical_facets = logical_facets[facet_order]
            logical_interfaces, logical_orientations = (
                logical_interfaces[facet_order],
                logical_orientations[facet_order],
            )
            observed = {name for name, _, _, _, _ in facets}
        else:
            (
                logical_cells,
                logical_regions,
                logical_facets,
                logical_interfaces,
                logical_orientations,
            ) = _logical_region_inventory(mesh, domain, definitions)
            observed = {
                name
                for index, (name, _, _, _) in enumerate(definitions)
                if bool(jax.device_get(jnp.any(logical_interfaces == index)))
            }
            local_regions, valid = _local_logical_lookup(
                logical_cells, logical_regions, jnp.asarray(ids, dtype=jnp.int64)
            )
            if not np.all(valid):
                raise ValueError(
                    "Local material cells are absent from the global inventory."
                )
            if tuple(domain.region_ids[index] for index in local_regions) != regions:
                raise ValueError(
                    "Local material assignments contradict the global scientific inventory."
                )
            expected_facets = _local_interface_records(
                mesh,
                logical_facets,
                logical_interfaces,
                logical_orientations,
                definitions,
            )
            if expected_facets != facets:
                raise ValueError(
                    "Local material facets contradict the global interface inventory."
                )
        if any(required and name not in observed for name, _, _, required in definitions):
            raise ValueError("A required source interface has no mesh facets.")
        if {value[0] for value in patch_ids} != observed:
            raise ValueError(
                "Every observed interface requires a canonical patch binding."
            )
        self.source_revision = revision
        self.source_complex_id = complex_id
        self.target_mesh_id = mesh.mesh_id
        self.target_topology_id = mesh.topology_id
        self.cell_global_ids = ids
        self.cell_region_ids = regions
        self.logical_cell_global_ids = logical_cells
        self.logical_cell_region_indices = logical_regions
        self.logical_facet_global_ids = logical_facets
        self.logical_facet_interface_indices = logical_interfaces
        self.logical_facet_orientations = logical_orientations
        self.region_zone_ids = zone_ids
        self.interface_patch_ids = patch_ids
        self.interface_definitions = definitions
        self.interface_facets = facets
        self.adjacency_pairs = tuple(
            sorted(
                {
                    (min(first, second), max(first, second))
                    for name, first, second, _ in definitions
                    if name in observed
                }
            )
        )
        self.domain = domain
        self.coverage = coverage
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "region-meshing-evidence",
                "source_revision": revision,
                "source_complex": complex_id,
                "mesh": mesh.mesh_id,
                "cell_inventory": logical_array_value_collection_digest(
                    {
                        "ids": logical_cells,
                        "regions": logical_regions,
                    }
                ),
                "zones": zone_ids,
                "patches": patch_ids,
                "definitions": definitions,
                "facet_inventory": logical_array_value_collection_digest(
                    {
                        "ids": logical_facets,
                        "interfaces": logical_interfaces,
                        "orientations": logical_orientations,
                    }
                ),
                "domain": domain.domain_id,
                "coverage": coverage.certificate_id,
            }
        )

    @property
    def coverage_id(self) -> str:
        return self.coverage.certificate_id

    def require_source(self, source: CompartmentComplex, /) -> None:
        if not isinstance(source, CompartmentComplex):
            raise TypeError("source must be CompartmentComplex.")
        if (
            self.source_revision != source.source_revision
            or self.source_complex_id != source.complex_id
            or set(self.domain.region_ids)
            != {value.compartment_id for value in source.compartments}
            or self.adjacency_pairs != source.observed_adjacencies
            or self.interface_definitions != _complete_interface_definitions(source)
        ):
            raise ValueError("Region evidence does not bind this compartment source.")

    def require_current(
        self,
        mesh: CellMesh,
        zones: tuple[MeshZone, ...],
        patches: tuple[MeshPatch, ...],
        /,
        *,
        geometry: CellGeometrySpec | None = None,
    ) -> None:
        if self.target_mesh_id != mesh.mesh_id:
            raise ValueError("Region evidence carries stale mesh coordinates.")
        geometry_ = (
            mesh.storage.restore_geometry()
            if geometry is None and mesh.storage is not None
            else CellGeometrySpec.affine(mesh)
            if geometry is None
            else geometry
        )
        self.coverage.binding.require(mesh, geometry_)
        ids = tuple(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64).tolist())
        if ids != self.cell_global_ids:
            raise ValueError("Region cell global IDs differ from the current mesh.")
        local_regions, valid = _local_logical_lookup(
            self.logical_cell_global_ids,
            self.logical_cell_region_indices,
            jnp.asarray(ids, dtype=jnp.int64),
        )
        if not np.all(valid):
            raise ValueError(
                "Local region cells are absent from the global material inventory."
            )
        if (
            tuple(self.domain.region_ids[index] for index in local_regions)
            != self.cell_region_ids
        ):
            raise ValueError(
                "Local region assignments contradict the global material inventory."
            )
        zone_by_id = {value.zone_id: value for value in zones}
        for region, zone_id in self.region_zone_ids:
            if zone_id not in zone_by_id:
                raise ValueError("A region zone binding is absent from the result.")
            zone = zone_by_id[zone_id]
            expected = _selected_ids(
                self.logical_cell_global_ids,
                self.logical_cell_region_indices == self.domain.region_ids.index(region),
            )
            if (
                zone.role is not MeshZoneRole.REGION
                or zone.scope.entity_dimension != 3
                or zone.scope.source_id != mesh.mesh_id
                or zone.scope.global_entity_ids.shape != expected.shape
                or not bool(
                    jax.device_get(jnp.all(zone.scope.global_entity_ids == expected))
                )
            ):
                raise ValueError(
                    "A canonical zone contradicts its source-region assignment."
                )
        patch_by_id = {value.patch_id: value for value in patches}
        zone_for_region = dict(self.region_zone_ids)
        definitions = {value[0]: value[1:3] for value in self.interface_definitions}
        for interface, patch_id in self.interface_patch_ids:
            if patch_id not in patch_by_id:
                raise ValueError("A material-interface patch binding is absent.")
            patch = patch_by_id[patch_id]
            pair = definitions[interface]
            interface_index = tuple(
                value[0] for value in self.interface_definitions
            ).index(interface)
            expected = _selected_ids(
                self.logical_facet_global_ids,
                self.logical_facet_interface_indices == interface_index,
            )
            if (
                patch.scope.source_id != mesh.mesh_id
                or patch.scope.entity_dimension != 2
                or patch.scope.global_entity_ids.shape != expected.shape
                or not bool(
                    jax.device_get(jnp.all(patch.scope.global_entity_ids == expected))
                )
                or patch.adjacent_zone_ids
                != tuple(sorted(zone_for_region[region] for region in pair))
            ):
                raise ValueError("A material-interface patch contradicts its evidence.")
        if mesh.storage is None:
            expected_facets = material_interface_facets(
                mesh, self.cell_region_ids, self.interface_definitions
            )
            if expected_facets != self.interface_facets:
                raise ValueError("Material facet orientation or adjacency is stale.")
        else:
            arrays = _logical_region_inventory(
                mesh, self.domain, self.interface_definitions
            )
            for actual, expected in zip(
                (
                    self.logical_cell_global_ids,
                    self.logical_cell_region_indices,
                    self.logical_facet_global_ids,
                    self.logical_facet_interface_indices,
                    self.logical_facet_orientations,
                ),
                arrays,
                strict=True,
            ):
                if actual.shape != expected.shape or not bool(
                    jax.device_get(jnp.all(actual == expected))
                ):
                    raise ValueError("Global material incidence or orientation is stale.")
            if (
                _local_interface_records(
                    mesh,
                    self.logical_facet_global_ids,
                    self.logical_facet_interface_indices,
                    self.logical_facet_orientations,
                    self.interface_definitions,
                )
                != self.interface_facets
            ):
                raise ValueError(
                    "Local material facets contradict the global interface inventory."
                )


def _logical_region_inventory(
    mesh: CellMesh,
    domain: PiecewiseLinearDomain,
    definitions: tuple[tuple[str, str, str, bool], ...],
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    storage = mesh.storage
    if storage is None:
        raise ValueError("Global material inventory requires accepted logical storage.")
    arrays = dict(storage.logical_arrays)
    cell_count, facet_count = (
        storage.global_entity_counts[3],
        storage.global_entity_counts[2],
    )
    cells = arrays["cell_global_ids"][:cell_count]
    regions = arrays["organization/region/cell_indices"][:cell_count]
    facets = arrays["entity_global_ids_2"][:facet_count]
    interfaces = arrays["organization/region/facet_indices"][:facet_count]
    orientations = arrays["organization/region/facet_orientations"][:facet_count]
    neighbors = arrays["organization/region/facet_cell_regions"][:facet_count]
    if (
        cells.shape != regions.shape
        or facets.shape != interfaces.shape
        or facets.shape != orientations.shape
        or neighbors.shape != (facet_count, 2)
        or any(
            value.dtype.kind not in "iu"
            for value in (cells, regions, facets, interfaces, orientations, neighbors)
        )
    ):
        raise ValueError(
            "Material logical banks must have exact aligned integer entity axes."
        )
    invalid = (
        jnp.any((regions < 0) | (regions >= len(domain.region_ids)))
        | jnp.any((interfaces < -1) | (interfaces >= len(definitions)))
        | jnp.any(
            jnp.where(interfaces < 0, orientations != 0, jnp.abs(orientations) != 1)
        )
        | jnp.any((neighbors < -1) | (neighbors >= len(domain.region_ids)))
    )
    for index in range(len(domain.region_ids)):
        invalid = invalid | ~jnp.any(regions == index)
    cross_material = jnp.all(neighbors >= 0, axis=1) & (
        neighbors[:, 0] != neighbors[:, 1]
    )
    invalid = invalid | jnp.any(cross_material != (interfaces >= 0))
    for index, (_, first, second, _) in enumerate(definitions):
        first_index, second_index = (
            domain.region_ids.index(first),
            domain.region_ids.index(second),
        )
        selected = interfaces == index
        outward_first = jnp.where(orientations > 0, neighbors[:, 0], neighbors[:, 1])
        outward_second = jnp.where(orientations > 0, neighbors[:, 1], neighbors[:, 0])
        invalid = invalid | jnp.any(
            selected & ((outward_first != first_index) | (outward_second != second_index))
        )
    if bool(jax.device_get(invalid)):
        raise ValueError(
            "Global material coverage, adjacency, or oriented interface inventory is inconsistent."
        )
    cell_order, facet_order = (
        jnp.argsort(cells, stable=True),
        jnp.argsort(facets, stable=True),
    )
    return (
        cells[cell_order],
        regions[cell_order],
        facets[facet_order],
        interfaces[facet_order],
        orientations[facet_order],
    )


def _local_interface_records(
    mesh: CellMesh,
    facets: Array,
    interfaces: Array,
    orientations: Array,
    definitions: tuple[tuple[str, str, str, bool], ...],
    /,
) -> tuple[tuple[str, int, int, str, str], ...]:
    local_facets = mesh.entity_set(2).entity_ids
    local_values, valid = _local_logical_lookup(
        facets,
        jnp.stack((interfaces, orientations.astype(interfaces.dtype)), axis=1),
        local_facets,
    )
    if not np.all(valid):
        raise ValueError(
            "Local material facets are absent from the global interface inventory."
        )
    local_interfaces, local_orientations = local_values[:, 0], local_values[:, 1]
    local_ids = np.asarray(local_facets, dtype=np.int64)
    return tuple(
        sorted(
            (
                definitions[index][0],
                int(facet),
                int(sign),
                definitions[index][1],
                definitions[index][2],
            )
            for facet, index, sign in zip(
                local_ids, local_interfaces, local_orientations, strict=True
            )
            if index >= 0
        )
    )


def lower_region_meshing_evidence(
    original: CellMeshingResult,
    target: CellMesh,
    geometry: CellGeometrySpec,
    zones: tuple[MeshZone, ...],
    patches: tuple[MeshPatch, ...],
    coverage: DomainCoverageCertificate,
    /,
) -> RegionMeshingEvidence:
    """Bind source material definitions to accepted global banks and their local views."""
    evidence = original.region_evidence
    if evidence is None:
        raise ValueError("Material renewal requires original scientific region evidence.")
    cells, regions, facets, interfaces, orientations = _logical_region_inventory(
        target, evidence.domain, evidence.interface_definitions
    )
    local_cells = jnp.asarray(target.entity_set(3).entity_ids, dtype=jnp.int64)
    local_regions, valid = _local_logical_lookup(cells, regions, local_cells)
    if not np.all(valid):
        raise ValueError(
            "Material renewal selects cells absent from the global inventory."
        )
    assignments = tuple(evidence.domain.region_ids[index] for index in local_regions)
    facet_records = _local_interface_records(
        target, facets, interfaces, orientations, evidence.interface_definitions
    )
    if len(original.zones) != len(zones) or len(original.patches) != len(patches):
        raise ValueError(
            "Material renewal requires complete scientific organization definitions."
        )
    zone_ids = {
        before.zone_id: after.zone_id
        for before, after in zip(original.zones, zones, strict=True)
    }
    patch_ids = {
        before.patch_id: after.patch_id
        for before, after in zip(original.patches, patches, strict=True)
    }
    renewed = RegionMeshingEvidence(
        target,
        geometry,
        evidence.source_revision,
        evidence.source_complex_id,
        assignments,
        tuple(
            (region, zone_ids[identifier])
            for region, identifier in evidence.region_zone_ids
        ),
        tuple(
            (interface, patch_ids[identifier])
            for interface, identifier in evidence.interface_patch_ids
        ),
        evidence.interface_definitions,
        facet_records,
        evidence.domain,
        coverage,
    )
    renewed.require_current(target, zones, patches, geometry=geometry)
    return renewed


def material_interface_facets(
    mesh: CellMesh,
    cell_region_ids: tuple[str, ...],
    definitions: tuple[tuple[str, str, str, bool], ...],
    /,
) -> tuple[tuple[str, int, int, str, str], ...]:
    """Recover material adjacency from whole-cell ownership, never a point sample."""
    connectivity = mesh.connectivity
    match connectivity:
        case TetrahedralConnectivity():
            face_values = np.asarray(connectivity.cell_faces, dtype=np.int64).reshape(-1)
            sign_values = np.asarray(connectivity.cell_face_signs, dtype=np.int8).reshape(
                -1
            )
            cell_rows = np.repeat(np.arange(connectivity.cell_count, dtype=np.int64), 4)
            slot_global_ids = np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
            )
            face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
        case PolyhedralConnectivity():
            face_values = np.asarray(connectivity.cell_face_values, dtype=np.int64)
            sign_values = np.asarray(connectivity.cell_face_sign_values, dtype=np.int8)
            cell_rows = np.repeat(
                np.arange(connectivity.cell_count, dtype=np.int64),
                np.diff(np.asarray(connectivity.cell_face_offsets, dtype=np.int64)),
            )
            slot_global_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
            face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
        case _:
            raise TypeError("Material interface evidence requires volume-cell incidence.")
    if len(cell_region_ids) != connectivity.cell_count:
        raise ValueError("Material assignments must cover every volume cell.")
    assignments_by_id = dict(
        zip(
            np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64).tolist(),
            cell_region_ids,
            strict=True,
        )
    )
    if set(slot_global_ids.tolist()) != set(assignments_by_id):
        raise ValueError(
            "Material assignments and volume incidence cell identities differ."
        )
    slot_regions = tuple(assignments_by_id[value] for value in slot_global_ids.tolist())
    owners: list[list[tuple[str, int]]] = [[] for _ in range(face_ids.size)]
    for row, face, sign in zip(
        cell_rows.tolist(), face_values.tolist(), sign_values.tolist(), strict=True
    ):
        owners[face].append((slot_regions[row], sign))
    by_pair = {
        tuple(sorted((first, second))): (name, first, second)
        for name, first, second, _ in definitions
    }
    facets: list[tuple[str, int, int, str, str]] = []
    for face, assigned in enumerate(owners):
        if len(assigned) > 2:
            raise ValueError("A material facet has more than two incident cells.")
        if len(assigned) != 2 or assigned[0][0] == assigned[1][0]:
            continue
        pair = tuple(sorted((assigned[0][0], assigned[1][0])))
        if pair not in by_pair:
            raise ValueError("The mesh contains an undeclared material adjacency.")
        name, first, second = by_pair[pair]
        orientation = assigned[0][1] if assigned[0][0] == first else assigned[1][1]
        facets.append((name, int(face_ids[face]), orientation, first, second))
    return tuple(sorted(facets))


def retained_mesh_organization_projections(
    result: CellMeshingResult,
    /,
) -> tuple[
    tuple[tuple[str, MeshScopeProjection], ...] | None,
    tuple[tuple[str, MeshAttributeProjection], ...] | None,
]:
    """Recover complete actual retained numerical receipts without global recomputation."""
    from ._result import CellMeshingResult

    if not isinstance(result, CellMeshingResult):
        raise TypeError(
            "Retained organization receipts require an actual accepted result."
        )
    storage = result.mesh.storage
    if storage is None:
        return None, None
    evidence = result.collective_evidence
    if evidence is None or evidence.evidence_id != storage.evidence_id:
        raise ValueError(
            "Retained organization receipts lack their actual collective source theorem."
        )
    scopes: list[tuple[str, MeshScopeProjection]] = []
    for family, records in (
        ("patch", result.patches),
        ("zone", result.zones),
        ("label", result.labels),
        ("attribute", result.attributes),
    ):
        for index, record in enumerate(records):
            receipt = record.scope._scope_projection
            name = f"organization/{family}/{index}/membership"
            if receipt is None or (
                receipt.membership_name != name
                or receipt.evidence_id != evidence.evidence_id
                or receipt.source_id != result.mesh.mesh_id
                or receipt.source_revision != result.mesh.numeric_version
                or receipt.scope_id != record.scope.scope_id
                or receipt.members is not record.scope.global_entity_ids
            ):
                raise ValueError(
                    "Organization records lost their exact retained scientific scope receipts."
                )
            receipt.local_mask(result.mesh)
            scopes.append((name, receipt))
    attributes: list[tuple[str, MeshAttributeProjection]] = []
    for index, record in enumerate(result.attributes):
        receipt = record._attribute_projection
        name = f"organization/attribute/{index}/values"
        if receipt is None or (
            receipt.value_name != name
            or receipt.global_values is not record.global_values
            or receipt.values_content_id != record.values_content_id
        ):
            raise ValueError(
                "Scientific attributes lost their complete retained numerical value receipts."
            )
        actual = receipt.local_values(record.scope)
        if (
            actual.shape != record.values.shape
            or actual.dtype != record.values.dtype
            or not np.array_equal(
                np.asarray(actual),
                np.asarray(record.values),
            )
        ):
            raise ValueError(
                "Retained scientific attribute values differ from actual local numerical receipts."
            )
        identity = canonical_fingerprint(
            {
                "kind": "mesh-attribute",
                "name": record.name,
                "role": record.role.value,
                "scope": record.scope.scope_id,
                "unit": None if record.unit is None else record.unit.unit_id,
                "values": record.values_content_id,
            }
        )
        if identity != record.attribute_id:
            raise ValueError(
                "Retained scientific attribute definition lost its complete value-content binding."
            )
        attributes.append((name, receipt))
    return tuple(scopes), tuple(attributes)


def prepare_mesh_organization_scopes(
    original: CellMeshingResult | InitialCollectiveMeshEvidence,
    evidence: CollectiveMeshEvidence | InitialCollectiveMeshEvidence,
    publication: PublicationProjection,
    source_revision: str,
    /,
) -> tuple[tuple[str, MeshScopeProjection], ...]:
    """Prepare all global scientific scope identities before owner-local construction."""
    from ._collective_organization import initial_organization_definitions
    from ._initial_certification import InitialCollectiveMeshEvidence

    groups = (
        initial_organization_definitions(original)
        if isinstance(original, InitialCollectiveMeshEvidence)
        else (
            ("patch", original.patches),
            ("zone", original.zones),
            ("label", original.labels),
            ("attribute", original.attributes),
        )
    )
    prepared = tuple(
        (
            name,
            MeshScopeProjection(
                evidence,
                publication,
                source_revision,
                record.scope.entity_dimension,
                name,
            ),
        )
        for family, records in groups
        for index, record in enumerate(records)
        for name in (f"organization/{family}/{index}/membership",)
    )
    result = []
    for name, receipt in prepared:
        if name.startswith("organization/zone/"):
            exclusive = tuple(
                other
                for other_name, other in prepared
                if other_name.startswith("organization/zone/")
                and other_name != name
                and other.dimension == receipt.dimension
            )
            receipt = MeshScopeProjection(
                evidence,
                publication,
                source_revision,
                receipt.dimension,
                name,
                exclusive_scopes=exclusive,
            )
        result.append((name, receipt))
    return tuple(result)


def lower_mesh_organization(
    original: CellMeshingResult | InitialCollectiveMeshEvidence,
    target: CellMesh,
    logical_arrays: Mapping[str, Array],
    /,
    *,
    scope_projections: tuple[tuple[str, MeshScopeProjection], ...] | None = None,
    attribute_projections: tuple[tuple[str, MeshAttributeProjection], ...] | None = None,
) -> tuple[
    tuple[MeshPatch, ...],
    tuple[MeshZone, ...],
    tuple[MeshLabel, ...],
    tuple[MeshAttribute, ...],
]:
    """Lower complete scientific definitions using certified global membership banks."""
    from ._collective_organization import initial_organization_definitions
    from ._initial_certification import InitialCollectiveMeshEvidence

    if isinstance(original, InitialCollectiveMeshEvidence):
        patch_group, zone_group, label_group, attribute_group = (
            initial_organization_definitions(original)
        )
        patch_records, zone_records = patch_group[1], zone_group[1]
        label_records, attribute_records = label_group[1], attribute_group[1]
    else:
        patch_records, zone_records = original.patches, original.zones
        label_records, attribute_records = original.labels, original.attributes
    if target.storage is None:
        raise ValueError(
            "Collective organization lowering requires certified logical storage."
        )
    storage = target.storage
    projections = {} if scope_projections is None else dict(scope_projections)
    local_bases: dict[int, MeshingScope] = {}

    def scope_for(collection: str, index: int, scope: MeshingScope) -> MeshingScope:
        membership_name = f"organization/{collection}/{index}/membership"
        if scope_projections is not None:
            receipt = projections[membership_name]
            source_banks = dict(receipt.publication.source_arrays)
            if logical_arrays[membership_name] is not source_banks[membership_name]:
                raise ValueError(
                    "Scientific scope lowering consumes a different accepted membership bank."
                )
            return MeshingScope(
                target.mesh_id,
                target.numeric_version,
                MeshingEntityKind.MESH,
                scope.entity_dimension,
                target.entity_set(scope.entity_dimension).entity_set_id,
                receipt.members,
                local_mesh=target,
                _projection=receipt,
            )
        degree = scope.entity_dimension
        count = storage.global_entity_counts[degree]
        name = (
            "vertex_global_ids"
            if degree == 0
            else "cell_global_ids"
            if degree == target.topological_dimension
            else f"entity_global_ids_{degree}"
        )
        ids = logical_arrays[name]
        membership = logical_arrays[f"organization/{collection}/{index}/membership"]
        if membership.shape != ids.shape or membership.dtype != jnp.bool_:
            raise ValueError(
                "Organization membership must exactly match its logical entity table."
            )
        if bool(jax.device_get(jnp.any(membership[count:]))):
            raise ValueError("Organization membership cannot select padded entity rows.")
        members = _selected_ids(ids[:count], membership[:count])
        basis = local_bases.get(degree)
        lowered = MeshingScope(
            target.mesh_id,
            target.numeric_version,
            MeshingEntityKind.MESH,
            degree,
            target.entity_set(degree).entity_set_id,
            members,
            local_mesh=target if basis is None else None,
            _local_basis=basis,
        )
        local_bases.setdefault(degree, lowered)
        return lowered

    zones = tuple(
        MeshZone(
            zone.name,
            zone.role,
            scope_for("zone", index, zone.scope),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        for index, zone in enumerate(zone_records)
    )
    validate_mesh_zones(zones)
    zone_ids = {
        before.zone_id: after.zone_id
        for before, after in zip(zone_records, zones, strict=True)
    }
    patches = tuple(
        MeshPatch(
            patch.name,
            scope_for("patch", index, patch.scope),
            connected=patch.connected,
            adjacent_zone_ids=tuple(
                zone_ids[identifier] for identifier in patch.adjacent_zone_ids
            ),
            source_adjacent_region_ids=patch.source_adjacent_region_ids,
        )
        for index, patch in enumerate(patch_records)
    )
    labels = tuple(
        MeshLabel(label.name, scope_for("label", index, label.scope))
        for index, label in enumerate(label_records)
    )
    attributes = []
    attribute_receipts = (
        {} if attribute_projections is None else dict(attribute_projections)
    )
    for index, attribute in enumerate(attribute_records):
        scope = scope_for("attribute", index, attribute.scope)
        if attribute_projections is not None:
            receipt = attribute_receipts[f"organization/attribute/{index}/values"]
            if (
                logical_arrays[receipt.value_name]
                is not dict(receipt.scope_projection.publication.source_arrays)[
                    receipt.value_name
                ]
            ):
                raise ValueError(
                    "Attribute lowering consumes a different accepted numerical value bank."
                )
            attributes.append(
                MeshAttribute(
                    attribute.name,
                    attribute.role,
                    scope,
                    receipt.global_values,
                    unit=attribute.unit,
                    _projection=receipt,
                )
            )
            continue
        degree = scope.entity_dimension
        name = (
            "vertex_global_ids"
            if degree == 0
            else "cell_global_ids"
            if degree == target.topological_dimension
            else f"entity_global_ids_{degree}"
        )
        ids = logical_arrays[name][: storage.global_entity_counts[degree]]
        rows = jnp.argsort(ids, stable=True)
        ordered_ids = ids[rows]
        positions = jnp.searchsorted(ordered_ids, scope.global_entity_ids)
        values = logical_arrays[f"organization/attribute/{index}/values"]
        if (
            values.shape[0] != logical_arrays[name].shape[0]
            or values.shape[1:] != attribute.component_shape
        ):
            raise ValueError(
                "Organization attribute values must match the logical entity and component axes."
            )
        attributes.append(
            MeshAttribute(
                attribute.name,
                attribute.role,
                scope,
                values[rows[positions]],
                unit=attribute.unit,
            )
        )
    return patches, zones, labels, tuple(attributes)


def _organization_definition_id(
    record: MeshPatch | MeshZone | MeshLabel | MeshAttribute, /
) -> str:
    match record:
        case MeshPatch():
            return canonical_fingerprint(
                {
                    "record_id": record.patch_id,
                    "kind": "patch",
                    "name": record.name,
                    "connected": record.connected,
                    "adjacent_zones": record.adjacent_zone_ids,
                    "source_adjacent_regions": record.source_adjacent_region_ids,
                }
            )
        case MeshZone():
            return canonical_fingerprint(
                {
                    "record_id": record.zone_id,
                    "kind": "zone",
                    "name": record.name,
                    "role": record.role.value,
                    "material": record.material_id,
                    "region_role": None
                    if record.region_role is None
                    else record.region_role.value,
                }
            )
        case MeshLabel():
            return canonical_fingerprint(
                {"kind": "label", "name": record.name, "record_id": record.label_id}
            )
        case MeshAttribute():
            return canonical_fingerprint(
                {
                    "record_id": record.attribute_id,
                    "kind": "attribute",
                    "name": record.name,
                    "role": record.role.value,
                    "unit": None if record.unit is None else record.unit.unit_id,
                    "component_shape": record.component_shape,
                }
            )
        case _:
            raise TypeError(
                "Organization records must have their actual registered class."
            )


def validate_local_mesh_organization(
    original: CellMeshingResult | InitialCollectiveMeshEvidence,
    target: CellMesh,
    logical_arrays: Mapping[str, Array],
    patches: tuple[MeshPatch, ...],
    zones: tuple[MeshZone, ...],
    labels: tuple[MeshLabel, ...],
    attributes: tuple[MeshAttribute, ...],
    /,
    *,
    scope_projections: tuple[tuple[str, MeshScopeProjection], ...] | None = None,
    attribute_projections: tuple[tuple[str, MeshAttributeProjection], ...] | None = None,
) -> None:
    """Require complete definitions and exact lowered membership, including empty views."""
    expected = lower_mesh_organization(
        original,
        target,
        logical_arrays,
        scope_projections=scope_projections,
        attribute_projections=attribute_projections,
    )
    actual = (patches, zones, labels, attributes)
    for provided, required in zip(actual, expected, strict=True):
        if len(provided) != len(required):
            raise ValueError(
                "A collective organization definition is missing or duplicated."
            )
        for record, witness in zip(provided, required, strict=True):
            if type(record) is not type(witness) or _organization_definition_id(
                record
            ) != _organization_definition_id(witness):
                raise ValueError(
                    "A collective organization scientific definition is inconsistent."
                )
            scope, expected_scope = record.scope, witness.scope
            if (
                scope.scope_id != expected_scope.scope_id
                or scope.logical_entity_set_id != expected_scope.logical_entity_set_id
                or (
                    scope.source_id,
                    scope.source_revision,
                    scope.entity_kind,
                    scope.entity_dimension,
                    scope.entity_set_id,
                )
                != (
                    expected_scope.source_id,
                    expected_scope.source_revision,
                    expected_scope.entity_kind,
                    expected_scope.entity_dimension,
                    expected_scope.entity_set_id,
                )
                or scope.local_coverage_id != expected_scope.local_coverage_id
                or scope.local_partition_index != expected_scope.local_partition_index
                or scope.local_partition_count != expected_scope.local_partition_count
            ):
                raise ValueError(
                    "A collective organization scope lacks its exact local coverage binding."
                )
            for values, expected_values in (
                (scope.global_entity_ids, expected_scope.global_entity_ids),
                (scope.entity_ids, expected_scope.entity_ids),
                (scope.entity_owner, expected_scope.entity_owner),
                (scope._local_entity_universe, expected_scope._local_entity_universe),
                (scope._local_owner_universe, expected_scope._local_owner_universe),
                (scope._global_entity_universe, expected_scope._global_entity_universe),
            ):
                if values is None or expected_values is None:
                    raise ValueError(
                        "A collective organization scope lacks its certified inventory universe."
                    )
                if scope_projections is not None and (
                    values is scope.global_entity_ids
                    or values is scope._global_entity_universe
                ):
                    if values is not expected_values:
                        raise ValueError(
                            "Scientific scope receipt lost its exact accepted global member bank."
                        )
                    continue
                if (
                    values.dtype != expected_values.dtype
                    or values.shape != expected_values.shape
                    or not bool(jax.device_get(jnp.all(values == expected_values)))
                ):
                    raise ValueError(
                        "A collective organization scope has a membership or ownership gap."
                    )
            if isinstance(record, MeshAttribute) and isinstance(witness, MeshAttribute):
                for values, expected_values in (
                    (record.global_values, witness.global_values),
                    (record.values, witness.values),
                ):
                    if (
                        attribute_projections is not None
                        and values is record.global_values
                    ):
                        if values is not expected_values:
                            raise ValueError(
                                "Attribute receipt lost its exact accepted global scientific value bank."
                            )
                        continue
                    if (
                        values.shape != expected_values.shape
                        or values.dtype != expected_values.dtype
                        or not bool(jax.device_get(jnp.all(values == expected_values)))
                    ):
                        raise ValueError(
                            "A collective organization attribute has inconsistent values."
                        )


def validate_mesh_zones(zones: tuple[MeshZone, ...], /) -> tuple[MeshZone, ...]:
    if not all(isinstance(zone, MeshZone) for zone in zones):
        raise TypeError("zones must contain MeshZone values.")
    names = tuple(zone.name for zone in zones)
    if len(set(names)) != len(names):
        raise ValueError("Mesh zone names must be unique.")
    for index, zone in enumerate(zones):
        scope = zone.scope
        for previous in zones[:index]:
            other = previous.scope
            if (
                scope.source_id,
                scope.source_revision,
                scope.entity_kind,
                scope.entity_dimension,
                scope.entity_set_id,
            ) != (
                other.source_id,
                other.source_revision,
                other.entity_kind,
                other.entity_dimension,
                other.entity_set_id,
            ):
                continue
            if scope._scope_projection is not None or other._scope_projection is not None:
                projection, previous_projection = (
                    scope._scope_projection,
                    other._scope_projection,
                )
                if (
                    projection is None
                    or previous_projection is None
                    or (
                        projection.publication.topology_id
                        != previous_projection.publication.topology_id
                        or projection.publication.organization_id
                        != previous_projection.publication.organization_id
                        or projection.publication.coordinate_geometry_id
                        != previous_projection.publication.coordinate_geometry_id
                        or projection.publication.partition_count
                        != previous_projection.publication.partition_count
                        or projection.publication.entity_ids[scope.entity_dimension]
                        is not previous_projection.publication.entity_ids[
                            other.entity_dimension
                        ]
                        or projection.evidence_id != previous_projection.evidence_id
                        or not any(
                            proof.scope_id == other.scope_id
                            for proof in projection.exclusive_scopes
                        )
                    )
                ):
                    raise ValueError(
                        "Scientific zone exclusivity lacks its actual globally prepared proof."
                    )
                continue
            rows = jnp.minimum(
                jnp.searchsorted(other.global_entity_ids, scope.global_entity_ids),
                other.global_entity_ids.size - 1,
            )
            if bool(
                jax.device_get(
                    jnp.any(other.global_entity_ids[rows] == scope.global_entity_ids)
                )
            ):
                raise ValueError("Mesh zones on one entity set must be disjoint.")
    return zones


def validate_mesh_labels(labels: tuple[MeshLabel, ...], /) -> tuple[MeshLabel, ...]:
    if not all(isinstance(label, MeshLabel) for label in labels):
        raise TypeError("labels must contain MeshLabel values.")
    names = tuple(label.name for label in labels)
    if len(set(names)) != len(names):
        raise ValueError("Mesh label names must be unique.")
    return labels


__all__ = [
    "MeshAttribute",
    "MeshAttributeRole",
    "MeshLabel",
    "MeshPatch",
    "MeshZone",
    "MeshZoneRole",
    "RegionRole",
    "RegionMeshingEvidence",
    "RegionBoundaryEvidence",
    "material_interface_facets",
    "validate_mesh_labels",
    "validate_mesh_zones",
]
