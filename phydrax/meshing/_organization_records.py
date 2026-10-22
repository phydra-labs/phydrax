#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum
from typing import final, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_complex as cell_complex_owner
import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..geometry._compartments import CompartmentComplex, complete_interface_definitions
from ..geometry._domain_coverage import DomainCoverageCertificate, PiecewiseLinearDomain
from ..geometry._embedding_state import MeshCertificateLimits
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..typing import checked
from ..units import UnitDefinition
from ._collective_failure import agreed_local_step
# Module references defer symbol resolution across mutually dependent execution owners.
from ._scope import (
    local_logical_lookup,
    MeshingEntityKind,
    MeshingScope,
    MeshingScopeBank,
    MeshScopeProjection,
    overlapping_ids,
    resolve_mesh_scope,
    selected_ids,
    unique_ids,
)

type RegionSourceDomain = PiecewiseLinearDomain | MappedReferenceDomain



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
            raise ValueError("MeshPatch.name must be non-empty; got an empty name")
        if isinstance(adjacent_zone_ids, str):
            raise TypeError(
                "MeshPatch.adjacent_zone_ids must be an iterable of zone IDs; got a string"
            )
        adjacent = tuple(
            sorted(str(identifier).strip() for identifier in adjacent_zone_ids)
        )
        if any(not identifier for identifier in adjacent):
            raise ValueError(
                "MeshPatch.adjacent_zone_ids must contain non-empty IDs; got an empty zone ID"
            )
        if len(adjacent) > 2 or len(set(adjacent)) != len(adjacent):
            raise ValueError(
                "MeshPatch.adjacent_zone_ids must name at most two distinct zone IDs; got excess or repeated zone IDs"
            )
        source_pair = source_adjacent_region_ids
        if source_pair is not None:
            if len(source_pair) != 2 or all(
                identifier is None for identifier in source_pair
            ):
                raise ValueError(
                    "MeshPatch.source_adjacent_region_ids must name at least one region; got two exterior sides"
                )
            if any(
                identifier is not None
                and (not isinstance(identifier, str) or not identifier.strip())
                for identifier in source_pair
            ):
                raise ValueError(
                    "MeshPatch.source_adjacent_region_ids must be nonempty identifiers or exterior None; got an invalid side identifier"
                )
            if source_pair[0] is not None and source_pair[0] == source_pair[1]:
                raise ValueError(
                    "MeshPatch.source_adjacent_region_ids must name distinct regions; got the same region on both sides"
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


@final
class MeshPatchBank(StrictModule, NonTrainableState):
    """Ordered patches of one publication stored once as columns.

    Names keep their order; connectivity is a boolean column; zone adjacency
    and ordered source-region sides index static dictionaries of their distinct
    values in first-use order (``-1`` marks absent sides); scopes are one
    ``MeshingScopeBank``. Each row is the ``MeshPatch`` its columns construct,
    so patch identities are recomputed rather than stored.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    connected: np.ndarray
    adjacency: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    adjacency_codes: np.ndarray
    source_sides: tuple[tuple[str | None, str | None], ...] = eqx.field(static=True)
    source_side_codes: np.ndarray
    scopes: MeshingScopeBank

    @checked
    def __init__(self, patches: tuple[MeshPatch, ...] = (), /) -> None:
        if any(type(patch) is not MeshPatch for patch in patches):
            raise TypeError(
                "MeshPatchBank.patches must contain MeshPatch rows; got another record kind"
            )
        adjacency: dict[tuple[str, ...], int] = {}
        sides: dict[tuple[str | None, str | None], int] = {}
        adjacency_codes, side_codes = [], []
        for patch in patches:
            adjacency_codes.append(
                adjacency.setdefault(patch.adjacent_zone_ids, len(adjacency))
            )
            pair = patch.source_adjacent_region_ids
            side_codes.append(-1 if pair is None else sides.setdefault(pair, len(sides)))
        self.names = tuple(patch.name for patch in patches)
        self.connected = _read_only_bank(
            np.asarray([patch.connected for patch in patches], dtype=np.bool_)
        )
        self.adjacency = tuple(adjacency)
        self.adjacency_codes = _read_only_bank(
            np.asarray(adjacency_codes, dtype=np.int32)
        )
        self.source_sides = tuple(sides)
        self.source_side_codes = _read_only_bank(np.asarray(side_codes, dtype=np.int32))
        self.scopes = MeshingScopeBank(tuple(patch.scope for patch in patches))

    @classmethod
    @checked
    def from_stored_fields(cls, value: MeshPatchBank, /) -> MeshPatchBank:
        """Rebuild every row through ``MeshPatch``; replay compares the content."""
        return cls(tuple(value))

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[MeshPatch]:
        self._require_columns()
        return (self.patch(index) for index in range(len(self)))

    @overload
    def __getitem__(self, index: int, /) -> MeshPatch: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[MeshPatch, ...]: ...

    def __getitem__(self, index: int | slice, /) -> MeshPatch | tuple[MeshPatch, ...]:
        if isinstance(index, slice):
            return tuple(self.patch(row) for row in range(len(self))[index])
        if type(index) is not int:
            raise TypeError(
                f"MeshPatchBank.index must be an integer or slice; got {type(index).__name__}"
            )
        return self.patch(index if index >= 0 else len(self) + index)

    def patch(self, index: int, /) -> MeshPatch:
        """Construct row ``index``; rows are built on demand and never retained."""
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(
                f"MeshPatchBank.index must be in [0, {len(self)}); got {index!r}"
            )
        side = int(self.source_side_codes[index])
        return MeshPatch(
            self.names[index],
            self.scopes.scope(index),
            connected=bool(self.connected[index]),
            adjacent_zone_ids=self.adjacency[int(self.adjacency_codes[index])],
            source_adjacent_region_ids=None if side < 0 else self.source_sides[side],
        )

    def index(self, name: str, /) -> int:
        """Return the row of the patch named ``name``."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"MeshPatchBank.name must name a patch; got {name!r}") from None

    def _require_columns(self) -> None:
        count = len(self.names)
        if (
            self.connected.shape != (count,)
            or self.connected.dtype != np.bool_
            or self.adjacency_codes.shape != (count,)
            or self.source_side_codes.shape != (count,)
            or len(self.scopes) != count
            or np.any(self.adjacency_codes < 0)
            or np.any(self.adjacency_codes >= len(self.adjacency))
            or np.any(self.source_side_codes < -1)
            or np.any(self.source_side_codes >= len(self.source_sides))
        ):
            raise ValueError(
                "MeshPatchBank.columns must align names, flags, dictionaries and scopes; got inconsistent columns"
            )
        self.scopes.require_columns()


def _read_only_bank(value: np.ndarray, /) -> np.ndarray:
    value = np.ascontiguousarray(value)
    value.flags.writeable = False
    return value


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
            raise ValueError("MeshZone.name must be non-empty; got an empty name")
        if not isinstance(role, MeshZoneRole):
            raise TypeError(
                "MeshZone.role must be MeshZoneRole; got a different role kind"
            )
        material = None if material_id is None else str(material_id).strip()
        if material == "":
            raise ValueError(
                "MeshZone.material_id must be non-empty when supplied; got an empty material ID"
            )
        if (material is None) != (region_role is None):
            raise ValueError(
                "MeshZone.material_id must be supplied together with region_role; got an incomplete material declaration"
            )
        if region_role is not None and not isinstance(region_role, RegionRole):
            raise TypeError(
                "MeshZone.region_role must be RegionRole or None; got a different role kind"
            )
        if material is not None and role is not MeshZoneRole.REGION:
            raise ValueError(
                "MeshZone.material_id must be declared together with region_role only on REGION zones; got a non-region zone"
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
            raise ValueError("MeshLabel.name must be non-empty; got an empty name")
        self.name = value
        self.scope = scope
        self.label_id = canonical_fingerprint(
            {"kind": "mesh-label", "name": value, "scope": scope.scope_id}
        )


@final
class MeshAttributeProjection(StrictModule, NonTrainableState):
    """Bind finite global attribute values to owner-local packet projections."""

    scope_projection: MeshScopeProjection
    global_values: Array
    packet_values: Array
    value_name: str = eqx.field(static=True)
    values_content_id: str = eqx.field(static=True)

    @checked
    def __init__(self, scope_projection: MeshScopeProjection, /) -> None:
        with agreed_local_step():
            name = scope_projection.membership_name
            if not name.startswith("organization/attribute/") or not name.endswith(
                "/membership"
            ):
                raise ValueError(
                    "MeshAttributeProjection.scope_projection must identify attribute membership; got a different membership bank"
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
                    "MeshAttributeProjection.global_values must match the entity axis; got incompatible value axes"
                )
            if values.dtype.kind not in "biuf":
                raise TypeError(
                    "MeshAttributeProjection.global_values must be real numeric arrays; got a non-real dtype"
                )
        rows = jnp.argsort(ids, stable=True)
        positions = jnp.searchsorted(ids[rows], scope_projection.members)
        scoped_values = values[rows[positions]]
        finite = scoped_values.dtype.kind != "f" or bool(
            jax.device_get(jnp.all(jnp.isfinite(scoped_values)))
        )
        with agreed_local_step():
            if not finite:
                raise ValueError(
                    "MeshAttributeProjection.global_values must be finite; got nonfinite values"
                )
        with agreed_local_step():
            packet_values = dict(publication.projected_arrays)[value_name]
            if (
                packet_values.shape != (*scope_projection.packet_ids.shape, *values.shape[1:])
                or packet_values.dtype != values.dtype
            ):
                raise ValueError(
                    "MeshAttributeProjection.packet_values must match part-leading component axes; got incompatible packet axes"
                )
        packet_rows = jnp.minimum(
            jnp.searchsorted(ids[rows], scope_projection.packet_ids), count - 1
        )
        expected = values[rows[packet_rows]]
        valid = scope_projection.packet_valid.reshape(
            (*scope_projection.packet_valid.shape, *((1,) * (values.ndim - 1)))
        )
        accepted = bool(jax.device_get(jnp.all(~valid | (packet_values == expected))))
        with agreed_local_step():
            if not accepted:
                raise ValueError(
                    "MeshAttributeProjection.packet_values must match accepted values; got different packet values"
                )
        content_id = logical_array_value_collection_digest({"values": scoped_values})
        with agreed_local_step():
            self.scope_projection = scope_projection
            self.global_values = scoped_values
            self.packet_values = packet_values
            self.value_name = value_name
            self.values_content_id = content_id

    @checked
    def local_values(self, scope: MeshingScope, /) -> Array:
        local = scope.projection
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
                "MeshAttributeProjection.scope must bind the prepared local scope; got a different scope binding"
            )
        if (
            dict(self.scope_projection.publication.projected_arrays)[self.value_name]
            is not self.packet_values
        ):
            raise ValueError(
                "MeshAttributeProjection.packet_values must retain prepared packet values; got a changed value bank"
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
                "MeshAttributeProjection.scope must cover every local scoped entity; got a missing local entity"
            )
        return jnp.asarray(values[rows], dtype=self.global_values.dtype)


def prepare_mesh_attribute_projections(
    scopes: tuple[tuple[str, MeshScopeProjection], ...],
    /,
) -> tuple[tuple[str, MeshAttributeProjection], ...]:
    """Prepare shared attribute buffers during all-owner publication."""
    return tuple(
        (projection.value_name, projection)
        for name, scope in scopes
        if name.startswith("organization/attribute/")
        for projection in (MeshAttributeProjection(scope),)
    )


class MeshAttribute(StrictModule, NonTrainableState):
    """Associate typed numerical values with one mesh scope."""

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
            raise ValueError("MeshAttribute.name must be non-empty; got an empty name")
        if not isinstance(role, MeshAttributeRole):
            raise TypeError(
                "MeshAttribute.role must be MeshAttributeRole; got a different role kind"
            )
        array = (
            values if isinstance(values, Array) else jax.device_put(np.asarray(values))
        )
        if array.ndim == 0 or array.shape[0] != scope.global_entity_ids.shape[0]:
            raise ValueError(
                "MeshAttribute.values must match the global scoped entity count; got incompatible value axes"
            )
        if _projection is None:
            if array.dtype.kind not in "biuf" or (
                array.dtype.kind == "f"
                and not bool(jax.device_get(jnp.all(jnp.isfinite(array))))
            ):
                raise ValueError(
                    "MeshAttribute.values must contain finite real numeric values; got nonfinite or non-real values"
                )
        elif array is not _projection.global_values:
            raise ValueError(
                "MeshAttribute.values must retain the globally prepared value bank; got a different bank object"
            )
        if _projection is not None:
            local_values = _projection.local_values(scope)
        elif scope.entity_ids is scope.global_entity_ids:
            local_values = array
        else:
            selected, valid = local_logical_lookup(
                scope.global_entity_ids, array, scope.entity_ids
            )
            if not np.all(valid):
                raise ValueError(
                    "MeshAttribute.scope must contain local membership in the global scope; got absent global membership"
                )
            local_values = jax.device_put(np.asarray(selected, dtype=array.dtype))
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[MeshAttribute], value: MeshAttribute, /
    ) -> MeshAttribute:
        """Rebuild scoped values from the retained global bank and projection."""
        return cls(
            value.name,
            value.role,
            value.scope,
            value.global_values,
            unit=value.unit,
            _projection=value._attribute_projection,
        )


@final
class RegionBoundaryEvidence(StrictModule, NonTrainableState):
    """Bind a source volume-region identity to an overlapping surface selection."""

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

    @checked
    def __init__(
        self,
        mesh: cell_mesh_owner.CellMesh,
        source_scope: MeshingScope,
        source_region_id: str,
        control_id: str,
        material_id: str | None,
        role: RegionRole | None,
        boundary_label: MeshLabel,
        patch_sides: tuple[tuple[str, int], ...],
        /,
    ) -> None:
        if mesh.topological_dimension not in (2, 3):
            raise TypeError(
                "RegionBoundaryEvidence.mesh must have topological dimension two or three; got a different mesh dimension"
            )
        if (
            source_scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or source_scope.entity_dimension != 3
            or source_scope.global_entity_ids.shape != (1,)
        ):
            raise ValueError(
                "RegionBoundaryEvidence.source_scope must select one source region; got a different region scope"
            )
        target = boundary_label.scope
        if (
            target.source_id != mesh.mesh_id
            or target.source_revision != mesh.numeric_version
            or target.entity_kind is not MeshingEntityKind.MESH
            or target.entity_dimension != 2
            or target.entity_set_id != mesh.entity_set(2).entity_set_id
        ):
            raise ValueError(
                "RegionBoundaryEvidence.boundary_label must bind this surface mesh revision; got a different mesh binding"
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
                "RegionBoundaryEvidence.material_id must be supplied together with role; got an incomplete material declaration"
            )
        sides = tuple(sorted(patch_sides))
        if (
            not sides
            or len({patch for patch, _ in sides}) != len(sides)
            or any(not patch or side not in (-1, 1) for patch, side in sides)
        ):
            raise ValueError(
                "RegionBoundaryEvidence.patch_sides must declare distinct oriented source patches; got absent, repeated, or unoriented patches"
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
        mesh: cell_mesh_owner.CellMesh,
        labels: tuple[MeshLabel, ...],
        patches: tuple[MeshPatch, ...] | MeshPatchBank,
        /,
    ) -> None:
        if (
            mesh.mesh_id != self.target_mesh_id
            or mesh.topology_id != self.target_topology_id
        ):
            raise ValueError(
                "RegionBoundaryEvidence.target_mesh_id must bind this surface mesh; got stale boundary evidence"
            )
        if self.boundary_label.label_id not in {label.label_id for label in labels}:
            raise ValueError(
                "RegionBoundaryEvidence.boundary_label must belong to the result; got an absent boundary label"
            )
        lookup = {patch.patch_id: patch for patch in patches}
        selected = []
        for identifier, side in self.patch_sides:
            if identifier not in lookup:
                raise ValueError(
                    "RegionBoundaryEvidence.patch_sides must identify result patches; got an absent boundary patch"
                )
            patch = lookup[identifier]
            pair = patch.source_adjacent_region_ids
            if pair is None or pair[0 if side > 0 else 1] != self.source_region_id:
                raise ValueError(
                    "RegionBoundaryEvidence.patch_sides must match source-region orientation; got a contradictory surface patch"
                )
            selected.append(patch.scope.global_entity_ids)
        union = unique_ids(jnp.concatenate(selected))
        expected = self.boundary_label.scope.global_entity_ids
        if union.shape != expected.shape or not bool(
            jax.device_get(jnp.all(union == expected))
        ):
            raise ValueError(
                "RegionBoundaryEvidence.boundary_label must select its oriented source patches; got omitted or added patches"
            )


class _RegionInventory(StrictModule):
    cells: Array
    regions: Array
    facets: Array
    interfaces: Array
    orientations: Array
    observed: frozenset[str] = eqx.field(static=True)


def _prepare_host_region_inventory(
    mesh: cell_mesh_owner.CellMesh,
    domain: RegionSourceDomain,
    ids: tuple[int, ...],
    regions: tuple[str, ...],
    definitions: tuple[tuple[str, str, str, bool], ...],
    facets: tuple[tuple[str, int, int, str, str], ...],
) -> _RegionInventory:
    if set(regions) != set(domain.region_ids):
        raise ValueError(
            "RegionMeshingEvidence.cell_region_ids must cover every source region; got an unassigned region"
        )
    cells = np.asarray(ids, dtype=np.int64)
    region_rows = {name: index for index, name in enumerate(domain.region_ids)}
    region_indices = np.asarray(
        tuple(region_rows[name] for name in regions), dtype=np.int32
    )
    facet_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    facet_lookup = {int(value): index for index, value in enumerate(facet_ids)}
    interface_rows = {name: index for index, (name, _, _, _) in enumerate(definitions)}
    interface_indices = np.full(facet_ids.shape, -1, dtype=np.int32)
    orientations = np.zeros(facet_ids.shape, dtype=np.int32)
    for name, facet, sign, _, _ in facets:
        interface_indices[facet_lookup[facet]] = interface_rows[name]
        orientations[facet_lookup[facet]] = sign
    cell_order, facet_order = (
        np.argsort(cells, stable=True),
        np.argsort(facet_ids, stable=True),
    )
    return _RegionInventory(
        jnp.asarray(cells[cell_order]),
        jnp.asarray(region_indices[cell_order]),
        jnp.asarray(facet_ids[facet_order]),
        jnp.asarray(interface_indices[facet_order]),
        jnp.asarray(orientations[facet_order]),
        frozenset(name for name, _, _, _, _ in facets),
    )


def _prepare_region_inventory(
    mesh: cell_mesh_owner.CellMesh,
    domain: RegionSourceDomain,
    ids: tuple[int, ...],
    regions: tuple[str, ...],
    definitions: tuple[tuple[str, str, str, bool], ...],
    facets: tuple[tuple[str, int, int, str, str], ...],
) -> _RegionInventory:
    if mesh.storage is None:
        return _prepare_host_region_inventory(
            mesh, domain, ids, regions, definitions, facets
        )
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
    local_regions, valid = local_logical_lookup(
        logical_cells, logical_regions, jnp.asarray(ids, dtype=jnp.int64)
    )
    if not np.all(valid):
        raise ValueError(
            "RegionMeshingEvidence.cell_global_ids must belong to the global inventory; got absent local cells"
        )
    if tuple(domain.region_ids[index] for index in local_regions) != regions:
        raise ValueError(
            "RegionMeshingEvidence.cell_region_ids must match the global inventory; got different local assignments"
        )
    expected_facets = local_interface_records(
        mesh,
        logical_facets,
        logical_interfaces,
        logical_orientations,
        definitions,
    )
    if expected_facets != facets:
        raise ValueError(
            "RegionMeshingEvidence.interface_facets must match the global interface inventory; got different local facets"
        )
    return _RegionInventory(
        logical_cells,
        logical_regions,
        logical_facets,
        logical_interfaces,
        logical_orientations,
        frozenset(observed),
    )


def _require_material_certificate_limits(
    coverage: DomainCoverageCertificate, limits: MeshCertificateLimits, /
) -> None:
    reconstructed = replace(limits)
    if (
        limits.limits_id != reconstructed.limits_id
        or limits.limits_id != coverage.binding.limits_id
    ):
        raise ValueError(
            "RegionMeshingEvidence.certificate_limits must match their validated controls and coverage binding; got different material certificate limits"
        )


@final
class RegionMeshingEvidence(StrictModule, NonTrainableState):
    """Bind source-material assignments to certified domain coverage.

    Interface orientation is relative to the mesh facet: ``+1`` points from
    the definition's first region to its second. The retained source domain
    supports coverage revalidation after topology or coordinate changes.
    Retained certificate controls authenticate the coverage limits binding
    and support independent renewal without substituting final-request caps.
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
    domain: RegionSourceDomain
    coverage: DomainCoverageCertificate
    certificate_limits: MeshCertificateLimits
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: cell_mesh_owner.CellMesh,
        geometry: cell_geometry_owner.CellGeometrySpec,
        source_revision: str,
        source_complex_id: str,
        cell_region_ids: tuple[str, ...],
        region_zone_ids: tuple[tuple[str, str], ...],
        interface_patch_ids: tuple[tuple[str, str], ...],
        interface_definitions: tuple[tuple[str, str, str, bool], ...],
        interface_facets: tuple[tuple[str, int, int, str, str], ...],
        domain: RegionSourceDomain,
        coverage: DomainCoverageCertificate,
        /,
        *,
        certificate_limits: MeshCertificateLimits,
    ) -> None:
        _require_material_certificate_limits(coverage, certificate_limits)
        coverage.binding.require(mesh, geometry)
        if coverage.status != "certified" or coverage.domain_id != domain.domain_id:
            raise ValueError(
                "RegionMeshingEvidence.coverage must certify its domain; got uncertified or different domain coverage"
            )
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
                "RegionMeshingEvidence.cell_region_ids must assign every local cell to a source region; got absent or unknown assignments"
            )
        zone_ids = tuple(sorted(region_zone_ids))
        patch_ids = tuple(sorted(interface_patch_ids))
        definitions = tuple(sorted(interface_definitions))
        facets = tuple(sorted(interface_facets))
        if tuple(value[0] for value in zone_ids) != tuple(
            sorted(domain.region_ids)
        ) or len({value[1] for value in zone_ids}) != len(zone_ids):
            raise ValueError(
                "RegionMeshingEvidence.region_zone_ids must bind one exclusive zone per region; got incomplete or repeated zone bindings"
            )
        if len({value[0] for value in definitions}) != len(definitions):
            raise ValueError(
                "RegionMeshingEvidence.interface_definitions must have unique IDs; got duplicate interface identities"
            )
        known = {value[0]: value[1:3] for value in definitions}
        if any(
            first not in domain.region_ids
            or second not in domain.region_ids
            or first == second
            for _, first, second, _ in definitions
        ):
            raise ValueError(
                "RegionMeshingEvidence.interface_definitions must name distinct known regions; got unknown or repeated regions"
            )
        if len({value[1] for value in facets}) != len(facets) or any(
            name not in known or (first, second) != known[name] or sign not in (-1, 1)
            for name, _, sign, first, second in facets
        ):
            raise ValueError(
                "RegionMeshingEvidence.interface_facets must be exclusive and oriented; got repeated, unknown, or unoriented facets"
            )
        inventory = _prepare_region_inventory(
            mesh, domain, ids, regions, definitions, facets
        )
        logical_cells, logical_regions = inventory.cells, inventory.regions
        logical_facets, logical_interfaces, logical_orientations = (
            inventory.facets,
            inventory.interfaces,
            inventory.orientations,
        )
        observed = inventory.observed
        if any(required and name not in observed for name, _, _, required in definitions):
            raise ValueError(
                "RegionMeshingEvidence.interface_facets must include required interfaces; got a required interface without facets"
            )
        if {value[0] for value in patch_ids} != observed:
            raise ValueError(
                "RegionMeshingEvidence.interface_patch_ids must bind every observed interface; got missing or extra interface bindings"
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
        self.certificate_limits = certificate_limits
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

    @checked
    def require_source(self, source: CompartmentComplex, /) -> None:
        if (
            self.source_revision != source.source_revision
            or self.source_complex_id != source.complex_id
            or set(self.domain.region_ids)
            != {value.compartment_id for value in source.compartments}
            or self.adjacency_pairs != source.observed_adjacencies
            or self.interface_definitions != complete_interface_definitions(source)
        ):
            raise ValueError(
                "RegionMeshingEvidence.source_complex_id must bind this compartment source; got different source evidence"
            )

    @checked
    def require_current(
        self,
        mesh: cell_mesh_owner.CellMesh,
        zones: tuple[MeshZone, ...],
        patches: tuple[MeshPatch, ...] | MeshPatchBank,
        /,
        *,
        geometry: cell_geometry_owner.CellGeometrySpec | None = None,
    ) -> None:
        _require_material_certificate_limits(self.coverage, self.certificate_limits)
        if self.target_mesh_id != mesh.mesh_id:
            raise ValueError(
                "RegionMeshingEvidence.target_mesh_id must bind current mesh coordinates; got stale mesh coordinates"
            )
        geometry_ = (
            mesh.storage.restore_geometry()
            if geometry is None and mesh.storage is not None
            else cell_geometry_owner.CellGeometrySpec.affine(mesh)
            if geometry is None
            else geometry
        )
        self.coverage.binding.require(mesh, geometry_)
        ids = tuple(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64).tolist())
        if ids != self.cell_global_ids:
            raise ValueError(
                "RegionMeshingEvidence.cell_global_ids must match the current mesh; got different cell IDs"
            )
        local_regions, valid = local_logical_lookup(
            self.logical_cell_global_ids,
            self.logical_cell_region_indices,
            jnp.asarray(ids, dtype=jnp.int64),
        )
        if not np.all(valid):
            raise ValueError(
                "RegionMeshingEvidence.cell_global_ids must belong to the global material inventory; got absent local cells"
            )
        if (
            tuple(self.domain.region_ids[index] for index in local_regions)
            != self.cell_region_ids
        ):
            raise ValueError(
                "RegionMeshingEvidence.cell_region_ids must match the global material inventory; got different local assignments"
            )
        zone_by_id = {value.zone_id: value for value in zones}
        for region, zone_id in self.region_zone_ids:
            if zone_id not in zone_by_id:
                raise ValueError(
                    "RegionMeshingEvidence.region_zone_ids must identify result zones; got an absent zone binding"
                )
            zone = zone_by_id[zone_id]
            expected = selected_ids(
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
                    "RegionMeshingEvidence.region_zone_ids must match source-region assignments; got a contradictory zone"
                )
        patch_by_id = {value.patch_id: value for value in patches}
        zone_for_region = dict(self.region_zone_ids)
        definitions = {value[0]: value[1:3] for value in self.interface_definitions}
        for interface, patch_id in self.interface_patch_ids:
            if patch_id not in patch_by_id:
                raise ValueError(
                    "RegionMeshingEvidence.interface_patch_ids must identify result patches; got an absent interface patch"
                )
            patch = patch_by_id[patch_id]
            pair = definitions[interface]
            interface_index = tuple(
                value[0] for value in self.interface_definitions
            ).index(interface)
            expected = selected_ids(
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
                raise ValueError(
                    "RegionMeshingEvidence.interface_patch_ids must match interface evidence; got a contradictory patch"
                )
        if mesh.storage is None:
            expected_facets = material_interface_facets(
                mesh, self.cell_region_ids, self.interface_definitions
            )
            if expected_facets != self.interface_facets:
                raise ValueError(
                    "RegionMeshingEvidence.interface_facets must match current orientation and adjacency; got stale material facets"
                )
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
                    raise ValueError(
                        "RegionMeshingEvidence.logical_arrays must match current incidence and orientation; got stale global material incidence"
                    )
            if (
                local_interface_records(
                    mesh,
                    self.logical_facet_global_ids,
                    self.logical_facet_interface_indices,
                    self.logical_facet_orientations,
                    self.interface_definitions,
                )
                != self.interface_facets
            ):
                raise ValueError(
                    "RegionMeshingEvidence.interface_facets must match the global interface inventory; got different local facets"
                )


def _logical_region_inventory(
    mesh: cell_mesh_owner.CellMesh,
    domain: RegionSourceDomain,
    definitions: tuple[tuple[str, str, str, bool], ...],
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    storage = mesh.storage
    if storage is None:
        raise ValueError(
            "RegionMeshingEvidence.mesh.storage must contain accepted logical storage; got missing storage"
        )
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
            "RegionMeshingEvidence.logical_arrays must have aligned integer entity axes; got incompatible bank axes"
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
            "RegionMeshingEvidence.logical_arrays must agree on coverage, adjacency, and interface orientation; got inconsistent material inventory"
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


def local_interface_records(
    mesh: cell_mesh_owner.CellMesh,
    facets: Array,
    interfaces: Array,
    orientations: Array,
    definitions: tuple[tuple[str, str, str, bool], ...],
    /,
) -> tuple[tuple[str, int, int, str, str], ...]:
    local_facets = mesh.entity_set(2).entity_ids
    local_values, valid = local_logical_lookup(
        facets,
        jnp.stack((interfaces, orientations.astype(interfaces.dtype)), axis=1),
        local_facets,
    )
    if not np.all(valid):
        raise ValueError(
            "RegionMeshingEvidence.interface_facets must belong to the global interface inventory; got an absent local facet"
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


@checked
def material_interface_facets(
    mesh: cell_mesh_owner.CellMesh,
    cell_region_ids: tuple[str, ...],
    definitions: tuple[tuple[str, str, str, bool], ...],
    /,
) -> tuple[tuple[str, int, int, str, str], ...]:
    """Recover material adjacency from whole-cell ownership, never a point sample."""
    connectivity = mesh.connectivity
    match connectivity:
        case cell_complex_owner.TetrahedralConnectivity():
            face_values = np.asarray(connectivity.cell_faces, dtype=np.int64).reshape(-1)
            sign_values = np.asarray(connectivity.cell_face_signs, dtype=np.int8).reshape(
                -1
            )
            cell_rows = np.repeat(np.arange(connectivity.cell_count, dtype=np.int64), 4)
            slot_global_ids = np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
            )
            face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
        case cell_complex_owner.PolyhedralConnectivity():
            face_values = np.asarray(connectivity.cell_face_values, dtype=np.int64)
            sign_values = np.asarray(connectivity.cell_face_sign_values, dtype=np.int8)
            cell_rows = np.repeat(
                np.arange(connectivity.cell_count, dtype=np.int64),
                np.diff(np.asarray(connectivity.cell_face_offsets, dtype=np.int64)),
            )
            slot_global_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
            face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
        case _:
            raise TypeError(
                "material_interface_facets.mesh must have volume-cell incidence; got a different connectivity kind"
            )
    if len(cell_region_ids) != connectivity.cell_count:
        raise ValueError(
            "material_interface_facets.cell_region_ids must cover every volume cell; got incomplete material assignments"
        )
    assignments_by_id = dict(
        zip(
            np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64).tolist(),
            cell_region_ids,
            strict=True,
        )
    )
    if set(slot_global_ids.tolist()) != set(assignments_by_id):
        raise ValueError(
            "material_interface_facets.cell_region_ids must match volume incidence cell identities; got different cell IDs"
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
            raise ValueError(
                "material_interface_facets.mesh must have at most two cells per facet; got a nonmanifold material facet"
            )
        if len(assigned) != 2 or assigned[0][0] == assigned[1][0]:
            continue
        pair = tuple(sorted((assigned[0][0], assigned[1][0])))
        if pair not in by_pair:
            raise ValueError(
                "material_interface_facets.interface_definitions must cover every material adjacency; got an undeclared adjacency"
            )
        name, first, second = by_pair[pair]
        orientation = assigned[0][1] if assigned[0][0] == first else assigned[1][1]
        facets.append((name, int(face_ids[face]), orientation, first, second))
    return tuple(sorted(facets))


@checked
def organization_definition_id(
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
                "organization_definition_id.record must have a registered organization class; got a different record kind"
            )


def validate_mesh_zones(zones: tuple[MeshZone, ...], /) -> tuple[MeshZone, ...]:
    if not all(isinstance(zone, MeshZone) for zone in zones):
        raise TypeError(
            "validate_mesh_zones.zones must contain MeshZone values; got a different record kind"
        )
    names = tuple(zone.name for zone in zones)
    if len(set(names)) != len(names):
        raise ValueError(
            "validate_mesh_zones.zones must have unique names; got duplicate zone names"
        )
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
            if scope.projection is not None or other.projection is not None:
                projection, previous_projection = (
                    scope.projection,
                    other.projection,
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
                        "validate_mesh_zones.zones must retain globally prepared exclusivity evidence; got different projection bindings"
                    )
                continue
            if overlapping_ids(other.global_entity_ids, scope.global_entity_ids):
                raise ValueError(
                    "validate_mesh_zones.zones must be disjoint on one entity set; got overlapping zone membership"
                )
    return zones


def validate_mesh_labels(labels: tuple[MeshLabel, ...], /) -> tuple[MeshLabel, ...]:
    if not all(isinstance(label, MeshLabel) for label in labels):
        raise TypeError(
            "validate_mesh_labels.labels must contain MeshLabel values; got a different record kind"
        )
    names = tuple(label.name for label in labels)
    if len(set(names)) != len(names):
        raise ValueError(
            "validate_mesh_labels.labels must have unique names; got duplicate label names"
        )
    return labels


__all__ = [
    "MeshAttribute",
    "MeshAttributeRole",
    "MeshLabel",
    "MeshPatch",
    "MeshPatchBank",
    "MeshZone",
    "MeshZoneRole",
    "RegionRole",
    "RegionMeshingEvidence",
    "RegionBoundaryEvidence",
    "material_interface_facets",
    "validate_mesh_labels",
    "validate_mesh_zones",
]
