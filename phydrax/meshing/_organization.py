#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization import CellMesh
from ..typing import checked
from ..units import UnitDefinition
from ._scope import MeshingEntityKind, MeshingScope, resolve_mesh_scope


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
    source_adjacent_region_occurrence_paths: (
        tuple[tuple[str, ...] | None, tuple[str, ...] | None] | None
    ) = eqx.field(static=True)
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
        source_adjacent_region_occurrence_paths: (
            tuple[tuple[str, ...] | None, tuple[str, ...] | None] | None
        ) = None,
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
        source_paths = source_adjacent_region_occurrence_paths
        if source_pair is None:
            if source_paths is not None:
                raise ValueError(
                    "Source patch occurrence paths require source region identities."
                )
        else:
            if (
                not isinstance(source_pair, tuple)
                or len(source_pair) != 2
                or source_paths is None
                or not isinstance(source_paths, tuple)
                or len(source_paths) != 2
            ):
                raise ValueError(
                    "Source patch sides require parallel region IDs and occurrence paths."
                )
            first, second = source_pair
            paths = source_paths
            if (
                (first is None) != (paths[0] is None)
                or (second is None) != (paths[1] is None)
            ):
                raise ValueError(
                    "Each source patch side requires both a region ID and occurrence path."
                )
            source_pair = (
                None
                if first is None
                else canonical_identifier(first, "source_region_id"),
                None
                if second is None
                else canonical_identifier(second, "source_region_id"),
            )
            if any(path is not None and not isinstance(path, tuple) for path in paths):
                raise TypeError("Source occurrence paths must be tuples.")
            source_paths = paths
            if any(
                path is not None
                and any(
                    not isinstance(part, str) or not part or part != part.strip()
                    for part in path
                )
                for path in source_paths
            ):
                raise ValueError("Source occurrence paths must contain canonical names.")
            if source_pair == (None, None):
                raise ValueError(
                    "A source patch must name at least one adjacent region occurrence."
                )
        self.name = value
        self.scope = scope
        self.connected = bool(connected)
        self.adjacent_zone_ids = adjacent
        self.source_adjacent_region_ids = source_pair
        self.source_adjacent_region_occurrence_paths = source_paths
        self.patch_id = canonical_fingerprint(
            {
                "kind": "mesh-patch",
                "name": value,
                "scope": scope.scope_id,
                "connected": bool(connected),
                "adjacent_zone_ids": adjacent,
                "source_adjacent_region_ids": source_pair,
                "source_adjacent_region_occurrence_paths": source_paths,
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


class MeshAttribute(StrictModule, NonTrainableState):
    """Typed numeric data associated with one exact mesh scope."""

    name: str = eqx.field(static=True)
    role: MeshAttributeRole = eqx.field(static=True)
    scope: MeshingScope
    values: Array
    unit: UnitDefinition | None = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    attribute_id: str = eqx.field(static=True)

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
    ) -> None:
        value = str(name).strip()
        if not value:
            raise ValueError("Mesh attribute name must be non-empty.")
        if not isinstance(role, MeshAttributeRole):
            raise TypeError("role must be MeshAttributeRole.")
        if unit is not None and not isinstance(unit, UnitDefinition):
            raise TypeError("unit must be UnitDefinition or None.")
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != scope.entity_ids.shape[0]:
            raise ValueError("Mesh attribute values must match its scoped entity count.")
        if array.dtype.kind not in "biuf" or (
            array.dtype.kind == "f" and not np.all(np.isfinite(array))
        ):
            raise ValueError("Mesh attributes must contain finite real numeric values.")
        self.name = value
        self.role = role
        self.scope = scope
        self.values = jnp.asarray(array)
        self.unit = unit
        self.component_shape = tuple(array.shape[1:])
        self.attribute_id = canonical_fingerprint(
            {
                "kind": "mesh-attribute",
                "name": value,
                "role": role.value,
                "scope": scope.scope_id,
                "unit": None if unit is None else unit.unit_id,
                "values": array_tree_fingerprint(array),
            }
        )


def validate_mesh_zones(zones: tuple[MeshZone, ...], /) -> tuple[MeshZone, ...]:
    if not all(isinstance(zone, MeshZone) for zone in zones):
        raise TypeError("zones must contain MeshZone values.")
    names = tuple(zone.name for zone in zones)
    if len(set(names)) != len(names):
        raise ValueError("Mesh zone names must be unique.")
    occupied: dict[tuple[str, str, int, str], set[int]] = {}
    for zone in zones:
        scope = zone.scope
        key = (
            scope.source_id,
            scope.source_revision,
            scope.entity_dimension,
            scope.entity_set_id,
        )
        identifiers = {int(value) for value in np.asarray(scope.entity_ids)}
        previous = occupied.setdefault(key, set())
        if previous & identifiers:
            raise ValueError("Mesh zones on one entity set must be disjoint.")
        previous.update(identifiers)
    return zones


def validate_mesh_labels(labels: tuple[MeshLabel, ...], /) -> tuple[MeshLabel, ...]:
    if not all(isinstance(label, MeshLabel) for label in labels):
        raise TypeError("labels must contain MeshLabel values.")
    names = tuple(label.name for label in labels)
    if len(set(names)) != len(names):
        raise ValueError("Mesh label names must be unique.")
    return labels


class RegionBoundaryEvidence(StrictModule, NonTrainableState):
    """Bind a source region and oriented patches to a target boundary selection."""

    source_scope: MeshingScope
    source_region_id: str = eqx.field(static=True)
    source_occurrence_path: tuple[str, ...] = eqx.field(static=True)
    control_id: str | None = eqx.field(static=True)
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
        mesh: CellMesh,
        source_scope: MeshingScope,
        source_region_id: str,
        control_id: str | None,
        material_id: str | None,
        role: RegionRole | None,
        boundary_label: MeshLabel,
        patch_sides: tuple[tuple[str, int], ...],
        /,
        *,
        source_occurrence_path: tuple[str, ...],
    ) -> None:
        if (
            source_scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or source_scope.entity_dimension not in (2, 3)
            or source_scope.entity_ids.shape != (1,)
        ):
            raise ValueError("Region boundary evidence must select one source region.")
        dimension = source_scope.entity_dimension - 1
        target = boundary_label.scope
        if (
            target.source_id != mesh.mesh_id
            or target.source_revision != mesh.numeric_version
            or target.entity_kind is not MeshingEntityKind.MESH
            or target.entity_dimension != dimension
            or target.entity_set_id != mesh.entity_set(dimension).entity_set_id
        ):
            raise ValueError(
                "The boundary label must bind the current mesh boundary stratum."
            )
        resolve_mesh_scope(mesh, target)
        region = canonical_identifier(source_region_id, "source_region_id")
        if (
            not isinstance(source_occurrence_path, tuple)
            or any(
                not isinstance(part, str) or not part or part != part.strip()
                for part in source_occurrence_path
            )
        ):
            raise ValueError("Source occurrence path must contain canonical names.")
        control = (
            None
            if control_id is None
            else canonical_identifier(control_id, "control_id")
        )
        material = (
            None
            if material_id is None
            else canonical_identifier(material_id, "material_id")
        )
        if (material is None) != (role is None) or (
            role is not None and not isinstance(role, RegionRole)
        ):
            raise ValueError(
                "A material identity and region role must be supplied together."
            )
        sides = tuple(sorted(patch_sides))
        if (
            not sides
            or len({patch for patch, _ in sides}) != len(sides)
            or any(not patch or side not in (-1, 1) for patch, side in sides)
        ):
            raise ValueError(
                "Source patches must have distinct identifiers and explicit orientations."
            )
        self.source_scope = source_scope
        self.source_region_id = region
        self.source_occurrence_path = source_occurrence_path
        self.control_id = control
        self.material_id = material
        self.role = role
        self.boundary_label = boundary_label
        self.patch_sides = sides
        self.target_mesh_id = mesh.mesh_id
        self.target_topology_id = mesh.topology_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "region-boundary-evidence",
                "source": source_scope.scope_id,
                "region": region,
                "source_occurrence_path": source_occurrence_path,
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
        if (mesh.mesh_id, mesh.topology_id) != (
            self.target_mesh_id,
            self.target_topology_id,
        ):
            raise ValueError("Region boundary evidence is bound to a different mesh.")
        if self.boundary_label.label_id not in {label.label_id for label in labels}:
            raise ValueError(
                "The accepted result omits the declared region boundary label."
            )
        selected: list[np.ndarray] = []
        lookup = {patch.patch_id: patch for patch in patches}
        for identifier, side in self.patch_sides:
            if identifier not in lookup:
                raise ValueError("The accepted result omits a region boundary patch.")
            patch = lookup[identifier]
            pair = patch.source_adjacent_region_ids
            paths = patch.source_adjacent_region_occurrence_paths
            if (
                pair is None
                or paths is None
                or pair[0 if side > 0 else 1] != self.source_region_id
                or paths[0 if side > 0 else 1] != self.source_occurrence_path
            ):
                raise ValueError(
                    "Patch source sides contradict the region occurrence orientation."
                )
            resolve_mesh_scope(mesh, patch.scope)
            selected.append(np.asarray(patch.scope.entity_ids))
        union = np.unique(np.concatenate(selected))
        if not np.array_equal(union, np.asarray(self.boundary_label.scope.entity_ids)):
            raise ValueError(
                "Region boundary patches omit or add selected mesh entities."
            )


__all__ = [
    "MeshAttribute",
    "MeshAttributeRole",
    "MeshLabel",
    "MeshPatch",
    "MeshZone",
    "MeshZoneRole",
    "RegionRole",
    "RegionBoundaryEvidence",
    "validate_mesh_labels",
    "validate_mesh_zones",
]
