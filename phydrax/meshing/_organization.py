#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import UnitDefinition
from ._scope import MeshingScope


class MeshZoneRole(StrEnum):
    BOUNDARY = "boundary"
    MATERIAL = "material"
    REGION = "region"
    PARTITION = "partition"
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
    patch_id: str = eqx.field(static=True)

    def __init__(self, name: str, scope: MeshingScope, /, *, connected: bool = True):
        value = str(name).strip()
        if not value:
            raise ValueError("Mesh patch name must be non-empty.")
        if not isinstance(scope, MeshingScope):
            raise TypeError("Mesh patch scope must be MeshingScope.")
        self.name = value
        self.scope = scope
        self.connected = bool(connected)
        self.patch_id = canonical_fingerprint(
            {
                "kind": "mesh-patch",
                "name": value,
                "scope": scope.scope_id,
                "connected": bool(connected),
            }
        )


class MeshZone(StrictModule, NonTrainableState):
    """Named exclusive semantic assignment on one entity set."""

    name: str = eqx.field(static=True)
    role: MeshZoneRole = eqx.field(static=True)
    scope: MeshingScope
    zone_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        role: MeshZoneRole,
        scope: MeshingScope,
        /,
    ):
        value = str(name).strip()
        if not value:
            raise ValueError("Mesh zone name must be non-empty.")
        if not isinstance(role, MeshZoneRole):
            raise TypeError("role must be MeshZoneRole.")
        if not isinstance(scope, MeshingScope):
            raise TypeError("Mesh zone scope must be MeshingScope.")
        self.name = value
        self.role = role
        self.scope = scope
        self.zone_id = canonical_fingerprint(
            {
                "kind": "mesh-zone",
                "name": value,
                "role": role.value,
                "scope": scope.scope_id,
            }
        )


class MeshLabel(StrictModule, NonTrainableState):
    """Named overlapping semantic selection."""

    name: str = eqx.field(static=True)
    scope: MeshingScope
    label_id: str = eqx.field(static=True)

    def __init__(self, name: str, scope: MeshingScope, /):
        value = str(name).strip()
        if not value:
            raise ValueError("Mesh label name must be non-empty.")
        if not isinstance(scope, MeshingScope):
            raise TypeError("Mesh label scope must be MeshingScope.")
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

    def __init__(
        self,
        name: str,
        role: MeshAttributeRole,
        scope: MeshingScope,
        values: ArrayLike,
        /,
        *,
        unit: UnitDefinition | None = None,
    ):
        value = str(name).strip()
        if not value:
            raise ValueError("Mesh attribute name must be non-empty.")
        if not isinstance(role, MeshAttributeRole):
            raise TypeError("role must be MeshAttributeRole.")
        if not isinstance(scope, MeshingScope):
            raise TypeError("Mesh attribute scope must be MeshingScope.")
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
        self.component_shape = tuple(int(size) for size in array.shape[1:])
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


__all__ = [
    "MeshAttribute",
    "MeshAttributeRole",
    "MeshLabel",
    "MeshPatch",
    "MeshZone",
    "MeshZoneRole",
    "validate_mesh_labels",
    "validate_mesh_zones",
]
