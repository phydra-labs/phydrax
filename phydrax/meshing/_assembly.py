#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .._fingerprint import canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from . import (
    _coupling as coupling_owner,
    _interface_binding as interface_binding_owner,
    _mesh_part as mesh_part_owner,
)
from ._scope import MeshingScope, overlapping_ids


class AssemblyInventory(StrictModule, NonTrainableState):
    """Part inventory with its shared coordinate contract."""

    parts: tuple[mesh_part_owner.MeshPart, ...]
    coordinate_contract: SpatialCoordinateContract


def _prepare_parts(parts: tuple[mesh_part_owner.MeshPart, ...], /) -> AssemblyInventory:
    values = tuple(parts)
    if not values or not all(
        isinstance(part, mesh_part_owner.MeshPart) for part in values
    ):
        raise ValueError(
            f"MeshAssembly.parts must contain MeshPart values; got {len(values)} entries"
        )
    if len({part.name for part in values}) != len(values):
        raise ValueError(
            f"MeshAssembly.parts must have unique names; got {tuple(part.name for part in values)!r}"
        )
    values = tuple(sorted(values, key=lambda part: part.name))
    contract = values[0].coordinate_contract
    if any(
        part.coordinate_contract.spatial_id != contract.spatial_id
        or part.ambient_dimension != values[0].ambient_dimension
        for part in values
    ):
        raise ValueError(
            f"MeshAssembly.parts must share a coordinate contract and ambient dimension; got {tuple((part.coordinate_contract.spatial_id, part.ambient_dimension) for part in values)!r}"
        )
    return AssemblyInventory(values, contract)


class OversetOwnership(StrictModule, NonTrainableState):
    """Validated receptor and hole scopes grouped by source and entity set."""

    receptors: tuple[tuple[tuple[str, str], tuple[MeshingScope, ...]], ...]
    holes: tuple[tuple[tuple[str, str], tuple[MeshingScope, ...]], ...]


def _prepare_overset_ownership(
    overlays: tuple[coupling_owner.MeshCoupling, ...],
    by_name: dict[str, mesh_part_owner.MeshPart],
    /,
) -> OversetOwnership:
    receptors: dict[tuple[str, str], list[MeshingScope]] = {}
    holes: dict[tuple[str, str], list[MeshingScope]] = {}
    hole_scopes: dict[tuple[str, str], str] = {}
    for overlay in overlays:
        if not isinstance(overlay, coupling_owner.OversetCoupling):
            continue
        key = (overlay.target_scope.source_id, overlay.target_scope.entity_set_id)
        owned = receptors.setdefault(key, [])
        if any(
            overlapping_ids(
                previous.global_entity_ids, overlay.target_scope.global_entity_ids
            )
            for previous in owned
        ):
            raise ValueError(
                f"MeshAssembly.couplings must assign each receptor to one donor overlay; got overlapping targets in {key!r}"
            )
        owned.append(overlay.target_scope)
        if overlay.hole_scope is not None:
            by_name[overlay.hole_scope.source_id].require_scope(overlay.hole_scope)
            previous = hole_scopes.setdefault(key, overlay.hole_scope.scope_id)
            if previous != overlay.hole_scope.scope_id:
                raise ValueError(
                    f"MeshAssembly.couplings must share one hole scope per receptor entity set; got {previous!r} and {overlay.hole_scope.scope_id!r}"
                )
            holes.setdefault(key, []).append(overlay.hole_scope)
    for key, scopes in receptors.items():
        if any(
            overlapping_ids(scope.global_entity_ids, hole.global_entity_ids)
            for scope in scopes
            for hole in holes.get(key, ())
        ):
            raise ValueError(
                f"MeshAssembly.couplings must keep receptor and hole scopes disjoint; got overlapping ownership in {key!r}"
            )
    return OversetOwnership(
        tuple((key, tuple(scopes)) for key, scopes in receptors.items()),
        tuple((key, tuple(scopes)) for key, scopes in holes.items()),
    )


def _require_donor_ownership(
    overlays: tuple[coupling_owner.MeshCoupling, ...], ownership: OversetOwnership, /
) -> None:
    receptors, holes = dict(ownership.receptors), dict(ownership.holes)
    for overlay in overlays:
        if not isinstance(overlay, coupling_owner.OversetCoupling):
            continue
        if overlay.field_query is not None:
            support = overlay.support_cell_scope
            if support is None:
                raise ValueError(
                    "MeshAssembly.couplings must provide support_cell_scope for query-mode donors; got None"
                )
            key = (support.source_id, support.entity_set_id)
            used = support.global_entity_ids
        else:
            key = (overlay.source_scope.source_id, overlay.source_scope.entity_set_id)
            weights = overlay.donor_weights
            if weights is None:
                raise ValueError(
                    "MeshAssembly.couplings must provide donor_weights for vertex stencils; got None"
                )
            used = jnp.where(weights > 0, overlay.donor_ids, -1).reshape(-1)
        forbidden = (*receptors.get(key, ()), *holes.get(key, ()))
        if any(
            overlapping_ids(scope.global_entity_ids, used)
            for scope in forbidden
        ):
            raise ValueError(
                f"MeshAssembly.couplings must keep donors outside receptor and hole scopes; got intersecting donor support in {key!r}"
            )


class MeshAssembly(StrictModule, NonTrainableState):
    """Named parts and revision-bound coupling overlays, with no implicit welding."""

    parts: tuple[mesh_part_owner.MeshPart, ...]
    couplings: tuple[coupling_owner.MeshCoupling, ...]
    coordinate_contract: SpatialCoordinateContract
    assembly_id: str = eqx.field(static=True)

    def __init__(
        self,
        parts: tuple[mesh_part_owner.MeshPart, ...],
        /,
        *,
        couplings: tuple[coupling_owner.MeshCoupling, ...] = (),
    ) -> None:
        inventory = _prepare_parts(parts)
        values = inventory.parts
        contract = inventory.coordinate_contract
        overlays = tuple(couplings)
        if not all(
            isinstance(overlay, coupling_owner.MeshCoupling) for overlay in overlays
        ):
            raise TypeError(
                f"MeshAssembly.couplings must contain MeshCoupling values; got {tuple(type(overlay).__name__ for overlay in overlays)!r}"
            )
        if len({overlay.coupling_id for overlay in overlays}) != len(overlays):
            raise ValueError(
                f"MeshAssembly.couplings must have unique coupling IDs; got {tuple(overlay.coupling_id for overlay in overlays)!r}"
            )
        by_name = {part.name: part for part in values}
        for overlay in overlays:
            for scope in (overlay.source_scope, overlay.target_scope):
                if scope.source_id not in by_name:
                    raise ValueError(
                        f"MeshAssembly.couplings must reference assembly part names; got {scope.source_id!r}"
                    )
                by_name[scope.source_id].require_scope(scope)
        ownership = _prepare_overset_ownership(overlays, by_name)
        _require_donor_ownership(overlays, ownership)
        overlays = tuple(sorted(overlays, key=lambda overlay: overlay.coupling_id))
        self.parts = values
        self.couplings = overlays
        self.coordinate_contract = contract
        self.assembly_id = canonical_fingerprint(
            {
                "kind": "mesh-assembly",
                "parts": [part.part_id for part in values],
                "couplings": [overlay.coupling_id for overlay in overlays],
                "coordinates": contract.spatial_id,
            }
        )

    def part(self, name: str, /) -> mesh_part_owner.MeshPart:
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(f"MeshAssembly.part must select a registered name; got {name!r}")

    @checked
    def require_attachment(
        self, attachment: interface_binding_owner.MeshInterfaceAttachment, /
    ) -> mesh_part_owner.MeshPart:
        """Return the current part of a revision-bound interface attachment."""
        for part in self.parts:
            if part.name == attachment.part_name:
                attachment.require_current(part)
                return part
        raise ValueError(
            f"MeshAssembly.attachment must reference a registered part; got {attachment.part_name!r}"
        )


__all__ = ["MeshAssembly"]
