#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum, StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticSystemPlan, MolecularTopologyPlan
from ._recipes import PolymerConstructionResult, RealizedPolymerConnectionPort


class PolymerReactionKind(StrEnum):
    CURE = "cure"
    REPAIR = "repair"


class PolymerReactionStatus(IntEnum):
    ACCEPTED = 0
    UNKNOWN_PORT = 1
    EXHAUSTED_PORT = 2
    INCOMPATIBLE_PORT = 3
    DUPLICATE_BOND = 4
    INTRAMOLECULAR_REJECTED = 5
    DISTANCE_REJECTED = 6
    PERIODIC_WINDING_REQUIRED = 7
    INVALID_PERIODIC_WINDING = 8
    MISSING_BOND = 9


class PolymerReactionTemplate(StrictModule, NonTrainableState):
    template_id: str = eqx.field(static=True)
    left_compatibility_class: str = eqx.field(static=True)
    right_compatibility_class: str = eqx.field(static=True)
    bond_type_id: int = eqx.field(static=True)
    reaction_kind: PolymerReactionKind = eqx.field(static=True)
    allow_intramolecular: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        template_id: str,
        left_compatibility_class: str,
        right_compatibility_class: str,
        /,
        *,
        reaction_kind: PolymerReactionKind = PolymerReactionKind.CURE,
        bond_type_id: int = 0,
        allow_intramolecular: bool = False,
    ):
        identifier = str(template_id).strip()
        left = str(left_compatibility_class).strip()
        right = str(right_compatibility_class).strip()
        bond_type = int(bond_type_id)
        if not isinstance(reaction_kind, PolymerReactionKind):
            raise TypeError("reaction_kind must be PolymerReactionKind.")
        if not identifier or not left or not right or bond_type < 0:
            raise ValueError("Polymer reaction-template definition is invalid.")
        self.template_id = identifier
        self.left_compatibility_class = left
        self.right_compatibility_class = right
        self.reaction_kind = reaction_kind
        self.bond_type_id = bond_type
        self.allow_intramolecular = bool(allow_intramolecular)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polymer-reaction-template",
                "template_id": identifier,
                "left_compatibility_class": left,
                "right_compatibility_class": right,
                "bond_type_id": bond_type,
                "reaction_kind": reaction_kind.value,
                "allow_intramolecular": self.allow_intramolecular,
            }
        )


class PolymerReactionEvent(StrictModule, NonTrainableState):
    event_index: int = eqx.field(static=True)
    template_id: str = eqx.field(static=True)
    reaction_kind: PolymerReactionKind = eqx.field(static=True)
    left_port_id: str = eqx.field(static=True)
    right_port_id: str = eqx.field(static=True)
    left_particle_id: int = eqx.field(static=True)
    right_particle_id: int = eqx.field(static=True)
    image_shift: tuple[int, ...] = eqx.field(static=True)
    status: PolymerReactionStatus = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        event_index: int,
        template_id: str,
        reaction_kind: PolymerReactionKind,
        left_port_id: str,
        right_port_id: str,
        left_particle_id: int,
        right_particle_id: int,
        image_shift: tuple[int, ...],
        status: PolymerReactionStatus,
        /,
    ):
        self.event_index = int(event_index)
        self.template_id = str(template_id)
        self.reaction_kind = PolymerReactionKind(reaction_kind)
        self.left_port_id = str(left_port_id)
        self.right_port_id = str(right_port_id)
        self.left_particle_id = int(left_particle_id)
        self.right_particle_id = int(right_particle_id)
        self.image_shift = tuple(int(value) for value in image_shift)
        self.status = PolymerReactionStatus(status)
        self.accepted = self.status is PolymerReactionStatus.ACCEPTED
        self.event_id = canonical_fingerprint(
            {
                "kind": "polymer-reaction-event",
                "event_index": self.event_index,
                "template_id": self.template_id,
                "reaction_kind": self.reaction_kind.value,
                "left_port_id": self.left_port_id,
                "right_port_id": self.right_port_id,
                "left_particle_id": self.left_particle_id,
                "right_particle_id": self.right_particle_id,
                "image_shift": list(self.image_shift),
                "status": int(self.status),
            }
        )


class PolymerReactionState(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    ports: tuple[RealizedPolymerConnectionPort, ...]
    ledger: tuple[PolymerReactionEvent, ...]
    image_counts: Array
    initial_port_capacity: int = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        ports: tuple[RealizedPolymerConnectionPort, ...],
        ledger: tuple[PolymerReactionEvent, ...],
        image_counts: ArrayLike,
        initial_port_capacity: int,
        /,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        port_values = tuple(ports)
        event_values = tuple(ledger)
        rank = 0 if system.cell is None else int(system.cell.vectors.shape[0])
        images = np.asarray(image_counts, dtype=np.int32)
        if (
            any(
                not isinstance(port, RealizedPolymerConnectionPort)
                for port in port_values
            )
            or any(not isinstance(event, PolymerReactionEvent) for event in event_values)
            or images.shape != (system.particle_ids.size, rank)
            or int(initial_port_capacity) <= 0
        ):
            raise ValueError("Polymer reaction-state content is invalid.")
        self.system = system
        self.ports = port_values
        self.ledger = event_values
        self.image_counts = jnp.asarray(images)
        self.initial_port_capacity = int(initial_port_capacity)
        self.state_id = canonical_fingerprint(
            {
                "kind": "polymer-reaction-state",
                "system": system.system_id,
                "ports": [
                    {
                        "id": port.port_id,
                        "particle": port.particle_id,
                        "uses": port.uses,
                        "maximum": port.maximum_uses,
                    }
                    for port in port_values
                ],
                "ledger": [event.event_id for event in event_values],
                "image_counts": np.asarray(images).tolist(),
                "initial_port_capacity": self.initial_port_capacity,
            }
        )


class PolymerReactionResult(StrictModule, NonTrainableState):
    state: PolymerReactionState
    event: PolymerReactionEvent
    successful: bool = eqx.field(static=True)


def initialize_polymer_reaction_state(
    construction: PolymerConstructionResult,
    /,
    *,
    image_counts: ArrayLike | None = None,
) -> PolymerReactionState:
    if not isinstance(construction, PolymerConstructionResult):
        raise TypeError("construction must be PolymerConstructionResult.")
    system = construction.system
    rank = 0 if system.cell is None else int(system.cell.vectors.shape[0])
    images = (
        np.zeros((system.particle_ids.size, rank), dtype=np.int32)
        if image_counts is None
        else np.asarray(image_counts, dtype=np.int32)
    )
    capacity = sum(port.maximum_uses for port in construction.ports)
    if capacity <= 0:
        raise ValueError("Reaction state requires at least one declared connection port.")
    return PolymerReactionState(system, construction.ports, (), images, capacity)


def _replacement_system(system: AtomisticSystemPlan, topology: MolecularTopologyPlan, /):
    return AtomisticSystemPlan(
        system.particle_ids,
        system.atomic_numbers,
        system.masses,
        system.units,
        atom_type_ids=system.atom_type_ids,
        element_mask=system.element_mask,
        charges=system.charges,
        active_mask=system.active_mask,
        mobile_mask=system.mobile_mask,
        molecule_ids=system.molecule_ids,
        region_ids=system.region_ids,
        topology=topology,
        cell=system.cell,
        coordinate_map=system.coordinate_map,
        name=system.name,
        coordinate_dtype=system.coordinate_dtype,
    )


def apply_polymer_reaction(
    state: PolymerReactionState,
    template: PolymerReactionTemplate,
    left_port_id: str,
    right_port_id: str,
    /,
    *,
    positions: ArrayLike | None = None,
    maximum_distance: float | None = None,
    periodic_image_shift: tuple[int, ...] | None = None,
) -> PolymerReactionResult:
    if not isinstance(state, PolymerReactionState):
        raise TypeError("state must be PolymerReactionState.")
    if not isinstance(template, PolymerReactionTemplate):
        raise TypeError("template must be PolymerReactionTemplate.")
    ports = {port.port_id: port for port in state.ports}
    left = ports.get(str(left_port_id))
    right = ports.get(str(right_port_id))
    status = PolymerReactionStatus.ACCEPTED
    if left is None or right is None:
        status = PolymerReactionStatus.UNKNOWN_PORT
        left_particle = -1 if left is None else left.particle_id
        right_particle = -1 if right is None else right.particle_id
    else:
        left_particle = left.particle_id
        right_particle = right.particle_id
        if template.reaction_kind is PolymerReactionKind.CURE:
            unavailable = not left.available or not right.available
        else:
            unavailable = left.uses <= 0 or right.uses <= 0
        if unavailable:
            status = PolymerReactionStatus.EXHAUSTED_PORT
        compatible = (
            left.compatibility_class == template.left_compatibility_class
            and right.compatibility_class == template.right_compatibility_class
        ) or (
            left.compatibility_class == template.right_compatibility_class
            and right.compatibility_class == template.left_compatibility_class
        )
        if status is PolymerReactionStatus.ACCEPTED and not compatible:
            status = PolymerReactionStatus.INCOMPATIBLE_PORT
    system = state.system
    ids = np.asarray(system.particle_ids, dtype=np.int64)
    slot_by_id = {int(value): index for index, value in enumerate(ids)}
    shift: tuple[int, ...] = ()
    updated_images = np.asarray(state.image_counts).copy()
    if status is PolymerReactionStatus.ACCEPTED:
        left_slot = slot_by_id[left_particle]
        right_slot = slot_by_id[right_particle]
        same_molecule = int(system.molecule_ids[left_slot]) == int(
            system.molecule_ids[right_slot]
        )
        if same_molecule and not template.allow_intramolecular:
            status = PolymerReactionStatus.INTRAMOLECULAR_REJECTED
        existing = {
            tuple(sorted((int(a), int(b))))
            for a, b in np.asarray(system.topology.bonds, dtype=np.int64)
        }
        pair_exists = tuple(sorted((left_particle, right_particle))) in existing
        if template.reaction_kind is PolymerReactionKind.CURE and pair_exists:
            status = PolymerReactionStatus.DUPLICATE_BOND
        elif template.reaction_kind is PolymerReactionKind.REPAIR and not pair_exists:
            status = PolymerReactionStatus.MISSING_BOND
    if status is PolymerReactionStatus.ACCEPTED and system.cell is not None:
        rank = int(system.cell.vectors.shape[0])
        if periodic_image_shift is None:
            status = PolymerReactionStatus.PERIODIC_WINDING_REQUIRED
        else:
            shift = tuple(int(value) for value in periodic_image_shift)
            if len(shift) != rank:
                status = PolymerReactionStatus.INVALID_PERIODIC_WINDING
            else:
                right_slot = slot_by_id[right_particle]
                right_molecule = int(system.molecule_ids[right_slot])
                molecule_mask = np.asarray(system.molecule_ids) == right_molecule
                updated_images[molecule_mask] += np.asarray(shift, dtype=np.int32)
    elif status is PolymerReactionStatus.ACCEPTED:
        if periodic_image_shift not in (None, ()):
            status = PolymerReactionStatus.INVALID_PERIODIC_WINDING
    if status is PolymerReactionStatus.ACCEPTED and maximum_distance is not None:
        maximum = float(maximum_distance)
        if not math.isfinite(maximum) or maximum <= 0.0 or positions is None:
            raise ValueError("A positive distance gate requires positions.")
        coordinates = np.asarray(positions, dtype=float)
        if coordinates.shape != (ids.size, 3):
            raise ValueError("positions must match the atomistic system capacity.")
        if system.cell is not None:
            coordinates = coordinates + np.asarray(updated_images) @ np.asarray(
                system.cell.vectors
            )
        distance = np.linalg.norm(
            coordinates[slot_by_id[left_particle]]
            - coordinates[slot_by_id[right_particle]]
        )
        if not np.isfinite(distance) or distance > maximum:
            status = PolymerReactionStatus.DISTANCE_REJECTED
    event = PolymerReactionEvent(
        len(state.ledger),
        template.template_id,
        template.reaction_kind,
        str(left_port_id),
        str(right_port_id),
        left_particle,
        right_particle,
        shift,
        status,
    )
    if status is not PolymerReactionStatus.ACCEPTED:
        refused_state = PolymerReactionState(
            state.system,
            state.ports,
            state.ledger + (event,),
            state.image_counts,
            state.initial_port_capacity,
        )
        return PolymerReactionResult(refused_state, event, False)
    old = system.topology
    old_bonds = np.asarray(old.bonds, dtype=np.int64)
    old_bond_types = np.asarray(old.bond_type_ids, dtype=np.int32)
    if template.reaction_kind is PolymerReactionKind.CURE:
        new_bonds = np.concatenate(
            (
                old_bonds,
                np.asarray([[left_particle, right_particle]], dtype=np.int64),
            ),
            axis=0,
        )
        new_bond_types = np.concatenate(
            (
                old_bond_types,
                np.asarray([template.bond_type_id], dtype=np.int32),
            )
        )
        port_delta = 1
    else:
        canonical = np.sort(old_bonds, axis=1)
        selected = np.all(
            canonical
            == np.sort(
                np.asarray([[left_particle, right_particle]], dtype=np.int64),
                axis=1,
            ),
            axis=1,
        )
        new_bonds = old_bonds[~selected]
        new_bond_types = old_bond_types[~selected]
        port_delta = -1
    topology = MolecularTopologyPlan(
        bonds=new_bonds,
        angles=old.angles,
        torsions=old.torsions,
        impropers=old.impropers,
        constraints=old.constraints,
        constraint_distances=old.constraint_distances,
        pair_exceptions=old.pair_exceptions,
        lennard_jones_scales=old.lennard_jones_scales,
        electrostatic_scales=old.electrostatic_scales,
        bond_type_ids=new_bond_types,
        angle_type_ids=old.angle_type_ids,
        torsion_type_ids=old.torsion_type_ids,
        improper_type_ids=old.improper_type_ids,
    )
    replacement = _replacement_system(system, topology)
    updated_ports = tuple(
        RealizedPolymerConnectionPort(
            port.port_id,
            port.particle_id,
            port.compatibility_class,
            port.maximum_uses,
            port.uses
            + (port_delta if port.port_id in (left.port_id, right.port_id) else 0),
        )
        for port in state.ports
    )
    accepted_state = PolymerReactionState(
        replacement,
        updated_ports,
        state.ledger + (event,),
        updated_images,
        state.initial_port_capacity,
    )
    return PolymerReactionResult(accepted_state, event, True)


__all__ = [
    "PolymerReactionEvent",
    "PolymerReactionResult",
    "PolymerReactionState",
    "PolymerReactionKind",
    "PolymerReactionStatus",
    "PolymerReactionTemplate",
    "apply_polymer_reaction",
    "initialize_polymer_reaction_state",
]
