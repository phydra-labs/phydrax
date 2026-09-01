#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._contact import AbstractMPMFrictionPlan, SharpCoulombMPMFrictionPlan


class MPMMaterialBankEntry(StrictModule, NonTrainableState):
    material: Any
    particle_indices: Array
    entry_id: str = eqx.field(static=True)

    def __init__(self, material: Any, particle_indices: ArrayLike, /, *, entry_id: str):
        indices = np.asarray(particle_indices, dtype=np.int32)
        identifier = str(entry_id)
        if indices.ndim != 1 or np.any(indices < 0) or not identifier:
            raise ValueError("MPM material-bank entry is invalid.")
        self.material = material
        self.particle_indices = jnp.asarray(indices)
        self.entry_id = canonical_fingerprint(
            {
                "kind": "mpm-material-bank-entry",
                "declared_id": identifier,
                "material": material.plan_id,
                "particle_indices": array_tree_fingerprint(indices),
            }
        )


class MPMMaterialBank(StrictModule, NonTrainableState):
    entries: tuple[MPMMaterialBankEntry, ...]
    bank_id: str = eqx.field(static=True)

    def __init__(self, entries: Sequence[MPMMaterialBankEntry], /):
        entries_ = tuple(entries)
        if not entries_ or any(
            not isinstance(entry, MPMMaterialBankEntry) for entry in entries_
        ):
            raise TypeError("MPM material bank requires non-empty typed entries.")
        all_indices = np.concatenate(
            tuple(np.asarray(entry.particle_indices) for entry in entries_)
        )
        if np.unique(all_indices).size != all_indices.size:
            raise ValueError("MPM material-bank selections must be disjoint.")
        self.entries = entries_
        self.bank_id = canonical_fingerprint(
            {"kind": "mpm-material-bank", "entries": [e.entry_id for e in entries_]}
        )


class MPMMaterialBankState(StrictModule):
    histories: tuple[Array, ...]


class MPMNodalFieldPlan(StrictModule, NonTrainableState):
    field_ids: tuple[str, ...] = eqx.field(static=True)
    initial_particle_field_slots: Array
    contact_friction: AbstractMPMFrictionPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_ids: Sequence[str],
        particle_field_slots: ArrayLike,
        /,
        *,
        contact_friction: AbstractMPMFrictionPlan | None = None,
    ):
        ids = tuple(str(value) for value in field_ids)
        slots = np.asarray(particle_field_slots, dtype=np.int32)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(not value for value in ids)
            or slots.ndim != 1
            or np.any(slots < 0)
            or np.any(slots >= len(ids))
        ):
            raise ValueError("MPM nodal field plan is invalid.")
        if contact_friction is not None and not isinstance(
            contact_friction, AbstractMPMFrictionPlan
        ):
            raise TypeError("contact_friction must be AbstractMPMFrictionPlan or None.")
        self.field_ids = ids
        self.initial_particle_field_slots = jnp.asarray(slots)
        self.contact_friction = contact_friction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-nodal-field-plan",
                "field_ids": ids,
                "particle_field_slots": array_tree_fingerprint(slots),
                "contact_friction": (
                    None if contact_friction is None else contact_friction.plan_id
                ),
            }
        )

    @property
    def field_count(self) -> int:
        return len(self.field_ids)


class MPMMultifieldContactEvidence(StrictModule):
    velocity: Array
    impulse: Array
    contact_mask: Array
    normal: Array
    action_reaction_defect: Array
    dissipation: Array
    successful: Array


def project_two_field_contact(
    mass: ArrayLike,
    velocity: ArrayLike,
    mass_gradient: ArrayLike,
    /,
    *,
    friction: AbstractMPMFrictionPlan | None = None,
    mass_tolerance: float = 0.0,
) -> MPMMultifieldContactEvidence:
    mass_ = jnp.asarray(mass)
    velocity_ = jnp.asarray(velocity)
    gradient = jnp.asarray(mass_gradient)
    if (
        mass_.shape[0] != 2
        or velocity_.shape[0] != 2
        or gradient.shape != velocity_.shape
    ):
        raise ValueError("Initial multifield contact supports exactly two nodal fields.")
    if mass_.shape != velocity_.shape[:-1]:
        raise ValueError("Multifield mass and velocity layouts differ.")
    friction_ = SharpCoulombMPMFrictionPlan(0.0) if friction is None else friction
    if not isinstance(friction_, AbstractMPMFrictionPlan):
        raise TypeError("friction must be AbstractMPMFrictionPlan or None.")
    occupied = mass_ > float(mass_tolerance)
    both = occupied[0] & occupied[1]
    normalized_gradients = gradient / jnp.where(
        occupied[..., None], mass_[..., None], 1.0
    )
    normal_raw = normalized_gradients[0] - normalized_gradients[1]
    norm = jnp.sqrt(jnp.sum(normal_raw * normal_raw, axis=-1))
    reliable = jnp.isfinite(norm) & (norm > 1.0e-12)
    normal = normal_raw / jnp.where(reliable, norm, 1.0)[..., None]
    relative = velocity_[0] - velocity_[1]
    normal_speed = jnp.sum(relative * normal, axis=-1)
    contact = both & reliable & (normal_speed > 0.0)
    reduced_mass = (
        mass_[0]
        * mass_[1]
        / jnp.where(mass_[0] + mass_[1] > 0.0, mass_[0] + mass_[1], 1.0)
    )
    normal_impulse = reduced_mass * jnp.maximum(normal_speed, 0.0)
    tangential = relative - normal_speed[..., None] * normal
    tangential_speed = jnp.sqrt(jnp.sum(tangential * tangential, axis=-1))
    tangent = (
        tangential / jnp.where(tangential_speed > 0.0, tangential_speed, 1.0)[..., None]
    )
    tangential_impulse = friction_.impulse_magnitude(
        tangential_speed, normal_impulse, reduced_mass
    )
    impulse = (
        -normal_impulse[..., None] * normal - tangential_impulse[..., None] * tangent
    )
    impulse = jnp.where(contact[..., None], impulse, 0.0)
    first = velocity_[0] + impulse / jnp.where(occupied[0], mass_[0], 1.0)[..., None]
    second = velocity_[1] - impulse / jnp.where(occupied[1], mass_[1], 1.0)[..., None]
    next_velocity = jnp.stack((first, second), axis=0)
    total_first = compensated_sum(impulse.reshape((-1, velocity_.shape[-1])), axis=0)
    total_second = compensated_sum((-impulse).reshape((-1, velocity_.shape[-1])), axis=0)
    action_reaction = jnp.linalg.norm(total_first + total_second)
    dissipation = compensated_sum(
        jnp.where(contact, tangential_impulse * tangential_speed, 0.0)
    )
    successful = (
        jnp.all(~both | reliable)
        & jnp.all(jnp.isfinite(next_velocity))
        & jnp.isfinite(action_reaction)
        & jnp.isfinite(dissipation)
    )
    return MPMMultifieldContactEvidence(
        next_velocity,
        impulse,
        contact,
        normal,
        action_reaction,
        dissipation,
        successful,
    )


__all__ = [
    "MPMMaterialBank",
    "MPMMaterialBankEntry",
    "MPMMaterialBankState",
    "MPMMultifieldContactEvidence",
    "MPMNodalFieldPlan",
    "project_two_field_contact",
]
