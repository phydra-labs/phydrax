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
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


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
    contact_plan: object
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_ids: Sequence[str],
        particle_field_slots: ArrayLike,
        /,
        *,
        contact_plan: object = None,
    ):
        from ._contact_kway import KWayMPMContactPlan

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
        if contact_plan is not None:
            if not isinstance(contact_plan, KWayMPMContactPlan):
                raise TypeError("contact_plan must be KWayMPMContactPlan or None.")
            if contact_plan.field_count != len(ids):
                raise ValueError("Contact field count differs from nodal fields.")
        self.field_ids = ids
        self.initial_particle_field_slots = jnp.asarray(slots)
        self.contact_plan = contact_plan
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-nodal-field-plan",
                "field_ids": ids,
                "particle_field_slots": array_tree_fingerprint(slots),
                "contact_plan": (None if contact_plan is None else contact_plan.plan_id),
            }
        )

    @property
    def field_count(self) -> int:
        return len(self.field_ids)


__all__ = [
    "MPMMaterialBank",
    "MPMMaterialBankEntry",
    "MPMMaterialBankState",
    "MPMNodalFieldPlan",
]
