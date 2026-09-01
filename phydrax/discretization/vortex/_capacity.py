#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._population import VortexPopulationPlan, VortexPopulationState


class VortexCapacityGrowthRequest(StrictModule):
    old_capacity: int = eqx.field(static=True)
    requested_capacity: int = eqx.field(static=True)
    selected_capacity: int = eqx.field(static=True)
    backend_rebuild_required: Array
    admissible: Array
    request_id: str = eqx.field(static=True)


class VortexEpochMigration(StrictModule):
    state: VortexPopulationState
    old_to_new: Array
    new_to_old: Array
    exact_state_residual: Array
    stable_id_residual: Array
    successful: Array
    migration_id: str = eqx.field(static=True)


class VortexCapacityGrowthPlan(StrictModule, NonTrainableState):
    growth_factor: float = eqx.field(static=True)
    maximum_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, /, *, growth_factor: float = 2.0, maximum_capacity: int):
        if (
            not math.isfinite(float(growth_factor))
            or float(growth_factor) <= 1.0
            or int(maximum_capacity) <= 0
        ):
            raise ValueError("Vortex capacity growth controls are invalid.")
        self.growth_factor, self.maximum_capacity = (
            float(growth_factor),
            int(maximum_capacity),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vortex-capacity-growth",
                "growth_factor": self.growth_factor,
                "maximum_capacity": self.maximum_capacity,
            }
        )

    def request(
        self, old_capacity: int, requested_capacity: int, /
    ) -> VortexCapacityGrowthRequest:
        old, requested = int(old_capacity), int(requested_capacity)
        selected = max(requested, int(math.ceil(old * self.growth_factor)))
        selected = min(selected, self.maximum_capacity)
        admissible = requested > old and selected >= requested
        return VortexCapacityGrowthRequest(
            old,
            requested,
            selected,
            jnp.asarray(admissible),
            jnp.asarray(admissible),
            canonical_fingerprint(
                {
                    "kind": "vortex-capacity-growth-request",
                    "plan": self.plan_id,
                    "old": old,
                    "requested": requested,
                    "selected": selected,
                }
            ),
        )

    def migrate(
        self,
        old_plan: VortexPopulationPlan,
        state: VortexPopulationState,
        request: VortexCapacityGrowthRequest,
        /,
    ) -> tuple[VortexPopulationPlan, VortexEpochMigration]:
        if (
            not isinstance(old_plan, VortexPopulationPlan)
            or old_plan.capacity != request.old_capacity
            or state.positions.shape[0] != old_plan.capacity
        ):
            raise ValueError("Vortex capacity migration inputs are incompatible.")
        if not bool(request.admissible):
            raise ValueError("Vortex capacity request is inadmissible.")
        new_plan = VortexPopulationPlan(
            request.selected_capacity,
            old_plan.dimension,
            journal_capacity=old_plan.journal_capacity,
        )
        padding = request.selected_capacity - old_plan.capacity

        def pad(value, fill):
            return jnp.pad(
                value,
                ((0, padding),) + ((0, 0),) * (value.ndim - 1),
                constant_values=fill,
            )

        migrated = VortexPopulationState(
            pad(state.positions, 0.0),
            pad(state.strength, 0.0),
            pad(state.core_radius, 1.0),
            pad(state.volume, 1.0),
            pad(state.active_mask, False),
            pad(state.stable_ids, -1),
            pad(state.parent_ids, -1),
            pad(state.source_codes, 0),
            pad(state.age, 0.0),
            state.next_stable_id,
        )
        old_to_new = jnp.arange(old_plan.capacity, dtype=jnp.int32)
        new_to_old = jnp.concatenate(
            (old_to_new, jnp.full((padding,), -1, dtype=jnp.int32))
        )
        numeric_residual = jnp.max(
            jnp.abs(migrated.positions[: old_plan.capacity] - state.positions)
        ) + jnp.max(jnp.abs(migrated.strength[: old_plan.capacity] - state.strength))
        id_residual = jnp.max(
            jnp.abs(migrated.stable_ids[: old_plan.capacity] - state.stable_ids)
        )
        successful = (
            (numeric_residual == 0.0)
            & (id_residual == 0)
            & request.backend_rebuild_required
        )
        migration = VortexEpochMigration(
            migrated,
            old_to_new,
            new_to_old,
            numeric_residual,
            id_residual,
            successful,
            canonical_fingerprint(
                {
                    "kind": "vortex-epoch-migration",
                    "growth": request.request_id,
                    "old_plan": old_plan.plan_id,
                    "new_plan": new_plan.plan_id,
                }
            ),
        )
        return new_plan, migration


__all__ = [
    "VortexCapacityGrowthPlan",
    "VortexCapacityGrowthRequest",
    "VortexEpochMigration",
]
