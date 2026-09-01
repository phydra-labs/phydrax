#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._route_state import ContactRouteMode, ContactRouteState


class ContactStateTransferPlan(StrictModule, NonTrainableState):
    new_route_keys: Array
    parent_route_slots: Array
    parent_weights: Array
    valid: Array
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        new_route_keys: ArrayLike,
        parent_route_slots: ArrayLike,
        parent_weights: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
    ):
        keys = np.asarray(new_route_keys)
        parents = np.asarray(parent_route_slots)
        weights = np.asarray(parent_weights, dtype=float)
        if keys.ndim != 1 or not np.issubdtype(keys.dtype, np.integer):
            raise TypeError("New contact route keys must be one integer vector.")
        if (
            parents.ndim != 2
            or weights.shape != parents.shape
            or parents.shape[0] != keys.size
        ):
            raise ValueError("Contact state parent slots/weights are incompatible.")
        active = (
            np.ones((keys.size,), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if active.shape != (keys.size,):
            raise ValueError("Contact state transfer valid mask has invalid shape.")
        if np.any(~np.isfinite(weights)) or not np.allclose(
            weights[active].sum(axis=1), 1.0
        ):
            raise ValueError("Active contact state transfer weights must be affine.")
        self.new_route_keys = jnp.asarray(keys, dtype=jnp.int64)
        self.parent_route_slots = jnp.asarray(parents, dtype=jnp.int32)
        self.parent_weights = jnp.asarray(weights)
        self.valid = jnp.asarray(active)
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "contact-state-transfer-plan",
                "keys": array_tree_fingerprint(keys),
                "parents": array_tree_fingerprint(parents),
                "weights": array_tree_fingerprint(weights),
                "valid": array_tree_fingerprint(active),
            }
        )


class ContactStateTransferEvidence(StrictModule):
    transferred_routes: Array
    invalid_parents: Array
    maximum_damage_decrease: Array
    maximum_wear_decrease: Array
    finite: Array
    irreversible: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class ContactStateTransferResult(StrictModule):
    state: ContactRouteState
    evidence: ContactStateTransferEvidence


def transfer_contact_route_state(
    plan: ContactStateTransferPlan,
    previous: ContactRouteState,
    /,
) -> ContactStateTransferResult:
    if not isinstance(plan, ContactStateTransferPlan):
        raise TypeError("plan must be ContactStateTransferPlan.")
    if not isinstance(previous, ContactRouteState):
        raise TypeError("previous must be ContactRouteState.")
    parents = plan.parent_route_slots
    parent_valid = (parents >= 0) & (parents < previous.capacity)
    safe = jnp.clip(parents, 0, max(previous.capacity - 1, 0))
    weights = plan.parent_weights.astype(previous.accumulated_slip.dtype)
    normalized = jnp.where(parent_valid, weights, 0.0)
    weight_sum = jnp.sum(normalized, axis=1)
    normalized = normalized / jnp.maximum(
        weight_sum[:, None], jnp.finfo(weights.dtype).eps
    )

    def transfer(field):
        gathered = field[safe]
        local_weights = normalized
        while local_weights.ndim < gathered.ndim:
            local_weights = local_weights[..., None]
        return jnp.sum(local_weights * gathered, axis=1)

    accumulated = transfer(previous.accumulated_slip)
    damage_average = transfer(previous.adhesion_damage)
    wear_average = transfer(previous.wear_depth)
    damage_maximum = jnp.max(
        jnp.where(parent_valid, previous.adhesion_damage[safe], 0.0),
        axis=1,
    )
    wear_maximum = jnp.max(
        jnp.where(parent_valid, previous.wear_depth[safe], 0.0),
        axis=1,
    )
    damage = jnp.maximum(damage_average, damage_maximum)
    wear = jnp.maximum(wear_average, wear_maximum)
    rate_state = transfer(previous.rate_state)
    film = jnp.maximum(transfer(previous.film_thickness), 0.0)
    mode_parent = jnp.max(
        jnp.where(parent_valid, previous.mode[safe], int(ContactRouteMode.OPEN)),
        axis=1,
    )
    route_valid = plan.valid & (weight_sum > 0.0)
    state = ContactRouteState(
        plan.new_route_keys,
        route_valid,
        mode_parent,
        accumulated,
        damage,
        wear,
        rate_state,
        film,
        previous.state_version + 1,
        previous.tangent_dimension,
        previous.closure_id,
    )
    invalid_parents = jnp.sum(plan.valid & (weight_sum <= 0.0), dtype=jnp.int32)
    damage_decrease = jnp.max(jnp.maximum(damage_maximum - damage, 0.0), initial=0.0)
    wear_decrease = jnp.max(jnp.maximum(wear_maximum - wear, 0.0), initial=0.0)
    finite = (
        jnp.all(jnp.isfinite(accumulated))
        & jnp.all(jnp.isfinite(damage))
        & jnp.all(jnp.isfinite(wear))
        & jnp.all(jnp.isfinite(rate_state))
        & jnp.all(jnp.isfinite(film))
    )
    irreversible = (damage_decrease == 0.0) & (wear_decrease == 0.0)
    evidence = ContactStateTransferEvidence(
        jnp.sum(route_valid, dtype=jnp.int32),
        invalid_parents,
        damage_decrease,
        wear_decrease,
        finite,
        irreversible,
        finite & irreversible & (invalid_parents == 0),
        plan.transfer_id,
    )
    return ContactStateTransferResult(state, evidence)


__all__ = [
    "ContactStateTransferEvidence",
    "ContactStateTransferPlan",
    "ContactStateTransferResult",
    "transfer_contact_route_state",
]
