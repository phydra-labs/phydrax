#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._kinematics import ContactKinematicsEpoch


class ContactRouteMode(IntEnum):
    OPEN = 0
    STICK = 1
    SLIP = 2
    ADHERED = 3
    DEBONDED = 4


class ContactRouteStateDefaults(StrictModule, NonTrainableState):
    mode: ContactRouteMode = eqx.field(static=True)
    rate_state: float = eqx.field(static=True)
    film_thickness: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        mode: ContactRouteMode = ContactRouteMode.OPEN,
        rate_state: float = 1.0,
        film_thickness: float = 0.0,
    ):
        rate = float(rate_state)
        film = float(film_thickness)
        if rate <= 0.0 or film < 0.0:
            raise ValueError(
                "Initial rate-state value must be positive and film thickness nonnegative."
            )
        self.mode = ContactRouteMode(mode)
        self.rate_state = rate
        self.film_thickness = film


class ContactRouteState(StrictModule, NonTrainableState):
    route_keys: Array
    valid: Array
    mode: Array
    accumulated_slip: Array
    adhesion_damage: Array
    wear_depth: Array
    rate_state: Array
    film_thickness: Array
    state_version: Array
    tangent_dimension: int = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    @property
    def capacity(self) -> int:
        return int(self.valid.size)

    @classmethod
    def empty(
        cls,
        capacity: int,
        tangent_dimension: int,
        closure_id: str,
        /,
        *,
        dtype=jnp.float64,
        defaults: ContactRouteStateDefaults | None = None,
    ) -> ContactRouteState:
        count = int(capacity)
        tangent = int(tangent_dimension)
        if count < 0 or tangent not in (1, 2):
            raise ValueError("Contact state capacity/tangent dimension is invalid.")
        defaults_ = ContactRouteStateDefaults() if defaults is None else defaults
        if not isinstance(defaults_, ContactRouteStateDefaults):
            raise TypeError("defaults must be ContactRouteStateDefaults or None.")
        identifier = str(closure_id)
        if not identifier:
            raise ValueError("closure_id must be nonempty.")
        return cls(
            jnp.zeros((count,), dtype=jnp.int64),
            jnp.zeros((count,), dtype=bool),
            jnp.full((count,), int(defaults_.mode), dtype=jnp.int32),
            jnp.zeros((count, tangent), dtype=dtype),
            jnp.zeros((count,), dtype=dtype),
            jnp.zeros((count,), dtype=dtype),
            jnp.full((count,), defaults_.rate_state, dtype=dtype),
            jnp.full((count,), defaults_.film_thickness, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            tangent,
            identifier,
        )


class ContactRouteStateTransition(StrictModule):
    previous: ContactRouteState
    candidate: ContactRouteState
    continued: Array
    created: Array
    disappeared: Array
    duplicate_keys: Array
    finite: Array
    successful: Array
    transition_id: str = eqx.field(static=True)

    def commit(self, /) -> ContactRouteState:
        return self.candidate

    def rollback(self, /) -> ContactRouteState:
        return self.previous


def flatten_contact_routes(epoch: ContactKinematicsEpoch, /) -> tuple[Array, Array]:
    if not isinstance(epoch, ContactKinematicsEpoch):
        raise TypeError("epoch must be ContactKinematicsEpoch.")
    if not epoch.batches:
        return (
            jnp.empty((0,), dtype=jnp.int64),
            jnp.empty((0,), dtype=bool),
        )
    return (
        jnp.concatenate(tuple(batch.route_keys for batch in epoch.batches)),
        jnp.concatenate(tuple(batch.valid for batch in epoch.batches)),
    )


def remap_contact_route_state(
    previous: ContactRouteState,
    epoch: ContactKinematicsEpoch,
    /,
    *,
    defaults: ContactRouteStateDefaults | None = None,
) -> ContactRouteStateTransition:
    if not isinstance(previous, ContactRouteState):
        raise TypeError("previous must be ContactRouteState.")
    defaults_ = ContactRouteStateDefaults() if defaults is None else defaults
    if not isinstance(defaults_, ContactRouteStateDefaults):
        raise TypeError("defaults must be ContactRouteStateDefaults or None.")
    keys, valid = flatten_contact_routes(epoch)
    capacity = int(keys.size)
    dtype = previous.accumulated_slip.dtype
    if previous.capacity == 0:
        candidate = ContactRouteState(
            keys,
            valid,
            jnp.full((capacity,), int(defaults_.mode), dtype=jnp.int32),
            jnp.zeros((capacity, previous.tangent_dimension), dtype=dtype),
            jnp.zeros((capacity,), dtype=dtype),
            jnp.zeros((capacity,), dtype=dtype),
            jnp.full((capacity,), defaults_.rate_state, dtype=dtype),
            jnp.full((capacity,), defaults_.film_thickness, dtype=dtype),
            previous.state_version + 1,
            previous.tangent_dimension,
            previous.closure_id,
        )
        finite = epoch.evidence.successful
        return ContactRouteStateTransition(
            previous,
            candidate,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.sum(valid, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            finite,
            finite,
            canonical_fingerprint(
                {
                    "kind": "contact-route-state-transition",
                    "closure": previous.closure_id,
                    "epoch": epoch.epoch_id,
                    "capacity": capacity,
                }
            ),
        )
    equality = (
        (keys[:, None] == previous.route_keys[None, :])
        & valid[:, None]
        & previous.valid[None, :]
    )
    found = jnp.any(equality, axis=1)
    source = jnp.argmax(equality, axis=1)
    safe_source = jnp.clip(source, 0, max(previous.capacity - 1, 0))
    duplicate_new = jnp.any(
        (keys[:, None] == keys[None, :])
        & valid[:, None]
        & valid[None, :]
        & ~jnp.eye(capacity, dtype=bool),
        axis=1,
    )
    duplicate_old_match = jnp.sum(equality, axis=1) > 1
    duplicate = duplicate_new | duplicate_old_match

    def inherited(old, default):
        selected = old[safe_source]
        default_value = jnp.broadcast_to(
            jnp.asarray(default, dtype=selected.dtype), selected.shape
        )
        condition = found
        while condition.ndim < selected.ndim:
            condition = condition[..., None]
        return jnp.where(condition, selected, default_value)

    candidate = ContactRouteState(
        keys,
        valid,
        inherited(previous.mode, int(defaults_.mode)),
        inherited(
            previous.accumulated_slip,
            jnp.zeros((previous.tangent_dimension,), dtype=dtype),
        ),
        inherited(previous.adhesion_damage, 0.0),
        inherited(previous.wear_depth, 0.0),
        inherited(previous.rate_state, defaults_.rate_state),
        inherited(previous.film_thickness, defaults_.film_thickness),
        previous.state_version + 1,
        previous.tangent_dimension,
        previous.closure_id,
    )
    old_continued = jnp.any(equality, axis=0)
    continued = jnp.sum(found & valid, dtype=jnp.int32)
    created = jnp.sum((~found) & valid, dtype=jnp.int32)
    disappeared = jnp.sum(previous.valid & ~old_continued, dtype=jnp.int32)
    finite = (
        jnp.all(jnp.isfinite(candidate.accumulated_slip))
        & jnp.all(jnp.isfinite(candidate.adhesion_damage))
        & jnp.all(jnp.isfinite(candidate.wear_depth))
        & jnp.all(jnp.isfinite(candidate.rate_state))
        & jnp.all(jnp.isfinite(candidate.film_thickness))
    )
    successful = epoch.evidence.successful & finite & ~jnp.any(duplicate)
    return ContactRouteStateTransition(
        previous,
        candidate,
        continued,
        created,
        disappeared,
        jnp.sum(duplicate, dtype=jnp.int32),
        finite,
        successful,
        canonical_fingerprint(
            {
                "kind": "contact-route-state-transition",
                "closure": previous.closure_id,
                "epoch": epoch.epoch_id,
                "capacity": capacity,
            }
        ),
    )


__all__ = [
    "ContactRouteMode",
    "ContactRouteState",
    "ContactRouteStateDefaults",
    "ContactRouteStateTransition",
    "flatten_contact_routes",
    "remap_contact_route_state",
]
