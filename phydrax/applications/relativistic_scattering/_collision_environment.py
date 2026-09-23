#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...particle_physics import ParticleEventBatch


class CollisionBundleStatus(IntEnum):
    SUCCESS = 0
    PILEUP_CAPACITY_EXHAUSTED = 1
    EMPTY_PILEUP_POOL = 2
    INVALID_PRIMARY_EVENT = 3


class CollisionEnvironmentPlan(StrictModule, NonTrainableState):
    """Static pileup, luminosity, and beam-background composition contract."""

    mean_pileup: Array
    instantaneous_luminosity: Array
    bunch_spacing: Array
    maximum_pileup: int = eqx.field(static=True)
    luminosity_unit: str = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    pileup_profile_id: str = eqx.field(static=True)
    beam_background_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean_pileup: float,
        /,
        *,
        maximum_pileup: int,
        instantaneous_luminosity: float,
        bunch_spacing: float,
        luminosity_unit: str,
        time_unit: str,
        pileup_profile_id: str,
        beam_background_ids: Sequence[str] = (),
    ):
        mean = float(mean_pileup)
        maximum = int(maximum_pileup)
        luminosity = float(instantaneous_luminosity)
        spacing = float(bunch_spacing)
        luminosity_unit_ = str(luminosity_unit).strip()
        time_unit_ = str(time_unit).strip()
        profile = str(pileup_profile_id).strip()
        backgrounds = tuple(str(value).strip() for value in beam_background_ids)
        if not all(math.isfinite(value) for value in (mean, luminosity, spacing)):
            raise ValueError("Collision environment values must be finite.")
        if mean < 0.0 or maximum < 0 or luminosity < 0.0 or spacing <= 0.0:
            raise ValueError(
                "Pileup/luminosity must be nonnegative and bunch spacing positive."
            )
        if not luminosity_unit_ or not time_unit_ or not profile:
            raise ValueError("Collision environment units and profile ID are required.")
        if any(not value for value in backgrounds) or len(set(backgrounds)) != len(
            backgrounds
        ):
            raise ValueError("beam_background_ids must be distinct non-empty values.")
        self.mean_pileup = jnp.asarray(mean)
        self.instantaneous_luminosity = jnp.asarray(luminosity)
        self.bunch_spacing = jnp.asarray(spacing)
        self.maximum_pileup = maximum
        self.luminosity_unit = luminosity_unit_
        self.time_unit = time_unit_
        self.pileup_profile_id = profile
        self.beam_background_ids = backgrounds
        self.plan_id = canonical_fingerprint(
            {
                "kind": "collision-environment-plan",
                "mean_pileup": mean,
                "maximum_pileup": maximum,
                "instantaneous_luminosity": luminosity,
                "bunch_spacing": spacing,
                "luminosity_unit": luminosity_unit_,
                "time_unit": time_unit_,
                "pileup_profile": profile,
                "beam_backgrounds": list(backgrounds),
            }
        )


class CollisionEventBundle(StrictModule, NonTrainableState):
    """Primary and pileup truth remain separate, related by bounded stable indices."""

    primary: ParticleEventBatch
    pileup_pool: ParticleEventBatch
    pileup_indices: Array
    pileup_active: Array
    requested_pileup: Array
    overflow: Array
    valid: Array
    status: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.primary.event_active)


def assign_collision_pileup(
    plan: CollisionEnvironmentPlan,
    primary: ParticleEventBatch,
    pileup_pool: ParticleEventBatch,
    key: Key[Array, ""],
    /,
) -> CollisionEventBundle:
    """Assign pileup with event-ID-folded keys, independent of batch ordering."""
    if not isinstance(plan, CollisionEnvironmentPlan):
        raise TypeError("plan must be CollisionEnvironmentPlan.")
    if not isinstance(primary, ParticleEventBatch) or not isinstance(
        pileup_pool, ParticleEventBatch
    ):
        raise TypeError("primary and pileup_pool must be ParticleEventBatch values.")
    event_capacity = primary.event_ids.shape[0]
    pool_capacity = pileup_pool.event_ids.shape[0]
    active_pool_indices = jnp.nonzero(
        pileup_pool.event_active & pileup_pool.valid,
        size=pool_capacity,
        fill_value=0,
    )[0]
    active_pool_count = jnp.sum(
        pileup_pool.event_active & pileup_pool.valid, dtype=jnp.int32
    )

    def draw(event_id):
        identifier = jnp.asarray(event_id, dtype=jnp.uint64)
        low = identifier.astype(jnp.uint32)
        high = (identifier >> jnp.asarray(32, dtype=jnp.uint64)).astype(jnp.uint32)
        event_key = jr.fold_in(jr.fold_in(key, high), low)
        count_key, index_key = jr.split(event_key)
        requested = jr.poisson(count_key, plan.mean_pileup).astype(jnp.int32)
        safe_pool_count = jnp.maximum(active_pool_count, 1)
        selections = jr.randint(
            index_key,
            (plan.maximum_pileup,),
            0,
            safe_pool_count,
            dtype=jnp.int32,
        )
        return requested, active_pool_indices[selections]

    requested, indices = jax.vmap(draw)(primary.event_ids)
    active = jnp.arange(plan.maximum_pileup)[None, :] < jnp.minimum(
        requested[:, None], plan.maximum_pileup
    )
    active = active & primary.event_active[:, None]
    overflow = (requested > plan.maximum_pileup) & primary.event_active
    pool_missing = (active_pool_count == 0) & (requested > 0) & primary.event_active
    primary_invalid = primary.event_active & ~primary.valid
    valid = primary.event_active & ~overflow & ~pool_missing & ~primary_invalid
    status = jnp.where(
        overflow,
        int(CollisionBundleStatus.PILEUP_CAPACITY_EXHAUSTED),
        jnp.where(
            pool_missing,
            int(CollisionBundleStatus.EMPTY_PILEUP_POOL),
            jnp.where(
                primary_invalid,
                int(CollisionBundleStatus.INVALID_PRIMARY_EVENT),
                int(CollisionBundleStatus.SUCCESS),
            ),
        ),
    )
    return CollisionEventBundle(
        primary,
        pileup_pool,
        indices,
        active,
        requested,
        overflow,
        valid,
        status.astype(jnp.int32),
        jnp.zeros((event_capacity,), dtype=jnp.bool_),
        plan.plan_id,
    )


__all__ = [
    "CollisionBundleStatus",
    "CollisionEnvironmentPlan",
    "CollisionEventBundle",
    "assign_collision_pileup",
]
