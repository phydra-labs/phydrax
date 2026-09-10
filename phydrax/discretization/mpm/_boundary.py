#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PrescribedGridVelocityResult(StrictModule):
    velocity: Array
    impulse: Array
    work: Array
    successful: Array


class PrescribedGridVelocityPlan(StrictModule, NonTrainableState):
    """Static component-wise velocity values on one prepared nodal grid."""

    mask: Array
    values: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, mask: ArrayLike, values: ArrayLike = 0.0, /):
        mask_ = np.asarray(mask, dtype=bool)
        values_ = np.asarray(values)
        if mask_.ndim < 2 or mask_.shape[-1] not in (1, 2, 3):
            raise ValueError(
                "Velocity mask must have target shape followed by dimension."
            )
        values_ = np.broadcast_to(values_, mask_.shape)
        if np.any(~np.isfinite(values_)):
            raise ValueError("Prescribed grid velocities must be finite.")
        self.mask = jnp.asarray(mask_)
        self.values = jnp.asarray(values_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prescribed-grid-velocity",
                "mask": array_tree_fingerprint(mask_),
                "values": array_tree_fingerprint(values_),
            }
        )

    def apply(
        self,
        velocity: ArrayLike,
        mass: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> PrescribedGridVelocityResult:
        value = jnp.asarray(velocity)
        mass_ = jnp.asarray(mass)
        if value.shape != self.mask.shape:
            raise ValueError("Grid velocity must match the prescribed boundary layout.")
        if mass_.shape != value.shape[:-1]:
            raise ValueError("Grid mass must match the velocity target shape.")
        dt = jnp.asarray(step_size, dtype=value.dtype)
        prescribed = self.values.astype(value.dtype)
        next_velocity = jnp.where(self.mask, prescribed, value)
        delta = mass_[..., None] * (next_velocity - value)
        impulse = compensated_sum(delta.reshape((-1, value.shape[-1])), axis=0)
        kinetic_change = (
            0.5
            * mass_
            * (
                jnp.sum(next_velocity * next_velocity, axis=-1)
                - jnp.sum(value * value, axis=-1)
            )
        )
        work = compensated_sum(kinetic_change.reshape((-1,)))
        successful = (
            jnp.isfinite(dt)
            & (dt > 0.0)
            & jnp.all(jnp.isfinite(next_velocity))
            & jnp.all(jnp.isfinite(impulse))
            & jnp.isfinite(work)
        )
        return PrescribedGridVelocityResult(
            next_velocity,
            impulse,
            work,
            successful,
        )

    def apply_indexed(
        self,
        velocity: ArrayLike,
        mass: ArrayLike,
        step_size: ArrayLike,
        logical_node_ids: ArrayLike,
        node_valid: ArrayLike,
        /,
    ) -> PrescribedGridVelocityResult:
        """Apply a dense logical boundary declaration on compact nodal storage."""
        value = jnp.asarray(velocity)
        mass_ = jnp.asarray(mass)
        logical = jnp.asarray(logical_node_ids, dtype=jnp.int32)
        valid = jnp.asarray(node_valid, dtype=bool)
        if value.ndim != 2 or value.shape[-1] != self.mask.shape[-1]:
            raise ValueError(
                "Compact grid velocity must have shape (storage, dimension)."
            )
        if mass_.shape != value.shape[:-1] or logical.shape != mass_.shape:
            raise ValueError("Compact mass and logical node IDs must match velocity.")
        if valid.shape != mass_.shape:
            raise ValueError("node_valid must match compact storage capacity.")
        logical_size = int(np.prod(self.mask.shape[:-1], dtype=int))
        if logical_size == 0:
            raise ValueError("Prescribed boundary logical grid cannot be empty.")
        safe = jnp.clip(logical, 0, logical_size - 1)
        mask = self.mask.reshape((logical_size, self.mask.shape[-1]))[safe]
        prescribed = self.values.reshape((logical_size, self.values.shape[-1]))[safe]
        mask = mask & valid[:, None]
        dt = jnp.asarray(step_size, dtype=value.dtype)
        next_velocity = jnp.where(mask, prescribed.astype(value.dtype), value)
        delta = mass_[:, None] * (next_velocity - value)
        impulse = compensated_sum(delta, axis=0)
        kinetic_change = (
            0.5
            * mass_
            * (
                jnp.sum(next_velocity * next_velocity, axis=-1)
                - jnp.sum(value * value, axis=-1)
            )
        )
        work = compensated_sum(kinetic_change)
        successful = (
            jnp.isfinite(dt)
            & (dt > 0.0)
            & jnp.all(jnp.isfinite(next_velocity))
            & jnp.all(jnp.isfinite(impulse))
            & jnp.isfinite(work)
        )
        return PrescribedGridVelocityResult(
            next_velocity,
            impulse,
            work,
            successful,
        )


__all__ = ["PrescribedGridVelocityPlan", "PrescribedGridVelocityResult"]
