#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class BAOABLangevinPlan(StrictModule, NonTrainableState):
    """Fixed-step BAOAB Langevin splitting in declared atomistic units."""

    step_size: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    friction: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_size: float,
        temperature: float,
        friction: float,
        /,
        *,
        realization_id: int = 0,
    ):
        step = float(step_size)
        thermal = float(temperature)
        damping = float(friction)
        realization = int(realization_id)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(thermal)
            or thermal <= 0.0
            or not math.isfinite(damping)
            or damping <= 0.0
            or realization < 0
        ):
            raise ValueError(
                "BAOAB step, temperature, and friction must be positive finite; "
                "realization_id must be non-negative."
            )
        self.step_size = step
        self.temperature = thermal
        self.friction = damping
        self.realization_id = realization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-baoab-langevin",
                "step_size": step,
                "temperature": thermal,
                "friction": damping,
                "realization_id": realization,
            }
        )


def stable_particle_normals(
    key_data: ArrayLike,
    particle_ids: ArrayLike,
    step_index: ArrayLike,
    /,
    *,
    operator_id: int,
    realization_id: int,
    dtype,
) -> Array:
    """Draw Cartesian normals addressed by stable IDs, step, operator, and path."""

    data = jnp.asarray(key_data, dtype=jnp.uint32)
    if data.shape != (2,):
        raise ValueError("key_data must contain two uint32 words.")
    ids = jnp.asarray(particle_ids, dtype=jnp.int64)
    if ids.ndim != 1:
        raise ValueError("particle_ids must be a vector.")
    key = jr.wrap_key_data(data)
    key = jr.fold_in(key, jnp.asarray(realization_id, dtype=jnp.uint32))
    key = jr.fold_in(key, jnp.asarray(step_index, dtype=jnp.uint32))
    key = jr.fold_in(key, jnp.asarray(operator_id, dtype=jnp.uint32))
    unsigned = ids.astype(jnp.uint64)
    lower = (unsigned & jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32)
    upper = (unsigned >> jnp.uint64(32)).astype(jnp.uint32)

    def particle_key(low, high):
        return jr.fold_in(jr.fold_in(key, high), low)

    keys = jax.vmap(particle_key)(lower, upper)
    return jax.vmap(lambda value: jr.normal(value, (3,), dtype=dtype))(keys)


class ThermostatEvaluation(StrictModule):
    momenta: Array
    heat: Array
    decay: Array
    successful: Array


def apply_baoab_ornstein_uhlenbeck(
    plan: BAOABLangevinPlan,
    key_data: ArrayLike,
    particle_ids: ArrayLike,
    momenta: ArrayLike,
    masses: ArrayLike,
    mobile_mask: ArrayLike,
    step_index: ArrayLike,
    /,
    *,
    boltzmann_constant: float,
    kinetic_to_energy: float,
) -> ThermostatEvaluation:
    if not isinstance(plan, BAOABLangevinPlan):
        raise TypeError("plan must be a BAOABLangevinPlan.")
    momentum = jnp.asarray(momenta)
    mass = jnp.asarray(masses, dtype=momentum.dtype)
    mobile = jnp.asarray(mobile_mask, dtype=bool)
    if momentum.shape != mass.shape + (3,) or mobile.shape != mass.shape:
        raise ValueError("Momentum, mass, and mobile masks have incompatible shapes.")
    decay = jnp.exp(-jnp.asarray(plan.friction * plan.step_size, dtype=momentum.dtype))
    variance = (
        mass
        * (1.0 - decay * decay)
        * (boltzmann_constant * plan.temperature)
        / kinetic_to_energy
    )
    normals = stable_particle_normals(
        key_data,
        particle_ids,
        step_index,
        operator_id=2,
        realization_id=plan.realization_id,
        dtype=momentum.dtype,
    )
    before = (
        0.5 * kinetic_to_energy * jnp.sum(momentum * momentum / mass[:, None], axis=-1)
    )
    proposed = decay * momentum + jnp.sqrt(variance)[:, None] * normals
    proposed = jnp.where(mobile[:, None], proposed, 0.0)
    after = (
        0.5 * kinetic_to_energy * jnp.sum(proposed * proposed / mass[:, None], axis=-1)
    )
    heat = jnp.sum(jnp.where(mobile, after - before, 0.0))
    successful = jnp.all(jnp.isfinite(proposed)) & jnp.isfinite(heat)
    return ThermostatEvaluation(proposed, heat, decay, successful)


__all__ = [
    "BAOABLangevinPlan",
    "ThermostatEvaluation",
    "apply_baoab_ornstein_uhlenbeck",
    "stable_particle_normals",
]
