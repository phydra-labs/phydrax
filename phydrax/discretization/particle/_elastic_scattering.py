#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ElasticPairScatteringResult(StrictModule, NonTrainableState):
    """Elastic pair update with per-pair invariant and validity evidence."""

    first_velocity: Array
    second_velocity: Array
    relative_speed: Array
    scattered: Array
    momentum_before: Array
    momentum_after: Array
    momentum_defect: Array
    kinetic_energy_before: Array
    kinetic_energy_after: Array
    kinetic_energy_defect: Array
    finite: Array
    mass_valid: Array
    direction_valid: Array
    conservative: Array
    successful: Array


def scatter_elastic_pairs(
    first_velocity: ArrayLike,
    second_velocity: ArrayLike,
    first_mass: ArrayLike,
    second_mass: ArrayLike,
    direction: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
) -> ElasticPairScatteringResult:
    """Scatter unequal-mass pairs while preserving center-of-mass invariants.

    ``direction`` supplies the post-collision direction of ``v_first-v_second``.
    It is normalized internally, so its magnitude cannot inject kinetic energy.
    Unmasked pairs are returned exactly unchanged and do not require a valid
    direction. Masses are physical scalar masses over the velocity leading axes.
    """

    first = jnp.asarray(first_velocity)
    if not jnp.issubdtype(first.dtype, jnp.inexact):
        first = first.astype(jnp.float32)
    second = jnp.asarray(second_velocity, dtype=first.dtype)
    direction_ = jnp.asarray(direction, dtype=first.dtype)
    if first.ndim < 1 or second.shape != first.shape or direction_.shape != first.shape:
        raise ValueError(
            "Pair velocities and directions must have one matching vector shape."
        )
    leading_shape = first.shape[:-1]
    first_mass_ = jnp.asarray(first_mass, dtype=first.dtype)
    second_mass_ = jnp.asarray(second_mass, dtype=first.dtype)
    first_mass_ = jnp.broadcast_to(first_mass_, leading_shape)
    second_mass_ = jnp.broadcast_to(second_mass_, leading_shape)
    selected = (
        jnp.ones(leading_shape, dtype=jnp.bool_)
        if mask is None
        else jnp.asarray(mask, dtype=jnp.bool_)
    )
    selected = jnp.broadcast_to(selected, leading_shape)

    mass_valid = (
        jnp.isfinite(first_mass_)
        & jnp.isfinite(second_mass_)
        & (first_mass_ > 0.0)
        & (second_mass_ > 0.0)
    )
    safe_first_mass = jnp.where(mass_valid, first_mass_, 1.0)
    safe_second_mass = jnp.where(mass_valid, second_mass_, 1.0)
    total_mass = safe_first_mass + safe_second_mass
    relative = first - second
    relative_speed = jnp.sqrt(ein.contract("...i,...i->...", relative, relative))
    direction_norm = jnp.sqrt(ein.contract("...i,...i->...", direction_, direction_))
    positive_speed = relative_speed > 0.0
    direction_valid = jnp.all(jnp.isfinite(direction_), axis=-1) & (
        (direction_norm > 0.0) | ~positive_speed
    )
    safe_direction_norm = jnp.where(direction_norm > 0.0, direction_norm, 1.0)
    unit_direction = jnp.where(
        direction_valid[..., None], direction_ / safe_direction_norm[..., None], 0.0
    )

    center = (
        safe_first_mass[..., None] * first + safe_second_mass[..., None] * second
    ) / total_mass[..., None]
    scattered_relative = relative_speed[..., None] * unit_direction
    candidate_first = (
        center + (safe_second_mass / total_mass)[..., None] * scattered_relative
    )
    candidate_second = (
        center - (safe_first_mass / total_mass)[..., None] * scattered_relative
    )
    pair_valid = mass_valid & direction_valid
    applied = selected & pair_valid
    first_after = jnp.where(applied[..., None], candidate_first, first)
    second_after = jnp.where(applied[..., None], candidate_second, second)

    momentum_before = (
        safe_first_mass[..., None] * first + safe_second_mass[..., None] * second
    )
    momentum_after = (
        safe_first_mass[..., None] * first_after
        + safe_second_mass[..., None] * second_after
    )
    kinetic_before = 0.5 * (
        safe_first_mass * ein.contract("...i,...i->...", first, first)
        + safe_second_mass * ein.contract("...i,...i->...", second, second)
    )
    kinetic_after = 0.5 * (
        safe_first_mass * ein.contract("...i,...i->...", first_after, first_after)
        + safe_second_mass * ein.contract("...i,...i->...", second_after, second_after)
    )
    momentum_defect = jnp.where(
        selected[..., None], momentum_after - momentum_before, 0.0
    )
    kinetic_defect = jnp.where(selected, kinetic_after - kinetic_before, 0.0)
    finite = (
        jnp.all(jnp.isfinite(first_after), axis=-1)
        & jnp.all(jnp.isfinite(second_after), axis=-1)
        & jnp.all(jnp.isfinite(momentum_defect), axis=-1)
        & jnp.isfinite(kinetic_defect)
        & jnp.isfinite(relative_speed)
    )
    epsilon = jnp.finfo(first.dtype).eps
    tiny = jnp.finfo(first.dtype).tiny
    first_momentum_norm = jnp.sqrt(
        ein.contract(
            "...i,...i->...",
            safe_first_mass[..., None] * first,
            safe_first_mass[..., None] * first,
        )
    )
    second_momentum_norm = jnp.sqrt(
        ein.contract(
            "...i,...i->...",
            safe_second_mass[..., None] * second,
            safe_second_mass[..., None] * second,
        )
    )
    momentum_scale = jnp.maximum(
        first_momentum_norm + second_momentum_norm,
        tiny,
    )
    energy_scale = jnp.maximum(jnp.abs(kinetic_before), tiny)
    conservative = (
        jnp.sqrt(ein.contract("...i,...i->...", momentum_defect, momentum_defect))
        <= 512.0 * epsilon * momentum_scale
    ) & (jnp.abs(kinetic_defect) <= 512.0 * epsilon * energy_scale)
    successful = (~selected) | (pair_valid & finite & conservative)
    return ElasticPairScatteringResult(
        first_after,
        second_after,
        relative_speed,
        applied,
        momentum_before,
        momentum_after,
        momentum_defect,
        kinetic_before,
        kinetic_after,
        kinetic_defect,
        finite,
        mass_valid,
        direction_valid,
        conservative,
        successful,
    )


__all__ = ["ElasticPairScatteringResult", "scatter_elastic_pairs"]
