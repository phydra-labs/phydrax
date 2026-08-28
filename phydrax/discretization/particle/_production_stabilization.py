#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization
from ._free_surface import FreeSurfaceState
from ._pairwise import ParticlePairGeometry, ParticlePairRelation, scatter_pair_sum
from ._precision import ParticleExecutionPolicy
from ._smoothing import AbstractSPHSmoothingKernel


class ShockViscositySensorPlan(StrictModule, NonTrainableState):
    minimum_alpha: float = eqx.field(static=True)
    maximum_alpha: float = eqx.field(static=True)
    decay_time: float = eqx.field(static=True)
    trigger_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_alpha: float = 0.0,
        maximum_alpha: float = 1.0,
        decay_time: float = 0.1,
        trigger_scale: float = 1.0,
    ):
        if (
            not 0.0 <= minimum_alpha <= maximum_alpha
            or decay_time <= 0.0
            or trigger_scale <= 0.0
        ):
            raise ValueError("Shock viscosity sensor parameters are invalid.")
        self.minimum_alpha = float(minimum_alpha)
        self.maximum_alpha = float(maximum_alpha)
        self.decay_time = float(decay_time)
        self.trigger_scale = float(trigger_scale)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shock-viscosity-sensor",
                "minimum_alpha": minimum_alpha,
                "maximum_alpha": maximum_alpha,
                "decay_time": decay_time,
                "trigger_scale": trigger_scale,
            }
        )


class ShockViscosityState(StrictModule):
    alpha: Array
    shock_indicator: Array
    shear_limiter: Array
    update_count: Array


def update_shock_viscosity(
    plan: ShockViscositySensorPlan,
    previous: ShockViscosityState,
    divergence: ArrayLike,
    shear_magnitude: ArrayLike,
    step_size: ArrayLike,
    /,
) -> ShockViscosityState:
    divergence_ = jnp.asarray(divergence)
    shear = jnp.asarray(shear_magnitude)
    compression = jnp.maximum(-divergence_, 0.0)
    indicator = plan.trigger_scale * compression
    target = plan.minimum_alpha + (
        plan.maximum_alpha - plan.minimum_alpha
    ) * indicator / (1.0 + indicator)
    decay = jnp.exp(-jnp.asarray(step_size) / plan.decay_time)
    alpha = jnp.maximum(
        target, plan.minimum_alpha + decay * (previous.alpha - plan.minimum_alpha)
    )
    limiter = jnp.abs(divergence_) / (
        jnp.abs(divergence_) + shear + jnp.finfo(divergence_.dtype).eps
    )
    return ShockViscosityState(
        alpha,
        indicator,
        limiter,
        previous.update_count + 1,
    )


class BalsaraLimiterState(StrictModule):
    divergence: Array
    curl_magnitude: Array
    limiter: Array


def balsara_limiter(
    particles: ParticleDiscretization,
    velocity: ArrayLike,
    density: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: float,
    execution: ParticleExecutionPolicy,
    /,
) -> BalsaraLimiterState:
    velocity_ = jnp.asarray(velocity)
    density_ = jnp.asarray(density)
    left = pairs.left_indices
    right = pairs.right_indices
    valid = pairs.valid & jnp.asarray(physical_pairs, dtype=bool)
    gradient = kernel.gradient(geometry.displacement, geometry.distance, smoothing_length)
    difference = velocity_[right] - velocity_[left]
    volume = particles.safe_masses / density_
    divergence_pair = jnp.sum(difference * gradient, axis=-1)
    divergence = scatter_pair_sum(
        pairs,
        volume[right] * divergence_pair,
        volume[left] * divergence_pair,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    if particles.ambient_dimension == 1:
        curl = jnp.zeros_like(divergence)
    elif particles.ambient_dimension == 2:
        cross = difference[:, 0] * gradient[:, 1] - difference[:, 1] * gradient[:, 0]
        curl = jnp.abs(
            scatter_pair_sum(
                pairs,
                volume[right] * cross,
                volume[left] * cross,
                size=particles.capacity,
                accumulation=execution.accumulation,
                valid=valid,
            )
        )
    else:
        cross = jnp.cross(difference, gradient)
        curl_vector = scatter_pair_sum(
            pairs,
            volume[right, None] * cross,
            volume[left, None] * cross,
            size=particles.capacity,
            accumulation=execution.accumulation,
            valid=valid,
        )
        curl = jnp.sqrt(jnp.sum(curl_vector * curl_vector, axis=-1))
    limiter = jnp.abs(divergence) / (
        jnp.abs(divergence) + curl + jnp.finfo(divergence.dtype).eps
    )
    return BalsaraLimiterState(divergence, curl, limiter)


class RenormalizationAudit(StrictModule):
    pressure_jump: Array
    internal_energy_jump: Array
    density_variance_before: Array
    density_variance_after: Array
    maximum_relative_correction: Array
    successful: Array


def audit_density_renormalization(
    material,
    mass: ArrayLike,
    before: ArrayLike,
    after: ArrayLike,
    /,
) -> RenormalizationAudit:
    mass_ = jnp.asarray(mass)
    before_ = jnp.asarray(before)
    after_ = jnp.asarray(after)
    pressure_jump = jnp.max(
        jnp.abs(material.pressure(after_) - material.pressure(before_))
    )
    energy_jump = compensated_sum(
        mass_
        * (
            material.specific_internal_energy(after_)
            - material.specific_internal_energy(before_)
        )
    )
    before_centered = before_ - jnp.mean(before_)
    after_centered = after_ - jnp.mean(after_)
    relative = jnp.abs(after_ - before_) / jnp.maximum(jnp.abs(before_), 1e-14)
    return RenormalizationAudit(
        pressure_jump,
        energy_jump,
        jnp.mean(before_centered**2),
        jnp.mean(after_centered**2),
        jnp.max(relative),
        jnp.all(jnp.isfinite(after_) & (after_ > 0.0)),
    )


class ParticleShiftingPlan(StrictModule, NonTrainableState):
    velocity_scale: float = eqx.field(static=True)
    target_spacing: float = eqx.field(static=True)
    maximum_shift: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity_scale: float,
        target_spacing: float,
        maximum_shift: float,
        /,
    ):
        if velocity_scale < 0.0 or target_spacing <= 0.0 or maximum_shift <= 0.0:
            raise ValueError("Particle shifting parameters are invalid.")
        self.velocity_scale = float(velocity_scale)
        self.target_spacing = float(target_spacing)
        self.maximum_shift = float(maximum_shift)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-shifting-plan",
                "velocity_scale": velocity_scale,
                "target_spacing": target_spacing,
                "maximum_shift": maximum_shift,
            }
        )


class ParticleShiftingResult(StrictModule):
    shifted_position: Array
    shift: Array
    correction_norm: Array
    limited_count: Array
    successful: Array


def particle_shifting(
    plan: ParticleShiftingPlan,
    particles: ParticleDiscretization,
    position: ArrayLike,
    density: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: float,
    execution: ParticleExecutionPolicy,
    /,
    *,
    step_size: ArrayLike,
    free_surface: FreeSurfaceState | None = None,
) -> ParticleShiftingResult:
    position_ = jnp.asarray(position)
    density_ = jnp.asarray(density)
    left = pairs.left_indices
    right = pairs.right_indices
    valid = pairs.valid & jnp.asarray(physical_pairs, dtype=bool)
    gradient = kernel.gradient(geometry.displacement, geometry.distance, smoothing_length)
    volume = particles.safe_masses / density_
    raw_velocity = scatter_pair_sum(
        pairs,
        -plan.velocity_scale * volume[right, None] * gradient,
        plan.velocity_scale * volume[left, None] * gradient,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    shift = jnp.asarray(step_size) * raw_velocity
    if free_surface is not None:
        normal_component = (
            jnp.sum(shift * free_surface.normal, axis=-1)[:, None] * free_surface.normal
        )
        shift = jnp.where(
            free_surface.hard_mask[:, None], shift - normal_component, shift
        )
    magnitude = jnp.sqrt(jnp.sum(shift * shift, axis=-1))
    limited = magnitude > plan.maximum_shift
    shift = (
        shift
        * jnp.minimum(
            1.0, plan.maximum_shift / jnp.where(magnitude > 0.0, magnitude, 1.0)
        )[:, None]
    )
    shifted = position_ + jnp.where(particles.active_mask[:, None], shift, 0.0)
    return ParticleShiftingResult(
        shifted,
        shift,
        jnp.sqrt(jnp.sum(shift * shift)),
        jnp.sum(limited.astype(jnp.int32)),
        jnp.all(jnp.isfinite(shifted)),
    )


class StabilizationCheckpointState(StrictModule, NonTrainableState):
    shock: ShockViscosityState
    renormalization_count: Array
    shifting_count: Array
    cumulative_dissipation: Array
    checkpoint_id: str = eqx.field(static=True)


__all__ = [
    "BalsaraLimiterState",
    "ParticleShiftingPlan",
    "ParticleShiftingResult",
    "RenormalizationAudit",
    "ShockViscositySensorPlan",
    "ShockViscosityState",
    "StabilizationCheckpointState",
    "audit_density_renormalization",
    "balsara_limiter",
    "particle_shifting",
    "update_shock_viscosity",
]
