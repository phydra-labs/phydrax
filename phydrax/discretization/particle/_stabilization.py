#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ._core import ParticleDiscretization
from ._pairwise import (
    ParticlePairGeometry,
    ParticlePairRelation,
    scatter_pair_exchange,
    scatter_pair_sum,
)
from ._precision import ParticleExecutionPolicy
from ._smoothing import AbstractSPHSmoothingKernel


FreeSurfaceDiffusionPolicy: TypeAlias = Literal[
    "disable", "one-sided-corrected", "smooth-taper"
]
ArtificialViscosityActivation: TypeAlias = Literal[
    "approaching-only", "always", "smooth-approach"
]


class SPHKernelNormalizationState(StrictModule):
    completeness: Array
    inverse_completeness: Array
    deficient_mask: Array


class SPHFirstOrderCorrectionState(StrictModule):
    moment_matrix: Array
    correction_matrix: Array
    condition_estimate: Array
    residual_norm: Array
    successful: Array


class SPHFirstOrderGradientCorrectionPlan(StrictModule, NonTrainableState):
    regularization: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, regularization: float = 1e-10, maximum_condition: float = 1e10):
        if regularization < 0.0 or maximum_condition <= 1.0:
            raise ValueError("SPH correction regularization/condition are invalid.")
        self.regularization = float(regularization)
        self.maximum_condition = float(maximum_condition)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sph-first-order-gradient-correction",
                "regularization": regularization,
                "maximum_condition": maximum_condition,
            }
        )


def sph_kernel_normalization(
    particles: ParticleDiscretization,
    density: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: float,
    execution: ParticleExecutionPolicy,
    /,
    *,
    minimum_completeness: float = 0.2,
) -> SPHKernelNormalizationState:
    density_ = jnp.asarray(density)
    volume = particles.safe_masses / density_
    valid = pairs.valid & jnp.asarray(physical_pairs, dtype=bool)
    weights = kernel.value(geometry.distance, smoothing_length)
    left = pairs.left_indices
    right = pairs.right_indices
    neighbors = scatter_pair_sum(
        pairs,
        volume[right] * weights,
        volume[left] * weights,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    self_value = volume * kernel.value(jnp.asarray(0.0, weights.dtype), smoothing_length)
    completeness = jnp.where(particles.active_mask, self_value + neighbors, 1.0)
    deficient = particles.active_mask & (completeness < minimum_completeness)
    inverse = 1.0 / jnp.where(deficient, 1.0, completeness)
    return SPHKernelNormalizationState(completeness, inverse, deficient)


def sph_first_order_correction(
    plan: SPHFirstOrderGradientCorrectionPlan,
    particles: ParticleDiscretization,
    density: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: float,
    execution: ParticleExecutionPolicy,
    /,
) -> SPHFirstOrderCorrectionState:
    density_ = jnp.asarray(density)
    volume = particles.safe_masses / density_
    valid = pairs.valid & jnp.asarray(physical_pairs, dtype=bool)
    gradient = kernel.gradient(geometry.displacement, geometry.distance, smoothing_length)
    outer = contract("ei,ej->eij", geometry.displacement, gradient)
    left = pairs.left_indices
    right = pairs.right_indices
    moment = scatter_pair_sum(
        pairs,
        -volume[right, None, None] * outer,
        -volume[left, None, None] * outer,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    dimension = particles.ambient_dimension
    eye = jnp.eye(dimension, dtype=moment.dtype)
    regularized = moment + plan.regularization * eye[None, :, :]
    result = solve_small_linear(
        SmallLinearSolvePlan(
            dimension,
            singular_tolerance=max(plan.regularization, 1e-14),
            maximum_condition=plan.maximum_condition,
            refinement_iterations=1,
        ),
        regularized,
        jnp.broadcast_to(eye, regularized.shape),
    )
    correction = result.value
    residual_norm = result.residual_norm
    condition = result.condition_estimate
    successful = result.successful
    successful = jnp.where(particles.active_mask, successful, True)
    correction = jnp.where(successful[:, None, None], correction, eye)
    return SPHFirstOrderCorrectionState(
        moment, correction, condition, residual_norm, successful
    )


class AbstractSPHDensityDiffusionPlan(StrictModule, NonTrainableState):
    delta: AbstractAttribute[float]
    regularization: AbstractAttribute[float]
    plan_id: AbstractAttribute[str]


class MolteniColagrossiDensityDiffusionPlan(AbstractSPHDensityDiffusionPlan):
    delta: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, delta: float = 0.1, /, *, regularization: float = 0.01):
        if delta < 0.0 or regularization <= 0.0:
            raise ValueError("Density diffusion parameters are invalid.")
        self.delta = float(delta)
        self.regularization = float(regularization)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molteni-colagrossi-density-diffusion",
                "delta": delta,
                "regularization": regularization,
            }
        )


class AntuonoDeltaSPHDiffusionPlan(AbstractSPHDensityDiffusionPlan):
    correction: SPHFirstOrderGradientCorrectionPlan
    delta: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    free_surface_policy: FreeSurfaceDiffusionPolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        delta: float = 0.1,
        /,
        *,
        regularization: float = 0.01,
        correction: SPHFirstOrderGradientCorrectionPlan | None = None,
        free_surface_policy: FreeSurfaceDiffusionPolicy = "disable",
    ):
        if delta < 0.0 or regularization <= 0.0:
            raise ValueError("Density diffusion parameters are invalid.")
        if free_surface_policy not in (
            "disable",
            "one-sided-corrected",
            "smooth-taper",
        ):
            raise ValueError("Unknown free-surface density-diffusion policy.")
        self.delta = float(delta)
        self.regularization = float(regularization)
        self.correction = (
            SPHFirstOrderGradientCorrectionPlan() if correction is None else correction
        )
        self.free_surface_policy = free_surface_policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "antuono-delta-sph-density-diffusion",
                "delta": self.delta,
                "regularization": self.regularization,
                "correction": self.correction.plan_id,
                "free_surface_policy": free_surface_policy,
            }
        )


class SPHDensityDiffusionResult(StrictModule):
    rate: Array
    variance_rate: Array
    positive_variance_defect: Array
    correction_successful: Array


def sph_density_diffusion_rate(
    plan: AbstractSPHDensityDiffusionPlan,
    particles: ParticleDiscretization,
    density: ArrayLike,
    sound_speed: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: float,
    execution: ParticleExecutionPolicy,
    /,
    *,
    free_surface_weight: ArrayLike | None = None,
) -> SPHDensityDiffusionResult:
    density_ = jnp.asarray(density)
    sound = jnp.asarray(sound_speed)
    volume = particles.safe_masses / density_
    valid = pairs.valid & jnp.asarray(physical_pairs, dtype=bool)
    left = pairs.left_indices
    right = pairs.right_indices
    gradient = kernel.gradient(geometry.displacement, geometry.distance, smoothing_length)
    denominator = geometry.distance**2 + plan.regularization * smoothing_length**2
    base_vector = (
        2.0
        * (density_[right] - density_[left])[:, None]
        * (-geometry.displacement)
        / denominator[:, None]
    )
    correction_successful = jnp.ones((particles.capacity,), dtype=bool)
    if isinstance(plan, AntuonoDeltaSPHDiffusionPlan):
        correction = sph_first_order_correction(
            plan.correction,
            particles,
            density_,
            pairs,
            geometry,
            valid,
            kernel,
            smoothing_length,
            execution,
        )
        raw_gradient = scatter_pair_sum(
            pairs,
            volume[right, None] * (density_[right] - density_[left])[:, None] * gradient,
            volume[left, None]
            * (density_[left] - density_[right])[:, None]
            * (-gradient),
            size=particles.capacity,
            accumulation=execution.accumulation,
            valid=valid,
        )
        corrected_gradient = contract(
            "nij,nj->ni", correction.correction_matrix, raw_gradient
        )
        base_vector = base_vector - corrected_gradient[left] - corrected_gradient[right]
        correction_successful = correction.successful
    pair_diffusion = jnp.sum(base_vector * gradient, axis=-1)
    pair_scale = 0.5 * (sound[left] + sound[right]) * smoothing_length
    left_rate = plan.delta * pair_scale * volume[right] * pair_diffusion
    right_rate = plan.delta * pair_scale * volume[left] * pair_diffusion
    if free_surface_weight is not None and isinstance(plan, AntuonoDeltaSPHDiffusionPlan):
        weight = jnp.asarray(free_surface_weight)
        if plan.free_surface_policy == "disable":
            left_rate = jnp.where(weight[left] > 0.5, 0.0, left_rate)
            right_rate = jnp.where(weight[right] > 0.5, 0.0, right_rate)
        elif plan.free_surface_policy == "smooth-taper":
            left_rate = (1.0 - weight[left]) * left_rate
            right_rate = (1.0 - weight[right]) * right_rate
    rate = scatter_pair_sum(
        pairs,
        left_rate,
        right_rate,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    centered = density_ - compensated_sum(density_) / particles.active_count
    variance_rate = 2.0 * compensated_sum(centered * rate)
    return SPHDensityDiffusionResult(
        rate,
        variance_rate,
        jnp.maximum(variance_rate, 0.0),
        correction_successful,
    )


class MonaghanArtificialViscosityPlan(StrictModule, NonTrainableState):
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    activation: ArtificialViscosityActivation = eqx.field(static=True)
    smooth_sharpness: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        alpha: float,
        /,
        *,
        beta: float = 0.0,
        regularization: float = 0.01,
        activation: ArtificialViscosityActivation = "approaching-only",
        smooth_sharpness: float = 50.0,
    ):
        if alpha < 0.0 or beta < 0.0 or regularization <= 0.0:
            raise ValueError("Artificial-viscosity coefficients are invalid.")
        if activation not in ("approaching-only", "always", "smooth-approach"):
            raise ValueError("Unknown artificial-viscosity activation.")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.regularization = float(regularization)
        self.activation = activation
        self.smooth_sharpness = float(smooth_sharpness)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "monaghan-artificial-viscosity",
                "alpha": alpha,
                "beta": beta,
                "regularization": regularization,
                "activation": activation,
                "smooth_sharpness": smooth_sharpness,
            }
        )


class ArtificialViscosityResult(StrictModule):
    force: Array
    pair_power: Array
    dissipation_rate: Array
    positive_power_defect: Array
    active_pairs: Array


def sph_artificial_viscosity_force(
    plan: MonaghanArtificialViscosityPlan,
    particles: ParticleDiscretization,
    density: ArrayLike,
    sound_speed: ArrayLike,
    velocity: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: float,
    execution: ParticleExecutionPolicy,
    /,
) -> ArtificialViscosityResult:
    density_ = jnp.asarray(density)
    sound = jnp.asarray(sound_speed)
    velocity_ = jnp.asarray(velocity)
    left = pairs.left_indices
    right = pairs.right_indices
    velocity_difference = velocity_[left] - velocity_[right]
    radial_velocity = jnp.sum(velocity_difference * geometry.displacement, axis=-1)
    mu = (
        smoothing_length
        * radial_velocity
        / (geometry.distance**2 + plan.regularization * smoothing_length**2)
    )
    pi = (-plan.alpha * 0.5 * (sound[left] + sound[right]) * mu + plan.beta * mu**2) / (
        0.5 * (density_[left] + density_[right])
    )
    if plan.activation == "approaching-only":
        activation = radial_velocity < 0.0
        weight = activation.astype(pi.dtype)
    elif plan.activation == "always":
        activation = jnp.ones_like(radial_velocity, dtype=bool)
        weight = jnp.ones_like(pi)
    else:
        weight = jax.nn.sigmoid(-plan.smooth_sharpness * radial_velocity)
        activation = weight > 0.5
    physical = pairs.valid & jnp.asarray(physical_pairs, dtype=bool)
    valid = physical & activation
    gradient = kernel.gradient(geometry.displacement, geometry.distance, smoothing_length)
    pair_force = (
        -(particles.safe_masses[left] * particles.safe_masses[right] * pi * weight)[
            :, None
        ]
        * gradient
    )
    pair_force = jnp.where(physical[:, None], pair_force, 0.0)
    force = scatter_pair_exchange(
        pairs,
        pair_force,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=physical,
    )
    pair_power = jnp.sum(velocity_difference * pair_force, axis=-1)
    return ArtificialViscosityResult(
        force,
        pair_power,
        -compensated_sum(pair_power),
        compensated_sum(jnp.maximum(pair_power, 0.0)),
        jnp.sum(valid.astype(jnp.int32)),
    )


def shepard_renormalized_density(
    particles: ParticleDiscretization,
    density: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: float,
    execution: ParticleExecutionPolicy,
    /,
) -> tuple[Array, Array]:
    density_ = jnp.asarray(density)
    valid = pairs.valid & jnp.asarray(physical_pairs, dtype=bool)
    weights = kernel.value(geometry.distance, smoothing_length)
    left = pairs.left_indices
    right = pairs.right_indices
    self_weight = kernel.value(jnp.asarray(0.0, weights.dtype), smoothing_length)
    numerator = particles.safe_masses * self_weight + scatter_pair_sum(
        pairs,
        particles.safe_masses[right] * weights,
        particles.safe_masses[left] * weights,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    denominator = particles.safe_masses / density_ * self_weight + scatter_pair_sum(
        pairs,
        particles.safe_masses[right] / density_[right] * weights,
        particles.safe_masses[left] / density_[left] * weights,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    candidate = numerator / jnp.where(denominator > 0.0, denominator, 1.0)
    successful = particles.active_mask & (denominator > 0.0) & jnp.isfinite(candidate)
    return jnp.where(successful, candidate, density_), successful


__all__ = [
    "AbstractSPHDensityDiffusionPlan",
    "AntuonoDeltaSPHDiffusionPlan",
    "ArtificialViscosityResult",
    "MolteniColagrossiDensityDiffusionPlan",
    "MonaghanArtificialViscosityPlan",
    "SPHDensityDiffusionResult",
    "SPHFirstOrderCorrectionState",
    "SPHFirstOrderGradientCorrectionPlan",
    "SPHKernelNormalizationState",
    "shepard_renormalized_density",
    "sph_artificial_viscosity_force",
    "sph_density_diffusion_rate",
    "sph_first_order_correction",
    "sph_kernel_normalization",
]
