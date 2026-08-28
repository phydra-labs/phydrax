#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ._pairwise import (
    ParticlePairGeometry,
    ParticlePairRelation,
    scatter_pair_exchange,
    scatter_pair_sum,
)
from ._precision import ParticleExecutionPolicy, ParticlePrecisionPolicy
from ._smoothing import AbstractSPHSmoothingKernel


class SPHViscousForceResult(StrictModule):
    """Pairwise viscous force and kinetic-power evidence."""

    force: Array
    pair_force: Array
    pair_power: Array
    dissipation_rate: Array
    positive_power_defect: Array


def sph_summation_density(
    masses: ArrayLike,
    active_mask: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: ArrayLike,
    /,
    *,
    particle_count: int,
    execution: ParticleExecutionPolicy,
    precision: ParticlePrecisionPolicy,
) -> Array:
    """Evaluate self-inclusive SPH density from one unordered pair relation."""

    masses_ = precision.evaluation(masses)
    active = jnp.asarray(active_mask, dtype=bool)
    valid = jnp.asarray(physical_pairs, dtype=bool)
    pair_kernel = precision.evaluation(kernel.value(geometry.distance, smoothing_length))
    left_mass = masses_[pairs.left_indices]
    right_mass = masses_[pairs.right_indices]
    neighbor_density = scatter_pair_sum(
        pairs,
        right_mass * pair_kernel,
        left_mass * pair_kernel,
        size=int(particle_count),
        accumulation=execution.accumulation,
        valid=valid,
    )
    zero = jnp.asarray(0.0, dtype=pair_kernel.dtype)
    self_kernel = kernel.value(zero, smoothing_length)
    self_density = masses_ * self_kernel
    return precision.evaluation(jnp.where(active, self_density + neighbor_density, 0.0))


def sph_continuity_density_rate(
    masses: ArrayLike,
    velocity: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: ArrayLike,
    /,
    *,
    particle_count: int,
    execution: ParticleExecutionPolicy,
    precision: ParticlePrecisionPolicy,
) -> Array:
    """Evaluate the pair-once SPH continuity-density rate."""

    masses_ = precision.evaluation(masses)
    velocity_ = precision.evaluation(velocity)
    left = pairs.left_indices
    right = pairs.right_indices
    kernel_gradient = kernel.gradient(
        geometry.displacement,
        geometry.distance,
        smoothing_length,
    )
    velocity_difference = velocity_[left] - velocity_[right]
    pair_rate = jnp.sum(velocity_difference * kernel_gradient, axis=-1)
    return precision.output(
        scatter_pair_sum(
            pairs,
            masses_[right] * pair_rate,
            masses_[left] * pair_rate,
            size=int(particle_count),
            accumulation=execution.accumulation,
            valid=physical_pairs,
        )
    )


def sph_symmetric_pressure_gradient(
    masses: ArrayLike,
    density: ArrayLike,
    pressure: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: ArrayLike,
    /,
    *,
    particle_count: int,
    execution: ParticleExecutionPolicy,
    precision: ParticlePrecisionPolicy,
) -> Array:
    """Return the conservative discrete pressure-potential gradient."""

    masses_ = precision.evaluation(masses)
    density_ = precision.evaluation(density)
    pressure_ = precision.evaluation(pressure)
    left = pairs.left_indices
    right = pairs.right_indices
    coefficient = (
        pressure_[left] / density_[left] ** 2 + pressure_[right] / density_[right] ** 2
    )
    kernel_gradient = kernel.gradient(
        geometry.displacement,
        geometry.distance,
        smoothing_length,
    )
    pair_gradient = (
        masses_[left, None]
        * masses_[right, None]
        * coefficient[:, None]
        * kernel_gradient
    )
    return precision.output(
        scatter_pair_exchange(
            pairs,
            precision.accumulation(pair_gradient),
            size=int(particle_count),
            accumulation=execution.accumulation,
            valid=physical_pairs,
        )
    )


def sph_morris_viscous_force(
    masses: ArrayLike,
    density: ArrayLike,
    velocity: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: ArrayLike,
    kinematic_viscosity: ArrayLike,
    regularization: ArrayLike,
    /,
    *,
    particle_count: int,
    execution: ParticleExecutionPolicy,
    precision: ParticlePrecisionPolicy,
) -> SPHViscousForceResult:
    """Evaluate symmetric Morris viscosity and exact pair kinetic power."""

    masses_ = precision.evaluation(masses)
    density_ = precision.evaluation(density)
    velocity_ = precision.evaluation(velocity)
    viscosity = precision.evaluation(kinematic_viscosity)
    epsilon = precision.evaluation(regularization)
    left = pairs.left_indices
    right = pairs.right_indices
    displacement = geometry.displacement
    kernel_gradient = kernel.gradient(
        displacement,
        geometry.distance,
        smoothing_length,
    )
    radial_contraction = jnp.sum(displacement * kernel_gradient, axis=-1)
    velocity_difference = velocity_[left] - velocity_[right]
    dynamic_left = density_[left] * viscosity
    dynamic_right = density_[right] * viscosity
    denominator = (
        density_[left]
        * density_[right]
        * (geometry.distance**2 + epsilon * jnp.asarray(smoothing_length) ** 2)
    )
    scalar = (
        masses_[left]
        * masses_[right]
        * (dynamic_left + dynamic_right)
        * radial_contraction
        / denominator
    )
    pair_force = scalar[:, None] * velocity_difference
    valid = pairs.valid & jnp.asarray(physical_pairs, dtype=bool)
    pair_force = jnp.where(valid[:, None], pair_force, 0.0)
    pair_power = jnp.where(
        valid,
        jnp.sum(velocity_difference * pair_force, axis=-1),
        0.0,
    )
    force = scatter_pair_exchange(
        pairs,
        precision.accumulation(pair_force),
        size=int(particle_count),
        accumulation=execution.accumulation,
        valid=valid,
    )
    pair_power_certified = precision.certification(pair_power)
    return SPHViscousForceResult(
        force=precision.output(force),
        pair_force=precision.output(pair_force),
        pair_power=pair_power_certified,
        dissipation_rate=-compensated_sum(pair_power_certified),
        positive_power_defect=compensated_sum(jnp.maximum(pair_power_certified, 0.0)),
    )


__all__ = [
    "SPHViscousForceResult",
    "sph_continuity_density_rate",
    "sph_morris_viscous_force",
    "sph_summation_density",
    "sph_symmetric_pressure_gradient",
]
