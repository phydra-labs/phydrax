#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization
from ._pairwise import (
    ParticlePairGeometry,
    ParticlePairRelation,
    scatter_pair_exchange,
    scatter_pair_sum,
)
from ._precision import ParticleExecutionPolicy
from ._smoothing import AbstractSPHSmoothingKernel


class AlgebraicSmoothingLengthPlan(StrictModule, NonTrainableState):
    eta: float = eqx.field(static=True)
    minimum_h: float = eqx.field(static=True)
    maximum_h: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, eta: float, minimum_h: float, maximum_h: float, /):
        eta_ = float(eta)
        minimum = float(minimum_h)
        maximum = float(maximum_h)
        if eta_ <= 0.0 or minimum <= 0.0 or maximum < minimum:
            raise ValueError("Adaptive smoothing-length parameters are invalid.")
        self.eta = eta_
        self.minimum_h = minimum
        self.maximum_h = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "algebraic-smoothing-length",
                "eta": eta_,
                "minimum_h": minimum,
                "maximum_h": maximum,
            }
        )

    def evaluate(
        self, particles: ParticleDiscretization, density: ArrayLike, /
    ) -> tuple[Array, Array]:
        density_ = jnp.asarray(density)
        raw = self.eta * (particles.safe_masses / density_) ** (
            1.0 / particles.ambient_dimension
        )
        bounded = jnp.clip(raw, self.minimum_h, self.maximum_h)
        return bounded, particles.active_mask & (bounded != raw)


class CoupledSummationSmoothingLengthPlan(StrictModule, NonTrainableState):
    eta: float = eqx.field(static=True)
    minimum_h: float = eqx.field(static=True)
    maximum_h: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        eta: float,
        minimum_h: float,
        maximum_h: float,
        /,
        *,
        maximum_iterations: int = 20,
        tolerance: float = 1e-8,
        relaxation: float = 0.7,
    ):
        if eta <= 0.0 or minimum_h <= 0.0 or maximum_h < minimum_h:
            raise ValueError("Coupled smoothing-length bounds are invalid.")
        if maximum_iterations <= 0 or tolerance <= 0.0 or not 0.0 < relaxation <= 1.0:
            raise ValueError("Coupled smoothing solve parameters are invalid.")
        self.eta = float(eta)
        self.minimum_h = float(minimum_h)
        self.maximum_h = float(maximum_h)
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self.relaxation = float(relaxation)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupled-summation-smoothing-length",
                "eta": eta,
                "minimum_h": minimum_h,
                "maximum_h": maximum_h,
                "maximum_iterations": maximum_iterations,
                "tolerance": tolerance,
                "relaxation": relaxation,
            }
        )

    def evaluate(
        self, particles: ParticleDiscretization, density: ArrayLike, /
    ) -> tuple[Array, Array]:
        density_ = jnp.asarray(density)
        raw = self.eta * (particles.safe_masses / density_) ** (
            1.0 / particles.ambient_dimension
        )
        bounded = jnp.clip(raw, self.minimum_h, self.maximum_h)
        return bounded, particles.active_mask & (bounded != raw)


class AdaptiveSmoothingLengthState(StrictModule):
    smoothing_length: Array
    density: Array
    omega: Array
    residual: Array
    iterations: Array
    converged: Array
    bound_active: Array


def variable_h_density(
    particles: ParticleDiscretization,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: ArrayLike,
    execution: ParticleExecutionPolicy,
    /,
) -> Array:
    h = jnp.asarray(smoothing_length)
    left = pairs.left_indices
    right = pairs.right_indices
    valid = pairs.valid & (
        geometry.distance < kernel.support_factor * jnp.maximum(h[left], h[right])
    )
    left_weight = kernel.value(geometry.distance, h[left])
    right_weight = kernel.value(geometry.distance, h[right])
    density = scatter_pair_sum(
        pairs,
        particles.safe_masses[right] * left_weight,
        particles.safe_masses[left] * right_weight,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    self_weight = kernel.value(jnp.zeros_like(h), h)
    return jnp.where(
        particles.active_mask,
        density + particles.safe_masses * self_weight,
        0.0,
    )


def adaptive_smoothing_state(
    plan: AlgebraicSmoothingLengthPlan,
    particles: ParticleDiscretization,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    kernel: AbstractSPHSmoothingKernel,
    execution: ParticleExecutionPolicy,
    /,
    *,
    density: ArrayLike | None = None,
    initial_h: ArrayLike | None = None,
) -> AdaptiveSmoothingLengthState:
    if isinstance(plan, CoupledSummationSmoothingLengthPlan):
        h0 = (
            jnp.full((particles.capacity,), 0.5 * (plan.minimum_h + plan.maximum_h))
            if initial_h is None
            else jnp.asarray(initial_h)
        )

        def body(_, carry):
            h, _, iterations = carry
            rho = variable_h_density(particles, pairs, geometry, kernel, h, execution)
            target, _ = plan.evaluate(particles, rho)
            next_h = (1.0 - plan.relaxation) * h + plan.relaxation * target
            residual = jnp.max(jnp.abs(next_h - h))
            return next_h, residual, iterations + 1

        h, residual, iterations = jax.lax.fori_loop(
            0,
            plan.maximum_iterations,
            body,
            (h0, jnp.asarray(jnp.inf, h0.dtype), jnp.asarray(0, jnp.int32)),
        )
        rho = variable_h_density(particles, pairs, geometry, kernel, h, execution)
        converged = residual <= plan.tolerance
        bound = particles.active_mask & ((h <= plan.minimum_h) | (h >= plan.maximum_h))
    else:
        if density is None:
            raise ValueError("Algebraic smoothing length requires density.")
        rho = jnp.asarray(density)
        h, bound = plan.evaluate(particles, rho)
        residual = jnp.zeros((), dtype=h.dtype)
        iterations = jnp.zeros((), dtype=jnp.int32)
        converged = jnp.asarray(True)
    left = pairs.left_indices
    right = pairs.right_indices
    valid = pairs.valid & (
        geometry.distance < kernel.support_factor * jnp.maximum(h[left], h[right])
    )
    derivative_left = kernel.smoothing_length_derivative(geometry.distance, h[left])
    derivative_right = kernel.smoothing_length_derivative(geometry.distance, h[right])
    derivative_sum = scatter_pair_sum(
        pairs,
        particles.safe_masses[right] * derivative_left,
        particles.safe_masses[left] * derivative_right,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    derivative_sum = (
        derivative_sum
        + particles.safe_masses * kernel.smoothing_length_derivative(jnp.zeros_like(h), h)
    )
    dh_drho = -h / (particles.ambient_dimension * jnp.where(rho > 0.0, rho, 1.0))
    omega = 1.0 - dh_drho * derivative_sum
    return AdaptiveSmoothingLengthState(
        h, rho, omega, residual, iterations, converged, bound
    )


def variable_h_pressure_gradient(
    particles: ParticleDiscretization,
    density: ArrayLike,
    pressure: ArrayLike,
    adaptive: AdaptiveSmoothingLengthState,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    kernel: AbstractSPHSmoothingKernel,
    execution: ParticleExecutionPolicy,
    /,
) -> Array:
    density_ = jnp.asarray(density)
    pressure_ = jnp.asarray(pressure)
    h = adaptive.smoothing_length
    omega = adaptive.omega
    left = pairs.left_indices
    right = pairs.right_indices
    valid = pairs.valid & (
        geometry.distance < kernel.support_factor * jnp.maximum(h[left], h[right])
    )
    left_gradient = kernel.gradient(geometry.displacement, geometry.distance, h[left])
    right_gradient = kernel.gradient(geometry.displacement, geometry.distance, h[right])
    pair_gradient = (
        particles.safe_masses[left, None]
        * particles.safe_masses[right, None]
        * (
            pressure_[left, None]
            / (omega[left, None] * density_[left, None] ** 2)
            * left_gradient
            + pressure_[right, None]
            / (omega[right, None] * density_[right, None] ** 2)
            * right_gradient
        )
    )
    return scatter_pair_exchange(
        pairs,
        pair_gradient,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )


__all__ = [
    "AdaptiveSmoothingLengthState",
    "AlgebraicSmoothingLengthPlan",
    "CoupledSummationSmoothingLengthPlan",
    "adaptive_smoothing_state",
    "variable_h_density",
    "variable_h_pressure_gradient",
]
