#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization
from ._free_surface import detect_free_surface, FreeSurfaceDetectionPlan
from ._neighborhood import AbstractPreparedParticleNeighborhood
from ._pairwise import particle_pair_geometry, scatter_pair_sum
from ._precision import ParticleExecutionPolicy, ParticlePrecisionPolicy
from ._qualification import (
    particle_constraint_residuals,
    ParticleConstraintResiduals,
    ParticleQualificationProfile,
)
from ._smoothing import AbstractSPHSmoothingKernel
from ._sph_operators import (
    sph_continuity_density_rate,
    sph_summation_density,
    sph_symmetric_pressure_gradient,
)


class DFSPHStateLayout(StrictModule, NonTrainableState):
    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(self, capacity: int, dimension: int, /):
        self.capacity = int(capacity)
        self.dimension = int(dimension)
        self.width = 2 * self.dimension + 2

    def pack(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        density_multiplier: ArrayLike,
        divergence_multiplier: ArrayLike,
        /,
    ) -> Array:
        q = jnp.asarray(position)
        v = jnp.asarray(velocity)
        kd = jnp.asarray(density_multiplier)
        kv = jnp.asarray(divergence_multiplier)
        if q.shape != (self.capacity, self.dimension) or v.shape != q.shape:
            raise ValueError("DFSPH position/velocity shape mismatch.")
        if kd.shape != (self.capacity,) or kv.shape != kd.shape:
            raise ValueError("DFSPH multiplier shape mismatch.")
        return jnp.concatenate((q, v, kd[:, None], kv[:, None]), axis=-1)

    def unpack(self, state: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
        value = jnp.asarray(state)
        if value.shape != (self.capacity, self.width):
            raise ValueError("DFSPH state shape mismatch.")
        return (
            value[:, : self.dimension],
            value[:, self.dimension : 2 * self.dimension],
            value[:, -2],
            value[:, -1],
        )


class DFSPHMethodPlan(StrictModule, NonTrainableState):
    reference_density: float = eqx.field(static=True)
    divergence_iterations: int = eqx.field(static=True)
    density_iterations: int = eqx.field(static=True)
    divergence_tolerance: float = eqx.field(static=True)
    density_tolerance: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    qualification: ParticleQualificationProfile

    def __init__(
        self,
        reference_density: float,
        /,
        *,
        divergence_iterations: int = 20,
        density_iterations: int = 30,
        divergence_tolerance: float = 1e-4,
        density_tolerance: float = 1e-4,
        relaxation: float = 0.5,
        qualification: ParticleQualificationProfile | None = None,
    ):
        if (
            reference_density <= 0.0
            or divergence_iterations <= 0
            or density_iterations <= 0
        ):
            raise ValueError("DFSPH method parameters are invalid.")
        self.reference_density = float(reference_density)
        self.divergence_iterations = int(divergence_iterations)
        self.density_iterations = int(density_iterations)
        self.divergence_tolerance = float(divergence_tolerance)
        self.density_tolerance = float(density_tolerance)
        self.relaxation = float(relaxation)
        self.qualification = (
            ParticleQualificationProfile() if qualification is None else qualification
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dfsph-method",
                "reference_density": reference_density,
                "divergence_iterations": divergence_iterations,
                "density_iterations": density_iterations,
                "divergence_tolerance": divergence_tolerance,
                "density_tolerance": density_tolerance,
                "relaxation": relaxation,
                "qualification": self.qualification.profile_id,
            }
        )


class DFSPHFactorState(StrictModule):
    alpha: Array
    denominator: Array
    deficient: Array


class DFSPHStepResult(StrictModule):
    candidate_state: Array
    accepted_state: Array
    divergence_residual: Array
    density_residual: Array
    divergence_iterations: Array
    density_iterations: Array
    divergence_converged: Array
    density_converged: Array
    successful: Array
    factor: DFSPHFactorState
    free_surface_mask: Array
    constraints: ParticleConstraintResiduals
    numerical_constraints_satisfied: Array
    production_qualified: Array


class PreparedDFSPH(StrictModule, NonTrainableState):
    particles: ParticleDiscretization
    neighborhood: AbstractPreparedParticleNeighborhood
    kernel: AbstractSPHSmoothingKernel
    plan: DFSPHMethodPlan
    free_surface: FreeSurfaceDetectionPlan | None
    execution: ParticleExecutionPolicy
    precision: ParticlePrecisionPolicy
    layout: DFSPHStateLayout
    smoothing_length: float = eqx.field(static=True)
    external_acceleration: Callable[[Array, Array, Array, Any], Array] | None
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        neighborhood: AbstractPreparedParticleNeighborhood,
        kernel: AbstractSPHSmoothingKernel,
        smoothing_length: float,
        plan: DFSPHMethodPlan,
        /,
        *,
        free_surface: FreeSurfaceDetectionPlan | None = None,
        external_acceleration: Callable[[Array, Array, Array, Any], Array] | None = None,
        execution: ParticleExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
    ):
        self.particles = particles
        self.neighborhood = neighborhood
        self.kernel = kernel
        self.smoothing_length = float(smoothing_length)
        self.plan = plan
        self.free_surface = free_surface
        self.external_acceleration = external_acceleration
        self.execution = (
            ParticleExecutionPolicy(realization=neighborhood.backend)
            if execution is None
            else execution
        )
        self.precision = ParticlePrecisionPolicy() if precision is None else precision
        self.layout = DFSPHStateLayout(particles.capacity, particles.ambient_dimension)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-dfsph",
                "particles": particles.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "kernel": kernel.kernel_id,
                "smoothing_length": smoothing_length,
                "plan": plan.plan_id,
            }
        )

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        density_multiplier: ArrayLike | None = None,
        divergence_multiplier: ArrayLike | None = None,
        /,
    ) -> Array:
        zeros = jnp.zeros((self.particles.capacity,))
        return self.layout.pack(
            position,
            velocity,
            zeros if density_multiplier is None else density_multiplier,
            zeros if divergence_multiplier is None else divergence_multiplier,
        )

    def _geometry(self, position):
        neighborhood = self.neighborhood.build(position)
        position = neighborhood.require_success(position)
        geometry = particle_pair_geometry(
            position, neighborhood.pair_relation, box=self.neighborhood.box
        )
        valid = geometry.valid & (
            geometry.distance < self.kernel.support_radius(self.smoothing_length)
        )
        density = sph_summation_density(
            self.particles.safe_masses,
            self.particles.active_mask,
            neighborhood.pair_relation,
            geometry,
            valid,
            self.kernel,
            self.smoothing_length,
            particle_count=self.particles.capacity,
            execution=self.execution,
            precision=self.precision,
        )
        return neighborhood, geometry, valid, density

    def factor(self, position: Array) -> DFSPHFactorState:
        neighborhood, geometry, valid, _ = self._geometry(position)
        pairs = neighborhood.pair_relation
        gradient = self.kernel.gradient(
            geometry.displacement, geometry.distance, self.smoothing_length
        )
        left = pairs.left_indices
        right = pairs.right_indices
        left_coefficient = (
            self.particles.safe_masses[right, None]
            / self.plan.reference_density**2
            * gradient
        )
        right_coefficient = (
            -self.particles.safe_masses[left, None]
            / self.plan.reference_density**2
            * gradient
        )
        sum_gradient = scatter_pair_sum(
            pairs,
            left_coefficient,
            right_coefficient,
            size=self.particles.capacity,
            accumulation=self.execution.accumulation,
            valid=valid,
        )
        squared = scatter_pair_sum(
            pairs,
            jnp.sum(left_coefficient**2, axis=-1),
            jnp.sum(right_coefficient**2, axis=-1),
            size=self.particles.capacity,
            accumulation=self.execution.accumulation,
            valid=valid,
        )
        denominator = squared + jnp.sum(sum_gradient**2, axis=-1)
        deficient = self.particles.active_mask & (denominator <= 1e-14)
        alpha = 1.0 / jnp.where(deficient, 1.0, denominator)
        return DFSPHFactorState(alpha, denominator, deficient)

    def _correct_velocity(
        self,
        position,
        velocity,
        residual,
        factor,
        step_size,
        surface_mask,
    ):
        neighborhood, geometry, valid, density = self._geometry(position)
        multiplier = jnp.where(
            surface_mask,
            0.0,
            jnp.maximum(residual, 0.0) * factor.alpha / jnp.maximum(step_size, 1e-14),
        )
        pressure = multiplier * density**2
        gradient = sph_symmetric_pressure_gradient(
            self.particles.safe_masses,
            density,
            pressure,
            neighborhood.pair_relation,
            geometry,
            valid,
            self.kernel,
            self.smoothing_length,
            particle_count=self.particles.capacity,
            execution=self.execution,
            precision=self.precision,
        )
        corrected = (
            velocity
            - self.plan.relaxation
            * step_size
            * gradient
            / self.particles.safe_masses[:, None]
        )
        return corrected, multiplier

    def step_detailed(
        self, time: Array, state: Array, step_size: Array, args: Any = None, /
    ) -> DFSPHStepResult:
        position, velocity, density_multiplier0, divergence_multiplier0 = (
            self.layout.unpack(state)
        )
        neighborhood, geometry, valid, density = self._geometry(position)
        factor = self.factor(position)
        surface_mask = jnp.zeros((self.particles.capacity,), dtype=bool)
        if self.free_surface is not None:
            surface_mask = detect_free_surface(
                self.free_surface,
                self.particles,
                density,
                neighborhood.pair_relation,
                geometry,
                valid,
                self.kernel,
                self.smoothing_length,
                self.execution,
            ).hard_mask

        def divergence_body(_, carry):
            current_velocity, multiplier, _ = carry
            divergence = sph_continuity_density_rate(
                self.particles.safe_masses,
                current_velocity,
                neighborhood.pair_relation,
                geometry,
                valid,
                self.kernel,
                self.smoothing_length,
                particle_count=self.particles.capacity,
                execution=self.execution,
                precision=self.precision,
            )
            corrected, increment = self._correct_velocity(
                position,
                current_velocity,
                divergence,
                factor,
                step_size,
                surface_mask,
            )
            return (
                corrected,
                multiplier + increment,
                jnp.max(jnp.maximum(divergence, 0.0)),
            )

        divergence_velocity, divergence_multiplier, divergence_residual = (
            jax.lax.fori_loop(
                0,
                self.plan.divergence_iterations,
                divergence_body,
                (velocity, divergence_multiplier0, jnp.asarray(jnp.inf, position.dtype)),
            )
        )
        external = (
            jnp.zeros_like(position)
            if self.external_acceleration is None
            else self.external_acceleration(time, position, divergence_velocity, args)
        )
        predicted_velocity = divergence_velocity + step_size * external

        def density_body(_, carry):
            current_velocity, multiplier, _ = carry
            density_rate = sph_continuity_density_rate(
                self.particles.safe_masses,
                current_velocity,
                neighborhood.pair_relation,
                geometry,
                valid,
                self.kernel,
                self.smoothing_length,
                particle_count=self.particles.capacity,
                execution=self.execution,
                precision=self.precision,
            )
            predicted = density + step_size * density_rate
            residual = predicted - self.plan.reference_density
            corrected, increment = self._correct_velocity(
                position,
                current_velocity,
                residual,
                factor,
                step_size,
                surface_mask,
            )
            return corrected, multiplier + increment, jnp.max(jnp.maximum(residual, 0.0))

        corrected_velocity, density_multiplier, density_residual = jax.lax.fori_loop(
            0,
            self.plan.density_iterations,
            density_body,
            (
                predicted_velocity,
                density_multiplier0,
                jnp.asarray(jnp.inf, position.dtype),
            ),
        )
        candidate = self.layout.pack(
            position + step_size * corrected_velocity,
            corrected_velocity,
            density_multiplier,
            divergence_multiplier,
        )
        divergence_converged = divergence_residual <= self.plan.divergence_tolerance
        density_converged = density_residual <= self.plan.density_tolerance
        successful = (
            divergence_converged
            & density_converged
            & ~jnp.any(factor.deficient)
            & jnp.all(jnp.isfinite(candidate))
        )
        accepted = jnp.where(successful, candidate, state)
        final_density_rate = sph_continuity_density_rate(
            self.particles.safe_masses,
            corrected_velocity,
            neighborhood.pair_relation,
            geometry,
            valid,
            self.kernel,
            self.smoothing_length,
            particle_count=self.particles.capacity,
            execution=self.execution,
            precision=self.precision,
        )
        final_density = density + step_size * final_density_rate
        constraints = particle_constraint_residuals(
            final_density,
            self.plan.reference_density,
            self.particles.safe_masses / self.plan.reference_density,
            density_rate=final_density_rate,
            step_size=step_size,
            active_mask=self.particles.active_mask,
        )
        constraints_satisfied = self.plan.qualification.constraints_satisfied(constraints)
        production_qualified = successful & constraints_satisfied
        return DFSPHStepResult(
            candidate,
            accepted,
            divergence_residual,
            density_residual,
            jnp.asarray(self.plan.divergence_iterations, jnp.int32),
            jnp.asarray(self.plan.density_iterations, jnp.int32),
            divergence_converged,
            density_converged,
            successful,
            factor,
            surface_mask,
            constraints,
            constraints_satisfied,
            production_qualified,
        )


__all__ = [
    "DFSPHFactorState",
    "DFSPHMethodPlan",
    "DFSPHStateLayout",
    "DFSPHStepResult",
    "PreparedDFSPH",
]
