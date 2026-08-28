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
from ._pairwise import particle_pair_geometry
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


class IISPHStateLayout(StrictModule, NonTrainableState):
    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, capacity: int, dimension: int, /):
        self.capacity = int(capacity)
        self.dimension = int(dimension)
        self.width = 2 * self.dimension + 1
        self.layout_id = canonical_fingerprint(
            {"kind": "iisph-state-layout", "capacity": capacity, "dimension": dimension}
        )

    def pack(
        self, position: ArrayLike, velocity: ArrayLike, pressure: ArrayLike, /
    ) -> Array:
        q = jnp.asarray(position)
        v = jnp.asarray(velocity)
        p = jnp.asarray(pressure)
        if q.shape != (self.capacity, self.dimension) or v.shape != q.shape:
            raise ValueError("IISPH position/velocity shape mismatch.")
        if p.shape != (self.capacity,):
            raise ValueError("IISPH pressure shape mismatch.")
        return jnp.concatenate((q, v, p[:, None]), axis=-1)

    def unpack(self, state: ArrayLike, /) -> tuple[Array, Array, Array]:
        value = jnp.asarray(state)
        if value.shape != (self.capacity, self.width):
            raise ValueError("IISPH state shape mismatch.")
        return (
            value[:, : self.dimension],
            value[:, self.dimension : 2 * self.dimension],
            value[:, -1],
        )


class IISPHMethodPlan(StrictModule, NonTrainableState):
    reference_density: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    qualification: ParticleQualificationProfile
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_density: float,
        /,
        *,
        maximum_iterations: int = 50,
        tolerance: float = 1e-4,
        relaxation: float = 0.5,
        qualification: ParticleQualificationProfile | None = None,
    ):
        if reference_density <= 0.0 or maximum_iterations <= 0 or tolerance <= 0.0:
            raise ValueError("IISPH solve parameters are invalid.")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("IISPH relaxation must be in (0, 1].")
        self.reference_density = float(reference_density)
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self.relaxation = float(relaxation)
        self.qualification = (
            ParticleQualificationProfile() if qualification is None else qualification
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iisph-method",
                "reference_density": reference_density,
                "maximum_iterations": maximum_iterations,
                "tolerance": tolerance,
                "relaxation": relaxation,
                "qualification": self.qualification.profile_id,
            }
        )


class IISPHStepResult(StrictModule):
    candidate_state: Array
    accepted_state: Array
    pressure: Array
    predicted_density: Array
    corrected_density: Array
    residual: Array
    iterations: Array
    converged: Array
    successful: Array
    free_surface_mask: Array
    constraints: ParticleConstraintResiduals
    numerical_constraints_satisfied: Array
    production_qualified: Array


class PreparedIISPH(StrictModule, NonTrainableState):
    particles: ParticleDiscretization
    neighborhood: AbstractPreparedParticleNeighborhood
    kernel: AbstractSPHSmoothingKernel
    plan: IISPHMethodPlan
    free_surface: FreeSurfaceDetectionPlan | None
    execution: ParticleExecutionPolicy
    precision: ParticlePrecisionPolicy
    layout: IISPHStateLayout
    smoothing_length: float = eqx.field(static=True)
    external_acceleration: Callable[[Array, Array, Array, Any], Array] | None
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        neighborhood: AbstractPreparedParticleNeighborhood,
        kernel: AbstractSPHSmoothingKernel,
        smoothing_length: float,
        plan: IISPHMethodPlan,
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
        self.layout = IISPHStateLayout(particles.capacity, particles.ambient_dimension)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-iisph",
                "particles": particles.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "kernel": kernel.kernel_id,
                "smoothing_length": smoothing_length,
                "plan": plan.plan_id,
                "free_surface": None if free_surface is None else free_surface.plan_id,
            }
        )

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        pressure: ArrayLike | None = None,
        /,
    ) -> Array:
        p = (
            jnp.zeros((self.particles.capacity,))
            if pressure is None
            else jnp.asarray(pressure)
        )
        return self.layout.pack(position, velocity, p)

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

    def pressure_action(
        self, position: Array, pressure: Array, step_size: Array, /
    ) -> Array:
        neighborhood, geometry, valid, density = self._geometry(position)
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
        acceleration = -gradient / self.particles.safe_masses[:, None]
        density_rate = sph_continuity_density_rate(
            self.particles.safe_masses,
            acceleration,
            neighborhood.pair_relation,
            geometry,
            valid,
            self.kernel,
            self.smoothing_length,
            particle_count=self.particles.capacity,
            execution=self.execution,
            precision=self.precision,
        )
        return step_size**2 * density_rate

    def step_detailed(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any = None,
        /,
    ) -> IISPHStepResult:
        position, velocity, pressure0 = self.layout.unpack(state)
        neighborhood, geometry, valid, density = self._geometry(position)
        external = (
            jnp.zeros_like(position)
            if self.external_acceleration is None
            else self.external_acceleration(time, position, velocity, args)
        )
        predicted_velocity = velocity + step_size * external
        predicted_density = density + step_size * sph_continuity_density_rate(
            self.particles.safe_masses,
            predicted_velocity,
            neighborhood.pair_relation,
            geometry,
            valid,
            self.kernel,
            self.smoothing_length,
            particle_count=self.particles.capacity,
            execution=self.execution,
            precision=self.precision,
        )
        surface_mask = jnp.zeros((self.particles.capacity,), dtype=bool)
        if self.free_surface is not None:
            surface_state = detect_free_surface(
                self.free_surface,
                self.particles,
                density,
                neighborhood.pair_relation,
                geometry,
                valid,
                self.kernel,
                self.smoothing_length,
                self.execution,
            )
            surface_mask = surface_state.hard_mask
        diagonal = jnp.full(
            (self.particles.capacity,),
            step_size**2 / jnp.maximum(self.smoothing_length**2, 1e-14),
            dtype=position.dtype,
        )

        def body(_, carry):
            pressure, _ = carry
            corrected = predicted_density + self.pressure_action(
                position, pressure, step_size
            )
            residual = corrected - self.plan.reference_density
            update = pressure - self.plan.relaxation * residual / diagonal
            update = jnp.where(surface_mask, 0.0, jnp.maximum(update, 0.0))
            return update, jnp.max(jnp.maximum(residual, 0.0))

        pressure, residual = jax.lax.fori_loop(
            0,
            self.plan.maximum_iterations,
            body,
            (jnp.maximum(pressure0, 0.0), jnp.asarray(jnp.inf, position.dtype)),
        )
        neighborhood, geometry, valid, density = self._geometry(position)
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
        corrected_velocity = (
            predicted_velocity
            - step_size * gradient / self.particles.safe_masses[:, None]
        )
        corrected_density = predicted_density + self.pressure_action(
            position, pressure, step_size
        )
        candidate = self.layout.pack(
            position + step_size * corrected_velocity, corrected_velocity, pressure
        )
        converged = residual <= self.plan.tolerance
        successful = converged & jnp.all(jnp.isfinite(candidate))
        accepted = jnp.where(successful, candidate, state)
        volumes = self.particles.safe_masses / self.plan.reference_density
        constraints = particle_constraint_residuals(
            corrected_density,
            self.plan.reference_density,
            volumes,
            pressure=pressure,
            atmospheric_pressure=0.0,
            active_mask=self.particles.active_mask,
        )
        constraints_satisfied = self.plan.qualification.constraints_satisfied(constraints)
        production_qualified = successful & constraints_satisfied
        return IISPHStepResult(
            candidate,
            accepted,
            pressure,
            predicted_density,
            corrected_density,
            residual,
            jnp.asarray(self.plan.maximum_iterations, jnp.int32),
            converged,
            successful,
            surface_mask,
            constraints,
            constraints_satisfied,
            production_qualified,
        )


__all__ = [
    "IISPHMethodPlan",
    "IISPHStateLayout",
    "IISPHStepResult",
    "PreparedIISPH",
]
