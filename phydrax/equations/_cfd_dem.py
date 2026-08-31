#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import (
    DEMRuntimeState,
    ParticleGridRelation,
    PreparedParticleGridTransfer,
    PreparedSoftSphereDEMDynamics,
)


class FluidParticleSample(StrictModule):
    velocity: Array
    density: Array
    dynamic_viscosity: Array
    pressure_gradient: Array
    porosity: Array


class HydrodynamicClosureResult(StrictModule):
    particle_force: Array
    particle_torque: Array
    reynolds_number: Array
    validity: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class AbstractHydrodynamicClosurePlan(StrictModule, NonTrainableState):
    closure_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        sample: FluidParticleSample,
        particle_velocity: Array,
        particle_radius: Array,
        particle_volume: Array,
        /,
    ) -> HydrodynamicClosureResult:
        raise NotImplementedError


class StokesDragPlan(AbstractHydrodynamicClosurePlan):
    maximum_reynolds: float = eqx.field(static=True)
    include_pressure_gradient: bool = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_reynolds: float = 1.0,
        include_pressure_gradient: bool = True,
    ):
        maximum = float(maximum_reynolds)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_reynolds must be finite and positive.")
        self.maximum_reynolds = maximum
        self.include_pressure_gradient = bool(include_pressure_gradient)
        self.closure_id = canonical_fingerprint(
            {
                "kind": "stokes-drag-plan",
                "maximum_reynolds": maximum,
                "include_pressure_gradient": self.include_pressure_gradient,
            }
        )

    def evaluate(
        self,
        sample,
        particle_velocity,
        particle_radius,
        particle_volume,
        /,
    ):
        slip = sample.velocity - particle_velocity
        speed = jnp.linalg.norm(slip, axis=-1)
        diameter = 2.0 * particle_radius
        reynolds = sample.density * speed * diameter / sample.dynamic_viscosity
        drag = (
            6.0
            * jnp.pi
            * sample.dynamic_viscosity[:, None]
            * particle_radius[:, None]
            * slip
        )
        pressure = (
            -particle_volume[:, None] * sample.pressure_gradient
            if self.include_pressure_gradient
            else jnp.zeros_like(drag)
        )
        force = drag + pressure
        valid = (
            jnp.isfinite(reynolds)
            & (reynolds <= self.maximum_reynolds)
            & jnp.isfinite(sample.dynamic_viscosity)
            & (sample.dynamic_viscosity > 0.0)
            & jnp.isfinite(sample.density)
            & (sample.density > 0.0)
            & jnp.isfinite(sample.porosity)
            & (sample.porosity > 0.0)
            & (sample.porosity <= 1.0)
        )
        successful = jnp.all(valid) & jnp.all(jnp.isfinite(force))
        angular_dimension = 1 if force.shape[-1] == 2 else 3
        return HydrodynamicClosureResult(
            force,
            jnp.zeros((force.shape[0], angular_dimension), dtype=force.dtype),
            reynolds,
            valid,
            successful,
            self.closure_id,
        )


class UnresolvedCFDEMCouplingPlan(StrictModule, NonTrainableState):
    dynamics: PreparedSoftSphereDEMDynamics
    transfer: PreparedParticleGridTransfer
    closure: AbstractHydrodynamicClosurePlan
    minimum_porosity: float = eqx.field(static=True)
    maximum_porosity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedSoftSphereDEMDynamics,
        transfer: PreparedParticleGridTransfer,
        closure: AbstractHydrodynamicClosurePlan,
        /,
        *,
        minimum_porosity: float = 1.0e-3,
        maximum_porosity: float = 1.0,
    ):
        if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
            raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
        if not isinstance(transfer, PreparedParticleGridTransfer):
            raise TypeError("transfer must be PreparedParticleGridTransfer.")
        if not isinstance(closure, AbstractHydrodynamicClosurePlan):
            raise TypeError("closure must be AbstractHydrodynamicClosurePlan.")
        if transfer.particles.prepared_id != dynamics.bodies.particles.prepared_id:
            raise ValueError(
                "CFD-DEM transfer and dynamics use different particle supports."
            )
        minimum = float(minimum_porosity)
        maximum = float(maximum_porosity)
        if not 0.0 < minimum < maximum <= 1.0:
            raise ValueError("Porosity bounds must satisfy 0 < min < max <= 1.")
        self.dynamics = dynamics
        self.transfer = transfer
        self.closure = closure
        self.minimum_porosity = minimum
        self.maximum_porosity = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unresolved-cfd-dem-coupling-plan",
                "dynamics": dynamics.prepared_id,
                "transfer": transfer.prepared_id,
                "closure": closure.closure_id,
                "porosity": [minimum, maximum],
            }
        )


class CFDEMCouplingEvaluation(StrictModule):
    relation: ParticleGridRelation
    particle_force: Array
    particle_torque: Array
    fluid_momentum_source_rate: Array
    particle_impulse: Array
    fluid_impulse: Array
    porosity: Array
    solid_velocity: Array
    momentum_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def evaluate_unresolved_cfd_dem(
    plan: UnresolvedCFDEMCouplingPlan,
    dem_state: DEMRuntimeState,
    fluid_velocity: ArrayLike,
    fluid_density: ArrayLike,
    dynamic_viscosity: ArrayLike,
    pressure_gradient: ArrayLike,
    particle_volume: ArrayLike,
    step_size: ArrayLike,
    /,
) -> CFDEMCouplingEvaluation:
    if not isinstance(plan, UnresolvedCFDEMCouplingPlan):
        raise TypeError("plan must be an UnresolvedCFDEMCouplingPlan.")
    transfer = plan.transfer
    relation = transfer.relation(dem_state.kinematics.position)
    velocity = jnp.asarray(fluid_velocity)
    density = jnp.asarray(fluid_density)
    viscosity = jnp.asarray(dynamic_viscosity)
    pressure = jnp.asarray(pressure_gradient)
    volume = jnp.asarray(particle_volume, dtype=velocity.dtype)
    cell_count = transfer.cell_count
    dimension = plan.dynamics.bodies.ambient_dimension
    if velocity.shape != (cell_count, dimension) or pressure.shape != velocity.shape:
        raise ValueError("Fluid velocity/pressure-gradient shape is invalid.")
    if density.shape != (cell_count,) or viscosity.shape != density.shape:
        raise ValueError("Fluid density/viscosity shape is invalid.")
    if volume.shape != (plan.dynamics.bodies.capacity,):
        raise ValueError("particle_volume shape is invalid.")
    solid_volume = transfer.deposit_particle_content(relation, volume)
    porosity = 1.0 - solid_volume / transfer.plan.cell_volumes
    momentum = transfer.deposit_particle_content(
        relation, volume[:, None] * dem_state.kinematics.velocity
    )
    solid_velocity = momentum / jnp.where(solid_volume > 0.0, solid_volume, 1.0)[:, None]
    particle_porosity = transfer.gather(relation, porosity)
    sample = FluidParticleSample(
        transfer.gather(relation, velocity),
        transfer.gather(relation, density),
        transfer.gather(relation, viscosity),
        transfer.gather(relation, pressure),
        particle_porosity,
    )
    closure = plan.closure.evaluate(
        sample,
        dem_state.kinematics.velocity,
        plan.dynamics.bodies.radii,
        volume,
    )
    fluid_source = transfer.deposit_particle_content(relation, -closure.particle_force)
    dt = jnp.asarray(step_size, dtype=velocity.dtype)
    particle_impulse = dt * closure.particle_force
    fluid_impulse = dt * fluid_source
    momentum_residual = jnp.sum(particle_impulse, axis=0) + jnp.sum(fluid_impulse, axis=0)
    porosity_valid = jnp.all(
        jnp.isfinite(porosity)
        & (porosity >= plan.minimum_porosity)
        & (porosity <= plan.maximum_porosity)
    )
    successful = (
        relation.successful
        & closure.successful
        & porosity_valid
        & jnp.all(jnp.isfinite(solid_velocity))
        & (jnp.linalg.norm(momentum_residual) <= 1.0e-10)
    )
    return CFDEMCouplingEvaluation(
        relation,
        closure.particle_force,
        closure.particle_torque,
        fluid_source,
        particle_impulse,
        fluid_impulse,
        porosity,
        solid_velocity,
        momentum_residual,
        successful,
        plan.plan_id,
    )


__all__ = [
    "AbstractHydrodynamicClosurePlan",
    "CFDEMCouplingEvaluation",
    "FluidParticleSample",
    "HydrodynamicClosureResult",
    "StokesDragPlan",
    "UnresolvedCFDEMCouplingPlan",
    "evaluate_unresolved_cfd_dem",
]
