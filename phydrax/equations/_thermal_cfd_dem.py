#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..discretization.particle import PreparedParticleGridTransfer


class ThermalCFDEMCouplingPlan(StrictModule, NonTrainableState):
    transfer: PreparedParticleGridTransfer
    particle_heat_capacity: Array
    fluid_cell_heat_capacity: Array
    heat_transfer_coefficient: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedParticleGridTransfer,
        particle_heat_capacity: ArrayLike,
        fluid_cell_heat_capacity: ArrayLike,
        heat_transfer_coefficient: ArrayLike,
        /,
    ):
        if not isinstance(transfer, PreparedParticleGridTransfer):
            raise TypeError("transfer must be PreparedParticleGridTransfer.")
        particle_capacity = np.asarray(particle_heat_capacity)
        fluid_capacity = np.asarray(fluid_cell_heat_capacity)
        coefficient = np.asarray(heat_transfer_coefficient)
        if (
            particle_capacity.shape != (transfer.particles.capacity,)
            or coefficient.shape != particle_capacity.shape
        ):
            raise ValueError(
                "Particle thermal capacities/coefficients have invalid shape."
            )
        if fluid_capacity.shape != (transfer.cell_count,):
            raise ValueError("Fluid cell heat capacity has invalid shape.")
        if (
            np.any(~np.isfinite(particle_capacity))
            or np.any(particle_capacity <= 0.0)
            or np.any(~np.isfinite(fluid_capacity))
            or np.any(fluid_capacity <= 0.0)
            or np.any(~np.isfinite(coefficient))
            or np.any(coefficient < 0.0)
        ):
            raise ValueError("Thermal coupling capacities/coefficients are invalid.")
        self.transfer = transfer
        self.particle_heat_capacity = jnp.asarray(particle_capacity)
        self.fluid_cell_heat_capacity = jnp.asarray(fluid_capacity)
        self.heat_transfer_coefficient = jnp.asarray(coefficient)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "thermal-cfd-dem-coupling-plan",
                "transfer": transfer.prepared_id,
                "values": array_tree_fingerprint(
                    {
                        "particle_capacity": particle_capacity,
                        "fluid_capacity": fluid_capacity,
                        "coefficient": coefficient,
                    }
                ),
            }
        )


class ThermalCFDEMCouplingState(StrictModule):
    particle_temperature: Array
    fluid_temperature: Array
    cumulative_particle_heat: Array
    cumulative_fluid_heat: Array
    accepted_steps: Array


class ThermalCFDEMEvaluation(StrictModule):
    particle_heat_rate: Array
    fluid_heat_source_rate: Array
    energy_residual: Array
    entropy_production: Array
    step_restriction: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ThermalCFDEMStepResult(StrictModule):
    candidate_state: ThermalCFDEMCouplingState
    accepted_state: ThermalCFDEMCouplingState
    evaluation: ThermalCFDEMEvaluation
    successful: Array


def initialize_thermal_cfd_dem(
    plan: ThermalCFDEMCouplingPlan,
    particle_temperature: ArrayLike,
    fluid_temperature: ArrayLike,
    /,
) -> ThermalCFDEMCouplingState:
    particle = jnp.asarray(particle_temperature)
    fluid = jnp.asarray(fluid_temperature, dtype=particle.dtype)
    if (
        particle.shape != plan.particle_heat_capacity.shape
        or fluid.shape != plan.fluid_cell_heat_capacity.shape
    ):
        raise ValueError("Thermal CFD-DEM temperature shapes are invalid.")
    if bool(np.any(np.asarray(particle) <= 0.0)) or bool(
        np.any(np.asarray(fluid) <= 0.0)
    ):
        raise ValueError("Thermal CFD-DEM temperatures must be positive.")
    zero = jnp.zeros((), dtype=particle.dtype)
    return ThermalCFDEMCouplingState(
        particle,
        fluid,
        zero,
        zero,
        jnp.zeros((), dtype=jnp.int32),
    )


def evaluate_thermal_cfd_dem(
    plan: ThermalCFDEMCouplingPlan,
    positions: ArrayLike,
    state: ThermalCFDEMCouplingState,
    /,
) -> ThermalCFDEMEvaluation:
    relation = plan.transfer.relation(positions)
    sampled_fluid = plan.transfer.gather(relation, state.fluid_temperature)
    heat_particle = plan.heat_transfer_coefficient * (
        sampled_fluid - state.particle_temperature
    )
    heat_particle = jnp.where(plan.transfer.particles.active_mask, heat_particle, 0.0)
    fluid_source = plan.transfer.deposit_particle_content(relation, -heat_particle)
    residual = jnp.sum(heat_particle) + jnp.sum(fluid_source)
    entropy = jnp.sum(
        jnp.where(
            plan.transfer.particles.active_mask,
            plan.heat_transfer_coefficient
            * (sampled_fluid - state.particle_temperature) ** 2
            / (sampled_fluid * state.particle_temperature),
            0.0,
        )
    )
    particle_limit = jnp.where(
        plan.heat_transfer_coefficient > 0.0,
        plan.particle_heat_capacity / plan.heat_transfer_coefficient,
        jnp.inf,
    )
    deposited_coefficient = plan.transfer.deposit_particle_content(
        relation, plan.heat_transfer_coefficient
    )
    fluid_limit = jnp.where(
        deposited_coefficient > 0.0,
        plan.fluid_cell_heat_capacity / deposited_coefficient,
        jnp.inf,
    )
    restriction = jnp.minimum(jnp.min(particle_limit), jnp.min(fluid_limit))
    successful = (
        relation.successful
        & jnp.all(jnp.isfinite(heat_particle))
        & jnp.all(jnp.isfinite(fluid_source))
        & (jnp.abs(residual) <= 1.0e-10)
        & jnp.isfinite(entropy)
        & (entropy >= 0.0)
        & ~jnp.isnan(restriction)
    )
    return ThermalCFDEMEvaluation(
        heat_particle,
        fluid_source,
        residual,
        entropy,
        restriction,
        successful,
        plan.plan_id,
    )


def step_thermal_cfd_dem(
    plan: ThermalCFDEMCouplingPlan,
    positions: ArrayLike,
    state: ThermalCFDEMCouplingState,
    step_size: ArrayLike,
    /,
) -> ThermalCFDEMStepResult:
    evaluation = evaluate_thermal_cfd_dem(plan, positions, state)
    dt = jnp.asarray(step_size, dtype=state.particle_temperature.dtype)
    particle = state.particle_temperature + dt * (
        evaluation.particle_heat_rate / plan.particle_heat_capacity
    )
    fluid = state.fluid_temperature + dt * (
        evaluation.fluid_heat_source_rate / plan.fluid_cell_heat_capacity
    )
    successful = (
        evaluation.successful
        & (dt <= evaluation.step_restriction)
        & jnp.all(jnp.isfinite(particle) & (particle > 0.0))
        & jnp.all(jnp.isfinite(fluid) & (fluid > 0.0))
    )
    candidate = ThermalCFDEMCouplingState(
        particle,
        fluid,
        state.cumulative_particle_heat + dt * jnp.sum(evaluation.particle_heat_rate),
        state.cumulative_fluid_heat + dt * jnp.sum(evaluation.fluid_heat_source_rate),
        state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
    )
    accepted = tree_where(successful, candidate, state)
    return ThermalCFDEMStepResult(candidate, accepted, evaluation, successful)


__all__ = [
    "ThermalCFDEMCouplingPlan",
    "ThermalCFDEMCouplingState",
    "ThermalCFDEMEvaluation",
    "ThermalCFDEMStepResult",
    "evaluate_thermal_cfd_dem",
    "initialize_thermal_cfd_dem",
    "step_thermal_cfd_dem",
]
