#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._driven_stress import (
    _atomistic_driven_stress_from_components,
    AtomisticDrivenStressPlan,
    AtomisticDrivenStressResult,
)
from .._dynamics import PreparedAtomisticDynamics
from ._cell import EvolvingFlowCellPlan, EvolvingFlowCellState, EvolvingFlowCellStepResult


SLLODThermostatKind = Literal["none", "gaussian-isokinetic"]


class SLLODIntegratorPlan(StrictModule, NonTrainableState):
    time_step: float = eqx.field(static=True)
    thermostat: SLLODThermostatKind = eqx.field(static=True)
    kinetic_relative_tolerance: float = eqx.field(static=True)
    maximum_displacement: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_step: float,
        /,
        *,
        thermostat: SLLODThermostatKind = "gaussian-isokinetic",
        kinetic_relative_tolerance: float = 1.0e-6,
        maximum_displacement: float | None = None,
    ):
        step = float(time_step)
        maximum = None if maximum_displacement is None else float(maximum_displacement)
        kinetic_tolerance = float(kinetic_relative_tolerance)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("SLLOD time_step must be finite and positive.")
        if thermostat not in ("none", "gaussian-isokinetic"):
            raise ValueError("Unknown SLLOD thermostat.")
        if not math.isfinite(kinetic_tolerance) or kinetic_tolerance < 0.0:
            raise ValueError("kinetic_relative_tolerance must be finite and nonnegative.")
        if maximum is not None and (not math.isfinite(maximum) or maximum <= 0.0):
            raise ValueError("maximum_displacement must be positive or None.")
        self.time_step = step
        self.thermostat = thermostat
        self.kinetic_relative_tolerance = kinetic_tolerance
        self.maximum_displacement = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sllod-integrator",
                "time_step": step,
                "thermostat": thermostat,
                "kinetic_relative_tolerance": kinetic_tolerance,
                "maximum_displacement": maximum,
                "momentum_frame": "peculiar",
            }
        )

    def prepare(
        self,
        dynamics: PreparedAtomisticDynamics,
        cell: EvolvingFlowCellPlan,
        /,
    ) -> PreparedSLLODIntegrator:
        return PreparedSLLODIntegrator(self, dynamics, cell)


class SLLODState(StrictModule):
    positions: Array
    image_counts: Array
    peculiar_momenta: Array
    forces: Array
    virial: Array
    potential_energy: Array
    flow_cell: EvolvingFlowCellState
    step_index: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class SLLODStepResult(StrictModule):
    candidate_state: SLLODState
    accepted_state: SLLODState
    cell: EvolvingFlowCellStepResult
    stress: AtomisticDrivenStressResult
    thermostat_multiplier: Array
    peculiar_kinetic_energy: Array
    kinetic_energy_residual: Array
    displacement_norm: Array
    accepted: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedSLLODIntegrator(StrictModule, NonTrainableState):
    plan: SLLODIntegratorPlan
    dynamics: PreparedAtomisticDynamics
    cell: EvolvingFlowCellPlan
    stress_plan: AtomisticDrivenStressPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SLLODIntegratorPlan,
        dynamics: PreparedAtomisticDynamics,
        cell: EvolvingFlowCellPlan,
        /,
    ):
        if not isinstance(plan, SLLODIntegratorPlan):
            raise TypeError("plan must be SLLODIntegratorPlan.")
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(cell, EvolvingFlowCellPlan):
            raise TypeError("cell must be EvolvingFlowCellPlan.")
        if (
            dynamics.system.cell is None
            or dynamics.system.cell.cell_id != cell.cell.cell_id
        ):
            raise ValueError(
                "SLLOD dynamics and evolving cell must share the reference cell."
            )
        if dynamics.constraints is not None:
            raise ValueError("This SLLOD route does not admit holonomic constraints.")
        self.plan = plan
        self.dynamics = dynamics
        self.cell = cell
        self.stress_plan = AtomisticDrivenStressPlan()
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-sllod-integrator",
                "plan": plan.plan_id,
                "dynamics": dynamics.prepared_id,
                "cell": cell.plan_id,
            }
        )

    def _evaluate(self, positions: Array, image_counts: Array, cell_vectors: Array, /):
        neighborhood = self.dynamics.neighborhood.build(positions)
        unwrapped = positions + contract(
            "ni,ij->nj", image_counts.astype(positions.dtype), cell_vectors
        )
        cell = self.cell.cell
        evaluation = self.dynamics.potential.evaluate(
            positions,
            neighborhood,
            unwrapped_positions=unwrapped,
            fractional_positions=cell.fractional_with_vectors(positions, cell_vectors),
            species=self.dynamics.system.plan.atom_type_ids,
            cell=cell,
            cell_vectors=cell_vectors,
        )
        return evaluation, unwrapped

    def initialize(
        self,
        positions: ArrayLike,
        peculiar_velocities: ArrayLike,
        /,
    ) -> SLLODState:
        coordinate = jnp.asarray(
            positions, dtype=self.dynamics.system.plan.coordinate_dtype
        )
        velocity = jnp.asarray(peculiar_velocities, dtype=coordinate.dtype)
        expected = (self.dynamics.system.capacity, 3)
        if coordinate.shape != expected or velocity.shape != expected:
            raise ValueError(
                "SLLOD positions and peculiar velocities must match system capacity."
            )
        flow_cell = self.cell.initialize()
        wrapped, images = self.cell.cell.wrap_with_vectors(coordinate, flow_cell.vectors)
        evaluation, _ = self._evaluate(wrapped, images, flow_cell.vectors)
        active = self.dynamics.system.active_mask
        masses = self.dynamics.system.plan.masses.astype(coordinate.dtype)
        momenta = jnp.where(active[:, None], masses[:, None] * velocity, 0.0)
        successful = (
            evaluation.successful
            & jnp.all(jnp.isfinite(momenta))
            & jnp.all(jnp.isfinite(coordinate))
        )
        return SLLODState(
            wrapped,
            images,
            momenta,
            evaluation.forces,
            evaluation.virial,
            evaluation.energy,
            flow_cell,
            jnp.asarray(0, dtype=jnp.int32),
            successful,
            self.prepared_id,
        )

    def _thermostat_multiplier(
        self, momenta: Array, forces: Array, gradient: Array
    ) -> Array:
        if self.plan.thermostat == "none":
            return jnp.asarray(0.0, dtype=momenta.dtype)
        masses = self.dynamics.system.plan.masses.astype(momenta.dtype)
        active = self.dynamics.system.active_mask
        streaming = contract("ij,nj->ni", gradient, momenta)
        inverse_mass = jnp.where(active, 1.0 / masses, 0.0)
        numerator = jnp.sum(inverse_mass[:, None] * momenta * (forces - streaming))
        denominator = jnp.sum(inverse_mass[:, None] * momenta * momenta)
        return jnp.where(
            denominator > jnp.finfo(momenta.dtype).tiny,
            numerator / denominator,
            0.0,
        )

    def step(
        self,
        state: SLLODState,
        velocity_gradient: ArrayLike,
        /,
    ) -> SLLODStepResult:
        if not isinstance(state, SLLODState) or state.prepared_id != self.prepared_id:
            raise ValueError("SLLOD state does not belong to this prepared integrator.")
        gradient = jnp.asarray(velocity_gradient, dtype=state.positions.dtype)
        if gradient.shape != (3, 3):
            raise ValueError("velocity_gradient must have shape (3, 3).")
        step = self.plan.time_step
        masses = self.dynamics.system.plan.masses.astype(state.positions.dtype)
        active = self.dynamics.system.active_mask
        inverse_mass = jnp.where(active, 1.0 / masses, 0.0)
        unwrapped = state.positions + contract(
            "ni,ij->nj",
            state.image_counts.astype(state.positions.dtype),
            state.flow_cell.vectors,
        )
        initial_kinetic = 0.5 * jnp.sum(
            inverse_mass[:, None] * state.peculiar_momenta * state.peculiar_momenta
        )
        alpha = self._thermostat_multiplier(
            state.peculiar_momenta, state.forces, gradient
        )
        streaming_momentum = contract("ij,nj->ni", gradient, state.peculiar_momenta)
        momentum_rate = state.forces - streaming_momentum - alpha * state.peculiar_momenta
        half_momenta = jnp.where(
            active[:, None], state.peculiar_momenta + 0.5 * step * momentum_rate, 0.0
        )
        peculiar_velocity = inverse_mass[:, None] * half_momenta
        initial_position_rate = peculiar_velocity + contract(
            "ij,nj->ni", gradient, unwrapped
        )
        midpoint = unwrapped + 0.5 * step * initial_position_rate
        proposed_unwrapped = unwrapped + step * (
            peculiar_velocity + contract("ij,nj->ni", gradient, midpoint)
        )
        cell_result = self.cell.step(state.flow_cell, gradient, step)
        next_vectors = cell_result.accepted_state.vectors
        wrapped, images = self.cell.cell.wrap_with_vectors(
            proposed_unwrapped, next_vectors
        )
        evaluation, reconstructed = self._evaluate(wrapped, images, next_vectors)
        next_alpha = self._thermostat_multiplier(
            half_momenta, evaluation.forces, gradient
        )
        next_streaming = contract("ij,nj->ni", gradient, half_momenta)
        final_momenta = jnp.where(
            active[:, None],
            half_momenta
            + 0.5
            * step
            * (evaluation.forces - next_streaming - next_alpha * half_momenta),
            0.0,
        )
        final_kinetic = 0.5 * jnp.sum(
            inverse_mass[:, None] * final_momenta * final_momenta
        )
        kinetic_residual = final_kinetic - initial_kinetic
        kinetic_valid = (self.plan.thermostat == "none") | (
            jnp.abs(kinetic_residual)
            <= self.plan.kinetic_relative_tolerance
            * jnp.maximum(jnp.abs(initial_kinetic), 1.0)
        )
        displacement = reconstructed - unwrapped
        displacement_norm = jnp.sqrt(jnp.sum(displacement * displacement))
        displacement_valid = (
            jnp.asarray(True)
            if self.plan.maximum_displacement is None
            else displacement_norm <= self.plan.maximum_displacement
        )
        finite = (
            jnp.all(jnp.isfinite(final_momenta))
            & jnp.all(jnp.isfinite(wrapped))
            & jnp.isfinite(final_kinetic)
            & jnp.isfinite(alpha)
            & jnp.isfinite(next_alpha)
        )
        accepted = (
            state.successful
            & cell_result.accepted
            & evaluation.successful
            & finite
            & displacement_valid
            & kinetic_valid
        )
        candidate = SLLODState(
            wrapped,
            images,
            final_momenta,
            evaluation.forces,
            evaluation.virial,
            evaluation.energy,
            cell_result.accepted_state,
            state.step_index + 1,
            state.successful & accepted,
            self.prepared_id,
        )
        accepted_state = SLLODState(
            jnp.where(accepted, candidate.positions, state.positions),
            jnp.where(accepted, candidate.image_counts, state.image_counts),
            jnp.where(accepted, candidate.peculiar_momenta, state.peculiar_momenta),
            jnp.where(accepted, candidate.forces, state.forces),
            jnp.where(accepted, candidate.virial, state.virial),
            jnp.where(accepted, candidate.potential_energy, state.potential_energy),
            EvolvingFlowCellState(
                jnp.where(accepted, candidate.flow_cell.vectors, state.flow_cell.vectors),
                jnp.where(
                    accepted,
                    candidate.flow_cell.deformation_gradient,
                    state.flow_cell.deformation_gradient,
                ),
                jnp.where(
                    accepted,
                    candidate.flow_cell.lattice_transform,
                    state.flow_cell.lattice_transform,
                ),
                jnp.where(accepted, candidate.flow_cell.time, state.flow_cell.time),
                jnp.where(
                    accepted, candidate.flow_cell.step_index, state.flow_cell.step_index
                ),
                jnp.where(
                    accepted, candidate.flow_cell.remap_count, state.flow_cell.remap_count
                ),
                state.flow_cell.successful,
                self.cell.plan_id,
            ),
            jnp.where(accepted, candidate.step_index, state.step_index),
            state.successful,
            self.prepared_id,
        )
        stress = _atomistic_driven_stress_from_components(
            self.stress_plan,
            accepted_state.peculiar_momenta,
            self.dynamics.system.inverse_masses,
            self.dynamics.system.mobile_mask,
            accepted_state.virial,
            accepted_state.flow_cell.vectors,
            gradient,
            self.dynamics.system.plan.units.kinetic_to_energy,
            accepted_state.successful,
        )
        successful = accepted & stress.successful
        return SLLODStepResult(
            candidate,
            accepted_state,
            cell_result,
            stress,
            0.5 * (alpha + next_alpha),
            final_kinetic,
            kinetic_residual,
            displacement_norm,
            accepted,
            successful,
            self.prepared_id,
        )


__all__ = [
    "PreparedSLLODIntegrator",
    "SLLODIntegratorPlan",
    "SLLODState",
    "SLLODStepResult",
    "SLLODThermostatKind",
]
