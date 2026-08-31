#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.mpm import MPMParticleState, MPMRuntimeState, PreparedMPMDynamics
from ..equations import MaterialPointArguments


class MPMPhaseFieldFracturePlan(StrictModule, NonTrainableState):
    maximum_staggered_iterations: int = eqx.field(static=True)
    maximum_damage_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_staggered_iterations: int = 3,
        maximum_damage_iterations: int = 100,
        tolerance: float = 1.0e-8,
    ):
        staggered = int(maximum_staggered_iterations)
        damage = int(maximum_damage_iterations)
        tolerance_ = float(tolerance)
        if (
            staggered <= 0
            or damage <= 0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Phase-field MPM solve policy is invalid.")
        self.maximum_staggered_iterations = staggered
        self.maximum_damage_iterations = damage
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-phase-field-fracture",
                "maximum_staggered_iterations": staggered,
                "maximum_damage_iterations": damage,
                "tolerance": tolerance_,
            }
        )


class MPMPhaseFieldRuntimeState(StrictModule):
    mechanics: MPMRuntimeState
    damage: Array
    history: Array


class MPMPhaseFieldEvidence(StrictModule):
    damage_residual_norm: Array
    damage_iterations: Array
    staggered_iterations: Array
    minimum_damage_increment: Array
    fracture_energy: Array
    irreversibility_valid: Array
    successful: Array


class MPMPhaseFieldStepResult(StrictModule):
    candidate_state: MPMPhaseFieldRuntimeState
    accepted_state: MPMPhaseFieldRuntimeState
    mechanics_result: object
    evidence: MPMPhaseFieldEvidence
    successful: Array
    suggested_step: Array


def _neighbor_sum(value, periodic):
    result = jnp.zeros_like(value)
    for axis, wraps in enumerate(periodic):
        if wraps:
            result = (
                result + jnp.roll(value, 1, axis=axis) + jnp.roll(value, -1, axis=axis)
            )
        else:
            lower = jnp.take(
                value, jnp.maximum(jnp.arange(value.shape[axis]) - 1, 0), axis=axis
            )
            upper = jnp.take(
                value,
                jnp.minimum(jnp.arange(value.shape[axis]) + 1, value.shape[axis] - 1),
                axis=axis,
            )
            result = result + lower + upper
    return result


class PreparedMPMPhaseFieldDynamics(StrictModule, NonTrainableState):
    mechanics: PreparedMPMDynamics
    plan: MPMPhaseFieldFracturePlan
    spacing: tuple[float, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanics: PreparedMPMDynamics,
        plan: MPMPhaseFieldFracturePlan | None = None,
        /,
    ):
        if not isinstance(mechanics, PreparedMPMDynamics):
            raise TypeError("mechanics must be PreparedMPMDynamics.")
        plan_ = MPMPhaseFieldFracturePlan() if plan is None else plan
        if not isinstance(plan_, MPMPhaseFieldFracturePlan):
            raise TypeError("plan must be MPMPhaseFieldFracturePlan or None.")
        if mechanics.material.state_shape != (2,):
            raise ValueError(
                "Phase-field dynamics require damage/history material state."
            )
        spacing = tuple(
            float(np.diff(np.asarray(coordinates))[0])
            for coordinates in mechanics.splat.layout.coordinates_by_axis
        )
        self.mechanics = mechanics
        self.plan = plan_
        self.spacing = spacing
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mpm-phase-field",
                "mechanics": mechanics.prepared_id,
                "plan": plan_.plan_id,
                "spacing": spacing,
            }
        )

    def initialize_state(self, mechanics_state: MPMRuntimeState, /):
        history = mechanics_state.particles.material_state
        if history.shape[-1] != 2:
            raise ValueError("Phase-field material history width must be two.")
        return MPMPhaseFieldRuntimeState(mechanics_state, history[:, 0], history[:, 1])

    def _grid_field(self, routes, volume, value):
        measure = self.mechanics.splat.deposit_content(routes, volume).content
        content = self.mechanics.splat.deposit_content(routes, volume * value).content
        return jnp.where(measure > 0.0, content / measure, 0.0), measure

    def _damage_solve(self, old_damage, history, parameters):
        coefficient = tuple(
            parameters.critical_energy_release_rate * parameters.length_scale / spacing**2
            for spacing in self.spacing
        )
        neighbor_coefficient = sum(coefficient)
        diagonal = (
            parameters.critical_energy_release_rate / parameters.length_scale
            + 2.0 * history
            + 2.0 * neighbor_coefficient
        )

        def iterate(_, damage):
            neighbor = jnp.zeros_like(damage)
            for axis, value in enumerate(coefficient):
                if self.mechanics.particle_domain.periodic[axis]:
                    neighbors = jnp.roll(damage, 1, axis=axis) + jnp.roll(
                        damage, -1, axis=axis
                    )
                else:
                    lower = jnp.take(
                        damage,
                        jnp.maximum(jnp.arange(damage.shape[axis]) - 1, 0),
                        axis=axis,
                    )
                    upper = jnp.take(
                        damage,
                        jnp.minimum(
                            jnp.arange(damage.shape[axis]) + 1,
                            damage.shape[axis] - 1,
                        ),
                        axis=axis,
                    )
                    neighbors = lower + upper
                neighbor = neighbor + value * neighbors
            candidate = (2.0 * history + neighbor) / diagonal
            return jnp.clip(jnp.maximum(old_damage, candidate), 0.0, 1.0)

        damage = jax.lax.fori_loop(
            0, self.plan.maximum_damage_iterations, iterate, old_damage
        )
        neighbor = _neighbor_sum(damage, self.mechanics.particle_domain.periodic)
        average_coefficient = neighbor_coefficient / len(self.spacing)
        residual = (
            parameters.critical_energy_release_rate / parameters.length_scale * damage
            + 2.0 * history * damage
            - 2.0 * history
            + 2.0 * neighbor_coefficient * damage
            - average_coefficient * neighbor
        )
        return damage, jnp.linalg.norm(residual.reshape((-1,)))

    def step_detailed(
        self,
        state: MPMPhaseFieldRuntimeState,
        step_size: ArrayLike,
        arguments: MaterialPointArguments,
        /,
    ) -> MPMPhaseFieldStepResult:
        from ..applications.solid_mechanics._mpm_fracture import (
            MPMPhaseFieldParameters,
        )

        if not isinstance(arguments.material_parameters, MPMPhaseFieldParameters):
            raise TypeError("Phase-field dynamics require MPMPhaseFieldParameters.")
        mechanics_result = self.mechanics.step_detailed(
            state.mechanics, step_size, arguments
        )
        candidate_mechanics = mechanics_result.accepted_state
        routes = self.mechanics.splat.build(
            candidate_mechanics.particles.position,
            assignment_input=candidate_mechanics.assignment_input,
        )
        volume = candidate_mechanics.particles.reference_volume
        trial_history = candidate_mechanics.particles.material_state[:, 1]
        grid_damage, grid_measure = self._grid_field(routes, volume, state.damage)
        grid_history, _ = self._grid_field(routes, volume, trial_history)
        damage_grid, residual = self._damage_solve(
            grid_damage, grid_history, arguments.material_parameters
        )
        damage = self.mechanics.splat.gather(routes, damage_grid).values
        damage = jnp.clip(jnp.maximum(state.damage, damage), 0.0, 1.0)
        history = jnp.maximum(state.history, trial_history)
        material_state = jnp.stack((damage, history), axis=-1)
        particles = candidate_mechanics.particles
        density = self.mechanics.particles.safe_masses / jnp.where(
            self.mechanics.particles.active_mask, particles.reference_volume, 1.0
        )
        material = self.mechanics.material.evaluate(
            particles.deformation_gradient,
            material_state,
            density,
            arguments.material_parameters,
            candidate_mechanics.time,
            step_size,
        )
        updated_particles = MPMParticleState(
            particles.position,
            particles.velocity,
            particles.deformation_gradient,
            particles.affine_velocity,
            particles.reference_volume,
            material.first_piola,
            material.reference_energy_density,
            material.maximum_wave_speed,
            material.trial_state,
        )
        updated_mechanics = MPMRuntimeState(
            updated_particles,
            candidate_mechanics.time,
            candidate_mechanics.accepted_step,
            candidate_mechanics.last_status,
            candidate_mechanics.topology_generation,
            candidate_mechanics.assignment_input,
            candidate_mechanics.material_slots,
            candidate_mechanics.body_ids,
            candidate_mechanics.velocity_field_slots,
            candidate_mechanics.storage_state,
        )
        candidate = MPMPhaseFieldRuntimeState(updated_mechanics, damage, history)
        irreversibility = jnp.all(damage >= state.damage - 1.0e-12) & jnp.all(
            history >= state.history - 1.0e-12
        )
        finite = (
            jnp.all(jnp.isfinite(damage))
            & jnp.all(jnp.isfinite(history))
            & jnp.isfinite(residual)
        )
        successful = (
            mechanics_result.successful
            & material.successful.all()
            & material.admissible.all()
            & irreversibility
            & finite
            & (residual <= self.plan.tolerance)
        )
        accepted = jax.tree.map(
            lambda trial, committed: jnp.where(successful, trial, committed),
            candidate,
            state,
        )
        gradients = []
        for axis, spacing in enumerate(self.spacing):
            gradients.append(
                (jnp.roll(damage_grid, -1, axis=axis) - damage_grid) / spacing
            )
        gradient_norm = sum(jnp.sum(value * value) for value in gradients)
        cell_measure = float(np.prod(self.spacing))
        fracture_energy = cell_measure * (
            arguments.material_parameters.critical_energy_release_rate
            / (2.0 * arguments.material_parameters.length_scale)
            * jnp.sum(damage_grid**2)
            + 0.5
            * arguments.material_parameters.critical_energy_release_rate
            * arguments.material_parameters.length_scale
            * gradient_norm
        )
        evidence = MPMPhaseFieldEvidence(
            residual,
            jnp.asarray(self.plan.maximum_damage_iterations, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.min(damage - state.damage),
            fracture_energy,
            irreversibility,
            successful,
        )
        return MPMPhaseFieldStepResult(
            candidate,
            accepted,
            mechanics_result,
            evidence,
            successful,
            jnp.where(successful, jnp.asarray(step_size), 0.5 * jnp.asarray(step_size)),
        )


__all__ = [
    "MPMPhaseFieldEvidence",
    "MPMPhaseFieldFracturePlan",
    "MPMPhaseFieldRuntimeState",
    "MPMPhaseFieldStepResult",
    "PreparedMPMPhaseFieldDynamics",
]
