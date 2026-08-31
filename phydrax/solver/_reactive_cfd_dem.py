#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite, tree_where
from ..discretization.particle import (
    DEMRuntimeState,
    ParticleConversionLedger,
    ParticleConversionState,
    ParticleMorphologyEvaluation,
    ParticleRadiationEvaluation,
    RigidSphereKinematics,
)
from ..equations import (
    evaluate_unresolved_cfd_dem,
    ParticleContinuumExchangeEvaluation,
    ReactiveCFDDEMCouplingPlan,
)
from ._particle_conversion import (
    advance_particle_conversion,
    ParticleConversionSolverPlan,
)


class ReactiveCouplingMode(StrEnum):
    STRANG_FROZEN_FLUID = "strang_frozen_fluid"
    ITERATED_STAGGERED = "iterated_staggered"


class ReactiveFluidFields(StrictModule):
    velocity: Array
    density: Array
    dynamic_viscosity: Array
    pressure_gradient: Array
    temperature: Array
    species_concentration: Array


class ReactiveParticleCouplingSchedulePlan(StrictModule, NonTrainableState):
    conversion_solver: ParticleConversionSolverPlan
    dem_substeps: int = eqx.field(static=True)
    mode: ReactiveCouplingMode = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    coupling_tolerance: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        conversion_solver: ParticleConversionSolverPlan,
        /,
        *,
        dem_substeps: int,
        mode: ReactiveCouplingMode = ReactiveCouplingMode.STRANG_FROZEN_FLUID,
        maximum_iterations: int = 1,
        coupling_tolerance: float = 1.0e-6,
        relaxation: float = 1.0,
    ):
        if not isinstance(conversion_solver, ParticleConversionSolverPlan):
            raise TypeError("conversion_solver must be a ParticleConversionSolverPlan.")
        substeps = int(dem_substeps)
        iterations = int(maximum_iterations)
        tolerance = float(coupling_tolerance)
        relaxation_ = float(relaxation)
        if not isinstance(mode, ReactiveCouplingMode):
            raise TypeError("mode must be a ReactiveCouplingMode.")
        if (
            substeps <= 0
            or iterations <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or not np.isfinite(relaxation_)
            or not 0.0 < relaxation_ <= 1.0
        ):
            raise ValueError("Reactive coupling controls are invalid.")
        if mode is ReactiveCouplingMode.STRANG_FROZEN_FLUID and iterations != 1:
            raise ValueError("Frozen-fluid coupling requires exactly one iteration.")
        self.conversion_solver = conversion_solver
        self.dem_substeps = substeps
        self.mode = mode
        self.maximum_iterations = iterations
        self.coupling_tolerance = tolerance
        self.relaxation = relaxation_
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "reactive-particle-coupling-schedule",
                "conversion_solver": conversion_solver.solver_id,
                "dem_substeps": substeps,
                "mode": mode.value,
                "maximum_iterations": iterations,
                "coupling_tolerance": tolerance,
                "relaxation": relaxation_,
            }
        )


class ReactiveCFDDEMCouplingState(StrictModule):
    dem_state: DEMRuntimeState
    conversion_state: ParticleConversionState
    fluid_state: Any
    cumulative_fluid_momentum: Array
    cumulative_fluid_energy: Array
    cumulative_fluid_species: Array
    accepted_windows: Array


class ReactiveCFDDEMEvaluation(StrictModule):
    continuum_initial: ParticleContinuumExchangeEvaluation
    continuum_final: ParticleContinuumExchangeEvaluation
    morphology: ParticleMorphologyEvaluation | None
    radiation: ParticleRadiationEvaluation | None
    fluid_momentum_increment: Array
    fluid_energy_increment: Array
    fluid_species_increment: Array
    momentum_residual: Array
    energy_residual: Array
    species_residual: Array
    coupling_residual: Array
    coupling_iterations: Array
    continuum_successful: Array
    conversion_successful: Array
    dem_successful: Array
    contact_successful: Array
    fluid_successful: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)


class ReactiveCFDDEMMacroStepResult(StrictModule):
    candidate_state: ReactiveCFDDEMCouplingState
    accepted_state: ReactiveCFDDEMCouplingState
    evaluation: ReactiveCFDDEMEvaluation
    successful: Array


def initialize_reactive_cfd_dem(
    plan: ReactiveCFDDEMCouplingPlan,
    dem_state: DEMRuntimeState,
    conversion_state: ParticleConversionState,
    fluid_state: Any,
    /,
) -> ReactiveCFDDEMCouplingState:
    species = plan.continuum_exchange.species_count
    dimension = plan.dem.bodies.ambient_dimension
    dtype = dem_state.kinematics.position.dtype
    return ReactiveCFDDEMCouplingState(
        dem_state,
        conversion_state,
        fluid_state,
        jnp.zeros((dimension,), dtype=dtype),
        jnp.zeros((), dtype=dtype),
        jnp.zeros((species,), dtype=dtype),
        jnp.zeros((), dtype=jnp.int32),
    )


def advance_reactive_cfd_dem_window(
    plan: ReactiveCFDDEMCouplingPlan,
    schedule: ReactiveParticleCouplingSchedulePlan,
    state: ReactiveCFDDEMCouplingState,
    fluid_sampler: Callable[[Any], ReactiveFluidFields],
    fluid_update: Callable[[Any, Array, Array, Array, Array], Any],
    conversion_boundaries,
    dem_boundary_temperatures,
    particle_volume: Array,
    time: Array,
    step_size: Array,
    /,
    *,
    args: Any = None,
) -> ReactiveCFDDEMMacroStepResult:
    if not isinstance(plan, ReactiveCFDDEMCouplingPlan):
        raise TypeError("plan must be a ReactiveCFDDEMCouplingPlan.")
    if not isinstance(schedule, ReactiveParticleCouplingSchedulePlan):
        raise TypeError("schedule must be a ReactiveParticleCouplingSchedulePlan.")
    if not isinstance(state, ReactiveCFDDEMCouplingState):
        raise TypeError("state must be a ReactiveCFDDEMCouplingState.")
    if not callable(fluid_sampler) or not callable(fluid_update):
        raise TypeError("fluid_sampler and fluid_update must be callable.")
    dt = jnp.asarray(step_size, dtype=state.dem_state.kinematics.position.dtype)
    guess = state
    final_candidate = state
    final_evaluation = None
    residual = jnp.asarray(jnp.inf, dtype=dt.dtype)
    successful = jnp.asarray(True)
    used_iterations = jnp.zeros((), dtype=jnp.int32)
    for iteration in range(schedule.maximum_iterations):
        fields = fluid_sampler(guess.fluid_state)
        if not isinstance(fields, ReactiveFluidFields):
            raise TypeError("fluid_sampler must return ReactiveFluidFields.")
        candidate, evaluation = _advance_once(
            plan,
            schedule,
            state,
            fields,
            fluid_update,
            tuple(conversion_boundaries),
            jnp.asarray(dem_boundary_temperatures, dtype=dt.dtype),
            particle_volume,
            jnp.asarray(time),
            dt,
            args,
        )
        residual = _coupling_distance(candidate, guess)
        final_candidate = candidate
        final_evaluation = evaluation
        used_iterations = jnp.asarray(iteration + 1, dtype=jnp.int32)
        successful = successful & evaluation.successful
        if schedule.mode is ReactiveCouplingMode.ITERATED_STAGGERED:
            guess = _relax_state(guess, candidate, schedule.relaxation)
        else:
            guess = candidate
    converged = (
        jnp.asarray(True)
        if schedule.mode is ReactiveCouplingMode.STRANG_FROZEN_FLUID
        else residual <= schedule.coupling_tolerance
    )
    successful = successful & converged & tree_allfinite(final_candidate)
    final_evaluation = eqx.tree_at(
        lambda value: (
            value.coupling_residual,
            value.coupling_iterations,
            value.successful,
        ),
        final_evaluation,
        (residual, used_iterations, successful),
    )
    candidate = eqx.tree_at(
        lambda value: value.accepted_windows,
        final_candidate,
        state.accepted_windows + jnp.asarray(1, dtype=jnp.int32),
    )
    accepted = tree_where(successful, candidate, state)
    return ReactiveCFDDEMMacroStepResult(
        candidate, accepted, final_evaluation, successful
    )


def _advance_once(
    plan,
    schedule,
    state,
    fields,
    fluid_update,
    conversion_boundaries,
    dem_boundary_temperatures,
    particle_volume,
    time,
    step_size,
    args,
):
    batches = plan.conversion.batches
    thermodynamics = tuple(
        value.thermodynamics for value in plan.conversion.problem.materials
    )
    continuum_initial = plan.continuum_exchange.evaluate(
        state.dem_state.kinematics.position,
        batches,
        state.conversion_state,
        thermodynamics,
        fields.temperature,
        fields.species_concentration,
        active_mask=state.dem_state.body_properties.active,
    )
    conversion = _kick_conversion(
        state.conversion_state,
        continuum_initial.batch_internal_energy_rate,
        continuum_initial.batch_species_amount_rate,
        0.5 * step_size,
        continuum=True,
    )
    first_conversion = advance_particle_conversion(
        plan.conversion,
        schedule.conversion_solver,
        conversion,
        conversion_boundaries,
        time,
        0.5 * step_size,
    )
    conversion = first_conversion.accepted_state
    dem = state.dem_state
    momentum_increment = jnp.zeros_like(state.cumulative_fluid_momentum)
    particle_momentum_increment = jnp.zeros_like(state.cumulative_fluid_momentum)
    contact_successful = jnp.asarray(True)
    radiation_successful = jnp.asarray(True)
    last_radiation = None
    dem_successful = jnp.asarray(True)
    dem_step = step_size / schedule.dem_substeps
    for substep in range(schedule.dem_substeps):
        subtime = time + substep * dem_step
        if plan.hydrodynamics is None:
            first_hydro = None
            pre = dem
        else:
            first_hydro = evaluate_unresolved_cfd_dem(
                plan.hydrodynamics,
                dem,
                fields.velocity,
                fields.density,
                fields.dynamic_viscosity,
                fields.pressure_gradient,
                particle_volume,
                dem_step,
            )
            pre = _hydrodynamic_kick(
                plan.dem, dem, first_hydro.particle_force, 0.5 * dem_step
            )
        detail = plan.dem.step_detailed(
            jnp.asarray(substep, dtype=jnp.int32),
            subtime,
            pre,
            dem_step,
            args,
        )
        dem = detail.accepted_state
        dem_successful = dem_successful & detail.successful
        if plan.hydrodynamics is not None:
            second_hydro = evaluate_unresolved_cfd_dem(
                plan.hydrodynamics,
                dem,
                fields.velocity,
                fields.density,
                fields.dynamic_viscosity,
                fields.pressure_gradient,
                particle_volume,
                dem_step,
            )
            dem = _hydrodynamic_kick(
                plan.dem, dem, second_hydro.particle_force, 0.5 * dem_step
            )
            momentum_increment = momentum_increment + 0.5 * dem_step * jnp.sum(
                first_hydro.fluid_momentum_source_rate
                + second_hydro.fluid_momentum_source_rate,
                axis=0,
            )
            particle_momentum_increment = (
                particle_momentum_increment
                + 0.5
                * dem_step
                * jnp.sum(
                    first_hydro.particle_force + second_hydro.particle_force,
                    axis=0,
                )
            )
        if plan.contact_exchange is not None:
            contact_exchange = plan.contact_exchange.evaluate(
                plan.dem,
                detail.evaluation,
                batches,
                conversion,
                thermodynamics,
                dem_step,
                boundary_temperatures=dem_boundary_temperatures,
            )
            contact_successful = contact_successful & contact_exchange.successful
            conversion = _kick_conversion(
                conversion,
                contact_exchange.batch_internal_energy_rate,
                tuple(
                    jnp.zeros_like(value.species_amount) for value in conversion.batches
                ),
                dem_step,
                contact=True,
            )
        if plan.radiation is not None:
            last_radiation = plan.radiation.evaluate(
                plan.dem,
                detail.evaluation,
                batches,
                conversion,
                thermodynamics,
                wall_temperatures=dem_boundary_temperatures,
            )
            radiation_successful = radiation_successful & last_radiation.successful
            conversion = _kick_conversion(
                conversion,
                last_radiation.batch_internal_energy_rate,
                tuple(
                    jnp.zeros_like(value.species_amount) for value in conversion.batches
                ),
                dem_step,
                radiative=True,
            )
    second_conversion = advance_particle_conversion(
        plan.conversion,
        schedule.conversion_solver,
        conversion,
        conversion_boundaries,
        time + 0.5 * step_size,
        0.5 * step_size,
    )
    conversion = second_conversion.accepted_state
    continuum_final = plan.continuum_exchange.evaluate(
        dem.kinematics.position,
        batches,
        conversion,
        thermodynamics,
        fields.temperature,
        fields.species_concentration,
        active_mask=dem.body_properties.active,
    )
    conversion = _kick_conversion(
        conversion,
        continuum_final.batch_internal_energy_rate,
        continuum_final.batch_species_amount_rate,
        0.5 * step_size,
        continuum=True,
    )
    morphology = None
    morphology_successful = jnp.asarray(True)
    if plan.morphology is not None:
        morphology = plan.morphology.evaluate(
            batches,
            conversion,
            tuple(material.schema.molar_masses for material in thermodynamics),
        )
        conversion = plan.morphology.apply(conversion, morphology)
        body_update = plan.dem.apply_body_properties(
            time + step_size,
            dem,
            morphology.body_properties,
            morphology.neighborhood_rebuild_required,
            args=args,
        )
        dem = body_update.accepted_state
        morphology_successful = morphology.successful & body_update.successful
    fluid_energy_increment = (
        0.5
        * step_size
        * (
            continuum_initial.fluid_energy_source_rate
            + continuum_final.fluid_energy_source_rate
        )
    )
    fluid_species_increment = (
        0.5
        * step_size
        * (
            continuum_initial.fluid_species_source_rate
            + continuum_final.fluid_species_source_rate
        )
    )
    fluid = fluid_update(
        state.fluid_state,
        momentum_increment,
        fluid_energy_increment,
        fluid_species_increment,
        step_size,
    )
    energy_residual = (
        0.5
        * step_size
        * (continuum_initial.energy_residual + continuum_final.energy_residual)
    )
    species_residual = (
        0.5
        * step_size
        * (continuum_initial.species_residual + continuum_final.species_residual)
    )
    momentum_residual = momentum_increment + particle_momentum_increment
    continuum_successful = continuum_initial.successful & continuum_final.successful
    conversion_successful = first_conversion.successful & second_conversion.successful
    fluid_successful = tree_allfinite(fluid)
    successful = (
        continuum_successful
        & conversion_successful
        & dem_successful
        & contact_successful
        & radiation_successful
        & morphology_successful
        & jnp.all(
            jnp.abs(momentum_residual)
            <= 128.0
            * jnp.finfo(step_size.dtype).eps
            * jnp.maximum(
                jnp.linalg.norm(momentum_increment)
                + jnp.linalg.norm(particle_momentum_increment),
                1.0,
            )
        )
        & tree_allfinite(dem)
        & tree_allfinite(conversion)
        & fluid_successful
    )
    candidate = ReactiveCFDDEMCouplingState(
        dem,
        conversion,
        fluid,
        state.cumulative_fluid_momentum + momentum_increment,
        state.cumulative_fluid_energy + jnp.sum(fluid_energy_increment),
        state.cumulative_fluid_species + jnp.sum(fluid_species_increment, axis=0),
        state.accepted_windows,
    )
    evaluation = ReactiveCFDDEMEvaluation(
        continuum_initial,
        continuum_final,
        morphology,
        last_radiation,
        momentum_increment,
        fluid_energy_increment,
        fluid_species_increment,
        momentum_residual,
        energy_residual,
        species_residual,
        jnp.asarray(jnp.inf, dtype=step_size.dtype),
        jnp.zeros((), dtype=jnp.int32),
        continuum_successful,
        conversion_successful,
        dem_successful,
        contact_successful,
        fluid_successful,
        successful,
        plan.plan_id,
        schedule.schedule_id,
    )
    return candidate, evaluation


def _kick_conversion(
    state,
    energy_rates,
    species_rates,
    scale,
    *,
    continuum=False,
    contact=False,
    radiative=False,
):
    batches = tuple(
        eqx.tree_at(
            lambda value: (value.internal_energy, value.species_amount),
            batch,
            (
                batch.internal_energy + scale * energy,
                batch.species_amount + scale * species,
            ),
        )
        for batch, energy, species in zip(
            state.batches, energy_rates, species_rates, strict=True
        )
    )
    boundary_heat = sum(jnp.sum(value) for value in energy_rates) * scale
    species_exchange = tuple(
        old + scale * jnp.sum(rate, axis=(0, 1))
        for old, rate in zip(
            state.ledger.cumulative_species_exchange,
            species_rates,
            strict=True,
        )
    )
    ledger = ParticleConversionLedger(
        state.ledger.initial_internal_energy,
        state.ledger.initial_species_amount,
        state.ledger.cumulative_boundary_heat + (boundary_heat if continuum else 0.0),
        state.ledger.cumulative_contact_heat + (boundary_heat if contact else 0.0),
        state.ledger.cumulative_radiative_heat + (boundary_heat if radiative else 0.0),
        species_exchange,
        state.ledger.cumulative_reaction_energy,
        state.ledger.cumulative_phase_change_energy,
        state.ledger.accepted_steps,
    )
    return ParticleConversionState(batches, ledger, state.state_id)


def _hydrodynamic_kick(dynamics, state, force, scale):
    mobile = (state.body_properties.active & ~dynamics.bodies.fixed_mask)[:, None]
    velocity = state.kinematics.velocity + scale * (
        state.body_properties.inverse_masses[:, None] * force
    )
    velocity = jnp.where(mobile, velocity, 0.0)
    kinematics = RigidSphereKinematics(
        state.kinematics.position,
        velocity,
        state.kinematics.angular_velocity,
    )
    return eqx.tree_at(lambda value: value.kinematics, state, kinematics)


def _coupling_distance(left, right):
    squared = jnp.zeros((), dtype=left.dem_state.kinematics.position.dtype)
    scale = jnp.zeros_like(squared)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        if eqx.is_inexact_array(left_leaf):
            difference = left_leaf - right_leaf
            squared = squared + jnp.sum(difference * difference)
            scale = scale + jnp.sum(left_leaf * left_leaf)
    return jnp.sqrt(squared / jnp.maximum(scale, 1.0))


def _relax_state(previous, candidate, relaxation):
    return jax.tree.map(
        lambda old, new: (
            old + relaxation * (new - old) if eqx.is_inexact_array(old) else new
        ),
        previous,
        candidate,
    )


__all__ = [
    "ReactiveCFDDEMCouplingState",
    "ReactiveCFDDEMEvaluation",
    "ReactiveCFDDEMMacroStepResult",
    "ReactiveCouplingMode",
    "ReactiveFluidFields",
    "ReactiveParticleCouplingSchedulePlan",
    "advance_reactive_cfd_dem_window",
    "initialize_reactive_cfd_dem",
]
