#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..discretization.particle import (
    conversion_state_admissible,
    ParticleConversionLedger,
    ParticleConversionState,
)
from ..discretization.particle._particle_internal_unstructured import (
    PreparedUnstructuredParticleInternalMesh,
)
from ..dynamics import TimeGrid
from ..equations._particle_conversion import (
    ParticleConversionEvaluation,
    ParticleConversionRejectionReason,
    PreparedParticleConversionDynamics,
)
from ..linalg import (
    DenseLinearOperator,
    LinearSystem,
    solve,
    TridiagonalLinearOperator,
)
from ._differential import DifferentialProblem
from ._rosenbrock import solve_rosenbrock


class ParticleConversionBackend(StrEnum):
    REFERENCE_ROSENBROCK = "reference_rosenbrock"
    STRUCTURED_NATIVE = "structured_native"


class ParticleConversionSolverPlan(StrictModule, NonTrainableState):
    backend: ParticleConversionBackend = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        backend: ParticleConversionBackend,
        /,
        *,
        substeps: int = 1,
        solver_id: str | None = None,
    ):
        if not isinstance(backend, ParticleConversionBackend):
            raise TypeError("backend must be a ParticleConversionBackend.")
        count = int(substeps)
        if count <= 0:
            raise ValueError("substeps must be positive.")
        generated = canonical_fingerprint(
            {
                "kind": "particle-conversion-solver-plan",
                "backend": backend.value,
                "substeps": count,
            }
        )
        self.backend = backend
        self.substeps = count
        self.solver_id = generated if solver_id is None else str(solver_id)
        if not self.solver_id:
            raise ValueError("solver_id must be nonempty.")


class ParticleConversionReplayRecord(StrictModule, NonTrainableState):
    successful: Array
    rejection_reasons: Array
    internal_energy_residual: Array
    element_residual: tuple[Array, ...]
    solver_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)


class ParticleConversionStepResult(StrictModule):
    candidate_state: ParticleConversionState
    accepted_state: ParticleConversionState
    evaluation: ParticleConversionEvaluation
    replay: ParticleConversionReplayRecord
    successful: Array


def advance_particle_conversion(
    dynamics: PreparedParticleConversionDynamics,
    plan: ParticleConversionSolverPlan,
    state: ParticleConversionState,
    boundaries,
    time: Array,
    step_size: Array,
    /,
) -> ParticleConversionStepResult:
    if not isinstance(dynamics, PreparedParticleConversionDynamics):
        raise TypeError("dynamics must be PreparedParticleConversionDynamics.")
    if not isinstance(plan, ParticleConversionSolverPlan):
        raise TypeError("plan must be a ParticleConversionSolverPlan.")
    dt = jnp.asarray(step_size)
    if plan.backend is ParticleConversionBackend.REFERENCE_ROSENBROCK:
        candidate, source_integrals, solver_successful = _reference_step(
            dynamics, state, tuple(boundaries), jnp.asarray(time), dt, plan.substeps
        )
    else:
        candidate, source_integrals, solver_successful = _structured_step(
            dynamics, state, tuple(boundaries), dt, plan.substeps
        )
    candidate = _update_ledger(state, candidate, source_integrals)
    evaluation = dynamics.evaluate(candidate, tuple(boundaries))
    admissible = conversion_state_admissible(candidate)
    successful = solver_successful & evaluation.successful & admissible
    reasons = evaluation.rejection_reasons | jnp.where(
        ~solver_successful,
        int(ParticleConversionRejectionReason.SOLVER),
        0,
    ).astype(jnp.int32)
    reasons = reasons | jnp.where(
        ~admissible,
        int(ParticleConversionRejectionReason.ADMISSIBILITY),
        0,
    ).astype(jnp.int32)
    energy_residual = (
        jnp.sum(
            jnp.stack(
                tuple(
                    jnp.sum(new.internal_energy) - jnp.sum(old.internal_energy)
                    for old, new in zip(state.batches, candidate.batches, strict=True)
                )
            )
        )
        - source_integrals[0]
        - source_integrals[2]
        - source_integrals[3]
    )
    element_residual = tuple(
        material.thermodynamics.schema.element_amount(
            jnp.sum(
                new.species_amount - old.species_amount,
                axis=(0, 1),
            )
            - source
        )
        for old, new, source, material in zip(
            state.batches,
            candidate.batches,
            source_integrals[1],
            dynamics.problem.materials,
            strict=True,
        )
    )
    energy_change_scale = jnp.sum(
        jnp.stack(
            tuple(
                jnp.sum(jnp.abs(new.internal_energy - old.internal_energy))
                for old, new in zip(state.batches, candidate.batches, strict=True)
            )
        )
    )
    energy_state_scale = jnp.sum(
        jnp.stack(
            tuple(
                jnp.sum(jnp.abs(old.internal_energy))
                + jnp.sum(jnp.abs(new.internal_energy))
                for old, new in zip(state.batches, candidate.batches, strict=True)
            )
        )
    )
    energy_tolerance = (
        4096.0
        * jnp.finfo(dt.dtype).eps
        * jnp.maximum(
            energy_state_scale
            + energy_change_scale
            + jnp.abs(source_integrals[0])
            + jnp.abs(source_integrals[2])
            + jnp.abs(source_integrals[3]),
            1.0,
        )
    )
    element_balanced = tuple(
        jnp.all(
            jnp.abs(residual)
            <= 4096.0
            * jnp.finfo(dt.dtype).eps
            * jnp.maximum(jnp.max(jnp.abs(residual)), 1.0)
        )
        for residual in element_residual
    )
    balance_successful = (jnp.abs(energy_residual) <= energy_tolerance) & jnp.all(
        jnp.stack(element_balanced)
    )
    successful = successful & balance_successful
    reasons = reasons | jnp.where(
        ~balance_successful,
        int(ParticleConversionRejectionReason.BALANCE),
        0,
    ).astype(jnp.int32)
    accepted = tree_where(successful, candidate, state)
    replay_id = canonical_fingerprint(
        {
            "kind": "particle-conversion-replay",
            "dynamics": dynamics.dynamics_id,
            "solver": plan.solver_id,
        }
    )
    replay = ParticleConversionReplayRecord(
        successful,
        reasons,
        energy_residual,
        element_residual,
        plan.solver_id,
        replay_id,
    )
    return ParticleConversionStepResult(
        candidate, accepted, evaluation, replay, successful
    )


def _reference_step(dynamics, state, boundaries, time, step_size, substeps):
    initial, main_size = _pack_state(state)
    auxiliary_size = 3 + sum(
        value.shape[-1] for value in state.ledger.initial_species_amount
    )
    initial = jnp.concatenate(
        (initial, jnp.zeros((auxiliary_size,), dtype=initial.dtype))
    )

    def drift(current_time, vector, args):
        del current_time, args
        current = _unpack_state(vector[:main_size], state)
        evaluation = dynamics.evaluate(current, boundaries)
        rates = _pack_rates(evaluation)
        boundary_heat, boundary_species, reaction_heat, phase_heat = _source_rates(
            evaluation
        )
        auxiliary = jnp.concatenate(
            (
                jnp.asarray([boundary_heat, reaction_heat, phase_heat]),
                *boundary_species,
            )
        )
        return jnp.concatenate((rates, auxiliary))

    problem = DifferentialProblem(
        drift,
        initial,
        t0=time,
        t1=time + step_size,
        problem_id=f"{dynamics.dynamics_id}:reference-conversion-step",
    )
    grid = TimeGrid(
        jnp.linspace(time, time + step_size, substeps + 1),
        time_id=f"{dynamics.dynamics_id}:reference-grid:{substeps}",
    )
    solution = solve_rosenbrock(problem, grid)
    final = solution.states[-1]
    candidate = _unpack_state(final[:main_size], state)
    auxiliary = final[main_size:]
    boundary_heat, reaction_heat, phase_heat = auxiliary[:3]
    cursor = 3
    boundary_species = []
    for value in state.ledger.initial_species_amount:
        width = value.shape[0]
        boundary_species.append(auxiliary[cursor : cursor + width])
        cursor += width
    return (
        candidate,
        (boundary_heat, tuple(boundary_species), reaction_heat, phase_heat),
        solution.successful,
    )


def _structured_step(dynamics, state, boundaries, step_size, substeps):
    current = state
    total_boundary_heat = jnp.zeros((), dtype=step_size.dtype)
    total_reaction_heat = jnp.zeros((), dtype=step_size.dtype)
    total_phase_heat = jnp.zeros((), dtype=step_size.dtype)
    total_boundary_species = tuple(
        jnp.zeros_like(value) for value in state.ledger.initial_species_amount
    )
    successful = jnp.asarray(True)
    substep = step_size / substeps
    for _ in range(substeps):
        transported_batches = []
        for prepared, batch_state, material, boundary in zip(
            dynamics.batches,
            current.batches,
            dynamics.problem.materials,
            boundaries,
            strict=True,
        ):
            transported, transport_successful = _implicit_transport_step(
                prepared, batch_state, material, boundary, substep
            )
            transported_batches.append(transported)
            successful = successful & transport_successful
        transported_state = ParticleConversionState(
            tuple(transported_batches), current.ledger, current.state_id
        )
        boundary_heat_increment = jnp.sum(
            jnp.stack(
                tuple(
                    jnp.sum(after.internal_energy - before.internal_energy)
                    for before, after in zip(
                        current.batches, transported_state.batches, strict=True
                    )
                )
            )
        )
        boundary_species_increment = tuple(
            jnp.sum(after.species_amount - before.species_amount, axis=(0, 1))
            for before, after in zip(
                current.batches, transported_state.batches, strict=True
            )
        )
        source_evaluation = dynamics.evaluate(transported_state, boundaries)
        candidate_batches = []
        reaction_heat_increment = jnp.zeros((), dtype=step_size.dtype)
        phase_heat_increment = jnp.zeros((), dtype=step_size.dtype)
        for batch_state, batch_evaluation in zip(
            transported_state.batches, source_evaluation.batches, strict=True
        ):
            reaction_energy = (
                jnp.zeros_like(batch_state.internal_energy)
                if batch_evaluation.reaction is None
                else batch_evaluation.reaction.internal_energy_rate
            )
            reaction_species = (
                jnp.zeros_like(batch_state.species_amount)
                if batch_evaluation.reaction is None
                else batch_evaluation.reaction.species_amount_rate
            )
            phase_energy = (
                jnp.zeros_like(batch_state.internal_energy)
                if batch_evaluation.phase_change is None
                else batch_evaluation.phase_change.internal_energy_rate
            )
            phase_species = (
                jnp.zeros_like(batch_state.species_amount)
                if batch_evaluation.phase_change is None
                else batch_evaluation.phase_change.species_amount_rate
            )
            reaction_heat_increment = reaction_heat_increment + substep * jnp.sum(
                reaction_energy
            )
            phase_heat_increment = phase_heat_increment + substep * jnp.sum(phase_energy)
            candidate_batches.append(
                eqx.tree_at(
                    lambda value: (value.internal_energy, value.species_amount),
                    batch_state,
                    (
                        batch_state.internal_energy
                        + substep * (reaction_energy + phase_energy),
                        batch_state.species_amount
                        + substep * (reaction_species + phase_species),
                    ),
                )
            )
        current = ParticleConversionState(
            tuple(candidate_batches), current.ledger, current.state_id
        )
        total_boundary_heat = total_boundary_heat + boundary_heat_increment
        total_reaction_heat = total_reaction_heat + reaction_heat_increment
        total_phase_heat = total_phase_heat + phase_heat_increment
        total_boundary_species = tuple(
            total + increment
            for total, increment in zip(
                total_boundary_species, boundary_species_increment, strict=True
            )
        )
        successful = (
            successful
            & source_evaluation.successful
            & conversion_state_admissible(current)
        )
    return (
        current,
        (
            total_boundary_heat,
            total_boundary_species,
            total_reaction_heat,
            total_phase_heat,
        ),
        successful,
    )


def _implicit_transport_step(prepared, state, material, boundary, step_size):
    if isinstance(prepared.mesh, PreparedUnstructuredParticleInternalMesh):
        return _implicit_unstructured_transport_step(
            prepared, state, material, boundary, step_size
        )
    metrics = prepared.mesh.metrics(state.outer_scale)
    thermo = material.thermodynamics.state(
        state.internal_energy,
        state.species_amount,
        metrics.cell_measures,
        state.porosity,
    )
    amount_sum = jnp.sum(state.species_amount, axis=-1, keepdims=True)
    fraction = state.species_amount / jnp.maximum(amount_sum, 1.0e-30)
    conductivity = jnp.sum(fraction * material.transport.thermal_conductivity, axis=-1)
    face_conductivity = _harmonic_mean(conductivity[:, :-1], conductivity[:, 1:])
    heat_conductance = (
        face_conductivity * metrics.face_measures[:, 1:-1] / metrics.center_distances
    )
    heat_boundary = boundary.heat_transfer_coefficient * metrics.surface_measure
    temperatures = []
    linear_successful = jnp.asarray(True)
    for particle in range(prepared.particle_count):
        capacity = thermo.heat_capacity[particle]
        conductance = heat_conductance[particle]
        diagonal = capacity / step_size
        diagonal = diagonal.at[:-1].add(conductance)
        diagonal = diagonal.at[1:].add(conductance)
        diagonal = diagonal.at[-1].add(heat_boundary[particle])
        off = -conductance
        right = capacity / step_size * thermo.temperature[particle]
        right = right.at[-1].add(
            heat_boundary[particle] * boundary.temperature[particle]
            + boundary.prescribed_heat_rate[particle]
        )
        result = solve(LinearSystem(TridiagonalLinearOperator(off, diagonal, off)), right)
        temperatures.append(result.value)
        linear_successful = linear_successful & result.successful
    temperature = jnp.stack(temperatures)
    energy = state.internal_energy + thermo.heat_capacity * (
        temperature - thermo.temperature
    )
    effective_diffusivity = (
        material.transport.species_diffusivity[None, None, :]
        * state.porosity[:, :, None] ** material.transport.tortuosity_exponent
    )
    face_diffusivity = _harmonic_mean(
        effective_diffusivity[:, :-1, :],
        effective_diffusivity[:, 1:, :],
    )
    conductance = (
        face_diffusivity
        * metrics.face_measures[:, 1:-1, None]
        / metrics.center_distances[:, :, None]
    )
    species = jnp.zeros_like(state.species_amount)
    for particle in range(prepared.particle_count):
        for species_index in range(prepared.species_count):
            face = conductance[particle, :, species_index]
            volume = metrics.cell_measures[particle]
            boundary_conductance = (
                boundary.mass_transfer_coefficient[particle, species_index]
                * metrics.surface_measure[particle]
            )
            diagonal = jnp.ones((prepared.cell_capacity,), dtype=energy.dtype) / step_size
            diagonal = diagonal.at[:-1].add(face / volume[:-1])
            diagonal = diagonal.at[1:].add(face / volume[1:])
            diagonal = diagonal.at[-1].add(boundary_conductance / volume[-1])
            lower = -face / volume[:-1]
            upper = -face / volume[1:]
            right = state.species_amount[particle, :, species_index] / step_size
            right = right.at[-1].add(
                boundary_conductance
                * boundary.species_concentration[particle, species_index]
                + boundary.prescribed_species_rate[particle, species_index]
            )
            result = solve(
                LinearSystem(TridiagonalLinearOperator(lower, diagonal, upper)),
                right,
            )
            species = species.at[particle, :, species_index].set(result.value)
            linear_successful = linear_successful & result.successful
    candidate = eqx.tree_at(
        lambda value: (value.internal_energy, value.species_amount),
        state,
        (energy, species),
    )
    successful = (
        linear_successful
        & jnp.all(jnp.isfinite(energy))
        & jnp.all(jnp.isfinite(species) & (species >= 0.0))
        & metrics.successful
    )
    return candidate, successful


def _implicit_unstructured_transport_step(
    prepared,
    state,
    material,
    boundary,
    step_size,
):
    active_cells = jnp.broadcast_to(
        state.active[:, None], (prepared.particle_count, prepared.cell_capacity)
    )
    metrics = prepared.mesh.metrics(state.outer_scale, active_cells=active_cells)
    thermo = material.thermodynamics.state(
        state.internal_energy,
        state.species_amount,
        metrics.cell_measures,
        state.porosity,
    )
    owner = metrics.owner_cells
    neighbour = metrics.neighbour_cells
    safe_neighbour = jnp.maximum(neighbour, 0)
    interior = (~metrics.boundary_faces)[None, :] & metrics.active_faces
    amount_sum = jnp.sum(state.species_amount, axis=-1, keepdims=True)
    fraction = state.species_amount / jnp.maximum(amount_sum, 1.0e-30)
    conductivity = jnp.sum(fraction * material.transport.thermal_conductivity, axis=-1)
    face_conductivity = _harmonic_mean(
        conductivity[:, owner], conductivity[:, safe_neighbour]
    )
    heat_conductance = (
        face_conductivity * metrics.face_measures / metrics.center_distances
    )
    temperature_values = []
    linear_successful = jnp.asarray(True)
    for particle in range(prepared.particle_count):
        capacity = thermo.heat_capacity[particle]
        matrix = jnp.diag(capacity / step_size)
        face = jnp.where(interior[particle], heat_conductance[particle], 0.0)
        matrix = matrix.at[owner, owner].add(face)
        matrix = matrix.at[safe_neighbour, safe_neighbour].add(face)
        matrix = matrix.at[owner, safe_neighbour].add(-face)
        matrix = matrix.at[safe_neighbour, owner].add(-face)
        boundary_face = metrics.boundary_faces & metrics.active_faces[particle]
        boundary_conductance = jnp.where(
            boundary_face,
            boundary.heat_transfer_coefficient[particle]
            * metrics.face_measures[particle],
            0.0,
        )
        matrix = matrix.at[owner, owner].add(boundary_conductance)
        area_fraction = metrics.face_measures[particle] / jnp.maximum(
            metrics.surface_measure[particle], 1.0e-30
        )
        right = capacity / step_size * thermo.temperature[particle]
        right = right.at[owner].add(
            boundary_conductance * boundary.temperature[particle]
            + jnp.where(
                boundary_face,
                boundary.prescribed_heat_rate[particle] * area_fraction,
                0.0,
            )
        )
        result = solve(LinearSystem(DenseLinearOperator(matrix)), right)
        temperature_values.append(result.value)
        linear_successful = linear_successful & result.successful
    temperature = jnp.stack(temperature_values)
    energy = state.internal_energy + thermo.heat_capacity * (
        temperature - thermo.temperature
    )
    effective_diffusivity = (
        material.transport.species_diffusivity[None, None, :]
        * state.porosity[:, :, None] ** material.transport.tortuosity_exponent
    )
    face_diffusivity = _harmonic_mean(
        effective_diffusivity[:, owner, :],
        effective_diffusivity[:, safe_neighbour, :],
    )
    species_conductance = (
        face_diffusivity
        * metrics.face_measures[:, :, None]
        / metrics.center_distances[:, :, None]
    )
    species = jnp.zeros_like(state.species_amount)
    for particle in range(prepared.particle_count):
        volume = metrics.cell_measures[particle]
        boundary_face = metrics.boundary_faces & metrics.active_faces[particle]
        area_fraction = metrics.face_measures[particle] / jnp.maximum(
            metrics.surface_measure[particle], 1.0e-30
        )
        for species_index in range(prepared.species_count):
            face = jnp.where(
                interior[particle],
                species_conductance[particle, :, species_index],
                0.0,
            )
            matrix = jnp.eye(prepared.cell_capacity, dtype=energy.dtype) / step_size
            matrix = matrix.at[owner, owner].add(face / volume[owner])
            matrix = matrix.at[safe_neighbour, safe_neighbour].add(
                face / volume[safe_neighbour]
            )
            matrix = matrix.at[owner, safe_neighbour].add(-face / volume[owner])
            matrix = matrix.at[safe_neighbour, owner].add(-face / volume[safe_neighbour])
            boundary_conductance = jnp.where(
                boundary_face,
                boundary.mass_transfer_coefficient[particle, species_index]
                * metrics.face_measures[particle],
                0.0,
            )
            matrix = matrix.at[owner, owner].add(boundary_conductance / volume[owner])
            right = state.species_amount[particle, :, species_index] / step_size
            right = right.at[owner].add(
                boundary_conductance
                * boundary.species_concentration[particle, species_index]
                + jnp.where(
                    boundary_face,
                    boundary.prescribed_species_rate[particle, species_index]
                    * area_fraction,
                    0.0,
                )
            )
            result = solve(LinearSystem(DenseLinearOperator(matrix)), right)
            species = species.at[particle, :, species_index].set(result.value)
            linear_successful = linear_successful & result.successful
    candidate = eqx.tree_at(
        lambda value: (value.internal_energy, value.species_amount),
        state,
        (energy, species),
    )
    successful = (
        linear_successful
        & jnp.all(jnp.isfinite(energy))
        & jnp.all(jnp.isfinite(species) & (species >= 0.0))
        & metrics.successful
    )
    return candidate, successful


def _pack_state(state):
    values = []
    for batch in state.batches:
        values.append(batch.internal_energy.reshape(-1))
        values.append(batch.species_amount.reshape(-1))
    packed = jnp.concatenate(tuple(values))
    return packed, packed.shape[0]


def _unpack_state(vector, template):
    cursor = 0
    batches = []
    for batch in template.batches:
        energy_size = batch.internal_energy.size
        species_size = batch.species_amount.size
        energy = vector[cursor : cursor + energy_size].reshape(
            batch.internal_energy.shape
        )
        cursor += energy_size
        species = vector[cursor : cursor + species_size].reshape(
            batch.species_amount.shape
        )
        cursor += species_size
        batches.append(
            eqx.tree_at(
                lambda value: (value.internal_energy, value.species_amount),
                batch,
                (energy, species),
            )
        )
    return ParticleConversionState(tuple(batches), template.ledger, template.state_id)


def _pack_rates(evaluation):
    values = []
    for batch in evaluation.batches:
        values.append(batch.internal_energy_rate.reshape(-1))
        values.append(batch.species_amount_rate.reshape(-1))
    return jnp.concatenate(tuple(values))


def _source_rates(evaluation):
    boundary_heat = jnp.sum(
        jnp.stack(
            tuple(
                jnp.sum(value.transport.boundary_heat_rate)
                for value in evaluation.batches
            )
        )
    )
    boundary_species = tuple(
        jnp.sum(value.transport.boundary_species_rate, axis=0)
        for value in evaluation.batches
    )
    reaction_heat = jnp.sum(
        jnp.stack(
            tuple(
                jnp.zeros((), dtype=value.internal_energy_rate.dtype)
                if value.reaction is None
                else jnp.sum(value.reaction.internal_energy_rate)
                for value in evaluation.batches
            )
        )
    )
    phase_heat = jnp.sum(
        jnp.stack(
            tuple(
                jnp.zeros((), dtype=value.internal_energy_rate.dtype)
                if value.phase_change is None
                else jnp.sum(value.phase_change.internal_energy_rate)
                for value in evaluation.batches
            )
        )
    )
    return boundary_heat, boundary_species, reaction_heat, phase_heat


def _update_ledger(previous, candidate, sources):
    boundary_heat, boundary_species, reaction_heat, phase_heat = sources
    ledger = ParticleConversionLedger(
        previous.ledger.initial_internal_energy,
        previous.ledger.initial_species_amount,
        previous.ledger.cumulative_boundary_heat + boundary_heat,
        previous.ledger.cumulative_contact_heat,
        previous.ledger.cumulative_radiative_heat,
        tuple(
            old + increment
            for old, increment in zip(
                previous.ledger.cumulative_species_exchange,
                boundary_species,
                strict=True,
            )
        ),
        previous.ledger.cumulative_reaction_energy + reaction_heat,
        previous.ledger.cumulative_phase_change_energy + phase_heat,
        previous.ledger.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
    )
    return ParticleConversionState(candidate.batches, ledger, candidate.state_id)


def _harmonic_mean(left, right):
    denominator = left + right
    return jnp.where(denominator > 0.0, 2.0 * left * right / denominator, 0.0)


__all__ = [
    "ParticleConversionBackend",
    "ParticleConversionReplayRecord",
    "ParticleConversionSolverPlan",
    "ParticleConversionStepResult",
    "advance_particle_conversion",
]
