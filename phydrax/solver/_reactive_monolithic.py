#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite, tree_where
from ..discretization.particle import ParticleConversionState
from ..equations._particle_thermochemistry import ParticleTransportBoundary
from ..equations._reactive_monolithic import (
    ReactiveFluidImplicitState,
    ReactiveMonolithicCouplingPlan,
    ReactiveMonolithicStage,
    ReactiveMonolithicUnknown,
)
from ..linalg import PyTreeSpace
from ..nonlinear import (
    FunctionLeftNonlinearPreconditioner,
    LeftPreconditionedSystem,
    NewtonKrylov,
    NewtonTrustRegion,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    solve_prepared_nonlinear,
)


class ReactiveMonolithicPreconditionerMode(StrEnum):
    LOCAL_BLOCK = "local_block"
    BLOCK_FACTORIZATION = "block_factorization"
    SCHUR_COMPLEMENT = "schur_complement"


class ReactiveMonolithicSolverPlan(StrictModule, NonTrainableState):
    method: NewtonKrylov | NewtonTrustRegion
    termination: NonlinearTermination
    precision: NonlinearPrecisionPolicy
    preconditioner_mode: ReactiveMonolithicPreconditionerMode = eqx.field(static=True)
    event_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        method: NewtonKrylov | NewtonTrustRegion | None = None,
        termination: NonlinearTermination | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
        preconditioner_mode: ReactiveMonolithicPreconditionerMode = (
            ReactiveMonolithicPreconditionerMode.LOCAL_BLOCK
        ),
        event_margin: float = 1.0e-10,
    ):
        method_ = NewtonKrylov() if method is None else method
        termination_ = (
            NonlinearTermination(
                absolute_residual=1.0e-10,
                relative_residual=1.0e-8,
                maximum_steps=24,
                maximum_evaluations=256,
                maximum_linear_iterations=2048,
            )
            if termination is None
            else termination
        )
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        margin = float(event_margin)
        if not isinstance(method_, (NewtonKrylov, NewtonTrustRegion)):
            raise TypeError("method must be NewtonKrylov or NewtonTrustRegion.")
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy.")
        if not isinstance(preconditioner_mode, ReactiveMonolithicPreconditionerMode):
            raise TypeError("preconditioner_mode is invalid.")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("event_margin must be finite and nonnegative.")
        self.method = method_
        self.termination = termination_
        self.precision = precision_
        self.preconditioner_mode = preconditioner_mode
        self.event_margin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reactive-monolithic-solver-plan",
                "method": method_.method_id,
                "preconditioner": preconditioner_mode.value,
                "event_margin": margin,
                "precision": precision_.policy_id,
            }
        )


class ReactiveMonolithicState(StrictModule):
    fluid: ReactiveFluidImplicitState
    conversion: ParticleConversionState
    particle_velocity: Array
    accepted_windows: Array
    state_id: str = eqx.field(static=True)


class ReactiveMonolithicPreconditionerEvidence(StrictModule):
    mode: ReactiveMonolithicPreconditionerMode = eqx.field(static=True)
    fluid_block_count: int = eqx.field(static=True)
    particle_block_count: int = eqx.field(static=True)
    conversion_block_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedReactiveMonolithicStep(StrictModule, NonTrainableState):
    coupling: ReactiveMonolithicCouplingPlan
    solver: ReactiveMonolithicSolverPlan
    stage: ReactiveMonolithicStage
    initial_unknown: ReactiveMonolithicUnknown
    physical_problem: NonlinearSystemProblem
    transformation: LeftPreconditionedSystem
    nonlinear: PreparedNonlinearSolve
    preconditioner: ReactiveMonolithicPreconditionerEvidence
    prepared_id: str = eqx.field(static=True)


class ReactiveMonolithicStepResult(StrictModule):
    candidate_state: ReactiveMonolithicState
    accepted_state: ReactiveMonolithicState
    nonlinear: NonlinearResult
    evaluation: object
    preconditioner: ReactiveMonolithicPreconditionerEvidence
    event_split_required: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


def initialize_reactive_monolithic_state(
    coupling: ReactiveMonolithicCouplingPlan,
    fluid: ReactiveFluidImplicitState,
    conversion: ParticleConversionState,
    particle_velocity: ArrayLike,
    /,
) -> ReactiveMonolithicState:
    if not isinstance(coupling, ReactiveMonolithicCouplingPlan):
        raise TypeError("coupling must be ReactiveMonolithicCouplingPlan.")
    coupling.fluid.validate_state(fluid)
    velocity = jnp.asarray(particle_velocity, dtype=fluid.velocity.dtype)
    capacity = coupling.continuum_exchange.transfer.particle_capacity
    dimension = fluid.velocity.shape[1]
    if velocity.shape != (capacity, dimension):
        raise ValueError("particle_velocity must have particle-dimension shape.")
    identifier = canonical_fingerprint(
        {
            "kind": "reactive-monolithic-state",
            "coupling": coupling.plan_id,
            "conversion": conversion.state_id,
        }
    )
    return ReactiveMonolithicState(
        fluid,
        conversion,
        velocity,
        jnp.zeros((), dtype=jnp.int32),
        identifier,
    )


def make_reactive_monolithic_stage(
    coupling: ReactiveMonolithicCouplingPlan,
    state: ReactiveMonolithicState,
    particle_position: ArrayLike,
    particle_mass: ArrayLike,
    particle_active: ArrayLike,
    conversion_boundaries,
    time: ArrayLike,
    step_size: ArrayLike,
    /,
    *,
    contact_energy_rate=None,
    radiative_energy_rate=None,
    external_fluid_momentum_rate=None,
    external_fluid_energy_rate=None,
    external_fluid_species_rate=None,
) -> ReactiveMonolithicStage:
    position = jnp.asarray(particle_position, dtype=state.fluid.velocity.dtype)
    mass = jnp.asarray(particle_mass, dtype=state.fluid.velocity.dtype)
    active = jnp.asarray(particle_active, dtype=bool)
    capacity = coupling.continuum_exchange.transfer.particle_capacity
    dimension = state.fluid.velocity.shape[1]
    if position.shape != (capacity, dimension):
        raise ValueError("particle_position must have particle-dimension shape.")
    if mass.shape != (capacity,) or active.shape != (capacity,):
        raise ValueError("particle mass/activity must have particle shape.")
    boundaries = tuple(conversion_boundaries)
    if len(boundaries) != len(state.conversion.batches) or any(
        not isinstance(value, ParticleTransportBoundary) for value in boundaries
    ):
        raise ValueError("conversion_boundaries do not match conversion batches.")
    contact = (
        tuple(jnp.zeros_like(value.internal_energy) for value in state.conversion.batches)
        if contact_energy_rate is None
        else tuple(jnp.asarray(value) for value in contact_energy_rate)
    )
    radiation = (
        tuple(jnp.zeros_like(value.internal_energy) for value in state.conversion.batches)
        if radiative_energy_rate is None
        else tuple(jnp.asarray(value) for value in radiative_energy_rate)
    )
    if len(contact) != len(state.conversion.batches) or len(radiation) != len(
        state.conversion.batches
    ):
        raise ValueError("Contact/radiative rates do not match conversion batches.")
    fluid_momentum = (
        jnp.zeros_like(state.fluid.velocity)
        if external_fluid_momentum_rate is None
        else jnp.asarray(external_fluid_momentum_rate, dtype=state.fluid.velocity.dtype)
    )
    fluid_energy = (
        jnp.zeros_like(state.fluid.temperature)
        if external_fluid_energy_rate is None
        else jnp.asarray(external_fluid_energy_rate, dtype=state.fluid.temperature.dtype)
    )
    fluid_species = (
        jnp.zeros_like(state.fluid.species_concentration)
        if external_fluid_species_rate is None
        else jnp.asarray(
            external_fluid_species_rate,
            dtype=state.fluid.species_concentration.dtype,
        )
    )
    dt = jnp.asarray(step_size, dtype=state.fluid.temperature.dtype).reshape(())
    time_ = jnp.asarray(time, dtype=state.fluid.temperature.dtype).reshape(())
    stage_id = canonical_fingerprint(
        {
            "kind": "reactive-monolithic-stage",
            "coupling": coupling.plan_id,
            "state": state.state_id,
        }
    )
    return ReactiveMonolithicStage(
        state.fluid,
        state.conversion,
        state.particle_velocity,
        position,
        mass,
        active,
        boundaries,
        contact,
        radiation,
        fluid_momentum,
        fluid_energy,
        fluid_species,
        time_,
        dt,
        stage_id,
    )


def _local_preconditioner(coupling, mode, stage, state, residual, args):
    del state, args
    fluid_velocity = residual.fluid_velocity / coupling.fluid.cell_mass[:, None]
    fluid_temperature = residual.fluid_temperature / coupling.fluid.cell_heat_capacity
    fluid_species = residual.fluid_species_concentration / coupling.fluid.species_storage
    mass = jnp.maximum(stage.particle_mass, 1.0)[:, None]
    particle_velocity = residual.particle_velocity / mass
    energy = tuple(
        value / jnp.maximum(jnp.abs(previous.internal_energy), 1.0)
        for value, previous in zip(
            residual.batch_internal_energy,
            stage.previous_conversion.batches,
            strict=True,
        )
    )
    species = tuple(
        value / jnp.maximum(jnp.abs(previous.species_amount), 1.0)
        for value, previous in zip(
            residual.batch_species_amount,
            stage.previous_conversion.batches,
            strict=True,
        )
    )
    if mode is not ReactiveMonolithicPreconditionerMode.LOCAL_BLOCK:
        transfer = coupling.continuum_exchange.transfer
        relation = transfer.routes(stage.particle_position, stage.particle_active)
        feedback_result = transfer.deposit(
            stage.particle_position,
            stage.particle_active,
            coupling.drag_coefficient[:, None] * particle_velocity,
        )
        feedback = feedback_result.content
        fluid_velocity = (
            fluid_velocity
            + stage.step_size * feedback / coupling.fluid.cell_mass[:, None]
        )
    return ReactiveMonolithicUnknown(
        fluid_velocity,
        fluid_temperature,
        fluid_species,
        particle_velocity,
        energy,
        species,
    )


def prepare_reactive_monolithic_step(
    coupling: ReactiveMonolithicCouplingPlan,
    solver: ReactiveMonolithicSolverPlan,
    stage: ReactiveMonolithicStage,
    /,
    *,
    initial_guess: ReactiveMonolithicUnknown | None = None,
) -> PreparedReactiveMonolithicStep:
    if not isinstance(coupling, ReactiveMonolithicCouplingPlan):
        raise TypeError("coupling must be ReactiveMonolithicCouplingPlan.")
    if not isinstance(solver, ReactiveMonolithicSolverPlan):
        raise TypeError("solver must be ReactiveMonolithicSolverPlan.")
    if not isinstance(stage, ReactiveMonolithicStage):
        raise TypeError("stage must be ReactiveMonolithicStage.")
    guess = coupling.initial_unknown(stage) if initial_guess is None else initial_guess
    space = PyTreeSpace(guess)

    def residual(unknown, stage_):
        return coupling.evaluate(unknown, stage_).residual

    def valid(unknown, residual_, auxiliary, stage_):
        del residual_, auxiliary
        return coupling.evaluate(unknown, stage_).successful

    problem = NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        validity=valid,
        problem_id=f"reactive-monolithic:{coupling.plan_id}",
    )

    def apply(state_, residual_, stage_):
        return _local_preconditioner(
            coupling,
            solver.preconditioner_mode,
            stage_,
            state_,
            residual_,
            None,
        )

    preconditioner = FunctionLeftNonlinearPreconditioner(
        apply,
        state_space=space,
        source=space,
        target=space,
        preconditioner_id=f"reactive-monolithic:{solver.preconditioner_mode.value}",
    )
    transformation = LeftPreconditionedSystem(problem, preconditioner)
    nonlinear = prepare_nonlinear(
        transformation.problem,
        guess,
        method=solver.method,
        termination=solver.termination,
        args=stage,
        precision=solver.precision,
    )
    evidence = ReactiveMonolithicPreconditionerEvidence(
        solver.preconditioner_mode,
        coupling.fluid.cell_count,
        coupling.continuum_exchange.transfer.particle_capacity,
        len(stage.previous_conversion.batches),
        canonical_fingerprint(
            {
                "kind": "reactive-monolithic-preconditioner-evidence",
                "coupling": coupling.plan_id,
                "solver": solver.plan_id,
                "mode": solver.preconditioner_mode.value,
            }
        ),
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-reactive-monolithic-step",
            "coupling": coupling.plan_id,
            "solver": solver.plan_id,
            "stage": stage.stage_id,
        }
    )
    return PreparedReactiveMonolithicStep(
        coupling,
        solver,
        stage,
        guess,
        problem,
        transformation,
        nonlinear,
        evidence,
        prepared_id,
    )


def solve_reactive_monolithic_step(
    prepared: PreparedReactiveMonolithicStep,
    previous: ReactiveMonolithicState,
    /,
) -> ReactiveMonolithicStepResult:
    if not isinstance(prepared, PreparedReactiveMonolithicStep):
        raise TypeError("prepared must be PreparedReactiveMonolithicStep.")
    if not isinstance(previous, ReactiveMonolithicState):
        raise TypeError("previous must be ReactiveMonolithicState.")
    transformed_result = solve_prepared_nonlinear(prepared.nonlinear)
    nonlinear = prepared.transformation.finalize_result(
        transformed_result,
        prepared.initial_unknown,
        prepared.solver.termination,
        args=prepared.stage,
    )
    evaluation = prepared.coupling.evaluate(nonlinear.state, prepared.stage)
    fluid = ReactiveFluidImplicitState(
        nonlinear.state.fluid_velocity,
        nonlinear.state.fluid_temperature,
        nonlinear.state.fluid_species_concentration,
    )
    candidate = ReactiveMonolithicState(
        fluid,
        evaluation.conversion_state,
        nonlinear.state.particle_velocity,
        previous.accepted_windows + jnp.asarray(1, dtype=jnp.int32),
        previous.state_id,
    )
    event_split = (
        evaluation.route.minimum_species_margin <= prepared.solver.event_margin
    ) | (evaluation.route.minimum_temperature_margin <= prepared.solver.event_margin)
    tolerance = 256.0 * jnp.finfo(fluid.temperature.dtype).eps
    conservative = (
        jnp.all(
            jnp.abs(evaluation.momentum_residual)
            <= tolerance * jnp.maximum(jnp.linalg.norm(evaluation.particle_force), 1.0)
        )
        & (
            jnp.abs(evaluation.energy_residual)
            <= tolerance
            * jnp.maximum(jnp.sum(jnp.abs(evaluation.exchange.owner_heat_rate)), 1.0)
        )
        & jnp.all(
            jnp.abs(evaluation.species_residual)
            <= tolerance
            * jnp.maximum(
                jnp.sum(jnp.abs(evaluation.exchange.owner_species_rate), axis=0),
                1.0,
            )
        )
    )
    successful = (
        nonlinear.successful
        & evaluation.successful
        & evaluation.route.unchanged
        & conservative
        & ~event_split
        & tree_allfinite(candidate)
    )
    accepted = tree_where(successful, candidate, previous)
    return ReactiveMonolithicStepResult(
        candidate,
        accepted,
        nonlinear,
        evaluation,
        prepared.preconditioner,
        event_split,
        successful,
        prepared.prepared_id,
    )


def reactive_monolithic_vjp(
    loss,
    prepared: PreparedReactiveMonolithicStep,
    previous: ReactiveMonolithicState,
    /,
):
    if not callable(loss):
        raise TypeError("loss must be callable.")

    def objective(initial_particle_velocity):
        stage = eqx.tree_at(
            lambda value: value.previous_particle_velocity,
            prepared.stage,
            initial_particle_velocity,
        )
        refreshed = prepare_reactive_monolithic_step(
            prepared.coupling,
            prepared.solver,
            stage,
            initial_guess=prepared.initial_unknown,
        )
        result = solve_reactive_monolithic_step(refreshed, previous)
        return jnp.where(result.successful, loss(result.accepted_state), jnp.nan)

    primal, pullback = jax.vjp(objective, prepared.stage.previous_particle_velocity)
    return primal, pullback(jnp.ones_like(primal))[0]


__all__ = [
    "PreparedReactiveMonolithicStep",
    "ReactiveMonolithicPreconditionerEvidence",
    "ReactiveMonolithicPreconditionerMode",
    "ReactiveMonolithicSolverPlan",
    "ReactiveMonolithicState",
    "ReactiveMonolithicStepResult",
    "initialize_reactive_monolithic_state",
    "make_reactive_monolithic_stage",
    "prepare_reactive_monolithic_step",
    "reactive_monolithic_vjp",
    "solve_reactive_monolithic_step",
]
