#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ..._strict import StrictModule
from ...nonlinear import (
    NonlinearSystemProblem,
    root_solution_jvp,
    root_solution_vjp,
    SensitivityEvidence,
    SensitivityPolicy,
)
from ._solver import (
    FiniteElementContactDynamicsPlan,
    FiniteElementContactEquilibriumPlan,
    FiniteElementContactEquilibriumResult,
    FiniteElementContactResult,
)


class ContactSensitivityArguments(StrictModule):
    rest_positions: Array
    stiffness: Array
    user_args: Any

    def __init__(
        self,
        rest_positions: ArrayLike,
        stiffness: ArrayLike,
        /,
        *,
        user_args: Any = None,
    ):
        rest = jnp.asarray(rest_positions)
        stiffness_ = jnp.asarray(stiffness, dtype=rest.dtype)
        if rest.ndim != 2 or rest.shape[-1] not in (2, 3):
            raise ValueError("rest_positions must have shape (vertices, dimension).")
        if stiffness_.shape != ():
            raise ValueError("stiffness must be scalar.")
        self.rest_positions = rest
        self.stiffness = stiffness_
        self.user_args = user_args


class ContactDynamicsSensitivityArguments(StrictModule):
    rest_positions: Array
    stiffness: Array
    user_args: Any
    accepted_displacement: Array
    accepted_velocity: Array
    accepted_acceleration: Array
    step_size: Array

    def __init__(
        self,
        rest_positions: ArrayLike,
        stiffness: ArrayLike,
        accepted_displacement: ArrayLike,
        accepted_velocity: ArrayLike,
        accepted_acceleration: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        user_args: Any = None,
    ):
        rest = jnp.asarray(rest_positions)
        stiffness_ = jnp.asarray(stiffness, dtype=rest.dtype)
        if rest.ndim != 2 or rest.shape[-1] not in (2, 3):
            raise ValueError("rest_positions must have shape (vertices, dimension).")
        if stiffness_.shape != ():
            raise ValueError("stiffness must be scalar.")
        self.rest_positions = rest
        self.stiffness = stiffness_
        self.user_args = user_args
        displacement = jnp.asarray(accepted_displacement)
        velocity = jnp.asarray(accepted_velocity, dtype=displacement.dtype)
        acceleration = jnp.asarray(accepted_acceleration, dtype=displacement.dtype)
        dt = jnp.asarray(step_size, dtype=displacement.dtype)
        if (
            displacement.shape != velocity.shape
            or displacement.shape != acceleration.shape
        ):
            raise ValueError(
                "Accepted displacement, velocity, and acceleration must agree."
            )
        if dt.shape != ():
            raise ValueError("step_size must be scalar.")
        self.accepted_displacement = displacement
        self.accepted_velocity = velocity
        self.accepted_acceleration = acceleration
        self.step_size = dt


class ContactSensitivityResult(StrictModule):
    value: PyTree[Array]
    root_evidence: SensitivityEvidence
    minimum_gap: Array
    minimum_feature_margin: Array
    route_complete: Array
    branch_qualified: Array
    successful: Array


def _shifted_scene_positions(scene, state, rest_positions):
    prepared_rest = jnp.concatenate(
        tuple(surface.rest_positions for surface in scene.surfaces), axis=0
    )
    return scene.positions(state) + (rest_positions - prepared_rest)


def _equilibrium_problem(
    plan: FiniteElementContactEquilibriumPlan,
    epoch,
    /,
) -> NonlinearSystemProblem:
    def residual(state, arguments):
        def energy(value):
            positions = _shifted_scene_positions(
                plan.scene, value, arguments.rest_positions
            )
            return plan.problem.potential(
                value, arguments.user_args
            ) + plan.contact.energy(
                positions,
                epoch,
                rest_positions=arguments.rest_positions,
                stiffness=arguments.stiffness,
            )

        return jax.grad(energy)(state)

    return NonlinearSystemProblem(
        residual,
        state_space=plan.problem.state_space,
        residual_space=plan.problem.state_space,
        problem_id=f"{plan.plan_id}:fixed-route-stationarity",
    )


def _dynamics_problem(
    plan: FiniteElementContactDynamicsPlan,
    epoch,
    friction_state,
    /,
) -> NonlinearSystemProblem:
    def residual(state, arguments):
        context = plan.problem._execution_context(arguments.user_args)
        _, reduced_mass = plan.problem._mass_operators(
            context,
            plan.mass_coefficient,
            plan.mass_policy,
            plan.mass_problem,
        )
        dt = arguments.step_size
        predictor = (
            arguments.accepted_displacement
            + dt * arguments.accepted_velocity
            + dt * dt * (0.5 - plan.method.beta) * arguments.accepted_acceleration
        )
        acceleration_scale = plan.method.position_to_acceleration_scale(dt, state.dtype)

        def energy(value):
            delta = value - predictor
            mass_delta = plan.problem.state_space.inverse_riesz(reduced_mass.mv(delta))
            inertia = (
                0.5
                * acceleration_scale
                * jnp.real(plan.problem.state_space.inner(delta, mass_delta))
            )
            positions = _shifted_scene_positions(
                plan.scene, value, arguments.rest_positions
            )
            friction_energy = jnp.asarray(0.0, dtype=value.dtype)
            if plan.friction is not None and friction_state is not None:
                acceleration = acceleration_scale * (value - predictor)
                predicted_velocity = (
                    arguments.accepted_velocity
                    + dt * (1.0 - plan.method.gamma) * arguments.accepted_acceleration
                )
                velocity = predicted_velocity + plan.method.gamma * dt * acceleration
                velocity_scale = plan.method.position_to_velocity_scale(dt, value.dtype)
                friction_energy = (
                    plan.friction.energy(plan.scene.map_values(velocity), friction_state)
                    / velocity_scale
                )
            return (
                inertia
                + plan.problem.potential(value, arguments.user_args)
                + plan.contact.energy(
                    positions,
                    epoch,
                    rest_positions=arguments.rest_positions,
                    stiffness=arguments.stiffness,
                )
                + friction_energy
            )

        return jax.grad(energy)(state)

    return NonlinearSystemProblem(
        residual,
        state_space=plan.problem.state_space,
        residual_space=plan.problem.state_space,
        problem_id=f"{plan.plan_id}:fixed-route-stationarity",
    )


def _wrap_derivative(derivative, contact, epoch, margin_tolerance):
    tolerance = jnp.asarray(margin_tolerance, dtype=contact.minimum_gap.dtype)
    branch = (
        epoch.successful
        & (contact.minimum_gap > tolerance)
        & (contact.minimum_feature_margin > tolerance)
    )
    successful = derivative.evidence.successful & branch
    value = jax.tree.map(
        lambda leaf: jnp.where(successful, leaf, jnp.full_like(leaf, jnp.nan)),
        derivative.value,
    )
    return ContactSensitivityResult(
        value,
        derivative.evidence,
        contact.minimum_gap,
        contact.minimum_feature_margin,
        epoch.successful,
        branch,
        successful,
    )


def contact_equilibrium_solution_jvp(
    plan: FiniteElementContactEquilibriumPlan,
    result: FiniteElementContactEquilibriumResult,
    arguments: ContactSensitivityArguments,
    tangent_arguments: ContactSensitivityArguments,
    /,
    *,
    policy: SensitivityPolicy | None = None,
    margin_tolerance: float = 1.0e-10,
) -> ContactSensitivityResult:
    if not isinstance(plan, FiniteElementContactEquilibriumPlan) or not isinstance(
        result, FiniteElementContactEquilibriumResult
    ):
        raise TypeError("plan/result are not a contact equilibrium pair.")
    epoch = result.contact.epoch_id
    del epoch
    replay = result.contact
    candidate_epoch = result.safety.epoch_id
    del candidate_epoch
    if not bool(result.accepted):
        raise ValueError("Contact equilibrium sensitivity requires an accepted result.")
    # The final contact evaluation retains the epoch identity; the solver result's
    # candidate routes are supplied by the accepted search replay through its plan.
    final_positions = plan.scene.positions(result.candidate)
    fixed_epoch = plan.search.build(plan.scene, final_positions)
    problem = _equilibrium_problem(plan, fixed_epoch)
    derivative = root_solution_jvp(
        problem,
        result.candidate,
        arguments,
        tangent_arguments,
        policy=policy,
    )
    return _wrap_derivative(derivative, replay, fixed_epoch, margin_tolerance)


def contact_equilibrium_solution_vjp(
    plan: FiniteElementContactEquilibriumPlan,
    result: FiniteElementContactEquilibriumResult,
    arguments: ContactSensitivityArguments,
    cotangent_state: ArrayLike,
    /,
    *,
    policy: SensitivityPolicy | None = None,
    margin_tolerance: float = 1.0e-10,
) -> ContactSensitivityResult:
    if not bool(result.accepted):
        raise ValueError("Contact equilibrium sensitivity requires an accepted result.")
    final_positions = plan.scene.positions(result.candidate)
    fixed_epoch = plan.search.build(plan.scene, final_positions)
    problem = _equilibrium_problem(plan, fixed_epoch)
    derivative = root_solution_vjp(
        problem,
        result.candidate,
        arguments,
        plan.problem.state_space.validate(cotangent_state),
        policy=policy,
    )
    return _wrap_derivative(derivative, result.contact, fixed_epoch, margin_tolerance)


def contact_dynamics_solution_jvp(
    plan: FiniteElementContactDynamicsPlan,
    result: FiniteElementContactResult,
    arguments: ContactDynamicsSensitivityArguments,
    tangent_arguments: ContactDynamicsSensitivityArguments,
    /,
    *,
    policy: SensitivityPolicy | None = None,
    margin_tolerance: float = 1.0e-10,
) -> ContactSensitivityResult:
    if not bool(result.accepted):
        raise ValueError("Contact dynamics sensitivity requires an accepted result.")
    epoch = result.candidate.replay_epoch
    if epoch is None:
        raise ValueError("Accepted contact result does not retain a replay epoch.")
    problem = _dynamics_problem(plan, epoch, result.candidate.friction_state)
    state = result.candidate.mechanics.displacement
    derivative = root_solution_jvp(
        problem,
        state,
        arguments,
        tangent_arguments,
        policy=policy,
    )
    return _wrap_derivative(derivative, result.contact, epoch, margin_tolerance)


def contact_dynamics_solution_vjp(
    plan: FiniteElementContactDynamicsPlan,
    result: FiniteElementContactResult,
    arguments: ContactDynamicsSensitivityArguments,
    cotangent_state: ArrayLike,
    /,
    *,
    policy: SensitivityPolicy | None = None,
    margin_tolerance: float = 1.0e-10,
) -> ContactSensitivityResult:
    if not bool(result.accepted):
        raise ValueError("Contact dynamics sensitivity requires an accepted result.")
    epoch = result.candidate.replay_epoch
    if epoch is None:
        raise ValueError("Accepted contact result does not retain a replay epoch.")
    problem = _dynamics_problem(plan, epoch, result.candidate.friction_state)
    state = result.candidate.mechanics.displacement
    derivative = root_solution_vjp(
        problem,
        state,
        arguments,
        plan.problem.state_space.validate(cotangent_state),
        policy=policy,
    )
    return _wrap_derivative(derivative, result.contact, epoch, margin_tolerance)


__all__ = [
    "ContactDynamicsSensitivityArguments",
    "ContactSensitivityArguments",
    "ContactSensitivityResult",
    "contact_dynamics_solution_jvp",
    "contact_dynamics_solution_vjp",
    "contact_equilibrium_solution_jvp",
    "contact_equilibrium_solution_vjp",
]
