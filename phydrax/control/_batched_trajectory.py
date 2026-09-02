#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import prepare_local_block_factorization, solve_local_blocks
from ._constraints import evaluate_sampled_feasibility
from ._cost import evaluate_sampled_cost
from ._ilqr import (
    _flow_map,
    _local_model,
    _trajectory_cost,
    _validate_solver_options,
    DifferentialControlFlow,
    ILQRDiagnostics,
    ILQRPolicy,
    ILQRResult,
    ILQRStatus,
)
from ._problem import ControlProblem
from ._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_SUCCESS,
    ControlResult,
    ControlTrajectory,
)


class ILQRPlan(StrictModule, NonTrainableState):
    maximum_iterations: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    gradient_tolerance: float = eqx.field(static=True)
    cost_tolerance: float = eqx.field(static=True)
    line_search_steps: int = eqx.field(static=True)
    line_search_decay: float = eqx.field(static=True)
    initial_step_size: float = eqx.field(static=True)
    armijo: float = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedILQR(StrictModule, NonTrainableState):
    plan: ILQRPlan
    problem: ControlProblem
    initial_controls: Array
    flow: Any = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def plan_ilqr(
    problem: ControlProblem,
    /,
    *,
    max_iterations: int = 100,
    regularization: float = 1.0e-6,
    gradient_tolerance: float = 1.0e-6,
    cost_tolerance: float = 1.0e-9,
    line_search_steps: int = 10,
    line_search_decay: float = 0.5,
    initial_step_size: float = 1.0,
    armijo: float = 1.0e-4,
) -> ILQRPlan:
    """Plan one homogeneous fixed-capacity case-axis iLQR kernel."""

    if not isinstance(problem, ControlProblem):
        raise TypeError("problem must be a ControlProblem.")
    if problem.path_constraints or problem.terminal_constraints:
        raise ValueError("iLQR supports unconstrained ControlProblem values only.")
    values = _validate_solver_options(
        max_iterations=max_iterations,
        regularization=regularization,
        gradient_tolerance=gradient_tolerance,
        cost_tolerance=cost_tolerance,
        line_search_steps=line_search_steps,
        line_search_decay=line_search_decay,
        initial_step_size=initial_step_size,
        armijo=armijo,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "prepared-ilqr-plan",
            "problem": problem.problem_id,
            "case_shape": list(problem.case_shape),
            "maximum_iterations": values[0],
            "line_search_steps": values[4],
            "regularization": values[1],
        }
    )
    return ILQRPlan(
        *values[:4],
        values[4],
        *values[5:],
        problem.case_shape,
        problem.problem_id,
        plan_id,
    )


def prepare_ilqr(
    plan: ILQRPlan,
    problem: ControlProblem,
    initial_controls: ArrayLike,
    /,
    *,
    differential_flow: DifferentialControlFlow | None = None,
) -> PreparedILQR:
    """Bind case-shaped controls and one explicit flow to an ILQRPlan."""

    if not isinstance(plan, ILQRPlan) or not isinstance(problem, ControlProblem):
        raise TypeError("plan/problem must be ILQRPlan/ControlProblem.")
    if plan.problem_id != problem.problem_id or plan.case_shape != problem.case_shape:
        raise ValueError("ILQR plan and problem topology identities do not match.")
    controls = jnp.asarray(initial_controls)
    expected = problem.case_shape + (problem.time_grid.num_steps,) + problem.control_shape
    if controls.shape != expected:
        raise ValueError(
            f"initial_controls must have shape {expected}; got {controls.shape}."
        )
    if not jnp.issubdtype(controls.dtype, jnp.inexact):
        controls = controls.astype(float)
    flow, discretization_id, backend_id = _flow_map(problem, differential_flow)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-ilqr",
            "plan": plan.plan_id,
            "discretization": discretization_id,
            "backend": backend_id,
            "dtype": np.dtype(controls.dtype).str,
        }
    )
    return PreparedILQR(
        plan,
        problem,
        controls,
        flow,
        discretization_id,
        backend_id,
        prepared_id,
    )


def _rollout(problem, flow, initial_state, controls):
    times = problem.time_grid.times

    def step(carry, inputs):
        state, active, failed_step = carry
        index, control = inputs
        index = jnp.asarray(index, dtype=jnp.int32)
        candidate = jnp.asarray(
            flow(times[index], times[index + 1], index, state, control)
        )
        valid = active & jnp.all(jnp.isfinite(control)) & jnp.all(jnp.isfinite(candidate))
        next_state = jnp.where(valid, candidate, state)
        failed_step = jnp.where(
            active & (~valid) & (failed_step < 0),
            index,
            failed_step,
        ).astype(jnp.int32)
        return (next_state, valid, failed_step), (next_state, valid)

    initial_valid = jnp.all(jnp.isfinite(initial_state))
    (_, active, failed_step), (tail, valid_tail) = jax.lax.scan(
        step,
        (initial_state, initial_valid, jnp.asarray(-1, dtype=jnp.int32)),
        (
            jnp.arange(problem.time_grid.num_steps, dtype=jnp.int32),
            controls,
        ),
    )
    states = jnp.concatenate((initial_state[None], tail), axis=0)
    valid = jnp.concatenate((initial_valid[None], valid_tail), axis=0)
    objective, cost_valid = _trajectory_cost(problem, states, controls)
    objective = jnp.where(active & cost_valid, objective, jnp.inf)
    return states, controls, valid, objective, failed_step


def _feedback_rollout(
    problem,
    flow,
    initial_state,
    nominal_states,
    nominal_controls,
    feedforward,
    feedback,
    step_size,
):
    times = problem.time_grid.times
    state_size = prod(problem.state_shape)
    control_size = prod(problem.control_shape)

    def step(carry, inputs):
        state, active, failed_step = carry
        index, nominal_state, nominal_control, feedforward_, feedback_ = inputs
        index = jnp.asarray(index, dtype=jnp.int32)
        delta = state.reshape((state_size,)) - nominal_state.reshape((state_size,))
        correction = oe.contract("ij,j->i", feedback_, delta)
        control = (
            nominal_control.reshape((control_size,))
            + step_size * feedforward_
            + correction
        ).reshape(problem.control_shape)
        candidate = jnp.asarray(
            flow(times[index], times[index + 1], index, state, control)
        )
        valid = active & jnp.all(jnp.isfinite(control)) & jnp.all(jnp.isfinite(candidate))
        next_state = jnp.where(valid, candidate, state)
        failed_step = jnp.where(
            active & (~valid) & (failed_step < 0),
            index,
            failed_step,
        ).astype(jnp.int32)
        return (next_state, valid, failed_step), (next_state, control, valid)

    initial_valid = jnp.all(jnp.isfinite(initial_state))
    (_, active, failed_step), (tail, controls, valid_tail) = jax.lax.scan(
        step,
        (initial_state, initial_valid, jnp.asarray(-1, dtype=jnp.int32)),
        (
            jnp.arange(problem.time_grid.num_steps, dtype=jnp.int32),
            nominal_states[:-1],
            nominal_controls,
            feedforward,
            feedback,
        ),
    )
    states = jnp.concatenate((initial_state[None], tail), axis=0)
    valid = jnp.concatenate((initial_valid[None], valid_tail), axis=0)
    objective, cost_valid = _trajectory_cost(problem, states, controls)
    objective = jnp.where(active & cost_valid, objective, jnp.inf)
    return states, controls, valid, objective, failed_step


def _backward(model, regularization):
    state_size = model.dynamics_state.shape[-1]
    control_size = model.dynamics_control.shape[-1]
    identity = jnp.eye(control_size, dtype=model.terminal_hessian.dtype)

    def step(carry, inputs):
        (
            value_gradient,
            value_hessian,
            active,
            linear,
            quadratic,
            curvature,
            failed_step,
        ) = carry
        index, a, b, lx, lu, lxx, luu, lux = inputs
        index = jnp.asarray(index, dtype=jnp.int32)
        qx = lx + oe.contract("ji,j->i", a, value_gradient)
        qu = lu + oe.contract("ji,j->i", b, value_gradient)
        qxx = lxx + oe.contract("ji,jk,kl->il", a, value_hessian, a)
        quu = luu + oe.contract("ji,jk,kl->il", b, value_hessian, b)
        qux = lux + oe.contract("ji,jk,kl->il", b, value_hessian, a)
        qxx = 0.5 * (qxx + jnp.swapaxes(qxx, -1, -2))
        regularized = 0.5 * (quu + jnp.swapaxes(quu, -1, -2)) + regularization * identity
        factorization = prepare_local_block_factorization(
            regularized[None], positive_definite=True
        )
        rhs = jnp.concatenate((qu[:, None], qux), axis=1)[None]
        solved, failed = solve_local_blocks(factorization, rhs)
        failed_ = jnp.ravel(failed)[0]
        usable = active & (~failed_)
        k = -solved[0, :, 0]
        gain = -solved[0, :, 1:]
        k = jnp.where(usable, k, 0)
        gain = jnp.where(usable, gain, 0)
        next_gradient = (
            qx
            + oe.contract("ji,j->i", gain, qu)
            + oe.contract("ji,j->i", qux, k)
            + oe.contract("ji,jk,k->i", gain, regularized, k)
        )
        next_hessian = (
            qxx
            + oe.contract("ji,jk->ik", gain, qux)
            + oe.contract("ji,jk->ik", qux, gain)
            + oe.contract("ji,jk,kl->il", gain, regularized, gain)
        )
        next_hessian = 0.5 * (next_hessian + jnp.swapaxes(next_hessian, -1, -2))
        diagonal = jnp.real(jnp.diagonal(factorization.factors[0]))
        minimum = jnp.min(diagonal * diagonal)
        next_carry = (
            jnp.where(usable, next_gradient, value_gradient),
            jnp.where(usable, next_hessian, value_hessian),
            usable,
            linear + jnp.where(usable, oe.contract("i,i->", qu, k), 0),
            quadratic
            + jnp.where(
                usable,
                0.5 * oe.contract("i,ij,j->", k, regularized, k),
                0,
            ),
            jnp.minimum(curvature, minimum),
            jnp.where(
                active & failed_ & (failed_step < 0),
                index,
                failed_step,
            ).astype(jnp.int32),
        )
        return next_carry, (k, gain)

    indices = jnp.arange(
        model.dynamics_state.shape[0] - 1,
        -1,
        -1,
        dtype=jnp.int32,
    )
    inputs = tuple(
        jnp.flip(value, axis=0)
        for value in (
            model.dynamics_state,
            model.dynamics_control,
            model.running_state_gradient,
            model.running_control_gradient,
            model.running_state_hessian,
            model.running_control_hessian,
            model.running_control_state_hessian,
        )
    )
    initial = (
        model.terminal_gradient,
        model.terminal_hessian,
        jnp.asarray(True),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(jnp.inf),
        jnp.asarray(-1, dtype=jnp.int32),
    )
    final, (feedforward, feedback) = jax.lax.scan(step, initial, (indices,) + inputs)
    return (
        jnp.flip(feedforward, axis=0),
        jnp.flip(feedback, axis=0),
        final[3],
        final[4],
        final[5],
        final[2],
        final[6],
    )


def _solve_case(prepared: PreparedILQR, initial_state: Array, initial_controls: Array):
    plan = prepared.plan
    problem = prepared.problem
    states, controls, valid, objective, failed_step = _rollout(
        problem, prepared.flow, initial_state, initial_controls
    )
    iterations = plan.maximum_iterations
    history_shape = (iterations,)
    zeros = jnp.zeros(history_shape, dtype=objective.dtype)
    nan_history = jnp.full(history_shape, jnp.nan, dtype=objective.dtype)
    initial_active = jnp.all(valid) & jnp.isfinite(objective)
    carry = (
        states,
        controls,
        valid,
        objective,
        jnp.zeros(
            (
                problem.time_grid.num_steps,
                prod(problem.control_shape),
                prod(problem.state_shape),
            ),
            dtype=controls.dtype,
        ),
        initial_active,
        jnp.where(
            initial_active,
            int(ILQRStatus.MAX_ITERATIONS),
            int(ILQRStatus.INITIAL_ROLLOUT_FAILED),
        ).astype(jnp.int32),
        failed_step,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        nan_history.at[0].set(objective),
        nan_history,
        nan_history,
        zeros,
        nan_history,
        nan_history,
        jnp.zeros(history_shape, dtype=jnp.int32),
    )

    def iteration(index, loop):
        (
            states_,
            controls_,
            valid_,
            objective_,
            feedback_,
            active,
            status,
            failed_step_,
            iteration_count,
            accepted_count,
            objective_history,
            gradient_history,
            curvature_history,
            step_history,
            expected_history,
            actual_history,
            evaluations_history,
        ) = loop

        def advance(_):
            model = _local_model(problem, states_, controls_, prepared.flow)
            gradient_norm = jnp.sqrt(jnp.sum(jnp.square(jnp.abs(model.control_gradient))))
            (
                feedforward,
                feedback,
                linear,
                quadratic,
                curvature,
                backward_valid,
                backward_failed,
            ) = _backward(model, plan.regularization)
            gradient_converged = gradient_norm <= plan.gradient_tolerance

            search_initial = (
                jnp.asarray(False),
                states_,
                controls_,
                valid_,
                objective_,
                jnp.asarray(0.0),
                jnp.asarray(jnp.nan),
                jnp.asarray(jnp.nan),
                jnp.asarray(0, dtype=jnp.int32),
                failed_step_,
            )

            def search(search_index, search_carry):
                (
                    found,
                    best_states,
                    best_controls,
                    best_valid,
                    best_objective,
                    best_step,
                    best_expected,
                    best_actual,
                    evaluations,
                    search_failed,
                ) = search_carry
                step_size = plan.initial_step_size * plan.line_search_decay**search_index
                candidate = _feedback_rollout(
                    problem,
                    prepared.flow,
                    initial_state,
                    states_,
                    controls_,
                    feedforward,
                    feedback,
                    step_size,
                )
                (
                    candidate_states,
                    candidate_controls,
                    candidate_valid,
                    candidate_objective,
                    candidate_failed,
                ) = candidate
                expected = -(step_size * linear + step_size**2 * quadratic)
                actual = objective_ - candidate_objective
                acceptable = (
                    (~found)
                    & backward_valid
                    & (~gradient_converged)
                    & jnp.all(candidate_valid)
                    & jnp.isfinite(candidate_objective)
                    & jnp.isfinite(expected)
                    & (expected > 0)
                    & (actual > 0)
                    & (actual >= plan.armijo * expected)
                )
                return (
                    found | acceptable,
                    jnp.where(acceptable, candidate_states, best_states),
                    jnp.where(acceptable, candidate_controls, best_controls),
                    jnp.where(acceptable, candidate_valid, best_valid),
                    jnp.where(acceptable, candidate_objective, best_objective),
                    jnp.where(acceptable, step_size, best_step),
                    jnp.where(acceptable, expected, best_expected),
                    jnp.where(acceptable, actual, best_actual),
                    jnp.where(acceptable, search_index + 1, evaluations),
                    jnp.where(
                        (candidate_failed >= 0) & (search_failed < 0),
                        candidate_failed,
                        search_failed,
                    ),
                )

            search = jax.lax.fori_loop(0, plan.line_search_steps, search, search_initial)
            (
                found,
                next_states,
                next_controls,
                next_valid,
                next_objective,
                step_size,
                expected,
                actual,
                evaluations,
                search_failed,
            ) = search
            cost_converged = found & (
                actual <= plan.cost_tolerance * jnp.maximum(1.0, jnp.abs(objective_))
            )
            converged = gradient_converged | cost_converged
            next_active = (
                backward_valid & (~gradient_converged) & found & (~cost_converged)
            )
            next_status = jnp.where(
                ~backward_valid,
                int(ILQRStatus.BACKWARD_PASS_NOT_POSITIVE_DEFINITE),
                jnp.where(
                    gradient_converged | cost_converged,
                    int(ILQRStatus.SUCCESS),
                    jnp.where(
                        found,
                        int(ILQRStatus.MAX_ITERATIONS),
                        int(ILQRStatus.LINE_SEARCH_FAILED),
                    ),
                ),
            ).astype(jnp.int32)
            next_failed = jnp.where(
                ~backward_valid,
                backward_failed,
                jnp.where(found, failed_step_, search_failed),
            )
            return (
                next_states,
                next_controls,
                next_valid,
                next_objective,
                jnp.where(backward_valid, feedback, feedback_),
                next_active,
                next_status,
                next_failed,
                iteration_count + 1,
                accepted_count + found.astype(jnp.int32),
                objective_history.at[index].set(next_objective),
                gradient_history.at[index].set(gradient_norm),
                curvature_history.at[index].set(curvature),
                step_history.at[index].set(step_size),
                expected_history.at[index].set(expected),
                actual_history.at[index].set(actual),
                evaluations_history.at[index].set(evaluations),
            )

        return jax.lax.cond(active, advance, lambda _: loop, None)

    final = jax.lax.fori_loop(0, iterations, iteration, carry)
    return final


def solve_prepared_ilqr(
    prepared: PreparedILQR,
    /,
    *,
    policy_id: str | None = None,
    result_id: str | None = None,
) -> ILQRResult:
    """Solve every active homogeneous case through one vmap/fixed-capacity kernel."""

    if not isinstance(prepared, PreparedILQR):
        raise TypeError("prepared must be a PreparedILQR.")
    problem = prepared.problem
    case_count = prod(problem.case_shape) if problem.case_shape else 1
    initial_states = problem.initial_state.reshape((case_count,) + problem.state_shape)
    controls = prepared.initial_controls.reshape(
        (case_count, problem.time_grid.num_steps) + problem.control_shape
    )
    outputs = jax.vmap(lambda state, control: _solve_case(prepared, state, control))(
        initial_states, controls
    )
    reshaped = tuple(
        value.reshape(problem.case_shape + value.shape[1:]) for value in outputs
    )
    (
        states,
        controls,
        valid,
        objective,
        feedback,
        active,
        status,
        failed_step,
        iteration_count,
        accepted_count,
        objective_history,
        gradient_history,
        curvature_history,
        step_history,
        expected_history,
        actual_history,
        evaluations_history,
    ) = reshaped
    del objective, active
    policy_name = f"ilqr-policy:{problem.problem_id}" if policy_id is None else policy_id
    policy = ILQRPolicy(
        problem.time_grid,
        states,
        controls,
        feedback,
        state_shape=problem.state_shape,
        control_shape=problem.control_shape,
        policy_id=policy_name,
    )
    trajectory_status = jnp.where(
        jnp.all(valid, axis=-1), CONTROL_SUCCESS, CONTROL_DYNAMICS_FAILED
    ).astype(jnp.int32)
    trajectory = ControlTrajectory(
        time_grid=problem.time_grid,
        states=states,
        controls=controls,
        valid=valid,
        status=trajectory_status,
        backend_status=trajectory_status,
        case_shape=problem.case_shape,
        state_shape=problem.state_shape,
        control_shape=problem.control_shape,
        problem_id=problem.problem_id,
        dynamics_id=problem.dynamics.dynamics_id,
        control_id=policy.parameterization_id,
        backend_id=prepared.backend_id,
        method_id="iterative-lqr:prepared-case-vmap",
        discretization_id=prepared.discretization_id,
        approximation_id=policy.approximation_id,
    )
    sampled_loss = evaluate_sampled_cost(problem, trajectory)
    feasibility = evaluate_sampled_feasibility(problem, trajectory)
    control_result = ControlResult(
        trajectory=trajectory,
        parameters=controls,
        sampled_loss=sampled_loss,
        feasibility=feasibility,
        result_id=f"ilqr-result:{problem.problem_id}" if result_id is None else result_id,
        method_id=trajectory.method_id,
    )
    diagnostics = ILQRDiagnostics(
        objective_history,
        gradient_history,
        curvature_history,
        step_history,
        expected_history,
        actual_history,
        evaluations_history,
        jnp.full(problem.case_shape, prepared.plan.regularization),
        status,
        iteration_count,
        accepted_count,
        failed_step,
        status == int(ILQRStatus.SUCCESS),
        "iterative-lqr:prepared-fixed-capacity",
    )
    return ILQRResult(control_result, policy, diagnostics)


__all__ = [
    "ILQRPlan",
    "PreparedILQR",
    "plan_ilqr",
    "prepare_ilqr",
    "solve_prepared_ilqr",
]
