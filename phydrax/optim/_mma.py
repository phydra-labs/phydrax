#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._iterative._base import AbstractMinimizationMethod
from ._iterative._types import (
    ConstrainedOptimalityCertificate,
    MinimizationProblem,
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationCertificate,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._nonlinear_constraints import _canonical_constraints, _constraint_layout


class MMAPolicy(StrictModule):
    """Moving-asymptote and separable dual-subproblem controls."""

    asymptote_initial: float = eqx.field(static=True)
    asymptote_shrink: float = eqx.field(static=True)
    asymptote_grow: float = eqx.field(static=True)
    move_limit: float = eqx.field(static=True)
    move_decay: float = eqx.field(static=True)
    minimum_move: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    dual_iterations: int = eqx.field(static=True)
    dual_bisections: int = eqx.field(static=True)
    dual_bracket_steps: int = eqx.field(static=True)
    feasibility_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        asymptote_initial: float = 0.5,
        asymptote_shrink: float = 0.7,
        asymptote_grow: float = 1.2,
        move_limit: float = 0.2,
        move_decay: float = 1.0,
        minimum_move: float = 1.0e-3,
        regularization: float = 1.0e-5,
        dual_iterations: int = 400,
        dual_bisections: int = 64,
        dual_bracket_steps: int = 48,
        feasibility_tolerance: float = 1.0e-9,
    ):
        values = tuple(
            float(value)
            for value in (
                asymptote_initial,
                asymptote_shrink,
                asymptote_grow,
                move_limit,
                move_decay,
                minimum_move,
                regularization,
                feasibility_tolerance,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("MMA controls must be finite and positive.")
        if values[1] >= 1.0 or values[2] <= 1.0:
            raise ValueError(
                "MMA asymptote factors must shrink below and grow above one."
            )
        if values[3] > 1.0 or values[4] > 1.0 or values[5] > values[3]:
            raise ValueError(
                "MMA move controls require minimum <= initial <= one and decay <= one."
            )
        iteration_values = (dual_iterations, dual_bisections, dual_bracket_steps)
        if any(isinstance(value, bool) or int(value) <= 0 for value in iteration_values):
            raise ValueError("MMA dual iteration counts must be positive integers.")
        (
            self.asymptote_initial,
            self.asymptote_shrink,
            self.asymptote_grow,
            self.move_limit,
            self.move_decay,
            self.minimum_move,
            self.regularization,
            self.feasibility_tolerance,
        ) = values
        self.dual_iterations = int(dual_iterations)
        self.dual_bisections = int(dual_bisections)
        self.dual_bracket_steps = int(dual_bracket_steps)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mma-policy",
                "asymptote_initial": self.asymptote_initial,
                "asymptote_shrink": self.asymptote_shrink,
                "asymptote_grow": self.asymptote_grow,
                "move_limit": self.move_limit,
                "move_decay": self.move_decay,
                "minimum_move": self.minimum_move,
                "regularization": self.regularization,
                "dual_iterations": self.dual_iterations,
                "dual_bisections": self.dual_bisections,
                "dual_bracket_steps": self.dual_bracket_steps,
                "feasibility_tolerance": self.feasibility_tolerance,
            }
        )


class MMAEvidence(StrictModule):
    """Final separable-subproblem and asymptote evidence."""

    lower_asymptotes: Array
    upper_asymptotes: Array
    inequality_multipliers: Array
    subproblem_residual: Array
    final_move_limit: Array
    subproblem_iterations: Array


class _MMAState(eqx.Module):
    parameters: Array
    previous: Array
    previous_previous: Array
    lower_asymptotes: Array
    upper_asymptotes: Array
    multipliers: Array
    subproblem_residual: Array
    status: Array
    iterations: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    constraint_evaluations: Array
    accepted_steps: Array
    final_step_norm: Array
    initial_optimality: Array
    final_move_limit: Array


class MethodOfMovingAsymptotes(AbstractMinimizationMethod):
    """Bounded inequality-constrained MMA with independent final KKT evidence."""

    policy: MMAPolicy

    def __init__(self, policy: MMAPolicy | None = None, /):
        policy_ = MMAPolicy() if policy is None else policy
        if not isinstance(policy_, MMAPolicy):
            raise TypeError("policy must be MMAPolicy or None.")
        self.policy = policy_

    @property
    def method_id(self) -> str:
        return f"mma/{self.policy.policy_id}"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=True,
            prepared_refresh=False,
            implicit_differentiation=False,
        )

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        return _solve_mma(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


def _split_gradient(
    gradient: Array,
    parameters: Array,
    lower_asymptotes: Array,
    upper_asymptotes: Array,
    span: Array,
    regularization: float,
) -> tuple[Array, Array]:
    positive = jnp.maximum(gradient, 0.0)
    negative = jnp.maximum(-gradient, 0.0)
    regularizer = jnp.asarray(regularization, dtype=parameters.dtype) / span
    p = (upper_asymptotes - parameters) ** 2 * (
        1.001 * positive + 0.001 * negative + regularizer
    )
    q = (parameters - lower_asymptotes) ** 2 * (
        0.001 * positive + 1.001 * negative + regularizer
    )
    return p, q


def _update_asymptotes(
    state: _MMAState,
    lower: Array,
    upper: Array,
    policy: MMAPolicy,
) -> tuple[Array, Array]:
    span = upper - lower
    initial_lower = state.parameters - policy.asymptote_initial * span
    initial_upper = state.parameters + policy.asymptote_initial * span
    sign = (state.parameters - state.previous) * (
        state.previous - state.previous_previous
    )
    factor = jnp.where(
        sign < 0.0,
        policy.asymptote_shrink,
        jnp.where(sign > 0.0, policy.asymptote_grow, 1.0),
    )
    history_lower = state.parameters - factor * (state.previous - state.lower_asymptotes)
    history_upper = state.parameters + factor * (state.upper_asymptotes - state.previous)
    history_lower = jnp.clip(
        history_lower,
        state.parameters - 10.0 * span,
        state.parameters - 0.01 * span,
    )
    history_upper = jnp.clip(
        history_upper,
        state.parameters + 0.01 * span,
        state.parameters + 10.0 * span,
    )
    early = state.iterations < 2
    return (
        jnp.where(early, initial_lower, history_lower),
        jnp.where(early, initial_upper, history_upper),
    )


def _mma_subproblem(
    state: _MMAState,
    objective_gradient: Array,
    inequalities: Array,
    inequality_jacobian: Array,
    lower: Array,
    upper: Array,
    policy: MMAPolicy,
) -> tuple[Array, Array, Array, Array, Array]:
    span = upper - lower
    asymptote_lower, asymptote_upper = _update_asymptotes(state, lower, upper, policy)
    move = jnp.maximum(policy.minimum_move, state.final_move_limit)
    alpha = jnp.maximum(
        lower,
        jnp.maximum(
            asymptote_lower + 0.1 * (state.parameters - asymptote_lower),
            state.parameters - move * span,
        ),
    )
    beta = jnp.minimum(
        upper,
        jnp.minimum(
            asymptote_upper - 0.1 * (asymptote_upper - state.parameters),
            state.parameters + move * span,
        ),
    )
    alpha = jnp.minimum(alpha, beta)
    p0, q0 = _split_gradient(
        objective_gradient,
        state.parameters,
        asymptote_lower,
        asymptote_upper,
        span,
        policy.regularization,
    )
    p_rows, q_rows = jax.vmap(
        lambda row: _split_gradient(
            row,
            state.parameters,
            asymptote_lower,
            asymptote_upper,
            span,
            policy.regularization,
        )
    )(inequality_jacobian)
    offset = (
        jnp.sum(
            p_rows / (asymptote_upper - state.parameters)
            + q_rows / (state.parameters - asymptote_lower),
            axis=1,
        )
        - inequalities
    )

    def primal(multipliers):
        p = p0 + contract("m,mn->n", multipliers, p_rows)
        q = q0 + contract("m,mn->n", multipliers, q_rows)
        root_p = jnp.sqrt(jnp.maximum(p, jnp.finfo(p.dtype).tiny))
        root_q = jnp.sqrt(jnp.maximum(q, jnp.finfo(q.dtype).tiny))
        candidate = (root_p * asymptote_lower + root_q * asymptote_upper) / (
            root_p + root_q
        )
        return jnp.clip(candidate, alpha, beta)

    def approximate_constraints(candidate):
        return (
            jnp.sum(
                p_rows / (asymptote_upper - candidate)
                + q_rows / (candidate - asymptote_lower),
                axis=1,
            )
            - offset
        )

    count = int(inequalities.shape[0])
    multipliers = jnp.zeros((count,), dtype=state.parameters.dtype)

    def update_coordinate(index, current):
        zeroed = current.at[index].set(0.0)

        def bracket_body(_, bracket):
            lo, hi = bracket
            trial = zeroed.at[index].set(hi)
            violated = approximate_constraints(primal(trial))[index] > 0.0
            return (
                jnp.where(violated, hi, lo),
                jnp.where(violated, 2.0 * hi, hi),
            )

        lo, hi = jax.lax.fori_loop(
            0,
            policy.dual_bracket_steps,
            bracket_body,
            (
                jnp.asarray(0.0, state.parameters.dtype),
                jnp.asarray(1.0, state.parameters.dtype),
            ),
        )

        def bisect_body(_, bracket):
            lower_multiplier, upper_multiplier = bracket
            midpoint = 0.5 * (lower_multiplier + upper_multiplier)
            trial = zeroed.at[index].set(midpoint)
            violated = approximate_constraints(primal(trial))[index] > 0.0
            return (
                jnp.where(violated, midpoint, lower_multiplier),
                jnp.where(violated, upper_multiplier, midpoint),
            )

        lo, hi = jax.lax.fori_loop(
            0,
            policy.dual_bisections,
            bisect_body,
            (lo, hi),
        )
        derivative_at_zero = approximate_constraints(primal(zeroed))[index]
        value = jnp.where(derivative_at_zero <= 0.0, 0.0, 0.5 * (lo + hi))
        return current.at[index].set(value)

    sweeps = max(1, policy.dual_iterations // max(count, 1))

    def sweep_body(_, current):
        return jax.lax.fori_loop(0, count, update_coordinate, current)

    multipliers = jax.lax.fori_loop(0, sweeps, sweep_body, multipliers)
    candidate = primal(multipliers)
    residual = jnp.max(jnp.maximum(approximate_constraints(candidate), 0.0), initial=0.0)
    return (
        candidate,
        asymptote_lower,
        asymptote_upper,
        multipliers,
        jnp.maximum(residual, 0.0),
    )


def _solve_mma(
    method: MethodOfMovingAsymptotes,
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> MinimizationResult:
    if not isinstance(problem, MinimizationProblem):
        raise TypeError("problem must be a MinimizationProblem.")
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")
    if problem.bounds is None:
        raise ValueError("MMA requires explicit finite parameter bounds.")

    flat_initial, unravel = ravel_pytree(initial_parameters)
    lower_tree, upper_tree = problem.bounds.materialize(initial_parameters)
    lower, _ = ravel_pytree(lower_tree)
    upper, _ = ravel_pytree(upper_tree)
    if not isinstance(lower, jax.core.Tracer):
        lower_host = np.asarray(lower)
        upper_host = np.asarray(upper)
        if np.any(
            ~np.isfinite(lower_host)
            | ~np.isfinite(upper_host)
            | (upper_host <= lower_host)
        ):
            raise ValueError("MMA requires finite parameter bounds with positive width.")
    flat_initial = eqx.error_if(
        flat_initial,
        jnp.any(~jnp.isfinite(lower) | ~jnp.isfinite(upper) | (upper <= lower)),
        "MMA requires finite parameter bounds with positive width.",
    )
    constraints_only = MinimizationProblem(
        problem.objective,
        has_aux=problem.has_aux,
        constraints=problem.constraints,
        problem_id=problem.problem_id,
    )
    layout = _constraint_layout(constraints_only, initial_parameters, args)
    if int(layout.equality_indices.size):
        raise ValueError("MMA currently supports inequalities, not equalities.")

    def objective_flat(coordinates):
        value, _ = problem.value(unravel(coordinates), args)
        return value

    def inequalities_flat(coordinates):
        _, inequalities = _canonical_constraints(
            constraints_only,
            layout,
            unravel(coordinates),
            args,
        )
        return inequalities

    def evaluate(coordinates):
        value, gradient = jax.value_and_grad(objective_flat)(coordinates)
        inequalities = inequalities_flat(coordinates)
        jacobian = jax.jacrev(inequalities_flat)(coordinates)
        return value, gradient, inequalities, jacobian

    initial_value, initial_gradient, initial_inequalities, _ = evaluate(flat_initial)
    initial_feasibility = jnp.max(jnp.maximum(initial_inequalities, 0.0), initial=0.0)
    initial_projected = problem.bounds.projected_gradient(
        initial_parameters,
        unravel(initial_gradient),
    )
    initial_projected_flat, _ = ravel_pytree(initial_projected)
    initial_optimality = jnp.maximum(
        jnp.max(jnp.abs(initial_projected_flat), initial=0.0),
        initial_feasibility,
    )
    feasible = (
        problem.bounds.contains(initial_parameters)
        & (initial_feasibility <= method.policy.feasibility_tolerance)
        & jnp.isfinite(initial_value)
        & jnp.all(jnp.isfinite(initial_gradient))
    )
    initial_status = jnp.where(
        feasible,
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.INFEASIBLE),
    ).astype(jnp.int32)
    span = upper - lower
    state = _MMAState(
        flat_initial,
        flat_initial,
        flat_initial,
        flat_initial - method.policy.asymptote_initial * span,
        flat_initial + method.policy.asymptote_initial * span,
        jnp.zeros_like(initial_inequalities),
        jnp.asarray(0.0, flat_initial.dtype),
        initial_status,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(1, jnp.int32),
        jnp.asarray(1, jnp.int32),
        jnp.asarray(2, jnp.int32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0.0, flat_initial.dtype),
        initial_optimality,
        jnp.asarray(method.policy.move_limit, flat_initial.dtype),
    )

    def condition(current):
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else current.objective_evaluations < termination.maximum_evaluations
        )
        return (
            (current.status == int(OptimizationStatus.ITERATING))
            & (current.iterations < termination.maximum_steps)
            & within_evaluations
        )

    def body(current):
        _, gradient, inequalities, jacobian = evaluate(current.parameters)
        candidate, asymptote_lower, asymptote_upper, multipliers, subproblem_residual = (
            _mma_subproblem(
                current,
                gradient,
                inequalities,
                jacobian,
                lower,
                upper,
                method.policy,
            )
        )
        candidate_value, candidate_gradient, candidate_inequalities, _ = evaluate(
            candidate
        )
        finite = (
            jnp.isfinite(candidate_value)
            & jnp.all(jnp.isfinite(candidate_gradient))
            & jnp.all(jnp.isfinite(candidate_inequalities))
            & jnp.all(jnp.isfinite(candidate))
        )
        step_norm = jnp.max(jnp.abs(candidate - current.parameters), initial=0.0)
        stagnated = step_norm <= termination.step_threshold(
            jnp.max(jnp.abs(current.parameters), initial=0.0)
        )
        status = jnp.where(
            ~finite,
            int(OptimizationStatus.NONFINITE_EVALUATION),
            jnp.where(
                stagnated,
                int(OptimizationStatus.STAGNATION),
                int(OptimizationStatus.ITERATING),
            ),
        ).astype(jnp.int32)
        next_iteration = current.iterations + 1
        move = jnp.maximum(
            method.policy.minimum_move,
            current.final_move_limit * method.policy.move_decay,
        )
        return _MMAState(
            jnp.where(finite, candidate, current.parameters),
            current.parameters,
            current.previous,
            asymptote_lower,
            asymptote_upper,
            multipliers,
            subproblem_residual,
            status,
            next_iteration,
            current.objective_evaluations + 2,
            current.gradient_evaluations + 2,
            current.constraint_evaluations + 4,
            current.accepted_steps + finite.astype(jnp.int32),
            step_norm,
            current.initial_optimality,
            move,
        )

    state = jax.lax.while_loop(condition, body, state)
    (final_value, auxiliary), final_gradient_tree = problem.value_and_gradient(
        unravel(state.parameters), args
    )
    final_gradient, _ = ravel_pytree(final_gradient_tree)
    final_inequalities = inequalities_flat(state.parameters)
    final_jacobian = jax.jacrev(inequalities_flat)(state.parameters)
    lagrangian_gradient = final_gradient + contract(
        "m,mn->n", state.multipliers, final_jacobian
    )
    projected_tree = problem.bounds.projected_gradient(
        unravel(state.parameters), unravel(lagrangian_gradient)
    )
    projected, _ = ravel_pytree(projected_tree)
    projected_norm = jnp.max(jnp.abs(projected), initial=0.0)
    primal = jnp.maximum(
        jnp.max(jnp.maximum(final_inequalities, 0.0), initial=0.0),
        problem.bounds.violation(unravel(state.parameters)),
    )
    dual = jnp.max(jnp.maximum(-state.multipliers, 0.0), initial=0.0)
    slacks = jnp.maximum(-final_inequalities, 0.0)
    complementarity = jnp.max(jnp.abs(state.multipliers * slacks), initial=0.0)
    optimality = jnp.maximum(
        jnp.maximum(projected_norm, primal), jnp.maximum(dual, complementarity)
    )
    threshold = termination.optimality_threshold(state.initial_optimality)
    finite = (
        jnp.isfinite(final_value)
        & jnp.all(jnp.isfinite(final_gradient))
        & jnp.all(jnp.isfinite(final_inequalities))
        & jnp.isfinite(optimality)
    )
    certified = finite & (optimality <= threshold)
    status = jnp.where(
        certified,
        int(OptimizationStatus.SUCCESS),
        jnp.where(
            state.status == int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
            state.status,
        ),
    ).astype(jnp.int32)
    active = final_inequalities >= -method.policy.feasibility_tolerance
    constrained_certificate = ConstrainedOptimalityCertificate(
        equality_multipliers=jnp.empty((0,), state.parameters.dtype),
        inequality_multipliers=state.multipliers,
        slacks=slacks,
        active_mask=active,
        stationarity_residual=unravel(projected),
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        inequality_sources=layout.inequality_sources,
    )
    certificate_id = canonical_fingerprint(
        {
            "kind": "mma-active-kkt",
            "problem": problem.problem_id,
            "policy": method.policy.policy_id,
        }
    )
    optimality_certificate = OptimizationCertificate(
        kind="active-kkt",
        tolerance=threshold,
        optimality_norm=optimality,
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        projected_stationarity=projected_norm,
        finite=finite,
        regular=jnp.isfinite(state.subproblem_residual),
        certified=certified,
        evaluation_work=state.objective_evaluations + 1,
        certificate_id=certificate_id,
    )
    diagnostics = OptimizationDiagnostics(
        iterations=state.iterations,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.iterations - state.accepted_steps,
        objective_evaluations=state.objective_evaluations + 1,
        gradient_evaluations=state.gradient_evaluations + 1,
        constraint_evaluations=state.constraint_evaluations + 2,
        initial_optimality_norm=state.initial_optimality,
        final_optimality_norm=optimality,
        final_step_norm=state.final_step_norm,
        accepted_step_size=state.final_move_limit,
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        active_constraints=jnp.sum(active, dtype=jnp.int32),
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        globalization="moving-asymptotes",
        matrix_free=True,
        implicit_differentiation=False,
        notes="Finite-box, feasible-start, inequality-constrained MMA.",
    )
    evidence = MMAEvidence(
        state.lower_asymptotes,
        state.upper_asymptotes,
        state.multipliers,
        state.subproblem_residual,
        state.final_move_limit,
        state.iterations * method.policy.dual_iterations,
    )
    return MinimizationResult(
        unravel(state.parameters),
        final_value,
        auxiliary,
        status,
        diagnostics,
        provenance,
        certificate=constrained_certificate,
        optimality_certificate=optimality_certificate,
        method_evidence=evidence,
    )


__all__ = [
    "MMAEvidence",
    "MMAPolicy",
    "MethodOfMovingAsymptotes",
]
