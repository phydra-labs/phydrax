#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree
from opt_einsum import contract

from ..linalg import (
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    PyTreeSpace,
    solve as solve_linear,
    transpose,
)
from ._iterative._types import (
    ConstrainedOptimalityCertificate,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._mma import _mma_subproblem, _MMAState, MMAEvidence, MMAPolicy
from ._pde_constrained import (
    _default_adjoint_policy,
    AbstractStateDesignMethod,
    StateDesignConstraint,
    StateDesignProblem,
    StateDesignResult,
    StateEquationResult,
)


@dataclass(frozen=True, slots=True)
class _StateInequality:
    constraint: StateDesignConstraint
    sign: float
    bound: float
    source: str

    def value(self, state, design, args):
        raw = jnp.asarray(self.constraint.value(state, design, args))
        return self.sign * (raw - self.bound)


class _ReducedMMAState(eqx.Module):
    state_result: StateEquationResult
    mma: _MMAState
    adjoint: PyTree[Array]
    status: Array
    state_solves: Array
    adjoint_solves: Array
    linear_iterations: Array
    rejected_steps: Array


class ReducedMMA(AbstractStateDesignMethod):
    """Reduced state/design MMA using exact matrix-free adjoint gradients."""

    policy: MMAPolicy
    linear_policy: LinearSolvePolicy

    def __init__(
        self,
        *,
        policy: MMAPolicy | None = None,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        policy_ = MMAPolicy() if policy is None else policy
        linear_policy_ = (
            _default_adjoint_policy() if linear_policy is None else linear_policy
        )
        if not isinstance(policy_, MMAPolicy):
            raise TypeError("policy must be MMAPolicy or None.")
        if not isinstance(linear_policy_, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        self.policy = policy_
        self.linear_policy = linear_policy_

    @property
    def method_id(self) -> str:
        return f"reduced-mma/{self.policy.policy_id}"

    def solve(
        self,
        problem: StateDesignProblem,
        initial_state: PyTree[Any],
        initial_design: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> StateDesignResult:
        return _solve_reduced_mma(
            self,
            problem,
            initial_state,
            initial_design,
            termination=termination,
            args=args,
        )


def _state_inequalities(
    problem: StateDesignProblem,
    state: PyTree[Any],
    design: PyTree[Any],
    args: Any,
    /,
) -> tuple[_StateInequality, ...]:
    inequalities: list[_StateInequality] = []
    for constraint in problem.constraints:
        abstract = jax.eval_shape(
            lambda current_state, current_design, constraint=constraint: constraint.value(
                current_state, current_design, args
            ),
            state,
            design,
        )
        leaves = jax.tree.leaves(abstract)
        if len(leaves) != 1 or leaves[0].shape != ():
            raise ValueError(
                "ReducedMMA currently requires scalar state-design constraints."
            )
        lower = np.asarray(constraint.lower)
        upper = np.asarray(constraint.upper)
        if lower.shape != () or upper.shape != ():
            raise ValueError("ReducedMMA constraint bounds must be scalar.")
        lower_value = float(lower)
        upper_value = float(upper)
        if (
            np.isfinite(lower_value)
            and np.isfinite(upper_value)
            and lower_value == upper_value
        ):
            raise ValueError("ReducedMMA supports inequalities, not equalities.")
        if np.isfinite(lower_value):
            inequalities.append(
                _StateInequality(
                    constraint,
                    -1.0,
                    lower_value,
                    f"{constraint.constraint_id}:lower",
                )
            )
        if np.isfinite(upper_value):
            inequalities.append(
                _StateInequality(
                    constraint,
                    1.0,
                    upper_value,
                    f"{constraint.constraint_id}:upper",
                )
            )
    if not inequalities:
        raise ValueError("ReducedMMA requires at least one finite inequality.")
    return tuple(inequalities)


def _reduced_values_and_gradients(
    problem: StateDesignProblem,
    inequalities: tuple[_StateInequality, ...],
    state: PyTree[Any],
    design: PyTree[Any],
    args: Any,
    linear_policy: LinearSolvePolicy,
    state_acceptance,
    /,
):
    def residual_function(current_state):
        return problem.residual(current_state, design, args)

    residual, state_linearization = jax.linearize(residual_function, state)
    _, state_pullback = jax.vjp(residual_function, state)

    def state_action(tangent):
        value = state_linearization(tangent)
        return jax.tree.map(
            lambda leaf, reference: jnp.asarray(
                leaf, dtype=jnp.asarray(reference).dtype
            ).reshape(jnp.asarray(reference).shape),
            value,
            residual,
        )

    def state_transpose_action(cotangent):
        value = state_pullback(cotangent)[0]
        return jax.tree.map(
            lambda leaf, reference: jnp.asarray(
                leaf, dtype=jnp.asarray(reference).dtype
            ).reshape(jnp.asarray(reference).shape),
            value,
            state,
        )

    state_jacobian = FunctionLinearOperator(
        state_action,
        source=PyTreeSpace(state),
        target=PyTreeSpace(residual),
        transpose_action=state_transpose_action,
        operator_id=f"{problem.problem_id}/state-jacobian",
        closure_convert=False,
    )
    transpose_system = LinearSystem(transpose(state_jacobian))
    _, design_residual_pullback = jax.vjp(
        lambda current_design: problem.residual(state, current_design, args),
        design,
    )

    def reduced_gradient(function, depends_on_state):
        direct = jax.grad(lambda current_design: function(state, current_design))(design)
        if not depends_on_state:
            zero = jax.tree.map(jnp.zeros_like, residual)
            return direct, zero, None, None
        state_gradient = jax.grad(lambda current_state: function(current_state, design))(
            state
        )
        adjoint_result = solve_linear(
            transpose_system,
            state_gradient,
            policy=linear_policy,
        )
        adjoint = adjoint_result.value
        adjoint_acceptance = problem.acceptance_policy.adjoint_evidence(
            adjoint,
            state_transpose_action(adjoint),
            state_gradient,
            adjoint_result.status,
            admissible=state_acceptance.admissible & state_acceptance.finite,
            realization_matches=state_acceptance.realization_matches,
        )
        residual_part = design_residual_pullback(adjoint)[0]
        gradient = jax.tree.map(
            lambda direct_part, implicit_part: direct_part - implicit_part,
            direct,
            residual_part,
        )
        return gradient, adjoint, adjoint_result, adjoint_acceptance

    objective_value, _ = problem.value(state, design, args)
    (
        objective_gradient,
        objective_adjoint,
        objective_result,
        objective_acceptance,
    ) = reduced_gradient(
        lambda current_state, current_design: problem.value(
            current_state, current_design, args
        )[0],
        True,
    )
    flat_objective_gradient, _ = ravel_pytree(objective_gradient)
    values = []
    rows = []
    adjoint_results = []
    adjoint_acceptances = []
    for inequality in inequalities:
        value = inequality.value(state, design, args)
        gradient, _, adjoint_result, adjoint_acceptance = reduced_gradient(
            lambda current_state, current_design, inequality=inequality: inequality.value(
                current_state, current_design, args
            ),
            inequality.constraint.depends_on_state,
        )
        flat_gradient, _ = ravel_pytree(gradient)
        values.append(value)
        rows.append(flat_gradient)
        if adjoint_result is not None:
            adjoint_results.append(adjoint_result)
            adjoint_acceptances.append(adjoint_acceptance)
    all_results = (objective_result,) + tuple(adjoint_results)
    all_acceptances = (objective_acceptance,) + tuple(adjoint_acceptances)
    usable = jnp.all(jnp.stack(tuple(evidence.accepted for evidence in all_acceptances)))
    linear_iterations = sum(
        (result.diagnostics.iterations for result in all_results),
        start=jnp.asarray(0, dtype=jnp.int32),
    )
    return (
        objective_value,
        flat_objective_gradient,
        jnp.stack(values),
        jnp.stack(rows),
        objective_adjoint,
        objective_acceptance,
        usable,
        jnp.asarray(len(all_results), dtype=jnp.int32),
        linear_iterations,
    )


def _solve_reduced_mma(
    method: ReducedMMA,
    problem: StateDesignProblem,
    initial_state: PyTree[Any],
    initial_design: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> StateDesignResult:
    if not isinstance(problem, StateDesignProblem):
        raise TypeError("problem must be a StateDesignProblem.")
    if problem.design_bounds is None:
        raise ValueError("ReducedMMA requires finite design bounds.")
    state = problem.solve_state(initial_design, initial_state, args=args)
    inequalities = _state_inequalities(problem, state.state, initial_design, args)
    flat_design, unravel = ravel_pytree(initial_design)
    lower_tree, upper_tree = problem.design_bounds.materialize(initial_design)
    lower, _ = ravel_pytree(lower_tree)
    upper, _ = ravel_pytree(upper_tree)
    if not isinstance(lower, jax.core.Tracer):
        lower_host, upper_host = np.asarray(lower), np.asarray(upper)
        if np.any(
            ~np.isfinite(lower_host)
            | ~np.isfinite(upper_host)
            | (upper_host <= lower_host)
        ):
            raise ValueError(
                "ReducedMMA requires finite design bounds with positive width."
            )

    (
        initial_value,
        initial_gradient,
        initial_constraints,
        _,
        initial_adjoint,
        _,
        initial_adjoint_usable,
        initial_adjoint_solves,
        initial_linear_iterations,
    ) = _reduced_values_and_gradients(
        problem,
        inequalities,
        state.state,
        initial_design,
        args,
        method.linear_policy,
        state.acceptance,
    )
    initial_projected = problem.design_bounds.projected_gradient(
        initial_design, unravel(initial_gradient)
    )
    initial_projected_flat, _ = ravel_pytree(initial_projected)
    feasibility = jnp.max(jnp.maximum(initial_constraints, 0.0), initial=0.0)
    initial_optimality = jnp.maximum(
        jnp.max(jnp.abs(initial_projected_flat), initial=0.0), feasibility
    )
    valid = (
        state.acceptance.accepted
        & initial_adjoint_usable
        & problem.design_bounds.contains(initial_design)
        & (feasibility <= method.policy.feasibility_tolerance)
        & jnp.isfinite(initial_value)
    )
    status = jnp.where(
        valid,
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.INFEASIBLE),
    ).astype(jnp.int32)
    span = upper - lower
    mma = _MMAState(
        flat_design,
        flat_design,
        flat_design,
        flat_design - method.policy.asymptote_initial * span,
        flat_design + method.policy.asymptote_initial * span,
        jnp.zeros_like(initial_constraints),
        jnp.asarray(0.0, flat_design.dtype),
        status,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(1, jnp.int32),
        jnp.asarray(1, jnp.int32),
        jnp.asarray(1, jnp.int32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0.0, flat_design.dtype),
        initial_optimality,
        jnp.asarray(method.policy.move_limit, flat_design.dtype),
    )
    carry = _ReducedMMAState(
        state,
        mma,
        initial_adjoint,
        status,
        jnp.asarray(1, jnp.int32),
        initial_adjoint_solves,
        initial_linear_iterations,
        jnp.asarray(0, jnp.int32),
    )

    def condition(current):
        return (current.status == int(OptimizationStatus.ITERATING)) & (
            current.mma.iterations < termination.maximum_steps
        )

    def body(current):
        design = unravel(current.mma.parameters)
        (
            _,
            gradient,
            constraint_values,
            constraint_jacobian,
            objective_adjoint,
            _,
            adjoints_usable,
            adjoint_solves,
            linear_iterations,
        ) = _reduced_values_and_gradients(
            problem,
            inequalities,
            current.state_result.state,
            design,
            args,
            method.linear_policy,
            current.state_result.acceptance,
        )
        candidate, asymptote_lower, asymptote_upper, multipliers, subproblem_residual = (
            _mma_subproblem(
                current.mma,
                gradient,
                constraint_values,
                constraint_jacobian,
                lower,
                upper,
                method.policy,
            )
        )
        candidate_design = unravel(candidate)
        candidate_state = problem.solve_state(
            candidate_design,
            current.state_result.state,
            args=args,
        )
        finite = (
            candidate_state.acceptance.accepted
            & adjoints_usable
            & jnp.all(jnp.isfinite(candidate))
        )
        step_norm = jnp.max(jnp.abs(candidate - current.mma.parameters), initial=0.0)
        accepted_move = jnp.maximum(
            method.policy.minimum_move,
            current.mma.final_move_limit * method.policy.move_decay,
        )
        rejected_move = jnp.maximum(
            method.policy.minimum_move,
            current.mma.final_move_limit * method.policy.asymptote_shrink,
        )
        exhausted = (~finite) & (
            current.mma.final_move_limit <= 1.01 * method.policy.minimum_move
        )
        stagnated = finite & (
            step_norm
            <= termination.step_threshold(
                jnp.max(jnp.abs(current.mma.parameters), initial=0.0)
            )
        )
        next_status = jnp.where(
            exhausted,
            int(OptimizationStatus.BACKEND_FAILED),
            jnp.where(
                stagnated,
                int(OptimizationStatus.STAGNATION),
                int(OptimizationStatus.ITERATING),
            ),
        ).astype(jnp.int32)
        next_parameters = jnp.where(finite, candidate, current.mma.parameters)
        next_mma = _MMAState(
            next_parameters,
            jnp.where(finite, current.mma.parameters, current.mma.previous),
            jnp.where(finite, current.mma.previous, current.mma.previous_previous),
            asymptote_lower,
            asymptote_upper,
            multipliers,
            subproblem_residual,
            next_status,
            current.mma.iterations + 1,
            current.mma.objective_evaluations + 1,
            current.mma.gradient_evaluations + 1,
            current.mma.constraint_evaluations + 1,
            current.mma.accepted_steps + finite.astype(jnp.int32),
            step_norm,
            current.mma.initial_optimality,
            jnp.where(finite, accepted_move, rejected_move),
        )
        next_state = jax.tree.map(
            lambda candidate_value, current_value: jnp.where(
                finite, candidate_value, current_value
            ),
            candidate_state,
            current.state_result,
        )
        return _ReducedMMAState(
            next_state,
            next_mma,
            jax.tree.map(
                lambda new, old: jnp.where(finite, new, old),
                objective_adjoint,
                current.adjoint,
            ),
            next_status,
            current.state_solves + 1,
            current.adjoint_solves + adjoint_solves,
            current.linear_iterations
            + linear_iterations
            + candidate_state.diagnostics.linear_iterations,
            current.rejected_steps + (~finite).astype(jnp.int32),
        )

    carry = jax.lax.while_loop(condition, body, carry)
    design = unravel(carry.mma.parameters)
    (
        final_value,
        final_gradient,
        final_constraints,
        final_jacobian,
        final_adjoint,
        final_adjoint_acceptance,
        final_adjoint_usable,
        final_adjoint_solves,
        final_linear_iterations,
    ) = _reduced_values_and_gradients(
        problem,
        inequalities,
        carry.state_result.state,
        design,
        args,
        method.linear_policy,
        carry.state_result.acceptance,
    )
    lagrangian_gradient = final_gradient + contract(
        "m,mn->n", carry.mma.multipliers, final_jacobian
    )
    projected_tree = problem.design_bounds.projected_gradient(
        design, unravel(lagrangian_gradient)
    )
    projected, _ = ravel_pytree(projected_tree)
    projected_norm = jnp.max(jnp.abs(projected), initial=0.0)
    primal = jnp.maximum(
        jnp.max(jnp.maximum(final_constraints, 0.0), initial=0.0),
        problem.design_bounds.violation(design),
    )
    dual = jnp.max(jnp.maximum(-carry.mma.multipliers, 0.0), initial=0.0)
    slacks = jnp.maximum(-final_constraints, 0.0)
    complementarity = jnp.max(jnp.abs(carry.mma.multipliers * slacks), initial=0.0)
    optimality = jnp.maximum(
        jnp.maximum(projected_norm, primal), jnp.maximum(dual, complementarity)
    )
    threshold = termination.optimality_threshold(carry.mma.initial_optimality)
    certified = (
        carry.state_result.acceptance.accepted
        & final_adjoint_usable
        & final_adjoint_acceptance.accepted
        & jnp.isfinite(final_value)
        & jnp.isfinite(optimality)
        & (optimality <= threshold)
    )
    status = jnp.where(
        certified,
        int(OptimizationStatus.SUCCESS),
        jnp.where(
            carry.status == int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
            carry.status,
        ),
    ).astype(jnp.int32)
    active = final_constraints >= -method.policy.feasibility_tolerance
    certificate = ConstrainedOptimalityCertificate(
        equality_multipliers=jnp.empty((0,), carry.mma.parameters.dtype),
        inequality_multipliers=carry.mma.multipliers,
        slacks=slacks,
        active_mask=active,
        stationarity_residual=unravel(projected),
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        inequality_sources=tuple(item.source for item in inequalities),
    )
    diagnostics = OptimizationDiagnostics(
        iterations=carry.mma.iterations,
        accepted_steps=carry.mma.accepted_steps,
        rejected_steps=carry.rejected_steps,
        objective_evaluations=carry.mma.objective_evaluations + 1,
        gradient_evaluations=carry.mma.gradient_evaluations + 1,
        constraint_evaluations=carry.mma.constraint_evaluations + 1,
        linear_solves=carry.adjoint_solves + final_adjoint_solves,
        linear_iterations=carry.linear_iterations + final_linear_iterations,
        initial_optimality_norm=carry.mma.initial_optimality,
        final_optimality_norm=optimality,
        final_step_norm=carry.mma.final_step_norm,
        accepted_step_size=carry.mma.final_move_limit,
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        active_constraints=jnp.sum(active, dtype=jnp.int32),
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        globalization="reduced-moving-asymptotes",
        matrix_free=True,
        implicit_differentiation=False,
        notes="State-converged candidates with exact reduced adjoint gradients.",
    )
    evidence = MMAEvidence(
        carry.mma.lower_asymptotes,
        carry.mma.upper_asymptotes,
        carry.mma.multipliers,
        carry.mma.subproblem_residual,
        carry.mma.final_move_limit,
        carry.mma.iterations * method.policy.dual_iterations,
    )
    _, auxiliary = problem.value(carry.state_result.state, design, args)
    return StateDesignResult(
        carry.state_result.state,
        design,
        final_value,
        auxiliary,
        final_adjoint,
        status,
        diagnostics,
        provenance,
        state_acceptance=carry.state_result.acceptance,
        adjoint_acceptance=final_adjoint_acceptance,
        certificate=certificate,
        method_evidence=evidence,
    )


__all__ = ["ReducedMMA"]
