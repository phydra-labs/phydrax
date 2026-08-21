#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from ..linalg import (
    bind_numeric,
    DifferentiationPolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    MINRES,
    OperatorProperties,
    prepare_template,
    PyTreeSpace,
    solve as solve_linear,
    TolerancePolicy,
    transpose,
)
from ._bounds import _projected_displacement
from ._iterative._globalization import ArmijoLineSearch
from ._iterative._types import (
    _tree_allfinite,
    _tree_inner,
    _tree_negative,
    _tree_norm,
    _validate_real_inexact_tree,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._pde_constrained import (
    _default_adjoint_policy,
    _state_design_line_search,
    _usable_linear_status,
    AbstractStateDesignMethod,
    StateDesignProblem,
    StateDesignResult,
)


def _default_reduced_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        MINRES(),
        tolerance=TolerancePolicy(relative=1e-7, absolute=1e-10, max_steps=100),
        differentiation=DifferentiationPolicy("algorithmic"),
    )


class ReducedNewtonKrylov(AbstractStateDesignMethod):
    """Reduced-space Newton--Krylov with incremental state and adjoint solves."""

    state_linear_policy: LinearSolvePolicy
    reduced_linear_policy: LinearSolvePolicy
    line_search: ArmijoLineSearch
    hessian_regularization: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_linear_policy: LinearSolvePolicy | None = None,
        reduced_linear_policy: LinearSolvePolicy | None = None,
        line_search: ArmijoLineSearch | None = None,
        hessian_regularization: float = 1e-8,
    ):
        state_policy = (
            _default_adjoint_policy()
            if state_linear_policy is None
            else state_linear_policy
        )
        reduced_policy = (
            _default_reduced_policy()
            if reduced_linear_policy is None
            else reduced_linear_policy
        )
        search = ArmijoLineSearch() if line_search is None else line_search
        regularization = float(hessian_regularization)
        if not isinstance(state_policy, LinearSolvePolicy):
            raise TypeError("state_linear_policy must be a LinearSolvePolicy or None.")
        if not isinstance(reduced_policy, LinearSolvePolicy):
            raise TypeError("reduced_linear_policy must be a LinearSolvePolicy or None.")
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        if not isfinite(regularization) or regularization < 0.0:
            raise ValueError("hessian_regularization must be finite and non-negative.")
        self.state_linear_policy = state_policy
        self.reduced_linear_policy = reduced_policy
        self.line_search = search
        self.hessian_regularization = regularization

    @property
    def method_id(self) -> str:
        return "reduced-newton-krylov"

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
        return _solve_reduced_newton_krylov(
            self,
            problem,
            initial_state,
            initial_design,
            termination=termination,
            args=args,
        )


def _state_jacobian(
    problem: StateDesignProblem,
    state: PyTree[Any],
    design: PyTree[Any],
    args: Any,
    /,
):
    def residual_function(current_state):
        return problem.residual(current_state, design, args)

    residual, state_action = jax.linearize(residual_function, state)
    _, state_pullback = jax.vjp(residual_function, state)
    operator = FunctionLinearOperator(
        state_action,
        source=PyTreeSpace(state),
        target=PyTreeSpace(residual),
        transpose_action=lambda cotangent: state_pullback(cotangent)[0],
        operator_id="reduced-newton-state-jacobian",
        closure_convert=False,
    )
    return residual, operator


def _prepare_linear_templates(
    method: ReducedNewtonKrylov,
    problem: StateDesignProblem,
    state: PyTree[Any],
    design: PyTree[Any],
    residual: PyTree[Any],
    /,
):
    state_jacobian = FunctionLinearOperator(
        lambda _: jax.tree.map(jnp.zeros_like, residual),
        source=PyTreeSpace(state),
        target=PyTreeSpace(residual),
        transpose_action=lambda _: jax.tree.map(jnp.zeros_like, state),
        operator_id="reduced-newton-state-jacobian",
        closure_convert=False,
    )
    state_system = LinearSystem(
        state_jacobian,
        problem_id=f"{problem.problem_id}/incremental-state",
    )
    adjoint_system = LinearSystem(
        transpose(state_jacobian),
        problem_id=f"{problem.problem_id}/incremental-adjoint",
    )
    state_template = prepare_template(
        state_system,
        method.state_linear_policy,
    )
    adjoint_template = prepare_template(
        adjoint_system,
        method.state_linear_policy,
    )

    design_space = PyTreeSpace(design)
    structural_reduced_operator = FunctionLinearOperator(
        lambda tangent: tangent,
        source=design_space,
        target=design_space,
        transpose_action=lambda tangent: tangent,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id="reduced-newton-hessian",
        closure_convert=False,
    )
    reduced_template = prepare_template(
        LinearSystem(
            structural_reduced_operator,
            problem_id=f"{problem.problem_id}/reduced-hessian",
        ),
        method.reduced_linear_policy,
    )
    return state_template, adjoint_template, reduced_template


def _reduced_model(
    method: ReducedNewtonKrylov,
    problem: StateDesignProblem,
    state: PyTree[Any],
    design: PyTree[Any],
    args: Any,
    state_template,
    adjoint_template,
    numeric_version: Any,
    /,
):
    residual, state_jacobian = _state_jacobian(problem, state, design, args)
    state_system = LinearSystem(
        state_jacobian,
        problem_id=f"{problem.problem_id}/incremental-state",
    )
    adjoint_system = LinearSystem(
        transpose(state_jacobian),
        problem_id=f"{problem.problem_id}/incremental-adjoint",
    )
    prepared_state = bind_numeric(
        state_template,
        state_system,
        numeric_version=numeric_version,
    )
    prepared_adjoint = bind_numeric(
        adjoint_template,
        adjoint_system,
        numeric_version=numeric_version,
    )
    state_objective_gradient = jax.grad(
        lambda current_state: problem.value(current_state, design, args)[0]
    )(state)
    adjoint_result = solve_linear(prepared_adjoint, state_objective_gradient)
    adjoint = adjoint_result.value
    design_objective_gradient = jax.grad(
        lambda current_design: problem.value(state, current_design, args)[0]
    )(design)
    _, design_pullback = jax.vjp(
        lambda current_design: problem.residual(state, current_design, args),
        design,
    )
    residual_design_adjoint = design_pullback(adjoint)[0]
    reduced_gradient = jax.tree.map(
        lambda objective_part, residual_part: objective_part - residual_part,
        design_objective_gradient,
        residual_design_adjoint,
    )

    def reduced_hessian_action(design_tangent):
        residual_design_tangent = jax.jvp(
            lambda current_design: problem.residual(state, current_design, args),
            (design,),
            (design_tangent,),
        )[1]
        incremental_state = solve_linear(
            prepared_state,
            jax.tree.map(lambda value: -value, residual_design_tangent),
        ).value

        def state_stationarity(current_state, current_design):
            objective_gradient = jax.grad(
                lambda state_value: problem.value(
                    state_value,
                    current_design,
                    args,
                )[0]
            )(current_state)
            _, pullback = jax.vjp(
                lambda state_value: problem.residual(
                    state_value,
                    current_design,
                    args,
                ),
                current_state,
            )
            residual_part = pullback(adjoint)[0]
            return jax.tree.map(
                lambda objective_part, constraint_part: objective_part - constraint_part,
                objective_gradient,
                residual_part,
            )

        incremental_adjoint_rhs = jax.jvp(
            state_stationarity,
            (state, design),
            (incremental_state, design_tangent),
        )[1]
        incremental_adjoint = solve_linear(
            prepared_adjoint,
            incremental_adjoint_rhs,
        ).value

        def design_stationarity(current_state, current_design):
            objective_gradient = jax.grad(
                lambda design_value: problem.value(
                    current_state,
                    design_value,
                    args,
                )[0]
            )(current_design)
            _, pullback = jax.vjp(
                lambda design_value: problem.residual(
                    current_state,
                    design_value,
                    args,
                ),
                current_design,
            )
            residual_part = pullback(adjoint)[0]
            return jax.tree.map(
                lambda objective_part, constraint_part: objective_part - constraint_part,
                objective_gradient,
                residual_part,
            )

        fixed_adjoint_tangent = jax.jvp(
            design_stationarity,
            (state, design),
            (incremental_state, design_tangent),
        )[1]
        incremental_adjoint_part = design_pullback(incremental_adjoint)[0]
        return jax.tree.map(
            lambda fixed_part, adjoint_part, tangent: (
                fixed_part - adjoint_part + method.hessian_regularization * tangent
            ),
            fixed_adjoint_tangent,
            incremental_adjoint_part,
            design_tangent,
        )

    return (
        residual,
        reduced_gradient,
        adjoint,
        adjoint_result,
        reduced_hessian_action,
    )


def _solve_reduced_newton_krylov(
    method: ReducedNewtonKrylov,
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
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")
    state = _validate_real_inexact_tree(initial_state, name="initial_state")
    design = _validate_real_inexact_tree(initial_design, name="initial_design")
    if problem.design_bounds is not None:
        design = problem.design_bounds.project(design)
    state_result = problem.solve_state(design, state, args=args)
    state = state_result.state
    state_template, adjoint_template, reduced_template = _prepare_linear_templates(
        method,
        problem,
        state,
        design,
        state_result.residual,
    )
    initial_status = jnp.where(
        state_result.successful,
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.BACKEND_FAILED),
    ).astype(jnp.int32)
    initial_adjoint = jax.tree.map(jnp.zeros_like, state_result.residual)
    design_scalar = _tree_norm(design)
    initial_carry = (
        state_result,
        design,
        initial_status,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        state_result.diagnostics.objective_evaluations,
        jnp.asarray(0, dtype=jnp.int32),
        state_result.diagnostics.residual_evaluations,
        state_result.diagnostics.jvp_evaluations,
        state_result.diagnostics.vjp_evaluations,
        state_result.diagnostics.hvp_evaluations,
        state_result.diagnostics.linear_solves,
        state_result.diagnostics.linear_iterations,
        state_result.diagnostics.setup_refreshes + 3,
        state_result.diagnostics.numeric_refreshes,
        state_result.diagnostics.globalization_evaluations,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros_like(design_scalar),
        jnp.zeros_like(design_scalar),
        jnp.full_like(design_scalar, jnp.nan),
        initial_adjoint,
    )

    def condition(carry):
        (
            _,
            _,
            status,
            iterations,
            _,
            objective_evaluations,
            *_,
        ) = carry
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else objective_evaluations < termination.maximum_evaluations
        )
        return (
            (status == int(OptimizationStatus.ITERATING))
            & (iterations < termination.maximum_steps)
            & within_evaluations
        )

    def body(carry):
        (
            current_state_result,
            current_design,
            status,
            iterations,
            state_solves,
            objective_evaluations,
            gradient_evaluations,
            residual_evaluations,
            jvp_evaluations,
            vjp_evaluations,
            hvp_evaluations,
            linear_solves,
            linear_iterations,
            setup_refreshes,
            numeric_refreshes,
            globalization_evaluations,
            accepted_steps,
            rejected_steps,
            direction_fallbacks,
            final_step_norm,
            accepted_rate,
            initial_optimality,
            _,
        ) = carry
        current_state = current_state_result.state
        value, _ = problem.value(current_state, current_design, args)
        (
            _,
            reduced_gradient,
            adjoint,
            adjoint_result,
            reduced_hessian_action,
        ) = _reduced_model(
            method,
            problem,
            current_state,
            current_design,
            args,
            state_template,
            adjoint_template,
            numeric_refreshes,
        )
        projected_gradient = (
            reduced_gradient
            if problem.design_bounds is None
            else problem.design_bounds.projected_gradient(
                current_design,
                reduced_gradient,
            )
        )
        optimality = _tree_norm(projected_gradient)
        next_initial_optimality = jnp.where(
            gradient_evaluations == 0,
            optimality,
            initial_optimality,
        )

        def fail_model(status_code):
            return (
                current_state_result,
                current_design,
                jnp.asarray(status_code, dtype=jnp.int32),
                iterations,
                state_solves,
                objective_evaluations + 3,
                gradient_evaluations + 1,
                residual_evaluations + 3,
                jvp_evaluations + adjoint_result.diagnostics.adjoint_matvec_count,
                vjp_evaluations + adjoint_result.diagnostics.matvec_count + 1,
                hvp_evaluations,
                linear_solves + 1,
                linear_iterations + adjoint_result.diagnostics.iterations,
                setup_refreshes,
                numeric_refreshes + 2,
                globalization_evaluations,
                accepted_steps,
                rejected_steps + 1,
                direction_fallbacks,
                final_step_norm,
                accepted_rate,
                next_initial_optimality,
                adjoint,
            )

        def evaluate_direction(_):
            design_space = PyTreeSpace(current_design)
            reduced_operator = FunctionLinearOperator(
                reduced_hessian_action,
                source=design_space,
                target=design_space,
                transpose_action=reduced_hessian_action,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "construction"},
                ),
                operator_id="reduced-newton-hessian",
                closure_convert=False,
            )
            reduced_system = LinearSystem(
                reduced_operator,
                problem_id=f"{problem.problem_id}/reduced-hessian",
            )
            prepared_reduced = bind_numeric(
                reduced_template,
                reduced_system,
                numeric_version=numeric_refreshes + 2,
            )
            direction_result = solve_linear(
                prepared_reduced,
                _tree_negative(reduced_gradient),
            )
            action_count = jnp.asarray(
                direction_result.diagnostics.matvec_count,
                dtype=jnp.int32,
            ) + jnp.asarray(
                direction_result.diagnostics.adjoint_matvec_count,
                dtype=jnp.int32,
            )
            newton_direction = direction_result.value
            if problem.design_bounds is not None:
                newton_direction = _projected_displacement(
                    problem.design_bounds,
                    current_design,
                    newton_direction,
                )
            newton_directional = _tree_inner(
                reduced_gradient,
                newton_direction,
            )
            usable_direction = (
                _usable_linear_status(direction_result.status)
                & _tree_allfinite(newton_direction)
                & jnp.isfinite(newton_directional)
                & (newton_directional < 0.0)
            )

            def use_newton(_):
                return newton_direction, newton_directional

            def use_gradient(_):
                fallback = _tree_negative(reduced_gradient)
                if problem.design_bounds is not None:
                    fallback = _projected_displacement(
                        problem.design_bounds,
                        current_design,
                        fallback,
                    )
                return fallback, _tree_inner(reduced_gradient, fallback)

            direction, directional = jax.lax.cond(
                usable_direction,
                use_newton,
                use_gradient,
                None,
            )
            used_fallback = (~usable_direction).astype(jnp.int32)
            valid_direction = (
                _tree_allfinite(direction)
                & jnp.isfinite(directional)
                & (directional < 0.0)
            )
            # Incremental solves occur inside opaque Hessian actions. Their
            # diagnostics cannot escape the operator interface, so the partial
            # counters report only solves with directly observed diagnostics.
            next_linear_solves = linear_solves + 2
            next_linear_iterations = (
                linear_iterations
                + adjoint_result.diagnostics.iterations
                + direction_result.diagnostics.iterations
            )
            next_objective_evaluations = objective_evaluations + 3 + 2 * action_count
            next_residual_evaluations = residual_evaluations + 3 + 3 * action_count
            next_jvp_evaluations = (
                jvp_evaluations
                + adjoint_result.diagnostics.adjoint_matvec_count
                + 3 * action_count
            )
            next_vjp_evaluations = (
                vjp_evaluations
                + adjoint_result.diagnostics.matvec_count
                + 1
                + 2 * action_count
            )
            next_hvp_evaluations = hvp_evaluations + action_count

            def search(_):
                (
                    candidate_state_result,
                    candidate_design,
                    _,
                    accepted,
                    rate,
                    trials,
                    trial_objective_evaluations,
                    trial_residual_evaluations,
                    trial_jvp_evaluations,
                    trial_vjp_evaluations,
                    trial_hvp_evaluations,
                    trial_setup_refreshes,
                    trial_numeric_refreshes,
                    trial_linear_solves,
                    trial_linear_iterations,
                    trial_globalization_evaluations,
                ) = _state_design_line_search(
                    problem,
                    method.line_search,
                    current_state_result,
                    current_design,
                    value,
                    direction,
                    directional,
                    args,
                )
                step_norm = rate * _tree_norm(direction)
                stagnated = accepted & (
                    step_norm <= termination.step_threshold(_tree_norm(candidate_design))
                )
                next_status = jnp.where(
                    accepted,
                    jnp.where(
                        stagnated,
                        int(OptimizationStatus.STAGNATION),
                        int(OptimizationStatus.ITERATING),
                    ),
                    int(OptimizationStatus.LINE_SEARCH_FAILED),
                ).astype(jnp.int32)
                return (
                    candidate_state_result,
                    candidate_design,
                    next_status,
                    iterations + 1,
                    state_solves + trials,
                    next_objective_evaluations + trial_objective_evaluations,
                    gradient_evaluations + 1,
                    next_residual_evaluations + trial_residual_evaluations,
                    next_jvp_evaluations + trial_jvp_evaluations,
                    next_vjp_evaluations + trial_vjp_evaluations,
                    next_hvp_evaluations + trial_hvp_evaluations,
                    next_linear_solves + trial_linear_solves,
                    next_linear_iterations + trial_linear_iterations,
                    setup_refreshes + trial_setup_refreshes,
                    numeric_refreshes + 3 + trial_numeric_refreshes,
                    globalization_evaluations + trial_globalization_evaluations,
                    accepted_steps + accepted.astype(jnp.int32),
                    rejected_steps + (~accepted).astype(jnp.int32),
                    direction_fallbacks + used_fallback,
                    step_norm,
                    rate,
                    next_initial_optimality,
                    adjoint,
                )

            def fail_direction(_):
                return (
                    current_state_result,
                    current_design,
                    jnp.asarray(
                        int(OptimizationStatus.INVALID_DIRECTION),
                        dtype=jnp.int32,
                    ),
                    iterations,
                    state_solves,
                    next_objective_evaluations,
                    gradient_evaluations + 1,
                    next_residual_evaluations,
                    next_jvp_evaluations,
                    next_vjp_evaluations,
                    next_hvp_evaluations,
                    next_linear_solves,
                    next_linear_iterations,
                    setup_refreshes,
                    numeric_refreshes + 3,
                    globalization_evaluations,
                    accepted_steps,
                    rejected_steps + 1,
                    direction_fallbacks + used_fallback,
                    final_step_norm,
                    accepted_rate,
                    next_initial_optimality,
                    adjoint,
                )

            return jax.lax.cond(
                valid_direction,
                search,
                fail_direction,
                None,
            )

        def evaluate_finite_model(_):
            converged = optimality <= termination.optimality_threshold(
                next_initial_optimality
            )

            def finish_success(_):
                return (
                    current_state_result,
                    current_design,
                    jnp.asarray(
                        int(OptimizationStatus.SUCCESS),
                        dtype=jnp.int32,
                    ),
                    iterations,
                    state_solves,
                    objective_evaluations + 3,
                    gradient_evaluations + 1,
                    residual_evaluations + 3,
                    jvp_evaluations + adjoint_result.diagnostics.adjoint_matvec_count,
                    vjp_evaluations + adjoint_result.diagnostics.matvec_count + 1,
                    hvp_evaluations,
                    linear_solves + 1,
                    linear_iterations + adjoint_result.diagnostics.iterations,
                    setup_refreshes,
                    numeric_refreshes + 2,
                    globalization_evaluations,
                    accepted_steps,
                    rejected_steps,
                    direction_fallbacks,
                    final_step_norm,
                    accepted_rate,
                    next_initial_optimality,
                    adjoint,
                )

            return jax.lax.cond(
                converged,
                finish_success,
                evaluate_direction,
                None,
            )

        finite_model = (
            jnp.isfinite(value)
            & jnp.isfinite(optimality)
            & _tree_allfinite(reduced_gradient)
        )
        return jax.lax.cond(
            finite_model,
            lambda _: jax.lax.cond(
                _usable_linear_status(adjoint_result.status),
                evaluate_finite_model,
                lambda _: fail_model(OptimizationStatus.LINEAR_SOLVE_FAILED),
                None,
            ),
            lambda _: fail_model(OptimizationStatus.NONFINITE_EVALUATION),
            None,
        )

    (
        state_result,
        design,
        status,
        iterations,
        state_solves,
        objective_evaluations,
        gradient_evaluations,
        residual_evaluations,
        jvp_evaluations,
        vjp_evaluations,
        hvp_evaluations,
        linear_solves,
        linear_iterations,
        setup_refreshes,
        numeric_refreshes,
        globalization_evaluations,
        accepted_steps,
        rejected_steps,
        direction_fallbacks,
        final_step_norm,
        accepted_rate,
        initial_optimality,
        adjoint,
    ) = jax.lax.while_loop(condition, body, initial_carry)
    if termination.maximum_evaluations is not None:
        status = jnp.where(
            (status == int(OptimizationStatus.ITERATING))
            & (objective_evaluations >= termination.maximum_evaluations),
            int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
            status,
        )
    status = jnp.where(
        status == int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        status,
    ).astype(jnp.int32)

    state = state_result.state
    final_value, auxiliary = problem.value(state, design, args)
    objective_evaluations = objective_evaluations + 3
    (
        residual,
        final_gradient,
        adjoint,
        adjoint_result,
        _,
    ) = _reduced_model(
        method,
        problem,
        state,
        design,
        args,
        state_template,
        adjoint_template,
        numeric_refreshes,
    )
    gradient_evaluations = gradient_evaluations + 1
    residual_evaluations = residual_evaluations + 3
    jvp_evaluations = jvp_evaluations + adjoint_result.diagnostics.adjoint_matvec_count
    vjp_evaluations = vjp_evaluations + adjoint_result.diagnostics.matvec_count + 1
    linear_solves = linear_solves + 1
    linear_iterations = linear_iterations + adjoint_result.diagnostics.iterations
    numeric_refreshes = numeric_refreshes + 2
    projected_final_gradient = (
        final_gradient
        if problem.design_bounds is None
        else problem.design_bounds.projected_gradient(design, final_gradient)
    )
    final_optimality = _tree_norm(projected_final_gradient)
    status_allows_final_success = (
        (status == int(OptimizationStatus.ITERATING))
        | (status == int(OptimizationStatus.MAXIMUM_STEPS_REACHED))
        | (status == int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED))
        | (status == int(OptimizationStatus.STAGNATION))
    )
    status = jnp.where(
        status_allows_final_success
        & (final_optimality <= termination.optimality_threshold(initial_optimality)),
        int(OptimizationStatus.SUCCESS),
        status,
    ).astype(jnp.int32)
    primal = _tree_norm(residual)
    if problem.design_bounds is not None:
        primal = jnp.maximum(primal, problem.design_bounds.violation(design))
    diagnostics = OptimizationDiagnostics(
        iterations=iterations,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        objective_evaluations=objective_evaluations,
        gradient_evaluations=gradient_evaluations,
        residual_evaluations=residual_evaluations,
        jvp_evaluations=jvp_evaluations,
        vjp_evaluations=vjp_evaluations,
        hvp_evaluations=hvp_evaluations,
        constraint_evaluations=state_solves,
        linear_solves=linear_solves,
        setup_refreshes=setup_refreshes,
        numeric_refreshes=numeric_refreshes,
        linear_iterations=linear_iterations,
        globalization_evaluations=globalization_evaluations,
        initial_optimality_norm=initial_optimality,
        final_optimality_norm=final_optimality,
        final_step_norm=final_step_norm,
        accepted_step_size=accepted_rate,
        direction_fallbacks=direction_fallbacks,
        primal_feasibility=primal,
        dual_feasibility=final_optimality,
        complementarity=0.0,
        counts_complete=False,
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax-linalg",
        backend_method=type(method.reduced_linear_policy.method).__name__.lower(),
        globalization="reduced-objective-armijo",
        matrix_free=True,
        notes=(
            "Each reduced Hessian action solves one incremental state and one "
            "incremental adjoint equation through prepared linear operators."
        ),
    )
    return StateDesignResult(
        state,
        design,
        final_value,
        auxiliary,
        adjoint,
        status,
        diagnostics,
        provenance,
    )


__all__ = ["ReducedNewtonKrylov"]
