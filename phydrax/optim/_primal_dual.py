#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._linear_refresh import prepare_refresh_state
from ..linalg import (
    ArraySpace,
    DifferentiationPolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    MINRES,
    OperatorProperties,
    saddle_point_system,
    solve as solve_linear,
    TolerancePolicy,
)
from ._iterative._base import AbstractMinimizationMethod
from ._iterative._types import (
    _tree_allfinite,
    ConstrainedOptimalityCertificate,
    MinimizationProblem,
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._nonlinear_constraints import (
    _canonical_constraints,
    _constraint_layout,
    _constraint_violation,
    _max_abs,
    _max_positive,
)


def _default_kkt_policy(*, tolerance: float, maximum_steps: int) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        MINRES(),
        tolerance=TolerancePolicy(
            relative=tolerance,
            absolute=tolerance,
            max_steps=maximum_steps,
        ),
        differentiation=DifferentiationPolicy("algorithmic"),
    )


def _usable_linear_status(status: Any, /) -> Array:
    status_ = jnp.asarray(status, dtype=jnp.int32)
    return (
        (status_ == int(LinearSolveStatus.SUCCESS))
        | (status_ == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED))
        | (status_ == int(LinearSolveStatus.STAGNATION))
        | (status_ == int(LinearSolveStatus.CONDITION_LIMIT_REACHED))
    )


def _fraction_to_boundary(
    value: Array,
    direction: Array,
    fraction: float,
    /,
) -> Array:
    if value.size == 0:
        return jnp.asarray(1.0, dtype=value.dtype)
    ratios = jnp.where(direction < 0.0, -value / direction, jnp.inf)
    return jnp.minimum(
        jnp.asarray(1.0, dtype=value.dtype),
        fraction * jnp.min(ratios),
    )


def _residual_norm(
    stationarity: Array,
    equality: Array,
    inequality_slack: Array,
    complementarity: Array,
    /,
) -> Array:
    values = (
        jnp.linalg.norm(stationarity),
        _max_abs(equality),
        _max_abs(inequality_slack),
        _max_abs(complementarity),
    )
    return jnp.max(jnp.stack(values))


class _AbstractPrimalDualInteriorMethod(AbstractMinimizationMethod):
    """Shared matrix-free primal-dual interior method contract."""

    linear_policy: LinearSolvePolicy
    initial_barrier: float = eqx.field(static=True)
    barrier_reduction: float = eqx.field(static=True)
    centering: float = eqx.field(static=True)
    minimum_slack: float = eqx.field(static=True)
    fraction_to_boundary: float = eqx.field(static=True)
    kkt_regularization: float = eqx.field(static=True)
    active_tolerance: float = eqx.field(static=True)
    sufficient_decrease: float = eqx.field(static=True)
    line_search_contraction: float = eqx.field(static=True)
    maximum_line_search_steps: int = eqx.field(static=True)
    maximum_restoration_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        linear_tolerance: float = 1e-8,
        linear_maximum_steps: int = 200,
        initial_barrier: float = 1e-1,
        barrier_reduction: float = 0.2,
        centering: float = 0.1,
        minimum_slack: float = 1e-10,
        fraction_to_boundary: float = 0.995,
        kkt_regularization: float = 1e-8,
        active_tolerance: float = 1e-7,
        sufficient_decrease: float = 1e-4,
        line_search_contraction: float = 0.5,
        maximum_line_search_steps: int = 20,
        maximum_restoration_steps: int = 20,
    ):
        tolerance = float(linear_tolerance)
        linear_steps = int(linear_maximum_steps)
        policy = (
            _default_kkt_policy(tolerance=tolerance, maximum_steps=linear_steps)
            if linear_policy is None
            else linear_policy
        )
        values = tuple(
            float(value)
            for value in (
                initial_barrier,
                barrier_reduction,
                centering,
                minimum_slack,
                fraction_to_boundary,
                kkt_regularization,
                active_tolerance,
                sufficient_decrease,
                line_search_contraction,
            )
        )
        line_steps = int(maximum_line_search_steps)
        restoration_steps = int(maximum_restoration_steps)
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if not isfinite(tolerance) or tolerance <= 0.0 or linear_steps < 1:
            raise ValueError("Linear tolerance and step limit must be positive.")
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Primal-dual scalar controls must be positive and finite.")
        if not 0.0 < values[1] < 1.0:
            raise ValueError("barrier_reduction must lie in (0, 1).")
        if not 0.0 < values[2] < 1.0:
            raise ValueError("centering must lie in (0, 1).")
        if not 0.0 < values[4] < 1.0:
            raise ValueError("fraction_to_boundary must lie in (0, 1).")
        if not 0.0 < values[7] < 1.0:
            raise ValueError("sufficient_decrease must lie in (0, 1).")
        if not 0.0 < values[8] < 1.0:
            raise ValueError("line_search_contraction must lie in (0, 1).")
        if line_steps < 1 or restoration_steps < 1:
            raise ValueError("Line-search and restoration limits must be positive.")
        self.linear_policy = policy
        (
            self.initial_barrier,
            self.barrier_reduction,
            self.centering,
            self.minimum_slack,
            self.fraction_to_boundary,
            self.kkt_regularization,
            self.active_tolerance,
            self.sufficient_decrease,
            self.line_search_contraction,
        ) = values
        self.maximum_line_search_steps = line_steps
        self.maximum_restoration_steps = restoration_steps

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def predictor_corrector(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def requires_feasible_start(self) -> bool:
        raise NotImplementedError

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=True,
            prepared_refresh=True,
            implicit_differentiation=True,
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
        return _solve_primal_dual_newton_krylov(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


class PrimalDualNewtonKrylov(_AbstractPrimalDualInteriorMethod):
    """Centered matrix-free primal-dual interior Newton method."""

    @property
    def method_id(self) -> str:
        return "primal-dual-newton-krylov"

    @property
    def predictor_corrector(self) -> bool:
        return False

    @property
    def requires_feasible_start(self) -> bool:
        return False


class PrimalDualPredictorCorrector(_AbstractPrimalDualInteriorMethod):
    """Feasible-start Mehrotra predictor and complementarity corrector."""

    centering_power: float = eqx.field(static=True)
    require_feasible_start: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        centering_power: float = 3.0,
        require_feasible_start: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        power = float(centering_power)
        if not isfinite(power) or power <= 0.0:
            raise ValueError("centering_power must be positive and finite.")
        self.centering_power = power
        self.require_feasible_start = bool(require_feasible_start)

    @property
    def method_id(self) -> str:
        return "primal-dual-predictor-corrector/mehrotra"

    @property
    def predictor_corrector(self) -> bool:
        return True

    @property
    def requires_feasible_start(self) -> bool:
        return self.require_feasible_start


def _point_data(problem, layout, unravel, flat_parameters, args, multipliers):
    equality_multipliers, inequality_multipliers = multipliers

    def objective(candidate):
        return problem.value(unravel(candidate), args)[0]

    def constraints(candidate):
        return _canonical_constraints(problem, layout, unravel(candidate), args)

    value, gradient = jax.value_and_grad(objective)(flat_parameters)
    equality, inequality = constraints(flat_parameters)
    _, pullback = jax.vjp(constraints, flat_parameters)
    multiplier_part = pullback((equality_multipliers, inequality_multipliers))[0]
    stationarity = gradient + multiplier_part
    return value, gradient, equality, inequality, stationarity, constraints


def _restore_feasibility(
    method: _AbstractPrimalDualInteriorMethod,
    constraints,
    flat_parameters: Array,
    equality: Array,
    inequality: Array,
    /,
):
    def feasibility(candidate):
        candidate_equality, candidate_inequality = constraints(candidate)
        positive_inequality = jnp.maximum(candidate_inequality, 0.0)
        return 0.5 * (
            jnp.vdot(candidate_equality, candidate_equality).real
            + jnp.vdot(positive_inequality, positive_inequality).real
        )

    value, gradient = jax.value_and_grad(feasibility)(flat_parameters)
    direction = -gradient
    directional = -jnp.vdot(gradient, gradient).real
    usable = (
        jnp.isfinite(value)
        & jnp.isfinite(directional)
        & jnp.all(jnp.isfinite(direction))
        & (directional < 0.0)
    )

    def condition(carry):
        step, _, accepted, *_ = carry
        return (step < method.maximum_restoration_steps) & (~accepted) & usable

    def body(carry):
        (
            step,
            rate,
            _,
            accepted_parameters,
            accepted_equality,
            accepted_inequality,
            accepted_rate,
        ) = carry
        candidate = flat_parameters + rate * direction
        candidate_equality, candidate_inequality = constraints(candidate)
        positive_inequality = jnp.maximum(candidate_inequality, 0.0)
        candidate_value = 0.5 * (
            jnp.vdot(candidate_equality, candidate_equality).real
            + jnp.vdot(positive_inequality, positive_inequality).real
        )
        accepted = jnp.isfinite(candidate_value) & (
            candidate_value <= value + method.sufficient_decrease * rate * directional
        )
        return (
            step + 1,
            rate * method.line_search_contraction,
            accepted,
            jnp.where(accepted, candidate, accepted_parameters),
            jnp.where(accepted, candidate_equality, accepted_equality),
            jnp.where(accepted, candidate_inequality, accepted_inequality),
            jnp.where(accepted, rate, accepted_rate),
        )

    (
        evaluations,
        _,
        restored,
        parameters,
        restored_equality,
        restored_inequality,
        accepted_rate,
    ) = jax.lax.while_loop(
        condition,
        body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(1.0, dtype=flat_parameters.dtype),
            jnp.asarray(False),
            flat_parameters,
            equality,
            inequality,
            jnp.asarray(0.0, dtype=flat_parameters.dtype),
        ),
    )
    return (
        parameters,
        restored_equality,
        restored_inequality,
        restored,
        evaluations,
        accepted_rate,
    )


class _PrimalDualCounters(NamedTuple):
    accepted_steps: Array
    rejected_steps: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    constraint_evaluations: Array
    linear_solves: Array
    numeric_refreshes: Array
    linear_iterations: Array
    globalization_evaluations: Array
    direction_fallbacks: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    hvp_evaluations: Array


class _PrimalDualState(NamedTuple):
    iteration: Array
    parameters: Array
    equality_multipliers: Array
    inequality_multipliers: Array
    slacks: Array
    barrier: Array
    status: Array
    initial_optimality: Array
    final_step_norm: Array
    accepted_rate: Array
    counters: _PrimalDualCounters
    linear_refresh_arrays: Any


def _build_kkt_system(
    method: _AbstractPrimalDualInteriorMethod,
    problem: MinimizationProblem,
    layout,
    unravel,
    args,
    flat_parameters: Array,
    equality_multipliers: Array,
    equality: Array,
    inequality_multipliers: Array,
    slacks: Array,
    constraints,
    /,
):
    inverse_slack = 1.0 / jnp.maximum(slacks, method.minimum_slack)
    diagonal = inequality_multipliers * inverse_slack

    def lagrangian(candidate):
        objective = problem.value(unravel(candidate), args)[0]
        candidate_equality, candidate_inequality = constraints(candidate)
        return (
            objective
            + jnp.vdot(equality_multipliers, candidate_equality).real
            + jnp.vdot(inequality_multipliers, candidate_inequality).real
        )

    def primal_action(tangent):
        hessian_tangent = jax.jvp(
            jax.grad(lagrangian),
            (flat_parameters,),
            (tangent,),
        )[1]
        if slacks.size:
            inequality_tangent = jax.jvp(
                lambda candidate: constraints(candidate)[1],
                (flat_parameters,),
                (tangent,),
            )[1]
            _, inequality_pullback = jax.vjp(
                lambda candidate: constraints(candidate)[1],
                flat_parameters,
            )
            curvature = inequality_pullback(diagonal * inequality_tangent)[0]
        else:
            curvature = jnp.zeros_like(tangent)
        return hessian_tangent + curvature + method.kkt_regularization * tangent

    def equality_action(tangent):
        return jax.jvp(
            lambda candidate: constraints(candidate)[0],
            (flat_parameters,),
            (tangent,),
        )[1]

    def equality_transpose(cotangent):
        _, pullback = jax.vjp(
            lambda candidate: constraints(candidate)[0],
            flat_parameters,
        )
        return pullback(cotangent)[0]

    primal_space = ArraySpace(flat_parameters.shape, dtype=flat_parameters.dtype)
    equality_space = ArraySpace(equality.shape, dtype=equality.dtype)
    primal_operator = FunctionLinearOperator(
        primal_action,
        source=primal_space,
        target=primal_space,
        transpose_action=primal_action,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id="primal-dual-reduced-hessian",
        closure_convert=False,
    )
    equality_operator = FunctionLinearOperator(
        equality_action,
        source=primal_space,
        target=equality_space,
        transpose_action=equality_transpose,
        operator_id="primal-dual-equality-jacobian",
        closure_convert=False,
    )
    return saddle_point_system(
        primal_operator,
        equality_operator,
        operator_id="primal-dual-kkt",
        problem_id=f"{problem.problem_id}/primal-dual-kkt",
    )


def _solve_primal_dual_newton_krylov(
    method: _AbstractPrimalDualInteriorMethod,
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> MinimizationResult:
    if not problem.constraints and problem.bounds is None:
        raise ValueError(
            "Primal-dual interior methods require nonlinear constraints or "
            "parameter bounds."
        )
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")

    layout = _constraint_layout(problem, initial_parameters, args)
    flat_parameters, unravel = ravel_pytree(initial_parameters)
    equality, inequality = _canonical_constraints(
        problem,
        layout,
        initial_parameters,
        args,
    )
    if method.predictor_corrector and inequality.size == 0:
        raise ValueError(
            "PrimalDualPredictorCorrector requires at least one inequality "
            "constraint or finite one-sided parameter bound."
        )
    equality_multipliers = jnp.zeros_like(equality)
    inequality_multipliers = jnp.ones_like(inequality)
    slacks = jnp.maximum(-inequality, method.initial_barrier)
    barrier = jnp.asarray(method.initial_barrier, dtype=flat_parameters.dtype)
    constraint_sources = len(problem.constraints) + int(problem.bounds is not None)
    constraint_sources_ = jnp.asarray(constraint_sources, dtype=jnp.int32)
    equality_action_factor = jnp.asarray(
        1 + int(equality.size > 0),
        dtype=jnp.int32,
    )

    def initial_constraints(candidate):
        return _canonical_constraints(problem, layout, unravel(candidate), args)

    initial_kkt = _build_kkt_system(
        method,
        problem,
        layout,
        unravel,
        args,
        flat_parameters,
        equality_multipliers,
        equality,
        inequality_multipliers,
        slacks,
        initial_constraints,
    )
    _, linear_refresh_state = prepare_refresh_state(
        initial_kkt,
        method.linear_policy,
    )
    linear_refresh_arrays, linear_refresh_static = eqx.partition(
        linear_refresh_state,
        eqx.is_array,
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    counters = _PrimalDualCounters(
        accepted_steps=zero,
        rejected_steps=zero,
        objective_evaluations=zero,
        gradient_evaluations=zero,
        constraint_evaluations=constraint_sources_,
        linear_solves=zero,
        numeric_refreshes=jnp.asarray(1, dtype=jnp.int32),
        linear_iterations=zero,
        globalization_evaluations=zero,
        direction_fallbacks=zero,
        jvp_evaluations=zero,
        vjp_evaluations=zero,
        hvp_evaluations=zero,
    )
    finite_initial = (
        _tree_allfinite(initial_parameters)
        & jnp.all(jnp.isfinite(equality))
        & jnp.all(jnp.isfinite(inequality))
    )
    infeasible_start = method.requires_feasible_start & (
        _constraint_violation(equality, inequality) > method.active_tolerance
    )
    initial_status = jnp.where(
        ~finite_initial,
        int(OptimizationStatus.NONFINITE_INPUT),
        jnp.where(
            infeasible_start,
            int(OptimizationStatus.INFEASIBLE),
            int(OptimizationStatus.ITERATING),
        ),
    ).astype(jnp.int32)
    state = _PrimalDualState(
        iteration=zero,
        parameters=flat_parameters,
        equality_multipliers=equality_multipliers,
        inequality_multipliers=inequality_multipliers,
        slacks=slacks,
        barrier=barrier,
        status=initial_status,
        initial_optimality=jnp.asarray(jnp.nan, dtype=flat_parameters.dtype),
        final_step_norm=jnp.asarray(0.0, dtype=flat_parameters.dtype),
        accepted_rate=jnp.asarray(0.0, dtype=flat_parameters.dtype),
        counters=counters,
        linear_refresh_arrays=linear_refresh_arrays,
    )

    def outer_condition(current):
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else current.counters.objective_evaluations < termination.maximum_evaluations
        )
        return (
            (current.status == int(OptimizationStatus.ITERATING))
            & (current.iteration < termination.maximum_steps)
            & within_evaluations
        )

    def outer_body(current):
        (
            value,
            _,
            current_equality,
            current_inequality,
            stationarity,
            constraints,
        ) = _point_data(
            problem,
            layout,
            unravel,
            current.parameters,
            args,
            (
                current.equality_multipliers,
                current.inequality_multipliers,
            ),
        )
        evaluated_counters = current.counters._replace(
            objective_evaluations=current.counters.objective_evaluations + 1,
            gradient_evaluations=current.counters.gradient_evaluations + 1,
            constraint_evaluations=(
                current.counters.constraint_evaluations + 2 * constraint_sources_
            ),
            vjp_evaluations=current.counters.vjp_evaluations + 1,
        )
        inequality_slack = current_inequality + current.slacks
        complementarity_vector = current.slacks * current.inequality_multipliers
        primal = _constraint_violation(current_equality, current_inequality)
        dual = jnp.maximum(
            jnp.linalg.norm(stationarity),
            _max_positive(-current.inequality_multipliers),
        )
        complementarity = _max_abs(complementarity_vector)
        optimality = jnp.maximum(
            jnp.maximum(jnp.maximum(primal, dual), complementarity),
            _max_abs(inequality_slack),
        )
        initial_optimality = jnp.where(
            current.iteration == 0,
            optimality,
            current.initial_optimality,
        )
        evaluated = current._replace(
            initial_optimality=initial_optimality,
            counters=evaluated_counters,
        )
        finite = (
            jnp.isfinite(value)
            & jnp.isfinite(optimality)
            & jnp.all(jnp.isfinite(stationarity))
        )
        converged = optimality <= termination.optimality_threshold(initial_optimality)

        def nonfinite_step(_):
            return evaluated._replace(
                status=jnp.asarray(
                    int(OptimizationStatus.NONFINITE_EVALUATION),
                    dtype=jnp.int32,
                ),
                counters=evaluated.counters._replace(
                    rejected_steps=evaluated.counters.rejected_steps + 1,
                ),
            )

        def converged_step(_):
            return evaluated._replace(
                status=jnp.asarray(
                    int(OptimizationStatus.SUCCESS),
                    dtype=jnp.int32,
                )
            )

        def newton_step(_):
            inverse_slack = (
                1.0 / jnp.maximum(current.slacks, method.minimum_slack)
                if current_inequality.size
                else current.slacks
            )
            kkt_system = _build_kkt_system(
                method,
                problem,
                layout,
                unravel,
                args,
                current.parameters,
                current.equality_multipliers,
                current_equality,
                current.inequality_multipliers,
                current.slacks,
                constraints,
            )
            current_refresh_state = eqx.combine(
                current.linear_refresh_arrays,
                linear_refresh_static,
            )
            prepared_kkt, next_refresh_state = current_refresh_state.refresh(kkt_system)
            next_refresh_arrays, _ = eqx.partition(
                next_refresh_state,
                eqx.is_array,
            )

            def solve_direction(target, correction):
                if current_inequality.size:
                    reduced_rhs_part = inverse_slack * (
                        complementarity_vector
                        + correction
                        - target
                        - current.inequality_multipliers * inequality_slack
                    )
                    _, constraint_pullback = jax.vjp(
                        constraints,
                        current.parameters,
                    )
                    reduced_rhs = (
                        -stationarity
                        + constraint_pullback(
                            (
                                jnp.zeros_like(current_equality),
                                reduced_rhs_part,
                            )
                        )[0]
                    )
                else:
                    reduced_rhs = -stationarity
                result = solve_linear(
                    prepared_kkt,
                    (reduced_rhs, -current_equality),
                )
                primal_direction, equality_direction_ = result.value
                if current_inequality.size:
                    inequality_direction = jax.jvp(
                        lambda candidate: constraints(candidate)[1],
                        (current.parameters,),
                        (primal_direction,),
                    )[1]
                    slack_direction_ = -inequality_slack - inequality_direction
                    multiplier_direction_ = inverse_slack * (
                        -complementarity_vector
                        - correction
                        + target
                        + current.inequality_multipliers * inequality_slack
                        + current.inequality_multipliers * inequality_direction
                    )
                else:
                    slack_direction_ = current.slacks
                    multiplier_direction_ = current.inequality_multipliers
                return (
                    result,
                    primal_direction,
                    equality_direction_,
                    slack_direction_,
                    multiplier_direction_,
                )

            zero_correction = jnp.zeros_like(complementarity_vector)
            if method.predictor_corrector and current_inequality.size:
                (
                    affine_result,
                    _,
                    _,
                    affine_slack_direction,
                    affine_multiplier_direction,
                ) = solve_direction(
                    jnp.asarray(0.0, dtype=current.parameters.dtype),
                    zero_correction,
                )
                affine_primal_rate = _fraction_to_boundary(
                    current.slacks,
                    affine_slack_direction,
                    1.0,
                )
                affine_dual_rate = _fraction_to_boundary(
                    current.inequality_multipliers,
                    affine_multiplier_direction,
                    1.0,
                )
                average_complementarity = jnp.mean(complementarity_vector)
                affine_complementarity = jnp.mean(
                    (current.slacks + affine_primal_rate * affine_slack_direction)
                    * (
                        current.inequality_multipliers
                        + affine_dual_rate * affine_multiplier_direction
                    )
                )
                centering_ratio = jnp.clip(
                    affine_complementarity / jnp.maximum(average_complementarity, 1e-30),
                    0.0,
                    1.0,
                )
                centering_parameter = centering_ratio**method.centering_power
                target_barrier = centering_parameter * average_complementarity
                complementarity_correction = (
                    affine_slack_direction * affine_multiplier_direction
                )
                (
                    linear_result,
                    direction,
                    equality_direction,
                    slack_direction,
                    multiplier_direction,
                ) = solve_direction(
                    target_barrier,
                    complementarity_correction,
                )
                solve_count = jnp.asarray(2, dtype=jnp.int32)
                operator_actions = jnp.asarray(
                    affine_result.diagnostics.matvec_count,
                    dtype=jnp.int32,
                ) + jnp.asarray(
                    linear_result.diagnostics.matvec_count,
                    dtype=jnp.int32,
                )
                linear_iterations = jnp.asarray(
                    affine_result.diagnostics.iterations,
                    dtype=jnp.int32,
                ) + jnp.asarray(
                    linear_result.diagnostics.iterations,
                    dtype=jnp.int32,
                )
                usable_linear_status = _usable_linear_status(
                    affine_result.status
                ) & _usable_linear_status(linear_result.status)
            else:
                target_barrier = (
                    jnp.minimum(
                        current.barrier,
                        method.centering * jnp.mean(complementarity_vector),
                    )
                    if current_inequality.size
                    else jnp.asarray(
                        0.0,
                        dtype=current.parameters.dtype,
                    )
                )
                (
                    linear_result,
                    direction,
                    equality_direction,
                    slack_direction,
                    multiplier_direction,
                ) = solve_direction(
                    target_barrier,
                    zero_correction,
                )
                solve_count = jnp.asarray(1, dtype=jnp.int32)
                operator_actions = jnp.asarray(
                    linear_result.diagnostics.matvec_count,
                    dtype=jnp.int32,
                )
                linear_iterations = jnp.asarray(
                    linear_result.diagnostics.iterations,
                    dtype=jnp.int32,
                )
                usable_linear_status = _usable_linear_status(linear_result.status)

            derivative_increment = jnp.where(
                current_inequality.size > 0,
                solve_count,
                jnp.asarray(0, dtype=jnp.int32),
            )
            solve_counters = evaluated.counters._replace(
                linear_solves=(evaluated.counters.linear_solves + solve_count),
                numeric_refreshes=evaluated.counters.numeric_refreshes + 1,
                linear_iterations=(
                    evaluated.counters.linear_iterations + linear_iterations
                ),
                hvp_evaluations=(evaluated.counters.hvp_evaluations + operator_actions),
                jvp_evaluations=(
                    evaluated.counters.jvp_evaluations
                    + equality_action_factor * operator_actions
                    + derivative_increment
                ),
                vjp_evaluations=(
                    evaluated.counters.vjp_evaluations
                    + equality_action_factor * operator_actions
                    + derivative_increment
                ),
            )
            usable_direction = (
                usable_linear_status
                & jnp.all(jnp.isfinite(direction))
                & jnp.all(jnp.isfinite(equality_direction))
                & jnp.all(jnp.isfinite(multiplier_direction))
                & jnp.all(jnp.isfinite(slack_direction))
            )
            current_residual = _residual_norm(
                stationarity,
                current_equality,
                inequality_slack,
                complementarity_vector - target_barrier,
            )

            def line_search(_):
                initial_rate = jnp.minimum(
                    _fraction_to_boundary(
                        current.slacks,
                        slack_direction,
                        method.fraction_to_boundary,
                    ),
                    _fraction_to_boundary(
                        current.inequality_multipliers,
                        multiplier_direction,
                        method.fraction_to_boundary,
                    ),
                )

                def line_condition(carry):
                    trial, _, accepted, *_ = carry
                    return (trial < method.maximum_line_search_steps) & (~accepted)

                def line_body(carry):
                    (
                        trial,
                        rate,
                        _,
                        accepted_parameters,
                        accepted_equality_multipliers,
                        accepted_inequality_multipliers,
                        accepted_slacks,
                        accepted_step_norm,
                        accepted_rate,
                    ) = carry
                    candidate_parameters = current.parameters + rate * direction
                    candidate_equality_multipliers = (
                        current.equality_multipliers + rate * equality_direction
                    )
                    candidate_inequality_multipliers = (
                        current.inequality_multipliers + rate * multiplier_direction
                    )
                    candidate_slacks = current.slacks + rate * slack_direction
                    (
                        candidate_value,
                        _,
                        candidate_equality,
                        candidate_inequality,
                        candidate_stationarity,
                        _,
                    ) = _point_data(
                        problem,
                        layout,
                        unravel,
                        candidate_parameters,
                        args,
                        (
                            candidate_equality_multipliers,
                            candidate_inequality_multipliers,
                        ),
                    )
                    candidate_residual = _residual_norm(
                        candidate_stationarity,
                        candidate_equality,
                        candidate_inequality + candidate_slacks,
                        (
                            candidate_slacks * candidate_inequality_multipliers
                            - target_barrier
                        ),
                    )
                    sufficient = (
                        candidate_residual
                        <= (1.0 - method.sufficient_decrease * rate) * current_residual
                    )
                    accepted = (
                        jnp.isfinite(candidate_value)
                        & jnp.isfinite(candidate_residual)
                        & jnp.all(candidate_slacks > 0.0)
                        & jnp.all(candidate_inequality_multipliers > 0.0)
                        & sufficient
                    )
                    return (
                        trial + 1,
                        rate * method.line_search_contraction,
                        accepted,
                        jnp.where(
                            accepted,
                            candidate_parameters,
                            accepted_parameters,
                        ),
                        jnp.where(
                            accepted,
                            candidate_equality_multipliers,
                            accepted_equality_multipliers,
                        ),
                        jnp.where(
                            accepted,
                            candidate_inequality_multipliers,
                            accepted_inequality_multipliers,
                        ),
                        jnp.where(
                            accepted,
                            candidate_slacks,
                            accepted_slacks,
                        ),
                        jnp.where(
                            accepted,
                            rate * jnp.linalg.norm(direction),
                            accepted_step_norm,
                        ),
                        jnp.where(accepted, rate, accepted_rate),
                    )

                return jax.lax.while_loop(
                    line_condition,
                    line_body,
                    (
                        jnp.asarray(0, dtype=jnp.int32),
                        initial_rate,
                        jnp.asarray(False),
                        current.parameters,
                        current.equality_multipliers,
                        current.inequality_multipliers,
                        current.slacks,
                        current.final_step_norm,
                        current.accepted_rate,
                    ),
                )

            def unusable_line_search(_):
                return (
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0.0, dtype=current.parameters.dtype),
                    jnp.asarray(False),
                    current.parameters,
                    current.equality_multipliers,
                    current.inequality_multipliers,
                    current.slacks,
                    current.final_step_norm,
                    current.accepted_rate,
                )

            (
                line_evaluations,
                _,
                accepted,
                candidate_parameters,
                candidate_equality_multipliers,
                candidate_inequality_multipliers,
                candidate_slacks,
                candidate_step_norm,
                candidate_rate,
            ) = jax.lax.cond(
                usable_direction,
                line_search,
                unusable_line_search,
                None,
            )
            trial_counters = solve_counters._replace(
                objective_evaluations=(
                    solve_counters.objective_evaluations + line_evaluations
                ),
                gradient_evaluations=(
                    solve_counters.gradient_evaluations + line_evaluations
                ),
                constraint_evaluations=(
                    solve_counters.constraint_evaluations
                    + 2 * constraint_sources_ * line_evaluations
                ),
                globalization_evaluations=(
                    solve_counters.globalization_evaluations + line_evaluations
                ),
                vjp_evaluations=(solve_counters.vjp_evaluations + line_evaluations),
            )

            def accept_newton_step(_):
                stagnated = candidate_step_norm <= termination.step_threshold(
                    jnp.linalg.norm(candidate_parameters)
                )
                next_status = jnp.where(
                    stagnated,
                    int(OptimizationStatus.STAGNATION),
                    int(OptimizationStatus.ITERATING),
                ).astype(jnp.int32)
                return evaluated._replace(
                    iteration=evaluated.iteration + 1,
                    parameters=candidate_parameters,
                    equality_multipliers=candidate_equality_multipliers,
                    inequality_multipliers=candidate_inequality_multipliers,
                    slacks=candidate_slacks,
                    barrier=jnp.maximum(
                        method.minimum_slack,
                        evaluated.barrier * method.barrier_reduction,
                    ),
                    status=next_status,
                    final_step_norm=candidate_step_norm,
                    accepted_rate=candidate_rate,
                    counters=trial_counters._replace(
                        accepted_steps=trial_counters.accepted_steps + 1,
                    ),
                    linear_refresh_arrays=next_refresh_arrays,
                )

            def restore_step(_):
                (
                    restored_parameters,
                    _,
                    restored_inequality,
                    restored,
                    restoration_evaluations,
                    restoration_rate,
                ) = _restore_feasibility(
                    method,
                    constraints,
                    current.parameters,
                    current_equality,
                    current_inequality,
                )
                restoration_counters = trial_counters._replace(
                    accepted_steps=(
                        trial_counters.accepted_steps + restored.astype(jnp.int32)
                    ),
                    rejected_steps=trial_counters.rejected_steps + 1,
                    gradient_evaluations=(trial_counters.gradient_evaluations + 1),
                    constraint_evaluations=(
                        trial_counters.constraint_evaluations
                        + constraint_sources_ * (restoration_evaluations + 1)
                    ),
                    globalization_evaluations=(
                        trial_counters.globalization_evaluations + restoration_evaluations
                    ),
                    direction_fallbacks=(trial_counters.direction_fallbacks + 1),
                    vjp_evaluations=trial_counters.vjp_evaluations + 1,
                )
                failed_restoration = evaluated._replace(
                    iteration=evaluated.iteration + 1,
                    status=jnp.asarray(
                        int(OptimizationStatus.RESTORATION_FAILED),
                        dtype=jnp.int32,
                    ),
                    counters=restoration_counters,
                    linear_refresh_arrays=next_refresh_arrays,
                )

                def commit_restoration(_):
                    return failed_restoration._replace(
                        parameters=restored_parameters,
                        inequality_multipliers=jnp.maximum(
                            current.inequality_multipliers,
                            method.initial_barrier,
                        ),
                        slacks=jnp.maximum(
                            -restored_inequality,
                            method.initial_barrier,
                        ),
                        status=jnp.asarray(
                            int(OptimizationStatus.ITERATING),
                            dtype=jnp.int32,
                        ),
                        final_step_norm=jnp.linalg.norm(
                            restored_parameters - current.parameters
                        ),
                        accepted_rate=restoration_rate,
                    )

                return jax.lax.cond(
                    restored,
                    commit_restoration,
                    lambda _: failed_restoration,
                    None,
                )

            if method.predictor_corrector:

                def reject_predictor_corrector(_):
                    return evaluated._replace(
                        iteration=evaluated.iteration + 1,
                        status=jnp.where(
                            usable_direction,
                            int(OptimizationStatus.LINE_SEARCH_FAILED),
                            int(OptimizationStatus.LINEAR_SOLVE_FAILED),
                        ).astype(jnp.int32),
                        counters=trial_counters._replace(
                            rejected_steps=trial_counters.rejected_steps + 1,
                        ),
                        linear_refresh_arrays=next_refresh_arrays,
                    )

                return jax.lax.cond(
                    accepted,
                    accept_newton_step,
                    reject_predictor_corrector,
                    None,
                )
            return jax.lax.cond(
                accepted,
                accept_newton_step,
                restore_step,
                None,
            )

        return jax.lax.cond(
            finite,
            lambda _: jax.lax.cond(
                converged,
                converged_step,
                newton_step,
                None,
            ),
            nonfinite_step,
            None,
        )

    state = jax.lax.while_loop(outer_condition, outer_body, state)
    if termination.maximum_evaluations is not None:
        state = state._replace(
            status=jnp.where(
                (state.status == int(OptimizationStatus.ITERATING))
                & (
                    state.counters.objective_evaluations
                    >= termination.maximum_evaluations
                ),
                int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                state.status,
            ).astype(jnp.int32)
        )
    state = state._replace(
        status=jnp.where(
            state.status == int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
            state.status,
        ).astype(jnp.int32)
    )

    parameters = unravel(state.parameters)
    (
        final_value,
        _,
        equality,
        inequality,
        stationarity,
        _,
    ) = _point_data(
        problem,
        layout,
        unravel,
        state.parameters,
        args,
        (
            state.equality_multipliers,
            state.inequality_multipliers,
        ),
    )
    primal = _constraint_violation(equality, inequality)
    dual = jnp.maximum(
        jnp.linalg.norm(stationarity),
        _max_positive(-state.inequality_multipliers),
    )
    complementarity = (
        _max_abs(state.inequality_multipliers * inequality)
        if inequality.size
        else jnp.asarray(0.0, dtype=state.parameters.dtype)
    )
    final_optimality = jnp.maximum(jnp.maximum(primal, dual), complementarity)
    success_eligible = (
        (state.status == int(OptimizationStatus.ITERATING))
        | (state.status == int(OptimizationStatus.MAXIMUM_STEPS_REACHED))
        | (state.status == int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED))
        | (state.status == int(OptimizationStatus.STAGNATION))
    )
    status = jnp.where(
        success_eligible
        & (
            final_optimality <= termination.optimality_threshold(state.initial_optimality)
        ),
        int(OptimizationStatus.SUCCESS),
        state.status,
    ).astype(jnp.int32)
    _, auxiliary = problem.value(parameters, args)
    packaged_counters = state.counters._replace(
        objective_evaluations=state.counters.objective_evaluations + 2,
        gradient_evaluations=state.counters.gradient_evaluations + 1,
        constraint_evaluations=(
            state.counters.constraint_evaluations + 2 * constraint_sources_
        ),
        vjp_evaluations=state.counters.vjp_evaluations + 1,
    )
    active_mask = jnp.maximum(-inequality, 0.0) <= method.active_tolerance
    diagnostics = OptimizationDiagnostics(
        iterations=state.iteration,
        accepted_steps=packaged_counters.accepted_steps,
        rejected_steps=packaged_counters.rejected_steps,
        objective_evaluations=packaged_counters.objective_evaluations,
        gradient_evaluations=packaged_counters.gradient_evaluations,
        jvp_evaluations=packaged_counters.jvp_evaluations,
        vjp_evaluations=packaged_counters.vjp_evaluations,
        hvp_evaluations=packaged_counters.hvp_evaluations,
        jacobian_evaluations=0,
        constraint_evaluations=packaged_counters.constraint_evaluations,
        linear_solves=packaged_counters.linear_solves,
        setup_refreshes=1,
        numeric_refreshes=packaged_counters.numeric_refreshes,
        linear_iterations=packaged_counters.linear_iterations,
        globalization_evaluations=packaged_counters.globalization_evaluations,
        initial_optimality_norm=state.initial_optimality,
        final_optimality_norm=final_optimality,
        final_step_norm=state.final_step_norm,
        accepted_step_size=state.accepted_rate,
        damping=state.barrier,
        direction_fallbacks=packaged_counters.direction_fallbacks,
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        active_constraints=jnp.sum(active_mask, dtype=jnp.int32),
        counts_complete=False,
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax-linalg",
        backend_method=type(method.linear_policy.method).__name__.lower(),
        globalization=(
            "mehrotra-predictor-corrector-residual"
            if method.predictor_corrector
            else "barrier-residual-with-feasibility-restoration"
        ),
        matrix_free=True,
        implicit_differentiation=True,
        notes=(
            "Each accepted predictor-corrector step uses an affine KKT predictor, "
            "the affine complementarity ratio for centering, and the product of "
            "affine slack and multiplier directions in the corrector equation."
            if method.predictor_corrector
            else "Centered matrix-free KKT steps with feasibility restoration."
        ),
    )
    certificate = ConstrainedOptimalityCertificate(
        equality_multipliers=state.equality_multipliers,
        inequality_multipliers=state.inequality_multipliers,
        slacks=jnp.maximum(-inequality, 0.0),
        active_mask=active_mask,
        stationarity_residual=unravel(stationarity),
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        equality_sources=layout.equality_sources,
        inequality_sources=layout.inequality_sources,
    )
    return MinimizationResult(
        parameters,
        final_value,
        auxiliary,
        status,
        diagnostics,
        provenance,
        certificate=certificate,
    )


__all__ = ["PrimalDualNewtonKrylov", "PrimalDualPredictorCorrector"]
