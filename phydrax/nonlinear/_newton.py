#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array, PyTree

from .._linear_refresh import LinearRefreshState, prepare_refresh_state
from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._strict import StrictModule
from .._tree_math import (
    tree_add_scaled,
    tree_allfinite,
    tree_scale,
    tree_where,
)
from ..linalg import (
    AbstractVectorSpace,
    DifferentiationPolicy,
    GMRES,
    initialize_recycling,
    LinearSolveControl,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    RecyclingState,
    solve as solve_linear,
    solve_recycled,
    TolerancePolicy,
)
from ._linearization import (
    _jacobian_solve_direction,
    _jacobian_solve_operator,
    _jacobian_solve_right_hand_side,
    JacobianPolicy,
    prepare_jacobian,
)
from ._preconditioning import AbstractNonlinearSystemTransformation
from ._types import (
    AbstractNonlinearMethod,
    NonlinearCapabilities,
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)


def _default_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        GMRES(restart=16),
        tolerance=TolerancePolicy(relative=1e-6, absolute=1e-10, max_steps=64),
    )


def _iteration_linear_policy(policy: LinearSolvePolicy, /) -> LinearSolvePolicy:
    """Disable nested solve derivatives inside an explicitly differentiated root."""
    if policy.differentiation.mode == "none":
        return policy
    return eqx.tree_at(
        lambda candidate: candidate.differentiation,
        policy,
        DifferentiationPolicy("none"),
    )


def _usable_linear_status(status: Any, /) -> Array:
    value = jnp.asarray(status, dtype=jnp.int32)
    return (
        (value == int(LinearSolveStatus.SUCCESS))
        | (value == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED))
        | (value == int(LinearSolveStatus.STAGNATION))
        | (value == int(LinearSolveStatus.CONDITION_LIMIT_REACHED))
    )


def _space_inner(
    space,
    left: PyTree[Any],
    right: PyTree[Any],
    precision: NonlinearPrecisionPolicy,
    /,
) -> Array:
    return precision.inner(space, left, right)


def _space_norm(
    space,
    vector: PyTree[Any],
    precision: NonlinearPrecisionPolicy,
    /,
) -> Array:
    return precision.norm(space, vector)


def _tree_cast_like(
    value: PyTree[Any],
    reference: PyTree[Any],
    /,
) -> PyTree[Array]:
    return jax.tree.map(
        lambda item, template: jnp.asarray(item, dtype=template.dtype),
        value,
        reference,
    )


def _bound_space(
    space: AbstractVectorSpace | None,
    name: str,
    /,
) -> AbstractVectorSpace:
    if space is None:
        raise ValueError(f"A Newton problem must have a bound {name} space.")
    return space


def _remaining_evaluations(
    termination: NonlinearTermination,
    used: Array,
    /,
) -> Array:
    if termination.maximum_evaluations is None:
        return jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    return jnp.maximum(
        jnp.asarray(termination.maximum_evaluations, dtype=jnp.int32) - used,
        0,
    )


def _remaining_linear_steps(
    termination: NonlinearTermination,
    used: Array,
    structural_limit: int,
    /,
) -> Array:
    if termination.maximum_linear_iterations is None:
        return jnp.asarray(structural_limit, dtype=jnp.int32)
    remaining = jnp.maximum(
        jnp.asarray(termination.maximum_linear_iterations, dtype=jnp.int32) - used,
        0,
    )
    return jnp.minimum(remaining, structural_limit)


def _linear_failure_status(status: Any, /) -> Array:
    values = jnp.asarray(status, dtype=jnp.int32)
    singular = jnp.any(
        (values == int(LinearSolveStatus.SINGULAR))
        | (values == int(LinearSolveStatus.RANK_DEFICIENT))
        | (values == int(LinearSolveStatus.BREAKDOWN))
    )
    nonfinite = jnp.any(
        (values == int(LinearSolveStatus.NONFINITE_INPUT))
        | (values == int(LinearSolveStatus.NONFINITE_OUTPUT))
    )
    capability = jnp.any(
        (values == int(LinearSolveStatus.INCOMPATIBLE_STRUCTURE))
        | (values == int(LinearSolveStatus.ADJOINT_FAILED))
        | (values == int(LinearSolveStatus.CAPABILITY_REJECTED))
    )
    return jnp.where(
        singular,
        int(NonlinearStatus.SINGULAR_JACOBIAN),
        jnp.where(
            nonfinite,
            int(NonlinearStatus.NONFINITE_EVALUATION),
            jnp.where(
                capability,
                int(NonlinearStatus.CAPABILITY_REJECTED),
                int(NonlinearStatus.LINEAR_SOLVE_FAILED),
            ),
        ),
    ).astype(jnp.int32)


class RootLineSearch(StrictModule):
    """Armijo search over the squared physical residual norm."""

    initial_rate: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    sufficient_decrease: float = eqx.field(static=True)
    minimum_rate: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_rate: float = 1.0,
        contraction: float = 0.5,
        sufficient_decrease: float = 1e-4,
        minimum_rate: float = 1e-12,
        maximum_steps: int = 24,
    ):
        values = tuple(
            float(value)
            for value in (
                initial_rate,
                contraction,
                sufficient_decrease,
                minimum_rate,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Line-search values must be finite and positive.")
        if values[1] >= 1.0 or values[2] >= 1.0 or values[3] > values[0]:
            raise ValueError("Invalid root line-search contraction or rate bounds.")
        steps = int(maximum_steps)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        (
            self.initial_rate,
            self.contraction,
            self.sufficient_decrease,
            self.minimum_rate,
        ) = values
        self.maximum_steps = steps


class RootTrustRegion(StrictModule):
    """Residual-model trust-region acceptance and radius policy."""

    initial_radius: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    acceptance_ratio: float = eqx.field(static=True)
    shrink_ratio: float = eqx.field(static=True)
    expansion_ratio: float = eqx.field(static=True)
    shrink: float = eqx.field(static=True)
    growth: float = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_radius: float = 1.0,
        minimum_radius: float = 1e-12,
        maximum_radius: float = 1e6,
        acceptance_ratio: float = 1e-4,
        shrink_ratio: float = 0.25,
        expansion_ratio: float = 0.75,
        shrink: float = 0.25,
        growth: float = 2.0,
        maximum_attempts: int = 12,
    ):
        values = tuple(
            float(value)
            for value in (
                initial_radius,
                minimum_radius,
                maximum_radius,
                acceptance_ratio,
                shrink_ratio,
                expansion_ratio,
                shrink,
                growth,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Trust-region values must be finite and positive.")
        if not values[1] <= values[0] <= values[2]:
            raise ValueError("Trust radii must satisfy minimum <= initial <= maximum.")
        if not values[3] < values[4] < values[5] < 1.0:
            raise ValueError(
                "Trust-region acceptance thresholds must be ordered below one."
            )
        if values[6] >= 1.0 or values[7] <= 1.0:
            raise ValueError("Trust-region shrink/growth factors are invalid.")
        attempts = int(maximum_attempts)
        if attempts < 1:
            raise ValueError("maximum_attempts must be positive.")
        (
            self.initial_radius,
            self.minimum_radius,
            self.maximum_radius,
            self.acceptance_ratio,
            self.shrink_ratio,
            self.expansion_ratio,
            self.shrink,
            self.growth,
        ) = values
        self.maximum_attempts = attempts


NewtonForcingStrategy: TypeAlias = Literal["constant", "eisenstat-walker"]


class NewtonForcingPolicy(StrictModule):
    """Per-step inexact-Newton forcing terms for prepared linear solves."""

    strategy: NewtonForcingStrategy = eqx.field(static=True)
    initial: float = eqx.field(static=True)
    minimum: float = eqx.field(static=True)
    maximum: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)

    def __init__(
        self,
        strategy: NewtonForcingStrategy = "eisenstat-walker",
        /,
        *,
        initial: float = 0.5,
        minimum: float = 1e-8,
        maximum: float = 0.9,
        gamma: float = 0.9,
        exponent: float = 1.5,
    ):
        if strategy not in ("constant", "eisenstat-walker"):
            raise ValueError("Unknown inexact-Newton forcing strategy.")
        values = tuple(
            float(value) for value in (initial, minimum, maximum, gamma, exponent)
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Forcing-policy values must be finite and positive.")
        if not values[1] <= values[0] <= values[2] < 1.0:
            raise ValueError(
                "Forcing terms must satisfy minimum <= initial <= maximum < 1."
            )
        self.strategy = strategy
        (
            self.initial,
            self.minimum,
            self.maximum,
            self.gamma,
            self.exponent,
        ) = values

    def next(
        self,
        current: Array,
        current_residual_norm: Array,
        candidate_residual_norm: Array,
        /,
    ) -> Array:
        if self.strategy == "constant":
            return current
        ratio = candidate_residual_norm / jnp.maximum(current_residual_norm, 1e-30)
        predicted = self.gamma * ratio**self.exponent
        safeguarded = jnp.maximum(predicted, self.gamma * current**self.exponent)
        return jnp.clip(safeguarded, self.minimum, self.maximum)


JacobianRefreshStrategy: TypeAlias = Literal[
    "every-step", "periodic", "stagnation", "rejection"
]


class JacobianRefreshPolicy(StrictModule):
    """Policy for rebuilding nonlinear derivative actions."""

    strategy: JacobianRefreshStrategy = eqx.field(static=True)
    period: int = eqx.field(static=True)
    residual_reduction: float = eqx.field(static=True)

    def __init__(
        self,
        strategy: JacobianRefreshStrategy = "every-step",
        /,
        *,
        period: int = 1,
        residual_reduction: float = 0.5,
    ):
        if strategy not in ("every-step", "periodic", "stagnation", "rejection"):
            raise ValueError("Unknown Jacobian refresh strategy.")
        period_ = int(period)
        reduction = float(residual_reduction)
        if period_ < 1:
            raise ValueError("period must be positive.")
        if not isfinite(reduction) or not 0.0 < reduction < 1.0:
            raise ValueError("residual_reduction must be finite and lie in (0, 1).")
        self.strategy = strategy
        self.period = period_
        self.residual_reduction = reduction

    def should_refresh(
        self,
        age: Array,
        current_residual_norm: Array,
        reference_residual_norm: Array,
        rejected_steps: Array,
        reference_rejected_steps: Array,
        /,
    ) -> Array:
        if self.strategy == "every-step":
            return age > 0
        if self.strategy == "periodic":
            return age >= self.period
        if self.strategy == "rejection":
            return rejected_steps > reference_rejected_steps
        return (age > 0) & (
            current_residual_norm
            > self.residual_reduction * jnp.maximum(reference_residual_norm, 1e-30)
        )


class _RootState(StrictModule):
    residual: PyTree[Array]
    auxiliary: Any
    initial_residual_norm: Array
    residual_norm: Array
    step_norm: Array
    iteration: Array
    residual_evaluations: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array
    accepted_steps: Array
    rejected_steps: Array
    globalization_rejections: Array
    domain_failures: Array
    nonfinite_trials: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    forcing: Array
    last_forcing: Array
    jacobian_age: Array
    jacobian_reference_residual_norm: Array
    jacobian_reference_rejected_steps: Array
    trust_radius: Array
    status: Array
    refresh_state: LinearRefreshState
    recycling: RecyclingState | None
    final_linear_status: Array
    final_linear_rank: Array
    final_linear_condition_estimate: Array
    final_linear_residual_norm: Array
    final_linear_converged: Array


class _SearchResult(StrictModule):
    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    rate: Array
    evaluations: Array
    rejections: Array
    accepted: Array
    finite_seen: Array
    domain_failures: Array
    nonfinite_trials: Array


class _TrustResult(StrictModule):
    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    step: PyTree[Array]
    radius: Array
    evaluations: Array
    rejections: Array
    accepted: Array
    finite_seen: Array
    domain_failures: Array
    nonfinite_trials: Array


def _root_line_search(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    residual: PyTree[Any],
    auxiliary: Any,
    direction: PyTree[Any],
    directional_derivative: Any,
    maximum_evaluations: Array,
    args: Any,
    policy: RootLineSearch,
    precision: NonlinearPrecisionPolicy,
    /,
) -> _SearchResult:
    merit = 0.5 * jnp.real(
        _space_inner(
            _bound_space(problem.residual_space, "residual"),
            residual,
            residual,
            precision,
        )
    )
    scalar_dtype = merit.dtype
    initial_rate = jnp.asarray(policy.initial_rate, dtype=scalar_dtype)
    initial_carry = (
        state,
        residual,
        auxiliary,
        initial_rate,
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def body(_, carry):
        (
            accepted_state,
            accepted_residual,
            accepted_auxiliary,
            rate,
            accepted,
            finite_seen,
            evaluations,
            domain_failures,
            nonfinite_trials,
        ) = carry

        def attempt(__):
            candidate = _tree_cast_like(
                tree_add_scaled(state, direction, rate),
                state,
            )
            candidate_residual, candidate_auxiliary = problem.evaluate(candidate, args)
            finite = tree_allfinite(candidate) & tree_allfinite(candidate_residual)
            valid = problem.valid(
                candidate, candidate_residual, candidate_auxiliary, args
            )
            candidate_merit = 0.5 * jnp.real(
                _space_inner(
                    _bound_space(problem.residual_space, "residual"),
                    candidate_residual,
                    candidate_residual,
                    precision,
                )
            )
            sufficient = candidate_merit <= (
                merit + policy.sufficient_decrease * rate * directional_derivative
            )
            use = finite & valid & sufficient
            next_rate = jnp.where(
                use,
                rate,
                jnp.maximum(rate * policy.contraction, policy.minimum_rate),
            )
            return (
                tree_where(use, candidate, accepted_state),
                tree_where(use, candidate_residual, accepted_residual),
                tree_where(use, candidate_auxiliary, accepted_auxiliary),
                next_rate,
                use,
                finite_seen | (finite & valid),
                evaluations + 1,
                domain_failures + (finite & ~valid).astype(jnp.int32),
                nonfinite_trials + (~finite).astype(jnp.int32),
            )

        can_attempt = (~accepted) & (evaluations < maximum_evaluations)
        return jax.lax.cond(can_attempt, attempt, lambda _: carry, operand=None)

    (
        accepted_state,
        accepted_residual,
        accepted_auxiliary,
        final_rate,
        accepted,
        finite_seen,
        evaluations,
        domain_failures,
        nonfinite_trials,
    ) = jax.lax.fori_loop(0, policy.maximum_steps, body, initial_carry)
    return _SearchResult(
        state=accepted_state,
        residual=accepted_residual,
        auxiliary=accepted_auxiliary,
        rate=jnp.where(accepted, final_rate, jnp.asarray(0.0, dtype=scalar_dtype)),
        evaluations=evaluations,
        rejections=evaluations - accepted.astype(jnp.int32),
        accepted=accepted,
        finite_seen=finite_seen,
        domain_failures=domain_failures,
        nonfinite_trials=nonfinite_trials,
    )


def _dogleg_step(
    state_space,
    newton_direction: PyTree[Any],
    cauchy_direction: PyTree[Any],
    radius: Array,
    precision: NonlinearPrecisionPolicy,
    /,
) -> tuple[PyTree[Array], Array]:
    newton_norm = _space_norm(state_space, newton_direction, precision)
    cauchy_norm = _space_norm(state_space, cauchy_direction, precision)
    scaled_cauchy = tree_scale(
        radius / jnp.maximum(cauchy_norm, 1e-30),
        cauchy_direction,
    )
    scaled_cauchy = _tree_cast_like(scaled_cauchy, cauchy_direction)
    difference = tree_add_scaled(newton_direction, cauchy_direction, -1.0)
    quadratic = jnp.real(_space_inner(state_space, difference, difference, precision))
    linear = 2.0 * jnp.real(
        _space_inner(state_space, cauchy_direction, difference, precision)
    )
    constant = (
        jnp.real(
            _space_inner(
                state_space,
                cauchy_direction,
                cauchy_direction,
                precision,
            )
        )
        - radius**2
    )
    discriminant = jnp.maximum(linear**2 - 4.0 * quadratic * constant, 0.0)
    interpolation = (-linear + jnp.sqrt(discriminant)) / jnp.maximum(
        2.0 * quadratic, 1e-30
    )
    dogleg = tree_add_scaled(cauchy_direction, difference, interpolation)
    dogleg = _tree_cast_like(dogleg, newton_direction)
    use_newton = newton_norm <= radius
    use_scaled_cauchy = (~use_newton) & (cauchy_norm >= radius)
    step = tree_where(
        use_newton,
        newton_direction,
        tree_where(use_scaled_cauchy, scaled_cauchy, dogleg),
    )
    step = _tree_cast_like(step, newton_direction)
    return step, ~use_newton


def _root_trust_region(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    residual: PyTree[Any],
    auxiliary: Any,
    newton_direction: PyTree[Any],
    cauchy_direction: PyTree[Any],
    jacobian,
    radius: Array,
    maximum_evaluations: Array,
    args: Any,
    policy: RootTrustRegion,
    precision: NonlinearPrecisionPolicy,
    /,
) -> _TrustResult:
    merit = 0.5 * jnp.real(
        _space_inner(
            _bound_space(problem.residual_space, "residual"),
            residual,
            residual,
            precision,
        )
    )
    scalar_dtype = merit.dtype
    zero_step = jax.tree.map(jnp.zeros_like, newton_direction)
    initial_carry = (
        state,
        residual,
        auxiliary,
        zero_step,
        radius,
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def body(_, carry):
        (
            accepted_state,
            accepted_residual,
            accepted_auxiliary,
            accepted_step,
            current_radius,
            accepted,
            finite_seen,
            evaluations,
            domain_failures,
            nonfinite_trials,
        ) = carry

        def attempt(__):
            step, boundary = _dogleg_step(
                problem.state_space,
                newton_direction,
                cauchy_direction,
                current_radius,
                precision,
            )
            model_residual = jax.tree.map(
                lambda value, change: value + change,
                residual,
                jacobian.mv(step),
            )
            predicted = merit - 0.5 * jnp.real(
                _space_inner(
                    _bound_space(problem.residual_space, "residual"),
                    model_residual,
                    model_residual,
                    precision,
                )
            )
            candidate = _tree_cast_like(
                tree_add_scaled(state, step, 1.0),
                state,
            )
            candidate_residual, candidate_auxiliary = problem.evaluate(candidate, args)
            finite = tree_allfinite(candidate) & tree_allfinite(candidate_residual)
            valid = problem.valid(
                candidate, candidate_residual, candidate_auxiliary, args
            )
            actual = merit - 0.5 * jnp.real(
                _space_inner(
                    _bound_space(problem.residual_space, "residual"),
                    candidate_residual,
                    candidate_residual,
                    precision,
                )
            )
            ratio = actual / jnp.maximum(
                predicted, jnp.asarray(1e-30, dtype=scalar_dtype)
            )
            usable_model = (
                jnp.isfinite(predicted) & (predicted > 0.0) & jnp.isfinite(ratio)
            )
            use = finite & valid & usable_model & (ratio >= policy.acceptance_ratio)
            expanded = jnp.minimum(policy.maximum_radius, policy.growth * current_radius)
            contracted = jnp.maximum(
                policy.minimum_radius, policy.shrink * current_radius
            )
            next_radius = jnp.where(
                use & (ratio >= policy.expansion_ratio) & boundary,
                expanded,
                jnp.where(
                    (~use) | (ratio < policy.shrink_ratio),
                    contracted,
                    current_radius,
                ),
            )
            return (
                tree_where(use, candidate, accepted_state),
                tree_where(use, candidate_residual, accepted_residual),
                tree_where(use, candidate_auxiliary, accepted_auxiliary),
                tree_where(use, step, accepted_step),
                next_radius,
                use,
                finite_seen | (finite & valid),
                evaluations + 1,
                domain_failures + (finite & ~valid).astype(jnp.int32),
                nonfinite_trials + (~finite).astype(jnp.int32),
            )

        can_attempt = (~accepted) & (evaluations < maximum_evaluations)
        return jax.lax.cond(can_attempt, attempt, lambda _: carry, operand=None)

    (
        accepted_state,
        accepted_residual,
        accepted_auxiliary,
        accepted_step,
        final_radius,
        accepted,
        finite_seen,
        evaluations,
        domain_failures,
        nonfinite_trials,
    ) = jax.lax.fori_loop(0, policy.maximum_attempts, body, initial_carry)
    return _TrustResult(
        state=accepted_state,
        residual=accepted_residual,
        auxiliary=accepted_auxiliary,
        step=accepted_step,
        radius=final_radius,
        evaluations=evaluations,
        rejections=evaluations - accepted.astype(jnp.int32),
        accepted=accepted,
        finite_seen=finite_seen,
        domain_failures=domain_failures,
        nonfinite_trials=nonfinite_trials,
    )


def _initial_root_state(
    problem: NonlinearSystemProblem,
    initial_state: PyTree[Any],
    jacobian_policy: JacobianPolicy,
    linear_policy: LinearSolvePolicy,
    forcing_policy: NewtonForcingPolicy,
    trust_radius: float,
    args: Any,
    precision: NonlinearPrecisionPolicy,
    /,
) -> tuple[NonlinearSystemProblem, PyTree[Array], _RootState, Any]:
    state = problem.validate_state(initial_state)
    prepared_jacobian = prepare_jacobian(problem, state, jacobian_policy, args)
    if prepared_jacobian.operator.source.size != prepared_jacobian.operator.target.size:
        raise ValueError("Newton methods require a square Jacobian coordinate map.")
    residual = prepared_jacobian.residual
    problem = problem.bind_spaces(state, residual)
    state_space = _bound_space(problem.state_space, "state")
    residual_space = _bound_space(problem.residual_space, "residual")
    precision.validate_trees(state, residual)
    precision.validate_accumulation_space(state_space)
    precision.validate_accumulation_space(residual_space)
    residual_norm = _space_norm(residual_space, residual, precision)
    linear_operator = _jacobian_solve_operator(prepared_jacobian.operator)
    prepared_linear, refresh_state = prepare_refresh_state(
        LinearSystem(linear_operator),
        _iteration_linear_policy(linear_policy),
        setup_operator=problem.linear_setup(state, args),
    )
    recycling = (
        None
        if prepared_linear.plan.policy.recycling is None
        else initialize_recycling(prepared_linear)
    )
    valid = problem.valid(state, residual, prepared_jacobian.auxiliary, args)
    finite = tree_allfinite(state) & tree_allfinite(residual)
    status = jnp.where(
        finite & valid,
        int(NonlinearStatus.ITERATING),
        jnp.where(
            finite,
            int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
            jnp.where(
                tree_allfinite(state),
                int(NonlinearStatus.NONFINITE_EVALUATION),
                int(NonlinearStatus.NONFINITE_INPUT),
            ),
        ),
    ).astype(jnp.int32)
    forcing = jnp.asarray(
        (
            linear_policy.tolerance.relative
            if forcing_policy.strategy == "constant"
            else forcing_policy.initial
        ),
        dtype=residual_norm.dtype,
    )
    run = _RootState(
        residual=residual,
        auxiliary=prepared_jacobian.auxiliary,
        initial_residual_norm=residual_norm,
        residual_norm=residual_norm,
        step_norm=jnp.asarray(0.0, dtype=residual_norm.dtype),
        iteration=jnp.asarray(0, dtype=jnp.int32),
        residual_evaluations=jnp.asarray(
            prepared_jacobian.residual_evaluations, dtype=jnp.int32
        ),
        jvp_evaluations=jnp.asarray(0, dtype=jnp.int32),
        vjp_evaluations=jnp.asarray(0, dtype=jnp.int32),
        jacobian_preparations=jnp.asarray(1, dtype=jnp.int32),
        linear_solves=jnp.asarray(0, dtype=jnp.int32),
        linear_iterations=jnp.asarray(0, dtype=jnp.int32),
        accepted_steps=jnp.asarray(0, dtype=jnp.int32),
        rejected_steps=jnp.asarray(0, dtype=jnp.int32),
        globalization_rejections=jnp.asarray(0, dtype=jnp.int32),
        domain_failures=(finite & ~valid).astype(jnp.int32),
        nonfinite_trials=jnp.asarray(0, dtype=jnp.int32),
        setup_refreshes=jnp.asarray(1, dtype=jnp.int32),
        numeric_refreshes=jnp.asarray(1, dtype=jnp.int32),
        trust_radius=jnp.asarray(trust_radius, dtype=residual_norm.dtype),
        forcing=forcing,
        last_forcing=jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        jacobian_age=jnp.asarray(0, dtype=jnp.int32),
        jacobian_reference_residual_norm=residual_norm,
        jacobian_reference_rejected_steps=jnp.asarray(0, dtype=jnp.int32),
        status=status,
        refresh_state=refresh_state,
        recycling=recycling,
        final_linear_status=jnp.asarray(-1, dtype=jnp.int32),
        final_linear_rank=jnp.asarray(-1, dtype=jnp.int32),
        final_linear_condition_estimate=jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        final_linear_residual_norm=jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        final_linear_converged=jnp.asarray(False),
    )
    return problem, state, run, prepared_jacobian


def _root_attempt_handoff(
    method: NewtonKrylov | NewtonTrustRegion,
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    residual: PyTree[Any],
    auxiliary: Any,
    previous_run: _RootState,
    prepared_jacobian: Any,
    args: Any,
    /,
):
    """Start a Newton attempt from retained physical and derivative evidence."""
    if not isinstance(method, (NewtonKrylov, NewtonTrustRegion)):
        raise TypeError("method must be NewtonKrylov or NewtonTrustRegion.")
    state_ = problem.validate_state(state)
    residual_ = problem.validate_residual(residual)
    problem_ = problem.bind_spaces(state_, residual_)
    residual_space = _bound_space(problem_.residual_space, "residual")
    residual_norm = jnp.sqrt(
        jnp.maximum(
            jnp.real(residual_space.inner(residual_, residual_)),
            0.0,
        )
    )
    finite = tree_allfinite(state_) & tree_allfinite(residual_)
    valid = problem_.valid(state_, residual_, auxiliary, args)
    status = jnp.where(
        finite & valid,
        int(NonlinearStatus.ITERATING),
        jnp.where(
            finite,
            int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
            jnp.where(
                tree_allfinite(state_),
                int(NonlinearStatus.NONFINITE_EVALUATION),
                int(NonlinearStatus.NONFINITE_INPUT),
            ),
        ),
    ).astype(jnp.int32)
    forcing = jnp.asarray(
        (
            method.linear_policy.tolerance.relative
            if method.forcing_policy.strategy == "constant"
            else method.forcing_policy.initial
        ),
        dtype=residual_norm.dtype,
    )
    trust_radius = (
        method.trust_region.initial_radius
        if isinstance(method, NewtonTrustRegion)
        else jnp.nan
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    run = _RootState(
        residual=residual_,
        auxiliary=auxiliary,
        initial_residual_norm=residual_norm,
        residual_norm=residual_norm,
        step_norm=jnp.asarray(0.0, dtype=residual_norm.dtype),
        iteration=zero,
        residual_evaluations=zero,
        jvp_evaluations=zero,
        vjp_evaluations=zero,
        jacobian_preparations=zero,
        linear_solves=zero,
        linear_iterations=zero,
        accepted_steps=zero,
        rejected_steps=zero,
        globalization_rejections=zero,
        domain_failures=(finite & ~valid).astype(jnp.int32),
        nonfinite_trials=zero,
        setup_refreshes=zero,
        numeric_refreshes=zero,
        forcing=forcing,
        last_forcing=jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        jacobian_age=zero,
        jacobian_reference_residual_norm=residual_norm,
        jacobian_reference_rejected_steps=zero,
        trust_radius=jnp.asarray(trust_radius, dtype=residual_norm.dtype),
        status=status,
        refresh_state=previous_run.refresh_state,
        recycling=previous_run.recycling,
        final_linear_status=jnp.asarray(-1, dtype=jnp.int32),
        final_linear_rank=jnp.asarray(-1, dtype=jnp.int32),
        final_linear_condition_estimate=jnp.asarray(
            jnp.nan,
            dtype=residual_norm.dtype,
        ),
        final_linear_residual_norm=jnp.asarray(
            jnp.nan,
            dtype=residual_norm.dtype,
        ),
        final_linear_converged=jnp.asarray(False),
    )
    jacobian = eqx.tree_at(
        lambda value: value.residual,
        prepared_jacobian,
        residual_,
    )
    jacobian = eqx.tree_at(
        lambda value: value.auxiliary,
        jacobian,
        auxiliary,
        is_leaf=lambda value: value is None,
    )
    return problem_, state_, run, jacobian


def _maybe_refresh_jacobian(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    current: Any,
    run: _RootState,
    jacobian_policy: JacobianPolicy,
    refresh_policy: JacobianRefreshPolicy,
    termination: NonlinearTermination,
    args: Any,
    /,
) -> tuple[Any, Array]:
    requested = refresh_policy.should_refresh(
        run.jacobian_age,
        run.residual_norm,
        run.jacobian_reference_residual_norm,
        run.globalization_rejections,
        run.jacobian_reference_rejected_steps,
    )
    remaining = _remaining_evaluations(termination, run.residual_evaluations)
    refresh = requested & (remaining > 1)
    current_dynamic, current_static = eqx.partition(current, eqx.is_array)

    def prepare_dynamic(_):
        prepared = prepare_jacobian(problem, state, jacobian_policy, args)
        return eqx.partition(prepared, eqx.is_array)[0]

    selected_dynamic = jax.lax.cond(
        refresh, prepare_dynamic, lambda _: current_dynamic, operand=None
    )
    return eqx.combine(selected_dynamic, current_static), refresh


def _solve_newton_linear(
    prepared: Any,
    right_hand_side: PyTree[Any],
    run: _RootState,
    termination: NonlinearTermination,
    /,
):
    control = _linear_control(prepared, run, termination)
    if run.recycling is None:
        return (
            solve_linear(prepared, right_hand_side, control=control),
            None,
        )
    recycled = solve_recycled(
        prepared,
        right_hand_side,
        recycling=run.recycling,
        control=control,
    )
    return recycled.result, recycled.recycling


def _linear_control(
    prepared: Any,
    run: _RootState,
    termination: NonlinearTermination,
    /,
) -> LinearSolveControl | None:
    if prepared.plan.backend != "native-krylov":
        return None
    structural_limit = (
        prepared.plan.policy.tolerance.max_steps or prepared.problem.operator.source.size
    )
    return LinearSolveControl(
        relative_tolerance=run.forcing,
        absolute_tolerance=prepared.plan.policy.tolerance.absolute,
        maximum_steps=_remaining_linear_steps(
            termination,
            run.linear_iterations,
            structural_limit,
        ),
    )


def _condition(termination: NonlinearTermination):
    def condition(carry):
        state = carry[1]
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else state.residual_evaluations < termination.maximum_evaluations
        )
        within_linear = (
            jnp.asarray(True)
            if termination.maximum_linear_iterations is None
            else state.linear_iterations < termination.maximum_linear_iterations
        )
        return (
            (state.status == int(NonlinearStatus.ITERATING))
            & (state.iteration < termination.maximum_steps)
            & within_evaluations
            & within_linear
        )

    return condition


def _terminal_status(state: _RootState, termination: NonlinearTermination, /) -> Array:
    status = state.status
    converged = (status == int(NonlinearStatus.ITERATING)) & (
        state.residual_norm <= termination.residual_threshold(state.initial_residual_norm)
    )
    status = jnp.where(converged, int(NonlinearStatus.SUCCESS), status)
    exhausted_evaluations = (
        jnp.asarray(False)
        if termination.maximum_evaluations is None
        else state.residual_evaluations >= termination.maximum_evaluations
    )
    exhausted_linear = (
        jnp.asarray(False)
        if termination.maximum_linear_iterations is None
        else state.linear_iterations >= termination.maximum_linear_iterations
    )
    status = jnp.where(
        (status == int(NonlinearStatus.ITERATING)) & exhausted_evaluations,
        int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
        status,
    )
    status = jnp.where(
        (status == int(NonlinearStatus.ITERATING)) & exhausted_linear,
        int(NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED),
        status,
    )
    return jnp.where(
        status == int(NonlinearStatus.ITERATING),
        int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
        status,
    ).astype(jnp.int32)


def _eager_initial_status(
    state: _RootState,
    termination: NonlinearTermination,
    /,
) -> Array | None:
    if any(
        isinstance(value, jax_core.Tracer)
        for value in (
            state.status,
            state.residual_norm,
            state.initial_residual_norm,
        )
    ):
        return None
    if int(state.status) != int(NonlinearStatus.ITERATING):
        return state.status
    converged = state.residual_norm <= termination.residual_threshold(
        state.initial_residual_norm
    )
    if isinstance(converged, jax_core.Tracer):
        return None
    if bool(converged):
        return jnp.asarray(int(NonlinearStatus.SUCCESS), dtype=jnp.int32)
    return None


def _package_result(
    method: AbstractNonlinearMethod,
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    run: _RootState,
    status: Array,
    termination: NonlinearTermination,
    jacobian_policy: JacobianPolicy,
    refresh_policy: JacobianRefreshPolicy,
    globalization_id: str,
    args: Any,
    precision: NonlinearPrecisionPolicy,
    /,
) -> NonlinearResult:
    residual = run.residual
    auxiliary = run.auxiliary
    residual_space = _bound_space(problem.residual_space, "residual")
    residual_norm = _space_norm(residual_space, residual, precision)
    finite = tree_allfinite(state) & tree_allfinite(residual)
    valid = problem.valid(state, residual, auxiliary, args)
    certified = (
        finite
        & valid
        & (residual_norm <= termination.residual_threshold(run.initial_residual_norm))
    )
    preserve_input_failure = status == int(NonlinearStatus.NONFINITE_INPUT)
    status = jnp.where(
        preserve_input_failure,
        status,
        jnp.where(
            ~finite,
            int(NonlinearStatus.NONFINITE_EVALUATION),
            jnp.where(
                ~valid,
                int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
                jnp.where(
                    (status == int(NonlinearStatus.SUCCESS)) & ~certified,
                    int(NonlinearStatus.RESIDUAL_STAGNATION),
                    status,
                ),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = NonlinearDiagnostics(
        initial_residual_norm=run.initial_residual_norm,
        final_residual_norm=residual_norm,
        final_step_norm=run.step_norm,
        iterations=run.iteration,
        residual_evaluations=run.residual_evaluations,
        jvp_evaluations=run.jvp_evaluations,
        vjp_evaluations=run.vjp_evaluations,
        jacobian_preparations=run.jacobian_preparations,
        linear_solves=run.linear_solves,
        linear_iterations=run.linear_iterations,
        accepted_steps=run.accepted_steps,
        rejected_steps=run.rejected_steps,
        domain_failures=run.domain_failures
        + ((run.domain_failures == 0) & finite & ~valid).astype(jnp.int32),
        nonfinite_trials=run.nonfinite_trials,
        setup_refreshes=run.setup_refreshes,
        numeric_refreshes=run.numeric_refreshes,
        final_forcing=run.last_forcing,
        final_trust_radius=run.trust_radius,
        final_linear_status=run.final_linear_status,
        final_linear_rank=run.final_linear_rank,
        final_linear_condition_estimate=run.final_linear_condition_estimate,
        final_linear_residual_norm=run.final_linear_residual_norm,
        final_linear_converged=run.final_linear_converged,
    )
    return NonlinearResult(
        state=jax.tree.map(precision.output, state),
        residual=residual,
        auxiliary=auxiliary,
        status=status,
        diagnostics=diagnostics,
        provenance=NonlinearProvenance(
            problem_id=problem.problem_id,
            method_id=method.method_id,
            derivative_id=jacobian_policy.policy_id,
            globalization_id=globalization_id,
            linear_plan_id=run.refresh_state.template.plan.plan_id,
            precision_policy_id=precision.policy_id,
            notes=(
                f"linear-method={run.refresh_state.template.plan.method};"
                f"linear-backend={run.refresh_state.template.plan.backend}"
            ),
        ),
        precision_evidence=precision.evidence_for(state, residual),
    )


class NewtonKrylov(AbstractNonlinearMethod):
    """Matrix-free inexact Newton root solve with residual-merit line search."""

    jacobian_policy: JacobianPolicy
    linear_policy: LinearSolvePolicy
    forcing_policy: NewtonForcingPolicy
    jacobian_refresh: JacobianRefreshPolicy
    line_search: RootLineSearch

    def __init__(
        self,
        *,
        jacobian_policy: JacobianPolicy | None = None,
        linear_policy: LinearSolvePolicy | None = None,
        forcing_policy: NewtonForcingPolicy | None = None,
        jacobian_refresh: JacobianRefreshPolicy | None = None,
        line_search: RootLineSearch | None = None,
    ):
        self.jacobian_policy = (
            JacobianPolicy() if jacobian_policy is None else jacobian_policy
        )
        self.linear_policy = (
            _default_linear_policy() if linear_policy is None else linear_policy
        )
        self.forcing_policy = (
            NewtonForcingPolicy() if forcing_policy is None else forcing_policy
        )
        self.jacobian_refresh = (
            JacobianRefreshPolicy() if jacobian_refresh is None else jacobian_refresh
        )
        self.line_search = RootLineSearch() if line_search is None else line_search
        if not isinstance(self.jacobian_policy, JacobianPolicy):
            raise TypeError("jacobian_policy must be JacobianPolicy or None.")
        if not isinstance(self.linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        if not isinstance(self.forcing_policy, NewtonForcingPolicy):
            raise TypeError("forcing_policy must be NewtonForcingPolicy or None.")
        if not isinstance(self.jacobian_refresh, JacobianRefreshPolicy):
            raise TypeError("jacobian_refresh must be JacobianRefreshPolicy or None.")
        if not isinstance(self.line_search, RootLineSearch):
            raise TypeError("line_search must be RootLineSearch or None.")

    @property
    def method_id(self) -> str:
        return "newton-krylov-line-search"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=True,
            prepared_refresh=True,
            jit=self.jacobian_policy.mode != "explicit",
            implicit_differentiation=True,
        )

    def solve(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
        precision: NonlinearPrecisionPolicy | None = None,
        _prepared_start: tuple[NonlinearSystemProblem, PyTree[Array], _RootState, Any]
        | None = None,
        _return_internal: bool = False,
    ) -> Any:
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be a NonlinearSystemProblem.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination.")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be a NonlinearPrecisionPolicy or None.")
        if _prepared_start is None:
            problem, state, run, prepared_jacobian = _initial_root_state(
                problem,
                initial_state,
                self.jacobian_policy,
                precision_.bind_linear(self.linear_policy),
                self.forcing_policy,
                jnp.nan,
                args,
                precision_,
            )
        else:
            problem, state, run, prepared_jacobian = _prepared_start
        initial_status = _eager_initial_status(run, termination)
        if initial_status is not None:
            result = _package_result(
                self,
                problem,
                state,
                run,
                initial_status,
                termination,
                self.jacobian_policy,
                self.jacobian_refresh,
                "residual-armijo",
                args,
                precision_,
            )
            return (result, state, run, prepared_jacobian) if _return_internal else result
        dynamic_run, static_run = eqx.partition(run, eqx.is_array)
        dynamic_jacobian, static_jacobian = eqx.partition(prepared_jacobian, eqx.is_array)

        def body(carry):
            current, dynamic, dynamic_derivative = carry
            current_run = eqx.combine(dynamic, static_run)
            carried_jacobian = eqx.combine(dynamic_derivative, static_jacobian)
            residual = current_run.residual
            auxiliary = current_run.auxiliary
            residual_norm = current_run.residual_norm
            current_valid = problem.valid(current, residual, auxiliary, args)
            current_finite = tree_allfinite(current) & tree_allfinite(residual)
            converged = current_valid & (
                residual_norm
                <= termination.residual_threshold(current_run.initial_residual_norm)
            )
            terminal_now = ~current_valid | converged
            terminal_status = jnp.where(
                current_valid,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    current_finite,
                    int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
                    int(NonlinearStatus.NONFINITE_EVALUATION),
                ),
            ).astype(jnp.int32)

            def terminal(_):
                updated = eqx.tree_at(
                    lambda item: item.status,
                    current_run,
                    terminal_status,
                )
                return (
                    current,
                    eqx.partition(updated, eqx.is_array)[0],
                    dynamic_derivative,
                )

            def step(_):
                selected_jacobian, refreshed = _maybe_refresh_jacobian(
                    problem,
                    current,
                    carried_jacobian,
                    current_run,
                    self.jacobian_policy,
                    self.jacobian_refresh,
                    termination,
                    args,
                )
                current_residual = tree_where(
                    refreshed, selected_jacobian.residual, residual
                )
                current_auxiliary = tree_where(
                    refreshed, selected_jacobian.auxiliary, auxiliary
                )
                current_residual_norm = _space_norm(
                    problem.residual_space,
                    current_residual,
                    precision_,
                )
                refresh_evaluations = refreshed.astype(jnp.int32) * (
                    selected_jacobian.residual_evaluations
                )
                linear_operator = _jacobian_solve_operator(selected_jacobian.operator)
                prepared, refresh_state = current_run.refresh_state.refresh(
                    LinearSystem(linear_operator),
                    setup_operator=problem.linear_setup(current, args),
                )
                right_hand_side = _jacobian_solve_right_hand_side(
                    linear_operator, current_residual
                )
                linear_result, recycling = _solve_newton_linear(
                    prepared,
                    right_hand_side,
                    current_run,
                    termination,
                )
                direction = _jacobian_solve_direction(
                    linear_operator, linear_result.value
                )
                image = selected_jacobian.operator.mv(direction)
                directional = jnp.real(
                    _space_inner(
                        _bound_space(problem.residual_space, "residual"),
                        current_residual,
                        image,
                        precision_,
                    )
                )
                usable = (
                    _usable_linear_status(linear_result.status)
                    & tree_allfinite(direction)
                    & jnp.isfinite(directional)
                    & (directional < 0.0)
                )
                used_before_search = (
                    current_run.residual_evaluations + refresh_evaluations
                )
                search_budget = _remaining_evaluations(termination, used_before_search)

                def search_direction(__):
                    return _root_line_search(
                        problem,
                        current,
                        current_residual,
                        current_auxiliary,
                        direction,
                        directional,
                        search_budget,
                        args,
                        self.line_search,
                        precision_,
                    )

                def failed_direction(__):
                    return _SearchResult(
                        state=current,
                        residual=current_residual,
                        auxiliary=current_auxiliary,
                        rate=jnp.asarray(0.0, dtype=directional.dtype),
                        evaluations=jnp.asarray(0, dtype=jnp.int32),
                        rejections=jnp.asarray(0, dtype=jnp.int32),
                        accepted=jnp.asarray(False),
                        finite_seen=jnp.asarray(False),
                        domain_failures=jnp.asarray(0, dtype=jnp.int32),
                        nonfinite_trials=jnp.asarray(0, dtype=jnp.int32),
                    )

                search = jax.lax.cond(
                    usable & (search_budget > 0),
                    search_direction,
                    failed_direction,
                    operand=None,
                )
                step_norm = search.rate * _space_norm(
                    problem.state_space,
                    direction,
                    precision_,
                )
                candidate_norm = _space_norm(
                    problem.residual_space,
                    search.residual,
                    precision_,
                )
                stagnated = search.accepted & (
                    step_norm
                    <= termination.step_threshold(
                        _space_norm(problem.state_space, current, precision_)
                    )
                )
                diverged = candidate_norm > (
                    termination.divergence_factor
                    * jnp.maximum(current_run.initial_residual_norm, 1e-30)
                )
                candidate_converged = search.accepted & (
                    candidate_norm
                    <= termination.residual_threshold(current_run.initial_residual_norm)
                )
                linear_iterations = jnp.sum(
                    linear_result.diagnostics.iterations, dtype=jnp.int32
                )
                next_residual_evaluations = used_before_search + search.evaluations
                next_linear_iterations = current_run.linear_iterations + linear_iterations
                evaluations_exhausted = (
                    jnp.asarray(False)
                    if termination.maximum_evaluations is None
                    else next_residual_evaluations >= termination.maximum_evaluations
                )
                linear_exhausted = (
                    jnp.asarray(False)
                    if termination.maximum_linear_iterations is None
                    else next_linear_iterations >= termination.maximum_linear_iterations
                )
                retry_rejected = (self.jacobian_refresh.strategy == "rejection") & (
                    search.rejections > 0
                )
                failed_status = jnp.where(
                    linear_exhausted,
                    int(NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED),
                    _linear_failure_status(linear_result.status),
                )
                status = jnp.where(
                    ~usable,
                    failed_status,
                    jnp.where(
                        candidate_converged,
                        int(NonlinearStatus.SUCCESS),
                        jnp.where(
                            stagnated,
                            int(NonlinearStatus.RESIDUAL_STAGNATION),
                            jnp.where(
                                diverged,
                                int(NonlinearStatus.DIVERGENCE),
                                jnp.where(
                                    evaluations_exhausted,
                                    int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
                                    jnp.where(
                                        linear_exhausted,
                                        int(
                                            NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED
                                        ),
                                        jnp.where(
                                            search.accepted | retry_rejected,
                                            int(NonlinearStatus.ITERATING),
                                            jnp.where(
                                                search.finite_seen,
                                                int(NonlinearStatus.LINE_SEARCH_FAILED),
                                                jnp.where(
                                                    search.domain_failures > 0,
                                                    int(
                                                        NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE
                                                    ),
                                                    int(
                                                        NonlinearStatus.NONFINITE_EVALUATION
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ).astype(jnp.int32)
                refreshed_age = jnp.where(
                    refreshed, jnp.asarray(0, dtype=jnp.int32), current_run.jacobian_age
                )
                reference_norm = jnp.where(
                    refreshed,
                    current_residual_norm,
                    current_run.jacobian_reference_residual_norm,
                )
                reference_rejections = jnp.where(
                    refreshed,
                    current_run.globalization_rejections,
                    current_run.jacobian_reference_rejected_steps,
                )
                next_forcing = jnp.where(
                    search.accepted,
                    self.forcing_policy.next(
                        current_run.forcing,
                        current_residual_norm,
                        candidate_norm,
                    ),
                    current_run.forcing,
                )
                updated = _RootState(
                    residual=search.residual,
                    auxiliary=search.auxiliary,
                    initial_residual_norm=current_run.initial_residual_norm,
                    residual_norm=candidate_norm,
                    step_norm=step_norm,
                    iteration=current_run.iteration + 1,
                    residual_evaluations=next_residual_evaluations,
                    jvp_evaluations=(
                        current_run.jvp_evaluations
                        + jnp.sum(linear_result.diagnostics.matvec_count, dtype=jnp.int32)
                        + 1
                    ),
                    vjp_evaluations=current_run.vjp_evaluations,
                    jacobian_preparations=current_run.jacobian_preparations
                    + refreshed.astype(jnp.int32),
                    linear_solves=current_run.linear_solves + 1,
                    linear_iterations=next_linear_iterations,
                    accepted_steps=current_run.accepted_steps
                    + search.accepted.astype(jnp.int32),
                    rejected_steps=current_run.rejected_steps
                    + (~search.accepted).astype(jnp.int32),
                    globalization_rejections=current_run.globalization_rejections
                    + search.rejections,
                    domain_failures=current_run.domain_failures + search.domain_failures,
                    nonfinite_trials=current_run.nonfinite_trials
                    + search.nonfinite_trials,
                    setup_refreshes=current_run.setup_refreshes,
                    numeric_refreshes=current_run.numeric_refreshes + 1,
                    forcing=next_forcing,
                    last_forcing=current_run.forcing,
                    jacobian_age=refreshed_age + search.accepted.astype(jnp.int32),
                    jacobian_reference_residual_norm=reference_norm,
                    jacobian_reference_rejected_steps=reference_rejections,
                    trust_radius=current_run.trust_radius,
                    status=status,
                    refresh_state=refresh_state,
                    recycling=recycling,
                    final_linear_status=linear_result.status,
                    final_linear_rank=linear_result.diagnostics.rank,
                    final_linear_condition_estimate=jnp.asarray(
                        linear_result.diagnostics.condition_estimate,
                        dtype=current_run.final_linear_condition_estimate.dtype,
                    ),
                    final_linear_residual_norm=jnp.asarray(
                        linear_result.diagnostics.residual_norm,
                        dtype=current_run.final_linear_residual_norm.dtype,
                    ),
                    final_linear_converged=linear_result.diagnostics.converged,
                )
                return (
                    search.state,
                    eqx.partition(updated, eqx.is_array)[0],
                    eqx.partition(selected_jacobian, eqx.is_array)[0],
                )

            return jax.lax.cond(terminal_now, terminal, step, operand=None)

        state, dynamic_run, dynamic_jacobian = jax.lax.while_loop(
            _condition(termination),
            body,
            (state, dynamic_run, dynamic_jacobian),
        )
        run = eqx.combine(dynamic_run, static_run)
        jacobian = eqx.combine(dynamic_jacobian, static_jacobian)
        status = _terminal_status(run, termination)
        result = _package_result(
            self,
            problem,
            state,
            run,
            status,
            termination,
            self.jacobian_policy,
            self.jacobian_refresh,
            "residual-armijo",
            args,
            precision_,
        )
        return (result, state, run, jacobian) if _return_internal else result


class NewtonTrustRegion(AbstractNonlinearMethod):
    """Inexact Newton solve with a residual-model dogleg trust region."""

    jacobian_policy: JacobianPolicy
    linear_policy: LinearSolvePolicy
    forcing_policy: NewtonForcingPolicy
    jacobian_refresh: JacobianRefreshPolicy
    trust_region: RootTrustRegion

    def __init__(
        self,
        *,
        jacobian_policy: JacobianPolicy | None = None,
        linear_policy: LinearSolvePolicy | None = None,
        forcing_policy: NewtonForcingPolicy | None = None,
        jacobian_refresh: JacobianRefreshPolicy | None = None,
        trust_region: RootTrustRegion | None = None,
    ):
        self.jacobian_policy = (
            JacobianPolicy() if jacobian_policy is None else jacobian_policy
        )
        self.linear_policy = (
            _default_linear_policy() if linear_policy is None else linear_policy
        )
        self.forcing_policy = (
            NewtonForcingPolicy() if forcing_policy is None else forcing_policy
        )
        self.jacobian_refresh = (
            JacobianRefreshPolicy() if jacobian_refresh is None else jacobian_refresh
        )
        self.trust_region = RootTrustRegion() if trust_region is None else trust_region
        if not isinstance(self.jacobian_policy, JacobianPolicy):
            raise TypeError("jacobian_policy must be JacobianPolicy or None.")
        if not isinstance(self.linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        if not isinstance(self.forcing_policy, NewtonForcingPolicy):
            raise TypeError("forcing_policy must be NewtonForcingPolicy or None.")
        if not isinstance(self.jacobian_refresh, JacobianRefreshPolicy):
            raise TypeError("jacobian_refresh must be JacobianRefreshPolicy or None.")
        if not isinstance(self.trust_region, RootTrustRegion):
            raise TypeError("trust_region must be RootTrustRegion or None.")

    @property
    def method_id(self) -> str:
        return "newton-trust-region"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=True,
            prepared_refresh=True,
            jit=self.jacobian_policy.mode != "explicit",
            implicit_differentiation=True,
        )

    def solve(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
        precision: NonlinearPrecisionPolicy | None = None,
        _prepared_start: tuple[NonlinearSystemProblem, PyTree[Array], _RootState, Any]
        | None = None,
        _return_internal: bool = False,
    ) -> Any:
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be a NonlinearSystemProblem.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination.")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be a NonlinearPrecisionPolicy or None.")
        if _prepared_start is None:
            problem, state, run, prepared_jacobian = _initial_root_state(
                problem,
                initial_state,
                self.jacobian_policy,
                precision_.bind_linear(self.linear_policy),
                self.forcing_policy,
                self.trust_region.initial_radius,
                args,
                precision_,
            )
        else:
            problem, state, run, prepared_jacobian = _prepared_start
        initial_status = _eager_initial_status(run, termination)
        if initial_status is not None:
            result = _package_result(
                self,
                problem,
                state,
                run,
                initial_status,
                termination,
                self.jacobian_policy,
                self.jacobian_refresh,
                "dogleg-residual-trust-region",
                args,
                precision_,
            )
            return (result, state, run, prepared_jacobian) if _return_internal else result
        dynamic_run, static_run = eqx.partition(run, eqx.is_array)
        dynamic_jacobian, static_jacobian = eqx.partition(prepared_jacobian, eqx.is_array)

        def body(carry):
            current, dynamic, dynamic_derivative = carry
            current_run = eqx.combine(dynamic, static_run)
            carried_jacobian = eqx.combine(dynamic_derivative, static_jacobian)
            residual = current_run.residual
            auxiliary = current_run.auxiliary
            residual_norm = current_run.residual_norm
            current_valid = problem.valid(current, residual, auxiliary, args)
            current_finite = tree_allfinite(current) & tree_allfinite(residual)
            converged = current_valid & (
                residual_norm
                <= termination.residual_threshold(current_run.initial_residual_norm)
            )
            terminal_now = ~current_valid | converged
            terminal_status = jnp.where(
                current_valid,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    current_finite,
                    int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
                    int(NonlinearStatus.NONFINITE_EVALUATION),
                ),
            ).astype(jnp.int32)

            def terminal(_):
                updated = eqx.tree_at(
                    lambda item: item.status,
                    current_run,
                    terminal_status,
                )
                return (
                    current,
                    eqx.partition(updated, eqx.is_array)[0],
                    dynamic_derivative,
                )

            def step(_):
                selected_jacobian, refreshed = _maybe_refresh_jacobian(
                    problem,
                    current,
                    carried_jacobian,
                    current_run,
                    self.jacobian_policy,
                    self.jacobian_refresh,
                    termination,
                    args,
                )
                current_residual = tree_where(
                    refreshed, selected_jacobian.residual, residual
                )
                current_auxiliary = tree_where(
                    refreshed, selected_jacobian.auxiliary, auxiliary
                )
                current_residual_norm = _space_norm(
                    problem.residual_space,
                    current_residual,
                    precision_,
                )
                refresh_evaluations = refreshed.astype(jnp.int32) * (
                    selected_jacobian.residual_evaluations
                )
                linear_operator = _jacobian_solve_operator(selected_jacobian.operator)
                prepared, refresh_state = current_run.refresh_state.refresh(
                    LinearSystem(linear_operator),
                    setup_operator=problem.linear_setup(current, args),
                )
                right_hand_side = _jacobian_solve_right_hand_side(
                    linear_operator, current_residual
                )
                linear_result, recycling = _solve_newton_linear(
                    prepared,
                    right_hand_side,
                    current_run,
                    termination,
                )
                raw_newton = _jacobian_solve_direction(
                    linear_operator, linear_result.value
                )
                gradient = selected_jacobian.operator.adjoint_mv(current_residual)
                gradient_image = selected_jacobian.operator.mv(gradient)
                gradient_scale = jnp.real(
                    _space_inner(
                        _bound_space(problem.state_space, "state"),
                        gradient,
                        gradient,
                        precision_,
                    )
                ) / jnp.maximum(
                    jnp.real(
                        _space_inner(
                            _bound_space(problem.residual_space, "residual"),
                            gradient_image,
                            gradient_image,
                            precision_,
                        )
                    ),
                    1e-30,
                )
                cauchy_direction = _tree_cast_like(
                    tree_scale(-gradient_scale, gradient),
                    gradient,
                )
                usable_newton = _usable_linear_status(
                    linear_result.status
                ) & tree_allfinite(raw_newton)
                newton_direction = tree_where(usable_newton, raw_newton, cauchy_direction)
                usable = tree_allfinite(cauchy_direction) & (
                    _space_norm(
                        problem.state_space,
                        cauchy_direction,
                        precision_,
                    )
                    > 0.0
                )
                used_before_search = (
                    current_run.residual_evaluations + refresh_evaluations
                )
                search_budget = _remaining_evaluations(termination, used_before_search)

                def search_direction(__):
                    return _root_trust_region(
                        problem,
                        current,
                        current_residual,
                        current_auxiliary,
                        newton_direction,
                        cauchy_direction,
                        selected_jacobian.operator,
                        current_run.trust_radius,
                        search_budget,
                        args,
                        self.trust_region,
                        precision_,
                    )

                def failed_direction(__):
                    return _TrustResult(
                        state=current,
                        residual=current_residual,
                        auxiliary=current_auxiliary,
                        step=jax.tree.map(jnp.zeros_like, current),
                        radius=current_run.trust_radius,
                        evaluations=jnp.asarray(0, dtype=jnp.int32),
                        rejections=jnp.asarray(0, dtype=jnp.int32),
                        accepted=jnp.asarray(False),
                        finite_seen=jnp.asarray(False),
                        domain_failures=jnp.asarray(0, dtype=jnp.int32),
                        nonfinite_trials=jnp.asarray(0, dtype=jnp.int32),
                    )

                search = jax.lax.cond(
                    usable & (search_budget > 0),
                    search_direction,
                    failed_direction,
                    operand=None,
                )
                step_norm = _space_norm(
                    problem.state_space,
                    search.step,
                    precision_,
                )
                candidate_norm = _space_norm(
                    problem.residual_space,
                    search.residual,
                    precision_,
                )
                stagnated = search.accepted & (
                    step_norm
                    <= termination.step_threshold(
                        _space_norm(problem.state_space, current, precision_)
                    )
                )
                diverged = candidate_norm > (
                    termination.divergence_factor
                    * jnp.maximum(current_run.initial_residual_norm, 1e-30)
                )
                candidate_converged = search.accepted & (
                    candidate_norm
                    <= termination.residual_threshold(current_run.initial_residual_norm)
                )
                linear_iterations = jnp.sum(
                    linear_result.diagnostics.iterations, dtype=jnp.int32
                )
                next_residual_evaluations = used_before_search + search.evaluations
                next_linear_iterations = current_run.linear_iterations + linear_iterations
                evaluations_exhausted = (
                    jnp.asarray(False)
                    if termination.maximum_evaluations is None
                    else next_residual_evaluations >= termination.maximum_evaluations
                )
                linear_exhausted = (
                    jnp.asarray(False)
                    if termination.maximum_linear_iterations is None
                    else next_linear_iterations >= termination.maximum_linear_iterations
                )
                retry_rejected = (self.jacobian_refresh.strategy == "rejection") & (
                    search.rejections > 0
                )
                failed_status = jnp.where(
                    linear_exhausted,
                    int(NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED),
                    _linear_failure_status(linear_result.status),
                )
                status = jnp.where(
                    ~usable,
                    failed_status,
                    jnp.where(
                        candidate_converged,
                        int(NonlinearStatus.SUCCESS),
                        jnp.where(
                            stagnated,
                            int(NonlinearStatus.RESIDUAL_STAGNATION),
                            jnp.where(
                                diverged,
                                int(NonlinearStatus.DIVERGENCE),
                                jnp.where(
                                    evaluations_exhausted,
                                    int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
                                    jnp.where(
                                        linear_exhausted,
                                        int(
                                            NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED
                                        ),
                                        jnp.where(
                                            search.accepted | retry_rejected,
                                            int(NonlinearStatus.ITERATING),
                                            jnp.where(
                                                search.finite_seen,
                                                int(NonlinearStatus.TRUST_REGION_FAILED),
                                                jnp.where(
                                                    search.domain_failures > 0,
                                                    int(
                                                        NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE
                                                    ),
                                                    int(
                                                        NonlinearStatus.NONFINITE_EVALUATION
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ).astype(jnp.int32)
                refreshed_age = jnp.where(
                    refreshed, jnp.asarray(0, dtype=jnp.int32), current_run.jacobian_age
                )
                reference_norm = jnp.where(
                    refreshed,
                    current_residual_norm,
                    current_run.jacobian_reference_residual_norm,
                )
                reference_rejections = jnp.where(
                    refreshed,
                    current_run.globalization_rejections,
                    current_run.jacobian_reference_rejected_steps,
                )
                next_forcing = jnp.where(
                    search.accepted,
                    self.forcing_policy.next(
                        current_run.forcing,
                        current_residual_norm,
                        candidate_norm,
                    ),
                    current_run.forcing,
                )
                updated = _RootState(
                    residual=search.residual,
                    auxiliary=search.auxiliary,
                    initial_residual_norm=current_run.initial_residual_norm,
                    residual_norm=candidate_norm,
                    step_norm=step_norm,
                    iteration=current_run.iteration + 1,
                    residual_evaluations=next_residual_evaluations,
                    jvp_evaluations=(
                        current_run.jvp_evaluations
                        + jnp.sum(linear_result.diagnostics.matvec_count, dtype=jnp.int32)
                        + search.evaluations
                        + 1
                    ),
                    vjp_evaluations=current_run.vjp_evaluations + 1,
                    jacobian_preparations=current_run.jacobian_preparations
                    + refreshed.astype(jnp.int32),
                    linear_solves=current_run.linear_solves + 1,
                    linear_iterations=next_linear_iterations,
                    accepted_steps=current_run.accepted_steps
                    + search.accepted.astype(jnp.int32),
                    rejected_steps=current_run.rejected_steps
                    + (~search.accepted).astype(jnp.int32),
                    globalization_rejections=current_run.globalization_rejections
                    + search.rejections,
                    domain_failures=current_run.domain_failures + search.domain_failures,
                    nonfinite_trials=current_run.nonfinite_trials
                    + search.nonfinite_trials,
                    setup_refreshes=current_run.setup_refreshes,
                    numeric_refreshes=current_run.numeric_refreshes + 1,
                    forcing=next_forcing,
                    last_forcing=current_run.forcing,
                    jacobian_age=refreshed_age + search.accepted.astype(jnp.int32),
                    jacobian_reference_residual_norm=reference_norm,
                    jacobian_reference_rejected_steps=reference_rejections,
                    trust_radius=search.radius,
                    status=status,
                    refresh_state=refresh_state,
                    recycling=recycling,
                    final_linear_status=linear_result.status,
                    final_linear_rank=linear_result.diagnostics.rank,
                    final_linear_condition_estimate=jnp.asarray(
                        linear_result.diagnostics.condition_estimate,
                        dtype=current_run.final_linear_condition_estimate.dtype,
                    ),
                    final_linear_residual_norm=jnp.asarray(
                        linear_result.diagnostics.residual_norm,
                        dtype=current_run.final_linear_residual_norm.dtype,
                    ),
                    final_linear_converged=linear_result.diagnostics.converged,
                )
                return (
                    search.state,
                    eqx.partition(updated, eqx.is_array)[0],
                    eqx.partition(selected_jacobian, eqx.is_array)[0],
                )

            return jax.lax.cond(terminal_now, terminal, step, operand=None)

        state, dynamic_run, dynamic_jacobian = jax.lax.while_loop(
            _condition(termination),
            body,
            (state, dynamic_run, dynamic_jacobian),
        )
        run = eqx.combine(dynamic_run, static_run)
        jacobian = eqx.combine(dynamic_jacobian, static_jacobian)
        status = _terminal_status(run, termination)
        result = _package_result(
            self,
            problem,
            state,
            run,
            status,
            termination,
            self.jacobian_policy,
            self.jacobian_refresh,
            "dogleg-residual-trust-region",
            args,
            precision_,
        )
        return (result, state, run, jacobian) if _return_internal else result


def root(
    problem: NonlinearSystemProblem | AbstractNonlinearSystemTransformation,
    initial_state: PyTree[Any],
    /,
    *,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
    args: Any = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> NonlinearResult:
    """Solve one physical nonlinear system with explicit transformation semantics."""
    method_ = NewtonKrylov() if method is None else method
    termination_ = NonlinearTermination() if termination is None else termination
    if not isinstance(method_, AbstractNonlinearMethod):
        raise TypeError("method must be an AbstractNonlinearMethod or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be a NonlinearTermination or None.")
    if precision is not None and not isinstance(
        method_,
        (NewtonKrylov, NewtonTrustRegion),
    ):
        raise ValueError(
            "A root precision override requires NewtonKrylov or NewtonTrustRegion; "
            "configure precision on other nonlinear methods directly."
        )

    def solve_selected(
        current_problem: NonlinearSystemProblem,
        current_termination: NonlinearTermination,
        /,
    ) -> NonlinearResult:
        if isinstance(method_, (NewtonKrylov, NewtonTrustRegion)):
            return method_.solve(
                current_problem,
                initial_state,
                termination=current_termination,
                args=args,
                precision=precision,
            )
        return method_.solve(
            current_problem,
            initial_state,
            termination=current_termination,
            args=args,
        )

    if isinstance(problem, AbstractNonlinearSystemTransformation):
        if (
            termination_.maximum_evaluations is not None
            and termination_.maximum_evaluations < 2
        ):
            raise ValueError(
                "A transformed solve requires at least two residual evaluations "
                "to solve and certify the physical system."
            )
        inner_termination = NonlinearTermination(
            absolute_residual=termination_.absolute_residual,
            relative_residual=termination_.relative_residual,
            absolute_step=termination_.absolute_step,
            relative_step=termination_.relative_step,
            maximum_steps=termination_.maximum_steps,
            maximum_evaluations=(
                None
                if termination_.maximum_evaluations is None
                else termination_.maximum_evaluations - 1
            ),
            maximum_linear_iterations=termination_.maximum_linear_iterations,
            divergence_factor=termination_.divergence_factor,
        )
        transformed = solve_selected(problem.problem, inner_termination)
        return problem.finalize_result(
            transformed,
            initial_state,
            termination_,
            args=args,
        )
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError(
            "problem must be a NonlinearSystemProblem or nonlinear transformation."
        )
    return solve_selected(problem, termination_)


__all__ = [
    "JacobianRefreshPolicy",
    "JacobianRefreshStrategy",
    "NewtonForcingPolicy",
    "NewtonForcingStrategy",
    "NewtonKrylov",
    "NewtonTrustRegion",
    "RootLineSearch",
    "RootTrustRegion",
    "root",
]
