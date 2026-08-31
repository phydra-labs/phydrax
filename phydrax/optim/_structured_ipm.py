#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    prepare as prepare_linear,
    refresh as refresh_linear,
    release as release_linear,
    solve as solve_linear,
    sparse_provider_capabilities,
    SparseLDLT,
)
from ._iterative import (
    MinimizationResult,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._sparse_kkt import (
    assemble_sparse_augmented_kkt,
    plan_sparse_augmented_kkt,
    SparseAugmentedKKTPlan,
)
from ._structured_nonlinear import (
    PreparedStructuredNonlinearProgram,
    StructuredNonlinearResult,
    StructuredNonlinearWarmStart,
    StructuredOptimizationWork,
)


class SparseStructuredIPMState(StrictModule):
    """Accepted bound-form primal-dual state for one structured NLP."""

    primal: Array
    slack: Array
    equality_multipliers: Array
    general_multipliers: Array
    lower_bound_multipliers: Array
    upper_bound_multipliers: Array
    lower_slack_multipliers: Array
    upper_slack_multipliers: Array
    barrier: Array
    iteration: Array
    status: Array
    accepted_steps: Array
    rejected_steps: Array
    final_step_norm: Array
    objective_evaluations: Array
    constraint_evaluations: Array
    gradient_evaluations: Array
    jacobian_evaluations: Array
    hessian_evaluations: Array
    kkt_assemblies: Array
    factorizations: Array
    right_hand_side_solves: Array


class SparseStructuredIPMEvidence(StrictModule):
    kkt_plan_id: str = eqx.field(static=True)
    kkt_dimension: int = eqx.field(static=True)
    factorizations: Array
    right_hand_side_solves: Array
    final_barrier: Array


def _maximum(value: Array, /) -> Array:
    return jnp.max(jnp.abs(value), initial=jnp.asarray(0.0, dtype=value.dtype))


def _strict_interior(value, lower, upper, push):
    lower_finite = jnp.isfinite(lower)
    upper_finite = jnp.isfinite(upper)
    width = upper - lower
    two_sided_push = jnp.minimum(push, 0.1 * width)
    lower_push = jnp.where(lower_finite, lower + push, value)
    upper_push = jnp.where(upper_finite, upper - push, value)
    result = jnp.where(lower_finite, jnp.maximum(value, lower_push), value)
    result = jnp.where(upper_finite, jnp.minimum(result, upper_push), result)
    result = jnp.where(
        lower_finite & upper_finite,
        jnp.clip(value, lower + two_sided_push, upper - two_sided_push),
        result,
    )
    return result


def _safe_gap(gap, finite):
    return jnp.where(finite, gap, 1.0)


def _positive_multiplier(value, finite):
    return jnp.where(finite, jnp.maximum(value, 1e-4), 0.0)


def _fraction(gap, direction, finite, fraction):
    ratio = jnp.where(finite & (direction < 0.0), -gap / direction, jnp.inf)
    return jnp.minimum(1.0, fraction * jnp.min(ratio, initial=jnp.inf))


def _split_values(prepared, plan, coordinates):
    evaluation = prepared.evaluate(coordinates)
    constraints = evaluation.constraints
    equality = (
        constraints[plan.equality_indices]
        - prepared.constraint_lower[plan.equality_indices]
    )
    general = constraints[plan.general_indices]
    return evaluation, equality, general


def _full_constraint_multipliers(prepared, plan, equality_dual, general_dual):
    values = jnp.zeros((prepared.program.num_constraints,), dtype=equality_dual.dtype)
    values = values.at[plan.equality_indices].set(equality_dual)
    return values.at[plan.general_indices].set(general_dual)


def _residuals(prepared, plan, state):
    evaluation, equality, general = _split_values(prepared, plan, state.primal)
    multipliers = _full_constraint_multipliers(
        prepared,
        plan,
        state.equality_multipliers,
        state.general_multipliers,
    )
    stationarity_primal = (
        evaluation.gradient
        + evaluation.jacobian.transpose_mv(multipliers)
        - state.lower_bound_multipliers
        + state.upper_bound_multipliers
    )
    stationarity_slack = (
        -state.general_multipliers
        - state.lower_slack_multipliers
        + state.upper_slack_multipliers
    )
    general_residual = general - state.slack
    lower_x_finite = jnp.isfinite(prepared.variable_lower)
    upper_x_finite = jnp.isfinite(prepared.variable_upper)
    lower_s = prepared.constraint_lower[plan.general_indices]
    upper_s = prepared.constraint_upper[plan.general_indices]
    lower_s_finite = jnp.isfinite(lower_s)
    upper_s_finite = jnp.isfinite(upper_s)
    lower_x_gap = _safe_gap(
        state.primal - prepared.variable_lower,
        lower_x_finite,
    )
    upper_x_gap = _safe_gap(
        prepared.variable_upper - state.primal,
        upper_x_finite,
    )
    lower_s_gap = _safe_gap(state.slack - lower_s, lower_s_finite)
    upper_s_gap = _safe_gap(upper_s - state.slack, upper_s_finite)
    complementarity = (
        jnp.where(
            lower_x_finite,
            lower_x_gap * state.lower_bound_multipliers,
            0.0,
        ),
        jnp.where(
            upper_x_finite,
            upper_x_gap * state.upper_bound_multipliers,
            0.0,
        ),
        jnp.where(
            lower_s_finite,
            lower_s_gap * state.lower_slack_multipliers,
            0.0,
        ),
        jnp.where(
            upper_s_finite,
            upper_s_gap * state.upper_slack_multipliers,
            0.0,
        ),
    )
    primal = jnp.maximum(_maximum(equality), _maximum(general_residual))
    primal = jnp.maximum(
        primal,
        _maximum(jnp.where(lower_x_finite, jnp.maximum(-lower_x_gap, 0.0), 0.0)),
    )
    primal = jnp.maximum(
        primal,
        _maximum(jnp.where(upper_x_finite, jnp.maximum(-upper_x_gap, 0.0), 0.0)),
    )
    primal = jnp.maximum(
        primal,
        _maximum(jnp.where(lower_s_finite, jnp.maximum(-lower_s_gap, 0.0), 0.0)),
    )
    primal = jnp.maximum(
        primal,
        _maximum(jnp.where(upper_s_finite, jnp.maximum(-upper_s_gap, 0.0), 0.0)),
    )
    dual = jnp.maximum(_maximum(stationarity_primal), _maximum(stationarity_slack))
    dual = jnp.maximum(dual, _maximum(jnp.minimum(state.lower_bound_multipliers, 0.0)))
    dual = jnp.maximum(dual, _maximum(jnp.minimum(state.upper_bound_multipliers, 0.0)))
    dual = jnp.maximum(dual, _maximum(jnp.minimum(state.lower_slack_multipliers, 0.0)))
    dual = jnp.maximum(dual, _maximum(jnp.minimum(state.upper_slack_multipliers, 0.0)))
    complementarity_norm = max(_maximum(value) for value in complementarity)
    norm = jnp.maximum(primal, jnp.maximum(dual, complementarity_norm))
    return (
        evaluation,
        multipliers,
        stationarity_primal,
        stationarity_slack,
        equality,
        general_residual,
        lower_x_gap,
        upper_x_gap,
        lower_s_gap,
        upper_s_gap,
        complementarity,
        norm,
    )


def _average_complementarity(state, prepared, plan):
    lower_x_finite = jnp.isfinite(prepared.variable_lower)
    upper_x_finite = jnp.isfinite(prepared.variable_upper)
    lower_s = prepared.constraint_lower[plan.general_indices]
    upper_s = prepared.constraint_upper[plan.general_indices]
    lower_s_finite = jnp.isfinite(lower_s)
    upper_s_finite = jnp.isfinite(upper_s)
    products = jnp.concatenate(
        (
            jnp.where(
                lower_x_finite,
                (state.primal - prepared.variable_lower) * state.lower_bound_multipliers,
                0.0,
            ),
            jnp.where(
                upper_x_finite,
                (prepared.variable_upper - state.primal) * state.upper_bound_multipliers,
                0.0,
            ),
            jnp.where(
                lower_s_finite,
                (state.slack - lower_s) * state.lower_slack_multipliers,
                0.0,
            ),
            jnp.where(
                upper_s_finite,
                (upper_s - state.slack) * state.upper_slack_multipliers,
                0.0,
            ),
        )
    )
    count = (
        jnp.sum(lower_x_finite)
        + jnp.sum(upper_x_finite)
        + jnp.sum(lower_s_finite)
        + jnp.sum(upper_s_finite)
    )
    return jnp.sum(products) / jnp.maximum(count, 1)


def initialize_sparse_structured_ipm(
    prepared: PreparedStructuredNonlinearProgram,
    plan: SparseAugmentedKKTPlan,
    initial_coordinates: ArrayLike,
    warm_start: StructuredNonlinearWarmStart | None,
    /,
) -> SparseStructuredIPMState:
    x = prepared.validate_coordinates(
        initial_coordinates if warm_start is None else warm_start.primal
    )
    fixed = np.asarray(prepared.template.fixed_variable_mask)
    if np.any(fixed):
        raise ValueError(
            "Sparse augmented IPM requires fixed variables to be eliminated or "
            "lowered as equalities."
        )
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(x), initial=1.0))
    push = jnp.asarray(1e-4, dtype=x.dtype) * scale
    x = _strict_interior(x, prepared.variable_lower, prepared.variable_upper, push)
    _, _, general = _split_values(prepared, plan, x)
    lower_s = prepared.constraint_lower[plan.general_indices]
    upper_s = prepared.constraint_upper[plan.general_indices]
    slack = _strict_interior(general, lower_s, upper_s, push)
    if warm_start is None:
        constraint_multipliers = jnp.zeros(
            (prepared.program.num_constraints,), dtype=x.dtype
        )
        lower_x_dual = jnp.ones_like(x)
        upper_x_dual = jnp.ones_like(x)
    else:
        constraint_multipliers = warm_start.constraint_multipliers
        lower_x_dual = warm_start.lower_bound_multipliers
        upper_x_dual = warm_start.upper_bound_multipliers
    equality_dual = constraint_multipliers[plan.equality_indices]
    general_dual = constraint_multipliers[plan.general_indices]
    lower_x_dual = _positive_multiplier(
        lower_x_dual,
        jnp.isfinite(prepared.variable_lower),
    )
    upper_x_dual = _positive_multiplier(
        upper_x_dual,
        jnp.isfinite(prepared.variable_upper),
    )
    lower_s_dual = _positive_multiplier(
        jnp.ones_like(slack),
        jnp.isfinite(lower_s),
    )
    upper_s_dual = _positive_multiplier(
        jnp.ones_like(slack),
        jnp.isfinite(upper_s),
    )
    state = SparseStructuredIPMState(
        x,
        slack,
        equality_dual,
        general_dual,
        lower_x_dual,
        upper_x_dual,
        lower_s_dual,
        upper_s_dual,
        jnp.asarray(1.0, dtype=x.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(int(OptimizationStatus.ITERATING), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=x.dtype),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return eqx.tree_at(
        lambda value: value.barrier,
        state,
        jnp.maximum(_average_complementarity(state, prepared, plan), 1e-8),
    )


def _complementarity_rhs(product, finite, barrier, affine_product=None):
    correction = 0.0 if affine_product is None else affine_product
    return jnp.where(finite, product + correction - barrier, 0.0)


def _direction_rhs(
    state,
    prepared,
    plan,
    residuals,
    complementarity,
):
    (
        _,
        _,
        stationarity_x,
        stationarity_s,
        equality,
        general,
        lower_x_gap,
        upper_x_gap,
        lower_s_gap,
        upper_s_gap,
        _,
        _,
    ) = residuals
    lower_x_finite = jnp.isfinite(prepared.variable_lower)
    upper_x_finite = jnp.isfinite(prepared.variable_upper)
    lower_s_finite = jnp.isfinite(prepared.constraint_lower[plan.general_indices])
    upper_s_finite = jnp.isfinite(prepared.constraint_upper[plan.general_indices])
    correction_x = jnp.where(
        lower_x_finite,
        complementarity[0] / lower_x_gap,
        0.0,
    ) - jnp.where(
        upper_x_finite,
        complementarity[1] / upper_x_gap,
        0.0,
    )
    correction_s = jnp.where(
        lower_s_finite,
        complementarity[2] / lower_s_gap,
        0.0,
    ) - jnp.where(
        upper_s_finite,
        complementarity[3] / upper_s_gap,
        0.0,
    )
    return jnp.concatenate(
        (
            -stationarity_x - correction_x,
            -stationarity_s - correction_s,
            -equality,
            -general,
        )
    )


def _recover_direction(state, prepared, plan, residuals, complementarity, solved):
    n = plan.num_primal
    nd = plan.num_slacks
    nc = plan.num_equalities
    dx = solved[:n]
    ds = solved[n : n + nd]
    dyc = solved[n + nd : n + nd + nc]
    dyd = solved[n + nd + nc :]
    lower_x_gap, upper_x_gap, lower_s_gap, upper_s_gap = residuals[6:10]
    lower_x_finite = jnp.isfinite(prepared.variable_lower)
    upper_x_finite = jnp.isfinite(prepared.variable_upper)
    lower_s_finite = jnp.isfinite(prepared.constraint_lower[plan.general_indices])
    upper_s_finite = jnp.isfinite(prepared.constraint_upper[plan.general_indices])
    dzl = jnp.where(
        lower_x_finite,
        (-complementarity[0] - state.lower_bound_multipliers * dx) / lower_x_gap,
        0.0,
    )
    dzu = jnp.where(
        upper_x_finite,
        (-complementarity[1] + state.upper_bound_multipliers * dx) / upper_x_gap,
        0.0,
    )
    dvl = jnp.where(
        lower_s_finite,
        (-complementarity[2] - state.lower_slack_multipliers * ds) / lower_s_gap,
        0.0,
    )
    dvu = jnp.where(
        upper_s_finite,
        (-complementarity[3] + state.upper_slack_multipliers * ds) / upper_s_gap,
        0.0,
    )
    return dx, ds, dyc, dyd, dzl, dzu, dvl, dvu


def _step_fraction(state, prepared, plan, direction, fraction):
    dx, ds, _, _, dzl, dzu, dvl, dvu = direction
    lower_x_finite = jnp.isfinite(prepared.variable_lower)
    upper_x_finite = jnp.isfinite(prepared.variable_upper)
    lower_s = prepared.constraint_lower[plan.general_indices]
    upper_s = prepared.constraint_upper[plan.general_indices]
    lower_s_finite = jnp.isfinite(lower_s)
    upper_s_finite = jnp.isfinite(upper_s)
    values = (
        _fraction(
            state.primal - prepared.variable_lower,
            dx,
            lower_x_finite,
            fraction,
        ),
        _fraction(
            prepared.variable_upper - state.primal,
            -dx,
            upper_x_finite,
            fraction,
        ),
        _fraction(state.slack - lower_s, ds, lower_s_finite, fraction),
        _fraction(upper_s - state.slack, -ds, upper_s_finite, fraction),
        _fraction(state.lower_bound_multipliers, dzl, lower_x_finite, fraction),
        _fraction(state.upper_bound_multipliers, dzu, upper_x_finite, fraction),
        _fraction(state.lower_slack_multipliers, dvl, lower_s_finite, fraction),
        _fraction(state.upper_slack_multipliers, dvu, upper_s_finite, fraction),
    )
    return min(values)


def _candidate(state, direction, rate):
    dx, ds, dyc, dyd, dzl, dzu, dvl, dvu = direction
    return eqx.tree_at(
        lambda value: (
            value.primal,
            value.slack,
            value.equality_multipliers,
            value.general_multipliers,
            value.lower_bound_multipliers,
            value.upper_bound_multipliers,
            value.lower_slack_multipliers,
            value.upper_slack_multipliers,
        ),
        state,
        (
            state.primal + rate * dx,
            state.slack + rate * ds,
            state.equality_multipliers + rate * dyc,
            state.general_multipliers + rate * dyd,
            state.lower_bound_multipliers + rate * dzl,
            state.upper_bound_multipliers + rate * dzu,
            state.lower_slack_multipliers + rate * dvl,
            state.upper_slack_multipliers + rate * dvu,
        ),
    )


def _prepare_kkt_linear(operator, policy, previous=None):
    if isinstance(policy.method, SparseLDLT):
        problem = LinearSystem(operator)
    else:
        problem = LinearSystem(DenseLinearOperator(operator.as_dense()))
    return (
        prepare_linear(problem, policy)
        if previous is None
        else refresh_linear(previous, problem)
    )


def advance_sparse_structured_ipm(
    prepared,
    plan,
    state,
    termination,
    linear_policy,
    *,
    fraction_to_boundary,
    sufficient_decrease,
    maximum_line_search_steps,
    regularization,
):
    residuals = _residuals(prepared, plan, state)
    current_norm = residuals[-1]
    threshold = termination.absolute_optimality
    if float(current_norm) <= float(threshold):
        return eqx.tree_at(
            lambda value: value.status,
            state,
            jnp.asarray(int(OptimizationStatus.SUCCESS), dtype=jnp.int32),
        )
    evaluation, multipliers = residuals[:2]
    hessian = prepared.hessian_operator(state.primal, multipliers)
    lower_x_gap, upper_x_gap, lower_s_gap, upper_s_gap = residuals[6:10]
    sigma_x = (
        state.lower_bound_multipliers / lower_x_gap
        + state.upper_bound_multipliers / upper_x_gap
    )
    sigma_s = (
        state.lower_slack_multipliers / lower_s_gap
        + state.upper_slack_multipliers / upper_s_gap
    )
    kkt_operator = assemble_sparse_augmented_kkt(
        plan,
        hessian,
        evaluation.jacobian,
        sigma_x,
        sigma_s,
        primal_regularization=regularization,
        dual_regularization=regularization,
    )
    linear = _prepare_kkt_linear(kkt_operator, linear_policy)
    affine_complementarity = residuals[10]
    affine_rhs = _direction_rhs(
        state,
        prepared,
        plan,
        residuals,
        affine_complementarity,
    )
    affine_solution = solve_linear(linear, affine_rhs)
    affine_direction = _recover_direction(
        state,
        prepared,
        plan,
        residuals,
        affine_complementarity,
        affine_solution.value,
    )
    affine_rate = _step_fraction(state, prepared, plan, affine_direction, 1.0)
    affine_state = _candidate(state, affine_direction, affine_rate)
    affine_barrier = _average_complementarity(affine_state, prepared, plan)
    sigma = jnp.clip(
        (affine_barrier / jnp.maximum(state.barrier, 1e-30)) ** 3,
        0.0,
        1.0,
    )
    affine_products = (
        affine_direction[4] * affine_direction[0],
        -affine_direction[5] * affine_direction[0],
        affine_direction[6] * affine_direction[1],
        -affine_direction[7] * affine_direction[1],
    )
    lower_x_finite = jnp.isfinite(prepared.variable_lower)
    upper_x_finite = jnp.isfinite(prepared.variable_upper)
    lower_s_finite = jnp.isfinite(prepared.constraint_lower[plan.general_indices])
    upper_s_finite = jnp.isfinite(prepared.constraint_upper[plan.general_indices])
    corrected = (
        _complementarity_rhs(
            residuals[10][0],
            lower_x_finite,
            sigma * state.barrier,
            affine_products[0],
        ),
        _complementarity_rhs(
            residuals[10][1],
            upper_x_finite,
            sigma * state.barrier,
            affine_products[1],
        ),
        _complementarity_rhs(
            residuals[10][2],
            lower_s_finite,
            sigma * state.barrier,
            affine_products[2],
        ),
        _complementarity_rhs(
            residuals[10][3],
            upper_s_finite,
            sigma * state.barrier,
            affine_products[3],
        ),
    )
    corrector_rhs = _direction_rhs(
        state,
        prepared,
        plan,
        residuals,
        corrected,
    )
    corrector_solution = solve_linear(linear, corrector_rhs)
    direction = _recover_direction(
        state,
        prepared,
        plan,
        residuals,
        corrected,
        corrector_solution.value,
    )
    rate = _step_fraction(
        state,
        prepared,
        plan,
        direction,
        fraction_to_boundary,
    )
    accepted = False
    candidate = state
    candidate_norm = current_norm
    for _ in range(maximum_line_search_steps):
        trial = _candidate(state, direction, rate)
        trial_norm = _residuals(prepared, plan, trial)[-1]
        finite = all(
            bool(jnp.all(jnp.isfinite(value)))
            for value in (
                trial.primal,
                trial.slack,
                trial.lower_bound_multipliers,
                trial.upper_bound_multipliers,
                trial.lower_slack_multipliers,
                trial.upper_slack_multipliers,
            )
        )
        if finite and float(trial_norm) <= float(
            (1.0 - sufficient_decrease * rate) * current_norm
        ):
            candidate = trial
            candidate_norm = trial_norm
            accepted = True
            break
        rate *= 0.5
    if isinstance(linear_policy.method, SparseLDLT):
        release_linear(linear)
    if not accepted:
        return eqx.tree_at(
            lambda value: (
                value.iteration,
                value.status,
                value.rejected_steps,
                value.hessian_evaluations,
                value.kkt_assemblies,
                value.factorizations,
                value.right_hand_side_solves,
            ),
            state,
            (
                state.iteration + 1,
                jnp.asarray(int(OptimizationStatus.RESTORATION_FAILED), dtype=jnp.int32),
                state.rejected_steps + 1,
                state.hessian_evaluations + 1,
                state.kkt_assemblies + 1,
                state.factorizations + 1,
                state.right_hand_side_solves + 2,
            ),
        )
    candidate = eqx.tree_at(
        lambda value: (
            value.barrier,
            value.iteration,
            value.accepted_steps,
            value.final_step_norm,
            value.objective_evaluations,
            value.constraint_evaluations,
            value.gradient_evaluations,
            value.jacobian_evaluations,
            value.hessian_evaluations,
            value.kkt_assemblies,
            value.factorizations,
            value.right_hand_side_solves,
        ),
        candidate,
        (
            jnp.maximum(_average_complementarity(candidate, prepared, plan), 1e-12),
            state.iteration + 1,
            state.accepted_steps + 1,
            jnp.linalg.norm(rate * direction[0]),
            state.objective_evaluations + 1,
            state.constraint_evaluations + 1,
            state.gradient_evaluations + 1,
            state.jacobian_evaluations + 1,
            state.hessian_evaluations + 1,
            state.kkt_assemblies + 1,
            state.factorizations + 1,
            state.right_hand_side_solves + 2,
        ),
    )
    status = jnp.where(
        candidate_norm <= threshold,
        int(OptimizationStatus.SUCCESS),
        int(OptimizationStatus.ITERATING),
    ).astype(jnp.int32)
    return eqx.tree_at(lambda value: value.status, candidate, status)


def finalize_sparse_structured_ipm(
    prepared: PreparedStructuredNonlinearProgram,
    plan: SparseAugmentedKKTPlan,
    state: SparseStructuredIPMState,
    /,
    *,
    termination: OptimizationTermination,
    linear_policy: LinearSolvePolicy,
    method_id: str,
    initial_norm: Array,
) -> StructuredNonlinearResult:
    residuals = _residuals(prepared, plan, state)
    final_norm = residuals[-1]
    full_multipliers = _full_constraint_multipliers(
        prepared,
        plan,
        state.equality_multipliers,
        state.general_multipliers,
    )
    certificate = prepared.certificate(
        state.primal,
        full_multipliers,
        state.lower_bound_multipliers,
        state.upper_bound_multipliers,
        active_tolerance=float(jnp.sqrt(termination.absolute_optimality)),
    )
    certified = residuals[0].finite & (
        jnp.maximum(
            _maximum(jnp.asarray(certificate.stationarity_residual)),
            jnp.maximum(
                certificate.primal_feasibility,
                jnp.maximum(
                    certificate.dual_feasibility,
                    certificate.complementarity,
                ),
            ),
        )
        <= termination.absolute_optimality
    )
    public_status = jnp.where(
        (state.status == int(OptimizationStatus.SUCCESS)) & ~certified,
        int(OptimizationStatus.CERTIFICATION_FAILED),
        state.status,
    ).astype(jnp.int32)
    evidence = SparseStructuredIPMEvidence(
        plan.plan_id,
        plan.kkt_dimension,
        state.factorizations,
        state.right_hand_side_solves,
        state.barrier,
    )
    diagnostics = OptimizationDiagnostics(
        iterations=state.iteration,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.rejected_steps,
        objective_evaluations=state.objective_evaluations + 1,
        gradient_evaluations=state.gradient_evaluations + 1,
        jacobian_evaluations=state.jacobian_evaluations + 1,
        hvp_evaluations=state.hessian_evaluations,
        constraint_evaluations=state.constraint_evaluations + 1,
        linear_solves=state.right_hand_side_solves,
        linear_iterations=state.right_hand_side_solves,
        globalization_evaluations=state.accepted_steps + state.rejected_steps,
        initial_optimality_norm=initial_norm,
        final_optimality_norm=final_norm,
        final_step_norm=state.final_step_norm,
        accepted_step_size=jnp.where(state.accepted_steps > 0, 1.0, 0.0),
        damping=state.barrier,
        primal_feasibility=certificate.primal_feasibility,
        dual_feasibility=certificate.dual_feasibility,
        complementarity=certificate.complementarity,
        active_constraints=jnp.sum(certificate.active_mask, dtype=jnp.int32),
        counts_complete=True,
    )
    optimization = MinimizationResult(
        state.primal,
        residuals[0].objective,
        None,
        public_status,
        diagnostics,
        OptimizationProvenance(
            problem_id=prepared.program.program_id,
            method=method_id,
            backend=(
                "spineax-cudss"
                if isinstance(linear_policy.method, SparseLDLT)
                else "phydrax-linalg"
            ),
            backend_method=linear_policy.method.name,
            globalization="primal-dual-residual-backtracking",
            matrix_free=False,
            implicit_differentiation=True,
            notes="Bound-form augmented KKT with factor reuse across predictor/corrector.",
        ),
        certificate=certificate,
        method_evidence=evidence,
    )
    warm = prepared.warm_start(
        state.primal,
        full_multipliers,
        state.lower_bound_multipliers,
        state.upper_bound_multipliers,
    )
    work = StructuredOptimizationWork(
        objective_evaluations=diagnostics.objective_evaluations,
        constraint_evaluations=diagnostics.constraint_evaluations,
        gradient_evaluations=diagnostics.gradient_evaluations,
        jacobian_evaluations=diagnostics.jacobian_evaluations,
        hessian_evaluations=diagnostics.hvp_evaluations,
        kkt_assemblies=state.kkt_assemblies,
        factorizations=state.factorizations,
        right_hand_side_solves=state.right_hand_side_solves,
        backtracking_evaluations=diagnostics.globalization_evaluations,
        certificate_evaluations=1,
        complete=True,
    )
    return StructuredNonlinearResult(
        optimization,
        warm,
        work,
        numeric_version=prepared.numeric_version,
        structure_id=prepared.structure_id,
        method_id=method_id,
    )


def solve_sparse_structured_ipm(
    prepared: PreparedStructuredNonlinearProgram,
    initial_coordinates: Any,
    /,
    *,
    termination: OptimizationTermination,
    warm_start: StructuredNonlinearWarmStart | None,
    linear_policy: LinearSolvePolicy,
    method_id: str,
    fraction_to_boundary: float = 0.995,
    sufficient_decrease: float = 1e-4,
    maximum_line_search_steps: int = 20,
    regularization: float = 1e-8,
) -> StructuredNonlinearResult:
    if not isinstance(prepared, PreparedStructuredNonlinearProgram):
        raise TypeError("prepared must be a PreparedStructuredNonlinearProgram.")
    if prepared.program.hessian_plan is None:
        raise ValueError("Sparse augmented IPM requires an exact Hessian plan.")
    if isinstance(linear_policy.method, SparseLDLT) and not (
        sparse_provider_capabilities("spineax-cudss").reliable_zero_inertia
    ):
        raise ValueError(
            "Spineax cuDSS cannot drive the nonconvex structured IPM until "
            "zero-inertia evidence is reliable."
        )
    plan = plan_sparse_augmented_kkt(prepared.template)
    state = initialize_sparse_structured_ipm(
        prepared,
        plan,
        initial_coordinates,
        warm_start,
    )
    initial_norm = _residuals(prepared, plan, state)[-1]
    while (
        int(state.status) == int(OptimizationStatus.ITERATING)
        and int(state.iteration) < termination.maximum_steps
    ):
        state = advance_sparse_structured_ipm(
            prepared,
            plan,
            state,
            termination,
            linear_policy,
            fraction_to_boundary=fraction_to_boundary,
            sufficient_decrease=sufficient_decrease,
            maximum_line_search_steps=maximum_line_search_steps,
            regularization=regularization,
        )
    if int(state.status) == int(OptimizationStatus.ITERATING):
        state = eqx.tree_at(
            lambda value: value.status,
            state,
            jnp.asarray(int(OptimizationStatus.MAXIMUM_STEPS_REACHED), dtype=jnp.int32),
        )
    return finalize_sparse_structured_ipm(
        prepared,
        plan,
        state,
        termination=termination,
        linear_policy=linear_policy,
        method_id=method_id,
        initial_norm=initial_norm,
    )


__all__ = [
    "SparseStructuredIPMEvidence",
    "SparseStructuredIPMState",
    "advance_sparse_structured_ipm",
    "finalize_sparse_structured_ipm",
    "initialize_sparse_structured_ipm",
    "solve_sparse_structured_ipm",
]
