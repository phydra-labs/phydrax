#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import jax
import jax.numpy as jnp

from .._bounds import Bounds
from .._numerics import weight_ess
from ..optim import (
    LinearProgram,
    MixedIntegerProgram,
    MixedIntegerSolvePolicy,
    solve_mixed_integer_program,
)
from ._canonical import _matrix, _target_rows
from ._problem import ExactMoments, IntervalMoments, MomentCalibrationProblem
from ._results import (
    MomentCalibrationDiagnostics,
    MomentCalibrationProvenance,
    MomentCalibrationResult,
    MomentCalibrationStatus,
)


def _program(problem):
    if problem.subset is None:
        raise ValueError("mixed-integer calibration requires EqualWeightSubset.")
    count = problem.source_points
    cardinality = problem.subset.cardinality
    prior = jax.nn.softmax(jnp.where(problem.mask, problem.prior_log_weights, -jnp.inf))
    exact_rows = [jnp.ones((1, count), dtype=prior.dtype)]
    exact_rhs = [jnp.asarray([cardinality], dtype=prior.dtype)]
    inequality_rows = []
    inequality_rhs = []
    exact, inequalities = _target_rows(
        _matrix(problem.moment_map) / cardinality, problem.target
    )
    for matrix, rhs in exact:
        exact_rows.append(matrix)
        exact_rhs.append(rhs)
    for matrix, rhs in inequalities:
        inequality_rows.append(matrix)
        inequality_rhs.append(rhs)
    if problem.group_constraints is not None:
        exact, inequalities = _target_rows(
            _matrix(problem.group_constraints.group_map) / cardinality,
            problem.group_constraints.target,
        )
        for matrix, rhs in exact:
            exact_rows.append(matrix)
            exact_rhs.append(rhs)
        for matrix, rhs in inequalities:
            inequality_rows.append(matrix)
            inequality_rhs.append(rhs)
    linear = -jnp.log(jnp.where(prior > 0.0, prior, 1.0)) / cardinality
    relaxation = LinearProgram(
        linear,
        equality_matrix=jnp.concatenate(exact_rows, axis=0),
        equality_rhs=jnp.concatenate(exact_rhs, axis=0),
        inequality_matrix=(
            None if not inequality_rows else jnp.concatenate(inequality_rows, axis=0)
        ),
        inequality_rhs=(
            None if not inequality_rhs else jnp.concatenate(inequality_rhs, axis=0)
        ),
        bounds=Bounds(
            jnp.zeros((count,)),
            problem.mask.astype(prior.dtype),
        ),
        problem_id=f"{problem.problem_id}:equal-subset",
    )
    return MixedIntegerProgram(
        relaxation,
        binary_indices=tuple(range(count)),
        program_id=f"{problem.problem_id}:equal-subset",
    )


def calibrate_moments_subset(problem: MomentCalibrationProblem, solver=None):
    """Solve fixed-cardinality equal weighting through MixedIntegerProgram."""
    policy = MixedIntegerSolvePolicy() if solver is None else solver
    if not isinstance(policy, MixedIntegerSolvePolicy):
        raise TypeError("mixed-integer solver must be MixedIntegerSolvePolicy.")
    result = solve_mixed_integer_program(_program(problem), policy)
    cardinality = problem.subset.cardinality
    selected = jnp.where(result.integral, jnp.rint(result.primal), result.primal)
    weights = selected / cardinality
    log_weights = jnp.where(weights > 0.0, jnp.log(weights), -jnp.inf)
    achieved = problem.moment_map.mv(weights)
    if isinstance(problem.target, ExactMoments):
        residual = achieved - problem.target.values
    elif isinstance(problem.target, IntervalMoments):
        residual = jnp.where(
            achieved < problem.target.lower,
            achieved - problem.target.lower,
            jnp.where(
                achieved > problem.target.upper, achieved - problem.target.upper, 0.0
            ),
        )
    else:
        raise TypeError("Equal subset supports exact or interval targets.")
    tolerance = policy.integrality_tolerance
    normalization = jnp.abs(jnp.sum(weights) - 1.0)
    valid = (
        result.successful
        & (jnp.max(jnp.abs(residual)) <= tolerance)
        & (normalization <= tolerance)
    )
    status = jnp.where(
        valid,
        int(MomentCalibrationStatus.SUCCESS),
        int(MomentCalibrationStatus.OPTIMIZATION_FAILED),
    ).astype(jnp.int32)
    prior = jax.nn.softmax(jnp.where(problem.mask, problem.prior_log_weights, -jnp.inf))
    diagnostics = MomentCalibrationDiagnostics(
        optimizer_status=result.status,
        optimization=result,
        prior_moments=problem.moment_map.mv(prior),
        target_residual=residual,
        scaled_target_residual=residual,
        maximum_absolute_residual=jnp.max(jnp.abs(residual)),
        maximum_scaled_residual=jnp.max(jnp.abs(residual)),
        affine_residual_norm=jnp.asarray(0.0),
        numerical_affine_rank=jnp.asarray(problem.moment_count),
        rank_cutoff=jnp.asarray(0.0),
        minimum_prior_eigenvalue=jnp.asarray(jnp.nan),
        maximum_prior_eigenvalue=jnp.asarray(jnp.nan),
        minimum_final_eigenvalue=jnp.asarray(jnp.nan),
        final_condition_estimate=jnp.asarray(jnp.nan),
        dual_gradient_norm=result.absolute_gap,
        dual_norm=jnp.asarray(0.0),
        relative_entropy=jnp.sum(
            jnp.where(weights > 0, weights * (log_weights - jnp.log(prior)), 0.0)
        ),
        effective_sample_size=weight_ess(weights, axis=0),
        active_support=jnp.sum(weights > 0),
        minimum_active_weight=jnp.min(jnp.where(weights > 0, weights, jnp.inf)),
        maximum_active_weight=jnp.max(weights),
        maximum_log_weight_ratio=jnp.max(jnp.abs(log_weights - jnp.log(prior))),
        normalization_residual=normalization,
        geometry_finite=jnp.all(jnp.isfinite(weights)),
        spectrum=(),
    )
    provenance = MomentCalibrationProvenance(
        problem_id=problem.problem_id,
        operator_id=problem.moment_map.operator_id,
        target_kind=type(problem.target).__name__,
        source_points=problem.source_points,
        moment_count=problem.moment_count,
        execution="mixed-integer",
        differentiation="none",
        optimizer=result.tree_result,
    )
    return MomentCalibrationResult(
        problem, log_weights, jnp.empty((0,)), achieved, status, diagnostics, provenance
    )


__all__ = ["calibrate_moments_subset"]
