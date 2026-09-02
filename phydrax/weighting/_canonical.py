#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .._bounds import Bounds
from .._numerics import weight_ess
from .._strict import StrictModule
from ..optim import (
    ConicProgram,
    ConvexSolvePolicy,
    ConvexTermination,
    ExponentialCone,
    LinearProgram,
    NativeHomogeneousConic,
    NonnegativeCone,
    ProductCone,
    solve_convex_program,
    ZeroCone,
)
from ._problem import ExactMoments, IntervalMoments
from ._results import (
    MomentCalibrationDiagnostics,
    MomentCalibrationProvenance,
    MomentCalibrationResult,
    MomentCalibrationStatus,
)


def _matrix(operator):
    if not operator.capabilities.materialize:
        raise ValueError("Canonical calibration requires an explicit moment map.")
    return operator._materialize()


def _target_rows(matrix, target):
    if isinstance(target, ExactMoments):
        return ((matrix, target.values),), ()
    if isinstance(target, IntervalMoments):
        return (), ((-matrix, -target.lower), (matrix, target.upper))
    raise TypeError("Canonical conic calibration supports exact/interval targets.")


def _program(problem):
    count = problem.source_points
    dtype = problem.prior_log_weights.dtype
    prior = jax.nn.softmax(jnp.where(problem.mask, problem.prior_log_weights, -jnp.inf))
    exact_rows = [(jnp.ones((1, count), dtype=dtype), jnp.ones((1,), dtype=dtype))]
    inequality_rows = []
    exact, inequalities = _target_rows(_matrix(problem.moment_map), problem.target)
    exact_rows.extend(exact)
    inequality_rows.extend(inequalities)
    if problem.group_constraints is not None:
        group_matrix = _matrix(problem.group_constraints.group_map)
        exact, inequalities = _target_rows(group_matrix, problem.group_constraints.target)
        exact_rows.extend(exact)
        inequality_rows.extend(inequalities)
    variables = 2 * count
    blocks = []
    rhs_blocks = []
    cones = []
    if exact_rows:
        matrix = jnp.concatenate([value[0] for value in exact_rows], axis=0)
        rhs = jnp.concatenate([value[1] for value in exact_rows], axis=0)
        block = (
            jnp.zeros((matrix.shape[0], variables), dtype=dtype).at[:, :count].set(matrix)
        )
        blocks.append(block)
        rhs_blocks.append(rhs)
        cones.append(ZeroCone(matrix.shape[0]))
    if inequality_rows:
        matrix = jnp.concatenate([value[0] for value in inequality_rows], axis=0)
        rhs = jnp.concatenate([value[1] for value in inequality_rows], axis=0)
        block = (
            jnp.zeros((matrix.shape[0], variables), dtype=dtype).at[:, :count].set(matrix)
        )
        blocks.append(block)
        rhs_blocks.append(rhs)
        cones.append(NonnegativeCone(matrix.shape[0]))
    exponential = jnp.zeros((3 * count, variables), dtype=dtype)
    exponential_rhs = jnp.zeros((3 * count,), dtype=dtype)
    for index in range(count):
        exponential = exponential.at[3 * index, count + index].set(1.0)
        exponential = exponential.at[3 * index + 1, index].set(-1.0)
        exponential_rhs = exponential_rhs.at[3 * index + 2].set(prior[index])
        cones.append(ExponentialCone())
    blocks.append(exponential)
    rhs_blocks.append(exponential_rhs)
    linear = jnp.concatenate(
        (jnp.zeros((count,), dtype=dtype), jnp.ones((count,), dtype=dtype))
    )
    lower = jnp.concatenate(
        (jnp.zeros((count,), dtype=dtype), jnp.full((count,), -jnp.inf, dtype=dtype))
    )
    upper = jnp.full((variables,), jnp.inf, dtype=dtype)
    return ConicProgram(
        None,
        linear,
        jnp.concatenate(blocks, axis=0),
        jnp.concatenate(rhs_blocks, axis=0),
        ProductCone(tuple(cones)),
        bounds=Bounds(lower, upper),
        problem_id=f"{problem.problem_id}:relative-entropy-conic",
        convexity_evidence="construction",
    )


class BoundaryFaceEvidence(StrictModule):
    forced_zero: jax.Array
    relative_interior_witness: jax.Array
    maximum_weights: jax.Array
    certified: jax.Array
    relaxations: tuple


def _face_program(problem, objective):
    count = problem.source_points
    dtype = problem.prior_log_weights.dtype
    exact_rows = [jnp.ones((1, count), dtype=dtype)]
    exact_rhs = [jnp.ones((1,), dtype=dtype)]
    inequality_rows = []
    inequality_rhs = []
    exact, inequalities = _target_rows(_matrix(problem.moment_map), problem.target)
    for matrix, rhs in exact:
        exact_rows.append(matrix)
        exact_rhs.append(rhs)
    for matrix, rhs in inequalities:
        inequality_rows.append(matrix)
        inequality_rhs.append(rhs)
    if problem.group_constraints is not None:
        exact, inequalities = _target_rows(
            _matrix(problem.group_constraints.group_map),
            problem.group_constraints.target,
        )
        for matrix, rhs in exact:
            exact_rows.append(matrix)
            exact_rhs.append(rhs)
        for matrix, rhs in inequalities:
            inequality_rows.append(matrix)
            inequality_rhs.append(rhs)
    return LinearProgram(
        objective,
        equality_matrix=jnp.concatenate(exact_rows),
        equality_rhs=jnp.concatenate(exact_rhs),
        inequality_matrix=(
            None if not inequality_rows else jnp.concatenate(inequality_rows)
        ),
        inequality_rhs=(None if not inequality_rhs else jnp.concatenate(inequality_rhs)),
        bounds=Bounds(
            jnp.zeros((count,), dtype=dtype),
            jnp.where(problem.mask, jnp.inf, 0.0),
        ),
        problem_id=f"{problem.problem_id}:face-certificate",
    )


def _monotone_equality_upper_bound(program, index):
    matrix = program.equality_matrix
    rhs = program.equality_rhs
    coefficient = matrix[:, index]
    nonnegative_row = jnp.all(matrix >= 0.0, axis=-1) & (coefficient > 0.0) & (rhs >= 0.0)
    nonpositive_row = jnp.all(matrix <= 0.0, axis=-1) & (coefficient < 0.0) & (rhs <= 0.0)
    positive_upper = jnp.where(
        nonnegative_row,
        rhs / jnp.where(coefficient > 0.0, coefficient, 1.0),
        jnp.inf,
    )
    negative_upper = jnp.where(
        nonpositive_row,
        (-rhs) / jnp.where(coefficient < 0.0, -coefficient, 1.0),
        jnp.inf,
    )
    return jnp.minimum(
        jnp.min(positive_upper, initial=jnp.inf),
        jnp.min(negative_upper, initial=jnp.inf),
    )


def _audit_face_coordinate(program, result, index, policy, zero_tolerance):
    canonical = program.canonical
    cone_dual = canonical.cone.project_dual(result.cone_dual)
    lower_dual = jnp.maximum(result.lower_bound_dual, 0.0)
    upper_dual = jnp.maximum(result.upper_bound_dual, 0.0)
    stationarity = (
        canonical.linear
        + canonical.constraint_matrix.T @ cone_dual
        - lower_dual
        + upper_dual
    )
    normalization_shift = -jnp.min(stationarity)
    cone_dual = cone_dual.at[0].add(normalization_shift)
    lower_dual = lower_dual + stationarity + normalization_shift
    corrected_stationarity = (
        canonical.linear
        + canonical.constraint_matrix.T @ cone_dual
        - lower_dual
        + upper_dual
    )
    scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.abs(result.objective),
            jnp.maximum(
                jnp.max(jnp.abs(canonical.linear), initial=0.0),
                jnp.max(jnp.abs(canonical.constraint_rhs), initial=0.0),
            ),
        ),
    )
    audit_tolerance = policy.termination.absolute + policy.termination.relative * scale
    finite_lower = jnp.where(
        jnp.isfinite(canonical.lower_bounds), canonical.lower_bounds, 0.0
    )
    finite_upper = jnp.where(
        jnp.isfinite(canonical.upper_bounds), canonical.upper_bounds, 0.0
    )
    linear_program_upper_bound = (
        canonical.constraint_rhs @ cone_dual
        - finite_lower @ lower_dual
        + finite_upper @ upper_dual
    )
    linear_program_upper_bound = jnp.maximum(linear_program_upper_bound, 0.0)
    dual_certified = (
        jnp.all(jnp.isfinite(cone_dual))
        & jnp.all(jnp.isfinite(lower_dual))
        & jnp.all(jnp.isfinite(upper_dual))
        & jnp.isfinite(linear_program_upper_bound)
        & jnp.all(jnp.isfinite(canonical.lower_bounds) | (lower_dual <= audit_tolerance))
        & jnp.all(jnp.isfinite(canonical.upper_bounds) | (upper_dual <= audit_tolerance))
        & (jnp.max(jnp.abs(corrected_stationarity), initial=0.0) <= audit_tolerance)
        & (canonical.cone.dual_residual(cone_dual) <= audit_tolerance)
    )
    monotone_upper_bound = _monotone_equality_upper_bound(program, index)
    maximum_upper_bound = jnp.minimum(
        jnp.where(dual_certified, linear_program_upper_bound, jnp.inf),
        monotone_upper_bound,
    )
    upper_bound_certified = jnp.isfinite(maximum_upper_bound)
    primal_certified = jnp.all(jnp.isfinite(result.primal)) & (
        result.primal_residual_norm <= audit_tolerance
    )
    forced_zero = upper_bound_certified & (maximum_upper_bound <= zero_tolerance)
    positive_witness = primal_certified & (
        result.primal[index] > zero_tolerance + audit_tolerance
    )
    certified = forced_zero | positive_witness
    witness = jnp.where(primal_certified, result.primal, 0.0)
    return maximum_upper_bound, witness, forced_zero, certified


def discover_boundary_face(problem, solver):
    """Certify forced-zero coordinates by bounded maximum-mass LPs."""
    if problem.boundary is None:
        raise ValueError("Boundary face discovery requires BoundaryFacePolicy.")
    if problem.source_points > problem.boundary.maximum_linear_programs:
        raise ValueError("Boundary face LP count exceeds the declared capacity.")
    relaxations = []
    maximum = []
    witnesses = []
    forced = []
    certified = []
    for index in range(problem.source_points):
        objective = jnp.zeros((problem.source_points,)).at[index].set(-1.0)
        program = _face_program(problem, objective)
        result = solve_convex_program(program, policy=solver).result
        upper, witness, is_forced, is_certified = _audit_face_coordinate(
            program,
            result,
            index,
            solver,
            problem.boundary.zero_tolerance,
        )
        relaxations.append(result)
        maximum.append(upper)
        witnesses.append(witness)
        forced.append(is_forced)
        certified.append(is_certified)
    maximum_weights = jnp.stack(maximum)
    certified = jnp.stack(certified)
    forced = (~problem.mask) | jnp.stack(forced)
    remaining = problem.mask & ~forced
    sole_remaining = remaining & (jnp.sum(remaining, dtype=jnp.int32) == 1)
    certified = certified | sole_remaining
    witness = jnp.sum(jnp.stack(witnesses), axis=0)
    witness = witness + sole_remaining.astype(witness.dtype)
    witness = witness / jnp.maximum(jnp.sum(witness), 1.0)
    return BoundaryFaceEvidence(
        forced,
        witness,
        maximum_weights,
        certified,
        tuple(relaxations),
    )


def calibrate_moments_conic(problem, solver=None):
    """Execute exact/interval/group relative entropy through ConicProgram."""
    if problem.subset is not None:
        raise ValueError("EqualWeightSubset requires the mixed-integer route.")
    policy = solver
    if policy is None:
        policy = ConvexSolvePolicy(
            NativeHomogeneousConic(primal_step=5e-3, dual_step=5e-3),
            termination=ConvexTermination(maximum_steps=4000, absolute=1e-7),
        )
    if not isinstance(policy, ConvexSolvePolicy):
        raise TypeError("canonical-conic solver must be ConvexSolvePolicy.")
    original_problem = problem
    face = None
    if problem.boundary is not None:
        face = discover_boundary_face(problem, policy)
        if not bool(jnp.all(face.certified | ~problem.mask)):
            raise ValueError("Boundary face discovery lacked audited LP certificates.")
        problem = eqx.tree_at(
            lambda value: value.mask,
            problem,
            problem.mask & ~face.forced_zero,
        )
    execution = solve_convex_program(_program(problem), policy=policy)
    convex = execution.result
    weights = convex.primal[: problem.source_points]
    finite_positive = jnp.isfinite(weights) & (weights > 0.0)
    log_weights = jnp.where(finite_positive, jnp.log(weights), -jnp.inf)
    achieved = problem.moment_map.mv(weights)
    if isinstance(problem.target, ExactMoments):
        residual = achieved - problem.target.values
        target_ok = jnp.max(jnp.abs(residual)) <= policy.termination.absolute
    else:
        residual = jnp.where(
            achieved < problem.target.lower,
            achieved - problem.target.lower,
            jnp.where(
                achieved > problem.target.upper,
                achieved - problem.target.upper,
                0.0,
            ),
        )
        target_ok = jnp.max(jnp.abs(residual)) <= policy.termination.absolute
    prior = jax.nn.softmax(jnp.where(problem.mask, problem.prior_log_weights, -jnp.inf))
    prior_moments = problem.moment_map.mv(prior)
    normalization = jnp.abs(jnp.sum(weights) - 1.0)
    valid = convex.successful & target_ok & (normalization <= policy.termination.absolute)
    status = jnp.where(
        valid,
        int(MomentCalibrationStatus.SUCCESS),
        int(MomentCalibrationStatus.OPTIMIZATION_FAILED),
    ).astype(jnp.int32)
    relative_entropy = jnp.sum(
        jnp.where(weights > 0.0, weights * (log_weights - jnp.log(prior)), 0.0)
    )
    diagnostics = MomentCalibrationDiagnostics(
        optimizer_status=convex.status,
        optimization=convex,
        prior_moments=prior_moments,
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
        dual_gradient_norm=convex.kkt_residual_norm,
        dual_norm=jnp.max(jnp.abs(convex.cone_dual), initial=0.0),
        relative_entropy=relative_entropy,
        effective_sample_size=weight_ess(weights, axis=0),
        active_support=jnp.sum(weights > 0.0),
        minimum_active_weight=jnp.min(jnp.where(weights > 0.0, weights, jnp.inf)),
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
        execution="canonical-conic",
        differentiation="fixed-regular-conic",
        optimizer=convex.provenance,
    )
    calibrated = MomentCalibrationResult(
        problem,
        log_weights,
        convex.cone_dual,
        achieved,
        status,
        diagnostics,
        provenance,
    )
    if face is not None:
        calibrated = eqx.tree_at(
            lambda value: (value.problem, value.diagnostics.spectrum),
            calibrated,
            (original_problem, face),
        )
    return calibrated


def implicit_calibrate_fixed_face(problem, face, **kwargs):
    """Differentiate calibration only while one certified face remains fixed."""
    if not isinstance(face, BoundaryFaceEvidence):
        raise TypeError("face must be BoundaryFaceEvidence.")
    if face.forced_zero.shape != problem.mask.shape:
        raise ValueError("Face evidence does not match the calibration support.")
    if not bool(jnp.all(face.certified | ~problem.mask)):
        raise ValueError("Every active forced-zero decision requires a certificate.")
    reduced = eqx.tree_at(
        lambda value: (value.mask, value.boundary),
        problem,
        (
            problem.mask & ~jax.lax.stop_gradient(face.forced_zero),
            None,
        ),
    )
    from ._relative_entropy import implicit_calibrate_moments

    return implicit_calibrate_moments(reduced, **kwargs)


__all__ = [
    "BoundaryFaceEvidence",
    "calibrate_moments_conic",
    "implicit_calibrate_fixed_face",
    "discover_boundary_face",
]
