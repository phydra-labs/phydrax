#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from itertools import combinations
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._bounds import Bounds
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import (
    tree_allfinite,
    tree_inner,
    tree_norm,
    tree_scale,
    tree_where,
    validate_inexact_tree,
)
from ..ein import contract
from ..linalg import FailurePolicy, SmallLinearSolvePlan, solve_small_linear
from ._programming import (
    ConvexSolvePolicy,
    ConvexTermination,
    QuadraticProgram,
    solve_quadratic_program,
)


ConflictFreeUpdateFailureMode = Literal["status", "error"]


class ConflictFreeUpdateStatus(IntEnum):
    """Terminal state of one optimizer-direction alignment."""

    PROJECTED = 0
    ALREADY_FEASIBLE = 1
    ZERO_PROPOSAL = 2
    NO_EFFECTIVE_OBJECTIVES = 3
    NONFINITE = 4
    INVALID_METRIC = 5
    DUAL_SOLVE_FAILED = 6
    POSTCHECK_FAILED = 7


class ConflictFreeUpdatePolicy(StrictModule, NonTrainableState):
    """Experimental contract for conflict-free optimizer-direction projection."""

    dual_solve_policy: ConvexSolvePolicy
    minimum_norm: float = eqx.field(static=True)
    feasibility_tolerance: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    failure: ConflictFreeUpdateFailureMode = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        dual_solve_policy: ConvexSolvePolicy | None = None,
        minimum_norm: float = 1e-12,
        feasibility_tolerance: float = 1e-10,
        maximum_condition: float = 1e12,
        failure: ConflictFreeUpdateFailureMode = "status",
    ):
        norm = float(minimum_norm)
        tolerance = float(feasibility_tolerance)
        condition = float(maximum_condition)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("minimum_norm must be finite and strictly positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("feasibility_tolerance must be finite and nonnegative.")
        if not np.isfinite(condition) or condition <= 1.0:
            raise ValueError("maximum_condition must be finite and greater than one.")
        if failure not in ("status", "error"):
            raise ValueError("failure must be 'status' or 'error'.")
        if dual_solve_policy is None:
            solver_tolerance = max(tolerance, 1e-10)
            dual = ConvexSolvePolicy(
                termination=ConvexTermination(
                    absolute=solver_tolerance,
                    relative=0.0,
                    primal_infeasible=solver_tolerance,
                    dual_infeasible=solver_tolerance,
                    maximum_steps=100,
                ),
                regularization=0.0,
                failure=FailurePolicy("status"),
            )
        else:
            dual = dual_solve_policy
        if not isinstance(dual, ConvexSolvePolicy):
            raise TypeError("dual_solve_policy must be a ConvexSolvePolicy or None.")
        if dual.failure.mode != "status":
            raise ValueError("dual_solve_policy must use status failure handling.")
        if dual.regularization != 0.0:
            raise ValueError(
                "dual_solve_policy must not regularize the exact projection."
            )
        self.dual_solve_policy = dual
        self.minimum_norm = norm
        self.feasibility_tolerance = tolerance
        self.maximum_condition = condition
        self.failure = failure
        self.policy_id = canonical_fingerprint(
            {
                "kind": "conflict-free-update-policy",
                "dual_solve_policy_id": dual.policy_id,
                "minimum_norm": norm,
                "feasibility_tolerance": tolerance,
                "maximum_condition": condition,
                "failure": failure,
            }
        )


class ConflictFreeUpdateResult(StrictModule):
    """Aligned descent direction and complete local projection evidence."""

    direction: PyTree[Array]
    gradient_norms: Array
    gradient_cosine_matrix: Array
    raw_projections: Array
    aligned_projections: Array
    raw_cosines: Array
    aligned_cosines: Array
    active: Array
    stationary: Array
    raw_conflicts: Array
    aligned_conflicts: Array
    multipliers: Array
    proposal_norm: Array
    direction_norm: Array
    correction_norm: Array
    relative_correction: Array
    metric_correction_norm: Array
    active_constraint_count: Array
    dual_feasibility_violation: Array
    complementarity_residual: Array
    kkt_residual_norm: Array
    projected: Array
    pareto_stationary: Array
    successful: Array
    status: Array
    solver_method: str = eqx.field(static=True)
    metric_kind: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)


class ConflictFreeUpdateStatistics(StrictModule):
    """Checkpointable aggregate gradient-update mismatch evidence."""

    steps: Array
    gradient_conflict_steps: Array
    constructed_conflict_steps: Array
    proposal_conflict_steps: Array
    applied_conflict_steps: Array
    projected_steps: Array
    zero_proposal_steps: Array
    pareto_stationary_steps: Array
    correction_norm_sum: Array
    relative_correction_sum: Array
    metric_correction_norm_sum: Array
    maximum_kkt_residual: Array

    @classmethod
    def zeros(cls, dtype: Any = jnp.float32) -> ConflictFreeUpdateStatistics:
        dtype_ = jnp.dtype(dtype)
        zero_count = jnp.asarray(0, dtype=jnp.int32)
        zero_value = jnp.asarray(0.0, dtype=dtype_)
        return cls(
            zero_count,
            zero_count,
            zero_count,
            zero_count,
            zero_count,
            zero_count,
            zero_count,
            zero_count,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
        )

    def update(
        self,
        result: ConflictFreeUpdateResult,
        /,
        *,
        gradient_conflict: Any,
        constructed_conflict: Any,
    ) -> ConflictFreeUpdateStatistics:
        if not isinstance(result, ConflictFreeUpdateResult):
            raise TypeError("result must be a ConflictFreeUpdateResult.")
        integer = jnp.asarray(1, dtype=self.steps.dtype)
        gradient = jnp.asarray(gradient_conflict, dtype=bool)
        constructed = jnp.asarray(constructed_conflict, dtype=bool)
        return ConflictFreeUpdateStatistics(
            self.steps + integer,
            self.gradient_conflict_steps + gradient.astype(self.steps.dtype),
            self.constructed_conflict_steps + constructed.astype(self.steps.dtype),
            self.proposal_conflict_steps
            + jnp.any(result.raw_conflicts).astype(self.steps.dtype),
            self.applied_conflict_steps
            + jnp.any(result.aligned_conflicts).astype(self.steps.dtype),
            self.projected_steps + result.projected.astype(self.steps.dtype),
            self.zero_proposal_steps
            + (result.status == int(ConflictFreeUpdateStatus.ZERO_PROPOSAL)).astype(
                self.steps.dtype
            ),
            self.pareto_stationary_steps
            + result.pareto_stationary.astype(self.steps.dtype),
            self.correction_norm_sum + result.correction_norm,
            self.relative_correction_sum + result.relative_correction,
            self.metric_correction_norm_sum + result.metric_correction_norm,
            jnp.maximum(self.maximum_kkt_residual, result.kkt_residual_norm),
        )

    def _rate(self, count: Array, /) -> Array:
        denominator = jnp.maximum(self.steps, 1).astype(self.correction_norm_sum.dtype)
        return count.astype(self.correction_norm_sum.dtype) / denominator

    @property
    def gradient_conflict_rate(self) -> Array:
        return self._rate(self.gradient_conflict_steps)

    @property
    def constructed_conflict_rate(self) -> Array:
        return self._rate(self.constructed_conflict_steps)

    @property
    def proposal_conflict_rate(self) -> Array:
        return self._rate(self.proposal_conflict_steps)

    @property
    def applied_conflict_rate(self) -> Array:
        return self._rate(self.applied_conflict_steps)

    @property
    def projection_rate(self) -> Array:
        return self._rate(self.projected_steps)

    @property
    def mean_correction_norm(self) -> Array:
        denominator = jnp.maximum(self.steps, 1).astype(self.correction_norm_sum.dtype)
        return self.correction_norm_sum / denominator

    @property
    def mean_relative_correction(self) -> Array:
        denominator = jnp.maximum(self.steps, 1).astype(self.correction_norm_sum.dtype)
        return self.relative_correction_sum / denominator

    @property
    def mean_metric_correction_norm(self) -> Array:
        denominator = jnp.maximum(self.steps, 1).astype(self.correction_norm_sum.dtype)
        return self.metric_correction_norm_sum / denominator


def _validate_congruent(
    reference: PyTree[Array],
    values: Sequence[PyTree[Array]],
    /,
    *,
    name: str,
) -> None:
    structure = jax.tree.structure(reference)
    shapes = tuple(leaf.shape for leaf in jax.tree.leaves(reference))
    if any(jax.tree.structure(value) != structure for value in values):
        raise ValueError(f"{name} must have congruent PyTree structures.")
    if any(
        tuple(leaf.shape for leaf in jax.tree.leaves(value)) != shapes for value in values
    ):
        raise ValueError(f"{name} leaves must have congruent shapes.")


def _tree_add(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def _tree_subtract(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x - y, left, right)


def _tree_zeros_like(vector: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(jnp.zeros_like, vector)


def _metric_inverse_apply(
    vector: PyTree[Array],
    metric_diagonal: PyTree[Array] | None,
    /,
) -> PyTree[Array]:
    if metric_diagonal is None:
        return vector
    return jax.tree.map(lambda value, diagonal: value / diagonal, vector, metric_diagonal)


def _metric_norm(
    vector: PyTree[Array],
    metric_diagonal: PyTree[Array] | None,
    /,
) -> Array:
    if metric_diagonal is None:
        return tree_norm(vector)
    squared = jax.tree.leaves(
        jax.tree.map(
            lambda value, diagonal: jnp.real(jnp.vdot(value, diagonal * value)),
            vector,
            metric_diagonal,
        )
    )
    total = squared[0]
    for value in squared[1:]:
        total = total + value
    return jnp.sqrt(jnp.maximum(total, 0.0))


def _small_active_set_dual(
    matrix: Array,
    linear: Array,
    /,
    *,
    tolerance: Array,
    policy: ConflictFreeUpdatePolicy,
) -> tuple[Array, Array]:
    count = int(linear.shape[0])
    zero = jnp.zeros_like(linear)
    empty_feasible = jnp.all(linear >= -tolerance)
    best = zero
    best_objective = jnp.where(empty_feasible, jnp.asarray(0.0, linear.dtype), jnp.inf)
    best_valid = empty_feasible
    for size in range(1, count + 1):
        plan = SmallLinearSolvePlan(
            size,
            singular_tolerance=max(policy.feasibility_tolerance, 1e-12),
            maximum_condition=policy.maximum_condition,
        )
        for subset in combinations(range(count), size):
            indices = jnp.asarray(subset, dtype=jnp.int32)
            submatrix = matrix[indices[:, None], indices[None, :]]
            sublinear = linear[indices]
            solve = solve_small_linear(plan, submatrix, -sublinear)
            candidate = zero.at[indices].set(jnp.maximum(solve.value, 0.0))
            gradient = linear + contract("ij,j->i", matrix, candidate)
            feasible = (
                solve.successful
                & jnp.all(solve.value >= -tolerance)
                & jnp.all(gradient >= -tolerance)
                & jnp.all(jnp.isfinite(candidate))
            )
            objective = 0.5 * contract(
                "i,ij,j->", candidate, matrix, candidate
            ) + contract("i,i->", linear, candidate)
            select = feasible & (~best_valid | (objective < best_objective))
            best = jnp.where(select, candidate, best)
            best_objective = jnp.where(select, objective, best_objective)
            best_valid = best_valid | feasible
    return best, best_valid


def _dual_solution(
    matrix: Array,
    linear: Array,
    /,
    *,
    tolerance: Array,
    policy: ConflictFreeUpdatePolicy,
) -> tuple[Array, Array, str]:
    count = int(linear.shape[0])
    if count <= 3:
        value, successful = _small_active_set_dual(
            matrix,
            linear,
            tolerance=tolerance,
            policy=policy,
        )
        return value, successful, "exact-active-set"
    problem = QuadraticProgram(
        matrix,
        linear,
        bounds=Bounds(0.0, jnp.inf),
        problem_id="conflict-free-update-dual",
        convexity_evidence="gradient-gram",
    )
    solved = solve_quadratic_program(problem, policy=policy.dual_solve_policy)
    return solved.primal, solved.successful, solved.method


def project_conflict_free_direction(
    proposal: PyTree[Any],
    objective_gradients: Sequence[PyTree[Any]],
    /,
    *,
    active: Any | None = None,
    metric_diagonal: PyTree[Any] | None = None,
    policy: ConflictFreeUpdatePolicy | None = None,
) -> ConflictFreeUpdateResult:
    """Project a positive descent proposal onto every active objective halfspace."""

    proposal_ = validate_inexact_tree(proposal, name="proposal")
    gradients = tuple(
        validate_inexact_tree(value, name=f"objective gradient {index}")
        for index, value in enumerate(objective_gradients)
    )
    if not gradients:
        raise ValueError("At least one objective gradient is required.")
    _validate_congruent(
        proposal_,
        gradients,
        name="Proposal and objective gradients",
    )
    resolved = ConflictFreeUpdatePolicy() if policy is None else policy
    if not isinstance(resolved, ConflictFreeUpdatePolicy):
        raise TypeError("policy must be a ConflictFreeUpdatePolicy.")
    count = len(gradients)
    active_mask = (
        jnp.ones((count,), dtype=bool)
        if active is None
        else jnp.asarray(active, dtype=bool)
    )
    if active_mask.shape != (count,):
        raise ValueError("active must contain one Boolean per objective gradient.")

    metric: PyTree[Array] | None
    if metric_diagonal is None:
        metric = None
        metric_valid = jnp.asarray(True)
        metric_kind = "euclidean"
    else:
        metric = validate_inexact_tree(
            metric_diagonal,
            name="metric diagonal",
            real=True,
        )
        _validate_congruent(proposal_, (metric,), name="Proposal and metric diagonal")
        metric_checks = tuple(
            jnp.all(jnp.isfinite(leaf) & (leaf > 0.0)) for leaf in jax.tree.leaves(metric)
        )
        metric_valid = metric_checks[0]
        for check in metric_checks[1:]:
            metric_valid = metric_valid & check
        metric = jax.tree.map(
            lambda leaf: jnp.where(jnp.isfinite(leaf) & (leaf > 0.0), leaf, 1.0),
            metric,
        )
        metric_kind = "diagonal"

    norms = jnp.stack(tuple(tree_norm(value) for value in gradients))
    gradient_finite = jnp.stack(tuple(tree_allfinite(value) for value in gradients))
    proposal_finite = tree_allfinite(proposal_)
    active_input_finite = jnp.all(~active_mask | gradient_finite)
    data_finite = proposal_finite & active_input_finite
    stationary = active_mask & gradient_finite & (norms <= resolved.minimum_norm)
    effective = active_mask & gradient_finite & ~stationary
    effective_count = jnp.sum(effective.astype(jnp.int32))
    safe_norms = jnp.where(effective, norms, jnp.ones_like(norms))
    normalized = tuple(
        jax.tree.map(
            lambda leaf, index=index: jnp.where(
                effective[index],
                leaf / safe_norms[index],
                jnp.zeros_like(leaf),
            ),
            value,
        )
        for index, value in enumerate(gradients)
    )

    proposal_norm = tree_norm(proposal_)
    usable_proposal = proposal_finite & (proposal_norm > resolved.minimum_norm)
    unit_proposal = tree_scale(
        jnp.where(usable_proposal, 1.0 / proposal_norm, 0.0),
        proposal_,
    )
    raw_cosines = jnp.stack(
        tuple(tree_inner(value, unit_proposal) for value in normalized)
    )
    effective_tolerance = jnp.maximum(
        jnp.asarray(resolved.feasibility_tolerance, dtype=raw_cosines.dtype),
        jnp.asarray(float(8 * count), dtype=raw_cosines.dtype)
        * jnp.finfo(raw_cosines.dtype).eps,
    )
    raw_conflicts = effective & (raw_cosines < -effective_tolerance)
    needs_projection = (
        data_finite
        & metric_valid
        & (effective_count > 0)
        & usable_proposal
        & jnp.any(raw_conflicts)
    )

    cosine_matrix = jnp.stack(
        tuple(
            jnp.stack(tuple(tree_inner(left, right) for right in normalized))
            for left in normalized
        )
    )
    inverse_normalized = tuple(
        _metric_inverse_apply(value, metric) for value in normalized
    )
    gram = (
        cosine_matrix
        if metric is None
        else jnp.stack(
            tuple(
                jnp.stack(tuple(tree_inner(left, right) for right in inverse_normalized))
                for left in normalized
            )
        )
    )
    inactive = ~effective
    gram = gram + jnp.diag(inactive.astype(gram.dtype))
    dual_linear = jnp.where(effective, raw_cosines, 0.0)

    def solve_dual(_: None) -> tuple[Array, Array]:
        multipliers_, successful_, _ = _dual_solution(
            gram,
            dual_linear,
            tolerance=effective_tolerance,
            policy=resolved,
        )
        return multipliers_, successful_

    def skip_dual(_: None) -> tuple[Array, Array]:
        return jnp.zeros_like(dual_linear), jnp.asarray(True)

    multipliers, solver_successful = jax.lax.cond(
        needs_projection,
        solve_dual,
        skip_dual,
        operand=None,
    )
    correction_unit = tree_scale(0.0, proposal_)
    for multiplier, normal in zip(multipliers, inverse_normalized, strict=True):
        correction_unit = _tree_add(
            correction_unit,
            tree_scale(multiplier, normal),
        )
    projected_unit = _tree_add(unit_proposal, correction_unit)
    projected_direction = tree_scale(proposal_norm, projected_unit)
    attempted_direction = tree_where(needs_projection, projected_direction, proposal_)
    attempted_norm = tree_norm(attempted_direction)
    attempted_finite = tree_allfinite(attempted_direction)
    attempted_unit = tree_scale(
        jnp.where(attempted_norm > resolved.minimum_norm, 1.0 / attempted_norm, 0.0),
        attempted_direction,
    )
    attempted_cosines = jnp.stack(
        tuple(tree_inner(value, attempted_unit) for value in normalized)
    )
    postcheck = attempted_finite & jnp.all(
        ~effective | (attempted_cosines >= -effective_tolerance)
    )

    no_effective = data_finite & metric_valid & (effective_count == 0)
    zero_proposal = data_finite & metric_valid & (effective_count > 0) & ~usable_proposal
    already_feasible = (
        data_finite
        & metric_valid
        & (effective_count > 0)
        & usable_proposal
        & ~jnp.any(raw_conflicts)
    )
    projected_successfully = needs_projection & solver_successful & postcheck
    successful = no_effective | zero_proposal | already_feasible | projected_successfully
    zero_direction = _tree_zeros_like(proposal_)
    direction = tree_where(successful, attempted_direction, zero_direction)
    if resolved.failure == "error":
        direction = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                ~successful,
                "Optimizer proposal could not be projected onto the objective cone.",
            ),
            direction,
        )

    direction_norm = tree_norm(direction)
    direction_unit = tree_scale(
        jnp.where(direction_norm > resolved.minimum_norm, 1.0 / direction_norm, 0.0),
        direction,
    )
    aligned_cosines = jnp.stack(
        tuple(tree_inner(value, direction_unit) for value in normalized)
    )
    raw_dimensional = jnp.stack(
        tuple(tree_inner(value, proposal_) for value in gradients)
    )
    aligned_dimensional = jnp.stack(
        tuple(tree_inner(value, direction) for value in gradients)
    )
    raw_projections = jnp.where(active_mask & gradient_finite, raw_dimensional, 0.0)
    aligned_projections = jnp.where(
        active_mask & gradient_finite,
        aligned_dimensional,
        0.0,
    )
    aligned_conflicts = effective & (aligned_cosines < -effective_tolerance)
    correction = _tree_subtract(direction, proposal_)
    correction_norm = tree_norm(correction)
    relative_correction = correction_norm / jnp.maximum(
        proposal_norm,
        jnp.asarray(resolved.minimum_norm, dtype=proposal_norm.dtype),
    )
    metric_correction_norm = _metric_norm(correction, metric)

    dual_gradient = dual_linear + contract("ij,j->i", gram, multipliers)
    multiplier_violation = jnp.max(jnp.maximum(-multipliers, 0.0))
    gradient_violation = jnp.max(jnp.maximum(-dual_gradient, 0.0))
    dual_feasibility_violation = jnp.maximum(
        multiplier_violation,
        gradient_violation,
    )
    complementarity_residual = jnp.abs(contract("i,i->", multipliers, dual_gradient))
    kkt_residual_norm = jnp.maximum(
        dual_feasibility_violation,
        complementarity_residual,
    )
    pareto_stationary = (
        successful & (effective_count > 0) & (direction_norm <= resolved.minimum_norm)
    )
    status = jnp.where(
        ~data_finite,
        int(ConflictFreeUpdateStatus.NONFINITE),
        jnp.where(
            ~metric_valid,
            int(ConflictFreeUpdateStatus.INVALID_METRIC),
            jnp.where(
                no_effective,
                int(ConflictFreeUpdateStatus.NO_EFFECTIVE_OBJECTIVES),
                jnp.where(
                    zero_proposal,
                    int(ConflictFreeUpdateStatus.ZERO_PROPOSAL),
                    jnp.where(
                        already_feasible,
                        int(ConflictFreeUpdateStatus.ALREADY_FEASIBLE),
                        jnp.where(
                            ~solver_successful,
                            int(ConflictFreeUpdateStatus.DUAL_SOLVE_FAILED),
                            jnp.where(
                                ~postcheck,
                                int(ConflictFreeUpdateStatus.POSTCHECK_FAILED),
                                int(ConflictFreeUpdateStatus.PROJECTED),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    solver_method = (
        "exact-active-set" if count <= 3 else resolved.dual_solve_policy.method.method_id
    )
    return ConflictFreeUpdateResult(
        direction,
        norms,
        cosine_matrix,
        raw_projections,
        aligned_projections,
        raw_cosines,
        aligned_cosines,
        active_mask,
        stationary,
        raw_conflicts,
        aligned_conflicts,
        multipliers,
        proposal_norm,
        direction_norm,
        correction_norm,
        relative_correction,
        metric_correction_norm,
        jnp.sum((multipliers > effective_tolerance).astype(jnp.int32)),
        dual_feasibility_violation,
        complementarity_residual,
        kkt_residual_norm,
        needs_projection & successful,
        pareto_stationary,
        successful,
        status.astype(jnp.int32),
        solver_method,
        metric_kind,
        resolved.policy_id,
    )


__all__ = [
    "ConflictFreeUpdateFailureMode",
    "ConflictFreeUpdatePolicy",
    "ConflictFreeUpdateResult",
    "ConflictFreeUpdateStatistics",
    "ConflictFreeUpdateStatus",
    "project_conflict_free_direction",
]
