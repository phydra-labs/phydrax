#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._binding import LinearSolveTemplate
from ._policies import FailurePolicy, LinearSolvePolicy
from ._problems import LinearSystem
from ._results import LinearSolveResult, LinearSolveStatus
from ._runtime import (
    _pack_rhs,
    _unpack_value,
    bind_numeric,
    prepare_template,
    solve,
)
from ._spaces import RHSLayout
from ._structured_operators import BasePlusLowRankLinearOperator


BaseNonsingularity = Literal["certified", "asserted"]


class LowRankSolveStatus(IntEnum):
    """Status for a base-plus-low-rank solve."""

    SUCCESS = 0
    BASE_SOLVE_FAILED = 1
    CORRECTION_ILL_CONDITIONED = 2
    RESIDUAL_TOLERANCE_NOT_MET = 3
    NONFINITE_OUTPUT = 4


class LowRankResourcePolicy(StrictModule):
    """Hard bounds for persistent Woodbury state and dense correction work."""

    max_rank: int = eqx.field(static=True)
    max_storage_bytes: int = eqx.field(static=True)
    max_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_rank: int = 4096,
        max_storage_bytes: int = 512 * 1024 * 1024,
        max_workspace_bytes: int = 512 * 1024 * 1024,
    ):
        values = tuple(
            int(value)
            for value in (
                max_rank,
                max_storage_bytes,
                max_workspace_bytes,
            )
        )
        if any(value < 1 for value in values):
            raise ValueError("Low-rank resource limits must be positive.")
        self.max_rank, self.max_storage_bytes, self.max_workspace_bytes = values


class LowRankSolvePolicy(StrictModule):
    """Base solve, conditioning, failure, and resource policy."""

    base: LinearSolvePolicy
    condition_limit: float = eqx.field(static=True)
    base_nonsingularity: BaseNonsingularity = eqx.field(static=True)
    failure: FailurePolicy
    resources: LowRankResourcePolicy

    def __init__(
        self,
        base: LinearSolvePolicy | None = None,
        *,
        condition_limit: float = 1e12,
        base_nonsingularity: BaseNonsingularity = "certified",
        failure: FailurePolicy | None = None,
        resources: LowRankResourcePolicy | None = None,
    ):
        base_ = LinearSolvePolicy() if base is None else base
        failure_ = FailurePolicy("error") if failure is None else failure
        resources_ = LowRankResourcePolicy() if resources is None else resources
        if not isinstance(base_, LinearSolvePolicy):
            raise TypeError("base must be a LinearSolvePolicy.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy.")
        if not isinstance(resources_, LowRankResourcePolicy):
            raise TypeError("resources must be a LowRankResourcePolicy.")
        limit = float(condition_limit)
        if not math.isfinite(limit) or limit < 1.0:
            raise ValueError("condition_limit must be finite and at least one.")
        if base_nonsingularity not in ("certified", "asserted"):
            raise ValueError("base_nonsingularity must be 'certified' or 'asserted'.")
        self.base = base_
        self.condition_limit = limit
        self.base_nonsingularity = base_nonsingularity
        self.failure = failure_
        self.resources = resources_


class LowRankCostEstimate(StrictModule):
    """Static storage and workspace estimate for a Woodbury correction."""

    dimension: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    solve_workspace_bytes_per_rhs: int = eqx.field(static=True)

    def __init__(self, dimension: int, rank: int, itemsize: int, /):
        n, r, size = int(dimension), int(rank), int(itemsize)
        self.dimension = n
        self.rank = r
        self.storage_bytes = size * (n * r + r * r + r * r + r)
        self.preparation_workspace_bytes = size * (n * r + 3 * r * r)
        self.solve_workspace_bytes_per_rhs = size * (2 * n + 3 * r)


class LowRankSolvePlan(StrictModule):
    """Immutable symbolic plan for an arbitrary-base Woodbury solve."""

    policy: LowRankSolvePolicy
    base_template: LinearSolveTemplate
    cost: LowRankCostEstimate
    operator_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        operator: BasePlusLowRankLinearOperator,
        policy: LowRankSolvePolicy,
        base_template: LinearSolveTemplate,
        cost: LowRankCostEstimate,
    ):
        self.policy = policy
        self.base_template = base_template
        self.cost = cost
        self.operator_id = operator.operator_id
        self.source_space_id = operator.source.space_id
        self.target_space_id = operator.target.space_id
        self.rank = operator.rank
        self.plan_id = canonical_fingerprint(
            {
                "kind": "low-rank-solve-plan",
                "operator": operator.operator_id,
                "source": operator.source.space_id,
                "target": operator.target.space_id,
                "rank": operator.rank,
                "base_template": base_template.template_id,
                "condition_limit": policy.condition_limit,
                "base_nonsingularity": policy.base_nonsingularity,
                "failure": policy.failure.mode,
                "resources": {
                    "max_rank": policy.resources.max_rank,
                    "max_storage_bytes": policy.resources.max_storage_bytes,
                    "max_workspace_bytes": policy.resources.max_workspace_bytes,
                },
            }
        )


class PreparedLowRankSolve(StrictModule):
    """Reusable numerical base state and Woodbury correction factorization."""

    operator: BasePlusLowRankLinearOperator
    plan: LowRankSolvePlan
    base_prepared: Any
    inverse_left_factor: Array
    correction_matrix: Array
    correction_lu: Array
    correction_pivots: Array
    correction_condition: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        operator: BasePlusLowRankLinearOperator,
        plan: LowRankSolvePlan,
        base_prepared: Any,
        inverse_left_factor: Array,
        correction_matrix: Array,
        correction_lu: Array,
        correction_pivots: Array,
        correction_condition: Array,
        numeric_version: Any,
    ):
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        self.operator = operator
        self.plan = plan
        self.base_prepared = base_prepared
        self.inverse_left_factor = jnp.asarray(inverse_left_factor)
        self.correction_matrix = jnp.asarray(correction_matrix)
        self.correction_lu = jnp.asarray(correction_lu)
        self.correction_pivots = jnp.asarray(correction_pivots)
        self.correction_condition = jnp.asarray(correction_condition)
        self.numeric_version = version
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-low-rank-solve",
                "plan": plan.plan_id,
                "operator": operator.operator_id,
                "state": "numeric",
            }
        )


class LowRankSolveDiagnostics(StrictModule):
    """Per-right-hand-side residuals plus shared correction evidence."""

    residual_norm: Array
    relative_residual: Array
    base_status: Array
    base_iterations: Array
    base_matvec_count: Array
    correction_condition: Array
    rank: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual_norm: Array,
        relative_residual: Array,
        base_status: Array,
        base_iterations: Array,
        base_matvec_count: Array,
        correction_condition: Array,
        rank: int,
    ):
        self.residual_norm = jnp.asarray(residual_norm)
        self.relative_residual = jnp.asarray(relative_residual)
        self.base_status = jnp.asarray(base_status, dtype=jnp.int32)
        self.base_iterations = jnp.asarray(base_iterations, dtype=jnp.int32)
        self.base_matvec_count = jnp.asarray(base_matvec_count, dtype=jnp.int32)
        self.correction_condition = jnp.asarray(correction_condition)
        self.rank = int(rank)


class LowRankSolveProvenance(StrictModule):
    """Static identities and dynamic versions for one specialized solve."""

    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    base_plan_id: str = eqx.field(static=True)
    base_template_id: str = eqx.field(static=True)
    base_nonsingularity: BaseNonsingularity = eqx.field(static=True)
    operator_numeric_version: Array
    base_numeric_version: Array

    def __init__(self, prepared: PreparedLowRankSolve, /):
        self.plan_id = prepared.plan.plan_id
        self.prepared_id = prepared.prepared_id
        self.operator_id = prepared.operator.operator_id
        self.base_plan_id = prepared.base_prepared.plan.plan_id
        self.base_template_id = prepared.base_prepared.template.template_id
        self.base_nonsingularity = prepared.plan.policy.base_nonsingularity
        self.operator_numeric_version = prepared.numeric_version
        self.base_numeric_version = prepared.base_prepared.numeric_version


class LowRankSolveResult(StrictModule):
    """Value, per-RHS status, diagnostics, provenance, and base solve evidence."""

    value: PyTree[Array]
    status: Array
    diagnostics: LowRankSolveDiagnostics
    provenance: LowRankSolveProvenance
    base_result: LinearSolveResult

    def __init__(
        self,
        value: PyTree[Array],
        status: Array,
        diagnostics: LowRankSolveDiagnostics,
        provenance: LowRankSolveProvenance,
        base_result: LinearSolveResult,
        /,
    ):
        self.value = value
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.base_result = base_result

    @property
    def successful(self) -> Array:
        return self.status == int(LowRankSolveStatus.SUCCESS)


def plan_low_rank_solve(
    operator: BasePlusLowRankLinearOperator,
    policy: LowRankSolvePolicy | None = None,
    /,
) -> LowRankSolvePlan:
    """Plan a bounded Woodbury solve without evaluating operator coefficients."""
    _validate_operator(operator)
    policy_ = LowRankSolvePolicy() if policy is None else policy
    if not isinstance(policy_, LowRankSolvePolicy):
        raise TypeError("policy must be a LowRankSolvePolicy or None.")
    if policy_.base_nonsingularity == "certified" and not _certifies_nonsingular(
        operator.base
    ):
        raise ValueError(
            "The base operator lacks a full-rank or positive-definite certificate; "
            "use base_nonsingularity='asserted' only when nonsingularity is known."
        )
    dtype = np.dtype(operator.left_factor.dtype)
    cost = LowRankCostEstimate(operator.source.size, operator.rank, dtype.itemsize)
    resources = policy_.resources
    failures = []
    if operator.rank > resources.max_rank:
        failures.append("rank exceeds max_rank")
    if cost.storage_bytes > resources.max_storage_bytes:
        failures.append("persistent state exceeds max_storage_bytes")
    if cost.preparation_workspace_bytes > resources.max_workspace_bytes:
        failures.append("preparation work exceeds max_workspace_bytes")
    if failures:
        raise ValueError(
            "Low-rank solve resource rejection: " + "; ".join(failures) + "."
        )
    base_problem = _base_problem(operator)
    base_template = prepare_template(base_problem, policy_.base)
    return LowRankSolvePlan(
        operator=operator,
        policy=policy_,
        base_template=base_template,
        cost=cost,
    )


def prepare_low_rank_solve(
    operator: BasePlusLowRankLinearOperator,
    policy: LowRankSolvePolicy | LowRankSolvePlan | None = None,
    /,
    *,
    numeric_version: Any = 0,
) -> PreparedLowRankSolve:
    """Bind numeric base state and factor the dense Woodbury correction."""
    plan = (
        policy
        if isinstance(policy, LowRankSolvePlan)
        else plan_low_rank_solve(operator, policy)
    )
    if not isinstance(plan, LowRankSolvePlan):
        raise TypeError("policy must be a LowRankSolvePolicy, LowRankSolvePlan, or None.")
    _validate_plan_operator(plan, operator)
    base_prepared = bind_numeric(
        plan.base_template,
        _base_problem(operator),
        numeric_version=numeric_version,
    )
    return _prepare_from_base(
        operator,
        plan,
        base_prepared,
        numeric_version=numeric_version,
    )


def refresh_low_rank_solve(
    prepared: PreparedLowRankSolve,
    operator: BasePlusLowRankLinearOperator,
    /,
) -> PreparedLowRankSolve:
    """Rebind changed coefficients while preserving the symbolic low-rank plan."""
    if not isinstance(prepared, PreparedLowRankSolve):
        raise TypeError("prepared must be a PreparedLowRankSolve.")
    _validate_plan_operator(prepared.plan, operator)
    version = prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32)
    base_prepared = bind_numeric(
        prepared.plan.base_template,
        _base_problem(operator),
        numeric_version=version,
    )
    return _prepare_from_base(
        operator,
        prepared.plan,
        base_prepared,
        numeric_version=version,
    )


def solve_low_rank(
    problem_or_prepared: BasePlusLowRankLinearOperator | PreparedLowRankSolve,
    rhs: PyTree[Any],
    policy: LowRankSolvePolicy | LowRankSolvePlan | None = None,
    /,
    *,
    rhs_layout: RHSLayout | None = None,
) -> LowRankSolveResult:
    """Solve ``(B + U C Vᴴ)x = rhs`` using reusable base and correction state."""
    if isinstance(problem_or_prepared, PreparedLowRankSolve):
        if policy is not None:
            raise ValueError("policy must be omitted when solving prepared state.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, BasePlusLowRankLinearOperator):
        prepared = prepare_low_rank_solve(problem_or_prepared, policy)
    else:
        raise TypeError(
            "Expected a BasePlusLowRankLinearOperator or PreparedLowRankSolve."
        )
    if rhs_layout is not None and not isinstance(rhs_layout, RHSLayout):
        raise TypeError("rhs_layout must be an RHSLayout or None.")

    operator = prepared.operator
    canonical_rhs, layout = _pack_rhs(
        operator.target,
        (),
        rhs,
        rhs_layout,
    )
    base_result = solve(prepared.base_prepared, rhs, rhs_layout=rhs_layout)
    base_value, base_layout = _pack_rhs(operator.source, (), base_result.value)
    if base_layout.rhs_shape != layout.rhs_shape:
        raise ValueError("The base solve changed the right-hand-side layout.")
    correction_rhs = operator.core @ (jnp.conj(operator.right_factor.T) @ base_value)
    correction = jsp.linalg.lu_solve(
        (prepared.correction_lu, prepared.correction_pivots),
        correction_rhs,
    )
    coordinates = base_value - prepared.inverse_left_factor @ correction
    value = _unpack_value(operator.source, coordinates, layout)

    applied = _operator_columns(operator, coordinates)
    residual = applied - canonical_rhs
    residual_norm = _column_norm(operator.target, residual)
    rhs_norm = _column_norm(operator.target, canonical_rhs)
    relative_residual = residual_norm / jnp.maximum(
        rhs_norm, jnp.finfo(residual_norm.dtype).tiny
    )
    base_status = jnp.asarray(base_result.status, dtype=jnp.int32).reshape(-1)
    iterations = jnp.asarray(base_result.diagnostics.iterations, dtype=jnp.int32).reshape(
        -1
    )
    matvec_count = jnp.asarray(
        base_result.diagnostics.matvec_count, dtype=jnp.int32
    ).reshape(-1)
    finite = jnp.all(jnp.isfinite(coordinates), axis=0) & jnp.isfinite(residual_norm)
    tolerance = prepared.plan.policy.base.tolerance
    converged = residual_norm <= (tolerance.absolute + tolerance.relative * rhs_norm)
    correction_valid = jnp.isfinite(prepared.correction_condition) & (
        prepared.correction_condition <= prepared.plan.policy.condition_limit
    )
    status = jnp.where(
        base_status != int(LinearSolveStatus.SUCCESS),
        int(LowRankSolveStatus.BASE_SOLVE_FAILED),
        int(LowRankSolveStatus.SUCCESS),
    )
    status = jnp.where(
        correction_valid,
        status,
        int(LowRankSolveStatus.CORRECTION_ILL_CONDITIONED),
    )
    status = jnp.where(
        converged,
        status,
        int(LowRankSolveStatus.RESIDUAL_TOLERANCE_NOT_MET),
    )
    status = jnp.where(
        finite,
        status,
        int(LowRankSolveStatus.NONFINITE_OUTPUT),
    )
    status = _restore_axes(status, layout)
    diagnostics = LowRankSolveDiagnostics(
        residual_norm=_restore_axes(residual_norm, layout),
        relative_residual=_restore_axes(relative_residual, layout),
        base_status=_restore_axes(base_status, layout),
        base_iterations=_restore_axes(iterations, layout),
        base_matvec_count=_restore_axes(matvec_count, layout),
        correction_condition=prepared.correction_condition,
        rank=operator.rank,
    )
    if prepared.plan.policy.failure.mode == "error":
        value = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                jnp.any(status != int(LowRankSolveStatus.SUCCESS)),
                "Base-plus-low-rank solve failed; inspect status-mode diagnostics.",
            ),
            value,
        )
    return LowRankSolveResult(
        value,
        status,
        diagnostics,
        LowRankSolveProvenance(prepared),
        base_result,
    )


def _prepare_from_base(
    operator: BasePlusLowRankLinearOperator,
    plan: LowRankSolvePlan,
    base_prepared: Any,
    *,
    numeric_version: Any,
) -> PreparedLowRankSolve:
    left_rhs = jax.vmap(operator.target.unflatten, in_axes=1, out_axes=1)(
        operator.left_factor
    )
    base_result = solve(base_prepared, left_rhs, rhs_layout=RHSLayout((operator.rank,)))
    inverse_left, _ = _pack_rhs(operator.source, (), base_result.value)
    inverse_left = eqx.error_if(
        inverse_left,
        jnp.any(jnp.asarray(base_result.status) != int(LinearSolveStatus.SUCCESS)),
        "The base solve failed while preparing inverse actions on the low-rank factor.",
    )
    correction_matrix = jnp.eye(operator.rank, dtype=operator.core.dtype) + (
        operator.core @ (jnp.conj(operator.right_factor.T) @ inverse_left)
    )
    correction_condition = jnp.linalg.cond(correction_matrix)
    correction_lu, correction_pivots = jsp.linalg.lu_factor(correction_matrix)
    correction_lu = eqx.error_if(
        correction_lu,
        jnp.any(~jnp.isfinite(correction_lu)),
        "The Woodbury correction factorization is non-finite.",
    )
    return PreparedLowRankSolve(
        operator=operator,
        plan=plan,
        base_prepared=base_prepared,
        inverse_left_factor=inverse_left,
        correction_matrix=correction_matrix,
        correction_lu=correction_lu,
        correction_pivots=correction_pivots,
        correction_condition=correction_condition,
        numeric_version=numeric_version,
    )


def _base_problem(operator: BasePlusLowRankLinearOperator, /) -> LinearSystem:
    return LinearSystem(
        operator.base,
        problem_id=f"{operator.operator_id}:woodbury-base",
    )


def _validate_operator(operator: BasePlusLowRankLinearOperator, /) -> None:
    if not isinstance(operator, BasePlusLowRankLinearOperator):
        raise TypeError("operator must be a BasePlusLowRankLinearOperator.")


def _validate_plan_operator(
    plan: LowRankSolvePlan,
    operator: BasePlusLowRankLinearOperator,
    /,
) -> None:
    _validate_operator(operator)
    if (
        operator.operator_id != plan.operator_id
        or operator.source.space_id != plan.source_space_id
        or operator.target.space_id != plan.target_space_id
        or operator.rank != plan.rank
    ):
        raise ValueError("Low-rank numeric binding changed symbolic operator structure.")


def _certifies_nonsingular(operator, /) -> bool:
    properties = operator.properties
    if properties.certifies("positive_definite"):
        return True
    rank = properties.rank
    return rank is not None and rank == operator.source.size


def _operator_columns(operator, coordinates: Array, /) -> Array:
    def apply(column):
        return operator.target.flatten(operator.mv(operator.source.unflatten(column)))

    return jax.vmap(apply, in_axes=1, out_axes=1)(coordinates)


def _column_norm(space, coordinates: Array, /) -> Array:
    def norm(column):
        vector = space.unflatten(column)
        squared = jnp.real(space.inner(vector, vector))
        return jnp.sqrt(jnp.maximum(squared, 0.0))

    return jax.vmap(norm, in_axes=1)(coordinates)


def _restore_axes(value: Array, layout, /) -> Array:
    return jnp.asarray(value).reshape(layout.rhs_shape)


__all__ = [
    "BaseNonsingularity",
    "LowRankCostEstimate",
    "LowRankResourcePolicy",
    "LowRankSolveDiagnostics",
    "LowRankSolvePlan",
    "LowRankSolvePolicy",
    "LowRankSolveProvenance",
    "LowRankSolveResult",
    "LowRankSolveStatus",
    "PreparedLowRankSolve",
    "plan_low_rank_solve",
    "prepare_low_rank_solve",
    "refresh_low_rank_solve",
    "solve_low_rank",
]
