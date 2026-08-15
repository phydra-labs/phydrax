#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._materialization import MaterializationPolicy, materialize
from ._operators import AbstractLinearOperator, adjoint
from ._policies import FailurePolicy, RankPolicy
from ._spaces import (
    _coordinate_dtype,
    _coordinate_pairing_matrix,
    AbstractVectorSpace,
)


SVDTarget: TypeAlias = Literal["largest", "smallest"]
SVDDifferentiationMode: TypeAlias = Literal["none", "singular-values"]


class SVDProblem(StrictModule):
    """One unbatched linear map with explicit source and target Hilbert pairings."""

    operator: AbstractLinearOperator
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        problem_id: str | None = None,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape:
            raise ValueError("SVDProblem requires an unbatched operator.")
        source_dtype = _coordinate_dtype(operator.source)
        target_dtype = _coordinate_dtype(operator.target)
        if source_dtype != target_dtype or not np.issubdtype(source_dtype, np.inexact):
            raise TypeError(
                "SVD source and target must share one real or complex coordinate dtype."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "svd-problem",
                    "operator": operator.operator_id,
                    "source": operator.source.space_id,
                    "target": operator.target.space_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.operator = operator
        self.problem_id = identifier

    @property
    def maximum_rank(self) -> int:
        return min(self.operator.source.size, self.operator.target.size)


class DenseSVD(StrictModule):
    """Pure-JAX dense singular value decomposition."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "dense-svd"


class SVDTolerancePolicy(StrictModule):
    residual: float = eqx.field(static=True)
    orthogonality: float = eqx.field(static=True)

    def __init__(self, *, residual: float = 1e-7, orthogonality: float = 1e-7):
        values = float(residual), float(orthogonality)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("SVD tolerances must be finite and non-negative.")
        self.residual, self.orthogonality = values


class SVDResourcePolicy(StrictModule):
    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    operator_matvecs: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        preparation_bytes: int = 512 * 1024 * 1024,
        workspace_bytes: int = 512 * 1024 * 1024,
        operator_matvecs: int = 1_000_000,
    ):
        values = (
            int(preparation_bytes),
            int(workspace_bytes),
            int(operator_matvecs),
        )
        if any(value < 0 for value in values):
            raise ValueError("SVD resource budgets must be non-negative.")
        self.preparation_bytes, self.workspace_bytes, self.operator_matvecs = values


class SVDSolvePolicy(StrictModule):
    method: DenseSVD
    count: int = eqx.field(static=True)
    which: SVDTarget = eqx.field(static=True)
    tolerance: SVDTolerancePolicy
    rank: RankPolicy
    materialization: MaterializationPolicy
    resources: SVDResourcePolicy
    differentiation: SVDDifferentiationMode = eqx.field(static=True)
    failure: FailurePolicy

    def __init__(
        self,
        method: DenseSVD | None = None,
        /,
        *,
        count: int = 1,
        which: SVDTarget = "largest",
        tolerance: SVDTolerancePolicy | None = None,
        rank: RankPolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        resources: SVDResourcePolicy | None = None,
        differentiation: SVDDifferentiationMode = "none",
        failure: FailurePolicy | None = None,
    ):
        method_ = DenseSVD() if method is None else method
        if not isinstance(method_, DenseSVD):
            raise TypeError("method must be DenseSVD.")
        count_ = int(count)
        if count_ < 1:
            raise ValueError("SVD count must be positive.")
        if which not in ("largest", "smallest"):
            raise ValueError("SVD target must be 'largest' or 'smallest'.")
        if differentiation not in ("none", "singular-values"):
            raise ValueError("SVD differentiation must be 'none' or 'singular-values'.")
        tolerance_ = SVDTolerancePolicy() if tolerance is None else tolerance
        rank_ = RankPolicy() if rank is None else rank
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        resources_ = SVDResourcePolicy() if resources is None else resources
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(tolerance_, SVDTolerancePolicy):
            raise TypeError("tolerance must be an SVDTolerancePolicy.")
        if not isinstance(rank_, RankPolicy):
            raise TypeError("rank must be a RankPolicy.")
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        if not isinstance(resources_, SVDResourcePolicy):
            raise TypeError("resources must be an SVDResourcePolicy.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy.")
        self.method = method_
        self.count = count_
        self.which = which
        self.tolerance = tolerance_
        self.rank = rank_
        self.materialization = materialization_
        self.resources = resources_
        self.differentiation = differentiation
        self.failure = failure_


class SVDCostEstimate(StrictModule):
    storage_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    apply_workspace_bytes: int = eqx.field(static=True)
    operator_matvec_count: int = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(
        self,
        storage_bytes: int,
        preparation_workspace_bytes: int,
        apply_workspace_bytes: int,
        operator_matvec_count: int,
        accepted: bool,
        reason: str,
        /,
    ):
        values = tuple(
            int(value)
            for value in (
                storage_bytes,
                preparation_workspace_bytes,
                apply_workspace_bytes,
                operator_matvec_count,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("SVD cost estimates must be non-negative.")
        reason_ = str(reason)
        if not reason_:
            raise ValueError("SVD cost reason must be non-empty.")
        (
            self.storage_bytes,
            self.preparation_workspace_bytes,
            self.apply_workspace_bytes,
            self.operator_matvec_count,
        ) = values
        self.accepted = bool(accepted)
        self.reason = reason_


class SVDSolvePlan(StrictModule):
    problem_id: str = eqx.field(static=True)
    policy: SVDSolvePolicy
    cost: SVDCostEstimate
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: SVDProblem,
        policy: SVDSolvePolicy,
        cost: SVDCostEstimate,
        /,
    ):
        if policy.count > problem.maximum_rank:
            raise ValueError("Requested SVD count exceeds min(target size, source size).")
        if not cost.accepted:
            raise ValueError(f"Dense SVD is infeasible: {cost.reason}.")
        self.problem_id = problem.problem_id
        self.policy = policy
        self.cost = cost
        self.plan_id = canonical_fingerprint(
            {
                "kind": "svd-solve-plan",
                "problem": problem.problem_id,
                "operator": problem.operator.operator_id,
                "source": problem.operator.source.space_id,
                "target": problem.operator.target.space_id,
                "method": policy.method.name,
                "count": policy.count,
                "which": policy.which,
                "rank_cutoff": policy.rank.relative_cutoff,
                "require_full_rank": policy.rank.require_full_rank,
                "differentiation": policy.differentiation,
                "failure": policy.failure.mode,
                "tolerance": {
                    "residual": policy.tolerance.residual,
                    "orthogonality": policy.tolerance.orthogonality,
                },
                "materialization": {
                    "max_entries": policy.materialization.max_entries,
                    "max_bytes": policy.materialization.max_bytes,
                },
                "resources": {
                    "preparation_bytes": policy.resources.preparation_bytes,
                    "workspace_bytes": policy.resources.workspace_bytes,
                    "operator_matvecs": policy.resources.operator_matvecs,
                },
                "cost": {
                    "storage_bytes": cost.storage_bytes,
                    "preparation_workspace_bytes": cost.preparation_workspace_bytes,
                    "apply_workspace_bytes": cost.apply_workspace_bytes,
                    "operator_matvec_count": cost.operator_matvec_count,
                },
            }
        )


class DenseSVDState(StrictModule):
    reduced_operator: Array
    source_factor: Array
    target_factor: Array

    def __init__(
        self,
        reduced_operator: Array,
        source_factor: Array,
        target_factor: Array,
        /,
    ):
        reduced = jnp.asarray(reduced_operator)
        source = jnp.asarray(source_factor)
        target = jnp.asarray(target_factor)
        if reduced.ndim != 2:
            raise ValueError("reduced_operator must be a matrix.")
        if source.shape != (reduced.shape[1], reduced.shape[1]):
            raise ValueError("source_factor shape does not match reduced_operator.")
        if target.shape != (reduced.shape[0], reduced.shape[0]):
            raise ValueError("target_factor shape does not match reduced_operator.")
        if source.dtype != reduced.dtype or target.dtype != reduced.dtype:
            raise TypeError("Dense SVD state arrays must share one dtype.")
        self.reduced_operator = reduced
        self.source_factor = source
        self.target_factor = target


class PreparedSVDSolve(StrictModule):
    problem: SVDProblem
    plan: SVDSolvePlan
    state: DenseSVDState
    numeric_version: Array

    def __init__(
        self,
        problem: SVDProblem,
        plan: SVDSolvePlan,
        state: DenseSVDState,
        /,
        *,
        numeric_version: Any = 0,
    ):
        if not isinstance(problem, SVDProblem):
            raise TypeError("problem must be an SVDProblem.")
        if not isinstance(plan, SVDSolvePlan):
            raise TypeError("plan must be an SVDSolvePlan.")
        if not isinstance(state, DenseSVDState):
            raise TypeError("state must be a DenseSVDState.")
        if problem.problem_id != plan.problem_id:
            raise ValueError("Prepared SVD problem and plan IDs must match.")
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        self.problem = problem
        self.plan = plan
        self.state = state
        self.numeric_version = version


class SVDSolveStatus(IntEnum):
    SUCCESS = 0
    RESIDUAL_TOLERANCE_NOT_MET = 1
    RANK_DEFICIENT = 2
    NONFINITE_OUTPUT = 3
    DIFFERENTIATION_REJECTED = 4


class SVDSolveDiagnostics(StrictModule):
    left_residual_norms: Array
    right_residual_norms: Array
    relative_residuals: Array
    left_orthogonality_error: Array
    right_orthogonality_error: Array
    isolation_gaps: Array
    converged: Array
    numerical_rank: Array
    operator_matvec_count: Array
    adjoint_matvec_count: Array


class SVDSolveProvenance(StrictModule):
    method: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    which: SVDTarget = eqx.field(static=True)
    differentiation: SVDDifferentiationMode = eqx.field(static=True)
    numeric_version: Array


class SVDSolveResult(StrictModule):
    singular_values: Array
    left_vectors: PyTree[Array]
    right_vectors: PyTree[Array]
    status: Array
    converged: Array
    diagnostics: SVDSolveDiagnostics
    provenance: SVDSolveProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SVDSolveStatus.SUCCESS)

    @property
    def numerical_rank(self) -> Array:
        return self.diagnostics.numerical_rank


class _SVDNumerics(tuple):
    __slots__ = ()

    def __new__(
        cls,
        values,
        left,
        right,
        left_residuals,
        right_residuals,
        relative_residuals,
        left_orthogonality,
        right_orthogonality,
        isolation_gaps,
        converged,
        numerical_rank,
        rank_cutoff,
    ):
        return tuple.__new__(
            cls,
            (
                values,
                left,
                right,
                left_residuals,
                right_residuals,
                relative_residuals,
                left_orthogonality,
                right_orthogonality,
                isolation_gaps,
                converged,
                numerical_rank,
                rank_cutoff,
            ),
        )


def plan_svd(
    problem: SVDProblem,
    policy: SVDSolvePolicy | None = None,
    /,
) -> SVDSolvePlan:
    if not isinstance(problem, SVDProblem):
        raise TypeError("problem must be an SVDProblem.")
    policy_ = SVDSolvePolicy() if policy is None else policy
    if not isinstance(policy_, SVDSolvePolicy):
        raise TypeError("policy must be an SVDSolvePolicy.")
    if policy_.count > problem.maximum_rank:
        raise ValueError("Requested SVD count exceeds min(target size, source size).")
    operator = problem.operator
    rows, columns = operator.target.size, operator.source.size
    itemsize = _coordinate_dtype(operator.source).itemsize
    operator_entries = rows * columns
    operator_bytes = operator_entries * itemsize
    source_bytes = columns * columns * itemsize
    target_bytes = rows * rows * itemsize
    storage = operator_bytes + source_bytes + target_bytes
    preparation = 4 * storage
    workspace = 3 * storage
    failures: list[str] = []
    if operator_entries > policy_.materialization.max_entries:
        failures.append(
            f"dense entries {operator_entries} exceed materialization limit "
            f"{policy_.materialization.max_entries}"
        )
    if operator_bytes > policy_.materialization.max_bytes:
        failures.append(
            f"dense bytes {operator_bytes} exceed materialization limit "
            f"{policy_.materialization.max_bytes}"
        )
    checks = (
        (preparation, policy_.resources.preparation_bytes, "preparation bytes"),
        (workspace, policy_.resources.workspace_bytes, "workspace bytes"),
        (columns, policy_.resources.operator_matvecs, "operator matvecs"),
    )
    failures.extend(
        f"{name} estimate {required} exceeds budget {budget}"
        for required, budget, name in checks
        if required > budget
    )
    cost = SVDCostEstimate(
        storage,
        preparation,
        workspace,
        columns,
        not failures,
        "dense pairing-aware SVD fits declared budgets"
        if not failures
        else "; ".join(failures),
    )
    return SVDSolvePlan(problem, policy_, cost)


def prepare_svd(
    problem: SVDProblem,
    policy: SVDSolvePolicy | SVDSolvePlan | None = None,
    /,
) -> PreparedSVDSolve:
    if isinstance(policy, SVDSolvePlan):
        selected_plan = policy
        if selected_plan.problem_id != problem.problem_id:
            raise ValueError("SVD plan and problem IDs must match.")
        replanned = plan_svd(problem, selected_plan.policy)
        if replanned.plan_id != selected_plan.plan_id:
            raise ValueError("SVD plan does not match the problem structure.")
    else:
        selected_plan = plan_svd(problem, policy)
    return PreparedSVDSolve(
        problem,
        selected_plan,
        _prepare_dense_svd_state(problem, selected_plan),
    )


def refresh_svd(
    prepared: PreparedSVDSolve,
    problem: SVDProblem,
    /,
) -> PreparedSVDSolve:
    if not isinstance(prepared, PreparedSVDSolve):
        raise TypeError("prepared must be a PreparedSVDSolve.")
    if not isinstance(problem, SVDProblem):
        raise TypeError("problem must be an SVDProblem.")
    if problem.problem_id != prepared.problem.problem_id:
        raise ValueError("SVD refreshes must preserve problem_id.")
    refreshed_plan = plan_svd(problem, prepared.plan.policy)
    if refreshed_plan.plan_id != prepared.plan.plan_id:
        raise ValueError("SVD refresh changed symbolic structure.")
    return PreparedSVDSolve(
        problem,
        refreshed_plan,
        _prepare_dense_svd_state(problem, refreshed_plan),
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def _prepare_dense_svd_state(
    problem: SVDProblem,
    plan: SVDSolvePlan,
    /,
) -> DenseSVDState:
    operator = problem.operator
    source_pairing = _coordinate_pairing_matrix(operator.source)
    target_pairing = _coordinate_pairing_matrix(operator.target)
    source_factor = _pairing_factor(source_pairing, "source")
    target_factor = _pairing_factor(target_pairing, "target")
    matrix = materialize(operator, plan.policy.materialization)
    right_scaled = jnp.conj(
        jsp.linalg.solve_triangular(
            source_factor,
            jnp.conj(matrix.T),
            lower=True,
        ).T
    )
    reduced = jnp.conj(target_factor.T) @ right_scaled
    reduced = eqx.error_if(
        reduced,
        jnp.any(~jnp.isfinite(reduced)),
        "Dense SVD transformed operator contains nonfinite values.",
    )
    return DenseSVDState(reduced, source_factor, target_factor)


def _pairing_factor(pairing: Array, name: str, /) -> Array:
    scale = jnp.maximum(jnp.max(jnp.abs(pairing)), 1)
    tolerance = 64 * max(pairing.shape[0], 1) * jnp.finfo(pairing.real.dtype).eps * scale
    hermitian_error = jnp.max(jnp.abs(pairing - jnp.conj(pairing.T)))
    pairing = eqx.error_if(
        pairing,
        jnp.any(~jnp.isfinite(pairing)) | (hermitian_error > tolerance),
        f"SVD {name} pairing is not numerically Hermitian.",
    )
    factor = jnp.linalg.cholesky(pairing, symmetrize_input=False)
    return eqx.error_if(
        factor,
        jnp.any(~jnp.isfinite(factor)) | jnp.any(jnp.real(jnp.diag(factor)) <= 0),
        f"SVD {name} pairing is not numerically positive-definite.",
    )


def svd(
    problem_or_prepared: SVDProblem | PreparedSVDSolve,
    /,
    *,
    policy: SVDSolvePolicy | SVDSolvePlan | None = None,
) -> SVDSolveResult:
    if isinstance(problem_or_prepared, PreparedSVDSolve):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared SVD solve.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, SVDProblem):
        prepared = prepare_svd(problem_or_prepared, policy)
    else:
        raise TypeError("Expected an SVDProblem or PreparedSVDSolve.")
    numerics = _solve_dense_svd(prepared)
    (
        values,
        left,
        right,
        left_residuals,
        right_residuals,
        relative_residuals,
        left_orthogonality,
        right_orthogonality,
        isolation_gaps,
        converged,
        numerical_rank,
        rank_cutoff,
    ) = numerics
    finite = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(left))
        & jnp.all(jnp.isfinite(right))
        & jnp.all(jnp.isfinite(relative_residuals))
    )
    status = jnp.where(
        finite & jnp.all(converged),
        int(SVDSolveStatus.SUCCESS),
        jnp.where(
            finite,
            int(SVDSolveStatus.RESIDUAL_TOLERANCE_NOT_MET),
            int(SVDSolveStatus.NONFINITE_OUTPUT),
        ),
    ).astype(jnp.int32)
    if prepared.plan.policy.rank.require_full_rank:
        status = jnp.where(
            numerical_rank < prepared.problem.maximum_rank,
            int(SVDSolveStatus.RANK_DEFICIENT),
            status,
        ).astype(jnp.int32)
    if prepared.plan.policy.differentiation == "singular-values":
        uncertainty = 4 * jnp.maximum(left_residuals, right_residuals)
        differentiation_valid = (
            (status == int(SVDSolveStatus.SUCCESS))
            & jnp.all(values > rank_cutoff)
            & jnp.all(jnp.isfinite(isolation_gaps))
            & jnp.all(isolation_gaps > uncertainty)
        )
        status = jnp.where(
            differentiation_valid,
            status,
            int(SVDSolveStatus.DIFFERENTIATION_REJECTED),
        ).astype(jnp.int32)
        values = jax.lax.cond(
            differentiation_valid,
            lambda payload: _mathematical_singular_values(
                prepared.problem,
                jax.lax.stop_gradient(payload[0]),
                jax.lax.stop_gradient(payload[1]),
                jax.lax.stop_gradient(payload[2]),
            ),
            lambda payload: jax.lax.stop_gradient(payload[0]),
            (values, left, right),
        )
    if prepared.plan.policy.failure.mode == "error":
        failed = status != int(SVDSolveStatus.SUCCESS)
        message = "SVD solve failed; inspect status-mode diagnostics."
        status = eqx.error_if(status, failed, message)
        values = eqx.error_if(values, failed, message)
    diagnostics = SVDSolveDiagnostics(
        left_residual_norms=left_residuals,
        right_residual_norms=right_residuals,
        relative_residuals=relative_residuals,
        left_orthogonality_error=left_orthogonality,
        right_orthogonality_error=right_orthogonality,
        isolation_gaps=isolation_gaps,
        converged=converged,
        numerical_rank=numerical_rank,
        operator_matvec_count=jnp.asarray(
            prepared.plan.cost.operator_matvec_count + prepared.plan.policy.count,
            dtype=jnp.int32,
        ),
        adjoint_matvec_count=jnp.asarray(
            prepared.plan.policy.count,
            dtype=jnp.int32,
        ),
    )
    provenance = SVDSolveProvenance(
        method=prepared.plan.policy.method.name,
        problem_id=prepared.problem.problem_id,
        plan_id=prepared.plan.plan_id,
        which=prepared.plan.policy.which,
        differentiation=prepared.plan.policy.differentiation,
        numeric_version=prepared.numeric_version,
    )
    result = SVDSolveResult(
        singular_values=values,
        left_vectors=_unflatten_columns(prepared.problem.operator.target, left),
        right_vectors=_unflatten_columns(prepared.problem.operator.source, right),
        status=status,
        converged=converged,
        diagnostics=diagnostics,
        provenance=provenance,
    )
    if prepared.plan.policy.differentiation == "none":
        return _stop_arrays(result)
    return eqx.tree_at(
        lambda value: (value.left_vectors, value.right_vectors, value.diagnostics),
        result,
        (
            _stop_arrays(result.left_vectors),
            _stop_arrays(result.right_vectors),
            _stop_arrays(result.diagnostics),
        ),
    )


def _solve_dense_svd(prepared: PreparedSVDSolve, /) -> _SVDNumerics:
    state = prepared.state
    policy = prepared.plan.policy
    left_reduced, all_values, right_adjoint_reduced = jnp.linalg.svd(
        state.reduced_operator,
        full_matrices=False,
        compute_uv=True,
    )
    right_reduced = jnp.conj(right_adjoint_reduced.T)
    left = jsp.linalg.solve_triangular(
        jnp.conj(state.target_factor.T),
        left_reduced,
        lower=False,
    )
    right = jsp.linalg.solve_triangular(
        jnp.conj(state.source_factor.T),
        right_reduced,
        lower=False,
    )
    order = (
        jnp.arange(all_values.shape[0])
        if policy.which == "largest"
        else jnp.arange(all_values.shape[0] - 1, -1, -1)
    )
    selected_indices = order[: policy.count]
    values = all_values[selected_indices]
    left = left[:, selected_indices]
    right = right[:, selected_indices]
    operator_right = _operator_columns(prepared.problem.operator, right)
    adjoint_left = _operator_columns(adjoint(prepared.problem.operator), left)
    left_residual = operator_right - left * values[None, :]
    right_residual = adjoint_left - right * values[None, :]
    left_residuals = _column_norms(
        prepared.problem.operator.target,
        left_residual,
    )
    right_residuals = _column_norms(
        prepared.problem.operator.source,
        right_residual,
    )
    left_scale = _column_norms(
        prepared.problem.operator.target,
        operator_right,
    ) + values * _column_norms(prepared.problem.operator.target, left)
    right_scale = _column_norms(
        prepared.problem.operator.source,
        adjoint_left,
    ) + values * _column_norms(prepared.problem.operator.source, right)
    tiny = jnp.finfo(values.dtype).tiny
    relative = jnp.maximum(
        left_residuals / jnp.maximum(left_scale, tiny),
        right_residuals / jnp.maximum(right_scale, tiny),
    )
    left_orthogonality = _orthogonality_error(
        prepared.problem.operator.target,
        left,
    )
    right_orthogonality = _orthogonality_error(
        prepared.problem.operator.source,
        right,
    )
    converged = relative <= jnp.asarray(policy.tolerance.residual, relative.dtype)
    orthogonality_ok = jnp.maximum(
        left_orthogonality,
        right_orthogonality,
    ) <= jnp.asarray(policy.tolerance.orthogonality, relative.dtype)
    converged = converged & orthogonality_ok
    rank_scale = jnp.maximum(jnp.max(all_values), 1)
    relative_cutoff = (
        max(state.reduced_operator.shape) * jnp.finfo(all_values.dtype).eps
        if policy.rank.relative_cutoff is None
        else policy.rank.relative_cutoff
    )
    rank_cutoff = jnp.asarray(relative_cutoff, all_values.dtype) * rank_scale
    numerical_rank = jnp.sum(all_values > rank_cutoff, dtype=jnp.int32)
    isolation_gaps = _singular_value_gaps(
        values,
        all_values,
        selected_indices,
        left_residuals,
        right_residuals,
    )
    return _SVDNumerics(
        values,
        left,
        right,
        left_residuals,
        right_residuals,
        relative,
        left_orthogonality,
        right_orthogonality,
        isolation_gaps,
        converged,
        numerical_rank,
        rank_cutoff,
    )


def _operator_columns(operator: AbstractLinearOperator, columns: Array, /) -> Array:
    return jax.vmap(
        lambda column: operator.target.flatten(
            operator.mv(operator.source.unflatten(column))
        ),
        in_axes=1,
        out_axes=1,
    )(columns)


def _column_norms(space: AbstractVectorSpace, columns: Array, /) -> Array:
    return jax.vmap(
        lambda column: jnp.sqrt(
            jnp.maximum(
                jnp.real(
                    space.inner(
                        space.unflatten(column),
                        space.unflatten(column),
                    )
                ),
                0.0,
            )
        ),
        in_axes=1,
    )(columns)


def _gram(space: AbstractVectorSpace, columns: Array, /) -> Array:
    return jax.vmap(
        lambda left: jax.vmap(
            lambda right: space.inner(
                space.unflatten(left),
                space.unflatten(right),
            ),
            in_axes=1,
        )(columns),
        in_axes=1,
    )(columns)


def _orthogonality_error(space: AbstractVectorSpace, columns: Array, /) -> Array:
    gram = _gram(space, columns)
    return jnp.max(jnp.abs(gram - jnp.eye(columns.shape[1], dtype=gram.dtype)))


def _singular_value_gaps(
    selected_values: Array,
    all_values: Array,
    selected_indices: Array,
    left_residuals: Array,
    right_residuals: Array,
    /,
) -> Array:
    distances = jnp.abs(selected_values[:, None] - all_values[None, :])
    uncertainty = 4 * jnp.maximum(left_residuals, right_residuals)
    roundoff = (
        jnp.sqrt(jnp.finfo(all_values.dtype).eps)
        * max(all_values.shape[0], 1)
        * jnp.maximum(all_values, 1)
    )
    distances = distances - uncertainty[:, None] - roundoff[None, :]
    neighbors = selected_indices[:, None] != jnp.arange(all_values.shape[0])[None, :]
    return jnp.min(
        jnp.where(neighbors, distances, jnp.asarray(jnp.inf)),
        axis=1,
    )


def _unflatten_columns(space: AbstractVectorSpace, columns: Array, /) -> PyTree[Array]:
    return jax.vmap(space.unflatten, in_axes=1, out_axes=-1)(columns)


@eqx.filter_custom_jvp
def _mathematical_singular_values(
    problem: SVDProblem,
    values: Array,
    left: Array,
    right: Array,
    /,
) -> Array:
    del problem, left, right
    return values


@_mathematical_singular_values.def_jvp
def _mathematical_singular_values_jvp(primals, tangents):
    problem, values, left, right = primals
    problem_tangent, _, _, _ = tangents
    target = problem.operator.target

    def perturbation(current_problem):
        contributions = []
        for index in range(values.shape[0]):
            left_vector = target.unflatten(left[:, index])
            right_vector = problem.operator.source.unflatten(right[:, index])
            image = current_problem.operator.mv(right_vector)
            contributions.append(jnp.real(target.inner(left_vector, image)))
        return jnp.stack(contributions)

    _, tangent = eqx.filter_jvp(
        perturbation,
        (problem,),
        (problem_tangent,),
    )
    if tangent is None:
        tangent = jnp.zeros_like(values)
    return values, tangent


def _stop_arrays(value: Any, /) -> Any:
    return jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
        value,
    )


__all__ = [
    "DenseSVD",
    "DenseSVDState",
    "PreparedSVDSolve",
    "SVDCostEstimate",
    "SVDDifferentiationMode",
    "SVDProblem",
    "SVDResourcePolicy",
    "SVDSolveDiagnostics",
    "SVDSolvePlan",
    "SVDSolvePolicy",
    "SVDSolveProvenance",
    "SVDSolveResult",
    "SVDSolveStatus",
    "SVDTarget",
    "SVDTolerancePolicy",
    "plan_svd",
    "prepare_svd",
    "refresh_svd",
    "svd",
]
