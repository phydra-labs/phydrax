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
from jaxtyping import Array, ArrayLike, PyTree

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._binding import LinearSolveTemplate
from ._factorizations import PreparedFactorization
from ._operators import AbstractLinearOperator
from ._pfaffian import evaluate_pfaffian, PfaffianPolicy, PfaffianResult
from ._policies import DenseLU, FailurePolicy, LinearSolvePolicy
from ._prepared import PreparedLinearSolve
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
from .backends._jax_dense import dense_lu_slogdet, DenseLUState


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


LowRankUpdateRoute = Literal[
    "dense",
    "row-indexed",
    "column-indexed",
    "skew-row-column-indexed",
]


class LowRankDeterminantStatus(IntEnum):
    """Status for an exact determinant-ratio proposal."""

    SUCCESS = 0
    BASE_SOLVE_FAILED = 1
    CURRENT_CORRECTION_ILL_CONDITIONED = 2
    PROPOSAL_CORRECTION_ILL_CONDITIONED = 3
    CAPACITY_EXCEEDED = 4
    NONFINITE_RATIO = 5


class LowRankPfaffianStatus(IntEnum):
    """Status for an exact skew low-rank Pfaffian-ratio proposal."""

    SUCCESS = 0
    DETERMINANT_PROPOSAL_FAILED = 1
    COMPACT_PFAFFIAN_FAILED = 2
    RATIO_IDENTITY_FAILED = 3


class LowRankUpdate(StrictModule):
    """Algebraic ``L Rᵀ`` data, optionally retaining an indexed identity factor."""

    left_factor: Array
    right_factor: Array
    indices: Array
    dimension: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    route: LowRankUpdateRoute = eqx.field(static=True)
    update_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_factor: ArrayLike,
        right_factor: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        dimension: int,
        route: LowRankUpdateRoute,
    ):
        left = jnp.asarray(left_factor)
        right = jnp.asarray(right_factor)
        indices_ = jnp.asarray(indices, dtype=jnp.int32)
        size = int(dimension)
        if size < 1:
            raise ValueError("Low-rank update dimension must be positive.")
        if route not in (
            "dense",
            "row-indexed",
            "column-indexed",
            "skew-row-column-indexed",
        ):
            raise ValueError("Unknown low-rank update route.")
        if route == "dense":
            if left.ndim != 2 or right.shape != left.shape or left.shape[0] != size:
                raise ValueError("Dense factors must have matching shape (n, rank).")
            rank = int(left.shape[1])
            expected_indices = (rank,)
        elif route == "row-indexed":
            if (
                left.ndim != 2
                or right.ndim != 2
                or left.shape[0] != 0
                or right.shape[0] != size
                or left.shape[1] != right.shape[1]
            ):
                raise ValueError(
                    "Indexed-row data must have factor shapes (0, rank) and (n, rank)."
                )
            rank = int(right.shape[1])
            expected_indices = (rank,)
        elif route == "column-indexed":
            if (
                left.ndim != 2
                or right.ndim != 2
                or right.shape[0] != 0
                or left.shape[0] != size
                or right.shape[1] != left.shape[1]
            ):
                raise ValueError(
                    "Indexed-column data must have factor shapes (n, rank) and (0, rank)."
                )
            rank = int(left.shape[1])
            expected_indices = (rank,)
        else:
            if left.shape != (size, 2) or right.shape != (size, 2):
                raise ValueError("Skew row/column data must have factor shapes (n, 2).")
            rank = 2
            expected_indices = (1,)
        if rank < 1:
            raise ValueError("Low-rank updates must contain at least one column.")
        if indices_.shape != expected_indices:
            raise ValueError("Low-rank update indices have an invalid shape.")
        dtype = jnp.result_type(left.dtype, right.dtype)
        left, right = left.astype(dtype), right.astype(dtype)
        coefficients = jnp.concatenate((left.reshape(-1), right.reshape(-1)))
        coefficients = eqx.error_if(
            coefficients,
            jnp.any(~jnp.isfinite(coefficients)),
            "Low-rank update coefficients must be finite.",
        )
        left_size = left.size
        left = coefficients[:left_size].reshape(left.shape)
        right = coefficients[left_size:].reshape(right.shape)
        if route != "dense":
            indices_ = eqx.error_if(
                indices_,
                jnp.any((indices_ < 0) | (indices_ >= size)),
                "Low-rank update indices are out of bounds.",
            )
        self.left_factor = left
        self.right_factor = right
        self.indices = indices_
        self.dimension = size
        self.rank = rank
        self.route = route
        self.update_id = canonical_fingerprint(
            {
                "kind": "bilinear-low-rank-update",
                "dimension": size,
                "rank": rank,
                "route": route,
            }
        )


class PreparedLowRankSequence(StrictModule):
    """Fixed-capacity accepted updates over one reusable base factorization."""

    prepared: PreparedLowRankSolve
    base_determinant_sign: Array
    base_log_abs_determinant: Array
    base_determinant_available: Array
    base_lineage: Array
    active_rank: Array
    accepted_count: Array
    determinant_sign: Array
    log_abs_determinant_ratio: Array
    status: Array
    sequence_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedLowRankSolve,
        /,
        *,
        active_rank: Any = 0,
        accepted_count: Any = 0,
        determinant_sign: Any = 1,
        log_abs_determinant_ratio: Any = 0,
        status: Any = LowRankDeterminantStatus.SUCCESS,
        base_lineage: ArrayLike | None = None,
    ):
        if not isinstance(prepared, PreparedLowRankSolve):
            raise TypeError("prepared must be a PreparedLowRankSolve.")
        active = jnp.asarray(active_rank, dtype=jnp.int32)
        accepted = jnp.asarray(accepted_count, dtype=jnp.int32)
        sign = jnp.asarray(determinant_sign, dtype=prepared.operator.core.dtype)
        log_abs = jnp.asarray(
            log_abs_determinant_ratio,
            dtype=prepared.operator.core.real.dtype,
        )
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if any(value.shape != () for value in (active, accepted, sign, log_abs, status_)):
            raise ValueError(
                "Low-rank sequence counters and determinant data are scalar."
            )
        active = eqx.error_if(
            active,
            (active < 0) | (active > prepared.plan.rank),
            "Low-rank sequence active rank is outside its fixed capacity.",
        )
        base_sign, base_log_abs, base_available = _sequence_base_determinant(prepared)
        lineage = (
            _array_tree_lineage(prepared.base_prepared)
            if base_lineage is None
            else jnp.asarray(base_lineage, dtype=jnp.uint64)
        )
        if lineage.shape != (2,):
            raise ValueError("base_lineage must have shape (2,).")
        state = prepared.base_prepared.state
        if isinstance(state, DenseLUState):
            status_ = jnp.where(
                state.singular,
                int(LowRankDeterminantStatus.BASE_SOLVE_FAILED),
                status_,
            )
        self.prepared = prepared
        self.base_determinant_sign = base_sign
        self.base_log_abs_determinant = base_log_abs
        self.base_determinant_available = base_available
        self.base_lineage = lineage
        self.active_rank = active
        self.accepted_count = accepted
        self.determinant_sign = sign
        self.log_abs_determinant_ratio = log_abs
        self.status = status_
        self.sequence_id = canonical_fingerprint(
            {
                "kind": "prepared-low-rank-sequence",
                "plan": prepared.plan.plan_id,
                "capacity": prepared.plan.rank,
            }
        )

    @property
    def operator(self) -> BasePlusLowRankLinearOperator:
        return self.prepared.operator

    @property
    def plan(self) -> LowRankSolvePlan:
        return self.prepared.plan

    @property
    def compact_condition(self) -> Array:
        return self.prepared.correction_condition

    @property
    def capacity(self) -> int:
        return self.prepared.plan.rank

    @property
    def remaining_capacity(self) -> Array:
        return jnp.asarray(self.capacity, dtype=jnp.int32) - self.active_rank

    @property
    def absolute_determinant_sign(self) -> Array:
        sign = self.base_determinant_sign * self.determinant_sign
        return jnp.where(
            self.base_determinant_available,
            sign,
            jnp.asarray(jnp.nan, dtype=sign.dtype),
        )

    @property
    def absolute_log_abs_determinant(self) -> Array:
        sign = self.base_determinant_sign * self.determinant_sign
        log_abs = self.base_log_abs_determinant + self.log_abs_determinant_ratio
        unavailable = jnp.asarray(jnp.nan, dtype=log_abs.dtype)
        return jnp.where(
            self.base_determinant_available,
            jnp.where(jnp.abs(sign) > 0, log_abs, -jnp.inf),
            unavailable,
        )

    @property
    def successful(self) -> Array:
        return self.status == int(LowRankDeterminantStatus.SUCCESS)

    @property
    def requires_rebase(self) -> Array:
        return ~self.successful

    def solve(
        self,
        rhs: PyTree[Any],
        /,
        *,
        rhs_layout: RHSLayout | None = None,
    ) -> LowRankSolveResult:
        """Apply the current inverse through base and compact prepared solves."""
        return solve_low_rank(self.prepared, rhs, rhs_layout=rhs_layout)


class LowRankDeterminantProvenance(StrictModule):
    """Sequence, solve, route, version, and fixed-capacity evidence."""

    solve: LowRankSolveProvenance
    sequence_id: str = eqx.field(static=True)
    update_id: str = eqx.field(static=True)
    route: LowRankUpdateRoute = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    proposed_rank: int = eqx.field(static=True)
    active_rank: Array
    accepted_count: Array
    base_lineage: Array

    def __init__(
        self,
        sequence: PreparedLowRankSequence,
        update: LowRankUpdate,
        /,
    ):
        self.solve = LowRankSolveProvenance(sequence.prepared)
        self.sequence_id = sequence.sequence_id
        self.update_id = update.update_id
        self.route = update.route
        self.capacity = sequence.capacity
        self.proposed_rank = update.rank
        self.active_rank = sequence.active_rank
        self.base_lineage = sequence.base_lineage
        self.accepted_count = sequence.accepted_count


class LowRankDeterminantResult(StrictModule):
    """Exact signed-log ratio and reusable candidate correction state."""

    value: Array
    sign: Array
    log_abs: Array
    compact_condition: Array
    aggregate_condition: Array
    status: Array
    source_prepared: PreparedLowRankSolve
    update: LowRankUpdate
    candidate_prepared: PreparedLowRankSolve
    base_inverse_left_factor: Array
    current_solved_left_factor: Array
    base_result: LinearSolveResult
    provenance: LowRankDeterminantProvenance

    def __init__(
        self,
        *,
        value: Any,
        sign: Any,
        log_abs: Any,
        compact_condition: Any,
        aggregate_condition: Any,
        status: Any,
        source_prepared: PreparedLowRankSolve,
        update: LowRankUpdate,
        candidate_prepared: PreparedLowRankSolve,
        base_inverse_left_factor: ArrayLike,
        current_solved_left_factor: ArrayLike,
        base_result: LinearSolveResult,
        provenance: LowRankDeterminantProvenance,
    ):
        self.value = jnp.asarray(value)
        self.sign = jnp.asarray(sign)
        self.log_abs = jnp.asarray(log_abs)
        self.compact_condition = jnp.asarray(compact_condition)
        self.aggregate_condition = jnp.asarray(aggregate_condition)
        self.source_prepared = source_prepared
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.update = update
        self.candidate_prepared = candidate_prepared
        self.base_inverse_left_factor = jnp.asarray(base_inverse_left_factor)
        self.current_solved_left_factor = jnp.asarray(current_solved_left_factor)
        self.base_result = base_result
        self.provenance = provenance

    @property
    def route(self) -> LowRankUpdateRoute:
        return self.update.route

    @property
    def successful(self) -> Array:
        return self.status == int(LowRankDeterminantStatus.SUCCESS)

    @property
    def valid(self) -> Array:
        return self.successful

    @property
    def requires_rebase(self) -> Array:
        return (
            (
                self.status
                == int(LowRankDeterminantStatus.CURRENT_CORRECTION_ILL_CONDITIONED)
            )
            | (
                self.status
                == int(LowRankDeterminantStatus.PROPOSAL_CORRECTION_ILL_CONDITIONED)
            )
            | (self.status == int(LowRankDeterminantStatus.CAPACITY_EXCEEDED))
            | (self.status == int(LowRankDeterminantStatus.NONFINITE_RATIO))
        )


class LowRankPfaffianResult(StrictModule):
    """Exact signed-log Pfaffian ratio with determinant-identity evidence."""

    value: Array
    sign: Array
    log_abs: Array
    compact: Array
    determinant_identity_residual: Array
    status: Array
    determinant: LowRankDeterminantResult
    compact_pfaffian: PfaffianResult

    def __init__(
        self,
        *,
        value: ArrayLike,
        sign: ArrayLike,
        log_abs: ArrayLike,
        compact: ArrayLike,
        determinant_identity_residual: ArrayLike,
        status: ArrayLike,
        determinant: LowRankDeterminantResult,
        compact_pfaffian: PfaffianResult,
    ):
        self.value = jnp.asarray(value)
        self.sign = jnp.asarray(sign)
        self.log_abs = jnp.asarray(log_abs)
        self.compact = jnp.asarray(compact)
        self.determinant_identity_residual = jnp.asarray(determinant_identity_residual)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.determinant = determinant
        self.compact_pfaffian = compact_pfaffian

    @property
    def successful(self) -> Array:
        return self.status == int(LowRankPfaffianStatus.SUCCESS)

    @property
    def valid(self) -> Array:
        return self.successful

    @property
    def requires_rebase(self) -> Array:
        return self.determinant.requires_rebase | ~self.successful


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
    correction_rhs = contract(
        "ab,bn,nr->ar",
        operator.core,
        jnp.conj(jnp.swapaxes(operator.right_factor, 0, 1)),
        base_value,
    )
    correction = jsp.linalg.lu_solve(
        (prepared.correction_lu, prepared.correction_pivots),
        correction_rhs,
    )
    coordinates = base_value - contract(
        "na,ar->nr",
        prepared.inverse_left_factor,
        correction,
    )
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


def dense_low_rank_update(
    left_factor: ArrayLike,
    right_factor: ArrayLike,
    /,
) -> LowRankUpdate:
    """Represent the bilinear update ``L Rᵀ`` without conjugating ``R``."""
    left, right = jnp.asarray(left_factor), jnp.asarray(right_factor)
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("Dense factors must have matching shape (n, rank).")
    rank = int(left.shape[1])
    return LowRankUpdate(
        left,
        right,
        jnp.full((rank,), -1, dtype=jnp.int32),
        dimension=int(left.shape[0]),
        route="dense",
    )


def row_low_rank_update(
    indices: ArrayLike,
    rows: ArrayLike,
    /,
) -> LowRankUpdate:
    """Represent additions to selected rows without retaining a one-hot factor."""
    indices_, rows_ = jnp.asarray(indices, dtype=jnp.int32), jnp.asarray(rows)
    if rows_.ndim != 2 or indices_.shape != (rows_.shape[0],):
        raise ValueError(
            "rows must have shape (rank, n) matching one-dimensional indices."
        )
    rank, dimension = (int(value) for value in rows_.shape)
    return LowRankUpdate(
        jnp.zeros((0, rank), dtype=rows_.dtype),
        jnp.swapaxes(rows_, 0, 1),
        indices_,
        dimension=dimension,
        route="row-indexed",
    )


def column_low_rank_update(
    indices: ArrayLike,
    columns: ArrayLike,
    /,
) -> LowRankUpdate:
    """Represent additions to selected columns without retaining a one-hot factor."""
    indices_, columns_ = jnp.asarray(indices, dtype=jnp.int32), jnp.asarray(columns)
    if columns_.ndim != 2 or indices_.shape != (columns_.shape[1],):
        raise ValueError(
            "columns must have shape (n, rank) matching one-dimensional indices."
        )
    dimension, rank = (int(value) for value in columns_.shape)
    return LowRankUpdate(
        columns_,
        jnp.zeros((0, rank), dtype=columns_.dtype),
        indices_,
        dimension=dimension,
        route="column-indexed",
    )


def skew_row_column_low_rank_update(
    index: ArrayLike,
    row_delta: ArrayLike,
    /,
) -> LowRankUpdate:
    """Represent ``eᵢ rᵀ - r eᵢᵀ`` as one joint rank-two proposal."""
    row = jnp.asarray(row_delta)
    index_ = jnp.asarray(index, dtype=jnp.int32)
    if row.ndim != 1 or index_.shape != ():
        raise ValueError("row_delta must be one-dimensional and index must be scalar.")
    dimension = int(row.shape[0])
    zeros = jnp.zeros_like(row)
    left = jnp.stack((zeros, -row), axis=1)
    right = jnp.stack((row, zeros), axis=1)
    return LowRankUpdate(
        left,
        right,
        index_[None],
        dimension=dimension,
        route="skew-row-column-indexed",
    )


def prepare_low_rank_sequence(
    base: AbstractLinearOperator,
    capacity: int,
    policy: LowRankSolvePolicy | None = None,
    /,
    *,
    numeric_version: Any = 0,
    operator_id: str | None = None,
) -> PreparedLowRankSequence:
    """Prepare an empty fixed-capacity update sequence over one base solve."""
    if not isinstance(base, AbstractLinearOperator):
        raise TypeError("base must be an AbstractLinearOperator.")
    capacity_ = int(capacity)
    operator = _empty_sequence_operator(base, capacity_, operator_id)
    prepared = prepare_low_rank_solve(
        operator,
        policy,
        numeric_version=numeric_version,
    )
    return PreparedLowRankSequence(prepared)


def prepare_factorized_low_rank_sequence(
    factorization: PreparedFactorization,
    capacity: int,
    policy: LowRankSolvePolicy | None = None,
    /,
    *,
    operator_id: str | None = None,
) -> PreparedLowRankSequence:
    """Reuse one prepared dense base factorization for an empty update sequence."""
    if not isinstance(factorization, PreparedFactorization):
        raise TypeError("factorization must be a PreparedFactorization.")
    if not isinstance(factorization.prepared_solve.state, DenseLUState):
        raise ValueError(
            "Factorized low-rank sequences currently require a prepared DenseLU base."
        )
    capacity_ = int(capacity)
    operator = _empty_sequence_operator(
        factorization.operator,
        capacity_,
        operator_id,
    )
    plan = plan_low_rank_solve(operator, policy)
    if not isinstance(plan.policy.base.method, DenseLU):
        raise ValueError(
            "Factorized low-rank sequence policy must use a DenseLU base method."
        )
    if plan.base_template.plan.backend != factorization.prepared_solve.plan.backend:
        raise ValueError(
            "Factorization and low-rank base policies select different backends."
        )
    base_prepared = PreparedLinearSolve(
        _base_problem(operator),
        plan.base_template,
        factorization.prepared_solve.state,
        preconditioning_state=factorization.prepared_solve.preconditioning_state,
        numeric_version=factorization.prepared_solve.numeric_version,
    )
    prepared = _prepare_from_base(
        operator,
        plan,
        base_prepared,
        numeric_version=factorization.prepared_solve.numeric_version,
    )
    return PreparedLowRankSequence(prepared)


def propose_low_rank_update(
    sequence: PreparedLowRankSequence,
    update: LowRankUpdate,
    /,
) -> LowRankDeterminantResult:
    """Evaluate ``det(M + L Rᵀ) / det(M)`` and prepare an accept-ready state."""
    _validate_sequence_update(sequence, update)
    prepared = sequence.prepared
    operator = prepared.operator
    left_coordinates = _update_left_coordinates(update, operator.left_factor.dtype)
    left_rhs = jax.vmap(operator.target.unflatten, in_axes=1, out_axes=1)(
        left_coordinates
    )
    base_result = solve(
        prepared.base_prepared,
        left_rhs,
        rhs_layout=RHSLayout((update.rank,)),
    )
    base_inverse_left, _ = _pack_rhs(operator.source, (), base_result.value)
    current_projection = contract(
        "ab,bn,nr->ar",
        operator.core,
        jnp.conj(jnp.swapaxes(operator.right_factor, 0, 1)),
        base_inverse_left,
    )
    current_coefficients = jsp.linalg.lu_solve(
        (prepared.correction_lu, prepared.correction_pivots),
        current_projection,
    )
    current_solved_left = base_inverse_left - contract(
        "na,ar->nr",
        prepared.inverse_left_factor,
        current_coefficients,
    )
    compact = jnp.eye(update.rank, dtype=base_inverse_left.dtype) + (
        _update_right_transpose_action(update, current_solved_left)
    )
    sign, log_abs = jnp.linalg.slogdet(compact)
    compact_condition = jnp.linalg.cond(compact)

    candidate_prepared = _candidate_prepared_low_rank(
        sequence,
        update,
        base_inverse_left,
    )
    aggregate_condition = candidate_prepared.correction_condition
    limit = prepared.plan.policy.condition_limit
    current_valid = (
        sequence.successful
        & jnp.isfinite(prepared.correction_condition)
        & (prepared.correction_condition <= limit)
    )
    base_valid = jnp.all(
        jnp.asarray(base_result.status) == int(LinearSolveStatus.SUCCESS)
    )
    capacity_valid = update.rank <= sequence.remaining_capacity
    compact_valid = (
        jnp.isfinite(compact_condition)
        & (compact_condition <= limit)
        & jnp.isfinite(aggregate_condition)
        & (aggregate_condition <= limit)
        & jnp.all(jnp.isfinite(candidate_prepared.correction_lu))
    )
    ratio_finite = (
        jnp.all(jnp.isfinite(sign)) & jnp.isfinite(log_abs) & (jnp.abs(sign) > 0)
    )
    status = jnp.asarray(int(LowRankDeterminantStatus.SUCCESS), dtype=jnp.int32)
    status = _set_low_rank_failure(
        status,
        ~current_valid,
        LowRankDeterminantStatus.CURRENT_CORRECTION_ILL_CONDITIONED,
    )
    status = _set_low_rank_failure(
        status,
        ~base_valid,
        LowRankDeterminantStatus.BASE_SOLVE_FAILED,
    )
    status = _set_low_rank_failure(
        status,
        ~capacity_valid,
        LowRankDeterminantStatus.CAPACITY_EXCEEDED,
    )
    status = _set_low_rank_failure(
        status,
        ~compact_valid,
        LowRankDeterminantStatus.PROPOSAL_CORRECTION_ILL_CONDITIONED,
    )
    status = _set_low_rank_failure(
        status,
        ~ratio_finite,
        LowRankDeterminantStatus.NONFINITE_RATIO,
    )
    value = sign * jnp.exp(log_abs)
    if prepared.plan.policy.failure.mode == "error":
        value = eqx.error_if(
            value,
            status != int(LowRankDeterminantStatus.SUCCESS),
            "Low-rank determinant proposal failed; use status failure mode for "
            "diagnostics.",
        )
    return LowRankDeterminantResult(
        value=value,
        sign=sign,
        log_abs=log_abs,
        compact_condition=compact_condition,
        aggregate_condition=aggregate_condition,
        status=status,
        source_prepared=prepared,
        update=update,
        candidate_prepared=candidate_prepared,
        base_inverse_left_factor=base_inverse_left,
        current_solved_left_factor=current_solved_left,
        base_result=base_result,
        provenance=LowRankDeterminantProvenance(sequence, update),
    )


def propose_pfaffian_update(
    sequence: PreparedLowRankSequence,
    update: LowRankUpdate,
    policy: PfaffianPolicy | None = None,
    /,
    *,
    determinant_identity_tolerance: float = 1.0e-8,
) -> LowRankPfaffianResult:
    """Evaluate one exact local Pfaffian ratio through native solve actions."""
    if update.route != "skew-row-column-indexed" or update.rank != 2:
        raise ValueError("Pfaffian updates require one skew_row_column_low_rank_update.")
    tolerance = float(determinant_identity_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "determinant_identity_tolerance must be finite and non-negative."
        )
    determinant = propose_low_rank_update(sequence, update)
    dtype = determinant.current_solved_left_factor.dtype
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    symplectic = jnp.stack((jnp.stack((zero, one)), jnp.stack((-one, zero))))
    left = _update_left_coordinates(update, dtype)
    compact = symplectic + contract(
        "ni,nj->ij",
        left,
        determinant.current_solved_left_factor,
    )
    compact_pfaffian = evaluate_pfaffian(compact, policy)

    pfaffian_log_abs_determinant = 2.0 * compact_pfaffian.log_abs
    finite_identity = (
        jnp.isfinite(pfaffian_log_abs_determinant)
        & jnp.isfinite(determinant.log_abs)
        & jnp.isfinite(compact_pfaffian.sign)
        & jnp.isfinite(determinant.sign)
    )
    scale_log = jnp.maximum(
        jnp.asarray(0.0, dtype=determinant.log_abs.dtype),
        jnp.maximum(
            jnp.where(finite_identity, pfaffian_log_abs_determinant, 0.0),
            jnp.where(finite_identity, determinant.log_abs, 0.0),
        ),
    )
    pfaffian_scaled_log = jnp.where(
        finite_identity,
        pfaffian_log_abs_determinant - scale_log,
        -jnp.inf,
    )
    determinant_scaled_log = jnp.where(
        finite_identity,
        determinant.log_abs - scale_log,
        -jnp.inf,
    )
    pfaffian_scaled = (
        compact_pfaffian.sign * compact_pfaffian.sign * jnp.exp(pfaffian_scaled_log)
    )
    determinant_scaled = determinant.sign * jnp.exp(determinant_scaled_log)
    identity_residual = jnp.where(
        finite_identity,
        jnp.abs(pfaffian_scaled - determinant_scaled),
        jnp.inf,
    )
    compact_valid = (
        compact_pfaffian.successful
        & ~compact_pfaffian.singular
        & jnp.isfinite(compact_pfaffian.log_abs)
        & jnp.isfinite(compact_pfaffian.sign)
        & (jnp.abs(compact_pfaffian.sign) > 0)
    )
    identity_valid = identity_residual <= tolerance
    status = jnp.asarray(int(LowRankPfaffianStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        determinant.successful,
        status,
        int(LowRankPfaffianStatus.DETERMINANT_PROPOSAL_FAILED),
    )
    status = jnp.where(
        (status == int(LowRankPfaffianStatus.SUCCESS)) & ~compact_valid,
        int(LowRankPfaffianStatus.COMPACT_PFAFFIAN_FAILED),
        status,
    )
    status = jnp.where(
        (status == int(LowRankPfaffianStatus.SUCCESS)) & ~identity_valid,
        int(LowRankPfaffianStatus.RATIO_IDENTITY_FAILED),
        status,
    )
    return LowRankPfaffianResult(
        value=compact_pfaffian.value,
        sign=compact_pfaffian.sign,
        log_abs=compact_pfaffian.log_abs,
        compact=compact,
        determinant_identity_residual=identity_residual,
        status=status,
        determinant=determinant,
        compact_pfaffian=compact_pfaffian,
    )


def accept_low_rank_update(
    sequence: PreparedLowRankSequence,
    proposal: LowRankDeterminantResult,
    /,
    *,
    accepted: Any = True,
) -> PreparedLowRankSequence:
    """Select a valid proposal without rebuilding its base or compact factors."""
    if not isinstance(sequence, PreparedLowRankSequence):
        raise TypeError("sequence must be a PreparedLowRankSequence.")
    if not isinstance(proposal, LowRankDeterminantResult):
        raise TypeError("proposal must be a LowRankDeterminantResult.")
    if proposal.provenance.sequence_id != sequence.sequence_id:
        raise ValueError("The determinant proposal belongs to another sequence.")
    source = proposal.source_prepared
    current = sequence.prepared
    versions_match = source.numeric_version == current.numeric_version
    source_matches = (
        jnp.array_equal(source.operator.left_factor, current.operator.left_factor)
        & jnp.array_equal(source.operator.right_factor, current.operator.right_factor)
        & jnp.array_equal(source.correction_matrix, current.correction_matrix)
        & jnp.array_equal(proposal.provenance.base_lineage, sequence.base_lineage)
        & (proposal.provenance.active_rank == sequence.active_rank)
        & (proposal.provenance.accepted_count == sequence.accepted_count)
    )
    accepted_ = jnp.asarray(accepted, dtype=bool)
    if accepted_.shape != ():
        raise ValueError("accepted must be scalar.")
    versions_match = jnp.asarray(versions_match, dtype=bool)
    selected = accepted_ & proposal.successful & versions_match & source_matches

    def perform(_: None) -> PreparedLowRankSequence:
        return PreparedLowRankSequence(
            proposal.candidate_prepared,
            active_rank=sequence.active_rank + proposal.update.rank,
            accepted_count=sequence.accepted_count + 1,
            determinant_sign=sequence.determinant_sign * proposal.sign,
            log_abs_determinant_ratio=(
                sequence.log_abs_determinant_ratio + proposal.log_abs
            ),
            base_lineage=sequence.base_lineage,
            status=LowRankDeterminantStatus.SUCCESS,
        )

    return jax.lax.cond(selected, perform, lambda _: sequence, None)


def refresh_low_rank_sequence(
    sequence: PreparedLowRankSequence,
    base: AbstractLinearOperator,
    /,
) -> PreparedLowRankSequence:
    """Refresh changed base coefficients while retaining accepted updates."""
    if not isinstance(sequence, PreparedLowRankSequence):
        raise TypeError("sequence must be a PreparedLowRankSequence.")
    if not isinstance(base, AbstractLinearOperator):
        raise TypeError("base must be an AbstractLinearOperator.")
    current = sequence.prepared.operator
    operator = BasePlusLowRankLinearOperator(
        base,
        current.left_factor,
        current.right_factor,
        current.core,
        operator_id=current.operator_id,
    )
    prepared = refresh_low_rank_solve(sequence.prepared, operator)
    sign, log_abs = jnp.linalg.slogdet(prepared.correction_matrix)
    condition_valid = jnp.isfinite(prepared.correction_condition) & (
        prepared.correction_condition <= prepared.plan.policy.condition_limit
    )
    ratio_valid = (
        jnp.all(jnp.isfinite(sign)) & jnp.isfinite(log_abs) & (jnp.abs(sign) > 0)
    )
    status = jnp.where(
        condition_valid & ratio_valid,
        int(LowRankDeterminantStatus.SUCCESS),
        int(LowRankDeterminantStatus.CURRENT_CORRECTION_ILL_CONDITIONED),
    )
    return PreparedLowRankSequence(
        prepared,
        active_rank=sequence.active_rank,
        accepted_count=sequence.accepted_count,
        determinant_sign=sign,
        log_abs_determinant_ratio=log_abs,
        status=status,
    )


def rebase_low_rank_sequence(
    sequence: PreparedLowRankSequence,
    base: AbstractLinearOperator,
    /,
) -> PreparedLowRankSequence:
    """Start an empty sequence after the caller folds accepted updates into a base."""
    if not isinstance(sequence, PreparedLowRankSequence):
        raise TypeError("sequence must be a PreparedLowRankSequence.")
    return prepare_low_rank_sequence(
        base,
        sequence.capacity,
        sequence.prepared.plan.policy,
        numeric_version=sequence.prepared.numeric_version + 1,
    )


def solve_low_rank_sequence(
    sequence: PreparedLowRankSequence,
    rhs: PyTree[Any],
    /,
    *,
    rhs_layout: RHSLayout | None = None,
) -> LowRankSolveResult:
    """Solve the current accumulated operator using its prepared compact state."""
    if not isinstance(sequence, PreparedLowRankSequence):
        raise TypeError("sequence must be a PreparedLowRankSequence.")
    return solve_low_rank(sequence.prepared, rhs, rhs_layout=rhs_layout)


def _empty_sequence_operator(
    base: AbstractLinearOperator,
    capacity: int,
    operator_id: str | None,
    /,
) -> BasePlusLowRankLinearOperator:
    if capacity < 1:
        raise ValueError("Low-rank sequence capacity must be positive.")
    dtype = base.source.flatten(base.source.zeros()).dtype
    zeros = jnp.zeros((base.source.size, capacity), dtype=dtype)
    identifier = (
        f"{base.operator_id}:low-rank-sequence:{capacity}"
        if operator_id is None
        else str(operator_id)
    )
    return BasePlusLowRankLinearOperator(
        base,
        zeros,
        zeros,
        jnp.eye(capacity, dtype=dtype),
        operator_id=identifier,
    )


def _sequence_base_determinant(
    prepared: PreparedLowRankSolve,
    /,
) -> tuple[Array, Array, Array]:
    state = prepared.base_prepared.state
    sign_dtype = prepared.operator.core.dtype
    log_dtype = prepared.operator.core.real.dtype
    if isinstance(state, DenseLUState):
        sign, log_abs = dense_lu_slogdet(state)
        finite_sign = jnp.isfinite(sign)
        nonzero = jnp.abs(sign) > 0
        available = finite_sign & (jnp.isfinite(log_abs) | ~nonzero)
        stored_log_abs = jnp.where(nonzero, log_abs, 0.0)
        return (
            sign.astype(sign_dtype),
            stored_log_abs.astype(log_dtype),
            jnp.asarray(available, dtype=bool),
        )
    return (
        jnp.ones((), dtype=sign_dtype),
        jnp.zeros((), dtype=log_dtype),
        jnp.asarray(False),
    )


def _validate_sequence_update(
    sequence: PreparedLowRankSequence,
    update: LowRankUpdate,
    /,
) -> None:
    if not isinstance(sequence, PreparedLowRankSequence):
        raise TypeError("sequence must be a PreparedLowRankSequence.")
    if not isinstance(update, LowRankUpdate):
        raise TypeError("update must be a LowRankUpdate.")
    operator = sequence.prepared.operator
    if update.dimension != operator.source.size:
        raise ValueError("The low-rank update dimension does not match the sequence.")
    coordinate_dtype = np.dtype(operator.left_factor.dtype)
    if (
        np.dtype(update.left_factor.dtype) != coordinate_dtype
        or np.dtype(update.right_factor.dtype) != coordinate_dtype
    ):
        raise TypeError("Low-rank update dtype must match the sequence coordinate dtype.")


def _update_left_coordinates(update: LowRankUpdate, dtype, /) -> Array:
    if update.route == "dense" or update.route == "column-indexed":
        return update.left_factor
    if update.route == "row-indexed":
        coordinates = jnp.zeros((update.dimension, update.rank), dtype=dtype)
        return coordinates.at[
            update.indices,
            jnp.arange(update.rank, dtype=jnp.int32),
        ].set(1)
    coordinates = update.left_factor
    return coordinates.at[update.indices[0], 0].set(1)


def _update_right_transpose_action(
    update: LowRankUpdate,
    coordinates: Array,
    /,
) -> Array:
    if update.route == "column-indexed":
        return coordinates[update.indices, :]
    if update.route == "skew-row-column-indexed":
        dense = contract("nr,nk->rk", update.right_factor, coordinates)
        return dense.at[1, :].set(coordinates[update.indices[0], :])
    return contract("nr,nk->rk", update.right_factor, coordinates)


def _dense_update_factors(update: LowRankUpdate, dtype, /) -> tuple[Array, Array]:
    left = _update_left_coordinates(update, dtype)
    if update.route == "dense" or update.route == "row-indexed":
        return left, update.right_factor
    if update.route == "column-indexed":
        right = jnp.zeros((update.dimension, update.rank), dtype=dtype)
        right = right.at[
            update.indices,
            jnp.arange(update.rank, dtype=jnp.int32),
        ].set(1)
        return left, right
    right = update.right_factor.at[update.indices[0], 1].set(1)
    return left, right


def _candidate_prepared_low_rank(
    sequence: PreparedLowRankSequence,
    update: LowRankUpdate,
    base_inverse_left: Array,
    /,
) -> PreparedLowRankSolve:
    prepared = sequence.prepared
    operator = prepared.operator
    capacity = sequence.capacity
    stored_rank = min(update.rank, capacity)
    safe_start = jnp.minimum(
        sequence.active_rank,
        jnp.asarray(max(capacity - stored_rank, 0), dtype=jnp.int32),
    )
    slots = safe_start + jnp.arange(stored_rank, dtype=jnp.int32)
    left, algebraic_right = _dense_update_factors(
        update,
        operator.left_factor.dtype,
    )
    candidate_left = operator.left_factor.at[:, slots].set(left[:, :stored_rank])
    current_algebraic_right = jnp.conj(operator.right_factor)
    candidate_algebraic_right = current_algebraic_right.at[:, slots].set(
        algebraic_right[:, :stored_rank]
    )
    candidate_inverse_left = prepared.inverse_left_factor.at[:, slots].set(
        base_inverse_left[:, :stored_rank]
    )
    candidate_operator = BasePlusLowRankLinearOperator(
        operator.base,
        candidate_left,
        jnp.conj(candidate_algebraic_right),
        operator.core,
        operator_id=operator.operator_id,
    )
    correction_matrix = jnp.eye(capacity, dtype=operator.core.dtype) + contract(
        "ab,bn,nc->ac",
        operator.core,
        candidate_algebraic_right.T,
        candidate_inverse_left,
    )
    correction_condition = jnp.linalg.cond(correction_matrix)
    correction_lu, correction_pivots = jsp.linalg.lu_factor(correction_matrix)
    return PreparedLowRankSolve(
        operator=candidate_operator,
        plan=prepared.plan,
        base_prepared=prepared.base_prepared,
        inverse_left_factor=candidate_inverse_left,
        correction_matrix=correction_matrix,
        correction_lu=correction_lu,
        correction_pivots=correction_pivots,
        correction_condition=correction_condition,
        numeric_version=prepared.numeric_version + 1,
    )


def _set_low_rank_failure(
    status: Array,
    failed: Array,
    failure: LowRankDeterminantStatus,
    /,
) -> Array:
    return jnp.where(
        (status == int(LowRankDeterminantStatus.SUCCESS)) & failed,
        int(failure),
        status,
    )


def _lineage_words(value: Array, /) -> Array:
    array = jnp.asarray(value)
    parts = (
        (jnp.real(array), jnp.imag(array))
        if jnp.issubdtype(array.dtype, jnp.complexfloating)
        else (array,)
    )
    words = []
    for part in parts:
        dtype = np.dtype(part.dtype)
        if jnp.issubdtype(part.dtype, jnp.floating):
            unsigned = {
                8: jnp.uint64,
                4: jnp.uint32,
                2: jnp.uint16,
            }.get(dtype.itemsize, jnp.uint8)
            bits = jax.lax.bitcast_convert_type(part, unsigned)
        else:
            bits = part.astype(jnp.uint64)
        words.append(jnp.ravel(bits).astype(jnp.uint64))
    return jnp.concatenate(words)


def _array_tree_lineage(tree: Any, /) -> Array:
    first = jnp.asarray(np.uint64(1469598103934665603), dtype=jnp.uint64)
    second = jnp.asarray(np.uint64(1099511628211), dtype=jnp.uint64)
    prime = jnp.asarray(np.uint64(1099511628211), dtype=jnp.uint64)
    mixer = jnp.asarray(np.uint64(11400714819323198485), dtype=jnp.uint64)
    leaves = tuple(leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_array(leaf))
    for leaf_index, leaf in enumerate(leaves):
        words = _lineage_words(leaf)
        if words.size == 0:
            continue
        indices = jnp.arange(1, words.size + 1, dtype=jnp.uint64)
        salt = jnp.asarray(np.uint64(leaf_index + 1), dtype=jnp.uint64) * mixer
        mixed = words ^ (indices * mixer + salt)
        first = (first ^ jnp.bitwise_xor.reduce(mixed)) * prime
        second = second + jnp.sum(
            (mixed ^ (mixed >> jnp.asarray(29, dtype=jnp.uint64))) * prime,
            dtype=jnp.uint64,
        )
    return jnp.stack((first, second))


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
    base_failed = jnp.any(
        jnp.asarray(base_result.status) != int(LinearSolveStatus.SUCCESS)
    )
    if plan.policy.failure.mode == "error":
        inverse_left = eqx.error_if(
            inverse_left,
            base_failed,
            "The base solve failed while preparing inverse actions on the "
            "low-rank factor.",
        )
    inverse_left = jnp.where(base_failed, jnp.zeros_like(inverse_left), inverse_left)
    correction_matrix = jnp.eye(operator.rank, dtype=operator.core.dtype) + contract(
        "ab,bn,nc->ac",
        operator.core,
        jnp.conj(jnp.swapaxes(operator.right_factor, 0, 1)),
        inverse_left,
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
    "LowRankDeterminantProvenance",
    "LowRankDeterminantResult",
    "LowRankDeterminantStatus",
    "LowRankPfaffianResult",
    "LowRankPfaffianStatus",
    "LowRankResourcePolicy",
    "LowRankSolveDiagnostics",
    "LowRankSolvePlan",
    "LowRankSolvePolicy",
    "LowRankSolveProvenance",
    "LowRankSolveResult",
    "LowRankSolveStatus",
    "LowRankUpdate",
    "LowRankUpdateRoute",
    "PreparedLowRankSequence",
    "PreparedLowRankSolve",
    "accept_low_rank_update",
    "column_low_rank_update",
    "dense_low_rank_update",
    "plan_low_rank_solve",
    "prepare_factorized_low_rank_sequence",
    "prepare_low_rank_sequence",
    "prepare_low_rank_solve",
    "propose_low_rank_update",
    "propose_pfaffian_update",
    "rebase_low_rank_sequence",
    "refresh_low_rank_sequence",
    "refresh_low_rank_solve",
    "row_low_rank_update",
    "skew_row_column_low_rank_update",
    "solve_low_rank",
    "solve_low_rank_sequence",
]
