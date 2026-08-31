#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._materialization import materialize
from .._operators import AbstractLinearOperator, DenseLinearOperator
from .._pairings import DiagonalPairing, EuclideanPairing
from .._spaces import ArraySpace
from ._schur import (
    prepare_schur_eigensolve,
    PreparedSchurSolve,
    schur_eigensolve,
    SchurEigenproblem,
    SchurSolvePolicy,
)


class ResolventScanStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    SCHUR_FAILURE = 2
    NONFINITE_OUTPUT = 3


class ResolventScanProblem(StrictModule):
    """One standard endomorphism and a fixed vector of complex shifts."""

    operator: AbstractLinearOperator
    shifts: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        shifts: ArrayLike,
        /,
        *,
        problem_id: str | None = None,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape or not operator.source.compatible(operator.target):
            raise ValueError("Resolvent scans require an unbatched endomorphism.")
        if not isinstance(operator.source, ArraySpace):
            raise TypeError("Dense resolvent scans initially require ArraySpace values.")
        shifts_ = jnp.asarray(shifts)
        if shifts_.ndim != 1 or shifts_.size == 0:
            raise ValueError("shifts must be one nonempty rank-one array.")
        if not jnp.issubdtype(shifts_.dtype, jnp.inexact):
            shifts_ = shifts_.astype(float)
        shifts_ = shifts_.astype(jnp.result_type(shifts_.dtype, 1j))
        shifts_ = eqx.error_if(
            shifts_,
            jnp.any(~jnp.isfinite(shifts_)),
            "Resolvent shifts must be finite.",
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "resolvent-scan-problem",
                    "operator": operator.operator_id,
                    "source": operator.source.space_id,
                    "shifts": array_tree_fingerprint(shifts_),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.operator = operator
        self.shifts = shifts_
        self.problem_id = identifier


class ResolventScanPolicy(StrictModule, NonTrainableState):
    schur: SchurSolvePolicy
    relative_singularity_tolerance: float = eqx.field(static=True)
    absolute_singularity_tolerance: float = eqx.field(static=True)
    maximum_shifts: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        schur: SchurSolvePolicy | None = None,
        relative_singularity_tolerance: float = 1e-12,
        absolute_singularity_tolerance: float = 0.0,
        maximum_shifts: int = 65_536,
        maximum_workspace_bytes: int = 1024 * 1024**2,
    ):
        schur_ = SchurSolvePolicy() if schur is None else schur
        relative = float(relative_singularity_tolerance)
        absolute = float(absolute_singularity_tolerance)
        maximum = int(maximum_shifts)
        workspace = int(maximum_workspace_bytes)
        if not isinstance(schur_, SchurSolvePolicy):
            raise TypeError("schur must be a SchurSolvePolicy or None.")
        if (
            not math.isfinite(relative)
            or not math.isfinite(absolute)
            or relative < 0.0
            or absolute < 0.0
            or maximum <= 0
            or workspace <= 0
        ):
            raise ValueError("Resolvent scan policy values are invalid.")
        self.schur = schur_
        self.relative_singularity_tolerance = relative
        self.absolute_singularity_tolerance = absolute
        self.maximum_shifts = maximum
        self.maximum_workspace_bytes = workspace


class ResolventScanPlan(StrictModule, NonTrainableState):
    policy: ResolventScanPolicy
    dimension: int = eqx.field(static=True)
    shift_count: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedResolventScan(StrictModule):
    problem: ResolventScanProblem
    schur: PreparedSchurSolve
    plan: ResolventScanPlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array


class ResolventScanDiagnostics(StrictModule):
    minimum_singular_values: Array
    singular_mask: Array
    finite: Array
    schur_status: Array
    decomposition_count: Array
    workspace_bytes: int = eqx.field(static=True)


class ResolventScanProvenance(StrictModule):
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    pairing_id: str = eqx.field(static=True)
    norm_definition: str = eqx.field(static=True)
    numeric_version: Array


class ResolventScanResult(StrictModule):
    shifts: Array
    minimum_singular_values: Array
    resolvent_norms: Array
    status: Array
    diagnostics: ResolventScanDiagnostics
    provenance: ResolventScanProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(ResolventScanStatus.SUCCESS)


def plan_resolvent_scan(
    problem: ResolventScanProblem,
    policy: ResolventScanPolicy | None = None,
    /,
) -> ResolventScanPlan:
    if not isinstance(problem, ResolventScanProblem):
        raise TypeError("problem must be a ResolventScanProblem.")
    policy_ = ResolventScanPolicy() if policy is None else policy
    if not isinstance(policy_, ResolventScanPolicy):
        raise TypeError("policy must be a ResolventScanPolicy or None.")
    dimension = problem.operator.source.size
    shift_count = int(problem.shifts.size)
    if shift_count > policy_.maximum_shifts:
        raise ValueError("Resolvent shift count exceeds maximum_shifts.")
    itemsize = np.dtype(
        jnp.result_type(problem.shifts.dtype, problem.operator.source.dtype)
    ).itemsize
    workspace = shift_count * dimension * dimension * itemsize
    if workspace > policy_.maximum_workspace_bytes:
        raise ValueError("Resolvent scan exceeds maximum_workspace_bytes.")
    return ResolventScanPlan(
        policy=policy_,
        dimension=dimension,
        shift_count=shift_count,
        workspace_bytes=workspace,
        problem_id=problem.problem_id,
        source_space_id=problem.operator.source.space_id,
        operator_id=problem.operator.operator_id,
        plan_id=canonical_fingerprint(
            {
                "kind": "resolvent-scan-plan",
                "problem": problem.problem_id,
                "operator": problem.operator.operator_id,
                "dimension": dimension,
                "shift_count": shift_count,
                "relative_tolerance": policy_.relative_singularity_tolerance,
                "absolute_tolerance": policy_.absolute_singularity_tolerance,
                "workspace_bytes": workspace,
            }
        ),
    )


def prepare_resolvent_scan(
    problem: ResolventScanProblem,
    policy: ResolventScanPolicy | ResolventScanPlan | None = None,
    /,
) -> PreparedResolventScan:
    plan = (
        policy
        if isinstance(policy, ResolventScanPlan)
        else plan_resolvent_scan(problem, policy)
    )
    return _prepare_resolvent(problem, plan, numeric_version=0)


def refresh_resolvent_scan(
    prepared: PreparedResolventScan,
    problem: ResolventScanProblem,
    /,
) -> PreparedResolventScan:
    if not isinstance(prepared, PreparedResolventScan):
        raise TypeError("prepared must be a PreparedResolventScan.")
    return _prepare_resolvent(
        problem,
        prepared.plan,
        numeric_version=int(np.asarray(prepared.numeric_version)) + 1,
        prepared_id=prepared.prepared_id,
    )


def resolvent_scan(
    problem_or_prepared: ResolventScanProblem | PreparedResolventScan,
    /,
    *,
    policy: ResolventScanPolicy | ResolventScanPlan | None = None,
) -> ResolventScanResult:
    if isinstance(problem_or_prepared, PreparedResolventScan):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared resolvent scan.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, ResolventScanProblem):
        prepared = prepare_resolvent_scan(problem_or_prepared, policy)
    else:
        raise TypeError("Expected a ResolventScanProblem or PreparedResolventScan.")
    schur = schur_eigensolve(prepared.schur)
    form = schur.schur_form
    identity = jnp.eye(prepared.plan.dimension, dtype=form.dtype)

    def minimum_singular_value(shift):
        singular_values = jnp.linalg.svd(
            form - shift.astype(form.dtype) * identity,
            full_matrices=False,
            compute_uv=False,
        )
        return singular_values[-1]

    minimum = jax.vmap(minimum_singular_value)(prepared.problem.shifts)
    scale = jnp.maximum(jnp.linalg.norm(form), jnp.asarray(1.0, dtype=form.real.dtype))
    tolerance = (
        prepared.plan.policy.absolute_singularity_tolerance
        + prepared.plan.policy.relative_singularity_tolerance * scale
    )
    singular = minimum <= tolerance
    norms = jnp.where(singular, jnp.inf, jnp.reciprocal(minimum))
    input_finite = jnp.all(jnp.isfinite(prepared.problem.shifts)) & jnp.all(
        jnp.isfinite(prepared.schur.matrix)
    )
    output_finite = jnp.all(jnp.isfinite(minimum))
    status = jnp.where(
        ~input_finite,
        int(ResolventScanStatus.NONFINITE_INPUT),
        jnp.where(
            ~schur.successful,
            int(ResolventScanStatus.SCHUR_FAILURE),
            jnp.where(
                ~output_finite,
                int(ResolventScanStatus.NONFINITE_OUTPUT),
                int(ResolventScanStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    pairing = prepared.problem.operator.source.pairing
    return ResolventScanResult(
        shifts=prepared.problem.shifts,
        minimum_singular_values=minimum,
        resolvent_norms=norms,
        status=status,
        diagnostics=ResolventScanDiagnostics(
            minimum_singular_values=minimum,
            singular_mask=singular,
            finite=input_finite & output_finite,
            schur_status=schur.status,
            decomposition_count=schur.diagnostics.decomposition_count,
            workspace_bytes=prepared.plan.workspace_bytes,
        ),
        provenance=ResolventScanProvenance(
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            source_space_id=prepared.problem.operator.source.space_id,
            pairing_id=pairing.pairing_id,
            norm_definition="induced Hilbert norm after pairing square-root coordinates",
            numeric_version=prepared.numeric_version,
        ),
    )


def _prepare_resolvent(
    problem: ResolventScanProblem,
    plan: ResolventScanPlan,
    /,
    *,
    numeric_version: int,
    prepared_id: str | None = None,
) -> PreparedResolventScan:
    if not isinstance(problem, ResolventScanProblem) or not isinstance(
        plan, ResolventScanPlan
    ):
        raise TypeError("Resolvent preparation requires a problem and plan.")
    if (
        problem.problem_id != plan.problem_id
        or problem.operator.operator_id != plan.operator_id
        or problem.operator.source.size != plan.dimension
        or int(problem.shifts.size) != plan.shift_count
        or problem.operator.source.space_id != plan.source_space_id
    ):
        raise ValueError("Resolvent problem is incompatible with the symbolic plan.")
    matrix = materialize(
        problem.operator,
        plan.policy.schur.materialization,
    )
    canonical = _canonical_pairing_matrix(problem.operator.source, matrix)
    space = ArraySpace((plan.dimension,), dtype=canonical.dtype)
    canonical_operator = DenseLinearOperator(
        canonical,
        source=space,
        target=space,
        operator_id=canonical_fingerprint(
            {
                "kind": "pairing-canonical-resolvent-operator",
                "operator": problem.operator.operator_id,
                "pairing": problem.operator.source.pairing.pairing_id,
            }
        ),
    )
    schur = prepare_schur_eigensolve(
        SchurEigenproblem(canonical_operator),
        plan.policy.schur,
    )
    version = jnp.asarray(numeric_version, dtype=jnp.int32)
    return PreparedResolventScan(
        problem=problem,
        schur=schur,
        plan=plan,
        prepared_id=(
            canonical_fingerprint(
                {
                    "kind": "prepared-resolvent-scan",
                    "plan": plan.plan_id,
                    "operator": problem.operator.operator_id,
                    "shifts": array_tree_fingerprint(problem.shifts),
                    "numeric_version": numeric_version,
                }
            )
            if prepared_id is None
            else str(prepared_id)
        ),
        numeric_version=version,
    )


def _canonical_pairing_matrix(space: ArraySpace, matrix: Array, /) -> Array:
    pairing = space.pairing
    if isinstance(pairing, EuclideanPairing):
        return jnp.asarray(matrix)
    if isinstance(pairing, DiagonalPairing):
        weights = jnp.asarray(pairing.weights).reshape((-1,))
        square_root = jnp.sqrt(weights)
        return square_root[:, None] * jnp.asarray(matrix) / square_root[None, :]
    raise TypeError("Resolvent scans require Euclidean or positive diagonal pairings.")


__all__ = [
    "PreparedResolventScan",
    "ResolventScanDiagnostics",
    "ResolventScanPlan",
    "ResolventScanPolicy",
    "ResolventScanProblem",
    "ResolventScanProvenance",
    "ResolventScanResult",
    "ResolventScanStatus",
    "plan_resolvent_scan",
    "prepare_resolvent_scan",
    "refresh_resolvent_scan",
    "resolvent_scan",
]
