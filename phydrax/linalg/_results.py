#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._differentiation import DerivativeContract, DerivativeRoute, DerivativeSurface
from .._iteration import IterationEvidence
from .._strict import StrictModule
from ._policies import DifferentiationMode, DifferentiationPolicy, MixedPrecisionPolicy
from ._recycling import RecyclingState


class LinearSolveStatus(IntEnum):
    SUCCESS = 0
    MAXIMUM_STEPS_REACHED = 1
    STAGNATION = 2
    BREAKDOWN = 3
    SINGULAR = 4
    RANK_DEFICIENT = 5
    NONFINITE_INPUT = 6
    NONFINITE_OUTPUT = 7
    RESIDUAL_TOO_LARGE = 8
    CAPABILITY_REJECTED = 9
    INCOMPATIBLE_STRUCTURE = 10
    ADJOINT_FAILED = 11
    CONDITION_LIMIT_REACHED = 12
    USER_STOPPED = 13


_STATUS_MESSAGES = {
    LinearSolveStatus.SUCCESS: "success",
    LinearSolveStatus.MAXIMUM_STEPS_REACHED: "maximum steps reached",
    LinearSolveStatus.STAGNATION: "iteration stagnated",
    LinearSolveStatus.BREAKDOWN: "iterative breakdown",
    LinearSolveStatus.SINGULAR: "operator is singular or numerically singular",
    LinearSolveStatus.RANK_DEFICIENT: "declared rank policy was not satisfied",
    LinearSolveStatus.NONFINITE_INPUT: "operator or right-hand side is non-finite",
    LinearSolveStatus.NONFINITE_OUTPUT: "solution contains non-finite values",
    LinearSolveStatus.RESIDUAL_TOO_LARGE: "true residual exceeds the requested tolerance",
    LinearSolveStatus.CAPABILITY_REJECTED: "operator lacks a required capability",
    LinearSolveStatus.INCOMPATIBLE_STRUCTURE: "problem structure is incompatible",
    LinearSolveStatus.ADJOINT_FAILED: "adjoint solve failed",
    LinearSolveStatus.CONDITION_LIMIT_REACHED: "condition limit reached",
    LinearSolveStatus.USER_STOPPED: "stopped by the iteration control rule",
}


def linear_status_message(status: int | LinearSolveStatus, /) -> str:
    return _STATUS_MESSAGES[LinearSolveStatus(int(status))]


class LinearSolveDiagnostics(StrictModule):
    """JAX-compatible evidence retained per operator batch or right-hand side."""

    residual_norm: Array
    relative_residual: Array
    normal_residual_norm: Array
    iterations: Array
    rank: Array
    condition_estimate: Array
    finite: Array
    converged: Array
    singular_values: Array | None
    rank_cutoff: Array
    compatibility_residual: Array
    gauge_residual: Array
    nullity: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    effective_block_rank: Array
    deflated_rhs_count: Array
    refinement_steps: Array

    def __init__(
        self,
        *,
        residual_norm: Any,
        relative_residual: Any,
        normal_residual_norm: Any = jnp.nan,
        iterations: Any = 0,
        rank: Any = -1,
        condition_estimate: Any = jnp.nan,
        finite: Any = True,
        converged: Any = True,
        singular_values: Any | None = None,
        rank_cutoff: Any = jnp.nan,
        compatibility_residual: Any = 0.0,
        gauge_residual: Any = 0.0,
        nullity: Any = -1,
        matvec_count: Any = 0,
        adjoint_matvec_count: Any = 0,
        effective_block_rank: Any = -1,
        deflated_rhs_count: Any = 0,
        refinement_steps: Any = 0,
    ):
        self.residual_norm = jnp.asarray(residual_norm)
        self.relative_residual = jnp.asarray(relative_residual)
        self.normal_residual_norm = jnp.asarray(normal_residual_norm)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.rank = jnp.asarray(rank, dtype=jnp.int32)
        self.condition_estimate = jnp.asarray(condition_estimate)
        self.rank_cutoff = jnp.asarray(rank_cutoff)
        self.finite = jnp.asarray(finite, dtype=jnp.bool_)
        self.converged = jnp.asarray(converged, dtype=jnp.bool_)
        self.compatibility_residual = jnp.asarray(compatibility_residual)
        self.gauge_residual = jnp.asarray(gauge_residual)
        self.nullity = jnp.asarray(nullity, dtype=jnp.int32)
        self.matvec_count = jnp.asarray(matvec_count, dtype=jnp.int32)
        self.adjoint_matvec_count = jnp.asarray(adjoint_matvec_count, dtype=jnp.int32)
        self.effective_block_rank = jnp.asarray(
            effective_block_rank,
            dtype=jnp.int32,
        )
        self.deflated_rhs_count = jnp.asarray(
            deflated_rhs_count,
            dtype=jnp.int32,
        )
        self.refinement_steps = jnp.asarray(refinement_steps, dtype=jnp.int32)
        self.singular_values = (
            None if singular_values is None else jnp.asarray(singular_values)
        )


class LinearIterationMetrics(StrictModule):
    """Portable scalar or batched metrics for a linear iteration."""

    residual_norm: Array
    relative_residual: Array
    normal_residual_norm: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    condition_estimate: Array
    breakdown_status: Array

    def __init__(
        self,
        *,
        residual_norm,
        relative_residual,
        normal_residual_norm=jnp.nan,
        iterations=0,
        matvec_count=0,
        adjoint_matvec_count=0,
        condition_estimate=jnp.nan,
        breakdown_status=0,
    ):
        self.residual_norm = jnp.asarray(residual_norm)
        self.relative_residual = jnp.asarray(relative_residual)
        self.normal_residual_norm = jnp.asarray(normal_residual_norm)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.matvec_count = jnp.asarray(matvec_count, dtype=jnp.int32)
        self.adjoint_matvec_count = jnp.asarray(adjoint_matvec_count, dtype=jnp.int32)
        self.condition_estimate = jnp.asarray(condition_estimate)
        self.breakdown_status = jnp.asarray(breakdown_status, dtype=jnp.int32)


class LinearPrecisionEvidence(StrictModule):
    """Requested-stage resolution for one capability-checked linear execution."""

    operator_dtype: str = eqx.field(static=True)
    factorization_dtype: str | None = eqx.field(static=True)
    preconditioner_dtype: str | None = eqx.field(static=True)
    krylov_dtype: str | None = eqx.field(static=True)
    residual_dtype: str = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    condition_limit: float | None = eqx.field(static=True)
    maximum_refinement_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        operator_dtype: str,
        factorization_dtype: str | None,
        preconditioner_dtype: str | None,
        krylov_dtype: str | None,
        residual_dtype: str,
        accumulation_dtype: str,
        condition_limit: float | None,
        maximum_refinement_steps: int,
    ):
        if not operator_dtype or not residual_dtype or not accumulation_dtype:
            raise ValueError(
                "Effective operator, residual, and accumulation dtypes must be non-empty."
            )
        optional_dtypes = (
            factorization_dtype,
            preconditioner_dtype,
            krylov_dtype,
        )
        if any(value is not None and not value for value in optional_dtypes):
            raise ValueError("Effective precision dtype names must be non-empty.")
        limit = None if condition_limit is None else float(condition_limit)
        steps = int(maximum_refinement_steps)
        if limit is not None and (not math.isfinite(limit) or limit <= 1.0):
            raise ValueError("Effective condition_limit must exceed one.")
        if steps < 0:
            raise ValueError("Effective refinement work must be non-negative.")
        self.operator_dtype = str(operator_dtype)
        self.factorization_dtype = (
            None if factorization_dtype is None else str(factorization_dtype)
        )
        self.preconditioner_dtype = (
            None if preconditioner_dtype is None else str(preconditioner_dtype)
        )
        self.krylov_dtype = None if krylov_dtype is None else str(krylov_dtype)
        self.residual_dtype = str(residual_dtype)
        self.condition_limit = limit
        self.maximum_refinement_steps = steps

        self.accumulation_dtype = str(accumulation_dtype)


class LinearSolveProvenance(StrictModule):
    """Static provider selection, candidate rejections, and plan identity."""

    backend: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    rejected: tuple[str, ...] = eqx.field(static=True)
    prepared: bool = eqx.field(static=True)
    rhs_mode: str = eqx.field(static=True)
    preconditioner_plan_id: str | None = eqx.field(static=True)
    preconditioner_id: str | None = eqx.field(static=True)
    preconditioning_side: str | None = eqx.field(static=True)
    preconditioner_refresh: str | None = eqx.field(static=True)
    preconditioner_numeric_version: Array
    preconditioner_built_numeric_version: Array
    preconditioner_storage_bytes: int = eqx.field(static=True)
    preconditioner_preparation_workspace_bytes: int = eqx.field(static=True)
    preconditioner_apply_workspace_bytes_per_rhs: int = eqx.field(static=True)
    preconditioner_setup_matvec_count: int = eqx.field(static=True)
    operator_numeric_version: Array
    recycling_capacity: int = eqx.field(static=True)
    recycling_state_bytes: int = eqx.field(static=True)
    recycling_update_count: Array
    requested_precision: MixedPrecisionPolicy | None
    effective_precision: LinearPrecisionEvidence | None

    def __init__(
        self,
        *,
        backend: str,
        method: str,
        plan_id: str,
        problem_id: str,
        reason: str,
        rejected: tuple[str, ...] = (),
        prepared: bool,
        rhs_mode: str = "single",
        preconditioner_plan_id: str | None = None,
        preconditioner_id: str | None = None,
        preconditioning_side: str | None = None,
        preconditioner_refresh: str | None = None,
        preconditioner_numeric_version: Any = -1,
        preconditioner_built_numeric_version: Any = -1,
        preconditioner_storage_bytes: int = 0,
        preconditioner_preparation_workspace_bytes: int = 0,
        preconditioner_apply_workspace_bytes_per_rhs: int = 0,
        preconditioner_setup_matvec_count: int = 0,
        operator_numeric_version: Any = 0,
        recycling_capacity: int = 0,
        recycling_state_bytes: int = 0,
        recycling_update_count: Any = 0,
        requested_precision: MixedPrecisionPolicy | None = None,
        effective_precision: LinearPrecisionEvidence | None = None,
    ):
        values = (
            str(backend),
            str(method),
            str(plan_id),
            str(problem_id),
            str(reason),
        )
        if any(not value for value in values):
            raise ValueError("Solve provenance identifiers and reason must be non-empty.")
        (
            self.backend,
            self.method,
            self.plan_id,
            self.problem_id,
            self.reason,
        ) = values
        self.rejected = tuple(str(value) for value in rejected)
        if rhs_mode not in ("single", "pseudo-block", "true-block"):
            raise ValueError(
                "rhs_mode must be 'single', 'pseudo-block', or 'true-block'."
            )
        self.rhs_mode = rhs_mode
        self.prepared = bool(prepared)
        if (requested_precision is None) != (effective_precision is None):
            raise ValueError(
                "Requested and effective precision evidence must be present together."
            )
        if requested_precision is not None and not isinstance(
            requested_precision,
            MixedPrecisionPolicy,
        ):
            raise TypeError("requested_precision must be a MixedPrecisionPolicy or None.")
        if effective_precision is not None and not isinstance(
            effective_precision,
            LinearPrecisionEvidence,
        ):
            raise TypeError(
                "effective_precision must be a LinearPrecisionEvidence or None."
            )
        self.requested_precision = requested_precision
        self.effective_precision = effective_precision
        optional_identifiers = (
            preconditioner_plan_id,
            preconditioner_id,
            preconditioner_refresh,
        )
        if any(value is not None and not str(value) for value in optional_identifiers):
            raise ValueError("Preconditioner provenance identifiers must be non-empty.")
        if preconditioning_side not in (None, "left", "right"):
            raise ValueError("preconditioning_side must be 'left', 'right', or None.")
        self.preconditioner_plan_id = (
            None if preconditioner_plan_id is None else str(preconditioner_plan_id)
        )
        self.preconditioner_id = (
            None if preconditioner_id is None else str(preconditioner_id)
        )
        self.preconditioning_side = preconditioning_side
        self.preconditioner_refresh = (
            None if preconditioner_refresh is None else str(preconditioner_refresh)
        )
        preconditioner_version = jnp.asarray(
            preconditioner_numeric_version,
            dtype=jnp.int32,
        )
        built_version = jnp.asarray(
            preconditioner_built_numeric_version,
            dtype=jnp.int32,
        )
        if preconditioner_version.ndim != 0 or built_version.ndim != 0:
            raise ValueError("Preconditioner provenance versions must be scalar.")
        invalid_versions = (
            (preconditioner_version < -1)
            | (built_version < -1)
            | ((preconditioner_version == -1) != (built_version == -1))
            | ((preconditioner_version >= 0) & (built_version > preconditioner_version))
        )
        self.preconditioner_numeric_version = eqx.error_if(
            preconditioner_version,
            invalid_versions,
            "Preconditioner provenance versions are invalid.",
        )
        self.preconditioner_built_numeric_version = eqx.error_if(
            built_version,
            invalid_versions,
            "Preconditioner provenance versions are invalid.",
        )
        preconditioner_costs = (
            int(preconditioner_storage_bytes),
            int(preconditioner_preparation_workspace_bytes),
            int(preconditioner_apply_workspace_bytes_per_rhs),
            int(preconditioner_setup_matvec_count),
        )
        if any(value < 0 for value in preconditioner_costs):
            raise ValueError("Preconditioner provenance costs must be non-negative.")
        (
            self.preconditioner_storage_bytes,
            self.preconditioner_preparation_workspace_bytes,
            self.preconditioner_apply_workspace_bytes_per_rhs,
            self.preconditioner_setup_matvec_count,
        ) = preconditioner_costs
        operator_version = jnp.asarray(operator_numeric_version, dtype=jnp.int32)
        recycling_costs = (
            int(recycling_capacity),
            int(recycling_state_bytes),
        )
        if operator_version.ndim != 0:
            raise ValueError("operator_numeric_version must be scalar.")
        operator_version = eqx.error_if(
            operator_version,
            operator_version < 0,
            "operator_numeric_version must be non-negative.",
        )
        if any(value < 0 for value in recycling_costs):
            raise ValueError("Recycling provenance costs must be non-negative.")
        self.operator_numeric_version = operator_version
        self.recycling_capacity, self.recycling_state_bytes = recycling_costs
        self.recycling_update_count = jnp.asarray(
            recycling_update_count,
            dtype=jnp.int32,
        )
        if self.recycling_update_count.ndim != 0:
            raise ValueError("recycling_update_count must be scalar.")


_SOLVE_CONVERGED = "solve-converged"
_DECISIONS_FROZEN = "decisions-frozen"


def _linear_solve_derivative_contract(mode: DifferentiationMode, /) -> DerivativeContract:
    """Return the canonical derivative contract of one differentiation mode.

    The right-hand side is the `SOLVER_ARGUMENT` surface and the operator arrays
    are the `PHYSICAL_PARAMETER` surface. Implicit contracts hold at a converged
    solution; unrolled contracts differentiate the executed iteration with its
    stopping and pivoting decisions held fixed.
    """
    both = (DerivativeSurface.SOLVER_ARGUMENT, DerivativeSurface.PHYSICAL_PARAMETER)
    match mode:
        case "mathematical":
            return DerivativeContract.smooth(
                both, route=DerivativeRoute.IMPLICIT, conditions=(_SOLVE_CONVERGED,)
            )
        case "rhs-only":
            return DerivativeContract.smooth(
                (DerivativeSurface.SOLVER_ARGUMENT,),
                route=DerivativeRoute.IMPLICIT,
                conditions=(_SOLVE_CONVERGED,),
            )
        case "algorithmic":
            return DerivativeContract.smooth(
                both, route=DerivativeRoute.UNROLLED, conditions=(_DECISIONS_FROZEN,)
            )
        case "none":
            return DerivativeContract(route=DerivativeRoute.STOPPED)
        case _:
            raise ValueError(f"Unknown differentiation mode {mode!r}.")


class LinearSolveResult(StrictModule):
    """Numerical value plus portable status, diagnostics, and provenance.

    `derivative_contract` is the canonical contract of the executed
    `DifferentiationPolicy`: `"mathematical"` is an implicit derivative in the
    right-hand side (`SOLVER_ARGUMENT`) and operator (`PHYSICAL_PARAMETER`);
    `"rhs-only"` is implicit in the right-hand side only, with the operator
    stopped; `"algorithmic"` unrolls the executed iteration; `"none"` is stopped.
    `derivative_valid` reports, per right-hand side, whether that contract holds
    for this solve.
    """

    value: PyTree[Array]
    status: Array
    diagnostics: LinearSolveDiagnostics
    provenance: LinearSolveProvenance
    iteration_evidence: IterationEvidence | None
    derivative_contract: DerivativeContract

    def __init__(
        self,
        value: PyTree[Array],
        status: Any,
        diagnostics: LinearSolveDiagnostics,
        provenance: LinearSolveProvenance,
        /,
        *,
        differentiation: DifferentiationPolicy,
        iteration_evidence: IterationEvidence | None = None,
    ):
        if not isinstance(diagnostics, LinearSolveDiagnostics):
            raise TypeError("diagnostics must be LinearSolveDiagnostics.")
        if not isinstance(provenance, LinearSolveProvenance):
            raise TypeError("provenance must be LinearSolveProvenance.")
        if not isinstance(differentiation, DifferentiationPolicy):
            raise TypeError("differentiation must be a DifferentiationPolicy.")
        if iteration_evidence is not None and not isinstance(
            iteration_evidence, IterationEvidence
        ):
            raise TypeError("iteration_evidence must be IterationEvidence or None.")
        self.value = value
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.iteration_evidence = iteration_evidence
        self.derivative_contract = _linear_solve_derivative_contract(differentiation.mode)

    @property
    def successful(self) -> Array:
        return self.status == int(LinearSolveStatus.SUCCESS)

    @property
    def derivative_valid(self) -> Array:
        """Whether `derivative_contract` holds for each right-hand side.

        Implicit derivatives require a converged solve, the same evidence that
        guards the returned value's derivative; unrolled derivatives of the
        executed iteration require finite arithmetic; a stopped contract claims
        no derivative.
        """
        match self.derivative_contract.route:
            case DerivativeRoute.IMPLICIT:
                return self.diagnostics.converged
            case DerivativeRoute.UNROLLED:
                return self.diagnostics.finite
            case DerivativeRoute.STOPPED:
                return jnp.zeros_like(self.diagnostics.converged)
            case route:
                raise ValueError(f"Linear solves do not use the {route.value} route.")


MatrixInversionKind: TypeAlias = Literal["inverse", "pseudoinverse"]


class MatrixInversionResult(StrictModule):
    """Explicit inverse matrix plus batch-level numerical evidence."""

    value: Array
    status: Array
    diagnostics: LinearSolveDiagnostics
    provenance: LinearSolveProvenance
    operation: MatrixInversionKind = eqx.field(static=True)

    def __init__(
        self,
        value: Any,
        status: Any,
        diagnostics: LinearSolveDiagnostics,
        provenance: LinearSolveProvenance,
        operation: MatrixInversionKind,
        /,
    ):
        if not isinstance(diagnostics, LinearSolveDiagnostics):
            raise TypeError("diagnostics must be LinearSolveDiagnostics.")
        if not isinstance(provenance, LinearSolveProvenance):
            raise TypeError("provenance must be LinearSolveProvenance.")
        if operation not in ("inverse", "pseudoinverse"):
            raise ValueError("operation must be 'inverse' or 'pseudoinverse'.")
        matrix = jnp.asarray(value)
        if matrix.ndim < 2:
            raise ValueError("Matrix inversion values must have at least two axes.")
        self.value = matrix
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.operation = operation

    @property
    def successful(self) -> Array:
        return self.status == int(LinearSolveStatus.SUCCESS)


LinearSolveCheckKind: TypeAlias = Literal["primal", "adjoint"]


class LinearSolveCheckEvidence(StrictModule):
    """Fail-closed assessment of one computed primal or adjoint solve."""

    status: Array
    true_residual_norm: Array
    rhs_norm: Array
    relative_residual: Array
    residual_threshold: Array
    finite: Array
    stability_lower_bound: Array
    forward_error_upper_bound: Array
    forward_error_bound_available: Array
    forward_error_bound_certified: Array
    status_ok: Array
    converged: Array
    residual_ok: Array
    stability_ok: Array
    compatibility_residual: Array
    gauge_residual: Array
    nullspace_ok: Array
    primal_valid: Array
    valid: Array
    kind: LinearSolveCheckKind = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    stability_checked: bool = eqx.field(static=True)
    stability_certificate_id: str | None = eqx.field(static=True)
    stability_evidence: str | None = eqx.field(static=True)
    stability_scope: str | None = eqx.field(static=True)
    nullspace_checked: bool = eqx.field(static=True)
    nullspace_certificate_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: LinearSolveCheckKind,
        operator_id: str,
        status: Any,
        true_residual_norm: Any,
        rhs_norm: Any,
        residual_threshold: Any,
        finite: Any,
        converged: Any,
        stability_lower_bound: Any,
        stability_checked: bool,
        stability_ok: Any,
        stability_certificate_id: str | None,
        stability_evidence: str | None,
        stability_scope: str | None,
        compatibility_residual: Any,
        gauge_residual: Any,
        nullspace_checked: bool,
        nullspace_ok: Any,
        nullspace_certificate_id: str | None,
        primal_valid: Any = True,
    ):
        if kind not in ("primal", "adjoint"):
            raise ValueError("kind must be 'primal' or 'adjoint'.")
        identifier = str(operator_id)
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        status_ = jnp.asarray(status, dtype=jnp.int32)
        shape = status_.shape

        def broadcast(value: Any, dtype: Any | None = None) -> Array:
            array = jnp.asarray(value, dtype=dtype)
            return jnp.broadcast_to(array, shape)

        residual = broadcast(true_residual_norm)
        rhs = broadcast(rhs_norm)
        threshold = broadcast(residual_threshold)
        finite_ = broadcast(finite, bool)
        stability_bound = broadcast(stability_lower_bound)
        converged_ = broadcast(converged, bool)
        stability_ok_ = broadcast(stability_ok, bool)
        forward_available = (
            jnp.asarray(stability_checked)
            & stability_ok_
            & finite_
            & jnp.isfinite(residual)
            & jnp.isfinite(stability_bound)
            & (stability_bound > 0.0)
        )
        forward_bound = jnp.where(
            forward_available,
            residual / jnp.where(stability_bound > 0.0, stability_bound, 1.0),
            jnp.asarray(jnp.inf, dtype=residual.dtype),
        )
        forward_certified = forward_available & jnp.asarray(
            stability_evidence in ("construction", "verified")
        )
        compatibility = broadcast(compatibility_residual)
        gauge = broadcast(gauge_residual)
        nullspace_ok_ = broadcast(nullspace_ok, bool)
        primal_valid_ = broadcast(primal_valid, bool)
        status_ok = status_ == int(LinearSolveStatus.SUCCESS)
        residual_ok = (
            jnp.isfinite(residual)
            & jnp.isfinite(rhs)
            & jnp.isfinite(threshold)
            & (residual <= threshold)
        )
        relative = jnp.where(rhs > 0.0, residual / rhs, residual)
        valid = (
            finite_
            & status_ok
            & converged_
            & residual_ok
            & stability_ok_
            & nullspace_ok_
            & primal_valid_
        )
        self.status = status_
        self.true_residual_norm = residual
        self.rhs_norm = rhs
        self.relative_residual = relative
        self.residual_threshold = threshold
        self.finite = finite_
        self.stability_lower_bound = stability_bound
        self.forward_error_upper_bound = forward_bound
        self.forward_error_bound_available = forward_available
        self.forward_error_bound_certified = forward_certified
        self.status_ok = status_ok
        self.converged = converged_
        self.residual_ok = residual_ok
        self.stability_ok = stability_ok_
        self.compatibility_residual = compatibility
        self.gauge_residual = gauge
        self.nullspace_ok = nullspace_ok_
        self.primal_valid = primal_valid_
        self.valid = valid
        self.kind = kind
        self.operator_id = identifier
        self.stability_checked = bool(stability_checked)
        self.stability_certificate_id = (
            None if stability_certificate_id is None else str(stability_certificate_id)
        )
        self.stability_evidence = (
            None if stability_evidence is None else str(stability_evidence)
        )
        self.stability_scope = None if stability_scope is None else str(stability_scope)
        self.nullspace_checked = bool(nullspace_checked)
        self.nullspace_certificate_id = (
            None if nullspace_certificate_id is None else str(nullspace_certificate_id)
        )


class RecycledLinearSolveResult(StrictModule):
    """An ordinary linear result paired with immutable updated recycling state."""

    result: LinearSolveResult
    recycling: RecyclingState

    def __init__(
        self,
        result: LinearSolveResult,
        recycling: RecyclingState,
        /,
    ):
        if not isinstance(result, LinearSolveResult):
            raise TypeError("result must be a LinearSolveResult.")
        if not isinstance(recycling, RecyclingState):
            raise TypeError("recycling must be a RecyclingState.")
        self.result = eqx.tree_at(
            lambda value: value.provenance.recycling_update_count,
            result,
            recycling.update_count,
        )
        self.recycling = recycling

    @property
    def value(self) -> PyTree[Array]:
        return self.result.value

    @property
    def status(self) -> Array:
        return self.result.status

    @property
    def diagnostics(self) -> LinearSolveDiagnostics:
        return self.result.diagnostics

    @property
    def provenance(self) -> LinearSolveProvenance:
        return self.result.provenance

    @property
    def iteration_evidence(self) -> IterationEvidence | None:
        return self.result.iteration_evidence

    @property
    def successful(self) -> Array:
        return self.result.successful

    @property
    def derivative_contract(self) -> DerivativeContract:
        return self.result.derivative_contract

    @property
    def derivative_valid(self) -> Array:
        return self.result.derivative_valid


__all__ = [
    "LinearIterationMetrics",
    "LinearPrecisionEvidence",
    "LinearSolveCheckEvidence",
    "LinearSolveCheckKind",
    "LinearSolveDiagnostics",
    "LinearSolveProvenance",
    "LinearSolveResult",
    "LinearSolveStatus",
    "MatrixInversionKind",
    "MatrixInversionResult",
    "RecycledLinearSolveResult",
    "linear_status_message",
]
