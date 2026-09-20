#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._certificates import SpectralInterval
from ._operators import AbstractLinearOperator
from ._spaces import _coordinate_dtype
from .backends._native_shifted_krylov import streaming_shifted_lanczos
from .krylov import (
    KrylovBreakdownStatus,
    KrylovProjectionPlan,
    KrylovProjectionPolicy,
    plan_krylov_projection,
    prepare_krylov_projection,
    PreparedKrylovProjection,
    refresh_krylov_projection,
)


ShiftedKrylovMethod: TypeAlias = Literal["auto", "arnoldi", "lanczos"]
ShiftedExecutionMode: TypeAlias = Literal["retained", "streaming"]
ShiftedDifferentiationMode: TypeAlias = Literal["runtime-shifts", "none"]


class ShiftedSolveStatus(IntEnum):
    """Portable per-shift status for one shared shifted family."""

    SUCCESS = 0
    MAX_DIMENSION_REACHED = 1
    SINGULAR = 2
    NONFINITE = 3
    KRYLOV_FAILURE = 4
    INADMISSIBLE_SHIFT = 5


class ShiftedLinearSystemFamily(StrictModule):
    """Systems ``(z_j I - A) x_j = b`` with optional spectral evidence."""

    operator: AbstractLinearOperator
    shifts: Array
    spectral_interval: SpectralInterval | None
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        shifts: ArrayLike,
        /,
        *,
        spectral_interval: SpectralInterval | None = None,
        family_id: str | None = None,
    ):
        _validate_operator(operator)
        values = jnp.asarray(shifts)
        if values.ndim != 1 or values.size < 1:
            raise ValueError("shifts must be one nonempty rank-one array.")
        if not jnp.issubdtype(values.dtype, jnp.number) or jnp.issubdtype(
            values.dtype, jnp.bool_
        ):
            raise TypeError("shifts must contain real or complex numbers.")
        dtype = jnp.result_type(_coordinate_dtype(operator.source), values.dtype)
        values = _validate_finite_shifts(values.astype(dtype))
        if spectral_interval is not None:
            if not isinstance(spectral_interval, SpectralInterval):
                raise TypeError("spectral_interval must be a SpectralInterval or None.")
            if not spectral_interval.matches(operator):
                raise ValueError(
                    "The spectral interval does not apply to the shifted operator."
                )
        interval_structure = (
            None if spectral_interval is None else spectral_interval.structure_id
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "shifted-linear-system-family",
                    "operator": operator.operator_id,
                    "source": operator.source.space_id,
                    "count": values.size,
                    "dtype": np.dtype(values.dtype).str,
                    "convention": "shift-minus-operator",
                    "spectral_interval": interval_structure,
                }
            )
            if family_id is None
            else str(family_id)
        )
        if not identifier:
            raise ValueError("family_id must be non-empty.")
        self.operator = operator
        self.shifts = values
        self.spectral_interval = spectral_interval
        self.family_id = identifier

    @property
    def num_shifts(self) -> int:
        return self.shifts.size


class ShiftedSolveResourcePolicy(StrictModule):
    """Optional hard budgets for one shared-basis shifted family."""

    max_matvec_count: int | None = eqx.field(static=True)
    max_storage_bytes: int | None = eqx.field(static=True)
    max_workspace_bytes: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_matvec_count: int | None = None,
        max_storage_bytes: int | None = None,
        max_workspace_bytes: int | None = None,
    ):
        self.max_matvec_count = _optional_nonnegative_int(
            max_matvec_count, "max_matvec_count"
        )
        self.max_storage_bytes = _optional_nonnegative_int(
            max_storage_bytes, "max_storage_bytes"
        )
        self.max_workspace_bytes = _optional_nonnegative_int(
            max_workspace_bytes, "max_workspace_bytes"
        )


class ShiftedSolvePolicy(StrictModule):
    """Krylov process, execution, differentiation, and residual policy."""

    method: ShiftedKrylovMethod = eqx.field(static=True)
    execution: ShiftedExecutionMode = eqx.field(static=True)
    differentiation: ShiftedDifferentiationMode = eqx.field(static=True)
    max_dimension: int = eqx.field(static=True)
    orthogonalization: str = eqx.field(static=True)
    breakdown_tolerance: float | None = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    resources: ShiftedSolveResourcePolicy = eqx.field(static=True)

    def __init__(
        self,
        method: ShiftedKrylovMethod = "auto",
        /,
        *,
        execution: ShiftedExecutionMode = "retained",
        differentiation: ShiftedDifferentiationMode = "runtime-shifts",
        max_dimension: int = 32,
        orthogonalization: Literal[
            "modified", "double", "selective", "full", "three-term"
        ] = "selective",
        breakdown_tolerance: float | None = None,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-10,
        resources: ShiftedSolveResourcePolicy | None = None,
    ):
        if method not in ("auto", "arnoldi", "lanczos"):
            raise ValueError("Unknown shifted Krylov method.")
        if execution not in ("retained", "streaming"):
            raise ValueError("Unknown shifted execution mode.")
        if differentiation not in ("runtime-shifts", "none"):
            raise ValueError("Unknown shifted differentiation mode.")
        if orthogonalization not in (
            "modified",
            "double",
            "selective",
            "full",
            "three-term",
        ):
            raise ValueError("Unknown orthogonalization policy.")
        if execution == "streaming":
            if method == "arnoldi":
                raise ValueError("Streaming shifted execution requires Lanczos.")
            if differentiation != "none":
                raise ValueError(
                    "Streaming shifted execution requires differentiation='none'."
                )
            if orthogonalization != "three-term":
                raise ValueError(
                    "Streaming shifted execution requires three-term orthogonalization."
                )
        elif orthogonalization == "three-term":
            raise ValueError(
                "Three-term orthogonalization is only valid for streaming execution."
            )
        dimension = int(max_dimension)
        if dimension < 1:
            raise ValueError("max_dimension must be positive.")
        if breakdown_tolerance is not None:
            breakdown = float(breakdown_tolerance)
            if not math.isfinite(breakdown) or breakdown < 0.0:
                raise ValueError("breakdown_tolerance must be finite and non-negative.")
        else:
            breakdown = None
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        if (
            not math.isfinite(relative)
            or not math.isfinite(absolute)
            or relative < 0.0
            or absolute < 0.0
        ):
            raise ValueError("Shifted solve tolerances must be finite and non-negative.")
        if resources is None:
            resources = ShiftedSolveResourcePolicy()
        if not isinstance(resources, ShiftedSolveResourcePolicy):
            raise TypeError("resources must be a ShiftedSolveResourcePolicy or None.")
        self.method = method
        self.execution = execution
        self.differentiation = differentiation
        self.max_dimension = dimension
        self.orthogonalization = orthogonalization
        self.breakdown_tolerance = breakdown
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.resources = resources


class ShiftedSolveCostEstimate(StrictModule):
    """Static retained/output/workspace and operator-action cost."""

    method: str = eqx.field(static=True)
    execution: ShiftedExecutionMode = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    num_shifts: int = eqx.field(static=True)
    matvec_count: int = eqx.field(static=True)
    preparation_matvec_count: int = eqx.field(static=True)
    execution_matvec_count: int = eqx.field(static=True)
    certification_matvec_count: int = eqx.field(static=True)
    basis_storage_bytes: int = eqx.field(static=True)
    recurrence_storage_bytes: int = eqx.field(static=True)
    solution_storage_bytes: int = eqx.field(static=True)
    total_storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)


class ShiftedSolvePlan(StrictModule):
    """Immutable symbolic plan for one fixed-size shifted family."""

    policy: ShiftedSolvePolicy = eqx.field(static=True)
    projection_plan: KrylovProjectionPlan | None = eqx.field(static=True)
    cost: ShiftedSolveCostEstimate = eqx.field(static=True)
    selected_method: str = eqx.field(static=True)
    selected_execution: ShiftedExecutionMode = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    family_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    spectral_interval_structure_id: str | None = eqx.field(static=True)
    shift_dtype: str = eqx.field(static=True)
    num_shifts: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class _RetainedShiftedState(StrictModule):
    projection: PreparedKrylovProjection


class _StreamingShiftedState(StrictModule):
    rhs: PyTree[Array]


PreparedShiftedState: TypeAlias = _RetainedShiftedState | _StreamingShiftedState


class PreparedShiftedSolve(StrictModule):
    """Execution-specific numerical state bound to one shifted family."""

    family: ShiftedLinearSystemFamily
    state: PreparedShiftedState
    plan: ShiftedSolvePlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array
    refresh_count: Array

    @property
    def projection(self) -> PreparedKrylovProjection | None:
        if isinstance(self.state, _RetainedShiftedState):
            return self.state.projection
        return None

    @property
    def rhs(self) -> PyTree[Array]:
        if isinstance(self.state, _RetainedShiftedState):
            return self.state.projection.initial
        return self.state.rhs


class ShiftedSolveDiagnostics(StrictModule):
    """Per-shift residual, recurrence, stability, and shared-work evidence."""

    residual_norm: Array
    recurrence_residual_norm: Array
    relative_residual: Array
    converged: Array
    finite: Array
    rank: Array
    condition_estimate: Array
    stability_lower_bound: Array
    forward_error_upper_bound: Array
    forward_error_bound_available: Array
    forward_error_bound_certified: Array
    curvature_failure: Array
    iterations: Array
    reference_shift: Array
    krylov_breakdown_status: Array
    orthogonality_error: Array
    setup_matvec_count: Array
    solve_matvec_count: Array
    certification_matvec_count: Array
    basis_storage_bytes: int = eqx.field(static=True)
    recurrence_storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class ShiftedSolveResult(StrictModule):
    """Batched PyTree solutions and evidence for all requested shifts."""

    value: PyTree[Array]
    shifts: Array
    status: Array
    diagnostics: ShiftedSolveDiagnostics
    method: str = eqx.field(static=True)
    execution: ShiftedExecutionMode = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    residual_source: str = eqx.field(static=True)
    spectral_interval_certificate_id: str | None = eqx.field(static=True)
    stability_evidence: str | None = eqx.field(static=True)
    stability_scope: str | None = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(ShiftedSolveStatus.SUCCESS)

    @property
    def all_successful(self) -> Array:
        return jnp.all(self.successful)

    def solution(self, index: int, /) -> PyTree[Array]:
        selected = int(index)
        if selected < 0 or selected >= self.shifts.size:
            raise IndexError("shifted solution index is out of range.")
        return jax.tree.map(lambda leaf: leaf[selected], self.value)


def plan_shifted_solve(
    family: ShiftedLinearSystemFamily,
    policy: ShiftedSolvePolicy | None = None,
    /,
) -> ShiftedSolvePlan:
    """Plan one retained or streaming shifted-system execution."""
    if not isinstance(family, ShiftedLinearSystemFamily):
        raise TypeError("family must be a ShiftedLinearSystemFamily.")
    selected = ShiftedSolvePolicy() if policy is None else policy
    if not isinstance(selected, ShiftedSolvePolicy):
        raise TypeError("policy must be a ShiftedSolvePolicy or None.")
    if selected.execution == "retained":
        projection_policy = KrylovProjectionPolicy(
            selected.method,
            max_dimension=selected.max_dimension,
            orthogonalization=selected.orthogonalization,
            breakdown_tolerance=selected.breakdown_tolerance,
        )
        projection_plan = plan_krylov_projection(family.operator, projection_policy)
        selected_method = projection_plan.selected_method
        dimension = projection_plan.dimension
    else:
        _validate_streaming_structure(family)
        projection_plan = None
        selected_method = "lanczos"
        dimension = min(selected.max_dimension, family.operator.source.size)
    cost = _shifted_cost(
        family,
        selected,
        selected_method,
        dimension,
        projection_plan,
    )
    _validate_resources(cost, selected.resources)
    interval_structure = _spectral_interval_structure_id(family)
    payload = {
        "kind": "shifted-solve-plan",
        "family": family.family_id,
        "operator": family.operator.operator_id,
        "method": selected_method,
        "execution": selected.execution,
        "differentiation": selected.differentiation,
        "orthogonalization": selected.orthogonalization,
        "dimension": dimension,
        "num_shifts": family.num_shifts,
        "shift_dtype": np.dtype(family.shifts.dtype).str,
        "spectral_interval": interval_structure,
        "relative_tolerance": selected.relative_tolerance,
        "absolute_tolerance": selected.absolute_tolerance,
    }
    return ShiftedSolvePlan(
        policy=selected,
        projection_plan=projection_plan,
        cost=cost,
        selected_method=selected_method,
        selected_execution=selected.execution,
        dimension=dimension,
        family_id=family.family_id,
        operator_id=family.operator.operator_id,
        spectral_interval_structure_id=interval_structure,
        shift_dtype=np.dtype(family.shifts.dtype).str,
        num_shifts=family.num_shifts,
        plan_id=canonical_fingerprint(payload),
    )


def prepare_shifted_solve(
    family: ShiftedLinearSystemFamily,
    rhs: PyTree[Any],
    policy: ShiftedSolvePolicy | ShiftedSolvePlan | None = None,
    /,
) -> PreparedShiftedSolve:
    """Bind one retained projection or one streaming right-hand side."""
    plan = (
        policy
        if isinstance(policy, ShiftedSolvePlan)
        else plan_shifted_solve(family, policy)
    )
    _validate_plan(family, plan)
    if plan.selected_execution == "retained":
        if plan.projection_plan is None:
            raise ValueError("Retained shifted plans require a projection plan.")
        state: PreparedShiftedState = _RetainedShiftedState(
            prepare_krylov_projection(
                family.operator,
                rhs,
                plan.projection_plan,
            )
        )
    else:
        validated = family.operator.source.validate(rhs)
        state = _StreamingShiftedState(_stop_arrays(validated))
    return _prepared_shifted(
        family,
        state,
        plan,
        numeric_version=0,
        refresh_count=0,
    )


def refresh_shifted_solve(
    prepared: PreparedShiftedSolve,
    family: ShiftedLinearSystemFamily,
    rhs: PyTree[Any] | None = None,
    /,
) -> PreparedShiftedSolve:
    """Refresh numerical state under one unchanged shifted plan."""
    if not isinstance(prepared, PreparedShiftedSolve):
        raise TypeError("prepared must be a PreparedShiftedSolve.")
    _validate_plan(family, prepared.plan)
    if prepared.plan.selected_execution == "retained":
        projection = prepared.projection
        if projection is None:
            raise ValueError("Retained shifted state is missing its projection.")
        state: PreparedShiftedState = _RetainedShiftedState(
            refresh_krylov_projection(
                projection,
                family.operator,
                rhs,
            )
        )
    else:
        selected_rhs = prepared.rhs if rhs is None else rhs
        validated = family.operator.source.validate(selected_rhs)
        state = _StreamingShiftedState(_stop_arrays(validated))
    return _prepared_shifted(
        family,
        state,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        refresh_count=prepared.refresh_count + jnp.asarray(1, dtype=jnp.int32),
        prepared_id=prepared.prepared_id,
    )


def solve_shifted(
    family_or_prepared: ShiftedLinearSystemFamily | PreparedShiftedSolve,
    rhs: PyTree[Any] | None = None,
    /,
    *,
    policy: ShiftedSolvePolicy | ShiftedSolvePlan | None = None,
    shifts: ArrayLike | None = None,
) -> ShiftedSolveResult:
    """Solve ``(z_j I - A)x_j=b`` by retained or streaming Lanczos work."""
    if isinstance(family_or_prepared, PreparedShiftedSolve):
        if rhs is not None or policy is not None:
            raise ValueError("rhs and policy must be omitted for prepared shifted state.")
        prepared = family_or_prepared
    elif isinstance(family_or_prepared, ShiftedLinearSystemFamily):
        if rhs is None:
            raise ValueError("rhs is required when solving an unprepared shifted family.")
        prepared = prepare_shifted_solve(family_or_prepared, rhs, policy)
    else:
        raise TypeError("Expected a ShiftedLinearSystemFamily or PreparedShiftedSolve.")
    shift_values = (
        prepared.family.shifts
        if shifts is None
        else _coerce_runtime_shifts(shifts, prepared.plan)
    )
    result = _execute_shifted(prepared, shift_values)
    return (
        _stop_arrays(result) if prepared.plan.policy.differentiation == "none" else result
    )


def _execute_shifted(
    prepared: PreparedShiftedSolve,
    shifts: Array,
    /,
) -> ShiftedSolveResult:
    if prepared.plan.selected_execution == "streaming":
        return _execute_streaming_shifted(prepared, shifts)
    return _execute_retained_shifted(prepared, shifts)


def _execute_retained_shifted(
    prepared: PreparedShiftedSolve,
    shifts: Array,
    /,
) -> ShiftedSolveResult:
    projection = prepared.projection
    if projection is None:
        raise ValueError("Retained shifted execution is missing its projection.")
    decomposition = projection.decomposition
    source = prepared.family.operator.source
    rhs_norm = jnp.sqrt(
        jnp.maximum(jnp.real(source.inner(prepared.rhs, prepared.rhs)), 0)
    )
    dtype = jnp.result_type(decomposition.projected.dtype, shifts.dtype)
    coefficients, statuses, residuals, relatives, finite, ranks, conditions = jax.vmap(
        lambda shift: _solve_one_shift(
            decomposition,
            rhs_norm,
            shift,
            prepared.plan.policy,
            dtype,
        )
    )(shifts)
    basis_rows = decomposition.basis[:-1].astype(dtype)
    solution_coordinates = coefficients @ basis_rows
    zero_rhs = rhs_norm == 0
    solution_coordinates = jnp.where(zero_rhs, 0, solution_coordinates)
    values = _unflatten_batched(prepared.rhs, solution_coordinates)
    converged = statuses == int(ShiftedSolveStatus.SUCCESS)
    iterations = jnp.full(
        shifts.shape,
        decomposition.effective_dimension,
        dtype=jnp.int32,
    )
    real_dtype = residuals.dtype
    interval = prepared.family.spectral_interval
    return ShiftedSolveResult(
        value=values,
        shifts=shifts,
        status=statuses,
        diagnostics=ShiftedSolveDiagnostics(
            residual_norm=residuals,
            recurrence_residual_norm=residuals,
            relative_residual=relatives,
            converged=converged,
            finite=finite,
            rank=ranks,
            condition_estimate=conditions,
            stability_lower_bound=jnp.full(shifts.shape, jnp.nan, dtype=real_dtype),
            forward_error_upper_bound=jnp.full(shifts.shape, jnp.inf, dtype=real_dtype),
            forward_error_bound_available=jnp.zeros(shifts.shape, dtype=jnp.bool_),
            forward_error_bound_certified=jnp.zeros(shifts.shape, dtype=jnp.bool_),
            curvature_failure=jnp.zeros(shifts.shape, dtype=jnp.bool_),
            iterations=iterations,
            reference_shift=jnp.asarray(jnp.nan, dtype=real_dtype),
            krylov_breakdown_status=decomposition.breakdown_status,
            orthogonality_error=decomposition.orthogonality_error,
            setup_matvec_count=decomposition.matvec_count,
            solve_matvec_count=jnp.asarray(0, dtype=jnp.int32),
            certification_matvec_count=jnp.asarray(0, dtype=jnp.int32),
            basis_storage_bytes=prepared.plan.cost.basis_storage_bytes,
            recurrence_storage_bytes=prepared.plan.cost.recurrence_storage_bytes,
            workspace_bytes=prepared.plan.cost.workspace_bytes,
        ),
        method=prepared.plan.selected_method,
        execution=prepared.plan.selected_execution,
        convention="shift-minus-operator",
        residual_source="projected-relation",
        spectral_interval_certificate_id=(
            None if interval is None else interval.certificate_id
        ),
        stability_evidence=None if interval is None else interval.evidence,
        stability_scope=None if interval is None else interval.scope,
        provenance=(
            "shared bound Krylov projection with per-shift projected-relation residuals"
        ),
    )


def _execute_streaming_shifted(
    prepared: PreparedShiftedSolve,
    shifts: Array,
    /,
) -> ShiftedSolveResult:
    family = prepared.family
    operator = _stop_arrays(family.operator)
    rhs = operator.source.flatten(_stop_arrays(prepared.rhs))
    stopped_shifts = jax.lax.stop_gradient(shifts)
    (
        lower,
        evidence_available,
        evidence_certified,
        certificate_id,
        evidence,
        scope,
    ) = _streaming_lower_evidence(family)
    lower = jax.lax.stop_gradient(lower)
    real_shifts = jnp.real(stopped_shifts).astype(rhs.real.dtype)
    real_valued = jnp.imag(stopped_shifts) == 0.0
    admissible = (
        jnp.asarray(evidence_available)
        & jnp.isfinite(real_shifts)
        & real_valued
        & (real_shifts < lower)
    )
    raw = streaming_shifted_lanczos(
        operator,
        rhs,
        real_shifts,
        admissible,
        max_steps=prepared.plan.dimension,
        relative_tolerance=prepared.plan.policy.relative_tolerance,
        absolute_tolerance=prepared.plan.policy.absolute_tolerance,
        breakdown_tolerance=prepared.plan.policy.breakdown_tolerance,
    )
    solution_coordinates = -raw.positive_solutions
    rhs_norm = _coordinate_norm(operator, rhs)
    zero_rhs = rhs_norm == 0.0

    def verify(_):
        images = jax.vmap(lambda value: _action_coordinates(operator, value))(
            solution_coordinates
        )
        return rhs[None, :] - (
            real_shifts[:, None].astype(rhs.dtype) * solution_coordinates - images
        )

    residual_coordinates = jax.lax.cond(
        zero_rhs,
        lambda _: jnp.zeros_like(solution_coordinates),
        verify,
        operand=None,
    )
    residuals = jax.vmap(lambda value: _coordinate_norm(operator, value))(
        residual_coordinates
    )
    tiny = jnp.asarray(jnp.finfo(rhs.real.dtype).tiny)
    relatives = jnp.where(
        zero_rhs,
        jnp.zeros_like(residuals),
        residuals / jnp.maximum(rhs_norm, tiny),
    )
    threshold = (
        jnp.asarray(prepared.plan.policy.absolute_tolerance, dtype=rhs.real.dtype)
        + jnp.asarray(prepared.plan.policy.relative_tolerance, dtype=rhs.real.dtype)
        * rhs_norm
    )
    converged = residuals <= threshold
    finite = (
        raw.finite
        & jnp.all(jnp.isfinite(solution_coordinates), axis=1)
        & jnp.all(jnp.isfinite(residual_coordinates), axis=1)
        & jnp.isfinite(residuals)
    )
    statuses = jnp.where(
        ~admissible,
        int(ShiftedSolveStatus.INADMISSIBLE_SHIFT),
        jnp.where(
            ~finite,
            int(ShiftedSolveStatus.NONFINITE),
            jnp.where(
                converged,
                int(ShiftedSolveStatus.SUCCESS),
                jnp.where(
                    raw.unfinished,
                    int(ShiftedSolveStatus.MAX_DIMENSION_REACHED),
                    int(ShiftedSolveStatus.KRYLOV_FAILURE),
                ),
            ),
        ),
    ).astype(jnp.int32)
    stability = lower - real_shifts
    bound_available = (
        admissible
        & jnp.asarray(evidence_available)
        & jnp.isfinite(stability)
        & (stability > 0.0)
        & jnp.isfinite(residuals)
    )
    forward_bound = jnp.where(
        bound_available,
        residuals / jnp.where(stability > 0.0, stability, 1.0),
        jnp.asarray(jnp.inf, dtype=residuals.dtype),
    )
    certification_count = jnp.where(
        zero_rhs,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(family.num_shifts, dtype=jnp.int32),
    )
    values = _unflatten_batched(prepared.rhs, solution_coordinates)
    return ShiftedSolveResult(
        value=values,
        shifts=stopped_shifts,
        status=statuses,
        diagnostics=ShiftedSolveDiagnostics(
            residual_norm=residuals,
            recurrence_residual_norm=raw.recurrence_residual_norm,
            relative_residual=relatives,
            converged=statuses == int(ShiftedSolveStatus.SUCCESS),
            finite=finite,
            rank=jnp.full(shifts.shape, -1, dtype=jnp.int32),
            condition_estimate=jnp.full(shifts.shape, jnp.nan, dtype=residuals.dtype),
            stability_lower_bound=stability,
            forward_error_upper_bound=forward_bound,
            forward_error_bound_available=bound_available,
            forward_error_bound_certified=bound_available
            & jnp.asarray(evidence_certified),
            curvature_failure=raw.curvature_failure,
            iterations=raw.iterations,
            reference_shift=raw.reference_shift,
            krylov_breakdown_status=raw.breakdown_status,
            orthogonality_error=jnp.asarray(jnp.nan, dtype=residuals.dtype),
            setup_matvec_count=jnp.asarray(0, dtype=jnp.int32),
            solve_matvec_count=raw.matvec_count + certification_count,
            certification_matvec_count=certification_count,
            basis_storage_bytes=prepared.plan.cost.basis_storage_bytes,
            recurrence_storage_bytes=prepared.plan.cost.recurrence_storage_bytes,
            workspace_bytes=prepared.plan.cost.workspace_bytes,
        ),
        method=prepared.plan.selected_method,
        execution=prepared.plan.selected_execution,
        convention="shift-minus-operator",
        residual_source="direct-operator",
        spectral_interval_certificate_id=certificate_id,
        stability_evidence=evidence,
        stability_scope=scope,
        provenance=(
            "streaming three-term multi-shift Lanczos with direct original-system residual verification"
        ),
    )


def _solve_one_shift(
    decomposition,
    rhs_norm: Array,
    shift: Array,
    policy: ShiftedSolvePolicy,
    dtype: Any,
    /,
):
    capacity = decomposition.projected.shape[1]
    projected = decomposition.projected.astype(dtype)
    real_dtype = projected.real.dtype
    tiny = jnp.asarray(jnp.finfo(real_dtype).tiny)
    krylov_ok = (decomposition.breakdown_status == int(KrylovBreakdownStatus.NONE)) | (
        decomposition.breakdown_status == int(KrylovBreakdownStatus.HAPPY)
    )

    def empty(_):
        zero = jnp.zeros((capacity,), dtype=dtype)
        residual = rhs_norm.astype(real_dtype)
        zero_rhs = residual == 0
        status = jnp.where(
            zero_rhs,
            int(ShiftedSolveStatus.SUCCESS),
            int(ShiftedSolveStatus.KRYLOV_FAILURE),
        ).astype(jnp.int32)
        return (
            zero,
            status,
            residual,
            jnp.where(zero_rhs, 0.0, 1.0).astype(real_dtype),
            jnp.isfinite(residual),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.nan, dtype=real_dtype),
        )

    def branch(size: int):
        def solve(_):
            hessenberg = projected[: size + 1, :size]
            embedded_identity = jnp.zeros((size + 1, size), dtype=dtype)
            embedded_identity = embedded_identity.at[:size, :].set(
                jnp.eye(size, dtype=dtype)
            )
            matrix = shift.astype(dtype) * embedded_identity - hessenberg
            target = jnp.zeros((size + 1,), dtype=dtype).at[0].set(rhs_norm.astype(dtype))
            value, _, rank, singular_values = jnp.linalg.lstsq(
                matrix,
                target,
                rcond=None,
            )
            residual = jnp.linalg.norm(target - matrix @ value)
            relative = residual / jnp.maximum(rhs_norm, tiny)
            finite = (
                jnp.all(jnp.isfinite(value))
                & jnp.isfinite(residual)
                & jnp.all(jnp.isfinite(singular_values))
            )
            full_rank = rank == size
            condition = jnp.where(
                full_rank & (singular_values[-1] > 0),
                singular_values[0] / singular_values[-1],
                jnp.asarray(jnp.inf, dtype=real_dtype),
            )
            converged = residual <= (
                policy.absolute_tolerance + policy.relative_tolerance * rhs_norm
            )
            status = jnp.where(
                ~finite,
                int(ShiftedSolveStatus.NONFINITE),
                jnp.where(
                    ~krylov_ok,
                    int(ShiftedSolveStatus.KRYLOV_FAILURE),
                    jnp.where(
                        ~full_rank,
                        int(ShiftedSolveStatus.SINGULAR),
                        jnp.where(
                            converged,
                            int(ShiftedSolveStatus.SUCCESS),
                            int(ShiftedSolveStatus.MAX_DIMENSION_REACHED),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            padded = jnp.zeros((capacity,), dtype=dtype).at[:size].set(value)
            return (
                padded,
                status,
                residual,
                relative,
                finite,
                rank.astype(jnp.int32),
                condition,
            )

        return solve

    branches = (empty,) + tuple(branch(size) for size in range(1, capacity + 1))
    return jax.lax.switch(decomposition.effective_dimension, branches, operand=None)


def _prepared_shifted(
    family: ShiftedLinearSystemFamily,
    state: PreparedShiftedState,
    plan: ShiftedSolvePlan,
    *,
    numeric_version: Any,
    refresh_count: Any,
    prepared_id: str | None = None,
) -> PreparedShiftedSolve:
    if isinstance(state, _RetainedShiftedState):
        state_identity = state.projection.projection_id
    else:
        state_identity = array_tree_fingerprint(state.rhs)
    identifier = (
        canonical_fingerprint(
            {
                "kind": "prepared-shifted-solve",
                "plan": plan.plan_id,
                "execution": plan.selected_execution,
                "state": state_identity,
            }
        )
        if prepared_id is None
        else prepared_id
    )
    return PreparedShiftedSolve(
        family=family,
        state=state,
        plan=plan,
        prepared_id=identifier,
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
    )


def _shifted_cost(
    family: ShiftedLinearSystemFamily,
    policy: ShiftedSolvePolicy,
    method: str,
    dimension: int,
    projection_plan: KrylovProjectionPlan | None,
    /,
) -> ShiftedSolveCostEstimate:
    count = family.num_shifts
    source_size = family.operator.source.size
    output_dtype = jnp.result_type(
        _coordinate_dtype(family.operator.source),
        family.shifts.dtype,
    )
    coordinate_itemsize = jnp.dtype(output_dtype).itemsize
    real_itemsize = jnp.empty((), dtype=output_dtype).real.dtype.itemsize
    solution_storage = count * source_size * coordinate_itemsize
    if policy.execution == "retained":
        if projection_plan is None:
            raise ValueError("Retained shifted cost requires a projection plan.")
        basis_storage = projection_plan.cost.storage_bytes
        recurrence_storage = 0
        total_storage = basis_storage + solution_storage + count * coordinate_itemsize
        small_entries = count * ((dimension + 1) * dimension + 3 * dimension + 1)
        workspace = (
            projection_plan.cost.workspace_bytes + small_entries * coordinate_itemsize
        )
        preparation_matvecs = projection_plan.cost.matvec_count
        execution_matvecs = 0
        certification_matvecs = 0
        exact = projection_plan.cost.exact
    else:
        basis_storage = 0
        scalar_storage = 8 * count * real_itemsize
        integer_storage = 3 * count * jnp.dtype(jnp.int32).itemsize
        mask_storage = 4 * count * jnp.dtype(jnp.bool_).itemsize
        recurrence_storage = (
            (count + 3) * source_size * coordinate_itemsize
            + scalar_storage
            + integer_storage
            + mask_storage
        )
        total_storage = solution_storage
        workspace = recurrence_storage
        preparation_matvecs = 0
        execution_matvecs = dimension
        certification_matvecs = count
        exact = True
    matvecs = preparation_matvecs + execution_matvecs + certification_matvecs
    return ShiftedSolveCostEstimate(
        method=method,
        execution=policy.execution,
        dimension=dimension,
        num_shifts=count,
        matvec_count=matvecs,
        preparation_matvec_count=preparation_matvecs,
        execution_matvec_count=execution_matvecs,
        certification_matvec_count=certification_matvecs,
        basis_storage_bytes=basis_storage,
        recurrence_storage_bytes=recurrence_storage,
        solution_storage_bytes=solution_storage,
        total_storage_bytes=total_storage,
        workspace_bytes=workspace,
        exact=exact,
    )


def _validate_resources(
    cost: ShiftedSolveCostEstimate,
    resources: ShiftedSolveResourcePolicy,
    /,
) -> None:
    checks = (
        ("matvec count", cost.matvec_count, resources.max_matvec_count),
        ("storage", cost.total_storage_bytes, resources.max_storage_bytes),
        ("workspace", cost.workspace_bytes, resources.max_workspace_bytes),
    )
    violations = [
        f"{name} estimate {value} exceeds budget {limit}"
        for name, value, limit in checks
        if limit is not None and value > limit
    ]
    if violations:
        raise ValueError("Shifted solve resource rejection: " + "; ".join(violations))


def _validate_operator(operator: AbstractLinearOperator, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Shifted families require an unbatched endomorphism.")
    if not jnp.issubdtype(_coordinate_dtype(operator.source), jnp.inexact):
        raise TypeError("Shifted families require real or complex coordinates.")


def _spectral_interval_structure_id(
    family: ShiftedLinearSystemFamily,
    /,
) -> str | None:
    interval = family.spectral_interval
    return None if interval is None else interval.structure_id


def _validate_streaming_structure(family: ShiftedLinearSystemFamily, /) -> None:
    operator = family.operator
    if not operator.properties.certifies("self_adjoint"):
        raise ValueError(
            "Streaming shifted execution requires certified self-adjoint structure."
        )
    interval = family.spectral_interval
    if interval is None:
        if not operator.properties.certifies("positive_semidefinite"):
            raise ValueError(
                "Streaming shifted execution requires a matching spectral interval "
                "or certified positive-semidefinite structure."
            )
        lower = 0.0
    else:
        if not interval.matches(operator):
            raise ValueError(
                "The streaming spectral interval does not match the operator."
            )
        lower = float(np.asarray(interval.lower))
    shifts = np.asarray(family.shifts)
    if np.any(np.imag(shifts) != 0.0):
        raise ValueError("Streaming shifted execution requires real-valued shifts.")
    if np.any(np.real(shifts) >= lower):
        raise ValueError(
            "Streaming shifts must lie strictly below the spectral lower bound."
        )


def _streaming_lower_evidence(
    family: ShiftedLinearSystemFamily,
    /,
) -> tuple[Array, bool, bool, str | None, str, str]:
    operator = family.operator
    self_adjoint_evidence = operator.properties.evidence_for("self_adjoint")
    interval = family.spectral_interval
    if interval is not None:
        available = interval.matches(operator)
        certified = (
            available
            and interval.evidence in ("construction", "verified")
            and self_adjoint_evidence in ("construction", "transformed", "verified")
        )
        return (
            interval.lower,
            available,
            certified,
            interval.certificate_id,
            interval.evidence,
            interval.scope,
        )
    semidefinite_evidence = operator.properties.evidence_for("positive_semidefinite")
    available = operator.properties.certifies(
        "self_adjoint"
    ) and operator.properties.certifies("positive_semidefinite")
    certified = (
        available
        and self_adjoint_evidence in ("construction", "transformed", "verified")
        and semidefinite_evidence in ("construction", "transformed", "verified")
    )
    real_dtype = jnp.empty((), dtype=_coordinate_dtype(operator.source)).real.dtype
    return (
        jnp.asarray(0.0, dtype=real_dtype),
        available,
        certified,
        None,
        semidefinite_evidence,
        "structural",
    )


def _action_coordinates(operator: AbstractLinearOperator, value: Array, /) -> Array:
    return operator.target.flatten(operator.mv(operator.source.unflatten(value)))


def _coordinate_norm(operator: AbstractLinearOperator, value: Array, /) -> Array:
    vector = operator.source.unflatten(value)
    squared = jnp.real(operator.source.inner(vector, vector))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def _stop_arrays(value: Any, /) -> Any:
    return jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
        value,
    )


def _validate_plan(
    family: ShiftedLinearSystemFamily,
    plan: ShiftedSolvePlan,
    /,
) -> None:
    if not isinstance(plan, ShiftedSolvePlan):
        raise TypeError("plan must be a ShiftedSolvePlan.")
    if (
        family.family_id != plan.family_id
        or family.operator.operator_id != plan.operator_id
    ):
        raise ValueError("Shifted solve plan belongs to a different symbolic family.")
    if family.num_shifts != plan.num_shifts:
        raise ValueError("Shift count changed under a fixed shifted solve plan.")
    if np.dtype(family.shifts.dtype).str != plan.shift_dtype:
        raise TypeError("Shift dtype changed under a fixed shifted solve plan.")
    if _spectral_interval_structure_id(family) != plan.spectral_interval_structure_id:
        raise ValueError("Spectral interval structure changed under a shifted plan.")
    if plan.selected_execution == "streaming":
        _validate_streaming_structure(family)


def _coerce_runtime_shifts(
    shifts: ArrayLike,
    plan: ShiftedSolvePlan,
    /,
) -> Array:
    values = jnp.asarray(shifts)
    if values.shape != (plan.num_shifts,):
        raise ValueError("Runtime shifts must preserve the planned shift count.")
    if not jnp.issubdtype(values.dtype, jnp.number) or jnp.issubdtype(
        values.dtype, jnp.bool_
    ):
        raise TypeError("Runtime shifts must contain real or complex numbers.")
    planned_dtype = np.dtype(plan.shift_dtype)
    if np.issubdtype(planned_dtype, np.floating) and jnp.issubdtype(
        values.dtype, jnp.complexfloating
    ):
        raise TypeError("Complex runtime shifts require a complex shifted plan.")
    return _validate_finite_shifts(values.astype(planned_dtype))


def _validate_finite_shifts(values: Array, /) -> Array:
    finite = jnp.all(jnp.isfinite(values))
    if isinstance(finite, jax_core.Tracer):
        return eqx.error_if(values, ~finite, "shifts must be finite.")
    if not bool(finite):
        raise ValueError("shifts must be finite.")
    return values


def _unflatten_batched(template: PyTree[Any], coordinates: Array, /) -> PyTree[Array]:
    leaves, treedef = jax.tree.flatten(template)
    rebuilt = []
    offset = 0
    for leaf in leaves:
        array = jnp.asarray(leaf)
        size = array.size
        rebuilt.append(
            coordinates[:, offset : offset + size].reshape(
                (coordinates.shape[0],) + array.shape
            )
        )
        offset += size
    if coordinates.shape[1] != offset:
        raise ValueError("Shifted solution coordinate width does not match the source.")
    return jax.tree.unflatten(treedef, rebuilt)


def _optional_nonnegative_int(value: int | None, name: str, /) -> int | None:
    if value is None:
        return None
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative or None.")
    return integer


__all__ = [
    "PreparedShiftedSolve",
    "ShiftedKrylovMethod",
    "ShiftedDifferentiationMode",
    "ShiftedExecutionMode",
    "ShiftedLinearSystemFamily",
    "ShiftedSolveCostEstimate",
    "ShiftedSolveDiagnostics",
    "ShiftedSolvePlan",
    "ShiftedSolvePolicy",
    "ShiftedSolveResourcePolicy",
    "ShiftedSolveResult",
    "ShiftedSolveStatus",
    "plan_shifted_solve",
    "prepare_shifted_solve",
    "refresh_shifted_solve",
    "solve_shifted",
]
