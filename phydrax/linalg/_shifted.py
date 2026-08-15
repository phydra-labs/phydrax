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

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._operators import AbstractLinearOperator
from ._spaces import _coordinate_dtype
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


class ShiftedSolveStatus(IntEnum):
    """Portable per-shift status for a shared-basis solve family."""

    SUCCESS = 0
    MAX_DIMENSION_REACHED = 1
    SINGULAR = 2
    NONFINITE = 3
    KRYLOV_FAILURE = 4


class ShiftedLinearSystemFamily(StrictModule):
    """Systems ``(z_j I - A) x_j = b`` sharing one operator and right-hand side."""

    operator: AbstractLinearOperator
    shifts: Array
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        shifts: ArrayLike,
        /,
        *,
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
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "shifted-linear-system-family",
                    "operator": operator.operator_id,
                    "source": operator.source.space_id,
                    "count": int(values.size),
                    "dtype": np.dtype(values.dtype).str,
                    "convention": "shift-minus-operator",
                }
            )
            if family_id is None
            else str(family_id)
        )
        if not identifier:
            raise ValueError("family_id must be non-empty.")
        self.operator = operator
        self.shifts = values
        self.family_id = identifier

    @property
    def num_shifts(self) -> int:
        return int(self.shifts.size)


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
    """Shared Krylov and residual policy for shifted systems."""

    method: ShiftedKrylovMethod = eqx.field(static=True)
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
        max_dimension: int = 32,
        orthogonalization: Literal[
            "modified", "double", "selective", "full"
        ] = "selective",
        breakdown_tolerance: float | None = None,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-10,
        resources: ShiftedSolveResourcePolicy | None = None,
    ):
        if method not in ("auto", "arnoldi", "lanczos"):
            raise ValueError("Unknown shifted Krylov method.")
        dimension = int(max_dimension)
        if dimension < 1:
            raise ValueError("max_dimension must be positive.")
        if orthogonalization not in ("modified", "double", "selective", "full"):
            raise ValueError("Unknown orthogonalization policy.")
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
        self.max_dimension = dimension
        self.orthogonalization = orthogonalization
        self.breakdown_tolerance = breakdown
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.resources = resources


class ShiftedSolveCostEstimate(StrictModule):
    """Static shared-basis, output, and projected-solve cost."""

    method: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    num_shifts: int = eqx.field(static=True)
    matvec_count: int = eqx.field(static=True)
    basis_storage_bytes: int = eqx.field(static=True)
    solution_storage_bytes: int = eqx.field(static=True)
    total_storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)


class ShiftedSolvePlan(StrictModule):
    """Immutable symbolic plan for one fixed-size shifted family."""

    policy: ShiftedSolvePolicy = eqx.field(static=True)
    projection_plan: KrylovProjectionPlan = eqx.field(static=True)
    cost: ShiftedSolveCostEstimate = eqx.field(static=True)
    family_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    shift_dtype: str = eqx.field(static=True)
    num_shifts: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def selected_method(self) -> str:
        return self.projection_plan.selected_method

    @property
    def dimension(self) -> int:
        return self.projection_plan.dimension


class PreparedShiftedSolve(StrictModule):
    """Shared Krylov projection bound to one operator and right-hand side."""

    family: ShiftedLinearSystemFamily
    projection: PreparedKrylovProjection
    plan: ShiftedSolvePlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array
    refresh_count: Array

    @property
    def rhs(self) -> PyTree[Array]:
        return self.projection.initial


class ShiftedSolveDiagnostics(StrictModule):
    """Per-shift residual and small-system evidence plus shared Krylov evidence."""

    residual_norm: Array
    relative_residual: Array
    converged: Array
    finite: Array
    rank: Array
    condition_estimate: Array
    iterations: Array
    krylov_breakdown_status: Array
    orthogonality_error: Array
    setup_matvec_count: Array
    solve_matvec_count: Array
    basis_storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class ShiftedSolveResult(StrictModule):
    """Batched PyTree solutions and evidence for all requested shifts."""

    value: PyTree[Array]
    shifts: Array
    status: Array
    diagnostics: ShiftedSolveDiagnostics
    method: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
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
    """Plan a shared Krylov basis and fixed family of projected solves."""
    if not isinstance(family, ShiftedLinearSystemFamily):
        raise TypeError("family must be a ShiftedLinearSystemFamily.")
    selected = ShiftedSolvePolicy() if policy is None else policy
    if not isinstance(selected, ShiftedSolvePolicy):
        raise TypeError("policy must be a ShiftedSolvePolicy or None.")
    projection_policy = KrylovProjectionPolicy(
        selected.method,
        max_dimension=selected.max_dimension,
        orthogonalization=selected.orthogonalization,
        breakdown_tolerance=selected.breakdown_tolerance,
    )
    projection_plan = plan_krylov_projection(family.operator, projection_policy)
    cost = _shifted_cost(family, projection_plan)
    _validate_resources(cost, selected.resources)
    payload = {
        "kind": "shifted-solve-plan",
        "family": family.family_id,
        "operator": family.operator.operator_id,
        "method": projection_plan.selected_method,
        "dimension": projection_plan.dimension,
        "num_shifts": family.num_shifts,
        "shift_dtype": np.dtype(family.shifts.dtype).str,
        "relative_tolerance": selected.relative_tolerance,
        "absolute_tolerance": selected.absolute_tolerance,
    }
    return ShiftedSolvePlan(
        policy=selected,
        projection_plan=projection_plan,
        cost=cost,
        family_id=family.family_id,
        operator_id=family.operator.operator_id,
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
    """Build the shared Krylov basis for one shifted family."""
    plan = (
        policy
        if isinstance(policy, ShiftedSolvePlan)
        else plan_shifted_solve(family, policy)
    )
    _validate_plan(family, plan)
    projection = prepare_krylov_projection(
        family.operator,
        rhs,
        plan.projection_plan,
    )
    return _prepared_shifted(
        family,
        projection,
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
    """Rebuild numerical basis state under one unchanged shifted plan."""
    if not isinstance(prepared, PreparedShiftedSolve):
        raise TypeError("prepared must be a PreparedShiftedSolve.")
    _validate_plan(family, prepared.plan)
    projection = refresh_krylov_projection(
        prepared.projection,
        family.operator,
        rhs,
    )
    return _prepared_shifted(
        family,
        projection,
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
    """Solve ``(z_j I - A)x_j=b`` from a shared Arnoldi or Lanczos basis."""
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
    return _execute_shifted(prepared, shift_values)


def _execute_shifted(
    prepared: PreparedShiftedSolve,
    shifts: Array,
    /,
) -> ShiftedSolveResult:
    decomposition = prepared.projection.decomposition
    capacity = prepared.plan.dimension
    rhs_coordinates = prepared.projection.initial_coordinates
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
    return ShiftedSolveResult(
        value=values,
        shifts=shifts,
        status=statuses,
        diagnostics=ShiftedSolveDiagnostics(
            residual_norm=residuals,
            relative_residual=relatives,
            converged=converged,
            finite=finite,
            rank=ranks,
            condition_estimate=conditions,
            iterations=iterations,
            krylov_breakdown_status=decomposition.breakdown_status,
            orthogonality_error=decomposition.orthogonality_error,
            setup_matvec_count=decomposition.matvec_count,
            solve_matvec_count=jnp.asarray(0, dtype=jnp.int32),
            basis_storage_bytes=prepared.plan.cost.basis_storage_bytes,
            workspace_bytes=prepared.plan.cost.workspace_bytes,
        ),
        method=prepared.plan.selected_method,
        convention="shift-minus-operator",
        provenance="shared bound Krylov projection with per-shift least-squares residuals",
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
    projection: PreparedKrylovProjection,
    plan: ShiftedSolvePlan,
    *,
    numeric_version: Any,
    refresh_count: Any,
    prepared_id: str | None = None,
) -> PreparedShiftedSolve:
    identifier = (
        canonical_fingerprint(
            {
                "kind": "prepared-shifted-solve",
                "plan": plan.plan_id,
                "projection": projection.projection_id,
            }
        )
        if prepared_id is None
        else prepared_id
    )
    return PreparedShiftedSolve(
        family=family,
        projection=projection,
        plan=plan,
        prepared_id=identifier,
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
    )


def _shifted_cost(
    family: ShiftedLinearSystemFamily,
    projection_plan: KrylovProjectionPlan,
    /,
) -> ShiftedSolveCostEstimate:
    dimension = projection_plan.dimension
    count = family.num_shifts
    itemsize = family.shifts.dtype.itemsize
    basis_storage = projection_plan.cost.storage_bytes
    solution_storage = count * family.operator.source.size * itemsize
    total_storage = basis_storage + solution_storage + count * itemsize
    small_entries = count * ((dimension + 1) * dimension + 3 * dimension + 1)
    workspace = projection_plan.cost.workspace_bytes + small_entries * itemsize
    return ShiftedSolveCostEstimate(
        method=projection_plan.selected_method,
        dimension=dimension,
        num_shifts=count,
        matvec_count=projection_plan.cost.matvec_count,
        basis_storage_bytes=basis_storage,
        solution_storage_bytes=solution_storage,
        total_storage_bytes=total_storage,
        workspace_bytes=workspace,
        exact=projection_plan.cost.exact,
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
        size = int(array.size)
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
