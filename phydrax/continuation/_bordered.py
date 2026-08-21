#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import tree_add_scaled, tree_allfinite
from ..linalg import (
    AbstractLinearOperator,
    LinearSolvePlan,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    plan as plan_linear,
    prepare as prepare_linear,
    PreparedLinearSolve,
    refresh as refresh_linear,
    solve as solve_linear,
)


class BorderedSolveStatus(IntEnum):
    """Portable terminal status for a bordered linear solve."""

    SUCCESS = 0
    PRINCIPAL_COLUMN_SOLVE_FAILED = 1
    CACHED_COLUMN_NONFINITE = 2
    SCHUR_NONFINITE = 3
    SCHUR_SINGULAR = 4
    PRINCIPAL_RIGHT_HAND_SIDE_SOLVE_FAILED = 5
    SOLUTION_NONFINITE = 6
    RESIDUAL_NONFINITE = 7


def _scalar(value: Any, /, *, name: str, dtype: Any | None = None) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if array.shape != () or not jnp.issubdtype(array.dtype, jnp.inexact):
        raise TypeError(f"{name} must be one real or complex inexact scalar array.")
    return array


def _tree_subtract(left: PyTree[Any], right: PyTree[Any], /) -> PyTree[Array]:
    return tree_add_scaled(left, right, -1.0)


def _finite_tree(tree: PyTree[Any], /) -> Array:
    return tree_allfinite(tree)


def _tree_where(
    condition: Any,
    proposed: PyTree[Any],
    current: PyTree[Any],
    /,
) -> PyTree[Array]:
    return jax.tree.map(
        lambda proposed_value, current_value: jnp.where(
            condition,
            proposed_value,
            current_value,
        ),
        proposed,
        current,
    )


class BorderedLinearSystem(StrictModule):
    """Immutable block system ``[A b; c* d]`` over explicit vector spaces.

    ``column`` belongs to the target of ``A`` and ``row`` belongs to its source;
    the source-space pairing evaluates ``c* x``. The principal operator is never
    materialized by this layer.
    """

    operator: AbstractLinearOperator
    column: PyTree[Array]
    row: PyTree[Array]
    corner: Array
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        column: PyTree[Any],
        row: PyTree[Any],
        corner: Any,
        /,
        *,
        system_id: str | None = None,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape:
            raise ValueError("Bordered systems do not accept batched operators.")
        if operator.source.size != operator.target.size:
            raise ValueError("The principal bordered operator must be square.")
        column_ = operator.target.validate(column)
        row_ = operator.source.validate(row)
        scalar_dtype = operator.source.inner(row_, row_).dtype
        corner_ = _scalar(corner, name="border corner", dtype=scalar_dtype)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "bordered-linear-system",
                    "operator_id": operator.operator_id,
                    "source_space_id": operator.source.space_id,
                    "target_space_id": operator.target.space_id,
                }
            )
            if system_id is None
            else str(system_id)
        )
        if not identifier:
            raise ValueError("system_id must be non-empty.")
        self.operator = operator
        self.column = column_
        self.row = row_
        self.corner = corner_
        self.system_id = identifier

    def row_action(self, vector: PyTree[Any], /) -> Array:
        return self.operator.source.inner(
            self.row,
            self.operator.source.validate(vector),
        )

    def apply(
        self,
        primal: PyTree[Any],
        scalar: Any,
        /,
    ) -> tuple[PyTree[Array], Array]:
        primal_ = self.operator.source.validate(primal)
        scalar_ = _scalar(
            scalar, name="bordered solution scalar", dtype=self.corner.dtype
        )
        target = tree_add_scaled(
            self.operator.mv(primal_),
            self.column,
            scalar_,
        )
        border = self.row_action(primal_) + self.corner * scalar_
        return target, border


class BorderedSolvePlan(StrictModule):
    """Reusable symbolic policy for one bordered-system structure."""

    principal_plan: LinearSolvePlan
    schur_tolerance: float = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        principal_plan: LinearSolvePlan,
        /,
        *,
        schur_tolerance: float,
        system_id: str,
        source_space_id: str,
        target_space_id: str,
        plan_id: str,
    ):
        if not isinstance(principal_plan, LinearSolvePlan):
            raise TypeError("principal_plan must be a LinearSolvePlan.")
        tolerance = float(schur_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("schur_tolerance must be finite and non-negative.")
        identifiers = tuple(
            str(value)
            for value in (
                system_id,
                source_space_id,
                target_space_id,
                plan_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Bordered solve plan identities must be non-empty.")
        self.principal_plan = principal_plan
        self.schur_tolerance = tolerance
        (
            self.system_id,
            self.source_space_id,
            self.target_space_id,
            self.plan_id,
        ) = identifiers


def _principal_problem(system: BorderedLinearSystem, /) -> LinearSystem:
    return LinearSystem(
        system.operator,
        problem_id=f"{system.system_id}/principal",
    )


def plan_bordered_solve(
    system: BorderedLinearSystem,
    policy: LinearSolvePolicy,
    /,
    *,
    schur_tolerance: float = 1e-12,
    plan_id: str | None = None,
) -> BorderedSolvePlan:
    """Plan the principal solves without materializing outside ``policy``."""
    if not isinstance(system, BorderedLinearSystem):
        raise TypeError("system must be a BorderedLinearSystem.")
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("policy must be a LinearSolvePolicy.")
    if policy.failure.mode != "status":
        raise ValueError(
            "Bordered solves require linear failure mode 'status' so inner failures "
            "remain explicit."
        )
    principal_plan = plan_linear(_principal_problem(system), policy)
    tolerance = float(schur_tolerance)
    identifier = (
        canonical_fingerprint(
            {
                "kind": "bordered-solve-plan",
                "system_id": system.system_id,
                "principal_plan_id": principal_plan.plan_id,
                "schur_tolerance": tolerance,
            }
        )
        if plan_id is None
        else str(plan_id)
    )
    return BorderedSolvePlan(
        principal_plan,
        schur_tolerance=tolerance,
        system_id=system.system_id,
        source_space_id=system.operator.source.space_id,
        target_space_id=system.operator.target.space_id,
        plan_id=identifier,
    )


class PreparedBorderedSolve(StrictModule):
    """Numerical principal state plus cached ``A^-1 b`` and Schur complement."""

    system: BorderedLinearSystem
    plan: BorderedSolvePlan
    principal: PreparedLinearSolve
    inverse_column: PyTree[Array]
    schur_complement: Array
    schur_scale: Array
    status: Array
    column_solve_status: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: BorderedLinearSystem,
        plan: BorderedSolvePlan,
        principal: PreparedLinearSolve,
        inverse_column: PyTree[Any],
        schur_complement: Any,
        schur_scale: Any,
        status: Any,
        column_solve_status: Any,
        numeric_version: Any,
        /,
        *,
        prepared_id: str,
    ):
        if not isinstance(system, BorderedLinearSystem):
            raise TypeError("system must be a BorderedLinearSystem.")
        if not isinstance(plan, BorderedSolvePlan):
            raise TypeError("plan must be a BorderedSolvePlan.")
        if not isinstance(principal, PreparedLinearSolve):
            raise TypeError("principal must be a PreparedLinearSolve.")
        if system.system_id != plan.system_id:
            raise ValueError("Bordered system and plan IDs must match.")
        inverse_column_ = system.operator.source.validate(inverse_column)
        schur = _scalar(
            schur_complement,
            name="Schur complement",
            dtype=system.corner.dtype,
        )
        scale = _scalar(schur_scale, name="Schur scale")
        status_ = jnp.asarray(status, dtype=jnp.int32)
        column_status = jnp.asarray(column_solve_status, dtype=jnp.int32)
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if any(value.shape != () for value in (status_, column_status, version)):
            raise ValueError("Prepared bordered statuses and version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        identifier = str(prepared_id)
        if not identifier:
            raise ValueError("prepared_id must be non-empty.")
        self.system = system
        self.plan = plan
        self.principal = principal
        self.inverse_column = inverse_column_
        self.schur_complement = schur
        self.schur_scale = scale
        self.status = status_
        self.column_solve_status = column_status
        self.numeric_version = version
        self.prepared_id = identifier

    @property
    def successful(self) -> Array:
        return self.status == int(BorderedSolveStatus.SUCCESS)


def _bind_bordered(
    system: BorderedLinearSystem,
    plan: BorderedSolvePlan,
    principal: PreparedLinearSolve,
    /,
    *,
    numeric_version: Any,
    prepared_id: str,
) -> PreparedBorderedSolve:
    column_result = solve_linear(principal, system.column)
    inverse_column = column_result.value
    column_status = column_result.status
    if int(column_status) != int(LinearSolveStatus.SUCCESS):
        status = BorderedSolveStatus.PRINCIPAL_COLUMN_SOLVE_FAILED
        inverse_column = system.operator.source.zeros()
        schur = jnp.asarray(jnp.nan, dtype=system.corner.dtype)
        scale = jnp.asarray(jnp.inf, dtype=jnp.real(system.corner).dtype)
    elif not bool(_finite_tree(inverse_column)):
        status = BorderedSolveStatus.CACHED_COLUMN_NONFINITE
        schur = jnp.asarray(jnp.nan, dtype=system.corner.dtype)
        scale = jnp.asarray(jnp.inf, dtype=jnp.real(system.corner).dtype)
    else:
        coupling = system.row_action(inverse_column)
        schur = system.corner - coupling
        scale = jnp.maximum(
            jnp.asarray(1.0, dtype=jnp.real(schur).dtype),
            jnp.maximum(jnp.abs(system.corner), jnp.abs(coupling)),
        )
        threshold = plan.schur_tolerance * scale
        if not bool(jnp.isfinite(schur)):
            status = BorderedSolveStatus.SCHUR_NONFINITE
        elif bool(jnp.abs(schur) <= threshold):
            status = BorderedSolveStatus.SCHUR_SINGULAR
        else:
            status = BorderedSolveStatus.SUCCESS
    return PreparedBorderedSolve(
        system,
        plan,
        principal,
        inverse_column,
        schur,
        scale,
        status,
        column_status,
        numeric_version,
        prepared_id=prepared_id,
    )


def prepare_bordered_solve(
    system: BorderedLinearSystem,
    plan: BorderedSolvePlan,
    /,
) -> PreparedBorderedSolve:
    """Bind numeric principal state and cache the bordered Schur data."""
    if not isinstance(system, BorderedLinearSystem):
        raise TypeError("system must be a BorderedLinearSystem.")
    if not isinstance(plan, BorderedSolvePlan):
        raise TypeError("plan must be a BorderedSolvePlan.")
    if system.system_id != plan.system_id:
        raise ValueError("Bordered system and plan IDs must match.")
    if system.operator.source.space_id != plan.source_space_id:
        raise ValueError("Bordered source space does not match the plan.")
    if system.operator.target.space_id != plan.target_space_id:
        raise ValueError("Bordered target space does not match the plan.")
    principal = prepare_linear(_principal_problem(system), plan.principal_plan)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-bordered-solve",
            "system_id": system.system_id,
            "plan_id": plan.plan_id,
        }
    )
    return _bind_bordered(
        system,
        plan,
        principal,
        numeric_version=principal.numeric_version,
        prepared_id=prepared_id,
    )


def refresh_bordered_solve(
    prepared: PreparedBorderedSolve,
    system: BorderedLinearSystem,
    /,
) -> PreparedBorderedSolve:
    """Refresh coefficients while preserving symbolic and prepared identities."""
    if not isinstance(prepared, PreparedBorderedSolve):
        raise TypeError("prepared must be a PreparedBorderedSolve.")
    if not isinstance(system, BorderedLinearSystem):
        raise TypeError("system must be a BorderedLinearSystem.")
    if system.system_id != prepared.system.system_id:
        raise ValueError("Numeric refreshes must preserve system_id.")
    if system.operator.source.space_id != prepared.plan.source_space_id:
        raise ValueError("Numeric refreshes must preserve the source space.")
    if system.operator.target.space_id != prepared.plan.target_space_id:
        raise ValueError("Numeric refreshes must preserve the target space.")
    principal = refresh_linear(prepared.principal, _principal_problem(system))
    return _bind_bordered(
        system,
        prepared.plan,
        principal,
        numeric_version=principal.numeric_version,
        prepared_id=prepared.prepared_id,
    )


class BorderedSolution(StrictModule):
    """Primal vector and scalar border coordinate from one solve."""

    primal: PyTree[Array]
    scalar: Array

    def __init__(self, primal: PyTree[Any], scalar: Any, /):
        self.primal = primal
        self.scalar = jnp.asarray(scalar)


class BorderedSolveDiagnostics(StrictModule):
    """Observable inner-solve, cache, Schur, and residual evidence."""

    column_solve_status: Array
    principal_solve_status: Array
    schur_complement: Array
    schur_threshold: Array
    residual_norm: Array
    principal_solve_count: Array
    cached_column_solve_reused: Array

    def __init__(
        self,
        *,
        column_solve_status: Any,
        principal_solve_status: Any,
        schur_complement: Any,
        schur_threshold: Any,
        residual_norm: Any,
        principal_solve_count: Any,
        cached_column_solve_reused: Any,
    ):
        self.column_solve_status = jnp.asarray(column_solve_status, dtype=jnp.int32)
        self.principal_solve_status = jnp.asarray(
            principal_solve_status,
            dtype=jnp.int32,
        )
        self.schur_complement = jnp.asarray(schur_complement)
        self.schur_threshold = jnp.asarray(schur_threshold)
        self.residual_norm = jnp.asarray(residual_norm)
        self.principal_solve_count = jnp.asarray(
            principal_solve_count,
            dtype=jnp.int32,
        )
        self.cached_column_solve_reused = jnp.asarray(
            cached_column_solve_reused,
            dtype=bool,
        )


class BorderedSolveProvenance(StrictModule):
    """Symbolic identities and numerical refresh version for one bordered solve."""

    numeric_version: Array
    system_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    principal_plan_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        numeric_version: Any,
        system_id: str,
        plan_id: str,
        prepared_id: str,
        principal_plan_id: str,
        operator_id: str,
    ):
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        identifiers = tuple(
            str(value)
            for value in (
                system_id,
                plan_id,
                prepared_id,
                principal_plan_id,
                operator_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Bordered solve provenance identities must be non-empty.")
        self.numeric_version = version
        (
            self.system_id,
            self.plan_id,
            self.prepared_id,
            self.principal_plan_id,
            self.operator_id,
        ) = identifiers


class BorderedSolveResult(StrictModule):
    """Bordered solution with explicit status, diagnostics, and provenance."""

    value: BorderedSolution
    status: Array
    diagnostics: BorderedSolveDiagnostics
    provenance: BorderedSolveProvenance

    def __init__(
        self,
        value: BorderedSolution,
        status: Any,
        diagnostics: BorderedSolveDiagnostics,
        provenance: BorderedSolveProvenance,
        /,
    ):
        if not isinstance(value, BorderedSolution):
            raise TypeError("value must be a BorderedSolution.")
        if not isinstance(diagnostics, BorderedSolveDiagnostics):
            raise TypeError("diagnostics must be BorderedSolveDiagnostics.")
        if not isinstance(provenance, BorderedSolveProvenance):
            raise TypeError("provenance must be BorderedSolveProvenance.")
        self.value = value
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance

    @property
    def successful(self) -> Array:
        return self.status == int(BorderedSolveStatus.SUCCESS)


def _provenance(prepared: PreparedBorderedSolve, /) -> BorderedSolveProvenance:
    return BorderedSolveProvenance(
        numeric_version=prepared.numeric_version,
        system_id=prepared.system.system_id,
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
        principal_plan_id=prepared.plan.principal_plan.plan_id,
        operator_id=prepared.system.operator.operator_id,
    )


def _diagnostics(
    prepared: PreparedBorderedSolve,
    /,
    *,
    principal_status: Any,
    residual_norm: Any,
    principal_solve_count: int,
    cache_reused: bool,
) -> BorderedSolveDiagnostics:
    return BorderedSolveDiagnostics(
        column_solve_status=prepared.column_solve_status,
        principal_solve_status=principal_status,
        schur_complement=prepared.schur_complement,
        schur_threshold=prepared.plan.schur_tolerance * prepared.schur_scale,
        residual_norm=residual_norm,
        principal_solve_count=principal_solve_count,
        cached_column_solve_reused=cache_reused,
    )


def _failed_result(
    prepared: PreparedBorderedSolve,
    status: Any,
    /,
    *,
    principal_status: Any,
    principal_solve_count: Any,
    cache_reused: Any,
) -> BorderedSolveResult:
    value = BorderedSolution(
        prepared.system.operator.source.zeros(),
        jnp.zeros((), dtype=prepared.system.corner.dtype),
    )
    return BorderedSolveResult(
        value,
        status,
        _diagnostics(
            prepared,
            principal_status=principal_status,
            residual_norm=jnp.asarray(jnp.inf),
            principal_solve_count=principal_solve_count,
            cache_reused=cache_reused,
        ),
        _provenance(prepared),
    )


def solve_bordered(
    prepared: PreparedBorderedSolve,
    right_hand_side: PyTree[Any],
    border_right_hand_side: Any,
    /,
) -> BorderedSolveResult:
    """Solve one RHS using cached ``A^-1 b`` and one new principal solve."""
    if not isinstance(prepared, PreparedBorderedSolve):
        raise TypeError("prepared must be a PreparedBorderedSolve.")
    system = prepared.system
    rhs = system.operator.target.validate(right_hand_side)
    border_rhs = _scalar(
        border_right_hand_side,
        name="border right-hand side",
        dtype=system.corner.dtype,
    )

    def prepared_failure(_: None) -> BorderedSolveResult:
        return _failed_result(
            prepared,
            prepared.status,
            principal_status=jnp.asarray(-1, dtype=jnp.int32),
            principal_solve_count=0,
            cache_reused=False,
        )

    def prepared_success(_: None) -> BorderedSolveResult:
        principal_result = solve_linear(prepared.principal, rhs)

        def principal_failure(_: None) -> BorderedSolveResult:
            return _failed_result(
                prepared,
                BorderedSolveStatus.PRINCIPAL_RIGHT_HAND_SIDE_SOLVE_FAILED,
                principal_status=principal_result.status,
                principal_solve_count=1,
                cache_reused=True,
            )

        def principal_success(_: None) -> BorderedSolveResult:
            unconstrained = principal_result.value
            scalar_candidate = (
                border_rhs - system.row_action(unconstrained)
            ) / prepared.schur_complement
            primal_candidate = tree_add_scaled(
                unconstrained,
                prepared.inverse_column,
                -scalar_candidate,
            )
            solution_finite = (
                _finite_tree(unconstrained)
                & jnp.isfinite(scalar_candidate)
                & _finite_tree(primal_candidate)
            )
            zero_primal = system.operator.source.zeros()
            primal = _tree_where(
                solution_finite,
                primal_candidate,
                zero_primal,
            )
            scalar = jnp.where(
                solution_finite,
                scalar_candidate,
                jnp.zeros((), dtype=system.corner.dtype),
            )
            target_value, border_value = system.apply(primal, scalar)
            target_residual = _tree_subtract(target_value, rhs)
            border_residual = border_value - border_rhs
            residual_squared = (
                jnp.real(
                    system.operator.target.inner(
                        target_residual,
                        target_residual,
                    )
                )
                + jnp.abs(border_residual) ** 2
            )
            computed_residual_norm = jnp.sqrt(jnp.maximum(residual_squared, 0.0))
            residual_norm = jnp.where(
                solution_finite,
                computed_residual_norm,
                jnp.asarray(jnp.inf, dtype=computed_residual_norm.dtype),
            )
            status = jnp.where(
                ~solution_finite,
                jnp.asarray(
                    BorderedSolveStatus.SOLUTION_NONFINITE,
                    dtype=jnp.int32,
                ),
                jnp.where(
                    jnp.isfinite(residual_norm),
                    jnp.asarray(
                        BorderedSolveStatus.SUCCESS,
                        dtype=jnp.int32,
                    ),
                    jnp.asarray(
                        BorderedSolveStatus.RESIDUAL_NONFINITE,
                        dtype=jnp.int32,
                    ),
                ),
            )
            return BorderedSolveResult(
                BorderedSolution(primal, scalar),
                status,
                _diagnostics(
                    prepared,
                    principal_status=principal_result.status,
                    residual_norm=residual_norm,
                    principal_solve_count=1,
                    cache_reused=True,
                ),
                _provenance(prepared),
            )

        return jax.lax.cond(
            principal_result.status == int(LinearSolveStatus.SUCCESS),
            principal_success,
            principal_failure,
            operand=None,
        )

    return jax.lax.cond(
        prepared.status == int(BorderedSolveStatus.SUCCESS),
        prepared_success,
        prepared_failure,
        operand=None,
    )


__all__ = [
    "BorderedLinearSystem",
    "BorderedSolution",
    "BorderedSolveDiagnostics",
    "BorderedSolvePlan",
    "BorderedSolveProvenance",
    "BorderedSolveResult",
    "BorderedSolveStatus",
    "PreparedBorderedSolve",
    "plan_bordered_solve",
    "prepare_bordered_solve",
    "refresh_bordered_solve",
    "solve_bordered",
]
