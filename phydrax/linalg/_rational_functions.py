#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._operators import AbstractLinearOperator
from ._policies import FailurePolicy
from ._shifted import (
    plan_shifted_solve,
    prepare_shifted_solve,
    PreparedShiftedSolve,
    refresh_shifted_solve,
    ShiftedLinearSystemFamily,
    ShiftedSolvePlan,
    ShiftedSolvePolicy,
    ShiftedSolveStatus,
    solve_shifted,
)
from ._spaces import _coordinate_dtype


class RationalFunctionStatus(IntEnum):
    """Portable status for one partial-fraction matrix-function action."""

    SUCCESS = 0
    SHIFTED_SOLVE_FAILURE = 1
    NONFINITE = 2


class PartialFractionRationalFunction(StrictModule):
    """Rational function ``sum_k c_k z^k + sum_j r_j / (p_j - z)``."""

    poles: Array
    residues: Array
    polynomial_coefficients: Array
    function_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        poles: ArrayLike,
        residues: ArrayLike,
        /,
        *,
        polynomial_coefficients: ArrayLike | tuple[float, ...] = (0.0,),
        function_id: str | None = None,
    ):
        poles_ = _numeric_vector(poles, "poles", nonempty=True)
        residues_ = _numeric_vector(residues, "residues", nonempty=True)
        polynomial_ = _numeric_vector(
            polynomial_coefficients,
            "polynomial_coefficients",
            nonempty=True,
        )
        if poles_.shape != residues_.shape:
            raise ValueError("poles and residues must have the same shape.")
        dtype = jnp.result_type(
            poles_.dtype, residues_.dtype, polynomial_.dtype, jnp.float32
        )
        poles_ = poles_.astype(dtype)
        residues_ = residues_.astype(dtype)
        polynomial_ = polynomial_.astype(dtype)
        structure = canonical_fingerprint(
            {
                "kind": "partial-fraction-rational-structure",
                "num_poles": int(poles_.size),
                "polynomial_size": int(polynomial_.size),
                "dtype": np.dtype(dtype).str,
                "convention": "pole-minus-argument",
            }
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "partial-fraction-rational-function",
                    "structure": structure,
                    "values": array_tree_fingerprint((poles_, residues_, polynomial_)),
                }
            )
            if function_id is None
            else str(function_id)
        )
        if not identifier:
            raise ValueError("function_id must be non-empty.")
        self.poles = poles_
        self.residues = residues_
        self.polynomial_coefficients = polynomial_
        self.function_id = identifier
        self.structure_id = structure

    @property
    def num_poles(self) -> int:
        return int(self.poles.size)

    @property
    def polynomial_degree(self) -> int:
        return int(self.polynomial_coefficients.size - 1)

    def __call__(self, value: ArrayLike, /) -> Array:
        argument = jnp.asarray(value)
        polynomial = jnp.zeros_like(argument, dtype=jnp.result_type(argument, self.poles))
        for coefficient in self.polynomial_coefficients[::-1]:
            polynomial = polynomial * argument + coefficient
        extra_axes = (1,) * argument.ndim
        poles = self.poles.reshape((self.num_poles,) + extra_axes)
        residues = self.residues.reshape((self.num_poles,) + extra_axes)
        return polynomial + jnp.sum(residues / (poles - argument), axis=0)


class RationalFunctionResourcePolicy(StrictModule):
    """Optional hard total-matvec and transient-workspace budgets."""

    max_matvec_count: int | None = eqx.field(static=True)
    max_workspace_bytes: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_matvec_count: int | None = None,
        max_workspace_bytes: int | None = None,
    ):
        self.max_matvec_count = _optional_nonnegative_int(
            max_matvec_count, "max_matvec_count"
        )
        self.max_workspace_bytes = _optional_nonnegative_int(
            max_workspace_bytes, "max_workspace_bytes"
        )


class RationalFunctionPolicy(StrictModule):
    """Shared shifted-solve, resource, and failure policy."""

    shifted: ShiftedSolvePolicy = eqx.field(static=True)
    resources: RationalFunctionResourcePolicy = eqx.field(static=True)
    failure: FailurePolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        shifted: ShiftedSolvePolicy | None = None,
        resources: RationalFunctionResourcePolicy | None = None,
        failure: FailurePolicy | None = None,
    ):
        shifted_ = ShiftedSolvePolicy() if shifted is None else shifted
        resources_ = RationalFunctionResourcePolicy() if resources is None else resources
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(shifted_, ShiftedSolvePolicy):
            raise TypeError("shifted must be a ShiftedSolvePolicy or None.")
        if not isinstance(resources_, RationalFunctionResourcePolicy):
            raise TypeError("resources must be a RationalFunctionResourcePolicy or None.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        self.shifted = shifted_
        self.resources = resources_
        self.failure = failure_


class RationalFunctionCostEstimate(StrictModule):
    """Shared-basis plus polynomial-action cost and memory estimate."""

    num_poles: int = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)
    shifted_setup_matvec_count: int = eqx.field(static=True)
    polynomial_matvec_count: int = eqx.field(static=True)
    total_matvec_count: int = eqx.field(static=True)
    retained_storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)


class RationalFunctionPlan(StrictModule):
    """Immutable symbolic plan for one rational-function structure."""

    shifted_plan: ShiftedSolvePlan = eqx.field(static=True)
    policy: RationalFunctionPolicy = eqx.field(static=True)
    cost: RationalFunctionCostEstimate = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    function_structure_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedRationalFunctionAction(StrictModule):
    """Reusable shared projection, right-hand side, and rational coefficients."""

    function: PartialFractionRationalFunction
    shifted: PreparedShiftedSolve
    plan: RationalFunctionPlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_fingerprint: str = eqx.field(static=True)
    function_fingerprint: str = eqx.field(static=True)
    numeric_version: Array
    refresh_count: Array

    @property
    def operator(self) -> AbstractLinearOperator:
        return self.shifted.family.operator

    @property
    def vector(self) -> PyTree[Array]:
        return self.shifted.rhs


class RationalFunctionDiagnostics(StrictModule):
    """Per-pole solve evidence and aggregate partial-fraction residual indicator."""

    shifted_status: Array
    shifted_residual_norm: Array
    shifted_relative_residual: Array
    shifted_condition_estimate: Array
    active_poles: Array
    residual_indicator: Array
    relative_residual_indicator: Array
    finite: Array
    converged: Array
    effective_dimension: Array
    krylov_breakdown_status: Array
    setup_matvec_count: Array
    polynomial_matvec_count: Array
    solve_matvec_count: Array
    retained_storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class RationalFunctionProvenance(StrictModule):
    """Plan, prepared state, function identity, convention, and numerical version."""

    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    function_id: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    numeric_version: Array


class RationalFunctionResult(StrictModule):
    """Rational matrix-function action with explicit solve-status evidence."""

    value: PyTree[Array]
    status: Array
    diagnostics: RationalFunctionDiagnostics
    provenance: RationalFunctionProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(RationalFunctionStatus.SUCCESS)


def plan_rational_function_action(
    operator: AbstractLinearOperator,
    function: PartialFractionRationalFunction,
    policy: RationalFunctionPolicy | None = None,
    /,
) -> RationalFunctionPlan:
    """Plan shared shifted solves and the polynomial part of a rational action."""
    _validate_operator_and_function(operator, function)
    selected = RationalFunctionPolicy() if policy is None else policy
    if not isinstance(selected, RationalFunctionPolicy):
        raise TypeError("policy must be a RationalFunctionPolicy or None.")
    family = ShiftedLinearSystemFamily(operator, function.poles)
    shifted_plan = plan_shifted_solve(family, selected.shifted)
    cost = _rational_cost(operator, function, shifted_plan)
    _validate_resources(cost, selected.resources)
    return RationalFunctionPlan(
        shifted_plan=shifted_plan,
        policy=selected,
        cost=cost,
        operator_id=operator.operator_id,
        function_structure_id=function.structure_id,
        plan_id=canonical_fingerprint(
            {
                "kind": "rational-function-plan",
                "operator": operator.operator_id,
                "function_structure": function.structure_id,
                "shifted_plan": shifted_plan.plan_id,
            }
        ),
    )


def prepare_rational_function_action(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    function: PartialFractionRationalFunction,
    policy: RationalFunctionPolicy | RationalFunctionPlan | None = None,
    /,
) -> PreparedRationalFunctionAction:
    """Build the shared pole projection for one operator and right-hand side."""
    plan = (
        policy
        if isinstance(policy, RationalFunctionPlan)
        else plan_rational_function_action(operator, function, policy)
    )
    _validate_plan(operator, function, plan)
    family = ShiftedLinearSystemFamily(operator, function.poles)
    shifted = prepare_shifted_solve(family, vector, plan.shifted_plan)
    return _prepared_rational(
        function,
        shifted,
        plan,
        numeric_version=0,
        refresh_count=0,
    )


def refresh_rational_function_action(
    prepared: PreparedRationalFunctionAction,
    operator: AbstractLinearOperator,
    function: PartialFractionRationalFunction,
    vector: PyTree[Any] | None = None,
    /,
) -> PreparedRationalFunctionAction:
    """Refresh operator, poles, coefficients, and optionally the right-hand side."""
    if not isinstance(prepared, PreparedRationalFunctionAction):
        raise TypeError("prepared must be a PreparedRationalFunctionAction.")
    _validate_plan(operator, function, prepared.plan)
    family = ShiftedLinearSystemFamily(operator, function.poles)
    shifted = refresh_shifted_solve(prepared.shifted, family, vector)
    return _prepared_rational(
        function,
        shifted,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        refresh_count=prepared.refresh_count + jnp.asarray(1, dtype=jnp.int32),
        prepared_id=prepared.prepared_id,
    )


def rational_function_action(
    operator_or_prepared: AbstractLinearOperator | PreparedRationalFunctionAction,
    vector: PyTree[Any] | None = None,
    function: PartialFractionRationalFunction | None = None,
    /,
    *,
    policy: RationalFunctionPolicy | RationalFunctionPlan | None = None,
) -> RationalFunctionResult:
    """Apply ``r(A)`` using one shared shifted basis and native polynomial actions."""
    if isinstance(operator_or_prepared, PreparedRationalFunctionAction):
        if vector is not None or function is not None or policy is not None:
            raise ValueError(
                "vector, function, and policy must be omitted for prepared rational state."
            )
        prepared = operator_or_prepared
    elif isinstance(operator_or_prepared, AbstractLinearOperator):
        if vector is None or function is None:
            raise ValueError("vector and function are required for an unprepared action.")
        prepared = prepare_rational_function_action(
            operator_or_prepared,
            vector,
            function,
            policy,
        )
    else:
        raise TypeError(
            "Expected an AbstractLinearOperator or PreparedRationalFunctionAction."
        )
    return _execute_rational(prepared)


def _execute_rational(
    prepared: PreparedRationalFunctionAction,
    /,
) -> RationalFunctionResult:
    operator = prepared.operator
    function = prepared.function
    shifted = solve_shifted(prepared.shifted)
    coefficients = function.polynomial_coefficients
    power = prepared.vector
    value = jax.tree.map(lambda leaf: coefficients[0] * leaf, power)
    for coefficient in coefficients[1:]:
        power = operator.mv(power)
        value = jax.tree.map(
            lambda accumulated, leaf: accumulated + coefficient * leaf,
            value,
            power,
        )
    active = jnp.abs(function.residues) > 0
    rational_term = jax.tree.map(
        lambda leaf: _weighted_shift_sum(leaf, function.residues, active),
        shifted.value,
    )
    value = jax.tree.map(lambda left, right: left + right, value, rational_term)
    value_norm = _tree_norm(value)
    indicator = jnp.sum(
        jnp.where(
            active, jnp.abs(function.residues) * shifted.diagnostics.residual_norm, 0
        )
    )
    tiny = jnp.asarray(jnp.finfo(value_norm.dtype).tiny)
    relative_indicator = indicator / jnp.maximum(value_norm, tiny)
    active_succeeded = jnp.all(
        jnp.where(active, shifted.status == int(ShiftedSolveStatus.SUCCESS), True)
    )
    finite = _tree_all_finite(value) & jnp.isfinite(indicator)
    converged = active_succeeded & finite
    status = jnp.where(
        ~finite,
        int(RationalFunctionStatus.NONFINITE),
        jnp.where(
            ~active_succeeded,
            int(RationalFunctionStatus.SHIFTED_SOLVE_FAILURE),
            int(RationalFunctionStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    if prepared.plan.policy.failure.mode == "error":
        value = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                status != int(RationalFunctionStatus.SUCCESS),
                "Rational matrix-function action did not satisfy its solve contract.",
            ),
            value,
        )
    setup_matvecs = shifted.diagnostics.setup_matvec_count + function.polynomial_degree
    diagnostics = RationalFunctionDiagnostics(
        shifted_status=shifted.status,
        shifted_residual_norm=shifted.diagnostics.residual_norm,
        shifted_relative_residual=shifted.diagnostics.relative_residual,
        shifted_condition_estimate=shifted.diagnostics.condition_estimate,
        active_poles=active,
        residual_indicator=indicator,
        relative_residual_indicator=relative_indicator,
        finite=finite,
        converged=converged,
        effective_dimension=shifted.diagnostics.iterations[0],
        krylov_breakdown_status=shifted.diagnostics.krylov_breakdown_status,
        setup_matvec_count=setup_matvecs,
        polynomial_matvec_count=jnp.asarray(function.polynomial_degree, dtype=jnp.int32),
        solve_matvec_count=shifted.diagnostics.solve_matvec_count,
        retained_storage_bytes=prepared.plan.cost.retained_storage_bytes,
        workspace_bytes=prepared.plan.cost.workspace_bytes,
    )
    return RationalFunctionResult(
        value=value,
        status=status,
        diagnostics=diagnostics,
        provenance=RationalFunctionProvenance(
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=operator.operator_id,
            function_id=function.function_id,
            convention="r(z)=sum_k c_k z^k + sum_j r_j/(p_j-z)",
            method=f"shared-{prepared.plan.shifted_plan.selected_method}",
            numeric_version=prepared.numeric_version,
        ),
    )


def _rational_cost(
    operator: AbstractLinearOperator,
    function: PartialFractionRationalFunction,
    shifted_plan: ShiftedSolvePlan,
    /,
) -> RationalFunctionCostEstimate:
    polynomial_matvecs = function.polynomial_degree
    shifted_matvecs = shifted_plan.cost.matvec_count
    output_itemsize = jnp.result_type(
        _coordinate_dtype(operator.source), function.poles.dtype
    ).itemsize
    vector_bytes = operator.source.size * output_itemsize
    retained = shifted_plan.cost.total_storage_bytes
    workspace = shifted_plan.cost.workspace_bytes + 2 * vector_bytes
    return RationalFunctionCostEstimate(
        num_poles=function.num_poles,
        polynomial_degree=polynomial_matvecs,
        shifted_setup_matvec_count=shifted_matvecs,
        polynomial_matvec_count=polynomial_matvecs,
        total_matvec_count=shifted_matvecs + polynomial_matvecs,
        retained_storage_bytes=retained,
        workspace_bytes=workspace,
        exact=False,
    )


def _prepared_rational(
    function: PartialFractionRationalFunction,
    shifted: PreparedShiftedSolve,
    plan: RationalFunctionPlan,
    *,
    numeric_version: Any,
    refresh_count: Any,
    prepared_id: str | None = None,
) -> PreparedRationalFunctionAction:
    operator_fingerprint = canonical_fingerprint(
        array_tree_fingerprint(shifted.family.operator)
    )
    function_fingerprint = canonical_fingerprint(
        array_tree_fingerprint(
            (function.poles, function.residues, function.polynomial_coefficients)
        )
    )
    identifier = (
        canonical_fingerprint(
            {
                "kind": "prepared-rational-function-action",
                "plan": plan.plan_id,
                "operator": operator_fingerprint,
                "function": function_fingerprint,
            }
        )
        if prepared_id is None
        else prepared_id
    )
    return PreparedRationalFunctionAction(
        function=function,
        shifted=shifted,
        plan=plan,
        prepared_id=identifier,
        operator_fingerprint=operator_fingerprint,
        function_fingerprint=function_fingerprint,
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
    )


def _validate_operator_and_function(
    operator: AbstractLinearOperator,
    function: PartialFractionRationalFunction,
    /,
) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Rational matrix functions require an unbatched endomorphism.")
    if not isinstance(function, PartialFractionRationalFunction):
        raise TypeError("function must be a PartialFractionRationalFunction.")


def _validate_plan(
    operator: AbstractLinearOperator,
    function: PartialFractionRationalFunction,
    plan: RationalFunctionPlan,
    /,
) -> None:
    _validate_operator_and_function(operator, function)
    if not isinstance(plan, RationalFunctionPlan):
        raise TypeError("plan must be a RationalFunctionPlan.")
    if operator.operator_id != plan.operator_id:
        raise ValueError("Rational plan belongs to a different symbolic operator.")
    if function.structure_id != plan.function_structure_id:
        raise ValueError("Rational function structure does not match the plan.")


def _validate_resources(
    cost: RationalFunctionCostEstimate,
    resources: RationalFunctionResourcePolicy,
    /,
) -> None:
    if (
        resources.max_matvec_count is not None
        and cost.total_matvec_count > resources.max_matvec_count
    ):
        raise ValueError(
            f"Rational action requires {cost.total_matvec_count} matvecs, exceeding "
            f"the policy limit {resources.max_matvec_count}."
        )
    if (
        resources.max_workspace_bytes is not None
        and cost.workspace_bytes > resources.max_workspace_bytes
    ):
        raise ValueError(
            f"Rational action requires {cost.workspace_bytes} workspace bytes, "
            f"exceeding the policy limit {resources.max_workspace_bytes}."
        )


def _weighted_shift_sum(values: Array, residues: Array, active: Array, /) -> Array:
    broadcast = (values.shape[0],) + (1,) * (values.ndim - 1)
    safe_values = jnp.where(active.reshape(broadcast), values, 0)
    return jnp.sum(residues.reshape(broadcast) * safe_values, axis=0)


def _tree_norm(tree: PyTree[Array], /) -> Array:
    squared = sum(
        (jnp.real(jnp.vdot(leaf, leaf)) for leaf in jax.tree.leaves(tree)),
        start=jnp.asarray(0.0),
    )
    return jnp.sqrt(jnp.maximum(squared, 0))


def _tree_all_finite(tree: PyTree[Array], /) -> Array:
    return jnp.all(
        jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)))
    )


def _numeric_vector(
    value: ArrayLike | tuple[float, ...],
    name: str,
    /,
    *,
    nonempty: bool,
) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1 or (nonempty and array.size < 1):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}rank-one array.")
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(
        array.dtype, jnp.bool_
    ):
        raise TypeError(f"{name} must contain real or complex numbers.")
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must be finite.")
    return array


def _optional_nonnegative_int(value: int | None, name: str, /) -> int | None:
    if value is None:
        return None
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative or None.")
    return integer


__all__ = [
    "PartialFractionRationalFunction",
    "PreparedRationalFunctionAction",
    "RationalFunctionCostEstimate",
    "RationalFunctionDiagnostics",
    "RationalFunctionPlan",
    "RationalFunctionPolicy",
    "RationalFunctionProvenance",
    "RationalFunctionResourcePolicy",
    "RationalFunctionResult",
    "RationalFunctionStatus",
    "plan_rational_function_action",
    "prepare_rational_function_action",
    "rational_function_action",
    "refresh_rational_function_action",
]
