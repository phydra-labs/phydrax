#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native operator-only scaled Taylor exponential action."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._admissibility import guard_derivative_validity
from ._exponential_taylor_planning import (
    plan_taylor_exponential_action,
    prepare_taylor_exponential_action,
    PreparedTaylorExponentialAction,
    refresh_taylor_exponential_action,
    TaylorExponentialPlan,
    TaylorExponentialPolicy,
    TaylorExponentialResourcePolicy,
)
from ._exponential_taylor_recurrence import select_taylor_degree, taylor_recurrence
from ._matrix_function_contracts import (
    MatrixFunctionDiagnostics,
    MatrixFunctionProvenance,
    MatrixFunctionResult,
    MatrixFunctionStatus,
)
from ._operators import AbstractLinearOperator


def _promoted_unflatten(template: PyTree[Array], coordinates: Array, /) -> PyTree[Array]:
    leaves, structure = jax.tree.flatten(template)
    values = []
    offset = 0
    for leaf in leaves:
        values.append(coordinates[offset : offset + leaf.size].reshape(leaf.shape))
        offset += leaf.size
    return jax.tree.unflatten(structure, values)


def _truncation_bound(
    input_norm: Array,
    operator_norm: Array,
    scale: Array,
    shift: Array,
    degree: Array,
    scaling_count: Array,
    /,
) -> Array:
    """Mathematical 1-norm truncation bound, excluding floating-point roundoff.

    The bound uses ||A-mu I||_1 <= ||A||_1+|mu|, and the Taylor
    remainder and polynomial norm are bounded by exp(||scale*(A-mu I)||_1).
    It is not valid if operator_norm comes from a randomized lower estimate.
    """
    real_dtype = input_norm.dtype
    total_norm = jnp.abs(scale) * (operator_norm + jnp.abs(shift))
    segments = scaling_count.astype(real_dtype)
    order = (degree + 1).astype(real_dtype)
    log_norm = jnp.log(jnp.where(total_norm > 0, total_norm / segments, 1.0))
    log_error = (
        jnp.log(segments)
        + total_norm
        + order * log_norm
        - jax.lax.lgamma(order + 1)
        + jnp.real(scale * shift)
        + jnp.log(jnp.where(input_norm > 0, input_norm, 1.0))
    )
    return jnp.where((input_norm == 0) | (total_norm == 0), 0.0, jnp.exp(log_error))


def _differentiable_execution_inputs(
    operator: AbstractLinearOperator,
    coordinates: Array,
    scale: Array,
    shift: Array,
    mode: str,
    /,
) -> tuple[AbstractLinearOperator, Array, Array, Array]:
    if mode not in ("rhs-only", "none"):
        return operator, coordinates, scale, shift
    execution_operator = jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_inexact_array(leaf) else leaf,
        operator,
    )
    coordinates_ = jax.lax.stop_gradient(coordinates) if mode == "none" else coordinates
    return (
        execution_operator,
        coordinates_,
        jax.lax.stop_gradient(scale),
        jax.lax.stop_gradient(shift),
    )


def _taylor_status(
    plan: TaylorExponentialPlan,
    prepared: PreparedTaylorExponentialAction,
    inputs_finite: Array,
    admissible: Array,
    finite: Array,
    converged: Array,
    /,
) -> Array:
    return jnp.where(
        jnp.asarray(not plan.feasible),
        int(MatrixFunctionStatus.RESOURCE_EXHAUSTED),
        jnp.where(
            ~prepared.norm_finite,
            int(MatrixFunctionStatus.PLANNING_FAILURE),
            jnp.where(
                ~inputs_finite,
                int(MatrixFunctionStatus.NONFINITE),
                jnp.where(
                    ~admissible,
                    int(MatrixFunctionStatus.RESOURCE_EXHAUSTED),
                    jnp.where(
                        converged,
                        int(MatrixFunctionStatus.SUCCESS),
                        jnp.where(
                            finite,
                            int(MatrixFunctionStatus.TOLERANCE_NOT_MET),
                            int(MatrixFunctionStatus.NONFINITE),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _finalize_taylor_value(
    value: PyTree[Array],
    status: Array,
    derivative_valid: Array,
    dependencies: Any,
    plan: TaylorExponentialPlan,
    /,
) -> PyTree[Array]:
    if plan.policy.failure.mode == "error":
        value = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                status != int(MatrixFunctionStatus.SUCCESS),
                "Taylor exponential action did not satisfy its numerical contract.",
            ),
            value,
        )
    if plan.policy.differentiation.mode == "none":
        return jax.tree.map(jax.lax.stop_gradient, value)
    return guard_derivative_validity(
        value,
        derivative_valid,
        dependencies=dependencies,
        failure=plan.policy.failure.mode,
        message=(
            "Taylor exponential action has no valid derivative; inspect "
            "status-mode diagnostics."
        ),
    )


def execute_taylor_exponential_action(
    operator_or_prepared: AbstractLinearOperator | PreparedTaylorExponentialAction,
    vector: PyTree[Any],
    scale: ArrayLike = 1.0,
    /,
    *,
    policy: TaylorExponentialPolicy | None = None,
    key: Array | None = None,
    trace: ArrayLike | None = None,
) -> MatrixFunctionResult:
    """Compute exp(scale*A) vector with fixed-degree, fixed-capacity JAX loops.

    A supplied ``trace`` is the full trace of A; the recurrence shifts by
    trace / dimension. Norm estimation is prepared once per operator, not per
    right-hand side. An estimated norm cannot certify the truncation error.
    """
    if isinstance(operator_or_prepared, PreparedTaylorExponentialAction):
        if policy is not None or key is not None:
            raise ValueError(
                "A prepared Taylor action already fixes its policy and norm key."
            )
        prepared = operator_or_prepared
    else:
        prepared = prepare_taylor_exponential_action(
            operator_or_prepared, policy, key=key
        )
    operator, plan = prepared.operator, prepared.plan
    differentiation = plan.policy.differentiation
    validated = operator.source.validate(vector)
    original = operator.source.flatten(validated)
    scale_value = jnp.asarray(scale)
    if scale_value.shape != () or not jnp.issubdtype(scale_value.dtype, jnp.number):
        raise TypeError("scale must be a real or complex scalar.")
    full_trace = prepared.trace if trace is None else jnp.asarray(trace)
    if full_trace.shape != () or not jnp.issubdtype(full_trace.dtype, jnp.number):
        raise TypeError("trace must be a real or complex scalar.")
    shift = full_trace / plan.dimension
    dtype = jnp.result_type(original.dtype, scale_value.dtype, shift.dtype)
    if not jnp.issubdtype(dtype, jnp.inexact):
        raise TypeError("Taylor action requires real or complex inexact coordinates.")
    coordinates = original.astype(dtype)
    scale_value = scale_value.astype(dtype)
    shift = shift.astype(dtype)
    execution_operator, coordinates, scale_value, shift = (
        _differentiable_execution_inputs(
            operator,
            coordinates,
            scale_value,
            shift,
            differentiation.mode,
        )
    )
    norm = prepared.norm_one
    norm_scale = jnp.abs(scale_value) * (norm + jnp.abs(shift))
    native_dtype = jax.tree.leaves(operator.source.structure())[0].dtype
    complex_extension = jnp.issubdtype(dtype, jnp.complexfloating) and not jnp.issubdtype(
        native_dtype, jnp.complexfloating
    )
    action_factor = 2 if complex_extension else 1
    alpha_scale = jnp.where(
        shift == 0,
        jnp.abs(scale_value) * prepared.alpha_p,
        jnp.full_like(prepared.alpha_p, norm_scale),
    )
    degree, scaling_count, admissible = select_taylor_degree(
        norm_scale, alpha_scale, plan, action_factor
    )
    input_norm = jnp.sum(jnp.abs(coordinates))
    inputs_finite = (
        prepared.norm_finite
        & jnp.isfinite(scale_value)
        & jnp.isfinite(shift)
        & jnp.all(jnp.isfinite(coordinates))
    )
    execute = plan.feasible & inputs_finite & admissible

    def calculate(_: None) -> tuple[Array, Array, Array]:
        return taylor_recurrence(
            execution_operator,
            coordinates,
            scale_value,
            shift,
            degree,
            scaling_count,
            plan.policy.resources.max_degree,
            plan.policy.resources.max_scaling_count,
        )

    def refuse(_: None) -> tuple[Array, Array, Array]:
        return (
            jnp.full_like(coordinates, jnp.nan),
            jnp.asarray(jnp.inf, dtype=input_norm.dtype),
            jnp.asarray(False),
        )

    answer, observed_tail, action_finite = jax.lax.cond(execute, calculate, refuse, None)
    bound_available = plan.norm_source != "estimated-block-1-norm"
    bound = (
        _truncation_bound(input_norm, norm, scale_value, shift, degree, scaling_count)
        if bound_available
        else jnp.asarray(jnp.inf, dtype=input_norm.dtype)
    )
    comparison = jnp.maximum(input_norm, jnp.sum(jnp.abs(answer)))
    comparison = jnp.maximum(comparison, jnp.asarray(1.0, dtype=comparison.dtype))
    error_estimate = observed_tail / comparison
    tolerance_met = observed_tail <= plan.policy.error_tolerance * comparison
    finite = inputs_finite & action_finite & jnp.isfinite(observed_tail)
    converged = execute & finite & tolerance_met
    status = _taylor_status(
        plan,
        prepared,
        inputs_finite,
        admissible,
        finite,
        converged,
    )
    performed = jnp.where(execute, action_factor * scaling_count * (degree + 1), 0)
    derivative_valid = converged & jnp.asarray(differentiation.mode != "none")
    value = _finalize_taylor_value(
        _promoted_unflatten(validated, answer),
        status,
        derivative_valid,
        (operator, scale_value),
        plan,
    )
    return MatrixFunctionResult(
        value=value,
        status=status,
        diagnostics=MatrixFunctionDiagnostics(
            error_estimate=error_estimate,
            residual_estimate=observed_tail,
            error_bound=bound,
            error_bound_available=jnp.asarray(bound_available) & execute,
            error_bound_certified=jnp.asarray(False),
            finite=finite,
            converged=converged,
            derivative_valid=derivative_valid,
            effective_dimension=jnp.asarray(plan.dimension, dtype=jnp.int32),
            selected_degree=jnp.where(execute, degree, 0),
            scaling_count=jnp.where(execute, scaling_count, 0),
            setup_matvec_count=jnp.asarray(
                plan.setup_matvec_count if plan.feasible else 0, dtype=jnp.int32
            ),
            action_matvec_count=performed,
            transpose_matvec_count=jnp.asarray(
                plan.transpose_matvec_count if plan.feasible else 0,
                dtype=jnp.int32,
            ),
            breakdown_status=jnp.asarray(0, dtype=jnp.int32),
            retained_storage_bytes=plan.retained_storage_bytes,
            workspace_bytes=plan.workspace_bytes,
        ),
        provenance=MatrixFunctionProvenance(
            method="taylor",
            kind="exp",
            description=(
                "Scaled full-degree Taylor; analytical truncation bound excludes "
                "floating-point roundoff and is not certified total error"
                if bound_available
                else "Scaled full-degree Taylor with randomized alpha-power norm "
                "estimates; error indicator is not certified"
            ),
            operator_id=operator.operator_id,
            plan_id=plan.plan_id,
            prepared_id=prepared.prepared_id,
            trace_source=("provided" if trace is not None else plan.trace_source),
            norm_source=plan.norm_source,
            numeric_version=prepared.numeric_version,
        ),
    )


__all__ = [
    "TaylorExponentialPolicy",
    "TaylorExponentialResourcePolicy",
    "TaylorExponentialPlan",
    "PreparedTaylorExponentialAction",
    "plan_taylor_exponential_action",
    "prepare_taylor_exponential_action",
    "refresh_taylor_exponential_action",
    "execute_taylor_exponential_action",
]
