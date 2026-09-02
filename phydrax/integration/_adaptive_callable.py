#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._breakpoints import discover_breakpoints
from ._estimates import (
    AdaptivePartition,
    AdaptiveQuadratureDiagnostics,
    AdaptiveTriangleDiagnostics,
    AdaptiveTrianglePartition,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._plans import AdaptiveQuadraturePlan, AdaptiveTrianglePlan
from ._precision import IntegrationPrecisionPolicy
from ._rules import (
    clenshaw_curtis_data,
    ClenshawCurtisRule,
    GaussKronrodRule,
    interval_rule_data,
    tanh_sinh_data,
    TanhSinhRule,
)
from ._status import IntegrationStatus


def _local_rule(plan: AdaptiveQuadraturePlan, /):
    high = interval_rule_data(plan.rule)
    if isinstance(plan.rule, GaussKronrodRule):
        if high.embedded_weights is None:
            raise RuntimeError("Gauss--Kronrod rule is missing embedded weights.")
        return high, None, int(high.nodes.shape[0])
    if isinstance(plan.rule, ClenshawCurtisRule):
        low_order = 2 if plan.rule.level == 1 else 2 ** (plan.rule.level - 1) + 1
        low = clenshaw_curtis_data(low_order)
        return high, low, int(high.nodes.shape[0] + low.nodes.shape[0])
    if isinstance(plan.rule, TanhSinhRule):
        low_order = max(3, plan.rule.order - 20)
        if low_order % 2 == 0:
            low_order -= 1
        low = tanh_sinh_data(low_order)
        return high, low, int(high.nodes.shape[0] + low.nodes.shape[0])
    raise TypeError(
        "AdaptiveQuadraturePlan requires GaussKronrodRule, "
        "ClenshawCurtisRule, or TanhSinhRule."
    )


def _error_norm(value: Array, /) -> Array:
    return jnp.max(jnp.abs(jnp.asarray(value)))


def _meets_plan_tolerance(
    value: Array,
    error: Array,
    plan: AdaptiveQuadraturePlan,
    precision: IntegrationPrecisionPolicy,
    /,
) -> Array:
    magnitude = precision.decision(_error_norm(value))
    error_ = precision.decision(error)
    real_dtype = magnitude.dtype
    absolute = (
        jnp.sqrt(jnp.finfo(real_dtype).eps)
        if plan.absolute_tolerance is None
        else jnp.asarray(plan.absolute_tolerance, dtype=real_dtype)
    )
    relative = (
        jnp.sqrt(jnp.finfo(real_dtype).eps)
        if plan.relative_tolerance is None
        else jnp.asarray(plan.relative_tolerance, dtype=real_dtype)
    )
    return error_ <= absolute + relative * magnitude


def adaptive_interval_callable(
    integrand: Callable[[Array], Array],
    bounds: Array,
    plan: AdaptiveQuadraturePlan,
    /,
    *,
    precision: IntegrationPrecisionPolicy | None = None,
) -> IntegrationEstimate:
    """Integrate a batched callable over one interval with a bounded plan.

    ``integrand`` receives an array of reference coordinates with shape ``(n,)``
    and returns values whose leading axis has the same length.  The callback is
    intentionally independent of domains, components, and boundary kernels so
    specialized evaluators can reuse the integration plan and diagnostics.
    """
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(plan, AdaptiveQuadraturePlan):
        raise TypeError("plan must be an AdaptiveQuadraturePlan.")
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy.")
    bounds_ = jnp.asarray(bounds)
    if bounds_.shape != (2,):
        raise ValueError("Adaptive callable bounds must have shape (2,).")

    def evaluate_points(points: Array) -> Array:
        values = precision_.evaluation(jnp.asarray(integrand(points)))
        if values.ndim == 0 or values.shape[0] != points.shape[0]:
            raise ValueError(
                "Adaptive callable integrands must preserve the leading point axis."
            )
        return values

    prototype = evaluate_points(jnp.asarray([0.5 * (bounds_[0] + bounds_[1])]))
    output_shape = prototype.shape[1:]
    discovery = None
    discovery_cost = jnp.asarray(0, dtype=jnp.int32)
    discovered_points = jnp.zeros((0,), dtype=bounds_.dtype)
    if plan.discovery is not None:
        discovery, discovery_cost = discover_breakpoints(
            evaluate_points,
            bounds_,
            plan.discovery,
            explicit=plan.breakpoints,
        )
        discovered_points = jnp.where(
            discovery.active,
            discovery.points,
            bounds_[1],
        )
    endpoints = precision_.accumulation(
        jnp.sort(
            jnp.concatenate(
                (
                    bounds_[:1],
                    jnp.asarray(plan.breakpoints, dtype=bounds_.dtype),
                    discovered_points,
                    bounds_[1:],
                )
            )
        )
    )
    initial_count = (
        len(plan.breakpoints)
        + (0 if plan.discovery is None else plan.discovery.max_candidates)
        + 1
    )

    high, low, local_cost = _local_rule(plan)
    discovery_static_cost = (
        0
        if plan.discovery is None
        else plan.discovery.pilot_count
        + plan.discovery.refinement_rounds * plan.discovery.max_candidates
    )
    initial_cost = initial_count * local_cost + 1 + discovery_static_cost
    if plan.max_evaluations is not None and plan.max_evaluations < initial_cost:
        bounds_valid = (bounds_[1] > bounds_[0]) & jnp.all(jnp.diff(endpoints) >= 0.0)
        finite = jnp.all(jnp.isfinite(prototype))
        status = jnp.where(
            bounds_valid,
            int(IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED),
            int(IntegrationStatus.INVALID_BOUNDS),
        )
        status = jnp.where(
            finite,
            status,
            int(IntegrationStatus.NONFINITE_INTEGRAND),
        )
        value_data = jnp.zeros(output_shape, dtype=prototype.dtype)
        if plan.throw:
            value_data = eqx.error_if(
                value_data,
                status != int(IntegrationStatus.CONVERGED),
                "Adaptive quadrature failed to meet its numerical contract.",
            )
        error = precision_.decision(jnp.asarray(jnp.inf))
        diagnostics = AdaptiveQuadratureDiagnostics(
            status=status,
            num_evaluations=jnp.asarray(1 + discovery_static_cost, dtype=jnp.int32),
            estimated_error=error,
            partition=None,
            rule=type(plan.rule).__name__,
            discovery=discovery,
            discovery_count=(
                0 if discovery is None else jnp.sum(discovery.active, dtype=jnp.int32)
            ),
            discovery_overflow=(
                False
                if discovery is None
                else discovery.status
                == int(IntegrationStatus.BREAKPOINT_CANDIDATE_OVERFLOW)
            ),
        )
        return IntegrationEstimate(
            value_data,
            status=status,
            num_evaluations=1 + discovery_static_cost,
            error_estimate=error,
            error_kind="embedded-rule",
            diagnostics=diagnostics,
            provenance=IntegrationProvenance(
                "adaptive-callable", "callable", type(plan.rule).__name__
            ),
        )

    def evaluate_interval(lower: Array, upper: Array) -> tuple[Array, Array, Array]:
        def evaluate(_):
            half = 0.5 * (upper - lower)
            center = 0.5 * (upper + lower)
            high_nodes = precision_.accumulation(high.nodes)
            high_weights = precision_.accumulation(high.weights)
            high_points = center + half * high_nodes
            high_values = precision_.accumulation(evaluate_points(high_points))
            high_estimate = precision_.accumulation(
                half * jnp.tensordot(high_weights, high_values, axes=(0, 0))
            )
            if high.embedded_weights is not None:
                low_estimate = precision_.accumulation(
                    half
                    * jnp.tensordot(
                        precision_.accumulation(high.embedded_weights),
                        high_values,
                        axes=(0, 0),
                    )
                )
                low_finite = jnp.asarray(True)
            else:
                if low is None:
                    raise RuntimeError("Nested adaptive rule is missing its low rule.")
                low_points = center + half * precision_.accumulation(low.nodes)
                low_values = precision_.accumulation(evaluate_points(low_points))
                low_estimate = precision_.accumulation(
                    half
                    * jnp.tensordot(
                        precision_.accumulation(low.weights),
                        low_values,
                        axes=(0, 0),
                    )
                )
                low_finite = jnp.all(jnp.isfinite(low_values))
            error = precision_.decision(_error_norm(high_estimate - low_estimate))
            finite = (
                jnp.all(jnp.isfinite(high_values))
                & jnp.all(jnp.isfinite(high_estimate))
                & low_finite
                & jnp.all(jnp.isfinite(low_estimate))
                & jnp.isfinite(error)
            )
            return high_estimate, error, finite

        return jax.lax.cond(
            upper > lower,
            evaluate,
            lambda _: (
                jnp.zeros(output_shape, dtype=prototype.dtype),
                precision_.decision(jnp.asarray(0.0)),
                jnp.asarray(True),
            ),
            operand=None,
        )

    initial_estimates, initial_errors, initial_finite = jax.vmap(evaluate_interval)(
        endpoints[:-1], endpoints[1:]
    )
    capacity = plan.max_intervals
    lower_bounds = (
        jnp.zeros((capacity,), dtype=endpoints.dtype)
        .at[:initial_count]
        .set(endpoints[:-1])
    )
    upper_bounds = (
        jnp.zeros((capacity,), dtype=endpoints.dtype)
        .at[:initial_count]
        .set(endpoints[1:])
    )
    estimates = (
        jnp.zeros((capacity,) + output_shape, dtype=initial_estimates.dtype)
        .at[:initial_count]
        .set(initial_estimates)
    )
    errors = (
        jnp.zeros((capacity,), dtype=initial_errors.dtype)
        .at[:initial_count]
        .set(initial_errors)
    )
    positive_initial = jnp.diff(endpoints) > 0.0
    active = (
        (jnp.arange(capacity) < initial_count).at[:initial_count].set(positive_initial)
    )
    global_estimate = precision_.accumulation(jnp.sum(initial_estimates, axis=0))
    global_error = precision_.decision(jnp.sum(initial_errors))
    evaluation_count = (
        jnp.sum(positive_initial, dtype=jnp.int32) * local_cost + 1 + discovery_cost
    )

    def converged(estimate: Array, error: Array) -> Array:
        return _meets_plan_tolerance(estimate, error, plan, precision_)

    bounds_valid = (bounds_[1] > bounds_[0]) & jnp.all(jnp.diff(endpoints) >= 0.0)
    finite_valid = jnp.all(initial_finite)
    initial_status = jnp.where(
        bounds_valid,
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.INVALID_BOUNDS),
    )
    initial_status = jnp.where(
        finite_valid,
        initial_status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    done = (~bounds_valid) | (~finite_valid) | converged(global_estimate, global_error)
    max_evaluations = (
        capacity * 2 * local_cost + initial_count * local_cost + 1 + discovery_static_cost
        if plan.max_evaluations is None
        else plan.max_evaluations
    )
    state = (
        lower_bounds,
        upper_bounds,
        estimates,
        errors,
        active,
        jnp.asarray(initial_count, dtype=jnp.int32),
        global_estimate,
        global_error,
        evaluation_count,
        jnp.asarray(initial_status, dtype=jnp.int32),
        done,
    )

    def iteration(carry, _):
        (
            lowers,
            uppers,
            interval_estimates,
            interval_errors,
            active_mask,
            count,
            total_estimate,
            total_error,
            evaluations,
            status,
            finished,
        ) = carry

        def refine(current):
            (
                lowers_,
                uppers_,
                estimates_,
                errors_,
                active_,
                count_,
                total_,
                error_,
                evaluations_,
                status_,
                _finished,
            ) = current
            capacity_exhausted = count_ >= capacity
            evaluation_exhausted = evaluations_ + 2 * local_cost > max_evaluations

            def fail_capacity(values):
                status_value = jnp.where(
                    capacity_exhausted,
                    int(IntegrationStatus.MAXIMUM_INTERVALS_REACHED),
                    int(IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED),
                )
                return values[:-2] + (
                    jnp.asarray(status_value, dtype=jnp.int32),
                    jnp.asarray(True),
                )

            def split(values):
                (
                    lower_values,
                    upper_values,
                    estimate_values,
                    error_values,
                    active_values,
                    count_value,
                    total_value,
                    total_error_value,
                    evaluation_value,
                    status_value,
                    _done_value,
                ) = values
                selected = jnp.argmax(jnp.where(active_values, error_values, -jnp.inf))
                lower = lower_values[selected]
                upper = upper_values[selected]
                midpoint = 0.5 * (lower + upper)
                stagnated = (midpoint == lower) | (midpoint == upper)
                left_estimate, left_error, left_finite = evaluate_interval(
                    lower, midpoint
                )
                right_estimate, right_error, right_finite = evaluate_interval(
                    midpoint, upper
                )
                finite = left_finite & right_finite
                new_total = precision_.accumulation(
                    total_value
                    - estimate_values[selected]
                    + left_estimate
                    + right_estimate
                )
                new_error = precision_.decision(
                    total_error_value - error_values[selected] + left_error + right_error
                )
                lower_values = lower_values.at[selected].set(lower)
                upper_values = upper_values.at[selected].set(midpoint)
                estimate_values = estimate_values.at[selected].set(left_estimate)
                error_values = error_values.at[selected].set(left_error)
                lower_values = lower_values.at[count_value].set(midpoint)
                upper_values = upper_values.at[count_value].set(upper)
                estimate_values = estimate_values.at[count_value].set(right_estimate)
                error_values = error_values.at[count_value].set(right_error)
                active_values = active_values.at[count_value].set(True)
                count_value = count_value + 1
                evaluation_value = evaluation_value + 2 * local_cost
                status_value = jnp.where(
                    finite,
                    status_value,
                    int(IntegrationStatus.NONFINITE_INTEGRAND),
                )
                status_value = jnp.where(
                    stagnated,
                    int(IntegrationStatus.REFINEMENT_STAGNATION),
                    status_value,
                )
                done_value = (~finite) | stagnated | converged(new_total, new_error)
                return (
                    lower_values,
                    upper_values,
                    estimate_values,
                    error_values,
                    active_values,
                    count_value,
                    new_total,
                    new_error,
                    evaluation_value,
                    jnp.asarray(status_value, dtype=jnp.int32),
                    done_value,
                )

            return jax.lax.cond(
                capacity_exhausted | evaluation_exhausted,
                fail_capacity,
                split,
                current,
            )

        next_carry = jax.lax.cond(finished, lambda value: value, refine, carry)
        return next_carry, None

    state, _ = jax.lax.scan(iteration, state, xs=None, length=capacity)
    (
        lower_bounds,
        upper_bounds,
        estimates,
        errors,
        active,
        count,
        global_estimate,
        global_error,
        evaluation_count,
        status,
        done,
    ) = state
    status = jnp.where(
        done,
        status,
        int(IntegrationStatus.MAXIMUM_INTERVALS_REACHED),
    )
    if discovery is not None:
        status = jnp.where(
            status == int(IntegrationStatus.CONVERGED),
            discovery.status,
            status,
        )
    value_data = global_estimate
    if plan.throw:
        value_data = eqx.error_if(
            value_data,
            status != int(IntegrationStatus.CONVERGED),
            "Adaptive quadrature failed to meet its numerical contract.",
        )
    partition = None
    if plan.collect_partition:
        partition = AdaptivePartition(
            count=jnp.sum(active, dtype=jnp.int32),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            integral_estimates=estimates,
            estimated_errors=errors,
            active=active,
        )
    rule_name = type(plan.rule).__name__
    diagnostics = AdaptiveQuadratureDiagnostics(
        status=status,
        num_evaluations=evaluation_count,
        estimated_error=global_error,
        partition=partition,
        rule=rule_name,
        discovery=discovery,
        discovery_count=(
            0 if discovery is None else jnp.sum(discovery.active, dtype=jnp.int32)
        ),
        discovery_overflow=(
            False
            if discovery is None
            else discovery.status == int(IntegrationStatus.BREAKPOINT_CANDIDATE_OVERFLOW)
        ),
    )
    return IntegrationEstimate(
        value_data,
        status=status,
        num_evaluations=evaluation_count,
        error_estimate=global_error,
        error_kind="embedded-rule",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("adaptive-callable", "callable", rule_name),
    )


def _triangle_children(vertices: Array, /) -> Array:
    first, second, third = vertices
    first_second = 0.5 * (first + second)
    second_third = 0.5 * (second + third)
    third_first = 0.5 * (third + first)
    return jnp.stack(
        (
            jnp.stack((first, first_second, third_first)),
            jnp.stack((first_second, second, second_third)),
            jnp.stack((third_first, second_third, third)),
            jnp.stack((first_second, second_third, third_first)),
        )
    )


def _error_norm(value: Array, /) -> Array:
    return jnp.max(jnp.abs(jnp.asarray(value)))


def adaptive_triangle_callable(
    integrand: Callable[[Array], Array],
    triangles: Array,
    plan: AdaptiveTrianglePlan,
    /,
    *,
    precision: IntegrationPrecisionPolicy | None = None,
) -> IntegrationEstimate:
    """Integrate a batched callable over affine triangles with a bounded plan.

    ``integrand`` receives physical points with shape ``(n, dimension)`` and
    returns values with the same leading point axis.  The triangle partition,
    statuses, estimates, and precision policy are the shared integration
    substrate used by domain and specialized evaluators alike.
    """
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(plan, AdaptiveTrianglePlan):
        raise TypeError("plan must be an AdaptiveTrianglePlan.")
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy.")
    initial_triangles = precision_.accumulation(jnp.asarray(triangles))
    if initial_triangles.ndim != 3 or initial_triangles.shape[1] != 3:
        raise ValueError("Adaptive triangles must have shape (count, 3, dimension).")
    initial_count = int(initial_triangles.shape[0])
    if initial_count == 0:
        raise ValueError("Adaptive triangles must be nonempty.")
    if initial_count > plan.max_cells:
        raise ValueError("max_cells cannot hold every initial triangle.")
    ambient_dimension = int(initial_triangles.shape[-1])
    if ambient_dimension not in (2, 3):
        raise ValueError("Adaptive triangles require ambient dimension two or three.")
    low_data = plan.low_rule.materialize()
    high_data = plan.high_rule.materialize()

    def evaluate_points(points: Array) -> Array:
        values = precision_.evaluation(jnp.asarray(integrand(points)))
        if values.ndim == 0 or values.shape[0] != points.shape[0]:
            raise ValueError(
                "Adaptive callable integrands must preserve the leading point axis."
            )
        return values

    def physical_jacobian(vertices: Array) -> Array:
        first = vertices[1] - vertices[0]
        second = vertices[2] - vertices[0]
        if ambient_dimension == 2:
            return jnp.abs(jnp.linalg.det(jnp.stack((first, second), axis=-1)))
        return jnp.linalg.norm(jnp.cross(first, second))

    def evaluate_field(vertices: Array, rule_data) -> Array:
        origin = vertices[0]
        first = vertices[1] - origin
        second = vertices[2] - origin
        reference = precision_.accumulation(rule_data.points)
        physical = origin + reference[:, :1] * first + reference[:, 1:2] * second
        values = evaluate_points(physical)
        weights = precision_.accumulation(
            physical_jacobian(vertices) * precision_.accumulation(rule_data.weights)
        )
        return precision_.accumulation(jnp.tensordot(weights, values, axes=(0, 0)))

    initial_cost = initial_count * (
        int(low_data.weights.shape[0]) + int(high_data.weights.shape[0])
    )
    if plan.max_evaluations is not None and plan.max_evaluations < initial_cost:
        prototype = evaluate_points(jnp.mean(initial_triangles[0], axis=0, keepdims=True))
        value_data = jnp.zeros(prototype.shape[1:], dtype=prototype.dtype)
        status = jnp.asarray(
            int(IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED), dtype=jnp.int32
        )
        if plan.throw:
            value_data = eqx.error_if(
                value_data,
                True,
                "Adaptive triangle quadrature exceeded its initial evaluation budget.",
            )
        error = precision_.decision(jnp.asarray(jnp.inf))
        diagnostics = AdaptiveTriangleDiagnostics(
            status=status,
            num_evaluations=jnp.asarray(1, dtype=jnp.int32),
            estimated_error=error,
            partition=None,
            low_rule=plan.low_rule.rule_id,
            high_rule=plan.high_rule.rule_id,
        )
        return IntegrationEstimate(
            value_data,
            status=status,
            num_evaluations=1,
            error_estimate=error,
            error_kind="paired-reference-rule",
            diagnostics=diagnostics,
            provenance=IntegrationProvenance(
                "adaptive-triangle-callable", "callable", plan.high_rule.rule_id
            ),
        )

    first_high = evaluate_field(initial_triangles[0], high_data)
    if initial_count == 1:
        high_estimates = first_high[None, ...]
    else:
        remaining_high = jax.vmap(lambda vertices: evaluate_field(vertices, high_data))(
            initial_triangles[1:]
        )
        high_estimates = jnp.concatenate((first_high[None, ...], remaining_high), axis=0)
    low_estimates = jax.vmap(lambda vertices: evaluate_field(vertices, low_data))(
        initial_triangles
    )
    initial_errors = precision_.decision(
        jax.vmap(_error_norm)(high_estimates - low_estimates)
    )
    finite = (
        jnp.all(jnp.isfinite(high_estimates))
        & jnp.all(jnp.isfinite(low_estimates))
        & jnp.all(jnp.isfinite(initial_errors))
    )
    capacity = plan.max_cells
    output_shape = high_estimates.shape[1:]
    vertices = (
        jnp.zeros((capacity, 3, ambient_dimension), dtype=initial_triangles.dtype)
        .at[:initial_count]
        .set(initial_triangles)
    )
    estimates = (
        jnp.zeros((capacity,) + output_shape, dtype=high_estimates.dtype)
        .at[:initial_count]
        .set(high_estimates)
    )
    errors = (
        jnp.zeros((capacity,), dtype=initial_errors.dtype)
        .at[:initial_count]
        .set(initial_errors)
    )
    active = jnp.arange(capacity) < initial_count
    total = precision_.accumulation(jnp.sum(high_estimates, axis=0))
    total_error = precision_.decision(jnp.sum(initial_errors))
    decision_dtype = total_error.dtype
    absolute = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if plan.absolute_tolerance is None
        else jnp.asarray(plan.absolute_tolerance, dtype=decision_dtype)
    )
    relative = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if plan.relative_tolerance is None
        else jnp.asarray(plan.relative_tolerance, dtype=decision_dtype)
    )

    def converged(value: Array, error: Array) -> Array:
        magnitude = precision_.decision(_error_norm(value))
        return precision_.decision(error) <= absolute + relative * magnitude

    done = (~finite) | converged(total, total_error)
    status = jnp.where(
        finite,
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    ).astype(jnp.int32)
    state = (
        vertices,
        estimates,
        errors,
        active,
        jnp.asarray(initial_count, dtype=jnp.int32),
        total,
        total_error,
        jnp.asarray(initial_cost, dtype=jnp.int32),
        status,
        done,
    )
    child_cost = 4 * (int(low_data.weights.shape[0]) + int(high_data.weights.shape[0]))
    maximum_evaluations = (
        2**31 - 1 if plan.max_evaluations is None else plan.max_evaluations
    )
    iterations = max(0, (capacity - initial_count) // 3)

    def iteration(carry, _):
        (
            vertices_,
            estimates_,
            errors_,
            active_,
            count_,
            total_,
            total_error_,
            evaluations_,
            status_,
            finished_,
        ) = carry

        def advance(current):
            (
                vertices__,
                estimates__,
                errors__,
                active__,
                count__,
                total__,
                total_error__,
                evaluations__,
                status__,
                finished__,
            ) = current
            budget_ok = evaluations__ + child_cost <= maximum_evaluations

            def budget_failure(values):
                values = list(values)
                values[-2] = jnp.asarray(
                    int(IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED),
                    dtype=jnp.int32,
                )
                values[-1] = jnp.asarray(True)
                return tuple(values)

            def refine(values):
                (
                    vertices___,
                    estimates___,
                    errors___,
                    active___,
                    count___,
                    total___,
                    total_error___,
                    evaluations___,
                    status___,
                    finished___,
                ) = values
                selected = jnp.argmax(jnp.where(active___, errors___, -jnp.inf))
                children = _triangle_children(vertices___[selected])
                child_high = jax.vmap(lambda child: evaluate_field(child, high_data))(
                    children
                )
                child_low = jax.vmap(lambda child: evaluate_field(child, low_data))(
                    children
                )
                child_errors = precision_.decision(
                    jax.vmap(_error_norm)(child_high - child_low)
                )
                finite_children = (
                    jnp.all(jnp.isfinite(child_high))
                    & jnp.all(jnp.isfinite(child_low))
                    & jnp.all(jnp.isfinite(child_errors))
                )
                append = count___ + jnp.arange(3, dtype=jnp.int32)
                next_vertices = vertices___.at[selected].set(children[0])
                next_vertices = next_vertices.at[append].set(children[1:])
                next_estimates = estimates___.at[selected].set(child_high[0])
                next_estimates = next_estimates.at[append].set(child_high[1:])
                next_errors = errors___.at[selected].set(child_errors[0])
                next_errors = next_errors.at[append].set(child_errors[1:])
                next_active = active___.at[append].set(True)
                next_total = precision_.accumulation(
                    total___ - estimates___[selected] + jnp.sum(child_high, axis=0)
                )
                next_total_error = precision_.decision(
                    total_error___ - errors___[selected] + jnp.sum(child_errors)
                )
                next_count = count___ + 3
                next_evaluations = evaluations___ + child_cost
                next_converged = converged(next_total, next_total_error)
                next_done = (~finite_children) | next_converged
                next_status = jnp.where(
                    finite_children,
                    int(IntegrationStatus.CONVERGED),
                    int(IntegrationStatus.NONFINITE_INTEGRAND),
                ).astype(jnp.int32)
                return (
                    next_vertices,
                    next_estimates,
                    next_errors,
                    next_active,
                    next_count,
                    next_total,
                    next_total_error,
                    next_evaluations,
                    next_status,
                    next_done,
                )

            return jax.lax.cond(budget_ok, refine, budget_failure, current)

        next_carry = jax.lax.cond(finished_, lambda value: value, advance, carry)
        return next_carry, None

    state, _ = jax.lax.scan(iteration, state, xs=None, length=iterations)
    (
        vertices,
        estimates,
        errors,
        active,
        count,
        total,
        total_error,
        evaluations,
        status,
        done,
    ) = state
    status = jnp.where(
        done,
        status,
        int(IntegrationStatus.MAXIMUM_CELLS_REACHED),
    ).astype(jnp.int32)
    value_data = total
    if plan.throw:
        value_data = eqx.error_if(
            value_data,
            status != int(IntegrationStatus.CONVERGED),
            "Adaptive triangle quadrature failed to meet its numerical contract.",
        )
    partition = (
        AdaptiveTrianglePartition(
            count=count,
            vertices=vertices,
            integral_estimates=estimates,
            estimated_errors=errors,
            active=active,
        )
        if plan.collect_partition
        else None
    )
    diagnostics = AdaptiveTriangleDiagnostics(
        status=status,
        num_evaluations=evaluations,
        estimated_error=total_error,
        partition=partition,
        low_rule=plan.low_rule.rule_id,
        high_rule=plan.high_rule.rule_id,
    )
    return IntegrationEstimate(
        value_data,
        status=status,
        num_evaluations=evaluations,
        error_estimate=total_error,
        error_kind="paired-reference-rule",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "adaptive-triangle-callable", "callable", plan.high_rule.rule_id
        ),
    )
