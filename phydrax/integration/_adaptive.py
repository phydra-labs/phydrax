#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._numerics import clenshaw_curtis_data, tanh_sinh_data
from .._strict import StrictModule
from ..domain._base import _AbstractGeometry
from ..domain._components import (
    Boundary,
    DomainComponent,
    DomainComponentUnion,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
)
from ..domain._domain import RelabeledDomain
from ..domain._function import DomainFunction
from ..domain._scalar import _AbstractScalarDomain, ScalarInterval
from ..domain._structure import PointsBatch, ProductStructure
from ..domain.geometry1d._primitives import Interval1d
from ._estimates import (
    AdaptivePartition,
    AdaptiveQuadratureDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._lowering import component_factor_fields, sum_over
from ._plans import AdaptiveQuadraturePlan
from ._rules import (
    ClenshawCurtisRule,
    GaussKronrodRule,
    interval_rule_data,
    TanhSinhRule,
)
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget


def _unwrap(factor: Any, /) -> Any:
    return factor.base if isinstance(factor, RelabeledDomain) else factor


def _fixed_field(factor: Any, selector: Any, /) -> cx.Field:
    factor = _unwrap(factor)
    if isinstance(factor, _AbstractScalarDomain):
        if isinstance(selector, FixedStart):
            value = factor.fixed("start")
        elif isinstance(selector, FixedEnd):
            value = factor.fixed("end")
        elif isinstance(selector, Fixed):
            value = selector.value
        else:
            raise TypeError("Non-integrated scalar factors must be fixed.")
        return cx.Field(jnp.asarray(value, dtype=float).reshape(()), dims=())
    if isinstance(factor, _AbstractGeometry) and isinstance(selector, Fixed):
        return cx.Field(
            jnp.asarray(selector.value, dtype=float).reshape((factor.var_dim,)),
            dims=(None,),
        )
    raise TypeError("Non-integrated adaptive factors must be fixed scalars or geometry.")


def _resolve_interval(
    component: DomainComponent,
    variable: str | None,
    /,
) -> tuple[
    str,
    ScalarInterval | Interval1d,
    ProductStructure,
    frozendict[str, cx.Field],
]:
    labels = component.domain.labels
    if variable is None:
        free = tuple(
            label
            for label in labels
            if isinstance(component.spec.component_for(label), Interior)
        )
        if len(free) != 1:
            raise ValueError(
                "Adaptive integration requires exactly one interior label; "
                "all remaining labels must be fixed."
            )
        variable_ = free[0]
    else:
        variable_ = str(variable)
        if variable_ not in labels:
            raise ValueError(f"Unknown integration variable {variable_!r}.")
    if not isinstance(component.spec.component_for(variable_), Interior):
        raise ValueError("The adaptive integration variable must select Interior().")
    factor = _unwrap(component.domain.factor(variable_))
    if not isinstance(factor, (ScalarInterval, Interval1d)):
        raise TypeError("Adaptive quadrature supports ScalarInterval and Interval1d.")
    fixed: dict[str, cx.Field] = {}
    fixed_labels: set[str] = set()
    for label in labels:
        if label == variable_:
            continue
        selector = component.spec.component_for(label)
        if isinstance(selector, (Interior, Boundary)):
            raise ValueError("Every non-integrated adaptive label must be fixed.")
        fixed[label] = _fixed_field(component.domain.factor(label), selector)
        fixed_labels.add(label)
    structure = ProductStructure(((variable_,),)).canonicalize(
        labels, fixed_labels=frozenset(fixed_labels)
    )
    return variable_, factor, structure, frozendict(fixed)


class DomainAdaptiveIntegrand(StrictModule):
    """Array callback applying component filters and weights at one coordinate."""

    integrand: DomainFunction
    component: DomainComponent
    fixed_points: frozendict[str, cx.Field]
    structure: ProductStructure
    log_density: DomainFunction | None
    key: Key[Array, ""]
    kwargs: frozendict[str, Any]
    variable: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    geometry_variable: bool = eqx.field(static=True)

    def field(self, coordinate: Array, /) -> cx.Field:
        coordinate_ = jnp.asarray(coordinate, dtype=float)
        if self.geometry_variable:
            variable = cx.Field(coordinate_.reshape((1, 1)), dims=(self.axis, None))
        else:
            variable = cx.Field(coordinate_.reshape((1,)), dims=(self.axis,))
        point_values = dict(self.fixed_points.items())
        point_values[self.variable] = variable
        points = PointsBatch(
            frozendict(
                {label: point_values[label] for label in self.component.domain.labels}
            ),
            self.structure,
        )
        values = self.integrand(points, key=self.key, **self.kwargs)
        if not isinstance(values, cx.Field):
            raise TypeError("Adaptive integrands must evaluate to coordax.Field.")
        mask, modifier = component_factor_fields(
            self.component,
            points,
            key=self.key,
            kwargs=dict(self.kwargs.items()),
        )
        result = values * mask * modifier
        if self.log_density is not None:
            log_values = self.log_density(points, key=self.key, **self.kwargs)
            result = result * cx.Field(
                jnp.exp(jnp.asarray(log_values.data)), dims=log_values.dims
            )
        if self.axis in result.named_dims:
            result = sum_over(result, self.axis)
        return result

    def __call__(self, coordinate: Array, /) -> Array:
        return jnp.asarray(self.field(coordinate).data)


def _as_domain_function(value: Any, component: DomainComponent, /) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    return DomainFunction(domain=component.domain, deps=(), func=value)


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
    /,
) -> Array:
    real_dtype = jnp.real(jnp.asarray(value)).dtype
    absolute = (
        jnp.sqrt(jnp.finfo(real_dtype).eps)
        if plan.absolute_tolerance is None
        else jnp.asarray(plan.absolute_tolerance, dtype=float)
    )
    relative = (
        jnp.sqrt(jnp.finfo(real_dtype).eps)
        if plan.relative_tolerance is None
        else jnp.asarray(plan.relative_tolerance, dtype=float)
    )
    return error <= absolute + relative * _error_norm(value)


def _run_adaptive_raw(
    integrand: Any,
    component: DomainComponent,
    plan: AdaptiveQuadraturePlan,
    /,
    *,
    variable: str | None,
    log_density: Any | None,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> IntegrationEstimate:
    variable_, factor, structure, fixed = _resolve_interval(component, variable)
    axis = structure.axis_for(variable_)
    if axis is None:
        raise RuntimeError("Adaptive variable has no integration axis.")
    function = _as_domain_function(integrand, component)
    density_function = (
        None if log_density is None else _as_domain_function(log_density, component)
    )
    callback = DomainAdaptiveIntegrand(
        integrand=function,
        component=component,
        fixed_points=fixed,
        structure=structure,
        log_density=density_function,
        key=key,
        kwargs=frozendict(kwargs),
        variable=variable_,
        axis=axis,
        geometry_variable=isinstance(factor, Interval1d),
    )
    endpoints = jnp.asarray((factor.start, *plan.breakpoints, factor.end), dtype=float)
    initial_count = len(plan.breakpoints) + 1
    prototype = callback.field(0.5 * (endpoints[0] + endpoints[-1]))
    output_shape = prototype.data.shape
    output_dims = prototype.dims
    high, low, local_cost = _local_rule(plan)
    initial_cost = initial_count * local_cost + 1
    if plan.max_evaluations is not None and plan.max_evaluations < initial_cost:
        bounds_valid = jnp.all(jnp.diff(endpoints) > 0.0)
        finite = jnp.all(jnp.isfinite(jnp.asarray(prototype.data)))
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
        value_data = jnp.zeros_like(jnp.asarray(prototype.data))
        if plan.throw:
            value_data = eqx.error_if(
                value_data,
                status != int(IntegrationStatus.CONVERGED),
                "Adaptive quadrature failed to meet its numerical contract.",
            )
        value = cx.Field(value_data, dims=output_dims)
        error = jnp.asarray(jnp.inf)
        diagnostics = AdaptiveQuadratureDiagnostics(
            status=status,
            num_evaluations=jnp.asarray(1, dtype=jnp.int32),
            estimated_error=error,
            partition=None,
            rule=type(plan.rule).__name__,
        )
        return IntegrationEstimate(
            value,
            status=status,
            num_evaluations=1,
            error_estimate=error,
            error_kind="embedded-rule",
            diagnostics=diagnostics,
            provenance=IntegrationProvenance(
                "adaptive", "component", type(plan.rule).__name__
            ),
        )

    def evaluate_interval(lower: Array, upper: Array) -> tuple[Array, Array, Array]:
        half = 0.5 * (upper - lower)
        center = 0.5 * (upper + lower)
        high_points = center + half * high.nodes
        high_values = jax.vmap(callback)(high_points)
        high_estimate = half * jnp.tensordot(high.weights, high_values, axes=(0, 0))
        if high.embedded_weights is not None:
            low_estimate = half * jnp.tensordot(
                high.embedded_weights, high_values, axes=(0, 0)
            )
        else:
            if low is None:
                raise RuntimeError("Nested adaptive rule is missing its low rule.")
            low_points = center + half * low.nodes
            low_values = jax.vmap(callback)(low_points)
            low_estimate = half * jnp.tensordot(low.weights, low_values, axes=(0, 0))
        error = _error_norm(high_estimate - low_estimate)
        finite = jnp.all(jnp.isfinite(high_values)) & jnp.all(jnp.isfinite(high_estimate))
        return high_estimate, error, finite

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
    errors = jnp.zeros((capacity,), dtype=float).at[:initial_count].set(initial_errors)
    active = jnp.arange(capacity) < initial_count
    global_estimate = jnp.sum(initial_estimates, axis=0)
    global_error = jnp.sum(initial_errors)
    evaluation_count = jnp.asarray(initial_count * local_cost + 1, dtype=jnp.int32)
    absolute = (
        jnp.sqrt(jnp.finfo(initial_estimates.dtype).eps)
        if plan.absolute_tolerance is None
        else jnp.asarray(plan.absolute_tolerance, dtype=float)
    )
    relative = (
        jnp.sqrt(jnp.finfo(initial_estimates.dtype).eps)
        if plan.relative_tolerance is None
        else jnp.asarray(plan.relative_tolerance, dtype=float)
    )

    def converged(estimate: Array, error: Array) -> Array:
        return error <= absolute + relative * _error_norm(estimate)

    bounds_valid = jnp.all(jnp.diff(endpoints) > 0.0)
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
        capacity * 2 * local_cost + initial_count * local_cost + 1
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
                new_total = (
                    total_value
                    - estimate_values[selected]
                    + left_estimate
                    + right_estimate
                )
                new_error = (
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
    value_data = global_estimate
    if plan.throw:
        value_data = eqx.error_if(
            value_data,
            status != int(IntegrationStatus.CONVERGED),
            "Adaptive quadrature failed to meet its numerical contract.",
        )
    value = cx.Field(value_data, dims=output_dims)
    partition = None
    if plan.collect_partition:
        partition = AdaptivePartition(
            count=count,
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
    )
    return IntegrationEstimate(
        value,
        status=status,
        num_evaluations=evaluation_count,
        error_estimate=global_error,
        error_kind="embedded-rule",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("adaptive", "component", rule_name),
    )


def _combine_ratio(
    numerator: IntegrationEstimate,
    denominator: IntegrationEstimate,
    /,
    *,
    plan: AdaptiveQuadraturePlan,
) -> IntegrationEstimate:
    denominator_data = jnp.asarray(denominator.value.data)
    valid = jnp.all(jnp.isfinite(denominator_data)) & jnp.all(denominator_data != 0)
    status = jnp.maximum(numerator.status, denominator.status)
    status = jnp.where(
        valid,
        status,
        int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
    )
    value_data = numerator.value.data / denominator_data
    value = cx.Field(value_data, dims=numerator.value.dims)
    if numerator.error_estimate is None or denominator.error_estimate is None:
        raise RuntimeError("Adaptive ratio terms require embedded-rule errors.")
    relative_numerator = numerator.error_estimate / jnp.maximum(
        jnp.abs(numerator.value.data), jnp.finfo(float).tiny
    )
    relative_denominator = denominator.error_estimate / jnp.maximum(
        jnp.abs(denominator_data), jnp.finfo(float).tiny
    )
    error = jnp.abs(value_data) * (relative_numerator + relative_denominator)
    error = _error_norm(error)
    ratio_converged = _meets_plan_tolerance(value_data, error, plan)
    status = jnp.where(
        (status == int(IntegrationStatus.CONVERGED)) & (~ratio_converged),
        int(IntegrationStatus.REFINEMENT_STAGNATION),
        status,
    )
    if plan.throw:
        value_data = eqx.error_if(
            value.data,
            status != int(IntegrationStatus.CONVERGED),
            "Normalized adaptive integration failed.",
        )
        value = cx.Field(value_data, dims=value.dims)
    diagnostics = AdaptiveQuadratureDiagnostics(
        status=status,
        num_evaluations=numerator.num_evaluations + denominator.num_evaluations,
        estimated_error=error,
        partition=None,
        rule=type(plan.rule).__name__,
    )
    return IntegrationEstimate(
        value,
        status=status,
        num_evaluations=numerator.num_evaluations + denominator.num_evaluations,
        error_estimate=error,
        error_kind="ratio-embedded-rule",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "adaptive-ratio", "component", type(plan.rule).__name__
        ),
    )


def integrate_adaptive(
    integrand: Any,
    target: ComponentTarget | DensityTarget,
    plan: AdaptiveQuadraturePlan,
    /,
    *,
    variable: str | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
) -> IntegrationEstimate:
    """Execute native adaptive interval integration for component/density targets."""
    callback_kwargs = {} if kwargs is None else kwargs
    if isinstance(target, DensityTarget):
        if not isinstance(target.base, ComponentTarget):
            raise TypeError("Adaptive density integration requires a component base.")
        component_target = target.base
        log_density = target.log_density
        normalized = target.normalized
        normalize_base = target.base.normalized and not target.normalized
    else:
        component_target = target
        log_density = None
        normalized = target.normalized
        normalize_base = False
    if isinstance(component_target.component, DomainComponentUnion):
        keys = jr.split(key, len(component_target.component.terms))
        estimates = tuple(
            _run_adaptive_raw(
                integrand,
                component,
                plan,
                variable=variable,
                log_density=log_density,
                key=term_key,
                kwargs=callback_kwargs,
            )
            for component, term_key in zip(
                component_target.component.terms, keys, strict=True
            )
        )
        value = estimates[0].value
        error = estimates[0].error_estimate
        if error is None:
            raise RuntimeError("Adaptive union terms require embedded-rule errors.")
        status = estimates[0].status
        evaluations = estimates[0].num_evaluations
        for estimate in estimates[1:]:
            if estimate.error_estimate is None:
                raise RuntimeError("Adaptive union terms require embedded-rule errors.")
            value = value + estimate.value
            error = error + estimate.error_estimate
            status = jnp.maximum(status, estimate.status)
            evaluations = evaluations + estimate.num_evaluations
        combined_converged = _meets_plan_tolerance(value.data, error, plan)
        status = jnp.where(
            (status == int(IntegrationStatus.CONVERGED)) & (~combined_converged),
            int(IntegrationStatus.REFINEMENT_STAGNATION),
            status,
        )
        if plan.throw:
            value_data = eqx.error_if(
                value.data,
                status != int(IntegrationStatus.CONVERGED),
                "Adaptive component-union integration failed.",
            )
            value = cx.Field(value_data, dims=value.dims)
        diagnostics = AdaptiveQuadratureDiagnostics(
            status=status,
            num_evaluations=evaluations,
            estimated_error=error,
            partition=None,
            rule=type(plan.rule).__name__,
        )
        raw = IntegrationEstimate(
            value,
            status=status,
            num_evaluations=evaluations,
            error_estimate=error,
            error_kind="embedded-rule",
            diagnostics=diagnostics,
            provenance=IntegrationProvenance(
                "adaptive", "component-union", type(plan.rule).__name__
            ),
        )
        if not normalized and not normalize_base:
            return raw
        base = ComponentTarget(
            component_target.component,
            axes=component_target.axes,
            normalized=False,
        )
        denominator_target = (
            DensityTarget(base, log_density, normalized=False)
            if normalized and log_density is not None
            else base
        )
        denominator = integrate_adaptive(
            1.0,
            denominator_target,
            plan,
            variable=variable,
            key=key,
            kwargs=callback_kwargs,
        )
        return _combine_ratio(raw, denominator, plan=plan)
    numerator = _run_adaptive_raw(
        integrand,
        component_target.component,
        plan,
        variable=variable,
        log_density=log_density,
        key=key,
        kwargs=callback_kwargs,
    )
    if not normalized and not normalize_base:
        return numerator
    denominator = _run_adaptive_raw(
        1.0,
        component_target.component,
        plan,
        variable=variable,
        log_density=log_density if normalized else None,
        key=key,
        kwargs=callback_kwargs,
    )
    return _combine_ratio(numerator, denominator, plan=plan)


__all__ = ["DomainAdaptiveIntegrand", "integrate_adaptive"]
