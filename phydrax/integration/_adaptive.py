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

from phydrax.domain import (
    AbstractGeometry,
    AbstractScalarDomain,
    Boundary,
    ComponentSum,
    DomainComponent,
    DomainFunction,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
    Interval1d,
    PointBatch,
    SampleLayout,
    ScalarInterval,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from ._adaptive_callable import (
    _error_norm,
    _meets_plan_tolerance,
    adaptive_interval_callable,
)
from ._estimates import (
    AdaptiveQuadratureDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._lowering import component_factor_fields, sum_over
from ._plans import AdaptiveQuadraturePlan
from ._precision import IntegrationPrecisionPolicy
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget


def _unwrap(factor: Any, /) -> Any:
    return factor


def _fixed_field(factor: Any, selector: Any, /) -> cx.Field:
    factor = _unwrap(factor)
    if isinstance(factor, AbstractScalarDomain):
        if isinstance(selector, FixedStart):
            value = factor.fixed("start")
        elif isinstance(selector, FixedEnd):
            value = factor.fixed("end")
        elif isinstance(selector, Fixed):
            value = selector.value
        else:
            raise TypeError("Non-integrated scalar factors must be fixed.")
        return cx.Field(jnp.asarray(value, dtype=float).reshape(()), dims=())
    if isinstance(factor, AbstractGeometry) and isinstance(selector, Fixed):
        return cx.Field(
            jnp.asarray(selector.value, dtype=float).reshape((factor.spatial_dim,)),
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
    SampleLayout,
    frozendict[str, cx.Field],
]:
    labels = component.domain.labels
    if variable is None:
        free = tuple(
            label
            for label in labels
            if isinstance(component.spec.selection_for(label), Interior)
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
    if not isinstance(component.spec.selection_for(variable_), Interior):
        raise ValueError("The adaptive integration variable must select Interior().")
    factor = _unwrap(component.domain.factor(variable_))
    if not isinstance(factor, (ScalarInterval, Interval1d)):
        raise TypeError("Adaptive quadrature supports ScalarInterval and Interval1d.")
    fixed: dict[str, cx.Field] = {}
    fixed_labels: set[str] = set()
    for label in labels:
        if label == variable_:
            continue
        selector = component.spec.selection_for(label)
        if isinstance(selector, (Interior, Boundary)):
            raise ValueError("Every non-integrated adaptive label must be fixed.")
        fixed[label] = _fixed_field(component.domain.factor(label), selector)
        fixed_labels.add(label)
    structure = SampleLayout(((variable_,),)).canonicalize(
        labels, fixed_labels=frozenset(fixed_labels)
    )
    return variable_, factor, structure, frozendict(fixed)


class DomainAdaptiveIntegrand(StrictModule):
    """Array callback applying component filters and weights at one coordinate."""

    integrand: DomainFunction
    component: DomainComponent
    fixed_points: frozendict[str, cx.Field]
    structure: SampleLayout
    log_density: DomainFunction | None
    key: Key[Array, ""]
    kwargs: frozendict[str, Any]
    variable: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    geometry_variable: bool = eqx.field(static=True)
    precision: IntegrationPrecisionPolicy

    def field(self, coordinate: Array, /) -> cx.Field:
        coordinate_ = jnp.asarray(coordinate, dtype=float)
        if self.geometry_variable:
            variable = cx.Field(coordinate_.reshape((1, 1)), dims=(self.axis, None))
        else:
            variable = cx.Field(coordinate_.reshape((1,)), dims=(self.axis,))
        point_values = dict(self.fixed_points.items())
        point_values[self.variable] = variable
        points = PointBatch(
            frozendict(
                {label: point_values[label] for label in self.component.domain.labels}
            ),
            self.structure,
        )
        values = self.integrand(points, key=self.key, **self.kwargs)
        if not isinstance(values, cx.Field):
            raise TypeError("Adaptive integrands must evaluate to coordax.Field.")
        values = cx.Field(
            self.precision.evaluation(values.data),
            dims=values.dims,
        )
        mask, modifier = component_factor_fields(
            self.component,
            points,
            key=self.key,
            kwargs=dict(self.kwargs.items()),
        )
        weight = cx.Field(
            self.precision.accumulation((mask * modifier).data),
            dims=(mask * modifier).dims,
        )
        result = cx.Field(
            self.precision.accumulation((values * weight).data),
            dims=(values * weight).dims,
        )
        if self.log_density is not None:
            log_values = self.log_density(points, key=self.key, **self.kwargs)
            density = cx.Field(
                self.precision.accumulation(
                    jnp.exp(self.precision.evaluation(log_values.data))
                ),
                dims=log_values.dims,
            )
            result = cx.Field(
                self.precision.accumulation((result * density).data),
                dims=(result * density).dims,
            )
        if self.axis in result.named_dims:
            result = sum_over(
                result,
                self.axis,
                accumulation_dtype=self.precision.accumulation_dtype,
            )
        return result

    def __call__(self, coordinate: Array, /) -> Array:
        return jnp.asarray(self.field(coordinate).data)


def _as_domain_function(value: Any, component: DomainComponent, /) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    return DomainFunction(domain=component.domain, deps=(), func=value)


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
    precision: IntegrationPrecisionPolicy,
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
        precision=precision,
    )
    endpoints = precision.accumulation(jnp.asarray((factor.start, factor.end)))
    prototype = callback.field(0.5 * (endpoints[0] + endpoints[-1]))
    raw = adaptive_interval_callable(
        jax.vmap(callback),
        endpoints,
        plan,
        precision=precision,
    )
    return IntegrationEstimate(
        cx.Field(raw.value, dims=prototype.dims),
        status=raw.status,
        num_evaluations=raw.num_evaluations,
        error_estimate=raw.error_estimate,
        error_kind=raw.error_kind,
        diagnostics=raw.diagnostics,
        provenance=IntegrationProvenance(
            "adaptive", "component", type(plan.rule).__name__
        ),
        precision_evidence=raw.precision_evidence,
    )


def _combine_ratio(
    numerator: IntegrationEstimate,
    denominator: IntegrationEstimate,
    /,
    *,
    plan: AdaptiveQuadraturePlan,
    precision: IntegrationPrecisionPolicy,
) -> IntegrationEstimate:
    denominator_data = precision.accumulation(denominator.value.data)
    valid = jnp.all(jnp.isfinite(denominator_data)) & jnp.all(denominator_data != 0)
    status = jnp.maximum(numerator.status, denominator.status)
    status = jnp.where(
        valid,
        status,
        int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
    )
    value_data = precision.accumulation(
        precision.accumulation(numerator.value.data) / denominator_data
    )
    value = cx.Field(value_data, dims=numerator.value.dims)
    if numerator.error_estimate is None or denominator.error_estimate is None:
        raise RuntimeError("Adaptive ratio terms require embedded-rule errors.")
    tiny = jnp.finfo(jnp.real(value_data).dtype).tiny
    relative_numerator = precision.decision(
        numerator.error_estimate
        / jnp.maximum(jnp.abs(precision.accumulation(numerator.value.data)), tiny)
    )
    relative_denominator = precision.decision(
        denominator.error_estimate / jnp.maximum(jnp.abs(denominator_data), tiny)
    )
    error = precision.decision(
        _error_norm(jnp.abs(value_data) * (relative_numerator + relative_denominator))
    )
    ratio_converged = _meets_plan_tolerance(value_data, error, plan, precision)
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
    precision: IntegrationPrecisionPolicy | None = None,
) -> IntegrationEstimate:
    """Execute native adaptive interval integration for component/density targets."""
    callback_kwargs = {} if kwargs is None else kwargs
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy.")
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
    if isinstance(component_target.component, ComponentSum):
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
                precision=precision_,
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
            value = cx.Field(
                precision_.accumulation((value + estimate.value).data),
                dims=value.dims,
            )
            error = precision_.decision(error + estimate.error_estimate)
            status = jnp.maximum(status, estimate.status)
            evaluations = evaluations + estimate.num_evaluations
        combined_converged = _meets_plan_tolerance(
            jnp.asarray(value.data),
            error,
            plan,
            precision_,
        )
        status = jnp.where(
            (status == int(IntegrationStatus.CONVERGED)) & (~combined_converged),
            int(IntegrationStatus.REFINEMENT_STAGNATION),
            status,
        )
        if plan.throw:
            value_data = eqx.error_if(
                value.data,
                status != int(IntegrationStatus.CONVERGED),
                "Adaptive component-sum integration failed.",
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
                "adaptive", "component-sum", type(plan.rule).__name__
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
            precision=precision_,
        )
        return _combine_ratio(
            raw,
            denominator,
            plan=plan,
            precision=precision_,
        )
    numerator = _run_adaptive_raw(
        integrand,
        component_target.component,
        plan,
        variable=variable,
        log_density=log_density,
        key=key,
        kwargs=callback_kwargs,
        precision=precision_,
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
        precision=precision_,
    )
    return _combine_ratio(
        numerator,
        denominator,
        plan=plan,
        precision=precision_,
    )


__all__ = ["DomainAdaptiveIntegrand", "integrate_adaptive"]
