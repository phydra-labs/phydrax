#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
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
    PointBatch,
    SampleLayout,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from ..geometry import CubatureAtlasProvider
from ._adaptive_callable import (
    _error_norm,
    adaptive_triangle_callable,
)
from ._estimates import (
    AdaptiveTriangleDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._lowering import component_factor_fields, sum_over
from ._plans import AdaptiveTrianglePlan
from ._precision import IntegrationPrecisionPolicy
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget


class _TriangleIntegrand(StrictModule):
    integrand: DomainFunction
    component: DomainComponent
    fixed_points: frozendict[str, cx.Field]
    structure: SampleLayout
    log_density: DomainFunction | None
    key: Key[Array, ""]
    kwargs: frozendict[str, Any]
    label: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    precision: IntegrationPrecisionPolicy

    def field(self, coordinates: Array, /) -> cx.Field:
        point_values = dict(self.fixed_points.items())
        point_values[self.label] = cx.Field(coordinates, dims=(self.axis, None))
        points = PointBatch(
            frozendict(
                {label: point_values[label] for label in self.component.domain.labels}
            ),
            self.structure,
        )
        values = self.integrand(points, key=self.key, **self.kwargs)
        if not isinstance(values, cx.Field):
            raise TypeError("Adaptive triangle integrands must return coordax.Field.")
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
        weight = mask * modifier
        result = cx.Field(
            self.precision.accumulation((values * weight).data),
            dims=(values * weight).dims,
        )
        if self.log_density is not None:
            log_values = self.log_density(points, key=self.key, **self.kwargs)
            if not isinstance(log_values, cx.Field):
                raise TypeError(
                    "Adaptive triangle log_density must return coordax.Field."
                )
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
        return result

    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.asarray(self.field(coordinates).data)


def _as_domain_function(value: Any, component: DomainComponent, /) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    return DomainFunction(domain=component.domain, deps=(), func=value)


def _fixed_field(factor: Any, selector: Any, /) -> cx.Field:
    if isinstance(factor, AbstractScalarDomain):
        if isinstance(selector, FixedStart):
            value = factor.fixed("start")
        elif isinstance(selector, FixedEnd):
            value = factor.fixed("end")
        elif isinstance(selector, Fixed):
            value = selector.value
        else:
            raise TypeError("Non-integrated adaptive factors must be fixed.")
        return cx.Field(jnp.asarray(value, dtype=float).reshape(()), dims=())
    if isinstance(factor, AbstractGeometry) and isinstance(selector, Fixed):
        return cx.Field(
            jnp.asarray(selector.value, dtype=float).reshape((factor.spatial_dim,)),
            dims=(None,),
        )
    raise TypeError("Unsupported fixed adaptive triangle factor.")


def _resolve_triangles(
    component: DomainComponent,
    /,
) -> tuple[str, str, SampleLayout, frozendict[str, cx.Field], Array]:
    if isinstance(component, ComponentSum):
        raise TypeError("Adaptive triangle component sums are not supported.")
    varying = tuple(
        label
        for label in component.domain.labels
        if not isinstance(
            component.spec.selection_for(label), (Fixed, FixedStart, FixedEnd)
        )
    )
    if len(varying) != 1:
        raise ValueError("Adaptive triangle integration requires one varying geometry.")
    label = varying[0]
    selector = component.spec.selection_for(label)
    factor = component.domain.factor(label)
    if not isinstance(factor, CubatureAtlasProvider):
        raise TypeError("The adaptive geometry does not expose cubature charts.")
    if isinstance(selector, Boundary):
        component_kind = "boundary"
    elif isinstance(selector, Interior):
        component_kind = "interior"
    else:
        raise TypeError("Adaptive triangles require Interior() or Boundary().")
    atlas = factor.cubature_atlas(component_kind)
    if isinstance(selector, Boundary) and (
        selector.tags is not None or selector.entity_ids is not None
    ):
        atlas = atlas.select(tags=selector.tags, entity_ids=selector.entity_ids)
    if atlas.reference_domain != "triangle":
        raise ValueError("AdaptiveTrianglePlan requires triangle cubature charts.")
    reference_vertices = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    reference = jnp.broadcast_to(
        reference_vertices[None, ...],
        (atlas.num_charts, 3, 2),
    )
    chart_indices = jnp.broadcast_to(
        jnp.arange(atlas.num_charts, dtype=jnp.int32)[:, None],
        (atlas.num_charts, 3),
    )
    triangles = atlas.map(chart_indices, reference)
    fixed_labels = frozenset(other for other in component.domain.labels if other != label)
    structure = SampleLayout(((label,),)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    axis = structure.axis_for(label)
    if axis is None:
        raise RuntimeError("Adaptive triangle structure has no integration axis.")
    fixed = frozendict(
        {
            other: _fixed_field(
                component.domain.factor(other), component.spec.selection_for(other)
            )
            for other in fixed_labels
        }
    )
    return label, axis, structure, fixed, triangles


def _run_triangle_raw(
    integrand: Any,
    component: DomainComponent,
    plan: AdaptiveTrianglePlan,
    /,
    *,
    log_density: Any | None,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
    precision: IntegrationPrecisionPolicy,
) -> IntegrationEstimate:
    label, axis, structure, fixed, initial_triangles = _resolve_triangles(component)
    function = _as_domain_function(integrand, component)
    density_function = (
        None if log_density is None else _as_domain_function(log_density, component)
    )
    callback = _TriangleIntegrand(
        integrand=function,
        component=component,
        fixed_points=fixed,
        structure=structure,
        log_density=density_function,
        key=key,
        kwargs=frozendict(kwargs),
        label=label,
        axis=axis,
        precision=precision,
    )
    prototype = callback.field(jnp.mean(initial_triangles[0], axis=0, keepdims=True))
    reduced = sum_over(
        prototype,
        axis,
        accumulation_dtype=precision.accumulation_dtype,
    )
    raw = adaptive_triangle_callable(
        callback,
        initial_triangles,
        plan,
        precision=precision,
    )
    return IntegrationEstimate(
        cx.Field(raw.value, dims=reduced.dims),
        status=raw.status,
        num_evaluations=raw.num_evaluations,
        error_estimate=raw.error_estimate,
        error_kind=raw.error_kind,
        diagnostics=raw.diagnostics,
        provenance=IntegrationProvenance(
            "adaptive-triangle", "component", plan.high_rule.rule_id
        ),
        precision_evidence=raw.precision_evidence,
    )


def _ratio_estimate(
    numerator: IntegrationEstimate,
    denominator: IntegrationEstimate,
    plan: AdaptiveTrianglePlan,
    precision: IntegrationPrecisionPolicy,
    /,
) -> IntegrationEstimate:
    denominator_data = precision.accumulation(denominator.value.data)
    valid_mass = jnp.all(jnp.isfinite(denominator_data)) & jnp.all(
        denominator_data != 0.0
    )
    successful = numerator.successful & denominator.successful & valid_mass
    status = jnp.where(
        valid_mass,
        jnp.where(numerator.successful, denominator.status, numerator.status),
        int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
    ).astype(jnp.int32)
    value_data = precision.accumulation(
        precision.accumulation(numerator.value.data) / denominator_data
    )
    value = cx.Field(value_data, dims=numerator.value.dims)
    denominator_norm = precision.decision(
        jnp.maximum(
            _error_norm(denominator_data),
            jnp.finfo(jnp.real(denominator_data).dtype).tiny,
        )
    )
    error = precision.decision(
        numerator.error_estimate / denominator_norm
        + precision.decision(_error_norm(numerator.value.data))
        * denominator.error_estimate
        / denominator_norm**2
    )
    if plan.throw:
        value_data = eqx.error_if(
            value_data,
            ~successful,
            "Adaptive triangle normalization failed.",
        )
    diagnostics = AdaptiveTriangleDiagnostics(
        status=status,
        num_evaluations=numerator.num_evaluations + denominator.num_evaluations,
        estimated_error=error,
        partition=numerator.diagnostics.partition,
        low_rule=plan.low_rule.rule_id,
        high_rule=plan.high_rule.rule_id,
    )
    return IntegrationEstimate(
        cx.Field(value_data, dims=value.dims),
        status=status,
        num_evaluations=numerator.num_evaluations + denominator.num_evaluations,
        error_estimate=error,
        error_kind="ratio-paired-reference-rule",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "adaptive-triangle", "density-ratio", plan.high_rule.rule_id
        ),
    )


def integrate_adaptive_triangle(
    integrand: Any,
    target: ComponentTarget | DensityTarget,
    plan: AdaptiveTrianglePlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
    precision: IntegrationPrecisionPolicy | None = None,
) -> IntegrationEstimate:
    """Integrate over affine triangle charts with bounded adaptive refinement."""
    callback_kwargs = {} if kwargs is None else kwargs
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy.")
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget) or isinstance(base.component, ComponentSum):
        raise TypeError("AdaptiveTrianglePlan requires one component-based target.")
    if base.axes is not None:
        requested = (base.axes,) if isinstance(base.axes, str) else base.axes
        if len(requested) != 1:
            raise ValueError("Adaptive triangles must reduce the geometry label.")
    log_density = target.log_density if isinstance(target, DensityTarget) else None
    numerator = _run_triangle_raw(
        integrand,
        base.component,
        plan,
        log_density=log_density,
        key=key,
        kwargs=callback_kwargs,
        precision=precision_,
    )
    if isinstance(target, DensityTarget) and target.normalized:
        denominator = _run_triangle_raw(
            1.0,
            base.component,
            plan,
            log_density=target.log_density,
            key=key,
            kwargs=callback_kwargs,
            precision=precision_,
        )
        return _ratio_estimate(numerator, denominator, plan, precision_)
    if isinstance(target, DensityTarget) and base.normalized:
        denominator = _run_triangle_raw(
            1.0,
            base.component,
            plan,
            log_density=None,
            key=key,
            kwargs=callback_kwargs,
            precision=precision_,
        )
        return _ratio_estimate(numerator, denominator, plan, precision_)
    if isinstance(target, ComponentTarget) and target.normalized:
        denominator = _run_triangle_raw(
            1.0,
            base.component,
            plan,
            log_density=None,
            key=key,
            kwargs=callback_kwargs,
            precision=precision_,
        )
        return _ratio_estimate(numerator, denominator, plan, precision_)
    return numerator


__all__ = ["integrate_adaptive_triangle"]
