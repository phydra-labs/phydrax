#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax
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
from ._estimates import (
    AdaptiveTriangleDiagnostics,
    AdaptiveTrianglePartition,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._lowering import component_factor_fields, sum_over
from ._plans import AdaptiveTrianglePlan
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
        mask, modifier = component_factor_fields(
            self.component,
            points,
            key=self.key,
            kwargs=dict(self.kwargs.items()),
        )
        result = values * mask * modifier
        if self.log_density is not None:
            log_values = self.log_density(points, key=self.key, **self.kwargs)
            if not isinstance(log_values, cx.Field):
                raise TypeError(
                    "Adaptive triangle log_density must return coordax.Field."
                )
            result = result * cx.Field(
                jnp.exp(jnp.asarray(log_values.data)), dims=log_values.dims
            )
        return result


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


def _run_triangle_raw(
    integrand: Any,
    component: DomainComponent,
    plan: AdaptiveTrianglePlan,
    /,
    *,
    log_density: Any | None,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> IntegrationEstimate:
    label, axis, structure, fixed, initial_triangles = _resolve_triangles(component)
    initial_count = int(initial_triangles.shape[0])
    if initial_count > plan.max_cells:
        raise ValueError("max_cells cannot hold every initial triangle chart.")
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
    )
    low_data = plan.low_rule.materialize()
    high_data = plan.high_rule.materialize()
    ambient_dimension = int(initial_triangles.shape[-1])
    if ambient_dimension not in (2, 3):
        raise ValueError("Adaptive triangles require ambient dimension two or three.")

    def physical_jacobian(vertices: Array) -> Array:
        first = vertices[1] - vertices[0]
        second = vertices[2] - vertices[0]
        if ambient_dimension == 2:
            return jnp.abs(jnp.linalg.det(jnp.stack((first, second), axis=-1)))
        return jnp.linalg.norm(jnp.cross(first, second))

    def evaluate_field(vertices: Array, rule_data) -> cx.Field:
        origin = vertices[0]
        first = vertices[1] - origin
        second = vertices[2] - origin
        reference = rule_data.points
        physical = origin + reference[:, :1] * first + reference[:, 1:2] * second
        values = callback.field(physical)
        weights = cx.Field(
            physical_jacobian(vertices) * rule_data.weights,
            dims=(axis,),
        )
        weighted = values * weights
        return sum_over(weighted, axis)

    initial_cost = initial_count * (
        int(low_data.weights.shape[0]) + int(high_data.weights.shape[0])
    )
    if plan.max_evaluations is not None and plan.max_evaluations < initial_cost:
        centroid = jnp.mean(initial_triangles[0], axis=0, keepdims=True)
        prototype = callback.field(centroid) * cx.Field(jnp.ones((1,)), dims=(axis,))
        reduced = sum_over(prototype, axis)
        value_data = jnp.zeros_like(jnp.asarray(reduced.data))
        status = jnp.asarray(
            int(IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED), dtype=jnp.int32
        )
        if plan.throw:
            value_data = eqx.error_if(
                value_data,
                True,
                "Adaptive triangle quadrature exceeded its initial evaluation budget.",
            )
        diagnostics = AdaptiveTriangleDiagnostics(
            status=status,
            num_evaluations=jnp.asarray(1, dtype=jnp.int32),
            estimated_error=jnp.asarray(jnp.inf),
            partition=None,
            low_rule=plan.low_rule.rule_id,
            high_rule=plan.high_rule.rule_id,
        )
        return IntegrationEstimate(
            cx.Field(value_data, dims=reduced.dims),
            status=status,
            num_evaluations=1,
            error_estimate=jnp.asarray(jnp.inf),
            error_kind="paired-reference-rule",
            diagnostics=diagnostics,
            provenance=IntegrationProvenance(
                "adaptive-triangle", "component", plan.high_rule.rule_id
            ),
        )

    first_high = evaluate_field(initial_triangles[0], high_data)
    if initial_count == 1:
        high_estimates = jnp.asarray(first_high.data)[None, ...]
    else:
        remaining_high = jax.vmap(
            lambda vertices: jnp.asarray(evaluate_field(vertices, high_data).data)
        )(initial_triangles[1:])
        high_estimates = jnp.concatenate(
            (jnp.asarray(first_high.data)[None, ...], remaining_high), axis=0
        )
    low_estimates = jax.vmap(
        lambda vertices: jnp.asarray(evaluate_field(vertices, low_data).data)
    )(initial_triangles)
    initial_errors = jax.vmap(_error_norm)(high_estimates - low_estimates)
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
    total = jnp.sum(high_estimates, axis=0)
    total_error = jnp.sum(initial_errors)
    real_dtype = jnp.real(total).dtype
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

    def converged(value: Array, error: Array) -> Array:
        return error <= absolute + relative * _error_norm(value)

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
                child_high = jax.vmap(
                    lambda child: jnp.asarray(evaluate_field(child, high_data).data)
                )(children)
                child_low = jax.vmap(
                    lambda child: jnp.asarray(evaluate_field(child, low_data).data)
                )(children)
                child_errors = jax.vmap(_error_norm)(child_high - child_low)
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
                next_total = (
                    total___ - estimates___[selected] + jnp.sum(child_high, axis=0)
                )
                next_total_error = (
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
        cx.Field(value_data, dims=first_high.dims),
        status=status,
        num_evaluations=evaluations,
        error_estimate=total_error,
        error_kind="paired-reference-rule",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "adaptive-triangle", "component", plan.high_rule.rule_id
        ),
    )


def _ratio_estimate(
    numerator: IntegrationEstimate,
    denominator: IntegrationEstimate,
    plan: AdaptiveTrianglePlan,
    /,
) -> IntegrationEstimate:
    denominator_data = jnp.asarray(denominator.value.data)
    valid_mass = jnp.all(jnp.isfinite(denominator_data)) & jnp.all(
        denominator_data != 0.0
    )
    successful = numerator.successful & denominator.successful & valid_mass
    status = jnp.where(
        valid_mass,
        jnp.where(numerator.successful, denominator.status, numerator.status),
        int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
    ).astype(jnp.int32)
    value = numerator.value / denominator.value
    denominator_norm = jnp.maximum(
        _error_norm(denominator_data), jnp.finfo(jnp.real(denominator_data).dtype).tiny
    )
    error = (
        numerator.error_estimate / denominator_norm
        + _error_norm(numerator.value.data)
        * denominator.error_estimate
        / denominator_norm**2
    )
    value_data = jnp.asarray(value.data)
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
) -> IntegrationEstimate:
    """Integrate over affine triangle charts with bounded adaptive refinement."""
    callback_kwargs = {} if kwargs is None else kwargs
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
    )
    if isinstance(target, DensityTarget) and target.normalized:
        denominator = _run_triangle_raw(
            1.0,
            base.component,
            plan,
            log_density=target.log_density,
            key=key,
            kwargs=callback_kwargs,
        )
        return _ratio_estimate(numerator, denominator, plan)
    if isinstance(target, DensityTarget) and base.normalized:
        denominator = _run_triangle_raw(
            1.0,
            base.component,
            plan,
            log_density=None,
            key=key,
            kwargs=callback_kwargs,
        )
        return _ratio_estimate(numerator, denominator, plan)
    if isinstance(target, ComponentTarget) and target.normalized:
        denominator = _run_triangle_raw(
            1.0,
            target.component,
            plan,
            log_density=None,
            key=key,
            kwargs=callback_kwargs,
        )
        return _ratio_estimate(numerator, denominator, plan)
    return numerator


__all__ = ["integrate_adaptive_triangle"]
