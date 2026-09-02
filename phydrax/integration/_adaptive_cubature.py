#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, Key

from phydrax.domain import (
    AbstractGeometry,
    AbstractScalarDomain,
    ComponentSum,
    DomainComponent,
    DomainFunction,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
    PointBatch,
    ProbabilityDomain,
    SampleLayout,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from ._adaptive_callable import _error_norm, _meets_plan_tolerance
from ._adaptive_triangle import integrate_adaptive_triangle
from ._estimates import (
    AdaptiveCubatureDiagnostics,
    AdaptiveCubaturePartition,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._lowering import component_factor_fields, sum_over
from ._plans import AdaptiveCubaturePlan, AdaptiveTrianglePlan
from ._precision import IntegrationPrecisionPolicy
from ._rules import CubatureRule
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget


class _ProductCubatureIntegrand(StrictModule):
    integrand: DomainFunction
    component: DomainComponent
    fixed_points: frozendict[str, cx.Field]
    varying: tuple[str, ...] = eqx.field(static=True)
    structure: SampleLayout
    axis: str = eqx.field(static=True)
    log_density: DomainFunction | None
    key: Key[Array, ""]
    kwargs: frozendict[str, Any]
    precision: IntegrationPrecisionPolicy

    def _physical(self, reference: Array, /) -> tuple[dict[str, cx.Field], Array]:
        points = dict(self.fixed_points.items())
        scale = jnp.asarray(1.0, dtype=reference.dtype)
        for position, label in enumerate(self.varying):
            factor = self.component.domain.factor(label)
            selector = self.component.spec.selection_for(label)
            if not isinstance(factor, AbstractScalarDomain) or not isinstance(
                selector, Interior
            ):
                raise TypeError(
                    "Adaptive hyperrectangle cubature requires scalar Interior factors."
                )
            coordinate = reference[:, position]
            if isinstance(factor, ProbabilityDomain):
                transport = factor.reference_transport
                if transport.reference_measure != "uniform":
                    raise ValueError(
                        f"Adaptive cubature probability axis {label!r} requires a "
                        "uniform reference transport."
                    )
                physical = transport.from_reference(coordinate)
                scale = scale * 0.5
            else:
                lower = factor.fixed("start")
                upper = factor.fixed("end")
                physical = 0.5 * (upper - lower) * coordinate + 0.5 * (upper + lower)
                scale = scale * 0.5 * (upper - lower)
            points[label] = cx.Field(jnp.asarray(physical), dims=(self.axis,))
        return points, scale

    def field(self, reference: Array, /) -> cx.Field:
        point_values, scale = self._physical(reference)
        points = PointBatch(
            frozendict(
                {label: point_values[label] for label in self.component.domain.labels}
            ),
            self.structure,
        )
        values = self.integrand(points, key=self.key, **self.kwargs)
        if not isinstance(values, cx.Field):
            raise TypeError("Adaptive cubature integrands must return coordax.Field.")
        values = cx.Field(self.precision.evaluation(values.data), dims=values.dims)
        mask, modifier = component_factor_fields(
            self.component,
            points,
            key=self.key,
            kwargs=dict(self.kwargs.items()),
        )
        weighted = values * mask * modifier
        result = cx.Field(
            self.precision.accumulation(weighted.data * scale),
            dims=weighted.dims,
        )
        if self.log_density is not None:
            log_values = self.log_density(points, key=self.key, **self.kwargs)
            if not isinstance(log_values, cx.Field):
                raise TypeError(
                    "Adaptive cubature log_density must return coordax.Field."
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

    def __call__(self, reference: Array, /) -> Array:
        return jnp.asarray(self.field(reference).data)


def _fixed_field(factor: Any, selector: Any, /) -> cx.Field:
    if not isinstance(factor, AbstractScalarDomain):
        raise TypeError("Adaptive product cubature fixed factors must be scalar.")
    if isinstance(selector, FixedStart):
        value = factor.fixed("start")
    elif isinstance(selector, FixedEnd):
        value = factor.fixed("end")
    elif isinstance(selector, Fixed):
        value = selector.value
    else:
        raise TypeError("Nonintegrated adaptive cubature factors must be fixed.")
    return cx.Field(jnp.asarray(value, dtype=float).reshape(()), dims=())


def _as_domain_function(value: Any, component: DomainComponent, /) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    return DomainFunction(domain=component.domain, deps=(), func=value)


def _tensor_rule(rule, dimension: int, dtype) -> tuple[Array, Array]:
    data = rule.data()
    nodes = jnp.asarray(data.nodes, dtype=dtype)
    weights = jnp.asarray(data.weights, dtype=dtype)
    mesh = jnp.meshgrid(*(nodes for _ in range(dimension)), indexing="ij")
    points = jnp.stack(tuple(axis.reshape((-1,)) for axis in mesh), axis=-1)
    return points, weights


def _contract_tensor(weights: Array, values: Array, dimension: int, order: int) -> Array:
    tensor = values.reshape((order,) * dimension + values.shape[1:])
    for _ in range(dimension):
        tensor = oe.contract("i,i...->...", weights, tensor)
    return tensor


def adaptive_cubature_callable(
    integrand,
    plan: AdaptiveCubaturePlan,
    /,
    *,
    precision: IntegrationPrecisionPolicy,
) -> IntegrationEstimate:
    """Adapt a coupled callable on ``[-1,1]^dimension`` with bounded cells."""
    prototype = precision.evaluation(
        jnp.asarray(integrand(jnp.zeros((1, plan.dimension), dtype=float)))
    )
    if prototype.ndim == 0 or prototype.shape[0] != 1:
        raise ValueError("Adaptive cubature callbacks must preserve the point axis.")
    output_shape = prototype.shape[1:]
    dtype = jnp.real(prototype).dtype
    high_points, high_weights = _tensor_rule(plan.high_rule, plan.dimension, dtype)
    low_points, low_weights = _tensor_rule(plan.low_rule, plan.dimension, dtype)
    high_cost = int(high_points.shape[0])
    low_cost = int(low_points.shape[0])
    local_cost = high_cost + low_cost

    def evaluate_cell(lower: Array, upper: Array):
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        volume = jnp.prod(half)
        high_coordinates = center + high_points * half
        low_coordinates = center + low_points * half
        high_values = precision.accumulation(jnp.asarray(integrand(high_coordinates)))
        low_values = precision.accumulation(jnp.asarray(integrand(low_coordinates)))
        high_estimate = precision.accumulation(
            volume
            * _contract_tensor(
                precision.accumulation(high_weights),
                high_values,
                plan.dimension,
                plan.high_rule.order,
            )
        )
        low_estimate = precision.accumulation(
            volume
            * _contract_tensor(
                precision.accumulation(low_weights),
                low_values,
                plan.dimension,
                plan.low_rule.order,
            )
        )
        error = precision.decision(_error_norm(high_estimate - low_estimate))
        finite = (
            jnp.all(jnp.isfinite(high_values))
            & jnp.all(jnp.isfinite(low_values))
            & jnp.all(jnp.isfinite(high_estimate))
            & jnp.all(jnp.isfinite(low_estimate))
            & jnp.isfinite(error)
        )
        return high_estimate, error, finite

    initial_lower = -jnp.ones((plan.dimension,), dtype=dtype)
    initial_upper = jnp.ones((plan.dimension,), dtype=dtype)
    initial_estimate, initial_error, initial_finite = evaluate_cell(
        initial_lower, initial_upper
    )
    capacity = plan.max_cells
    lowers = jnp.zeros((capacity, plan.dimension), dtype=dtype).at[0].set(initial_lower)
    uppers = jnp.zeros((capacity, plan.dimension), dtype=dtype).at[0].set(initial_upper)
    estimates = (
        jnp.zeros((capacity,) + output_shape, dtype=initial_estimate.dtype)
        .at[0]
        .set(initial_estimate)
    )
    errors = jnp.zeros((capacity,), dtype=initial_error.dtype).at[0].set(initial_error)
    active = jnp.arange(capacity) == 0
    status = jnp.where(
        initial_finite,
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    ).astype(jnp.int32)
    done = (~initial_finite) | _meets_plan_tolerance(
        initial_estimate, initial_error, plan, precision
    )
    evaluation_count = jnp.asarray(local_cost + 1, dtype=jnp.int32)
    max_evaluations = (
        capacity * 2 * local_cost + local_cost + 1
        if plan.max_evaluations is None
        else plan.max_evaluations
    )
    state = (
        lowers,
        uppers,
        estimates,
        errors,
        active,
        jnp.asarray(1, dtype=jnp.int32),
        initial_estimate,
        initial_error,
        evaluation_count,
        status,
        done,
    )

    def iteration(carry, _):
        def refine(current):
            (
                lower_all,
                upper_all,
                estimate_all,
                error_all,
                active_all,
                count,
                global_estimate,
                global_error,
                evaluations,
                current_status,
                current_done,
            ) = current
            del current_status, current_done
            capacity_exhausted = count >= capacity
            evaluation_exhausted = evaluations + 2 * local_cost > max_evaluations

            def fail(_):
                failure = jnp.where(
                    capacity_exhausted,
                    int(IntegrationStatus.MAXIMUM_CELLS_REACHED),
                    int(IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED),
                )
                return (
                    lower_all,
                    upper_all,
                    estimate_all,
                    error_all,
                    active_all,
                    count,
                    global_estimate,
                    global_error,
                    evaluations,
                    failure.astype(jnp.int32),
                    jnp.asarray(True),
                )

            def split(_):
                index = jnp.argmax(jnp.where(active_all, error_all, -jnp.inf))
                lower = lower_all[index]
                upper = upper_all[index]
                widths = (upper - lower) / jnp.asarray(plan.anisotropy, dtype=dtype)
                axis = jnp.argmax(widths)
                midpoint = 0.5 * (lower[axis] + upper[axis])
                left_upper = upper.at[axis].set(midpoint)
                right_lower = lower.at[axis].set(midpoint)
                child_estimates, child_errors, child_finite = jax.vmap(evaluate_cell)(
                    jnp.stack((lower, right_lower)),
                    jnp.stack((left_upper, upper)),
                )
                next_lower = lower_all.at[index].set(lower).at[count].set(right_lower)
                next_upper = upper_all.at[index].set(left_upper).at[count].set(upper)
                next_estimates = (
                    estimate_all.at[index]
                    .set(child_estimates[0])
                    .at[count]
                    .set(child_estimates[1])
                )
                next_errors = (
                    error_all.at[index]
                    .set(child_errors[0])
                    .at[count]
                    .set(child_errors[1])
                )
                next_active = active_all.at[count].set(True)
                estimate = (
                    global_estimate
                    - estimate_all[index]
                    + jnp.sum(child_estimates, axis=0)
                )
                error = global_error - error_all[index] + jnp.sum(child_errors)
                finite = jnp.all(child_finite)
                stagnated = (midpoint == lower[axis]) | (midpoint == upper[axis])
                converged = _meets_plan_tolerance(estimate, error, plan, precision)
                terminal = (~finite) | stagnated | converged
                next_status = jnp.where(
                    ~finite,
                    int(IntegrationStatus.NONFINITE_INTEGRAND),
                    jnp.where(
                        stagnated,
                        int(IntegrationStatus.REFINEMENT_STAGNATION),
                        int(IntegrationStatus.CONVERGED),
                    ),
                ).astype(jnp.int32)
                return (
                    next_lower,
                    next_upper,
                    next_estimates,
                    next_errors,
                    next_active,
                    count + 1,
                    estimate,
                    error,
                    evaluations + 2 * local_cost,
                    next_status,
                    terminal,
                )

            return jax.lax.cond(
                capacity_exhausted | evaluation_exhausted, fail, split, None
            )

        return jax.lax.cond(carry[-1], lambda value: value, refine, carry), None

    state, _ = jax.lax.scan(iteration, state, xs=None, length=capacity)
    (
        lowers,
        uppers,
        estimates,
        errors,
        active,
        count,
        value,
        error,
        evaluations,
        status,
        done,
    ) = state
    status = jnp.where(done, status, int(IntegrationStatus.MAXIMUM_CELLS_REACHED)).astype(
        jnp.int32
    )
    if plan.throw:
        value = eqx.error_if(
            value,
            status != int(IntegrationStatus.CONVERGED),
            "Adaptive cubature failed to meet its numerical contract.",
        )
    partition = None
    if plan.collect_partition:
        partition = AdaptiveCubaturePartition(
            count=count,
            lower_bounds=lowers,
            upper_bounds=uppers,
            integral_estimates=estimates,
            estimated_errors=errors,
            active=active,
        )
    diagnostics = AdaptiveCubatureDiagnostics(
        status=status,
        num_evaluations=evaluations,
        estimated_error=error,
        partition=partition,
        dimension=plan.dimension,
        low_rule=f"GaussLegendre({plan.low_rule.order})",
        high_rule=f"GaussLegendre({plan.high_rule.order})",
    )
    return IntegrationEstimate(
        value,
        status=status,
        num_evaluations=evaluations,
        error_estimate=error,
        error_kind="paired-tensor-rule",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "adaptive-cubature", "callable", f"dimension-{plan.dimension}"
        ),
    )


def _run_product(
    integrand: Any,
    component: DomainComponent,
    plan: AdaptiveCubaturePlan,
    /,
    *,
    log_density: Any | None,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
    precision: IntegrationPrecisionPolicy,
) -> IntegrationEstimate:
    varying = tuple(
        label
        for label in component.domain.labels
        if not isinstance(
            component.spec.selection_for(label), (Fixed, FixedStart, FixedEnd)
        )
    )
    if len(varying) != plan.dimension:
        raise ValueError(
            "AdaptiveCubaturePlan dimension must equal the number of varying labels."
        )
    if any(
        not isinstance(component.domain.factor(label), AbstractScalarDomain)
        for label in varying
    ):
        raise TypeError("AdaptiveCubaturePlan hyperrectangles require scalar factors.")
    fixed_labels = frozenset(
        label for label in component.domain.labels if label not in varying
    )
    structure = SampleLayout((varying,)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    axis = structure.axis_for(varying[0])
    if axis is None:
        raise RuntimeError("Adaptive cubature structure has no integration axis.")
    fixed = frozendict(
        {
            label: _fixed_field(
                component.domain.factor(label), component.spec.selection_for(label)
            )
            for label in fixed_labels
        }
    )
    callback = _ProductCubatureIntegrand(
        integrand=_as_domain_function(integrand, component),
        component=component,
        fixed_points=fixed,
        varying=varying,
        structure=structure,
        axis=axis,
        log_density=(
            None if log_density is None else _as_domain_function(log_density, component)
        ),
        key=key,
        kwargs=frozendict(kwargs),
        precision=precision,
    )
    prototype = callback.field(jnp.zeros((1, plan.dimension)))
    reduced = sum_over(
        prototype,
        axis,
        accumulation_dtype=precision.accumulation_dtype,
    )
    raw = adaptive_cubature_callable(callback, plan, precision=precision)
    return eqx.tree_at(
        lambda estimate: estimate.value,
        raw,
        cx.Field(raw.value, dims=reduced.dims),
    )


def _ratio(
    numerator: IntegrationEstimate,
    denominator: IntegrationEstimate,
    plan: AdaptiveCubaturePlan,
    /,
) -> IntegrationEstimate:
    mass = denominator.value.data
    valid = denominator.successful & jnp.all(jnp.isfinite(mass)) & jnp.all(mass != 0)
    status = jnp.where(
        numerator.successful & valid,
        int(IntegrationStatus.CONVERGED),
        jnp.where(
            valid,
            numerator.status,
            int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
        ),
    ).astype(jnp.int32)
    value = numerator.value.data / mass
    mass_norm = jnp.maximum(_error_norm(mass), jnp.finfo(jnp.real(mass).dtype).tiny)
    error = (
        numerator.error_estimate / mass_norm
        + _error_norm(numerator.value.data) * denominator.error_estimate / mass_norm**2
    )
    if plan.throw:
        value = eqx.error_if(
            value,
            status != int(IntegrationStatus.CONVERGED),
            "Adaptive cubature normalization failed.",
        )
    return IntegrationEstimate(
        cx.Field(value, dims=numerator.value.dims),
        status=status,
        num_evaluations=numerator.num_evaluations + denominator.num_evaluations,
        error_estimate=error,
        error_kind="ratio-paired-tensor-rule",
        diagnostics=numerator.diagnostics,
        provenance=IntegrationProvenance("adaptive-cubature", "density-ratio"),
    )


def integrate_adaptive_cubature(
    integrand: Any,
    target: ComponentTarget | DensityTarget,
    plan: AdaptiveCubaturePlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
    precision: IntegrationPrecisionPolicy | None = None,
) -> IntegrationEstimate:
    """Integrate a coupled finite-dimensional declared product."""
    if not isinstance(plan, AdaptiveCubaturePlan):
        raise TypeError("plan must be an AdaptiveCubaturePlan.")
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    callback_kwargs = {} if kwargs is None else kwargs
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget) or isinstance(base.component, ComponentSum):
        raise TypeError("AdaptiveCubaturePlan requires one finite component target.")
    varying = tuple(
        label
        for label in base.component.domain.labels
        if not isinstance(
            base.component.spec.selection_for(label), (Fixed, FixedStart, FixedEnd)
        )
    )
    if len(varying) == 1 and isinstance(
        base.component.domain.factor(varying[0]), AbstractGeometry
    ):
        triangle_plan = AdaptiveTrianglePlan(
            CubatureRule("triangle", 5),
            CubatureRule("triangle", 10),
            absolute_tolerance=plan.absolute_tolerance,
            relative_tolerance=plan.relative_tolerance,
            max_cells=plan.max_cells,
            max_evaluations=plan.max_evaluations,
            collect_partition=plan.collect_partition,
            throw=plan.throw,
        )
        return integrate_adaptive_triangle(
            integrand,
            target,
            triangle_plan,
            key=key,
            kwargs=callback_kwargs,
            precision=precision_,
        )
    log_density = target.log_density if isinstance(target, DensityTarget) else None
    numerator = _run_product(
        integrand,
        base.component,
        plan,
        log_density=log_density,
        key=key,
        kwargs=callback_kwargs,
        precision=precision_,
    )
    if isinstance(target, DensityTarget) and target.normalized:
        denominator = _run_product(
            1.0,
            base.component,
            plan,
            log_density=target.log_density,
            key=key,
            kwargs=callback_kwargs,
            precision=precision_,
        )
        return _ratio(numerator, denominator, plan)
    if isinstance(target, DensityTarget) and base.normalized:
        denominator = _run_product(
            1.0,
            base.component,
            plan,
            log_density=None,
            key=key,
            kwargs=callback_kwargs,
            precision=precision_,
        )
        return _ratio(numerator, denominator, plan)
    return numerator


__all__ = ["adaptive_cubature_callable", "integrate_adaptive_cubature"]
