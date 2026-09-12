#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

import phydrax.axes as cx
import phydrax.ein as ein
from phydrax.domain import (
    AbstractGeometry,
    AbstractScalarDomain,
    ComponentSum,
    DomainComponent,
    DomainFunction,
    Fixed,
    FixedEnd,
    FixedStart,
    HyperRectangle,
    Interior,
    PointBatch,
    ProbabilityDomain,
    SampleLayout,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._numerics._compensated import compensated_sum, two_sum
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
    fixed_points: frozendict[str, cx.AxisArray]
    varying: tuple[str, ...] = eqx.field(static=True)
    structure: SampleLayout
    axis: str = eqx.field(static=True)
    log_density: DomainFunction | None
    key: Key[Array, ""]
    kwargs: frozendict[str, Any]
    precision: IntegrationPrecisionPolicy

    def _physical(self, reference: Array, /) -> tuple[dict[str, cx.AxisArray], Array]:
        points = dict(self.fixed_points.items())
        scale = jnp.asarray(1.0, dtype=reference.dtype)
        offset = 0
        for label in self.varying:
            factor = self.component.domain.factor(label)
            selector = self.component.spec.selection_for(label)
            if not isinstance(selector, Interior):
                raise TypeError(
                    "Adaptive hyperrectangle cubature requires Interior factors."
                )
            if isinstance(factor, AbstractScalarDomain):
                coordinate = reference[:, offset]
                offset += 1
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
                points[label] = cx.AxisArray(jnp.asarray(physical), dims=(self.axis,))
            elif isinstance(factor, HyperRectangle):
                width = factor.spatial_dim
                coordinate = reference[:, offset : offset + width]
                offset += width
                lower = jnp.asarray(factor.lower, dtype=reference.dtype)
                upper = jnp.asarray(factor.upper, dtype=reference.dtype)
                physical = 0.5 * (upper - lower) * coordinate + 0.5 * (upper + lower)
                scale = scale * jnp.prod(0.5 * (upper - lower))
                points[label] = cx.AxisArray(
                    jnp.asarray(physical), dims=(self.axis, None)
                )
            else:
                raise TypeError(
                    "Adaptive cubature varying factors must be scalar domains or "
                    "HyperRectangle values."
                )
        if offset != reference.shape[1]:
            raise RuntimeError("Adaptive cubature coordinate layout is inconsistent.")
        return points, scale

    def field(self, reference: Array, /) -> cx.AxisArray:
        point_values, scale = self._physical(reference)
        points = PointBatch(
            frozendict(
                {label: point_values[label] for label in self.component.domain.labels}
            ),
            self.structure,
        )
        values = self.integrand(points, key=self.key, **self.kwargs)
        if not isinstance(values, cx.AxisArray):
            raise TypeError(
                "Adaptive cubature integrands must return phydrax.axes.AxisArray."
            )
        values = cx.AxisArray(self.precision.evaluation(values.data), dims=values.dims)
        mask, modifier = component_factor_fields(
            self.component,
            points,
            key=self.key,
            kwargs=dict(self.kwargs.items()),
        )
        weighted = values * mask * modifier
        result = cx.AxisArray(
            self.precision.accumulation(weighted.data * scale),
            dims=weighted.dims,
        )
        if self.log_density is not None:
            log_values = self.log_density(points, key=self.key, **self.kwargs)
            if not isinstance(log_values, cx.AxisArray):
                raise TypeError(
                    "Adaptive cubature log_density must return phydrax.axes.AxisArray."
                )
            density = cx.AxisArray(
                self.precision.accumulation(
                    jnp.exp(self.precision.evaluation(log_values.data))
                ),
                dims=log_values.dims,
            )
            result = cx.AxisArray(
                self.precision.accumulation((result * density).data),
                dims=(result * density).dims,
            )
        return result

    def __call__(self, reference: Array, /) -> Array:
        return jnp.asarray(self.field(reference).data)


def _fixed_field(factor: Any, selector: Any, /) -> cx.AxisArray:
    if isinstance(factor, AbstractScalarDomain):
        if isinstance(selector, FixedStart):
            value = factor.fixed("start")
        elif isinstance(selector, FixedEnd):
            value = factor.fixed("end")
        elif isinstance(selector, Fixed):
            value = selector.value
        else:
            raise TypeError("Nonintegrated adaptive scalar factors must be fixed.")
        return cx.AxisArray(jnp.asarray(value, dtype=float).reshape(()), dims=())
    if isinstance(factor, AbstractGeometry) and isinstance(selector, Fixed):
        return cx.AxisArray(
            jnp.asarray(selector.value, dtype=float).reshape((factor.spatial_dim,)),
            dims=(None,),
        )
    raise TypeError("Nonintegrated adaptive factors must be fixed scalars or geometry.")


def _factor_dimension(factor: Any, /) -> int:
    if isinstance(factor, AbstractScalarDomain):
        return 1
    if isinstance(factor, HyperRectangle):
        return factor.spatial_dim
    raise TypeError(
        "Adaptive cubature varying factors must be scalar domains or HyperRectangle "
        "values."
    )


def _reference_breakpoints(
    component: DomainComponent,
    varying: tuple[str, ...],
    plan: AdaptiveCubaturePlan,
    /,
) -> tuple[Array, ...]:
    result: list[Array] = []
    position = 0
    for label in varying:
        factor = component.domain.factor(label)
        if isinstance(factor, AbstractScalarDomain):
            physical = jnp.asarray(plan.breakpoints[position], dtype=float)
            position += 1
            if isinstance(factor, ProbabilityDomain):
                transport = factor.reference_transport
                if transport.reference_measure != "uniform":
                    raise ValueError(
                        f"Adaptive cubature probability axis {label!r} requires a "
                        "uniform reference transport."
                    )
                result.append(jnp.asarray(transport.to_reference(physical)))
            else:
                lower = factor.fixed("start")
                upper = factor.fixed("end")
                result.append((2.0 * physical - upper - lower) / (upper - lower))
        elif isinstance(factor, HyperRectangle):
            lower = jnp.asarray(factor.lower)
            upper = jnp.asarray(factor.upper)
            for axis in range(factor.spatial_dim):
                physical = jnp.asarray(plan.breakpoints[position], dtype=float)
                position += 1
                result.append(
                    (2.0 * physical - upper[axis] - lower[axis])
                    / (upper[axis] - lower[axis])
                )
        else:
            raise TypeError(
                "Adaptive cubature varying factors must be scalar domains or "
                "HyperRectangle values."
            )
    if position != plan.dimension:
        raise RuntimeError("Adaptive cubature breakpoint layout is inconsistent.")
    return tuple(result)


def _as_domain_function(value: Any, component: DomainComponent, /) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    return DomainFunction(domain=component.domain, deps=(), func=value)


_ROUNDOFF_FACTOR = 50.0


_ERROR_INFLATION = 200.0
_MAX_CELL_ASPECT_RATIO = 16.0


def _canonical_initial_cells(
    breakpoints: tuple[Any, ...], dtype, /
) -> tuple[Array, Array, Array]:
    edges: list[Array] = []
    valid = jnp.asarray(True)
    for axis_breakpoints in breakpoints:
        points = jnp.asarray(axis_breakpoints, dtype=dtype).reshape((-1,))
        valid = (
            valid
            & jnp.all(jnp.isfinite(points))
            & jnp.all(points > -1.0)
            & jnp.all(points < 1.0)
            & jnp.all(jnp.diff(points) > 0.0)
        )
        edges.append(
            jnp.concatenate(
                (
                    jnp.asarray((-1.0,), dtype=dtype),
                    points,
                    jnp.asarray((1.0,), dtype=dtype),
                )
            )
        )
    indices = tuple(
        itertools.product(*(range(axis_edges.shape[0] - 1) for axis_edges in edges))
    )
    lower = jnp.stack(
        tuple(
            jnp.stack(tuple(edges[axis][index[axis]] for axis in range(len(edges))))
            for index in indices
        )
    )
    upper = jnp.stack(
        tuple(
            jnp.stack(tuple(edges[axis][index[axis] + 1] for axis in range(len(edges))))
            for index in indices
        )
    )
    return lower, upper, valid


def _adaptive_cubature_solve(
    integrand,
    plan: AdaptiveCubaturePlan,
    reference_breakpoints: tuple[Any, ...],
    /,
    *,
    precision: IntegrationPrecisionPolicy,
) -> IntegrationEstimate:
    reference_template = precision.evaluation(jnp.zeros((1, plan.dimension), dtype=float))
    prototype = eqx.filter_eval_shape(
        lambda points: precision.evaluation(jnp.asarray(integrand(points))),
        reference_template,
    )
    if len(prototype.shape) == 0 or prototype.shape[0] != 1:
        raise ValueError("Adaptive cubature callbacks must preserve the point axis.")
    output_shape = prototype.shape[1:]
    rule = plan.rule.prepared
    coordinate_dtype = reference_template.dtype
    value_dtype = precision.accumulation(jnp.zeros((), dtype=prototype.dtype)).dtype
    real_dtype = jnp.real(jnp.zeros((), dtype=value_dtype)).dtype
    points = jnp.asarray(rule.points, dtype=coordinate_dtype)
    coefficients = precision.accumulation(
        jnp.concatenate(
            (
                rule.weights[None, :],
                rule.embedded_weights[None, :],
                rule.split_weights,
            ),
            axis=0,
        )
    )
    absolute_weights = precision.accumulation(jnp.abs(rule.weights))
    local_cost = rule.num_points
    batch_size = (
        local_cost
        if plan.max_batch_points is None
        else min(local_cost, plan.max_batch_points)
    )
    num_batches = (local_cost + batch_size - 1) // batch_size
    padding = num_batches * batch_size - local_cost
    if padding:
        points = jnp.concatenate(
            (
                points,
                jnp.broadcast_to(points[-1], (padding, plan.dimension)),
            ),
            axis=0,
        )
        coefficients = jnp.pad(coefficients, ((0, 0), (0, padding)))
        absolute_weights = jnp.pad(absolute_weights, (0, padding))
    point_batches = points.reshape((num_batches, batch_size, plan.dimension))
    coefficient_batches = jnp.swapaxes(
        coefficients.reshape((2 + plan.dimension, num_batches, batch_size)),
        0,
        1,
    )
    absolute_weight_batches = absolute_weights.reshape((num_batches, batch_size))

    def evaluate_cell(lower: Array, upper: Array):
        center = precision.evaluation(0.5 * (lower + upper))
        half = precision.evaluation(0.5 * (upper - lower))
        volume = precision.accumulation(jnp.prod(half))
        moment_shape = (2 + plan.dimension,) + output_shape
        moment_zero = jnp.zeros(moment_shape, dtype=value_dtype)
        value_zero = jnp.zeros(output_shape, dtype=value_dtype)
        absolute_zero = jnp.zeros(output_shape, dtype=real_dtype)

        def evaluate_batch(carry, batch):
            (
                moment_high,
                moment_correction,
                absolute_high,
                absolute_correction,
                weighted_high,
                weighted_correction,
                square_high,
                square_correction,
                finite,
            ) = carry
            reference, batch_coefficients, batch_absolute_weights = batch
            coordinates = precision.evaluation(center + reference * half)
            values = precision.accumulation(jnp.asarray(integrand(coordinates)))
            contributions = ein.contract("kn,n...->k...", batch_coefficients, values)
            absolute_contribution = ein.contract(
                "n,n...->...",
                batch_absolute_weights,
                jnp.abs(values),
            )
            weighted_contribution = ein.contract(
                "n,n...->...",
                batch_absolute_weights,
                values,
            )
            square_contribution = ein.contract(
                "n,n...->...",
                batch_absolute_weights,
                jnp.abs(values) ** 2,
            )
            next_moment, moment_error = two_sum(moment_high, contributions)
            next_absolute, absolute_error = two_sum(absolute_high, absolute_contribution)
            next_weighted, weighted_error = two_sum(weighted_high, weighted_contribution)
            next_square, square_error = two_sum(square_high, square_contribution)
            return (
                next_moment,
                moment_correction + moment_error,
                next_absolute,
                absolute_correction + absolute_error,
                next_weighted,
                weighted_correction + weighted_error,
                next_square,
                square_correction + square_error,
                finite & jnp.all(jnp.isfinite(values)),
            ), None

        initial = (
            moment_zero,
            moment_zero,
            absolute_zero,
            absolute_zero,
            value_zero,
            value_zero,
            absolute_zero,
            absolute_zero,
            jnp.asarray(True),
        )
        accumulated, _ = jax.lax.scan(
            evaluate_batch,
            initial,
            (point_batches, coefficient_batches, absolute_weight_batches),
        )
        (
            moment_high,
            moment_correction,
            absolute_high,
            absolute_correction,
            weighted_high,
            weighted_correction,
            square_high,
            square_correction,
            finite,
        ) = accumulated
        moments = precision.accumulation(moment_high + moment_correction)
        absolute_moment = precision.accumulation(absolute_high + absolute_correction)
        weighted_moment = precision.accumulation(weighted_high + weighted_correction)
        square_moment = precision.accumulation(square_high + square_correction)
        estimate = precision.accumulation(volume * moments[0])
        embedded = precision.accumulation(volume * moments[1])
        split_values = precision.accumulation(volume * moments[2:])
        raw_error = precision.decision(_error_norm(estimate - embedded))
        absolute_scale = precision.decision(
            _error_norm(jnp.abs(volume) * absolute_moment)
        )
        mean = moments[0] / float(2**plan.dimension)
        centered_square = jnp.maximum(
            square_moment
            - 2.0 * jnp.real(jnp.conj(mean) * weighted_moment)
            + rule.weight_l1_norm * jnp.abs(mean) ** 2,
            0.0,
        )
        variation = precision.decision(
            _error_norm(jnp.abs(volume) * jnp.sqrt(rule.weight_l1_norm * centered_square))
        )
        exponent = (
            (rule.exact_degree + 1 + plan.dimension)
            / (rule.embedded_degree + 1 + plan.dimension)
            if rule.exact_degree is not None and rule.embedded_degree is not None
            else 1.5
        )
        scalable = (variation > 0.0) & (raw_error > 0.0)
        variation_safe = jnp.where(scalable, variation, 1.0)
        ratio = jnp.where(scalable, raw_error / variation_safe, 1.0)
        rescaled_error = jnp.where(
            scalable,
            variation
            * jnp.minimum(
                _ERROR_INFLATION * jnp.minimum(ratio, 1.0),
                1.0,
            )
            ** exponent,
            raw_error,
        )
        epsilon = jnp.finfo(real_dtype).eps
        roundoff = precision.decision(_ROUNDOFF_FACTOR * epsilon * absolute_scale)
        error = precision.decision(jnp.maximum(rescaled_error, roundoff))
        split_indicators = precision.decision(jax.vmap(_error_norm)(split_values))
        finite = (
            finite
            & jnp.all(jnp.isfinite(estimate))
            & jnp.all(jnp.isfinite(embedded))
            & jnp.all(jnp.isfinite(split_indicators))
            & jnp.isfinite(error)
        )
        return estimate, error, split_indicators, finite

    initial_lower, initial_upper, initial_bounds_valid = _canonical_initial_cells(
        reference_breakpoints, coordinate_dtype
    )
    initial_count = int(initial_lower.shape[0])
    initial_cost = initial_count * local_cost
    maximum_evaluations = (
        initial_cost + (plan.max_cells - initial_count) * 2 * local_cost
        if plan.max_evaluations is None
        else plan.max_evaluations
    )
    capacity = plan.max_cells

    if initial_cost > maximum_evaluations:
        value = jnp.zeros(output_shape, dtype=value_dtype)
        error = precision.decision(jnp.asarray(jnp.inf))
        status = jnp.where(
            initial_bounds_valid,
            int(IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED),
            int(IntegrationStatus.INVALID_BOUNDS),
        ).astype(jnp.int32)
        partition = None
        if plan.collect_partition:
            partition = AdaptiveCubaturePartition(
                count=jnp.asarray(0, dtype=jnp.int32),
                lower_bounds=jnp.zeros(
                    (capacity, plan.dimension), dtype=coordinate_dtype
                ),
                upper_bounds=jnp.zeros(
                    (capacity, plan.dimension), dtype=coordinate_dtype
                ),
                integral_estimates=jnp.zeros(
                    (capacity,) + output_shape, dtype=value_dtype
                ),
                estimated_errors=jnp.zeros((capacity,), dtype=error.dtype),
                split_indicators=jnp.zeros((capacity, plan.dimension), dtype=error.dtype),
                active=jnp.zeros((capacity,), dtype=bool),
            )
        if plan.throw:
            value = eqx.error_if(
                value,
                True,
                "Adaptive cubature cannot afford its initial partition.",
            )
        diagnostics = AdaptiveCubatureDiagnostics(
            status=status,
            num_evaluations=jnp.asarray(0, dtype=jnp.int32),
            estimated_error=error,
            partition=partition,
            dimension=plan.dimension,
            rule_id=rule.rule_id,
            family=rule.family,
            num_rule_points=rule.num_points,
            exact_degree=rule.exact_degree,
            embedded_degree=rule.embedded_degree,
            weight_l1_norm=rule.weight_l1_norm,
            negative_weight_mass=rule.negative_weight_mass,
            max_batch_points=plan.max_batch_points,
        )
        return IntegrationEstimate(
            precision.output(value),
            status=status,
            num_evaluations=jnp.asarray(0, dtype=jnp.int32),
            error_estimate=error,
            error_kind="embedded-cubature-indicator",
            diagnostics=diagnostics,
            provenance=IntegrationProvenance(
                "adaptive-cubature", "callable", rule.rule_id
            ),
        )

    def evaluate_initial(carry, bounds):
        del carry
        return None, evaluate_cell(bounds[0], bounds[1])

    def evaluate_initial_partition(_):
        _, values = jax.lax.scan(evaluate_initial, None, (initial_lower, initial_upper))
        return values

    def invalid_initial_partition(_):
        return (
            jnp.zeros((initial_count,) + output_shape, dtype=value_dtype),
            jnp.full((initial_count,), jnp.inf, dtype=real_dtype),
            jnp.zeros((initial_count, plan.dimension), dtype=real_dtype),
            jnp.zeros((initial_count,), dtype=bool),
        )

    (
        initial_estimates,
        initial_errors,
        initial_split_indicators,
        initial_finite,
    ) = jax.lax.cond(
        initial_bounds_valid,
        evaluate_initial_partition,
        invalid_initial_partition,
        None,
    )
    lowers = jnp.zeros((capacity, plan.dimension), dtype=coordinate_dtype)
    uppers = jnp.zeros((capacity, plan.dimension), dtype=coordinate_dtype)
    estimates = jnp.zeros((capacity,) + output_shape, dtype=value_dtype)
    errors = jnp.zeros((capacity,), dtype=initial_errors.dtype)
    split_indicators = jnp.zeros(
        (capacity, plan.dimension), dtype=initial_split_indicators.dtype
    )
    lowers = lowers.at[:initial_count].set(initial_lower)
    uppers = uppers.at[:initial_count].set(initial_upper)
    estimates = estimates.at[:initial_count].set(initial_estimates)
    errors = errors.at[:initial_count].set(initial_errors)
    split_indicators = split_indicators.at[:initial_count].set(initial_split_indicators)
    effective_initial_count = jnp.where(initial_bounds_valid, initial_count, 0).astype(
        jnp.int32
    )
    anisotropy = jnp.asarray(plan.anisotropy, dtype=coordinate_dtype)

    def aspect_ratios(
        lower_bounds: Array, upper_bounds: Array, active_cells: Array, /
    ) -> Array:
        widths = (upper_bounds - lower_bounds) / anisotropy
        largest = jnp.max(widths, axis=1)
        smallest = jnp.min(jnp.where(widths > 0.0, widths, jnp.inf), axis=1)
        ratios = largest / jnp.where(smallest > 0.0, smallest, 1.0)
        return jnp.where(active_cells, ratios, 0.0)

    active = (jnp.arange(capacity) < initial_count) & initial_bounds_valid
    value = precision.accumulation(compensated_sum(initial_estimates, axis=0))
    error = precision.decision(compensated_sum(initial_errors, axis=0))
    finite = jnp.all(initial_finite)
    status = jnp.where(
        ~initial_bounds_valid,
        int(IntegrationStatus.INVALID_BOUNDS),
        jnp.where(
            finite,
            int(IntegrationStatus.CONVERGED),
            int(IntegrationStatus.NONFINITE_INTEGRAND),
        ),
    ).astype(jnp.int32)
    mesh_regular = jnp.all(
        aspect_ratios(lowers, uppers, active) <= _MAX_CELL_ASPECT_RATIO
    )
    done = (
        (~initial_bounds_valid)
        | (~finite)
        | (_meets_plan_tolerance(value, error, plan, precision) & mesh_regular)
    )
    state = (
        lowers,
        uppers,
        estimates,
        errors,
        split_indicators,
        active,
        effective_initial_count,
        value,
        error,
        jnp.where(initial_bounds_valid, initial_cost, 0).astype(jnp.int32),
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
                split_all,
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
            evaluation_exhausted = evaluations + 2 * local_cost > maximum_evaluations

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
                    split_all,
                    active_all,
                    count,
                    global_estimate,
                    global_error,
                    evaluations,
                    failure.astype(jnp.int32),
                    jnp.asarray(True),
                )

            def split(_):
                ratios = aspect_ratios(lower_all, upper_all, active_all)
                numerically_ready = _meets_plan_tolerance(
                    global_estimate, global_error, plan, precision
                )
                rebalance = numerically_ready & jnp.any(ratios > _MAX_CELL_ASPECT_RATIO)
                error_index = jnp.argmax(jnp.where(active_all, error_all, -jnp.inf))
                shape_index = jnp.argmax(ratios)
                index = jnp.where(rebalance, shape_index, error_index)
                lower = lower_all[index]
                upper = upper_all[index]
                local_split = split_all[index] / anisotropy
                widths = (upper - lower) / anisotropy
                smallest_width = jnp.min(jnp.where(widths > 0.0, widths, jnp.inf))
                irregular = jnp.max(widths) > (_MAX_CELL_ASPECT_RATIO * smallest_width)
                has_signal = jnp.any(jnp.isfinite(local_split) & (local_split > 0.0))
                rule_axis = jnp.argmax(
                    jnp.where(jnp.isfinite(local_split), local_split, -jnp.inf)
                )
                axis = jnp.where(
                    rebalance | irregular | (~has_signal),
                    jnp.argmax(widths),
                    rule_axis,
                )
                midpoint = 0.5 * (lower[axis] + upper[axis])
                stagnated = (midpoint == lower[axis]) | (midpoint == upper[axis])

                def stagnation(_):
                    return (
                        lower_all,
                        upper_all,
                        estimate_all,
                        error_all,
                        split_all,
                        active_all,
                        count,
                        global_estimate,
                        global_error,
                        evaluations,
                        jnp.asarray(
                            int(IntegrationStatus.REFINEMENT_STAGNATION),
                            dtype=jnp.int32,
                        ),
                        jnp.asarray(True),
                    )

                def evaluate_children(_):
                    left_upper = upper.at[axis].set(midpoint)
                    right_lower = lower.at[axis].set(midpoint)

                    def evaluate_child(child_carry, bounds):
                        del child_carry
                        return None, evaluate_cell(bounds[0], bounds[1])

                    _, children = jax.lax.scan(
                        evaluate_child,
                        None,
                        (
                            jnp.stack((lower, right_lower)),
                            jnp.stack((left_upper, upper)),
                        ),
                    )
                    (
                        child_estimates,
                        child_errors,
                        child_split_indicators,
                        child_finite,
                    ) = children
                    child_sum = compensated_sum(child_estimates, axis=0)
                    disagreement = precision.decision(
                        _error_norm(estimate_all[index] - child_sum)
                    )
                    child_error_sum = jnp.sum(child_errors)
                    shares = jnp.where(
                        child_error_sum > 0.0,
                        child_errors
                        / jnp.where(child_error_sum > 0.0, child_error_sum, 1.0),
                        jnp.asarray((0.5, 0.5), dtype=child_errors.dtype),
                    )
                    adjusted_errors = precision.decision(
                        child_errors + disagreement * shares
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
                        .set(adjusted_errors[0])
                        .at[count]
                        .set(adjusted_errors[1])
                    )
                    next_split = (
                        split_all.at[index]
                        .set(child_split_indicators[0])
                        .at[count]
                        .set(child_split_indicators[1])
                    )
                    next_active = active_all.at[count].set(True)
                    value_mask = next_active.reshape(
                        (capacity,) + (1,) * len(output_shape)
                    )
                    estimate = precision.accumulation(
                        compensated_sum(
                            jnp.where(value_mask, next_estimates, 0.0),
                            axis=0,
                        )
                    )
                    error = precision.decision(
                        compensated_sum(
                            jnp.where(next_active, next_errors, 0.0),
                            axis=0,
                        )
                    )
                    children_finite = jnp.all(child_finite)
                    mesh_regular = jnp.all(
                        aspect_ratios(next_lower, next_upper, next_active)
                        <= _MAX_CELL_ASPECT_RATIO
                    )
                    converged = (
                        _meets_plan_tolerance(estimate, error, plan, precision)
                        & mesh_regular
                    )
                    terminal = (~children_finite) | converged
                    next_status = jnp.where(
                        children_finite,
                        int(IntegrationStatus.CONVERGED),
                        int(IntegrationStatus.NONFINITE_INTEGRAND),
                    ).astype(jnp.int32)
                    return (
                        next_lower,
                        next_upper,
                        next_estimates,
                        next_errors,
                        next_split,
                        next_active,
                        count + 1,
                        estimate,
                        error,
                        evaluations + 2 * local_cost,
                        next_status,
                        terminal,
                    )

                return jax.lax.cond(stagnated, stagnation, evaluate_children, None)

            return jax.lax.cond(
                capacity_exhausted | evaluation_exhausted, fail, split, None
            )

        return jax.lax.cond(carry[-1], lambda value: value, refine, carry), None

    state, _ = jax.lax.scan(
        iteration,
        state,
        xs=None,
        length=capacity - initial_count,
    )
    (
        lowers,
        uppers,
        estimates,
        errors,
        split_indicators,
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
            split_indicators=split_indicators,
            active=active,
        )
    diagnostics = AdaptiveCubatureDiagnostics(
        status=status,
        num_evaluations=evaluations,
        estimated_error=error,
        partition=partition,
        dimension=plan.dimension,
        rule_id=rule.rule_id,
        family=rule.family,
        num_rule_points=rule.num_points,
        exact_degree=rule.exact_degree,
        embedded_degree=rule.embedded_degree,
        weight_l1_norm=rule.weight_l1_norm,
        negative_weight_mass=rule.negative_weight_mass,
        max_batch_points=plan.max_batch_points,
    )
    return IntegrationEstimate(
        precision.output(value),
        status=status,
        num_evaluations=evaluations,
        error_estimate=error,
        error_kind="embedded-cubature-indicator",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("adaptive-cubature", "callable", rule.rule_id),
    )


def adaptive_cubature_callable(
    integrand,
    plan: AdaptiveCubaturePlan,
    /,
    *,
    precision: IntegrationPrecisionPolicy,
) -> IntegrationEstimate:
    """Adapt a coupled callable on ``[-1,1]^dimension`` with bounded cells."""
    return _adaptive_cubature_solve(
        integrand,
        plan,
        plan.breakpoints,
        precision=precision,
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
    if not varying:
        raise ValueError("AdaptiveCubaturePlan requires at least one varying factor.")
    if any(
        not isinstance(component.spec.selection_for(label), Interior) for label in varying
    ):
        raise TypeError("Adaptive cubature varying factors must select Interior().")
    dimension = sum(
        _factor_dimension(component.domain.factor(label)) for label in varying
    )
    if dimension != plan.dimension:
        raise ValueError(
            "AdaptiveCubaturePlan rule dimension must equal the flattened varying "
            f"coordinate dimension {dimension}."
        )
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
    prototype_shape = eqx.filter_eval_shape(
        callback.field,
        precision.evaluation(jnp.zeros((1, plan.dimension), dtype=float)),
    )
    prototype = cx.AxisArray(
        jnp.zeros(prototype_shape.data.shape, dtype=prototype_shape.data.dtype),
        dims=prototype_shape.dims,
    )
    reduced = sum_over(
        prototype,
        axis,
        accumulation_dtype=precision.accumulation_dtype,
    )
    raw = _adaptive_cubature_solve(
        callback,
        plan,
        _reference_breakpoints(component, varying, plan),
        precision=precision,
    )
    return eqx.tree_at(
        lambda estimate: estimate.value,
        raw,
        cx.AxisArray(raw.value, dims=reduced.dims),
    )


def _ratio(
    numerator: IntegrationEstimate,
    denominator: IntegrationEstimate,
    plan: AdaptiveCubaturePlan,
    /,
) -> IntegrationEstimate:
    numerator_error = numerator.error_estimate
    denominator_error = denominator.error_estimate
    if numerator_error is None or denominator_error is None:
        raise RuntimeError("Adaptive cubature ratio requires both error indicators.")
    mass = denominator.value.data
    positive = (
        jnp.asarray(False)
        if jnp.issubdtype(mass.dtype, jnp.complexfloating)
        else jnp.all(mass > 0.0)
    )
    mass_valid = jnp.all(jnp.isfinite(mass)) & positive
    status = jnp.where(
        ~numerator.successful,
        numerator.status,
        jnp.where(
            ~denominator.successful,
            denominator.status,
            jnp.where(
                mass_valid,
                int(IntegrationStatus.CONVERGED),
                int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
            ),
        ),
    ).astype(jnp.int32)
    value = numerator.value.data / mass
    mass_norm = jnp.maximum(_error_norm(mass), jnp.finfo(jnp.real(mass).dtype).tiny)
    error = (
        numerator_error / mass_norm
        + _error_norm(numerator.value.data) * denominator_error / mass_norm**2
    )
    if plan.throw:
        value = eqx.error_if(
            value,
            status != int(IntegrationStatus.CONVERGED),
            "Adaptive cubature normalization failed.",
        )
    return IntegrationEstimate(
        cx.AxisArray(value, dims=numerator.value.dims),
        status=status,
        num_evaluations=numerator.num_evaluations + denominator.num_evaluations,
        error_estimate=error,
        error_kind="ratio-embedded-cubature-indicator",
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
    geometry_factor = (
        base.component.domain.factor(varying[0]) if len(varying) == 1 else None
    )
    if isinstance(geometry_factor, AbstractGeometry) and not isinstance(
        geometry_factor, HyperRectangle
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
