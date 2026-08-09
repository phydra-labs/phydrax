#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._numerics import solve_weighted_least_squares
from ..._strict import StrictModule
from ...operators.interpolation import BSplineInterpolationPlan, fit_bspline
from .._trajectory import TrajectoryData


FiniteDifferenceEndpoint: TypeAlias = Literal["invalid", "one-sided"]


def _event_scale(value: Array, event_rank: int, /) -> Array:
    return value.reshape(value.shape + (1,) * event_rank)


def _time_index(case_rank: int, index, /) -> tuple:
    return (slice(None),) * case_rank + (index,)


class DerivativeEstimate(StrictModule):
    """State derivatives plus local accuracy, conditioning, and validity evidence."""

    values: Array
    valid: Array
    order: Array
    condition_number: Array
    method_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)

    def attach(self, data: TrajectoryData, /) -> TrajectoryData:
        if not isinstance(data, TrajectoryData):
            raise TypeError("data must be TrajectoryData.")
        if data.dataset_id != self.dataset_id:
            raise ValueError(
                "Derivative estimate belongs to a different trajectory dataset."
            )
        return data.with_derivatives(
            self.values,
            self.valid,
            source_id=f"{data.source_id}:{self.method_id}",
        )


def finite_difference_derivative(
    data: TrajectoryData,
    /,
    *,
    endpoint: FiniteDifferenceEndpoint = "one-sided",
) -> DerivativeEstimate:
    """Estimate irregular-grid derivatives without crossing invalid transitions."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    if endpoint not in ("invalid", "one-sided"):
        raise ValueError("endpoint must be 'invalid' or 'one-sided'.")
    event_rank = len(data.state_layout.shape)
    case_rank = len(data.case_shape)
    derivative = jnp.full_like(data.states, jnp.nan)
    valid = jnp.zeros(data.case_shape + (data.capacity,), dtype=bool)
    order = jnp.zeros(data.case_shape + (data.capacity,), dtype=jnp.int32)
    condition = jnp.full(
        data.case_shape + (data.capacity,), jnp.inf, dtype=data.coordinates.dtype
    )

    for index in range(data.capacity):
        current_index = _time_index(case_rank, index)
        previous_valid = (
            jnp.zeros(data.case_shape, dtype=bool)
            if index == 0
            else data.transition_valid[..., index - 1]
        )
        next_valid = (
            jnp.zeros(data.case_shape, dtype=bool)
            if index + 1 == data.capacity
            else data.transition_valid[..., index]
        )
        if index > 0:
            previous_state = data.states[_time_index(case_rank, index - 1)]
            previous_dt = data.coordinates[..., index] - data.coordinates[..., index - 1]
            safe_previous_dt = jnp.where(previous_valid, previous_dt, 1.0)
            backward = (data.states[current_index] - previous_state) / _event_scale(
                safe_previous_dt, event_rank
            )
        else:
            previous_dt = jnp.ones(data.case_shape, dtype=data.coordinates.dtype)
            backward = jnp.zeros_like(data.states[current_index])
        if index + 1 < data.capacity:
            next_state = data.states[_time_index(case_rank, index + 1)]
            next_dt = data.coordinates[..., index + 1] - data.coordinates[..., index]
            safe_next_dt = jnp.where(next_valid, next_dt, 1.0)
            forward = (next_state - data.states[current_index]) / _event_scale(
                safe_next_dt, event_rank
            )
        else:
            next_dt = jnp.ones(data.case_shape, dtype=data.coordinates.dtype)
            forward = jnp.zeros_like(data.states[current_index])

        central_valid = previous_valid & next_valid
        safe_previous_dt = jnp.where(central_valid, previous_dt, 1.0)
        safe_next_dt = jnp.where(central_valid, next_dt, 1.0)
        total_dt = safe_previous_dt + safe_next_dt
        previous_weight = -safe_next_dt / (safe_previous_dt * total_dt)
        current_weight = (safe_next_dt - safe_previous_dt) / (
            safe_previous_dt * safe_next_dt
        )
        next_weight = safe_previous_dt / (safe_next_dt * total_dt)
        previous_state = data.states[_time_index(case_rank, max(index - 1, 0))]
        next_state = data.states[
            _time_index(case_rank, min(index + 1, data.capacity - 1))
        ]
        central = (
            _event_scale(previous_weight, event_rank) * previous_state
            + _event_scale(current_weight, event_rank) * data.states[current_index]
            + _event_scale(next_weight, event_rank) * next_state
        )
        one_sided_valid = (
            (next_valid | previous_valid)
            if endpoint == "one-sided"
            else jnp.zeros_like(next_valid)
        )
        resolved_valid = central_valid | one_sided_valid
        one_sided = jnp.where(_event_scale(next_valid, event_rank), forward, backward)
        resolved = jnp.where(_event_scale(central_valid, event_rank), central, one_sided)
        derivative = derivative.at[current_index].set(
            jnp.where(
                _event_scale(resolved_valid, event_rank),
                resolved,
                jnp.full_like(resolved, jnp.nan),
            )
        )
        valid = valid.at[..., index].set(resolved_valid)
        order = order.at[..., index].set(
            jnp.where(central_valid, 2, jnp.where(one_sided_valid, 1, 0))
        )
        spacing_condition = jnp.maximum(safe_previous_dt, safe_next_dt) / jnp.minimum(
            safe_previous_dt, safe_next_dt
        )
        condition = condition.at[..., index].set(
            jnp.where(
                central_valid, spacing_condition, jnp.where(one_sided_valid, 1.0, jnp.inf)
            )
        )

    return DerivativeEstimate(
        values=derivative,
        valid=valid,
        order=order,
        condition_number=condition,
        method_id=f"finite-difference:{endpoint}",
        dataset_id=data.dataset_id,
        state_shape=data.state_layout.shape,
    )


def _connected(data: TrajectoryData, source: int, target: int, /) -> Array:
    lower = min(source, target)
    upper = max(source, target)
    connected = jnp.ones(data.case_shape, dtype=bool)
    for index in range(lower, upper):
        connected = connected & data.transition_valid[..., index]
    return connected


def local_polynomial_derivative(
    data: TrajectoryData,
    /,
    *,
    degree: int = 2,
    window_radius: int = 2,
    rcond: float | None = None,
) -> DerivativeEstimate:
    """Fit one mask-aware local polynomial per sample on physical coordinates."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    polynomial_degree = int(degree)
    radius = int(window_radius)
    if polynomial_degree < 1:
        raise ValueError("degree must be positive.")
    if radius < 1 or 2 * radius + 1 < polynomial_degree + 1:
        raise ValueError("window_radius does not provide enough polynomial samples.")
    offsets = tuple(range(-radius, radius + 1))
    state_size = data.state_layout.size
    designs = []
    responses = []
    masks = []
    weights = []
    scales = []
    for target in range(data.capacity):
        target_coordinates = data.coordinates[..., target]
        target_design = []
        target_response = []
        target_mask = []
        target_weights = []
        for offset in offsets:
            source = target + offset
            in_bounds = 0 <= source < data.capacity
            clipped = min(max(source, 0), data.capacity - 1)
            difference = data.coordinates[..., clipped] - target_coordinates
            connected = (
                _connected(data, clipped, target)
                if in_bounds
                else jnp.zeros(data.case_shape, dtype=bool)
            )
            row_valid = (
                data.sample_valid[..., clipped]
                & data.sample_valid[..., target]
                & connected
            )
            target_design.append(difference)
            target_response.append(
                data.states[_time_index(len(data.case_shape), clipped)].reshape(
                    data.case_shape + (state_size,)
                )
            )
            target_mask.append(row_valid)
            target_weights.append(data.weights[..., clipped])
        differences = jnp.stack(target_design, axis=-1)
        row_mask = jnp.stack(target_mask, axis=-1)
        scale = jnp.max(jnp.where(row_mask, jnp.abs(differences), 0.0), axis=-1)
        scale = jnp.where(scale > 0.0, scale, 1.0)
        normalized = differences / scale[..., None]
        design = jnp.stack(
            tuple(normalized**power for power in range(polynomial_degree + 1)),
            axis=-1,
        )
        designs.append(design)
        responses.append(jnp.stack(target_response, axis=-2))
        masks.append(row_mask)
        weights.append(jnp.stack(target_weights, axis=-1))
        scales.append(scale)

    design_array = jnp.stack(designs, axis=len(data.case_shape))
    response_array = jnp.stack(responses, axis=len(data.case_shape))
    mask_array = jnp.stack(masks, axis=len(data.case_shape))
    weight_array = jnp.stack(weights, axis=len(data.case_shape))
    scale_array = jnp.stack(scales, axis=len(data.case_shape))
    problem_count = data.num_cases * data.capacity
    window_size = len(offsets)
    feature_count = polynomial_degree + 1
    flat_design = design_array.reshape((problem_count, window_size, feature_count))
    flat_response = response_array.reshape((problem_count, window_size, state_size))
    flat_mask = mask_array.reshape((problem_count, window_size))
    flat_weights = weight_array.reshape((problem_count, window_size))
    flat_scale = scale_array.reshape((problem_count,))

    def solve_one(design, response, mask, sample_weights, scale):
        result = solve_weighted_least_squares(
            design,
            response,
            mask=mask,
            weights=sample_weights,
            rcond=rcond,
            min_samples=feature_count,
        )
        derivative = result.coefficients[1] / scale
        return derivative, result.valid, result.condition_number

    flat_derivative, flat_valid, flat_condition = jax.vmap(solve_one)(
        flat_design,
        flat_response,
        flat_mask,
        flat_weights,
        flat_scale,
    )
    sample_shape = data.case_shape + (data.capacity,)
    valid = flat_valid.reshape(sample_shape) & data.sample_valid
    derivative = flat_derivative.reshape(sample_shape + data.state_layout.shape)
    derivative = jnp.where(
        _event_scale(valid, len(data.state_layout.shape)),
        derivative,
        jnp.full_like(derivative, jnp.nan),
    )
    return DerivativeEstimate(
        values=derivative,
        valid=valid,
        order=jnp.where(valid, polynomial_degree, 0),
        condition_number=flat_condition.reshape(sample_shape),
        method_id=f"local-polynomial:degree={polynomial_degree}:radius={radius}",
        dataset_id=data.dataset_id,
        state_shape=data.state_layout.shape,
    )


def bspline_derivative(
    data: TrajectoryData,
    /,
    *,
    plan: BSplineInterpolationPlan | None = None,
) -> DerivativeEstimate:
    """Fit each contiguous valid segment and differentiate its physical-time spline."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    resolved_plan = BSplineInterpolationPlan() if plan is None else plan
    if not isinstance(resolved_plan, BSplineInterpolationPlan):
        raise TypeError("plan must be a BSplineInterpolationPlan or None.")
    flat_coordinates = np.asarray(data.coordinates).reshape(
        (data.num_cases, data.capacity)
    )
    flat_states = np.asarray(data.states).reshape(
        (data.num_cases, data.capacity, data.state_layout.size)
    )
    flat_samples = np.asarray(data.sample_valid).reshape((data.num_cases, data.capacity))
    flat_transitions = np.asarray(data.transition_valid).reshape(
        (data.num_cases, data.capacity - 1)
    )
    flat_weights = np.asarray(data.weights).reshape((data.num_cases, data.capacity))
    derivative = np.full_like(flat_states, np.nan, dtype=float)
    valid = np.zeros((data.num_cases, data.capacity), dtype=bool)
    condition = np.full((data.num_cases, data.capacity), np.inf, dtype=float)
    minimum = resolved_plan.degree + 1

    for case in range(data.num_cases):
        start = 0
        while start < data.capacity:
            if not flat_samples[case, start]:
                start += 1
                continue
            end = start + 1
            while (
                end < data.capacity
                and flat_samples[case, end]
                and flat_transitions[case, end - 1]
            ):
                end += 1
            if end - start >= minimum:
                nodes = jnp.asarray(flat_coordinates[case, start:end])
                values = jnp.asarray(flat_states[case, start:end])
                sample_weights = (
                    None
                    if resolved_plan.mode == "interpolate"
                    else jnp.asarray(flat_weights[case, start:end])
                )
                interpolant = fit_bspline(
                    nodes,
                    values,
                    plan=resolved_plan,
                    sample_weights=sample_weights,
                )
                derivative[case, start:end] = np.asarray(interpolant.derivative(nodes))
                valid[case, start:end] = True
                condition[case, start:end] = interpolant.diagnostics.condition_estimate
            start = end

    sample_shape = data.case_shape + (data.capacity,)
    return DerivativeEstimate(
        values=jnp.asarray(derivative).reshape(sample_shape + data.state_layout.shape),
        valid=jnp.asarray(valid).reshape(sample_shape),
        order=jnp.where(
            jnp.asarray(valid).reshape(sample_shape), resolved_plan.degree, 0
        ),
        condition_number=jnp.asarray(condition).reshape(sample_shape),
        method_id=(f"bspline:degree={resolved_plan.degree}:mode={resolved_plan.mode}"),
        dataset_id=data.dataset_id,
        state_shape=data.state_layout.shape,
    )


__all__ = [
    "DerivativeEstimate",
    "FiniteDifferenceEndpoint",
    "bspline_derivative",
    "finite_difference_derivative",
    "local_polynomial_derivative",
]
