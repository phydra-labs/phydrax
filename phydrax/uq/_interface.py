#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry import regularized_heaviside_values


class InterfacePredictiveSummary(StrictModule):
    occupancy_probability: Array
    interface_probability: Array
    level_set_mean: Array
    level_set_standard_deviation: Array
    phase_measure_samples: Array
    phase_measure_mean: Array
    phase_measure_standard_deviation: Array


class InterfaceAcquisitionPolicy(StrictModule, NonTrainableState):
    uncertainty_weight: float
    residual_weight: float
    diversity_weight: float
    normalization_epsilon: float

    def __init__(
        self,
        *,
        uncertainty_weight: float = 1.0,
        residual_weight: float = 1.0,
        diversity_weight: float = 1.0,
        normalization_epsilon: float = 1.0e-12,
    ):
        values = tuple(
            float(value)
            for value in (uncertainty_weight, residual_weight, diversity_weight)
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Acquisition weights must be finite and nonnegative.")
        if not any(value > 0.0 for value in values):
            raise ValueError("At least one acquisition weight must be positive.")
        epsilon = float(normalization_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be finite and positive.")
        (
            self.uncertainty_weight,
            self.residual_weight,
            self.diversity_weight,
        ) = values
        self.normalization_epsilon = epsilon


class InterfaceAcquisitionResult(StrictModule):
    indices: Array
    scores: Array
    uncertainty: Array
    residual: Array
    diversity: Array


def interface_predictive_summary(
    level_set_samples: ArrayLike,
    quadrature_weights: ArrayLike,
    /,
    *,
    width: float,
    sample_axis: int = 0,
    point_axis: int = -1,
) -> InterfacePredictiveSummary:
    """Derive phase and interface uncertainty from coherent level-set draws."""

    values = jnp.asarray(level_set_samples)
    weights = jnp.asarray(quadrature_weights, dtype=float)
    if values.ndim < 2 or jnp.iscomplexobj(values):
        raise ValueError("level_set_samples must be real with draw and point axes.")
    draw_axis = int(sample_axis) % values.ndim
    point_axis_ = int(point_axis) % values.ndim
    if draw_axis == point_axis_:
        raise ValueError("sample_axis and point_axis must be distinct.")
    values = jnp.moveaxis(values, (draw_axis, point_axis_), (0, -1))
    weights = jnp.broadcast_to(weights, values.shape[1:])
    if bool(jnp.any(~jnp.isfinite(weights) | (weights < 0.0))):
        raise ValueError("quadrature_weights must be finite and nonnegative.")
    occupancy = regularized_heaviside_values(-values, width=width)
    probability = jnp.mean(occupancy, axis=0)
    phase_measure_samples = jnp.sum(occupancy * weights[None, ...], axis=-1)
    return InterfacePredictiveSummary(
        occupancy_probability=probability,
        interface_probability=4.0 * probability * (1.0 - probability),
        level_set_mean=jnp.mean(values, axis=0),
        level_set_standard_deviation=jnp.std(values, axis=0),
        phase_measure_samples=phase_measure_samples,
        phase_measure_mean=jnp.mean(phase_measure_samples, axis=0),
        phase_measure_standard_deviation=jnp.std(phase_measure_samples, axis=0),
    )


def select_interface_acquisition(
    candidate_points: ArrayLike,
    predictive_uncertainty: ArrayLike,
    physics_residual: ArrayLike,
    count: int,
    /,
    *,
    existing_points: ArrayLike | None = None,
    policy: InterfaceAcquisitionPolicy | None = None,
) -> InterfaceAcquisitionResult:
    """Select candidates by uncertainty, physics residual, and sequential diversity."""

    points = jnp.asarray(candidate_points, dtype=float)
    uncertainty = jnp.asarray(predictive_uncertainty, dtype=float)
    residual = jnp.asarray(physics_residual, dtype=float)
    if points.ndim != 2 or min(points.shape) <= 0:
        raise ValueError("candidate_points must have shape (candidate, coordinate).")
    candidate_count = int(points.shape[0])
    if uncertainty.shape != (candidate_count,) or residual.shape != (candidate_count,):
        raise ValueError("Acquisition signals must contain one value per candidate.")
    if bool(
        jnp.any(~jnp.isfinite(points))
        | jnp.any(~jnp.isfinite(uncertainty))
        | jnp.any(~jnp.isfinite(residual))
        | jnp.any(uncertainty < 0.0)
        | jnp.any(residual < 0.0)
    ):
        raise ValueError("Acquisition inputs must be finite and signals nonnegative.")
    selection_count = int(count)
    if selection_count <= 0 or selection_count > candidate_count:
        raise ValueError("count must lie in [1, candidate_count].")
    resolved = InterfaceAcquisitionPolicy() if policy is None else policy
    if not isinstance(resolved, InterfaceAcquisitionPolicy):
        raise TypeError("policy must be InterfaceAcquisitionPolicy or None.")

    normalized_uncertainty = _normalize(uncertainty, resolved.normalization_epsilon)
    normalized_residual = _normalize(residual, resolved.normalization_epsilon)
    if existing_points is None:
        minimum_distance = jnp.ones((candidate_count,), dtype=points.dtype)
    else:
        existing = jnp.asarray(existing_points, dtype=float)
        if (
            existing.ndim != 2
            or existing.shape[-1] != points.shape[-1]
            or existing.shape[0] == 0
        ):
            raise ValueError(
                "existing_points must have non-empty shape (point, candidate_coordinate)."
            )
        minimum_distance = jnp.min(_pairwise_distances(points, existing), axis=-1)
    selected = jnp.zeros((candidate_count,), dtype=bool)
    selected_indices = []
    selected_scores = []
    selected_diversity = []
    for _ in range(selection_count):
        diversity = _normalize(minimum_distance, resolved.normalization_epsilon)
        score = (
            resolved.uncertainty_weight * normalized_uncertainty
            + resolved.residual_weight * normalized_residual
            + resolved.diversity_weight * diversity
        )
        score = jnp.where(selected, -jnp.inf, score)
        index = jnp.argmax(score).astype(jnp.int32)
        selected_indices.append(index)
        selected_scores.append(score[index])
        selected_diversity.append(diversity[index])
        selected = selected.at[index].set(True)
        distance = jnp.sqrt(jnp.sum((points - points[index]) ** 2, axis=-1))
        minimum_distance = jnp.minimum(minimum_distance, distance)
    indices = jnp.stack(selected_indices)
    return InterfaceAcquisitionResult(
        indices=indices,
        scores=jnp.stack(selected_scores),
        uncertainty=uncertainty[indices],
        residual=residual[indices],
        diversity=jnp.stack(selected_diversity),
    )


def _normalize(values: Array, epsilon: float, /) -> Array:
    minimum = jnp.min(values)
    extent = jnp.max(values) - minimum
    return (values - minimum) / jnp.maximum(extent, epsilon)


def _pairwise_distances(left: Array, right: Array, /) -> Array:
    left_square = jnp.sum(left * left, axis=-1, keepdims=True)
    right_square = jnp.sum(right * right, axis=-1)
    cross = ein.contract("id,jd->ij", left, right)
    return jnp.sqrt(jnp.maximum(left_square + right_square[None, :] - 2.0 * cross, 0.0))


__all__ = [
    "InterfaceAcquisitionPolicy",
    "InterfaceAcquisitionResult",
    "InterfacePredictiveSummary",
    "interface_predictive_summary",
    "select_interface_acquisition",
]
