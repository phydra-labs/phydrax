#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization
from ._pairwise import ParticlePairGeometry, ParticlePairRelation, scatter_pair_sum
from ._precision import ParticleExecutionPolicy
from ._smoothing import AbstractSPHSmoothingKernel


class FreeSurfaceDetectionPlan(StrictModule, NonTrainableState):
    completeness_threshold: float = eqx.field(static=True)
    normal_threshold: float = eqx.field(static=True)
    cone_angle: float = eqx.field(static=True)
    smooth_sharpness: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        completeness_threshold: float = 0.85,
        normal_threshold: float = 0.15,
        cone_angle: float = np.pi / 3.0,
        smooth_sharpness: float = 30.0,
    ):
        values = tuple(
            float(value)
            for value in (
                completeness_threshold,
                normal_threshold,
                cone_angle,
                smooth_sharpness,
            )
        )
        if not 0.0 < values[0] < 1.5 or values[1] <= 0.0:
            raise ValueError(
                "Free-surface completeness and normal thresholds are invalid."
            )
        if not 0.0 < values[2] < np.pi or values[3] <= 0.0:
            raise ValueError("Free-surface cone angle and sharpness are invalid.")
        (
            self.completeness_threshold,
            self.normal_threshold,
            self.cone_angle,
            self.smooth_sharpness,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-surface-detection",
                "completeness_threshold": values[0],
                "normal_threshold": values[1],
                "cone_angle": values[2],
                "smooth_sharpness": values[3],
            }
        )


class FreeSurfaceState(StrictModule):
    completeness: Array
    normal: Array
    normal_magnitude: Array
    maximum_cone_projection: Array
    neighbor_count: Array
    hard_mask: Array
    smooth_weight: Array
    ambiguous_mask: Array


class FreeSurfacePressurePlan(StrictModule, NonTrainableState):
    atmospheric_pressure: float = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, atmospheric_pressure: float = 0.0, /, *, mode: str = "hard"):
        if mode not in ("hard", "smooth"):
            raise ValueError("Free-surface pressure mode must be 'hard' or 'smooth'.")
        pressure = float(atmospheric_pressure)
        if not np.isfinite(pressure):
            raise ValueError("atmospheric_pressure must be finite.")
        self.atmospheric_pressure = pressure
        self.mode = mode
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-surface-pressure",
                "atmospheric_pressure": pressure,
                "mode": mode,
            }
        )

    def apply(self, pressure: ArrayLike, state: FreeSurfaceState, /) -> Array:
        value = jnp.asarray(pressure)
        atmospheric = jnp.asarray(self.atmospheric_pressure, dtype=value.dtype)
        if self.mode == "hard":
            return jnp.where(state.hard_mask, atmospheric, value)
        return (1.0 - state.smooth_weight) * value + state.smooth_weight * atmospheric


class FreeSurfaceOperatorCorrectionPlan(StrictModule, NonTrainableState):
    minimum_completeness: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, minimum_completeness: float = 0.2, /):
        minimum = float(minimum_completeness)
        if not np.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_completeness must be finite and positive.")
        self.minimum_completeness = minimum
        self.plan_id = canonical_fingerprint(
            {"kind": "free-surface-operator-normalization", "minimum": minimum}
        )

    def normalize(self, value: ArrayLike, state: FreeSurfaceState, /) -> Array:
        array = jnp.asarray(value)
        denominator = jnp.maximum(
            state.completeness, jnp.asarray(self.minimum_completeness, array.dtype)
        )
        return array / denominator.reshape(
            denominator.shape + (1,) * (array.ndim - denominator.ndim)
        )


def detect_free_surface(
    plan: FreeSurfaceDetectionPlan,
    particles: ParticleDiscretization,
    density: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    physical_pairs: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: float,
    execution: ParticleExecutionPolicy,
    /,
) -> FreeSurfaceState:
    density_ = jnp.asarray(density)
    volume = particles.safe_masses / density_
    valid = pairs.valid & jnp.asarray(physical_pairs, dtype=bool)
    weights = kernel.value(geometry.distance, smoothing_length)
    left = pairs.left_indices
    right = pairs.right_indices
    neighbor_completeness = scatter_pair_sum(
        pairs,
        volume[right] * weights,
        volume[left] * weights,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    self_weight = kernel.value(jnp.asarray(0.0, weights.dtype), smoothing_length)
    completeness = volume * self_weight + neighbor_completeness
    gradient = kernel.gradient(geometry.displacement, geometry.distance, smoothing_length)
    normal = scatter_pair_sum(
        pairs,
        -volume[right, None] * gradient,
        volume[left, None] * gradient,
        size=particles.capacity,
        accumulation=execution.accumulation,
        valid=valid,
    )
    magnitude = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
    unit = normal / jnp.where(magnitude > 0.0, magnitude, 1.0)[:, None]
    neighbor_direction_left = -geometry.direction
    neighbor_direction_right = geometry.direction
    projection_left = jnp.sum(neighbor_direction_left * unit[left], axis=-1)
    projection_right = jnp.sum(neighbor_direction_right * unit[right], axis=-1)
    projections = jnp.full((particles.capacity,), -1.0, dtype=weights.dtype)
    projections = projections.at[left].max(jnp.where(valid, projection_left, -1.0))
    projections = projections.at[right].max(jnp.where(valid, projection_right, -1.0))
    counts = jnp.zeros((particles.capacity,), dtype=jnp.int32)
    counts = counts.at[left].add(valid.astype(jnp.int32))
    counts = counts.at[right].add(valid.astype(jnp.int32))
    completeness_test = completeness < plan.completeness_threshold
    normal_test = smoothing_length * magnitude > plan.normal_threshold
    cone_test = projections < jnp.cos(plan.cone_angle)
    hard = particles.active_mask & completeness_test & normal_test & cone_test
    completeness_score = jax.nn.sigmoid(
        plan.smooth_sharpness * (plan.completeness_threshold - completeness)
    )
    normal_score = jax.nn.sigmoid(
        plan.smooth_sharpness * (smoothing_length * magnitude - plan.normal_threshold)
    )
    cone_score = jax.nn.sigmoid(
        plan.smooth_sharpness * (jnp.cos(plan.cone_angle) - projections)
    )
    smooth = jnp.where(
        particles.active_mask, completeness_score * normal_score * cone_score, 0.0
    )
    ambiguous = particles.active_mask & (smooth > 0.1) & (smooth < 0.9)
    return FreeSurfaceState(
        completeness,
        unit,
        magnitude,
        projections,
        counts,
        hard,
        smooth,
        ambiguous,
    )


__all__ = [
    "FreeSurfaceDetectionPlan",
    "FreeSurfaceOperatorCorrectionPlan",
    "FreeSurfacePressurePlan",
    "FreeSurfaceState",
    "detect_free_surface",
]
