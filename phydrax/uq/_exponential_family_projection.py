#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._exponential_family import (
    AbstractExponentialFamily,
    EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT,
    EXPONENTIAL_FAMILY_INVALID_EVENT,
    EXPONENTIAL_FAMILY_NONFINITE,
    ExponentialFamilyLaw,
    MeanCoordinates,
)
from .._numerics._weighted_moments import (
    _canonical_values,
    LogWeightedAccumulator,
    WeightedMomentsDiagnostics,
)
from .._strict import StrictModule


class ExponentialFamilyEstimateResult(StrictModule):
    """Weighted sufficient-statistic projection with explicit MLE diagnostics."""

    law: ExponentialFamilyLaw
    mean_coordinates: MeanCoordinates
    statistic_standard_error: Array
    diagnostics: WeightedMomentsDiagnostics
    conversion_residual: Array
    conversion_iterations: Array
    valid: Array
    status: Array
    estimator_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        law: ExponentialFamilyLaw,
        mean_coordinates: MeanCoordinates,
        statistic_standard_error: ArrayLike,
        diagnostics: WeightedMomentsDiagnostics,
        conversion_residual: ArrayLike,
        conversion_iterations: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        estimator_id: str,
        method_id: str,
    ):
        if not estimator_id or not method_id:
            raise ValueError("Estimator provenance IDs must be non-empty.")
        shape = mean_coordinates.batch_shape
        if law.batch_shape != shape:
            raise ValueError("Estimated law and mean-coordinate batch shapes must match.")
        self.law = law
        self.mean_coordinates = mean_coordinates
        self.statistic_standard_error = jnp.asarray(statistic_standard_error)
        self.diagnostics = diagnostics
        self.conversion_residual = jnp.broadcast_to(
            jnp.asarray(conversion_residual), shape
        )
        self.conversion_iterations = jnp.broadcast_to(
            jnp.asarray(conversion_iterations, dtype=jnp.int32), shape
        )
        self.valid = jnp.broadcast_to(jnp.asarray(valid, dtype=bool), shape)
        self.status = jnp.broadcast_to(jnp.asarray(status, dtype=jnp.int32), shape)
        self.estimator_id = str(estimator_id)
        self.method_id = str(method_id)


class ExponentialFamilyProjectionAccumulator(StrictModule):
    """Mergeable log-weighted sufficient-statistic accumulator."""

    family: AbstractExponentialFamily
    moments: LogWeightedAccumulator
    nonfinite_observation_count: Array
    invalid_event_count: Array
    invalid_weight_count: Array
    minimum_log_weight: Array
    weighted_log_weight_sum: Array

    def __init__(
        self,
        *,
        family: AbstractExponentialFamily,
        moments: LogWeightedAccumulator,
        nonfinite_observation_count: ArrayLike,
        invalid_event_count: ArrayLike,
        invalid_weight_count: ArrayLike,
        minimum_log_weight: ArrayLike,
        weighted_log_weight_sum: ArrayLike,
    ):
        if not isinstance(family, AbstractExponentialFamily):
            raise TypeError("family must implement AbstractExponentialFamily.")
        batch_shape = moments.log_scale.shape
        if moments.weighted_value_sum.shape != batch_shape + (
            family.signature.dimension,
        ):
            raise ValueError("Weighted moments do not match the family dimension.")
        self.family = family
        self.moments = moments
        self.nonfinite_observation_count = jnp.broadcast_to(
            jnp.asarray(nonfinite_observation_count, dtype=jnp.int32), batch_shape
        )
        self.invalid_event_count = jnp.broadcast_to(
            jnp.asarray(invalid_event_count, dtype=jnp.int32), batch_shape
        )
        self.invalid_weight_count = jnp.broadcast_to(
            jnp.asarray(invalid_weight_count, dtype=jnp.int32), batch_shape
        )
        self.minimum_log_weight = jnp.broadcast_to(
            jnp.asarray(minimum_log_weight), batch_shape
        )
        self.weighted_log_weight_sum = jnp.broadcast_to(
            jnp.asarray(weighted_log_weight_sum), batch_shape
        )

    @classmethod
    def from_log_weights(
        cls,
        family: AbstractExponentialFamily,
        observations: ArrayLike,
        log_weights: ArrayLike,
        /,
        *,
        sample_axes: int | tuple[int, ...] = 0,
        mask: ArrayLike | None = None,
    ) -> "ExponentialFamilyProjectionAccumulator":
        if not isinstance(family, AbstractExponentialFamily):
            raise TypeError("family must implement AbstractExponentialFamily.")
        observation_array = jnp.asarray(observations)
        weight_array = jnp.asarray(log_weights, dtype=float)
        event_shape = family.signature.event_shape
        expected_observation_shape = weight_array.shape + event_shape
        if observation_array.shape != expected_observation_shape:
            raise ValueError(
                "observations must have shape log_weights.shape + event_shape; "
                f"got {observation_array.shape} and {weight_array.shape}."
            )
        statistics = family.sufficient_statistics(observation_array)
        if (
            statistics.values.shape != weight_array.shape + (family.signature.dimension,)
            or statistics.valid.shape != weight_array.shape
        ):
            raise ValueError(
                "Family sufficient statistics do not align with the observation batch."
            )
        if event_shape:
            event_axes = tuple(
                range(observation_array.ndim - len(event_shape), observation_array.ndim)
            )
            observation_finite = jnp.all(jnp.isfinite(observation_array), axis=event_axes)
        else:
            observation_finite = jnp.isfinite(observation_array)
        packed = jnp.concatenate(
            (
                statistics.values,
                statistics.valid[..., None].astype(statistics.values.dtype),
                observation_finite[..., None].astype(statistics.values.dtype),
            ),
            axis=-1,
        )
        canonical, canonical_weights, included = _canonical_values(
            packed,
            weight_array,
            sample_axes,
            None if mask is None else jnp.asarray(mask, dtype=bool),
        )
        dimension = family.signature.dimension
        statistic_values = canonical[..., :dimension]
        event_valid = canonical[..., dimension] != 0.0
        finite_observation = canonical[..., dimension + 1] != 0.0
        invalid_weight = included & (
            jnp.isnan(canonical_weights) | jnp.isposinf(canonical_weights)
        )
        active = included & jnp.isfinite(canonical_weights)
        nonfinite_observation = active & ~finite_observation
        invalid_event = active & finite_observation & ~event_valid
        valid_active = active & event_valid & finite_observation
        moments = LogWeightedAccumulator.from_values(
            statistic_values,
            canonical_weights,
            sample_axes=0,
            mask=valid_active,
        )
        safe_scale = jnp.where(jnp.isfinite(moments.log_scale), moments.log_scale, 0.0)
        scaled_weights = jnp.where(
            valid_active,
            jnp.exp(canonical_weights - safe_scale[None, ...]),
            0.0,
        )
        safe_log_weights = jnp.where(valid_active, canonical_weights, 0.0)
        minimum_log_weight = jnp.min(
            jnp.where(valid_active, canonical_weights, jnp.inf),
            axis=0,
            initial=jnp.inf,
        )
        return cls(
            family=family,
            moments=moments,
            nonfinite_observation_count=jnp.sum(
                nonfinite_observation, axis=0, dtype=jnp.int32
            ),
            invalid_event_count=jnp.sum(invalid_event, axis=0, dtype=jnp.int32),
            invalid_weight_count=jnp.sum(invalid_weight, axis=0, dtype=jnp.int32),
            minimum_log_weight=minimum_log_weight,
            weighted_log_weight_sum=jnp.sum(scaled_weights * safe_log_weights, axis=0),
        )

    def merge(
        self, other: "ExponentialFamilyProjectionAccumulator", /
    ) -> "ExponentialFamilyProjectionAccumulator":
        if not isinstance(other, ExponentialFamilyProjectionAccumulator):
            raise TypeError("other must be an ExponentialFamilyProjectionAccumulator.")
        if self.family.signature.key != other.family.signature.key:
            raise ValueError(
                "Projection accumulators must use identical family signatures."
            )
        merged_moments = self.moments.merge(other.moments)
        scale = merged_moments.log_scale
        left_scale = jnp.where(
            jnp.isfinite(self.moments.log_scale), self.moments.log_scale, 0.0
        )
        right_scale = jnp.where(
            jnp.isfinite(other.moments.log_scale), other.moments.log_scale, 0.0
        )
        left_factor = jnp.where(
            jnp.isfinite(self.moments.log_scale), jnp.exp(left_scale - scale), 0.0
        )
        right_factor = jnp.where(
            jnp.isfinite(other.moments.log_scale), jnp.exp(right_scale - scale), 0.0
        )
        return ExponentialFamilyProjectionAccumulator(
            family=self.family,
            moments=merged_moments,
            nonfinite_observation_count=(
                self.nonfinite_observation_count + other.nonfinite_observation_count
            ),
            invalid_event_count=self.invalid_event_count + other.invalid_event_count,
            invalid_weight_count=self.invalid_weight_count + other.invalid_weight_count,
            minimum_log_weight=jnp.minimum(
                self.minimum_log_weight, other.minimum_log_weight
            ),
            weighted_log_weight_sum=(
                left_factor * self.weighted_log_weight_sum
                + right_factor * other.weighted_log_weight_sum
            ),
        )

    @property
    def diagnostics(self) -> WeightedMomentsDiagnostics:
        count = self.moments.count
        ess = self.moments.weight_ess
        positive_mass = self.moments.weight_sum > 0.0
        log_total_weight = self.moments.log_scale + jnp.log(
            jnp.maximum(self.moments.weight_sum, jnp.finfo(float).tiny)
        )
        mean_log_weight = self.weighted_log_weight_sum / jnp.maximum(
            self.moments.weight_sum, jnp.finfo(float).tiny
        )
        entropy = jnp.where(
            positive_mass,
            jnp.maximum(log_total_weight - mean_log_weight, 0.0),
            0.0,
        )
        coefficient = jnp.sqrt(
            jnp.maximum(
                count / jnp.maximum(ess, jnp.finfo(float).tiny) - 1.0,
                0.0,
            )
        )
        return WeightedMomentsDiagnostics(
            weight_ess=ess,
            relative_weight_ess=self.moments.relative_weight_ess,
            coefficient_of_variation=coefficient,
            maximum_normalized_weight=self.moments.maximum_normalized_weight,
            entropy=entropy,
            log_weight_range=jnp.where(
                positive_mass,
                self.moments.log_scale - self.minimum_log_weight,
                jnp.inf,
            ),
            finite_count=count,
        )

    def finalize(self) -> ExponentialFamilyEstimateResult:
        mean = self.family.mean(self.moments.normalized_mean)
        conversion = self.family.natural_from_mean(mean)
        law = self.family.law(conversion.natural)
        sufficient_weight = self.moments.weight_sum > 0.0
        no_nonfinite = (self.nonfinite_observation_count == 0) & (
            self.invalid_weight_count == 0
        )
        events_valid = self.invalid_event_count == 0
        valid = no_nonfinite & events_valid & sufficient_weight & conversion.valid
        status = conversion.status
        status = jnp.where(
            ~sufficient_weight,
            EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT,
            status,
        )
        status = jnp.where(
            ~events_valid,
            EXPONENTIAL_FAMILY_INVALID_EVENT,
            status,
        )
        status = jnp.where(
            ~no_nonfinite,
            EXPONENTIAL_FAMILY_NONFINITE,
            status,
        )
        return ExponentialFamilyEstimateResult(
            law=law,
            mean_coordinates=mean,
            statistic_standard_error=self.moments.normalized_standard_error,
            diagnostics=self.diagnostics,
            conversion_residual=conversion.residual,
            conversion_iterations=conversion.iterations,
            valid=valid,
            status=status,
            estimator_id="weighted_sufficient_statistics",
            method_id=conversion.method_id,
        )


def _observation_batch_shape(
    family: AbstractExponentialFamily,
    observations: ArrayLike,
    /,
) -> tuple[int, ...]:
    shape = jnp.asarray(observations).shape
    event_shape = family.signature.event_shape
    if event_shape:
        if len(shape) < len(event_shape) or shape[-len(event_shape) :] != event_shape:
            raise ValueError("Observation trailing dimensions do not match event_shape.")
        return tuple(shape[: -len(event_shape)])
    return tuple(shape)


def project_exponential_family(
    family: AbstractExponentialFamily,
    observations: ArrayLike,
    /,
    *,
    log_weights: ArrayLike | None = None,
    sample_axes: int | tuple[int, ...] = 0,
    mask: ArrayLike | None = None,
) -> ExponentialFamilyEstimateResult:
    """Project weighted observations onto a regular exponential family."""
    batch_shape = _observation_batch_shape(family, observations)
    configured_weights = (
        jnp.zeros(batch_shape, dtype=float)
        if log_weights is None
        else jnp.asarray(log_weights, dtype=float)
    )
    return ExponentialFamilyProjectionAccumulator.from_log_weights(
        family,
        observations,
        configured_weights,
        sample_axes=sample_axes,
        mask=mask,
    ).finalize()


def fit_exponential_family(
    family: AbstractExponentialFamily,
    observations: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    sample_axes: int | tuple[int, ...] = 0,
    mask: ArrayLike | None = None,
) -> ExponentialFamilyEstimateResult:
    """Fit a regular exponential family by weighted sufficient-statistic MLE."""
    batch_shape = _observation_batch_shape(family, observations)
    if weights is None:
        log_weights = jnp.zeros(batch_shape, dtype=float)
    else:
        weight_array = jnp.asarray(weights, dtype=float)
        log_weights = jnp.where(
            weight_array > 0.0,
            jnp.log(weight_array),
            jnp.where(weight_array == 0.0, -jnp.inf, jnp.nan),
        )
    return project_exponential_family(
        family,
        observations,
        log_weights=log_weights,
        sample_axes=sample_axes,
        mask=mask,
    )


__all__ = [
    "ExponentialFamilyEstimateResult",
    "ExponentialFamilyProjectionAccumulator",
    "fit_exponential_family",
    "project_exponential_family",
]
