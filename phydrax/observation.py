#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared, fixed-shape observation operators and scientific likelihoods.

Instrument geometry and sampling capacity live in immutable, content-addressed
plans. Prepared runtimes expose JAX-compatible forward and evaluation paths whose
typed results carry explicit finite, identifiable, and successful evidence.
"""

from __future__ import annotations

import math
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


class CoordinateLayout(StrictModule, NonTrainableState):
    labels: tuple[str, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, labels: tuple[str, ...], /):
        labels_ = tuple(str(label).strip() for label in labels)
        if (
            not labels_
            or len(set(labels_)) != len(labels_)
            or any(not label for label in labels_)
        ):
            raise ValueError("Coordinate labels must be non-empty and unique.")
        self.labels = labels_
        self.layout_id = canonical_fingerprint(
            {"kind": "coordinate-layout", "labels": list(labels_)}
        )

    @property
    def size(self) -> int:
        return len(self.labels)


class TheoryVector(StrictModule):
    values: Array
    layout: CoordinateLayout
    product_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, layout: CoordinateLayout, product_id: str, /):
        value = jnp.asarray(values)
        if value.shape != (layout.size,):
            raise ValueError("Observation product must match its coordinate layout.")
        product_id_ = str(product_id).strip()
        if not product_id_:
            raise ValueError("Observation product ID must be non-empty.")
        self.values = value
        self.layout = layout
        self.product_id = product_id_


ObservationProduct = TheoryVector


class LinearObservationPlan(StrictModule, NonTrainableState):
    matrix: Array
    source: CoordinateLayout
    target: CoordinateLayout
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        source: CoordinateLayout,
        target: CoordinateLayout,
        /,
    ):
        values = jax.lax.stop_gradient(jnp.asarray(matrix))
        if values.shape != (target.size, source.size):
            raise ValueError("Observation matrix shape must match layouts.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Observation matrix must be finite.",
        )
        self.matrix = values
        self.source = source
        self.target = target
        self.plan_id = canonical_fingerprint(
            {
                "kind": "linear-observation-plan",
                "source": source.layout_id,
                "target": target.layout_id,
                "matrix": array_tree_fingerprint(values),
            }
        )

    def apply(self, theory: TheoryVector, /) -> TheoryVector:
        if theory.layout.layout_id != self.source.layout_id:
            raise ValueError("Observation product layout does not match response source.")
        values = contract("oi,i->o", self.matrix, theory.values)
        return TheoryVector(
            values,
            self.target,
            canonical_fingerprint(
                {
                    "kind": "observed-product",
                    "parent": theory.product_id,
                    "plan": self.plan_id,
                }
            ),
        )


class PrecisionCovarianceAction(StrictModule, NonTrainableState):
    precision: Array
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        precision: ArrayLike,
        logdet_covariance: ArrayLike,
        layout: CoordinateLayout,
        /,
    ):
        matrix = jax.lax.stop_gradient(jnp.asarray(precision))
        logdet = jax.lax.stop_gradient(jnp.asarray(logdet_covariance, dtype=matrix.dtype))
        if matrix.shape != (layout.size, layout.size) or logdet.shape != ():
            raise ValueError("Precision/covariance determinant shapes are invalid.")
        matrix = eqx.error_if(
            matrix,
            jnp.any(~jnp.isfinite(matrix))
            | ~jnp.isfinite(logdet)
            | jnp.any(jnp.abs(matrix - matrix.T) > 1.0e-10)
            | jnp.any(jnp.diag(matrix) <= 0.0),
            "Precision action must be finite, symmetric, and positive on the diagonal.",
        )
        self.precision = matrix
        self.logdet_covariance = logdet
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "precision-covariance-action",
                "layout": layout.layout_id,
                "precision": array_tree_fingerprint(matrix),
                "logdet_covariance": array_tree_fingerprint(logdet),
            }
        )

    def quadratic(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual, dtype=self.precision.dtype)
        if value.shape != (self.layout.size,):
            raise ValueError("Residual must match covariance layout.")
        return contract("i,ij,j->", value, self.precision, value)


class CholeskyCovarianceAction(StrictModule, NonTrainableState):
    lower_cholesky: Array
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(self, lower_cholesky: ArrayLike, layout: CoordinateLayout, /):
        cholesky = jax.lax.stop_gradient(jnp.asarray(lower_cholesky))
        if cholesky.shape != (layout.size, layout.size):
            raise ValueError("Covariance Cholesky shape must match its layout.")
        cholesky = eqx.error_if(
            cholesky,
            jnp.any(~jnp.isfinite(cholesky))
            | jnp.any(jnp.triu(cholesky, 1) != 0.0)
            | jnp.any(jnp.diag(cholesky) <= 0.0),
            "Covariance Cholesky must be finite, lower triangular, and positive.",
        )
        self.lower_cholesky = cholesky
        self.logdet_covariance = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "cholesky-covariance-action",
                "layout": layout.layout_id,
                "cholesky": array_tree_fingerprint(cholesky),
            }
        )

    def whiten(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual, dtype=self.lower_cholesky.dtype)
        if value.shape != (self.layout.size,):
            raise ValueError("Residual must match covariance layout.")
        return jsp.linalg.solve_triangular(self.lower_cholesky, value, lower=True)

    def quadratic(self, residual: ArrayLike, /) -> Array:
        whitened = self.whiten(residual)
        return jnp.sum(whitened * whitened)


CovarianceAction = PrecisionCovarianceAction | CholeskyCovarianceAction


class CorrelatedGaussianResult(StrictModule):
    residual: Array
    quadratic: Array
    log_probability: Array
    finite: Array
    successful: Array


class CorrelatedGaussianPlan(StrictModule, NonTrainableState):
    data: Array
    observation: LinearObservationPlan
    covariance: CovarianceAction
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        data: ArrayLike,
        observation: LinearObservationPlan,
        covariance: CovarianceAction,
        /,
    ):
        values = jax.lax.stop_gradient(jnp.asarray(data))
        if values.shape != (observation.target.size,):
            raise ValueError("Observed data must match response target layout.")
        if covariance.layout.layout_id != observation.target.layout_id:
            raise ValueError("Covariance and response target layouts disagree.")
        values = eqx.error_if(
            values, jnp.any(~jnp.isfinite(values)), "Observed data must be finite."
        )
        self.data = values
        self.observation = observation
        self.covariance = covariance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "correlated-gaussian-plan",
                "observation": observation.plan_id,
                "covariance": covariance.action_id,
                "data": array_tree_fingerprint(values),
            }
        )

    def evaluate(self, theory: TheoryVector, /) -> CorrelatedGaussianResult:
        observed = self.observation.apply(theory)
        residual = self.data - observed.values
        quadratic = self.covariance.quadratic(residual)
        size = jnp.asarray(self.data.size, dtype=residual.dtype)
        log_probability = -0.5 * (
            quadratic
            + self.covariance.logdet_covariance
            + size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=residual.dtype))
        )
        finite = jnp.all(jnp.isfinite(residual)) & jnp.isfinite(log_probability)
        return CorrelatedGaussianResult(
            residual, quadratic, log_probability, finite, finite
        )


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _finite_positive(value: float, name: str, /) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _unit(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical unit label.")
    return value


def _floating_array(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return jnp.asarray(array, dtype=jnp.result_type(array.dtype, jnp.float32))


class MeanSquareDisplacementResult(StrictModule):
    """Lag-resolved mean-square displacement and validity evidence."""

    lag_times: Array
    values: Array
    pair_counts: Array
    finite: Array
    identifiable: Array
    successful: Array


class MeanSquareDisplacementPlan(StrictModule, NonTrainableState):
    """Fixed-capacity trajectory geometry for unbiased MSD estimates."""

    sample_count: int = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    max_lag: int = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    distance_unit: str = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int,
        spatial_dimension: int,
        max_lag: int,
        time_step: float,
        /,
        *,
        distance_unit: str = "m",
        time_unit: str = "s",
    ):
        samples = _positive_integer(sample_count, "sample_count")
        dimension = _positive_integer(spatial_dimension, "spatial_dimension")
        lag = _positive_integer(max_lag, "max_lag")
        if lag >= samples:
            raise ValueError("max_lag must be smaller than sample_count.")
        step = _finite_positive(time_step, "time_step")
        distance = _unit(distance_unit, "distance_unit")
        time = _unit(time_unit, "time_unit")
        self.sample_count = samples
        self.spatial_dimension = dimension
        self.max_lag = lag
        self.time_step = step
        self.distance_unit = distance
        self.time_unit = time
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mean-square-displacement-plan",
                "sample_count": samples,
                "spatial_dimension": dimension,
                "max_lag": lag,
                "time_step": step,
                "distance_unit": distance,
                "time_unit": time,
            }
        )

    def prepare(self) -> PreparedMeanSquareDisplacement:
        return PreparedMeanSquareDisplacement(self)


class PreparedMeanSquareDisplacement(StrictModule, NonTrainableState):
    """Prepared pair indices for one fixed-shape MSD evaluation."""

    plan: MeanSquareDisplacementPlan
    source_indices: Array
    target_indices: Array
    valid_pairs: Array
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: MeanSquareDisplacementPlan, /):
        if not isinstance(plan, MeanSquareDisplacementPlan):
            raise TypeError("plan must be a MeanSquareDisplacementPlan.")
        lag = jnp.arange(plan.max_lag + 1, dtype=jnp.int32)[:, None]
        source = jnp.arange(plan.sample_count, dtype=jnp.int32)[None, :]
        valid = source + lag < plan.sample_count
        target = jnp.minimum(source + lag, plan.sample_count - 1)
        self.plan = plan
        self.source_indices = jnp.broadcast_to(source, target.shape)
        self.target_indices = target
        self.valid_pairs = valid
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-mean-square-displacement", "plan": plan.plan_id}
        )

    def forward(self, positions: ArrayLike, /) -> MeanSquareDisplacementResult:
        value = _floating_array(positions)
        expected = (self.plan.sample_count, self.plan.spatial_dimension)
        if value.shape != expected:
            raise ValueError(
                "positions must match the planned sample and dimension shape."
            )
        displacement = value[self.target_indices, :] - value[self.source_indices, :]
        squared = jnp.sum(displacement * displacement, axis=-1)
        counts = jnp.sum(self.valid_pairs, axis=1)
        values = jnp.sum(
            jnp.where(self.valid_pairs, squared, jnp.zeros_like(squared)), axis=1
        ) / counts.astype(value.dtype)
        lag_times = (
            jnp.arange(self.plan.max_lag + 1, dtype=value.dtype) * self.plan.time_step
        )
        finite = jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(values))
        identifiable = jnp.all(counts > 0)
        successful = finite & identifiable
        return MeanSquareDisplacementResult(
            lag_times, values, counts, finite, identifiable, successful
        )


class AutocorrelationResult(StrictModule):
    """Lag-resolved scalar autocorrelation and normalization evidence."""

    lag_times: Array
    values: Array
    pair_counts: Array
    mean: Array
    variance: Array
    finite: Array
    identifiable: Array
    successful: Array


class AutocorrelationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity, sample-mean-centered autocorrelation plan."""

    sample_count: int = eqx.field(static=True)
    max_lag: int = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    signal_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int,
        max_lag: int,
        time_step: float,
        /,
        *,
        normalized: bool = True,
        time_unit: str = "s",
        signal_unit: str = "1",
    ):
        samples = _positive_integer(sample_count, "sample_count")
        lag = _positive_integer(max_lag, "max_lag")
        if lag >= samples:
            raise ValueError("max_lag must be smaller than sample_count.")
        step = _finite_positive(time_step, "time_step")
        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a boolean.")
        time = _unit(time_unit, "time_unit")
        signal = _unit(signal_unit, "signal_unit")
        self.sample_count = samples
        self.max_lag = lag
        self.time_step = step
        self.normalized = normalized
        self.time_unit = time
        self.signal_unit = signal
        self.plan_id = canonical_fingerprint(
            {
                "kind": "autocorrelation-plan",
                "sample_count": samples,
                "max_lag": lag,
                "time_step": step,
                "normalized": normalized,
                "time_unit": time,
                "signal_unit": signal,
            }
        )

    def prepare(self) -> PreparedAutocorrelation:
        return PreparedAutocorrelation(self)


class PreparedAutocorrelation(StrictModule, NonTrainableState):
    """Prepared sample-pair geometry for autocorrelation evaluation."""

    plan: AutocorrelationPlan
    source_indices: Array
    target_indices: Array
    valid_pairs: Array
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: AutocorrelationPlan, /):
        if not isinstance(plan, AutocorrelationPlan):
            raise TypeError("plan must be an AutocorrelationPlan.")
        lag = jnp.arange(plan.max_lag + 1, dtype=jnp.int32)[:, None]
        source = jnp.arange(plan.sample_count, dtype=jnp.int32)[None, :]
        valid = source + lag < plan.sample_count
        target = jnp.minimum(source + lag, plan.sample_count - 1)
        self.plan = plan
        self.source_indices = jnp.broadcast_to(source, target.shape)
        self.target_indices = target
        self.valid_pairs = valid
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-autocorrelation", "plan": plan.plan_id}
        )

    def forward(self, signal: ArrayLike, /) -> AutocorrelationResult:
        value = _floating_array(signal)
        if value.shape != (self.plan.sample_count,):
            raise ValueError("signal must match the planned sample capacity.")
        mean = jnp.mean(value)
        centered = value - mean
        scale = jnp.max(jnp.abs(centered))
        scale_valid = jnp.isfinite(scale) & (scale > 0.0)
        scaled = centered / jnp.where(scale_valid, scale, jnp.ones_like(scale))
        scaled_products = scaled[self.source_indices] * scaled[self.target_indices]
        counts = jnp.sum(self.valid_pairs, axis=1)
        scaled_covariance = jnp.sum(
            jnp.where(
                self.valid_pairs,
                scaled_products,
                jnp.zeros_like(scaled_products),
            ),
            axis=1,
        ) / counts.astype(value.dtype)
        variance_scaled = scaled_covariance[0]
        covariance = scaled_covariance * scale * scale
        variance = covariance[0]
        normalizable = (
            scale_valid & jnp.isfinite(variance_scaled) & (variance_scaled > 0.0)
        )
        if self.plan.normalized:
            values = scaled_covariance / jnp.where(
                normalizable, variance_scaled, jnp.ones_like(variance_scaled)
            )
            values = jnp.where(normalizable, values, jnp.full_like(values, jnp.nan))
            identifiable = normalizable
        else:
            values = covariance
            identifiable = jnp.isfinite(variance)
        lag_times = (
            jnp.arange(self.plan.max_lag + 1, dtype=value.dtype) * self.plan.time_step
        )
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.isfinite(mean)
            & jnp.isfinite(variance)
            & jnp.all(jnp.isfinite(values))
        )
        successful = finite & identifiable
        return AutocorrelationResult(
            lag_times,
            values,
            counts,
            mean,
            variance,
            finite,
            identifiable,
            successful,
        )


class FluorescenceCorrelationResult(StrictModule):
    """Fluorescence fluctuation correlation with brightness evidence."""

    lag_times: Array
    correlation: Array
    pair_counts: Array
    mean_intensity: Array
    finite: Array
    identifiable: Array
    successful: Array


class FluorescenceCorrelationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity fluorescence correlation spectroscopy plan."""

    sample_count: int = eqx.field(static=True)
    max_lag: int = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    intensity_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int,
        max_lag: int,
        time_step: float,
        /,
        *,
        time_unit: str = "s",
        intensity_unit: str = "count/s",
    ):
        samples = _positive_integer(sample_count, "sample_count")
        lag = _positive_integer(max_lag, "max_lag")
        if lag >= samples:
            raise ValueError("max_lag must be smaller than sample_count.")
        step = _finite_positive(time_step, "time_step")
        time = _unit(time_unit, "time_unit")
        intensity = _unit(intensity_unit, "intensity_unit")
        self.sample_count = samples
        self.max_lag = lag
        self.time_step = step
        self.time_unit = time
        self.intensity_unit = intensity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fluorescence-correlation-plan",
                "sample_count": samples,
                "max_lag": lag,
                "time_step": step,
                "time_unit": time,
                "intensity_unit": intensity,
            }
        )

    def prepare(self) -> PreparedFluorescenceCorrelation:
        return PreparedFluorescenceCorrelation(self)


class PreparedFluorescenceCorrelation(StrictModule, NonTrainableState):
    """Prepared lag geometry for fluorescence fluctuation correlation."""

    plan: FluorescenceCorrelationPlan
    source_indices: Array
    target_indices: Array
    valid_pairs: Array
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: FluorescenceCorrelationPlan, /):
        if not isinstance(plan, FluorescenceCorrelationPlan):
            raise TypeError("plan must be a FluorescenceCorrelationPlan.")
        lag = jnp.arange(plan.max_lag + 1, dtype=jnp.int32)[:, None]
        source = jnp.arange(plan.sample_count, dtype=jnp.int32)[None, :]
        valid = source + lag < plan.sample_count
        target = jnp.minimum(source + lag, plan.sample_count - 1)
        self.plan = plan
        self.source_indices = jnp.broadcast_to(source, target.shape)
        self.target_indices = target
        self.valid_pairs = valid
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-fluorescence-correlation", "plan": plan.plan_id}
        )

    def forward(
        self, intensity: ArrayLike, /, *, background: ArrayLike = 0.0
    ) -> FluorescenceCorrelationResult:
        value = _floating_array(intensity)
        if value.shape != (self.plan.sample_count,):
            raise ValueError("intensity must match the planned sample capacity.")
        background_ = jnp.asarray(background, dtype=value.dtype)
        if background_.shape != ():
            raise ValueError("background must be scalar.")
        corrected = value - background_
        mean = jnp.mean(corrected)
        scale = jnp.max(jnp.abs(corrected))
        scale_valid = jnp.isfinite(scale) & (scale > 0.0)
        scaled = corrected / jnp.where(scale_valid, scale, jnp.ones_like(scale))
        mean_scaled = jnp.mean(scaled)
        fluctuation = scaled - mean_scaled
        products = fluctuation[self.source_indices] * fluctuation[self.target_indices]
        counts = jnp.sum(self.valid_pairs, axis=1)
        covariance = jnp.sum(
            jnp.where(self.valid_pairs, products, jnp.zeros_like(products)), axis=1
        ) / counts.astype(value.dtype)
        identifiable = scale_valid & jnp.isfinite(mean_scaled) & (mean_scaled > 0.0)
        denominator = jnp.where(
            identifiable, mean_scaled * mean_scaled, jnp.ones_like(mean_scaled)
        )
        correlation = covariance / denominator
        correlation = jnp.where(
            identifiable, correlation, jnp.full_like(correlation, jnp.nan)
        )
        lag_times = (
            jnp.arange(self.plan.max_lag + 1, dtype=value.dtype) * self.plan.time_step
        )
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.isfinite(background_)
            & jnp.all(value >= 0.0)
            & jnp.all(jnp.isfinite(correlation))
        )
        successful = finite & identifiable
        return FluorescenceCorrelationResult(
            lag_times,
            correlation,
            counts,
            mean,
            finite,
            identifiable,
            successful,
        )


class PairCorrelationResult(StrictModule):
    """Signed-lag pair correlation and directional peak evidence."""

    lag_times: Array
    correlation: Array
    pair_counts: Array
    peak_lag: Array
    peak_correlation: Array
    directionality: Array
    finite: Array
    identifiable: Array
    successful: Array


class PairCorrelationPlan(StrictModule, NonTrainableState):
    """Cross-channel plan where positive lag means the first channel leads."""

    sample_count: int = eqx.field(static=True)
    max_lag: int = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int,
        max_lag: int,
        time_step: float,
        /,
        *,
        time_unit: str = "s",
    ):
        samples = _positive_integer(sample_count, "sample_count")
        lag = _positive_integer(max_lag, "max_lag")
        if lag >= samples:
            raise ValueError("max_lag must be smaller than sample_count.")
        step = _finite_positive(time_step, "time_step")
        time = _unit(time_unit, "time_unit")
        self.sample_count = samples
        self.max_lag = lag
        self.time_step = step
        self.time_unit = time
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pair-correlation-plan",
                "sample_count": samples,
                "max_lag": lag,
                "time_step": step,
                "time_unit": time,
            }
        )

    def prepare(self) -> PreparedPairCorrelation:
        return PreparedPairCorrelation(self)


class PreparedPairCorrelation(StrictModule, NonTrainableState):
    """Prepared signed-lag channel-pair geometry."""

    plan: PairCorrelationPlan
    signed_lags: Array
    source_indices: Array
    target_indices: Array
    valid_pairs: Array
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: PairCorrelationPlan, /):
        if not isinstance(plan, PairCorrelationPlan):
            raise TypeError("plan must be a PairCorrelationPlan.")
        lag = jnp.arange(-plan.max_lag, plan.max_lag + 1, dtype=jnp.int32)[:, None]
        sample = jnp.arange(plan.sample_count, dtype=jnp.int32)[None, :]
        magnitude = jnp.abs(lag)
        valid = sample < plan.sample_count - magnitude
        source = jnp.minimum(
            jnp.where(lag >= 0, sample, sample + magnitude), plan.sample_count - 1
        )
        target = jnp.minimum(
            jnp.where(lag >= 0, sample + magnitude, sample), plan.sample_count - 1
        )
        self.plan = plan
        self.signed_lags = lag[:, 0]
        self.source_indices = source
        self.target_indices = target
        self.valid_pairs = valid
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-pair-correlation", "plan": plan.plan_id}
        )

    def forward(
        self, leading_channel: ArrayLike, trailing_channel: ArrayLike, /
    ) -> PairCorrelationResult:
        leading = _floating_array(leading_channel)
        trailing = jnp.asarray(trailing_channel, dtype=leading.dtype)
        expected = (self.plan.sample_count,)
        if leading.shape != expected or trailing.shape != expected:
            raise ValueError("Both channels must match the planned sample capacity.")
        centered_leading = leading - jnp.mean(leading)
        centered_trailing = trailing - jnp.mean(trailing)
        leading_scale = jnp.max(jnp.abs(centered_leading))
        trailing_scale = jnp.max(jnp.abs(centered_trailing))
        scales_valid = (
            jnp.isfinite(leading_scale)
            & (leading_scale > 0.0)
            & jnp.isfinite(trailing_scale)
            & (trailing_scale > 0.0)
        )
        scaled_leading = centered_leading / jnp.where(
            scales_valid, leading_scale, jnp.ones_like(leading_scale)
        )
        scaled_trailing = centered_trailing / jnp.where(
            scales_valid, trailing_scale, jnp.ones_like(trailing_scale)
        )
        products = (
            scaled_leading[self.source_indices] * scaled_trailing[self.target_indices]
        )
        counts = jnp.sum(self.valid_pairs, axis=1)
        covariance = jnp.sum(
            jnp.where(self.valid_pairs, products, jnp.zeros_like(products)), axis=1
        ) / counts.astype(leading.dtype)
        variance_product = jnp.mean(scaled_leading * scaled_leading) * jnp.mean(
            scaled_trailing * scaled_trailing
        )
        denominator = jnp.sqrt(jnp.maximum(variance_product, 0.0))
        normalizable = scales_valid & jnp.isfinite(denominator) & (denominator > 0.0)
        correlation = covariance / jnp.where(
            normalizable, denominator, jnp.ones_like(denominator)
        )
        correlation = jnp.where(
            normalizable, correlation, jnp.full_like(correlation, jnp.nan)
        )
        directional = self.signed_lags != 0
        candidates = jnp.where(
            directional & jnp.isfinite(correlation), correlation, -jnp.inf
        )
        peak_index = jnp.argmax(candidates)
        peak_value = candidates[peak_index]
        tie_tolerance = (
            32.0 * jnp.finfo(leading.dtype).eps * jnp.maximum(1.0, jnp.abs(peak_value))
        )
        unique_peak = (
            jnp.sum(
                directional
                & jnp.isfinite(correlation)
                & (jnp.abs(correlation - peak_value) <= tie_tolerance)
            )
            == 1
        )
        peak_steps = self.signed_lags[peak_index]
        magnitude = jnp.abs(peak_steps)
        positive_index = self.plan.max_lag + magnitude
        negative_index = self.plan.max_lag - magnitude
        directionality = correlation[positive_index] - correlation[negative_index]
        asymmetric = jnp.abs(directionality) > tie_tolerance
        lag_times = self.signed_lags.astype(leading.dtype) * self.plan.time_step
        input_finite = jnp.all(jnp.isfinite(leading)) & jnp.all(jnp.isfinite(trailing))
        identifiable = (
            input_finite
            & normalizable
            & unique_peak
            & asymmetric
            & jnp.any(directional & jnp.isfinite(correlation))
        )
        peak_lag = jnp.where(
            identifiable,
            peak_steps.astype(leading.dtype) * self.plan.time_step,
            jnp.asarray(jnp.nan, dtype=leading.dtype),
        )
        peak_correlation = jnp.where(
            identifiable,
            correlation[peak_index],
            jnp.asarray(jnp.nan, dtype=leading.dtype),
        )
        directionality = jnp.where(
            identifiable,
            directionality,
            jnp.asarray(jnp.nan, dtype=leading.dtype),
        )
        finite = input_finite & jnp.all(jnp.isfinite(correlation))
        successful = finite & identifiable
        return PairCorrelationResult(
            lag_times,
            correlation,
            counts,
            peak_lag,
            peak_correlation,
            directionality,
            finite,
            identifiable,
            successful,
        )


class DiffusionForwardResult(StrictModule):
    """Analytic anomalous or confined-diffusion MSD prediction."""

    lag_times: Array
    mean_squared_displacement: Array
    finite: Array
    identifiable: Array
    successful: Array


class DiffusionEvaluationResult(StrictModule):
    """Observed-MSD residual and Gaussian goodness-of-fit evidence."""

    prediction: Array
    residual: Array
    chi_square: Array
    log_probability: Array
    finite: Array
    identifiable: Array
    successful: Array


class DiffusionModelPlan(StrictModule, NonTrainableState):
    """Analytic MSD plan for anomalous power laws or confined Ornstein-Uhlenbeck motion."""

    lag_times: Array
    spatial_dimension: int = eqx.field(static=True)
    model: str = eqx.field(static=True)
    distance_unit: str = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lag_times: ArrayLike,
        spatial_dimension: int,
        model: str,
        /,
        *,
        distance_unit: str = "m",
        time_unit: str = "s",
    ):
        lag = jax.lax.stop_gradient(_floating_array(lag_times))
        if lag.ndim != 1 or lag.size < 2:
            raise ValueError(
                "lag_times must be a one-dimensional array with two or more entries."
            )
        dimension = _positive_integer(spatial_dimension, "spatial_dimension")
        if not isinstance(model, str):
            raise TypeError("model must be a string.")
        model_ = model
        if model_ not in ("anomalous", "confined"):
            raise ValueError("model must be 'anomalous' or 'confined'.")
        distance = _unit(distance_unit, "distance_unit")
        time = _unit(time_unit, "time_unit")
        lag = eqx.error_if(
            lag,
            jnp.any(~jnp.isfinite(lag))
            | jnp.any(lag < 0.0)
            | jnp.any(jnp.diff(lag) <= 0.0),
            "lag_times must be finite, nonnegative, and strictly increasing.",
        )
        self.lag_times = lag
        self.spatial_dimension = dimension
        self.model = model_
        self.distance_unit = distance
        self.time_unit = time
        self.plan_id = canonical_fingerprint(
            {
                "kind": "diffusion-model-plan",
                "lag_times": array_tree_fingerprint(lag),
                "spatial_dimension": dimension,
                "model": model_,
                "distance_unit": distance,
                "time_unit": time,
            }
        )

    def prepare(self) -> PreparedDiffusionModel:
        return PreparedDiffusionModel(self)


class PreparedDiffusionModel(StrictModule, NonTrainableState):
    """Prepared analytic diffusion forward model and likelihood."""

    plan: DiffusionModelPlan
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: DiffusionModelPlan, /):
        if not isinstance(plan, DiffusionModelPlan):
            raise TypeError("plan must be a DiffusionModelPlan.")
        self.plan = plan
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-diffusion-model", "plan": plan.plan_id}
        )

    def forward(
        self,
        diffusion_coefficient: ArrayLike,
        /,
        *,
        exponent: ArrayLike = 1.0,
        confinement_time: ArrayLike = 1.0,
        localization_variance: ArrayLike = 0.0,
    ) -> DiffusionForwardResult:
        dtype = self.plan.lag_times.dtype
        diffusion = jnp.asarray(diffusion_coefficient, dtype=dtype)
        exponent_ = jnp.asarray(exponent, dtype=dtype)
        confinement = jnp.asarray(confinement_time, dtype=dtype)
        localization = jnp.asarray(localization_variance, dtype=dtype)
        if any(
            value.shape != ()
            for value in (diffusion, exponent_, confinement, localization)
        ):
            raise ValueError("Diffusion-model parameters must be scalar.")
        base_valid = (
            jnp.isfinite(diffusion)
            & (diffusion >= 0.0)
            & jnp.isfinite(localization)
            & (localization >= 0.0)
        )
        dimension = jnp.asarray(self.plan.spatial_dimension, dtype=dtype)
        localization_offset = jnp.where(
            self.plan.lag_times > 0.0,
            2.0 * dimension * localization,
            jnp.asarray(0.0, dtype=dtype),
        )
        positive_lag = self.plan.lag_times > 0.0
        safe_lag = jnp.where(
            positive_lag, self.plan.lag_times, jnp.ones_like(self.plan.lag_times)
        )
        if self.plan.model == "anomalous":
            parameter_valid = (
                jnp.isfinite(exponent_) & (exponent_ > 0.0) & (exponent_ <= 2.0)
            )
            prediction = (
                2.0 * dimension * diffusion * jnp.power(self.plan.lag_times, exponent_)
                + localization_offset
            )
            anomalous_basis = jnp.power(safe_lag, exponent_)
            first_sensitivity = jnp.where(
                positive_lag, anomalous_basis, jnp.zeros_like(anomalous_basis)
            )
            second_sensitivity = jnp.where(
                positive_lag,
                diffusion * anomalous_basis * jnp.log(safe_lag),
                jnp.zeros_like(anomalous_basis),
            )
        else:
            parameter_valid = jnp.isfinite(confinement) & (confinement > 0.0)
            safe_confinement = jnp.where(
                parameter_valid, confinement, jnp.ones_like(confinement)
            )
            prediction = (
                2.0
                * dimension
                * diffusion
                * safe_confinement
                * (-jnp.expm1(-self.plan.lag_times / safe_confinement))
                + localization_offset
            )
            scaled_lag = self.plan.lag_times / safe_confinement
            confined_response = -jnp.expm1(-scaled_lag)
            first_sensitivity = jnp.where(
                positive_lag,
                safe_confinement * confined_response,
                jnp.zeros_like(confined_response),
            )
            second_sensitivity = jnp.where(
                positive_lag,
                diffusion * (confined_response - scaled_lag * jnp.exp(-scaled_lag)),
                jnp.zeros_like(confined_response),
            )
        valid = base_valid & parameter_valid
        first_scale = jnp.max(jnp.abs(first_sensitivity))
        second_scale = jnp.max(jnp.abs(second_sensitivity))
        scaled_first = first_sensitivity / jnp.where(
            first_scale > 0.0, first_scale, jnp.ones_like(first_scale)
        )
        scaled_second = second_sensitivity / jnp.where(
            second_scale > 0.0, second_scale, jnp.ones_like(second_scale)
        )
        first_norm = jnp.sqrt(jnp.sum(scaled_first * scaled_first))
        second_norm = jnp.sqrt(jnp.sum(scaled_second * scaled_second))
        cosine = jnp.sum(scaled_first * scaled_second) / jnp.where(
            (first_norm > 0.0) & (second_norm > 0.0),
            first_norm * second_norm,
            jnp.ones_like(first_norm),
        )
        rank_residual = jnp.maximum(0.0, 1.0 - cosine * cosine)
        shape_identifiable = (
            jnp.isfinite(first_scale)
            & (first_scale > 0.0)
            & jnp.isfinite(second_scale)
            & (second_scale > 0.0)
            & jnp.isfinite(rank_residual)
            & (rank_residual > 64.0 * jnp.finfo(dtype).eps)
        )
        prediction = jnp.where(valid, prediction, jnp.full_like(prediction, jnp.nan))
        finite = valid & jnp.all(jnp.isfinite(prediction))
        identifiable = finite & (diffusion > 0.0) & shape_identifiable
        successful = finite & identifiable
        return DiffusionForwardResult(
            self.plan.lag_times, prediction, finite, identifiable, successful
        )

    def evaluate(
        self,
        observed_msd: ArrayLike,
        standard_error: ArrayLike,
        diffusion_coefficient: ArrayLike,
        /,
        *,
        exponent: ArrayLike = 1.0,
        confinement_time: ArrayLike = 1.0,
        localization_variance: ArrayLike = 0.0,
    ) -> DiffusionEvaluationResult:
        observed = jnp.asarray(observed_msd, dtype=self.plan.lag_times.dtype)
        uncertainty = jnp.asarray(standard_error, dtype=self.plan.lag_times.dtype)
        if observed.shape != self.plan.lag_times.shape:
            raise ValueError("observed_msd must match the planned lag shape.")
        if uncertainty.shape not in ((), self.plan.lag_times.shape):
            raise ValueError("standard_error must be scalar or match the lag shape.")
        forward = self.forward(
            diffusion_coefficient,
            exponent=exponent,
            confinement_time=confinement_time,
            localization_variance=localization_variance,
        )
        uncertainty = jnp.broadcast_to(uncertainty, observed.shape)
        uncertainty_valid = jnp.all(jnp.isfinite(uncertainty)) & jnp.all(
            uncertainty > 0.0
        )
        residual = observed - forward.mean_squared_displacement
        scaled = residual / jnp.where(
            uncertainty_valid, uncertainty, jnp.ones_like(uncertainty)
        )
        chi_square = jnp.sum(scaled * scaled)
        log_probability = -0.5 * jnp.sum(
            scaled * scaled + 2.0 * jnp.log(uncertainty) + jnp.log(2.0 * jnp.pi)
        )
        finite = (
            forward.finite
            & uncertainty_valid
            & jnp.all(jnp.isfinite(observed))
            & jnp.isfinite(log_probability)
        )
        identifiable = finite & forward.identifiable
        successful = finite & identifiable
        return DiffusionEvaluationResult(
            forward.mean_squared_displacement,
            residual,
            chi_square,
            log_probability,
            finite,
            identifiable,
            successful,
        )


class BrightnessConditionedTransportResult(StrictModule):
    """Brightness-bin transport estimates and per-bin identifiability."""

    bin_centers: Array
    counts: Array
    mean_squared_displacement: Array
    diffusion_coefficient: Array
    identifiable_bins: Array
    finite: Array
    identifiable: Array
    successful: Array


class BrightnessConditionedTransportPlan(StrictModule, NonTrainableState):
    """Fixed-capacity brightness-conditioned single-step transport plan."""

    brightness_edges: Array
    sample_capacity: int = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    minimum_count: int = eqx.field(static=True)
    brightness_unit: str = eqx.field(static=True)
    distance_unit: str = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        brightness_edges: ArrayLike,
        sample_capacity: int,
        spatial_dimension: int,
        time_step: float,
        /,
        *,
        minimum_count: int = 2,
        brightness_unit: str = "count/s",
        distance_unit: str = "m",
        time_unit: str = "s",
    ):
        edges = jax.lax.stop_gradient(_floating_array(brightness_edges))
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                "brightness_edges must contain at least two one-dimensional edges."
            )
        capacity = _positive_integer(sample_capacity, "sample_capacity")
        dimension = _positive_integer(spatial_dimension, "spatial_dimension")
        step = _finite_positive(time_step, "time_step")
        count = _positive_integer(minimum_count, "minimum_count")
        brightness = _unit(brightness_unit, "brightness_unit")
        distance = _unit(distance_unit, "distance_unit")
        time = _unit(time_unit, "time_unit")
        edges = eqx.error_if(
            edges,
            jnp.any(~jnp.isfinite(edges))
            | jnp.any(edges < 0.0)
            | jnp.any(jnp.diff(edges) <= 0.0),
            "brightness_edges must be finite, nonnegative, and strictly increasing.",
        )
        self.brightness_edges = edges
        self.sample_capacity = capacity
        self.spatial_dimension = dimension
        self.time_step = step
        self.minimum_count = count
        self.brightness_unit = brightness
        self.distance_unit = distance
        self.time_unit = time
        self.plan_id = canonical_fingerprint(
            {
                "kind": "brightness-conditioned-transport-plan",
                "brightness_edges": array_tree_fingerprint(edges),
                "sample_capacity": capacity,
                "spatial_dimension": dimension,
                "time_step": step,
                "minimum_count": count,
                "brightness_unit": brightness,
                "distance_unit": distance,
                "time_unit": time,
            }
        )

    def prepare(self) -> PreparedBrightnessConditionedTransport:
        return PreparedBrightnessConditionedTransport(self)


class PreparedBrightnessConditionedTransport(StrictModule, NonTrainableState):
    """Prepared bin geometry for brightness-conditioned transport."""

    plan: BrightnessConditionedTransportPlan
    bin_centers: Array
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: BrightnessConditionedTransportPlan, /):
        if not isinstance(plan, BrightnessConditionedTransportPlan):
            raise TypeError("plan must be a BrightnessConditionedTransportPlan.")
        self.plan = plan
        self.bin_centers = 0.5 * (plan.brightness_edges[:-1] + plan.brightness_edges[1:])
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-brightness-conditioned-transport", "plan": plan.plan_id}
        )

    def evaluate(
        self,
        brightness: ArrayLike,
        displacements: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
    ) -> BrightnessConditionedTransportResult:
        brightness_ = jnp.asarray(brightness, dtype=self.plan.brightness_edges.dtype)
        displacement = jnp.asarray(displacements, dtype=brightness_.dtype)
        if brightness_.shape != (self.plan.sample_capacity,):
            raise ValueError("brightness must match the planned sample capacity.")
        if displacement.shape != (
            self.plan.sample_capacity,
            self.plan.spatial_dimension,
        ):
            raise ValueError(
                "displacements must match the planned sample and dimension shape."
            )
        if active is None:
            active_ = jnp.ones((self.plan.sample_capacity,), dtype=jnp.bool_)
            active_valid = jnp.asarray(True)
        else:
            active_ = jnp.asarray(active)
            if active_.shape != (self.plan.sample_capacity,):
                raise ValueError("active must match the planned sample capacity.")
            active_valid = (active_ == 0) | (active_ == 1)
            active_ = active_.astype(jnp.bool_)
        lower = self.plan.brightness_edges[:-1, None]
        upper = self.plan.brightness_edges[1:, None]
        membership = (brightness_[None, :] >= lower) & (
            (brightness_[None, :] < upper)
            | (
                (jnp.arange(self.bin_centers.size)[:, None] == self.bin_centers.size - 1)
                & (brightness_[None, :] <= upper)
            )
        )
        membership = membership & active_[None, :]
        assigned = jnp.sum(membership, axis=0) == 1
        counts = jnp.sum(membership, axis=1)
        squared = contract("nd,nd->n", displacement, displacement)
        squared = jnp.where(active_, squared, jnp.zeros_like(squared))
        sums = contract("bn,n->b", membership.astype(brightness_.dtype), squared)
        safe_counts = jnp.maximum(counts, 1).astype(brightness_.dtype)
        mean_squared = sums / safe_counts
        mean_squared = jnp.where(
            counts > 0, mean_squared, jnp.full_like(mean_squared, jnp.nan)
        )
        denominator = (
            2.0
            * jnp.asarray(self.plan.spatial_dimension, dtype=brightness_.dtype)
            * self.plan.time_step
        )
        diffusion = mean_squared / denominator
        identifiable_bins = counts >= self.plan.minimum_count
        input_finite = (
            jnp.all(
                jnp.where(
                    active_,
                    jnp.isfinite(brightness_) & (brightness_ >= 0.0),
                    True,
                )
            )
            & jnp.all(jnp.where(active_[:, None], jnp.isfinite(displacement), True))
            & jnp.all(active_valid)
            & jnp.all(jnp.where(active_, assigned, True))
        )
        finite = input_finite & jnp.all(jnp.isfinite(diffusion))
        identifiable = jnp.all(identifiable_bins)
        successful = finite & identifiable
        return BrightnessConditionedTransportResult(
            self.bin_centers,
            counts,
            mean_squared,
            diffusion,
            identifiable_bins,
            finite,
            identifiable,
            successful,
        )


class FluorescencePhotonExpectation(StrictModule):
    """Instrument-convolved fluorescence-lifetime photon expectation."""

    effective_lifetime: Array
    intrinsic_probability: Array
    detected_probability: Array
    expected_counts: Array
    fret_efficiency: Array
    finite: Array
    identifiable: Array
    successful: Array


class FluorescencePhotonResult(StrictModule):
    """Expected and reproducibly Poisson-sampled photon histogram."""

    expected_counts: Array
    photon_counts: Array
    effective_lifetime: Array
    fret_efficiency: Array
    finite: Array
    identifiable: Array
    successful: Array


class FluorescencePhotonEvaluation(StrictModule):
    """Poisson fluorescence-histogram likelihood and evidence."""

    expected_counts: Array
    residual: Array
    log_likelihood: Array
    finite: Array
    identifiable: Array
    successful: Array


class FluorescencePhotonPlan(StrictModule, NonTrainableState):
    """TCSPC lifetime/FRET plan with a discrete instrument response function."""

    bin_edges: Array
    instrument_response: Array
    time_unit: str = eqx.field(static=True)
    count_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bin_edges: ArrayLike,
        instrument_response: ArrayLike,
        /,
        *,
        time_unit: str = "s",
        count_unit: str = "photon",
    ):
        edges = jax.lax.stop_gradient(_floating_array(bin_edges))
        response = jax.lax.stop_gradient(
            jnp.asarray(instrument_response, dtype=edges.dtype)
        )
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges must contain at least two one-dimensional edges.")
        if response.shape != (edges.size - 1,):
            raise ValueError("instrument_response must contain one value per time bin.")
        time = _unit(time_unit, "time_unit")
        count = _unit(count_unit, "count_unit")
        widths = jnp.diff(edges)
        uniform = jnp.all(
            jnp.abs(widths / widths[0] - 1.0) <= 32.0 * jnp.finfo(edges.dtype).eps
        )
        edges = eqx.error_if(
            edges,
            jnp.any(~jnp.isfinite(edges))
            | (edges[0] != 0.0)
            | jnp.any(widths <= 0.0)
            | ~uniform,
            "bin_edges must be finite, start at zero, be strictly increasing, and be uniformly spaced.",
        )
        response = eqx.error_if(
            response,
            jnp.any(~jnp.isfinite(response))
            | jnp.any(response < 0.0)
            | (jnp.sum(response) <= 0.0),
            "instrument_response must be finite, nonnegative, and have positive mass.",
        )
        response = response / jnp.sum(response)
        self.bin_edges = edges
        self.instrument_response = response
        self.time_unit = time
        self.count_unit = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fluorescence-photon-plan",
                "bin_edges": array_tree_fingerprint(edges),
                "instrument_response": array_tree_fingerprint(response),
                "time_unit": time,
                "count_unit": count,
            }
        )

    def prepare(self) -> PreparedFluorescencePhotonModel:
        return PreparedFluorescencePhotonModel(self)


class PreparedFluorescencePhotonModel(StrictModule, NonTrainableState):
    """Prepared causal IRF convolution for lifetime/FRET histograms."""

    plan: FluorescencePhotonPlan
    response_matrix: Array
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: FluorescencePhotonPlan, /):
        if not isinstance(plan, FluorescencePhotonPlan):
            raise TypeError("plan must be a FluorescencePhotonPlan.")
        size = plan.instrument_response.size
        row = jnp.arange(size, dtype=jnp.int32)[:, None]
        column = jnp.arange(size, dtype=jnp.int32)[None, :]
        offset = row - column
        matrix = jnp.where(
            offset >= 0,
            plan.instrument_response[jnp.clip(offset, 0, size - 1)],
            jnp.asarray(0.0, dtype=plan.instrument_response.dtype),
        )
        self.plan = plan
        self.response_matrix = matrix
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-fluorescence-photon-model", "plan": plan.plan_id}
        )

    def expected(
        self,
        donor_lifetime: ArrayLike,
        photon_count: ArrayLike,
        /,
        *,
        fret_efficiency: ArrayLike = 0.0,
        background_count: ArrayLike = 0.0,
    ) -> FluorescencePhotonExpectation:
        dtype = self.plan.bin_edges.dtype
        lifetime = jnp.asarray(donor_lifetime, dtype=dtype)
        photons = jnp.asarray(photon_count, dtype=dtype)
        efficiency = jnp.asarray(fret_efficiency, dtype=dtype)
        background = jnp.asarray(background_count, dtype=dtype)
        if any(
            value.shape != () for value in (lifetime, photons, efficiency, background)
        ):
            raise ValueError("Photon-model parameters must be scalar.")
        valid = (
            jnp.isfinite(lifetime)
            & (lifetime > 0.0)
            & jnp.isfinite(photons)
            & (photons >= 0.0)
            & jnp.isfinite(efficiency)
            & (efficiency >= 0.0)
            & (efficiency <= 1.0)
            & jnp.isfinite(background)
            & (background >= 0.0)
        )
        effective = lifetime * (1.0 - efficiency)
        positive_lifetime = valid & (effective > 0.0)
        safe_lifetime = jnp.where(positive_lifetime, effective, jnp.ones_like(effective))
        left = self.plan.bin_edges[:-1]
        right = self.plan.bin_edges[1:]
        intrinsic = jnp.exp(-left / safe_lifetime) * (
            -jnp.expm1(-(right - left) / safe_lifetime)
        )
        prompt = jnp.zeros_like(intrinsic).at[0].set(1.0)
        intrinsic = jnp.where(efficiency == 1.0, prompt, intrinsic)
        intrinsic_sum = jnp.sum(intrinsic)
        intrinsic_mass_valid = jnp.isfinite(intrinsic_sum) & (intrinsic_sum > 0.0)
        intrinsic = intrinsic / jnp.where(
            intrinsic_mass_valid, intrinsic_sum, jnp.ones_like(intrinsic_sum)
        )
        detected = contract("ij,j->i", self.response_matrix, intrinsic)
        detected_sum = jnp.sum(detected)
        detected_mass_valid = jnp.isfinite(detected_sum) & (detected_sum > 0.0)
        detected = detected / jnp.where(
            detected_mass_valid, detected_sum, jnp.ones_like(detected_sum)
        )
        model_valid = valid & intrinsic_mass_valid & detected_mass_valid
        expected_counts = photons * detected + background / jnp.asarray(
            detected.size, dtype=dtype
        )
        expected_counts = jnp.where(
            model_valid, expected_counts, jnp.full_like(expected_counts, jnp.nan)
        )
        finite = model_valid & jnp.all(jnp.isfinite(expected_counts))
        identifiable = finite & (photons > 0.0) & (efficiency < 1.0)
        successful = finite & identifiable
        return FluorescencePhotonExpectation(
            effective,
            intrinsic,
            detected,
            expected_counts,
            efficiency,
            finite,
            identifiable,
            successful,
        )

    def forward(
        self,
        key: Array,
        donor_lifetime: ArrayLike,
        photon_count: ArrayLike,
        /,
        *,
        fret_efficiency: ArrayLike = 0.0,
        background_count: ArrayLike = 0.0,
    ) -> FluorescencePhotonResult:
        expectation = self.expected(
            donor_lifetime,
            photon_count,
            fret_efficiency=fret_efficiency,
            background_count=background_count,
        )
        safe_expected = jnp.where(
            expectation.finite,
            expectation.expected_counts,
            jnp.zeros_like(expectation.expected_counts),
        )
        counts = jax.random.poisson(key, safe_expected, shape=safe_expected.shape)
        finite = expectation.finite & jnp.all(jnp.isfinite(counts))
        successful = finite & expectation.identifiable
        return FluorescencePhotonResult(
            expectation.expected_counts,
            counts,
            expectation.effective_lifetime,
            expectation.fret_efficiency,
            finite,
            expectation.identifiable,
            successful,
        )

    def evaluate(
        self,
        observed_counts: ArrayLike,
        donor_lifetime: ArrayLike,
        photon_count: ArrayLike,
        /,
        *,
        fret_efficiency: ArrayLike = 0.0,
        background_count: ArrayLike = 0.0,
    ) -> FluorescencePhotonEvaluation:
        expectation = self.expected(
            donor_lifetime,
            photon_count,
            fret_efficiency=fret_efficiency,
            background_count=background_count,
        )
        observed = jnp.asarray(observed_counts, dtype=expectation.expected_counts.dtype)
        if observed.shape != expectation.expected_counts.shape:
            raise ValueError("observed_counts must contain one value per time bin.")
        observed_valid = (
            jnp.all(jnp.isfinite(observed))
            & jnp.all(observed >= 0.0)
            & jnp.all(observed == jnp.floor(observed))
        )
        possible = jnp.all((expectation.expected_counts > 0.0) | (observed == 0.0))
        safe_expected = jnp.where(
            expectation.expected_counts > 0.0,
            expectation.expected_counts,
            jnp.ones_like(expectation.expected_counts),
        )
        log_likelihood = jnp.sum(
            jnp.where(
                observed == 0.0,
                jnp.zeros_like(observed),
                observed * jnp.log(safe_expected),
            )
            - expectation.expected_counts
            - jsp.special.gammaln(observed + 1.0)
        )
        finite = (
            expectation.finite & observed_valid & possible & jnp.isfinite(log_likelihood)
        )
        identifiable = finite & expectation.identifiable
        successful = finite & identifiable
        return FluorescencePhotonEvaluation(
            expectation.expected_counts,
            observed - expectation.expected_counts,
            log_likelihood,
            finite,
            identifiable,
            successful,
        )


class DwellTimeLikelihoodResult(StrictModule):
    """Right-censored exponential dwell-time likelihood and MLE evidence."""

    log_likelihood: Array
    event_count: Array
    censored_count: Array
    total_exposure: Array
    maximum_likelihood_rate: Array
    finite: Array
    identifiable: Array
    successful: Array


class DwellTimeLikelihoodPlan(StrictModule, NonTrainableState):
    """Fixed-capacity right-censored exponential channel dwell-time plan."""

    sample_capacity: int = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, sample_capacity: int, /, *, time_unit: str = "s"):
        capacity = _positive_integer(sample_capacity, "sample_capacity")
        time = _unit(time_unit, "time_unit")
        self.sample_capacity = capacity
        self.time_unit = time
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dwell-time-likelihood-plan",
                "sample_capacity": capacity,
                "time_unit": time,
            }
        )

    def prepare(self) -> PreparedDwellTimeLikelihood:
        return PreparedDwellTimeLikelihood(self)


class PreparedDwellTimeLikelihood(StrictModule, NonTrainableState):
    """Prepared fixed-capacity censored exponential likelihood."""

    plan: DwellTimeLikelihoodPlan
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: DwellTimeLikelihoodPlan, /):
        if not isinstance(plan, DwellTimeLikelihoodPlan):
            raise TypeError("plan must be a DwellTimeLikelihoodPlan.")
        self.plan = plan
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-dwell-time-likelihood", "plan": plan.plan_id}
        )

    def evaluate(
        self,
        durations: ArrayLike,
        event_observed: ArrayLike,
        rate: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
    ) -> DwellTimeLikelihoodResult:
        duration = _floating_array(durations)
        events_raw = jnp.asarray(event_observed)
        if duration.shape != (self.plan.sample_capacity,):
            raise ValueError("durations must match the planned sample capacity.")
        if events_raw.shape != (self.plan.sample_capacity,):
            raise ValueError("event_observed must match the planned sample capacity.")
        rate_ = jnp.asarray(rate, dtype=duration.dtype)
        if rate_.shape != ():
            raise ValueError("rate must be scalar.")
        events = events_raw.astype(jnp.bool_)
        if active is None:
            active_ = jnp.ones_like(events)
            active_valid = jnp.asarray(True)
        else:
            active_raw = jnp.asarray(active)
            if active_raw.shape != (self.plan.sample_capacity,):
                raise ValueError("active must match the planned sample capacity.")
            active_valid = jnp.all((active_raw == 0) | (active_raw == 1))
            active_ = active_raw.astype(jnp.bool_)
        events_valid = jnp.all(
            jnp.where(active_, (events_raw == 0) | (events_raw == 1), True)
        )
        selected_events = active_ & events
        event_count = jnp.sum(selected_events)
        active_count = jnp.sum(active_)
        censored_count = active_count - event_count
        exposure = jnp.sum(jnp.where(active_, duration, jnp.zeros_like(duration)))
        valid = (
            jnp.all(jnp.where(active_, jnp.isfinite(duration), True))
            & jnp.all(jnp.where(active_, duration >= 0.0, True))
            & jnp.isfinite(rate_)
            & (rate_ > 0.0)
            & events_valid
            & active_valid
            & (active_count > 0)
            & (exposure > 0.0)
        )
        safe_rate = jnp.where(valid, rate_, jnp.ones_like(rate_))
        log_likelihood = (
            event_count.astype(duration.dtype) * jnp.log(safe_rate) - safe_rate * exposure
        )
        maximum_likelihood_rate = event_count.astype(duration.dtype) / jnp.where(
            exposure > 0.0, exposure, jnp.ones_like(exposure)
        )
        identifiable = valid & (event_count > 0)
        maximum_likelihood_rate = jnp.where(
            identifiable,
            maximum_likelihood_rate,
            jnp.asarray(jnp.nan, dtype=duration.dtype),
        )
        finite = valid & jnp.isfinite(log_likelihood)
        successful = finite & identifiable
        return DwellTimeLikelihoodResult(
            log_likelihood,
            event_count,
            censored_count,
            exposure,
            maximum_likelihood_rate,
            finite,
            identifiable,
            successful,
        )


class IVReversalResult(StrictModule):
    """Weighted linear I-V fit with reversal-potential identifiability."""

    conductance: Array
    intercept: Array
    reversal_potential: Array
    fitted_current: Array
    residual: Array
    weighted_residual_sum_squares: Array
    finite: Array
    identifiable: Array
    successful: Array


class IVReversalPlan(StrictModule, NonTrainableState):
    """Prepared weighted I-V regression for I = g(V - E_rev)."""

    voltages: Array
    weights: Array
    minimum_conductance: float = eqx.field(static=True)
    voltage_unit: str = eqx.field(static=True)
    current_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        voltages: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        minimum_conductance: float = 0.0,
        voltage_unit: str = "V",
        current_unit: str = "A",
    ):
        voltage = jax.lax.stop_gradient(_floating_array(voltages))
        if voltage.ndim != 1 or voltage.size < 2:
            raise ValueError(
                "voltages must be a one-dimensional array with two or more entries."
            )
        if weights is None:
            weight = jnp.ones_like(voltage)
        else:
            weight = jax.lax.stop_gradient(jnp.asarray(weights, dtype=voltage.dtype))
            if weight.shape != voltage.shape:
                raise ValueError("weights must match the voltage shape.")
        minimum = float(minimum_conductance)
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_conductance must be finite and nonnegative.")
        voltage_label = _unit(voltage_unit, "voltage_unit")
        current_label = _unit(current_unit, "current_unit")
        voltage = eqx.error_if(
            voltage,
            jnp.any(~jnp.isfinite(voltage)),
            "voltages must be finite.",
        )
        weight_sum = jnp.sum(weight)
        weight = eqx.error_if(
            weight,
            jnp.any(~jnp.isfinite(weight))
            | jnp.any(weight < 0.0)
            | ~jnp.isfinite(weight_sum)
            | (weight_sum <= 0.0),
            "weights must be finite, nonnegative, and have finite positive mass.",
        )
        weight = weight / weight_sum
        self.voltages = voltage
        self.weights = weight
        self.minimum_conductance = minimum
        self.voltage_unit = voltage_label
        self.current_unit = current_label
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iv-reversal-plan",
                "voltages": array_tree_fingerprint(voltage),
                "weights": array_tree_fingerprint(weight),
                "minimum_conductance": minimum,
                "voltage_unit": voltage_label,
                "current_unit": current_label,
            }
        )

    def prepare(self) -> PreparedIVReversalInference:
        return PreparedIVReversalInference(self)


class PreparedIVReversalInference(StrictModule, NonTrainableState):
    """Prepared weighted voltage moments for I-V reversal inference."""

    plan: IVReversalPlan
    weight_sum: Array
    voltage_mean: Array
    voltage_scale: Array
    voltage_variation: Array
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: IVReversalPlan, /):
        if not isinstance(plan, IVReversalPlan):
            raise TypeError("plan must be an IVReversalPlan.")
        weight_sum = jnp.sum(plan.weights)
        voltage_mean = jnp.sum(plan.weights * plan.voltages) / weight_sum
        centered = plan.voltages - voltage_mean
        scale = jnp.max(jnp.abs(centered))
        scaled = centered / jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
        variation = jnp.sum(plan.weights * scaled * scaled)
        self.plan = plan
        self.weight_sum = weight_sum
        self.voltage_mean = voltage_mean
        self.voltage_scale = scale
        self.voltage_variation = variation
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-iv-reversal-inference", "plan": plan.plan_id}
        )

    def evaluate(self, currents: ArrayLike, /) -> IVReversalResult:
        current = jnp.asarray(currents, dtype=self.plan.voltages.dtype)
        if current.shape != self.plan.voltages.shape:
            raise ValueError("currents must match the planned voltage shape.")
        current_mean = jnp.sum(self.plan.weights * current) / self.weight_sum
        centered_voltage = self.plan.voltages - self.voltage_mean
        centered_scaled = centered_voltage / jnp.where(
            self.voltage_scale > 0.0,
            self.voltage_scale,
            jnp.ones_like(self.voltage_scale),
        )
        centered_current = current - current_mean
        normalized_variation = self.voltage_variation / self.weight_sum
        voltage_identifiable = (
            jnp.isfinite(self.voltage_scale)
            & (self.voltage_scale > 0.0)
            & jnp.isfinite(normalized_variation)
            & (normalized_variation > 0.0)
        )
        safe_denominator = jnp.where(
            voltage_identifiable,
            self.voltage_variation * self.voltage_scale,
            jnp.ones_like(self.voltage_variation),
        )
        conductance = (
            jnp.sum(self.plan.weights * centered_scaled * centered_current)
            / safe_denominator
        )
        intercept = current_mean - conductance * self.voltage_mean
        slope_identifiable = jnp.abs(conductance) > self.plan.minimum_conductance
        candidate_identifiable = voltage_identifiable & slope_identifiable
        reversal_candidate = -intercept / jnp.where(
            candidate_identifiable, conductance, jnp.ones_like(conductance)
        )
        identifiable = candidate_identifiable & jnp.isfinite(reversal_candidate)
        reversal = jnp.where(
            identifiable,
            reversal_candidate,
            jnp.asarray(jnp.nan, dtype=current.dtype),
        )
        fitted = conductance * self.plan.voltages + intercept
        residual = current - fitted
        weighted_rss = jnp.sum(self.plan.weights * residual * residual)
        finite = (
            jnp.all(jnp.isfinite(current))
            & jnp.isfinite(conductance)
            & jnp.isfinite(intercept)
            & jnp.all(jnp.isfinite(fitted))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(weighted_rss)
        )
        identifiable = finite & identifiable
        successful = finite & identifiable
        return IVReversalResult(
            conductance,
            intercept,
            reversal,
            fitted,
            residual,
            weighted_rss,
            finite,
            identifiable,
            successful,
        )


__all__ = [
    "AutocorrelationPlan",
    "AutocorrelationResult",
    "BrightnessConditionedTransportPlan",
    "BrightnessConditionedTransportResult",
    "CholeskyCovarianceAction",
    "CoordinateLayout",
    "CorrelatedGaussianPlan",
    "CorrelatedGaussianResult",
    "CovarianceAction",
    "DiffusionEvaluationResult",
    "DiffusionForwardResult",
    "DiffusionModelPlan",
    "DwellTimeLikelihoodPlan",
    "DwellTimeLikelihoodResult",
    "FluorescenceCorrelationPlan",
    "FluorescenceCorrelationResult",
    "FluorescencePhotonEvaluation",
    "FluorescencePhotonExpectation",
    "FluorescencePhotonPlan",
    "FluorescencePhotonResult",
    "IVReversalPlan",
    "IVReversalResult",
    "LinearObservationPlan",
    "MeanSquareDisplacementPlan",
    "MeanSquareDisplacementResult",
    "ObservationProduct",
    "PairCorrelationPlan",
    "PairCorrelationResult",
    "PrecisionCovarianceAction",
    "PreparedAutocorrelation",
    "PreparedBrightnessConditionedTransport",
    "PreparedDiffusionModel",
    "PreparedDwellTimeLikelihood",
    "PreparedFluorescenceCorrelation",
    "PreparedFluorescencePhotonModel",
    "PreparedIVReversalInference",
    "PreparedMeanSquareDisplacement",
    "PreparedPairCorrelation",
    "TheoryVector",
]
