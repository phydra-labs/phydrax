#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite, prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from .._frozendict import frozendict
from .._strict import StrictModule
from .._uncertainty import UNCERTAINTY_SOURCES, UncertaintySource
from ..stochastic._process import AbstractMarginalTransitionLaw, semigroup_objective
from ._metrics import energy_score, ensemble_crps
from ._predictive import PredictiveField


ProcessScoreReduction = Literal["mean", "sum", "none"]


def _axis(axis: int, rank: int, /, *, name: str) -> int:
    value = int(axis)
    if value < 0:
        value += rank
    if not 0 <= value < rank:
        raise ValueError(f"{name}={axis} is out of bounds for rank {rank}.")
    return value


def _confidence(value: float, /) -> float:
    confidence = float(value)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    return confidence


def _normal_quantile(confidence: float, /) -> Array:
    return jsp.special.ndtri(jnp.asarray(0.5 + 0.5 * confidence))


def _relative_error(value: Array, reference: Array, /) -> Array:
    numerator = jnp.linalg.norm(value - reference)
    denominator = jnp.maximum(jnp.linalg.norm(reference), 1e-12)
    return numerator / denominator


def _canonical_forecasts(
    samples: ArrayLike,
    targets: ArrayLike,
    /,
    *,
    sample_axis: int,
    case_axis: int | None,
    horizon_axis: int,
) -> tuple[Array, Array, tuple[int, ...]]:
    sample_values = jnp.asarray(samples, dtype=float)
    target_values = jnp.asarray(targets, dtype=float)
    if sample_values.ndim != target_values.ndim + 1:
        raise ValueError("samples must have exactly one more axis than targets.")
    sample_position = _axis(sample_axis, sample_values.ndim, name="sample_axis")
    sample_values = jnp.moveaxis(sample_values, sample_position, 0)
    if sample_values.shape[1:] != target_values.shape:
        raise ValueError(
            "Removing sample_axis from samples must produce the target shape; "
            f"got {sample_values.shape[1:]} and {target_values.shape}."
        )

    target_rank = target_values.ndim
    horizon_position = _axis(horizon_axis, target_rank, name="horizon_axis")
    if case_axis is None:
        remaining = tuple(
            index for index in range(target_rank) if index != horizon_position
        )
        permutation = (horizon_position,) + remaining
        target_values = jnp.transpose(target_values, permutation)[None, ...]
        sample_values = jnp.transpose(
            sample_values,
            (0,) + tuple(index + 1 for index in permutation),
        )[:, None, ...]
    else:
        case_position = _axis(case_axis, target_rank, name="case_axis")
        if case_position == horizon_position:
            raise ValueError("case_axis and horizon_axis must be distinct.")
        remaining = tuple(
            index
            for index in range(target_rank)
            if index not in (case_position, horizon_position)
        )
        permutation = (case_position, horizon_position) + remaining
        target_values = jnp.transpose(target_values, permutation)
        sample_values = jnp.transpose(
            sample_values,
            (0,) + tuple(index + 1 for index in permutation),
        )

    event_shape = tuple(int(size) for size in target_values.shape[2:])
    target_values = target_values.reshape(
        (
            target_values.shape[0],
            target_values.shape[1],
            prod(event_shape) if event_shape else 1,
        )
    )
    sample_values = sample_values.reshape(
        (
            sample_values.shape[0],
            sample_values.shape[1],
            sample_values.shape[2],
            target_values.shape[-1],
        )
    )
    return sample_values, target_values, event_shape


def _canonical_target_value(
    value: ArrayLike | None,
    target_shape: tuple[int, ...],
    /,
    *,
    name: str,
    case_axis: int | None,
    horizon_axis: int,
    dtype: type | None = None,
) -> Array | None:
    if value is None:
        return None
    array = jnp.broadcast_to(jnp.asarray(value, dtype=dtype), target_shape)
    target_rank = len(target_shape)
    horizon_position = _axis(horizon_axis, target_rank, name="horizon_axis")
    if case_axis is None:
        remaining = tuple(
            index for index in range(target_rank) if index != horizon_position
        )
        array = jnp.transpose(array, (horizon_position,) + remaining)[None, ...]
    else:
        case_position = _axis(case_axis, target_rank, name="case_axis")
        remaining = tuple(
            index
            for index in range(target_rank)
            if index not in (case_position, horizon_position)
        )
        array = jnp.transpose(array, (case_position, horizon_position) + remaining)
    return array.reshape((array.shape[0], array.shape[1], -1))


def _weighted_mean(values: Array, mask: Array, weights: Array, /) -> Array:
    active = jnp.where(mask, weights, 0.0)
    denominator = jnp.sum(active)
    numerator = jnp.sum(jnp.where(mask, values * active, 0.0))
    return jnp.where(denominator > 0.0, numerator / denominator, jnp.nan)


class HorizonScoreDiagnostics(StrictModule):
    """Proper scores and interval diagnostics indexed by forecast horizon."""

    horizons: Array
    marginal_crps: Array
    energy_score: Array
    pointwise_coverage: Array
    simultaneous_coverage: Array
    interval_width: Array
    valid_cases: Array
    lower_quantile: float = eqx.field(static=True)
    upper_quantile: float = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def nominal_coverage(self) -> float:
        return self.upper_quantile - self.lower_quantile


def horizon_score_diagnostics(
    samples: ArrayLike,
    targets: ArrayLike,
    horizons: ArrayLike,
    /,
    *,
    sample_axis: int = 0,
    case_axis: int | None = 0,
    horizon_axis: int = 1,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> HorizonScoreDiagnostics:
    """Score finite-sample forecasts without merging cases, horizons, or events.

    ``case_axis`` and ``horizon_axis`` refer to ``targets``. ``sample_axis`` refers
    to ``samples``. Removing the sample axis must leave exactly the target shape.
    All remaining axes form one multivariate event for the energy score.
    """

    lower, upper = float(lower_quantile), float(upper_quantile)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Require 0 <= lower_quantile < upper_quantile <= 1.")
    target_shape = tuple(jnp.shape(targets))
    sample_values, target_values, event_shape = _canonical_forecasts(
        samples,
        targets,
        sample_axis=sample_axis,
        case_axis=case_axis,
        horizon_axis=horizon_axis,
    )
    horizon_values = jnp.asarray(horizons, dtype=float)
    if horizon_values.ndim != 1 or horizon_values.shape[0] != target_values.shape[1]:
        raise ValueError("horizons must be a vector aligned with target horizon_axis.")
    if bool(jnp.any(~jnp.isfinite(horizon_values))) or bool(
        jnp.any(jnp.diff(horizon_values) <= 0.0)
    ):
        raise ValueError("horizons must be finite and strictly increasing.")

    declared_mask = _canonical_target_value(
        mask,
        target_shape,
        name="mask",
        case_axis=case_axis,
        horizon_axis=horizon_axis,
        dtype=bool,
    )
    if declared_mask is None:
        declared_mask = jnp.ones_like(target_values, dtype=bool)
    declared_weights = _canonical_target_value(
        weights,
        target_shape,
        name="weights",
        case_axis=case_axis,
        horizon_axis=horizon_axis,
        dtype=float,
    )
    if declared_weights is None:
        declared_weights = jnp.ones_like(target_values)
    if bool(jnp.any(jnp.where(declared_mask, declared_weights < 0.0, False))):
        raise ValueError("weights must be nonnegative at active target locations.")

    finite_samples = jnp.all(jnp.isfinite(sample_values), axis=0)
    finite_targets = jnp.isfinite(target_values)
    location_valid = declared_mask & finite_samples & finite_targets
    case_valid = jnp.all(location_valid | ~declared_mask, axis=-1) & jnp.any(
        declared_mask, axis=-1
    )
    active_mask = declared_mask & case_valid[..., None]

    lower_values = jnp.quantile(sample_values, lower, axis=0)
    upper_values = jnp.quantile(sample_values, upper, axis=0)
    crps_values = ensemble_crps(sample_values, target_values, sample_axis=0)
    covered = (target_values >= lower_values) & (target_values <= upper_values)
    widths = upper_values - lower_values

    crps_by_horizon: list[Array] = []
    energy_by_horizon: list[Array] = []
    pointwise_by_horizon: list[Array] = []
    simultaneous_by_horizon: list[Array] = []
    width_by_horizon: list[Array] = []
    for index in range(int(horizon_values.shape[0])):
        horizon_mask = active_mask[:, index]
        horizon_weights = declared_weights[:, index]
        crps_by_horizon.append(
            _weighted_mean(
                crps_values[:, index],
                horizon_mask,
                horizon_weights,
            )
        )
        pointwise_by_horizon.append(
            _weighted_mean(
                covered[:, index].astype(float),
                horizon_mask,
                horizon_weights,
            )
        )
        width_by_horizon.append(
            _weighted_mean(
                widths[:, index],
                horizon_mask,
                horizon_weights,
            )
        )
        per_case_simultaneous = jnp.all(
            covered[:, index] | ~horizon_mask,
            axis=-1,
        ).astype(float)
        simultaneous_by_horizon.append(
            _weighted_mean(
                per_case_simultaneous,
                case_valid[:, index],
                jnp.ones_like(per_case_simultaneous),
            )
        )

        per_case_energy: list[Array] = []
        for case_index in range(int(target_values.shape[0])):
            event_mask = horizon_mask[case_index]
            event_weights = jnp.where(
                event_mask,
                horizon_weights[case_index],
                0.0,
            )
            normalized = event_weights / jnp.maximum(jnp.sum(event_weights), 1e-12)
            scale = jnp.sqrt(normalized)
            score = energy_score(
                sample_values[:, case_index, index] * scale,
                target_values[case_index, index] * scale,
                sample_axis=0,
            )
            per_case_energy.append(
                jnp.where(case_valid[case_index, index], score, jnp.nan)
            )
        energy_by_horizon.append(jnp.nanmean(jnp.stack(per_case_energy)))

    return HorizonScoreDiagnostics(
        horizons=horizon_values,
        marginal_crps=jnp.stack(crps_by_horizon),
        energy_score=jnp.stack(energy_by_horizon),
        pointwise_coverage=jnp.stack(pointwise_by_horizon),
        simultaneous_coverage=jnp.stack(simultaneous_by_horizon),
        interval_width=jnp.stack(width_by_horizon),
        valid_cases=jnp.sum(case_valid, axis=0),
        lower_quantile=lower,
        upper_quantile=upper,
        event_shape=event_shape,
    )


class UniformRankDiagnostics(StrictModule):
    """Uniformity diagnostics for scalar PIT values or ensemble ranks."""

    values: Array
    valid: Array
    histogram: Array
    expected_histogram: Array
    empirical_cdf: Array
    reference_cdf: Array
    max_cdf_deviation: Array
    simultaneous_bound: Array
    valid_count: Array
    confidence: float = eqx.field(static=True)
    kind: Literal["pit", "ensemble_rank"] = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(self.valid_count > 0) and bool(
            self.max_cdf_deviation <= self.simultaneous_bound
        )


def _uniform_rank_summary(
    values: Array,
    valid: Array,
    /,
    *,
    bins: int,
    confidence: float,
    kind: Literal["pit", "ensemble_rank"],
) -> UniformRankDiagnostics:
    if bins <= 1:
        raise ValueError("bins must exceed one.")
    count = jnp.sum(valid)
    clipped = jnp.clip(values, 0.0, jnp.nextafter(1.0, 0.0))
    indices = jnp.floor(clipped * bins).astype(jnp.int32)
    histogram = jnp.sum(
        jax.nn.one_hot(indices, bins, dtype=float) * valid[..., None],
        axis=tuple(range(valid.ndim)),
    )
    expected = jnp.full((bins,), count / float(bins))
    empirical_cdf = jnp.cumsum(histogram) / jnp.maximum(count, 1)
    reference_cdf = jnp.arange(1, bins + 1, dtype=float) / float(bins)
    deviation = jnp.max(jnp.abs(empirical_cdf - reference_cdf))
    alpha = 1.0 - confidence
    bound = jnp.sqrt(jnp.log(2.0 / alpha) / (2.0 * jnp.maximum(count, 1)))
    return UniformRankDiagnostics(
        values=jnp.where(valid, values, jnp.nan),
        valid=valid,
        histogram=histogram,
        expected_histogram=expected,
        empirical_cdf=empirical_cdf,
        reference_cdf=reference_cdf,
        max_cdf_deviation=deviation,
        simultaneous_bound=bound,
        valid_count=count,
        confidence=confidence,
        kind=kind,
    )


def pit_diagnostics(
    values: ArrayLike,
    /,
    *,
    bins: int = 10,
    confidence: float = 0.95,
) -> UniformRankDiagnostics:
    """Assess uniformity of scalar probability-integral-transform values."""

    level = _confidence(confidence)
    pit = jnp.asarray(values, dtype=float)
    valid = jnp.isfinite(pit) & (pit >= 0.0) & (pit <= 1.0)
    return _uniform_rank_summary(
        pit,
        valid,
        bins=int(bins),
        confidence=level,
        kind="pit",
    )


def observable_rank_diagnostics(
    samples: ArrayLike,
    targets: ArrayLike,
    /,
    *,
    sample_axis: int = 0,
    key: Key[Array, ""] | None = None,
    confidence: float = 0.95,
) -> UniformRankDiagnostics:
    """Compute tie-randomized ranks for declared scalar observables.

    Every non-sample position is one scalar verification event. Complete fields must
    first be reduced through an explicitly declared scalar observable.
    """

    level = _confidence(confidence)
    sample_values = jnp.asarray(samples, dtype=float)
    target_values = jnp.asarray(targets, dtype=float)
    position = _axis(sample_axis, sample_values.ndim, name="sample_axis")
    sample_values = jnp.moveaxis(sample_values, position, 0)
    if sample_values.shape[1:] != target_values.shape:
        raise ValueError("Observable samples and targets have incompatible shapes.")
    count = int(sample_values.shape[0])
    if count <= 0:
        raise ValueError("Observable samples must be non-empty.")
    valid = jnp.isfinite(target_values) & jnp.all(jnp.isfinite(sample_values), axis=0)
    less = jnp.sum(sample_values < target_values, axis=0)
    equal = jnp.sum(sample_values == target_values, axis=0)
    if key is None:
        offset = equal // 2
    else:
        uniform = jr.uniform(key, target_values.shape)
        offset = jnp.floor(uniform * (equal + 1)).astype(equal.dtype)
    ranks = less + offset
    values = (ranks.astype(float) + 0.5) / float(count + 1)
    return _uniform_rank_summary(
        values,
        valid,
        bins=count + 1,
        confidence=level,
        kind="ensemble_rank",
    )


class MonteCarloEstimate(StrictModule):
    """Replicated Monte Carlo estimate with a normal confidence interval."""

    replicates: Array
    mean: Array
    standard_error: Array
    lower: Array
    upper: Array
    confidence: float = eqx.field(static=True)


def monte_carlo_estimate(
    values: ArrayLike,
    /,
    *,
    confidence: float = 0.95,
) -> MonteCarloEstimate:
    """Summarize independent leading-axis Monte Carlo replicates."""

    level = _confidence(confidence)
    replicates = jnp.asarray(values, dtype=float)
    if replicates.ndim < 1 or replicates.shape[0] < 2:
        raise ValueError("At least two leading-axis Monte Carlo replicates are required.")
    mean = jnp.mean(replicates, axis=0)
    standard_error = jnp.std(replicates, axis=0, ddof=1) / jnp.sqrt(
        float(replicates.shape[0])
    )
    half_width = _normal_quantile(level) * standard_error
    return MonteCarloEstimate(
        replicates=replicates,
        mean=mean,
        standard_error=standard_error,
        lower=mean - half_width,
        upper=mean + half_width,
        confidence=level,
    )


class SemigroupMonteCarloDiagnostics(StrictModule):
    """Replicated Chapman--Kolmogorov error and optional reference floor."""

    candidate: MonteCarloEstimate
    reference: MonteCarloEstimate | None
    excess: MonteCarloEstimate | None
    num_samples: int = eqx.field(static=True)
    num_replicates: int = eqx.field(static=True)

    def passes_reference(self, /, *, tolerance: float = 0.0) -> bool:
        if self.excess is None:
            raise ValueError("No reference law was supplied.")
        return bool(self.excess.upper <= float(tolerance))


def semigroup_mc_diagnostics(
    law: AbstractMarginalTransitionLaw,
    state: ArrayLike,
    /,
    *,
    t0: ArrayLike,
    tmid: ArrayLike,
    t1: ArrayLike,
    key: Key[Array, ""],
    num_samples: int = 256,
    num_replicates: int = 16,
    observable: Callable[[Array], Array] | None = None,
    reference_law: AbstractMarginalTransitionLaw | None = None,
    confidence: float = 0.95,
) -> SemigroupMonteCarloDiagnostics:
    """Estimate semigroup error repeatedly instead of gating one noisy draw."""

    sample_count = int(num_samples)
    replicate_count = int(num_replicates)
    if sample_count <= 0 or replicate_count < 2:
        raise ValueError("num_samples must be positive and num_replicates at least two.")
    candidate_values: list[Array] = []
    reference_values: list[Array] = []
    for index in range(replicate_count):
        candidate_values.append(
            semigroup_objective(
                law,
                state,
                t0=t0,
                tmid=tmid,
                t1=t1,
                key=jr.fold_in(key, index),
                num_samples=sample_count,
                observable=observable,
            )
        )
        if reference_law is not None:
            reference_values.append(
                semigroup_objective(
                    reference_law,
                    state,
                    t0=t0,
                    tmid=tmid,
                    t1=t1,
                    key=jr.fold_in(key, replicate_count + index),
                    num_samples=sample_count,
                    observable=observable,
                )
            )
    candidate_array = jnp.stack(candidate_values)
    candidate = monte_carlo_estimate(candidate_array, confidence=confidence)
    if reference_law is None:
        reference = None
        excess = None
    else:
        reference_array = jnp.stack(reference_values)
        reference = monte_carlo_estimate(reference_array, confidence=confidence)
        excess = monte_carlo_estimate(
            candidate_array - reference_array,
            confidence=confidence,
        )
    return SemigroupMonteCarloDiagnostics(
        candidate=candidate,
        reference=reference,
        excess=excess,
        num_samples=sample_count,
        num_replicates=replicate_count,
    )


class TemporalMomentDiagnostics(StrictModule):
    """Time-indexed mean, covariance, cross-covariance, and autocorrelation."""

    times: Array
    mean: Array
    covariance: Array
    cross_covariance: Array
    correlation: Array
    lag_autocorrelation: Array
    mean_relative_error: Array | None
    covariance_relative_error: Array | None
    event_shape: tuple[int, ...] = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)


def temporal_moment_diagnostics(
    samples: ArrayLike,
    times: ArrayLike,
    /,
    *,
    sample_axis: int = 0,
    time_axis: int = 1,
    observable: Callable[[Array], Array] | None = None,
    reference_mean: ArrayLike | None = None,
    reference_covariance: ArrayLike | None = None,
) -> TemporalMomentDiagnostics:
    """Compute dependence diagnostics from complete, shared-path trajectories."""

    values = jnp.asarray(samples, dtype=float)
    sample_position = _axis(sample_axis, values.ndim, name="sample_axis")
    values = jnp.moveaxis(values, sample_position, 0)
    reduced_time_position = _axis(time_axis, jnp.asarray(samples).ndim, name="time_axis")
    if reduced_time_position == sample_position:
        raise ValueError("sample_axis and time_axis must be distinct.")
    if reduced_time_position > sample_position:
        reduced_time_position -= 1
    values = jnp.moveaxis(values, reduced_time_position + 1, 1)
    if observable is not None:
        values = jnp.asarray(observable(values), dtype=float)
        if values.ndim < 2 or values.shape[:2] != (
            jnp.asarray(samples).shape[sample_position],
            jnp.asarray(times).shape[0],
        ):
            raise ValueError("observable must preserve leading sample and time axes.")
    if values.shape[0] < 2:
        raise ValueError("Temporal covariance requires at least two trajectories.")
    time_values = jnp.asarray(times, dtype=float)
    if time_values.ndim != 1 or values.shape[1] != time_values.shape[0]:
        raise ValueError("times must align with the trajectory time axis.")
    if bool(jnp.any(~jnp.isfinite(values))):
        raise ValueError("Temporal diagnostics require finite complete trajectories.")
    event_shape = tuple(int(size) for size in values.shape[2:])
    flat = values.reshape((values.shape[0], values.shape[1], -1))
    mean = jnp.mean(flat, axis=0)
    centered = flat - mean
    denominator = float(flat.shape[0] - 1)
    cross_covariance = oe.contract("mti,msj->tsij", centered, centered) / denominator
    component_indices = jnp.arange(flat.shape[-1])
    covariance = cross_covariance[:, :, component_indices, component_indices]
    variances = jnp.diagonal(covariance, axis1=0, axis2=1).T
    scale = jnp.sqrt(jnp.maximum(variances[:, None, :] * variances[None, :, :], 0.0))
    correlation = jnp.where(scale > 0.0, covariance / scale, 0.0)
    lag_values: list[Array] = []
    for lag in range(int(time_values.shape[0])):
        diagonal = jnp.diagonal(correlation, offset=lag, axis1=0, axis2=1)
        lag_values.append(jnp.mean(diagonal, axis=-1))
    lag_autocorrelation = jnp.stack(lag_values)

    mean_values = mean.reshape((mean.shape[0],) + event_shape)
    covariance_values = covariance.reshape(
        (covariance.shape[0], covariance.shape[1]) + event_shape
    )
    mean_error = None
    if reference_mean is not None:
        reference = jnp.asarray(reference_mean, dtype=float)
        if reference.shape != mean_values.shape:
            raise ValueError("reference_mean must match the time-indexed mean shape.")
        mean_error = _relative_error(mean_values, reference)
    covariance_error = None
    if reference_covariance is not None:
        reference = jnp.asarray(reference_covariance, dtype=float)
        if reference.shape != covariance_values.shape:
            raise ValueError(
                "reference_covariance must match the componentwise covariance shape."
            )
        covariance_error = _relative_error(covariance_values, reference)
    return TemporalMomentDiagnostics(
        times=time_values,
        mean=mean_values,
        covariance=covariance_values,
        cross_covariance=cross_covariance.reshape(
            (cross_covariance.shape[0], cross_covariance.shape[1])
            + event_shape
            + event_shape
        ),
        correlation=correlation.reshape(
            (correlation.shape[0], correlation.shape[1]) + event_shape
        ),
        lag_autocorrelation=lag_autocorrelation.reshape(
            (lag_autocorrelation.shape[0],) + event_shape
        ),
        mean_relative_error=mean_error,
        covariance_relative_error=covariance_error,
        event_shape=event_shape,
        num_samples=int(flat.shape[0]),
    )


class PairedNumericalUncertainty(StrictModule):
    """Richardson-estimated numerical uncertainty from coupled refinements."""

    mean_correction: Array
    variance: Array
    mean_squared_error: Array
    valid_pairs: Array
    refinement_ratio: float = eqx.field(static=True)
    convergence_order: float = eqx.field(static=True)


def paired_refinement_uncertainty(
    coarse: ArrayLike,
    fine: ArrayLike,
    /,
    *,
    refinement_ratio: float,
    convergence_order: float,
    pair_axis: int = 0,
    mask: ArrayLike | None = None,
) -> PairedNumericalUncertainty:
    """Estimate fine-grid error from pathwise- or casewise-coupled refinements."""

    coarse_values = jnp.asarray(coarse, dtype=float)
    fine_values = jnp.asarray(fine, dtype=float)
    if coarse_values.shape != fine_values.shape or coarse_values.ndim == 0:
        raise ValueError("coarse and fine must have equal non-scalar shapes.")
    position = _axis(pair_axis, coarse_values.ndim, name="pair_axis")
    coarse_values = jnp.moveaxis(coarse_values, position, 0)
    fine_values = jnp.moveaxis(fine_values, position, 0)
    if int(coarse_values.shape[0]) < 2:
        raise ValueError("Paired refinement estimation requires at least two pairs.")
    ratio, order = float(refinement_ratio), float(convergence_order)
    if not isfinite(ratio) or ratio <= 1.0:
        raise ValueError("refinement_ratio must be finite and greater than one.")
    if not isfinite(order) or order <= 0.0:
        raise ValueError("convergence_order must be finite and positive.")
    active = jnp.ones_like(coarse_values, dtype=bool)
    if mask is not None:
        active = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), jnp.shape(coarse))
        active = jnp.moveaxis(active, position, 0)
    active = active & jnp.isfinite(coarse_values) & jnp.isfinite(fine_values)
    valid_pairs = jnp.sum(active, axis=0)
    if bool(jnp.any(valid_pairs < 2)):
        raise ValueError(
            "Every numerical output requires at least two valid refinement pairs."
        )
    correction = (fine_values - coarse_values) / (ratio**order - 1.0)
    correction = jnp.where(active, correction, 0.0)
    denominator = valid_pairs.astype(float)
    mean = jnp.sum(correction, axis=0) / denominator
    second = jnp.sum(correction**2, axis=0) / denominator
    variance = jnp.maximum(second - mean**2, 0.0)
    return PairedNumericalUncertainty(
        mean_correction=mean,
        variance=variance,
        mean_squared_error=second,
        valid_pairs=valid_pairs,
        refinement_ratio=ratio,
        convergence_order=order,
    )


class PredictiveVarianceDecomposition(StrictModule):
    """Order-explicit iterated total-variance decomposition."""

    components: frozendict[str, Array]
    total: Array
    reconstructed: Array
    remainder: Array
    order: tuple[str, ...] = eqx.field(static=True)
    event_dims: tuple[str | None, ...] = eqx.field(static=True)


def _nan_group_variance(values: Array, axes: tuple[int, ...], /) -> Array:
    mean = jnp.nanmean(values, axis=axes, keepdims=True)
    second = jnp.nanmean(values**2, axis=axes, keepdims=True)
    return jnp.maximum(second - mean**2, 0.0)


def predictive_variance_decomposition(
    prediction: PredictiveField,
    /,
    *,
    order: Sequence[UncertaintySource] | None = None,
    numerical_uncertainty: PairedNumericalUncertainty | None = None,
) -> PredictiveVarianceDecomposition:
    """Decompose nested predictive variance from inner to outer uncertainty sources.

    Sources sharing multiple axes are reduced as one group. The declared order is
    mathematically significant for crossed designs. Conditional observation variance
    is the innermost component. Numerical uncertainty must come from explicitly
    coupled coarse/fine refinements, never from an axis label alone.
    """

    if not isinstance(prediction, PredictiveField):
        raise TypeError("prediction must be a PredictiveField.")
    source_dims: dict[str, list[str]] = {}
    for sample_axis in prediction.sample_axes:
        source_dims.setdefault(sample_axis.source, []).append(sample_axis.dim)
    if "numerical" in source_dims:
        raise ValueError(
            "A numerical SampleAxis is not refinement evidence; pass "
            "numerical_uncertainty from paired_refinement_uncertainty instead."
        )
    present = tuple(source_dims)
    if order is None:
        default = ("observation", "process", "input", "epistemic")
        resolved_order = tuple(source for source in default if source in present)
        resolved_order += tuple(
            source for source in present if source not in resolved_order
        )
    else:
        resolved_order = tuple(str(source) for source in order)
        if len(set(resolved_order)) != len(resolved_order):
            raise ValueError("order must not contain duplicate uncertainty sources.")
        invalid = tuple(
            source for source in resolved_order if source not in UNCERTAINTY_SOURCES
        )
        if invalid:
            raise ValueError(f"Unknown uncertainty sources: {invalid!r}.")
        if "numerical" in resolved_order:
            raise ValueError(
                "order covers explicit predictive sample sources only; numerical "
                "uncertainty is appended from paired refinement evidence."
            )
        if set(resolved_order) != set(present):
            raise ValueError(
                "order must contain every explicit predictive uncertainty source exactly once."
            )

    values = jnp.asarray(prediction.samples.data, dtype=float)
    if prediction.valid is not None:
        valid = jnp.asarray(prediction.valid.data, dtype=bool)
        valid_dims = prediction.valid.dims
        reshape = tuple(
            int(valid.shape[valid_dims.index(dim)]) if dim in valid_dims else 1
            for dim in prediction.samples.dims
        )
        values = jnp.where(valid.reshape(reshape), values, jnp.nan)
    sample_dims = tuple(axis.dim for axis in prediction.sample_axes)
    event_dims = tuple(dim for dim in prediction.samples.dims if dim not in sample_dims)
    current_dims = list(prediction.samples.dims)
    current = values
    components: dict[str, Array] = {}

    if prediction.conditional_variance is not None:
        conditional = jnp.asarray(prediction.conditional_variance.data, dtype=float)
        conditional_dims = prediction.conditional_variance.dims
        reshape = tuple(
            int(conditional.shape[conditional_dims.index(dim)])
            if dim in conditional_dims
            else 1
            for dim in prediction.samples.dims
        )
        conditional = jnp.broadcast_to(conditional.reshape(reshape), values.shape)
        sample_positions = tuple(current_dims.index(dim) for dim in sample_dims)
        observation_component = jnp.nanmean(conditional, axis=sample_positions)
        components["observation"] = observation_component

    for source in resolved_order:
        dims = tuple(source_dims[source])
        positions = tuple(current_dims.index(dim) for dim in dims)
        conditional_variance = _nan_group_variance(current, positions)
        remaining_sample_positions = tuple(
            index
            for index, dim in enumerate(current_dims)
            if dim in sample_dims and dim not in dims
        )
        component = conditional_variance
        if remaining_sample_positions:
            component = jnp.nanmean(
                component,
                axis=remaining_sample_positions,
                keepdims=True,
            )
        component = jnp.squeeze(
            component,
            axis=tuple(sorted(positions + remaining_sample_positions)),
        )
        components[source] = component
        current = jnp.nanmean(current, axis=positions)
        for position in sorted(positions, reverse=True):
            del current_dims[position]

    original_sample_positions = tuple(
        prediction.samples.dims.index(dim) for dim in sample_dims
    )
    total = _nan_group_variance(values, original_sample_positions)
    total = jnp.squeeze(total, axis=original_sample_positions)
    if prediction.conditional_variance is not None:
        total = total + components["observation"]
    if numerical_uncertainty is not None:
        if not isinstance(numerical_uncertainty, PairedNumericalUncertainty):
            raise TypeError(
                "numerical_uncertainty must be PairedNumericalUncertainty or None."
            )
        numerical_component = jnp.asarray(
            numerical_uncertainty.mean_squared_error, dtype=float
        )
        if jnp.broadcast_shapes(numerical_component.shape, total.shape) != total.shape:
            raise ValueError(
                "Paired numerical uncertainty must broadcast exactly to the event shape."
            )
        numerical_component = jnp.broadcast_to(numerical_component, total.shape)
        if bool(jnp.any(~jnp.isfinite(numerical_component))) or bool(
            jnp.any(numerical_component < 0.0)
        ):
            raise ValueError(
                "Paired numerical uncertainty must be finite and non-negative."
            )
        components["numerical"] = numerical_component
        total = total + numerical_component
    reconstructed = sum(components.values(), start=jnp.zeros_like(total))
    remainder = total - reconstructed
    component_order = (
        (("observation",) if prediction.conditional_variance is not None else ())
        + resolved_order
        + (("numerical",) if numerical_uncertainty is not None else ())
    )
    return PredictiveVarianceDecomposition(
        components=frozendict(components),
        total=total,
        reconstructed=reconstructed,
        remainder=remainder,
        order=component_order,
        event_dims=event_dims,
    )


__all__ = [
    "HorizonScoreDiagnostics",
    "horizon_score_diagnostics",
    "MonteCarloEstimate",
    "PairedNumericalUncertainty",
    "paired_refinement_uncertainty",
    "monte_carlo_estimate",
    "observable_rank_diagnostics",
    "pit_diagnostics",
    "PredictiveVarianceDecomposition",
    "predictive_variance_decomposition",
    "ProcessScoreReduction",
    "SemigroupMonteCarloDiagnostics",
    "semigroup_mc_diagnostics",
    "TemporalMomentDiagnostics",
    "temporal_moment_diagnostics",
    "UniformRankDiagnostics",
]
