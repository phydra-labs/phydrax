#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, cast, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._uncertainty import UNCERTAINTY_SOURCES, UncertaintySource
from ..stochastic._jump import JUMP_MAX_EVENTS, JUMP_SUCCESS, JumpEventBatch
from ._conformal import FunctionalConformal, NormalizedConformal, SplitConformal
from ._metrics import energy_score
from ._predictive import PredictionInterval
from ._process_diagnostics import (
    _axis,
    _canonical_forecasts,
    _canonical_target_value,
    _confidence,
    _normal_quantile,
    _relative_error,
    horizon_score_diagnostics,
    HorizonScoreDiagnostics,
    monte_carlo_estimate,
    MonteCarloEstimate,
    paired_refinement_uncertainty,
    PairedNumericalUncertainty,
    PredictiveVarianceDecomposition,
    SemigroupMonteCarloDiagnostics,
    TemporalMomentDiagnostics,
)


ProcessShiftKind = Literal[
    "in_distribution",
    "rollout_horizon",
    "covariance",
    "initial_condition",
    "parameter_regime",
]
ProcessConformalKind = Literal["trajectory", "observable"]


class TrajectoryScoreDiagnostics(StrictModule):
    """Proper scores over complete trajectories and their pairwise dependence."""

    trajectory_energy_score: Array
    variogram_score: Array
    energy_by_case: Array
    variogram_by_case: Array
    valid_energy_cases: Array
    valid_variogram_cases: Array
    num_samples: int = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    variogram_power: float = eqx.field(static=True)
    max_lag: int | None = eqx.field(static=True)


def trajectory_score_diagnostics(
    samples: ArrayLike,
    targets: ArrayLike,
    /,
    *,
    sample_axis: int = 0,
    case_axis: int | None = 0,
    time_axis: int = 1,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    variogram_power: float = 0.5,
    max_lag: int | None = None,
) -> TrajectoryScoreDiagnostics:
    """Score each shared-path rollout as one multivariate forecast event.

    The energy score sees the complete time-by-state trajectory. The variogram
    score compares all active coordinate pairs, optionally restricted to a
    temporal lag. Masking never turns independently resampled marginals into a
    path: the leading sample axis must identify complete shared realizations.
    """

    power = float(variogram_power)
    if not 0.0 < power <= 2.0:
        raise ValueError("variogram_power must lie in (0, 2].")
    lag = None if max_lag is None else int(max_lag)
    if lag is not None and lag < 1:
        raise ValueError("max_lag must be positive or None.")

    target_shape = tuple(jnp.shape(targets))
    sample_values, target_values, event_shape = _canonical_forecasts(
        samples,
        targets,
        sample_axis=sample_axis,
        case_axis=case_axis,
        horizon_axis=time_axis,
    )
    if int(sample_values.shape[0]) < 2:
        raise ValueError("Trajectory scores require at least two forecast realizations.")
    time_count = int(target_values.shape[1])
    event_size = int(target_values.shape[2])
    flattened_size = time_count * event_size
    if flattened_size < 2:
        raise ValueError("Variogram scores require at least two trajectory coordinates.")

    declared_mask = _canonical_target_value(
        mask,
        target_shape,
        name="mask",
        case_axis=case_axis,
        horizon_axis=time_axis,
        dtype=bool,
    )
    if declared_mask is None:
        declared_mask = jnp.ones_like(target_values, dtype=bool)
    declared_weights = _canonical_target_value(
        weights,
        target_shape,
        name="weights",
        case_axis=case_axis,
        horizon_axis=time_axis,
        dtype=float,
    )
    if declared_weights is None:
        declared_weights = jnp.ones_like(target_values)
    if bool(
        jnp.any(
            jnp.where(
                declared_mask,
                ~jnp.isfinite(declared_weights) | (declared_weights < 0.0),
                False,
            )
        )
    ):
        raise ValueError("Active trajectory weights must be finite and non-negative.")

    finite_samples = jnp.all(jnp.isfinite(sample_values), axis=0)
    finite_targets = jnp.isfinite(target_values)
    location_valid = declared_mask & finite_samples & finite_targets
    energy_valid = jnp.all(location_valid | ~declared_mask, axis=(1, 2)) & jnp.any(
        declared_mask, axis=(1, 2)
    )

    sample_flat = sample_values.reshape(
        (sample_values.shape[0], sample_values.shape[1], flattened_size)
    )
    target_flat = target_values.reshape((target_values.shape[0], flattened_size))
    mask_flat = declared_mask.reshape((declared_mask.shape[0], flattened_size))
    weight_flat = declared_weights.reshape((declared_weights.shape[0], flattened_size))

    pair_left_np, pair_right_np = np.triu_indices(flattened_size, k=1)
    if lag is not None:
        time_indices = np.repeat(np.arange(time_count), event_size)
        selected = np.abs(time_indices[pair_left_np] - time_indices[pair_right_np]) <= lag
        pair_left_np = pair_left_np[selected]
        pair_right_np = pair_right_np[selected]
    if pair_left_np.size == 0:
        raise ValueError("max_lag selected no variogram coordinate pairs.")
    pair_left = jnp.asarray(pair_left_np, dtype=jnp.int32)
    pair_right = jnp.asarray(pair_right_np, dtype=jnp.int32)

    energy_values: list[Array] = []
    variogram_values: list[Array] = []
    variogram_valid_values: list[Array] = []
    for case_index in range(int(target_flat.shape[0])):
        active_weight = jnp.where(mask_flat[case_index], weight_flat[case_index], 0.0)
        normalized_weight = active_weight / jnp.maximum(jnp.sum(active_weight), 1e-12)
        energy = energy_score(
            sample_flat[:, case_index] * jnp.sqrt(normalized_weight),
            target_flat[case_index] * jnp.sqrt(normalized_weight),
            sample_axis=0,
        )
        energy_values.append(jnp.where(energy_valid[case_index], energy, jnp.nan))

        pair_active = mask_flat[case_index, pair_left] & mask_flat[case_index, pair_right]
        pair_weight = jnp.where(
            pair_active,
            active_weight[pair_left] * active_weight[pair_right],
            0.0,
        )
        pair_weight = pair_weight / jnp.maximum(jnp.sum(pair_weight), 1e-12)
        forecast_difference = (
            jnp.abs(
                sample_flat[:, case_index, pair_left]
                - sample_flat[:, case_index, pair_right]
            )
            ** power
        )
        target_difference = (
            jnp.abs(
                target_flat[case_index, pair_left] - target_flat[case_index, pair_right]
            )
            ** power
        )
        pair_residual = jnp.mean(forecast_difference, axis=0) - target_difference
        variogram = jnp.sum(pair_weight * pair_residual**2)
        variogram_valid = energy_valid[case_index] & (jnp.sum(pair_active) > 0)
        variogram_values.append(jnp.where(variogram_valid, variogram, jnp.nan))
        variogram_valid_values.append(variogram_valid)

    energy_by_case = jnp.stack(energy_values)
    variogram_by_case = jnp.stack(variogram_values)
    variogram_valid = jnp.stack(variogram_valid_values)
    return TrajectoryScoreDiagnostics(
        trajectory_energy_score=jnp.nanmean(energy_by_case),
        variogram_score=jnp.nanmean(variogram_by_case),
        energy_by_case=energy_by_case,
        variogram_by_case=variogram_by_case,
        valid_energy_cases=jnp.sum(energy_valid),
        valid_variogram_cases=jnp.sum(variogram_valid),
        num_samples=int(sample_values.shape[0]),
        event_shape=(time_count,) + event_shape,
        variogram_power=power,
        max_lag=lag,
    )


class JumpEventSummary(StrictModule):
    """Count, waiting-time, channel, and mark statistics from successful paths."""

    path_counts: Array
    successful: Array
    count_mean: Array
    count_variance: Array
    count_histogram: Array
    count_probabilities: Array
    interarrival_mean: Array
    interarrival_variance: Array
    channel_counts: Array
    channel_probabilities: Array
    mark_counts: Array
    mark_mean: Array
    mark_covariance: Array
    num_paths: int = eqx.field(static=True)
    num_successful: int = eqx.field(static=True)
    num_failed: int = eqx.field(static=True)
    num_overflow: int = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)


class JumpEventDiagnostics(StrictModule):
    """Event summary plus reference discrepancies when a reference batch exists."""

    candidate: JumpEventSummary
    reference: JumpEventSummary | None
    count_mean_relative_error: Array | None
    count_variance_relative_error: Array | None
    count_wasserstein_distance: Array | None
    channel_frequency_l1: Array | None
    mark_mean_relative_error: Array | None
    mark_covariance_relative_error: Array | None


def _jump_event_summary(
    events: JumpEventBatch,
    /,
    *,
    t0: float,
    t1: float,
    num_channels: int,
) -> JumpEventSummary:
    if not isinstance(events, JumpEventBatch):
        raise TypeError("events must be a JumpEventBatch.")
    start, end = float(t0), float(t1)
    if not np.isfinite(start) or not np.isfinite(end) or not start < end:
        raise ValueError("Require finite t0 < t1.")
    channels_count = int(num_channels)
    if channels_count <= 0:
        raise ValueError("num_channels must be positive.")

    valid = jnp.asarray(events.valid, dtype=bool)
    times = jnp.asarray(events.times, dtype=float)
    channels = jnp.asarray(events.channels, dtype=jnp.int32)
    invalid_seen = jnp.cumsum(~valid, axis=-1) > 0
    if bool(jnp.any(valid & invalid_seen)):
        raise ValueError("JumpEventBatch.valid must be a prefix mask on every path.")
    active_time_invalid = valid & (~jnp.isfinite(times) | (times < start) | (times > end))
    if bool(jnp.any(active_time_invalid)):
        raise ValueError("Active event times must be finite and lie in [t0, t1].")
    preceding = jnp.concatenate(
        (jnp.full(times.shape[:-1] + (1,), start), times[..., :-1]),
        axis=-1,
    )
    gaps = times - preceding
    if bool(jnp.any(valid & (gaps < 0.0))):
        raise ValueError("Active event times must be nondecreasing on every path.")
    if bool(jnp.any(valid & ((channels < 0) | (channels >= channels_count)))):
        raise ValueError("Active event channels lie outside num_channels.")

    successful = jnp.asarray(events.status == JUMP_SUCCESS, dtype=bool)
    usable = valid & successful[..., None]
    path_counts = jnp.sum(valid, axis=-1, dtype=jnp.int32)
    successful_count = int(jnp.sum(successful))
    path_count = int(prod(events.batch_shape)) if events.batch_shape else 1
    if successful_count == 0:
        raise ValueError("Event diagnostics require at least one successful path.")
    success_weight = successful.astype(float)
    count_mean = jnp.sum(path_counts * success_weight) / successful_count
    centered_count = path_counts - count_mean
    count_variance = jnp.sum(centered_count**2 * success_weight) / successful_count
    count_histogram = jnp.sum(
        jax.nn.one_hot(path_counts, events.max_events + 1, dtype=float)
        * successful[..., None],
        axis=tuple(range(successful.ndim)),
    )
    count_probabilities = count_histogram / successful_count

    gap_count = jnp.sum(usable)
    interarrival_mean = jnp.sum(jnp.where(usable, gaps, 0.0)) / jnp.maximum(gap_count, 1)
    interarrival_variance = jnp.sum(
        jnp.where(usable, (gaps - interarrival_mean) ** 2, 0.0)
    ) / jnp.maximum(gap_count, 1)

    channel_one_hot = jax.nn.one_hot(channels, channels_count, dtype=float)
    channel_counts = jnp.sum(
        channel_one_hot * usable[..., None],
        axis=tuple(range(usable.ndim)),
    )
    total_events = jnp.sum(channel_counts)
    channel_probabilities = jnp.where(
        total_events > 0,
        channel_counts / total_events,
        jnp.full_like(channel_counts, jnp.nan),
    )

    mark_size = prod(events.mark_shape) if events.mark_shape else 1
    marks = jnp.asarray(events.marks, dtype=float).reshape(valid.shape + (mark_size,))
    if bool(jnp.any(usable[..., None] & ~jnp.isfinite(marks))):
        raise ValueError("Active event marks must be finite.")
    mark_weight = channel_one_hot * usable[..., None]
    mark_counts = jnp.sum(mark_weight, axis=tuple(range(usable.ndim)))
    mark_sum = oe.contract("...k,...i->ki", mark_weight, marks)
    mark_mean_flat = mark_sum / jnp.maximum(mark_counts[:, None], 1)
    centered_marks = marks[..., None, :] - mark_mean_flat
    mark_covariance_flat = oe.contract(
        "...k,...ki,...kj->kij",
        mark_weight,
        centered_marks,
        centered_marks,
    ) / jnp.maximum(mark_counts[:, None, None], 1)
    mark_mean_flat = jnp.where(mark_counts[:, None] > 0, mark_mean_flat, jnp.nan)
    mark_covariance_flat = jnp.where(
        mark_counts[:, None, None] > 0,
        mark_covariance_flat,
        jnp.nan,
    )
    mark_mean = mark_mean_flat.reshape((channels_count,) + events.mark_shape)
    mark_covariance = mark_covariance_flat.reshape(
        (channels_count,) + events.mark_shape + events.mark_shape
    )

    return JumpEventSummary(
        path_counts=path_counts,
        successful=successful,
        count_mean=count_mean,
        count_variance=count_variance,
        count_histogram=count_histogram,
        count_probabilities=count_probabilities,
        interarrival_mean=interarrival_mean,
        interarrival_variance=interarrival_variance,
        channel_counts=channel_counts,
        channel_probabilities=channel_probabilities,
        mark_counts=mark_counts,
        mark_mean=mark_mean,
        mark_covariance=mark_covariance,
        num_paths=path_count,
        num_successful=successful_count,
        num_failed=path_count - successful_count,
        num_overflow=int(jnp.sum(events.status == JUMP_MAX_EVENTS)),
        num_channels=channels_count,
        mark_shape=events.mark_shape,
        t0=start,
        t1=end,
    )


def _shared_mark_relative_error(
    candidate: Array,
    reference: Array,
    candidate_counts: Array,
    reference_counts: Array,
    /,
) -> Array:
    shared = (candidate_counts > 0) & (reference_counts > 0)
    if not bool(jnp.any(shared)):
        return jnp.asarray(jnp.nan)
    trailing = (1,) * (candidate.ndim - 1)
    mask = shared.reshape(shared.shape + trailing)
    return _relative_error(
        jnp.where(mask, candidate, 0.0),
        jnp.where(mask, reference, 0.0),
    )


def jump_event_diagnostics(
    events: JumpEventBatch,
    /,
    *,
    t0: float,
    t1: float,
    num_channels: int,
    reference: JumpEventBatch | None = None,
) -> JumpEventDiagnostics:
    """Summarize finite-activity events and compare an optional reference sample."""

    candidate = _jump_event_summary(
        events,
        t0=t0,
        t1=t1,
        num_channels=num_channels,
    )
    if reference is None:
        return JumpEventDiagnostics(
            candidate=candidate,
            reference=None,
            count_mean_relative_error=None,
            count_variance_relative_error=None,
            count_wasserstein_distance=None,
            channel_frequency_l1=None,
            mark_mean_relative_error=None,
            mark_covariance_relative_error=None,
        )
    reference_summary = _jump_event_summary(
        reference,
        t0=t0,
        t1=t1,
        num_channels=num_channels,
    )
    if candidate.mark_shape != reference_summary.mark_shape:
        raise ValueError("Candidate and reference event marks must have equal shape.")
    histogram_size = max(
        int(candidate.count_probabilities.shape[0]),
        int(reference_summary.count_probabilities.shape[0]),
    )
    candidate_probabilities = jnp.pad(
        candidate.count_probabilities,
        (0, histogram_size - int(candidate.count_probabilities.shape[0])),
    )
    reference_probabilities = jnp.pad(
        reference_summary.count_probabilities,
        (0, histogram_size - int(reference_summary.count_probabilities.shape[0])),
    )
    count_wasserstein = jnp.sum(
        jnp.abs(jnp.cumsum(candidate_probabilities) - jnp.cumsum(reference_probabilities))
    )
    return JumpEventDiagnostics(
        candidate=candidate,
        reference=reference_summary,
        count_mean_relative_error=_relative_error(
            candidate.count_mean, reference_summary.count_mean
        ),
        count_variance_relative_error=_relative_error(
            candidate.count_variance, reference_summary.count_variance
        ),
        count_wasserstein_distance=count_wasserstein,
        channel_frequency_l1=jnp.sum(
            jnp.abs(
                candidate.channel_probabilities - reference_summary.channel_probabilities
            )
        ),
        mark_mean_relative_error=_shared_mark_relative_error(
            candidate.mark_mean,
            reference_summary.mark_mean,
            candidate.mark_counts,
            reference_summary.mark_counts,
        ),
        mark_covariance_relative_error=_shared_mark_relative_error(
            candidate.mark_covariance,
            reference_summary.mark_covariance,
            candidate.mark_counts,
            reference_summary.mark_counts,
        ),
    )


class FirstPassageDiagnostics(StrictModule):
    """Fixed-horizon first-passage CDF check against a declared analytic law."""

    evaluation_times: Array
    empirical_cdf: Array
    reference_cdf: Array
    absolute_error: Array
    max_cdf_deviation: Array
    simultaneous_bound: Array
    observed_fraction: Array
    num_paths: int = eqx.field(static=True)
    horizon: float = eqx.field(static=True)
    confidence: float = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(self.max_cdf_deviation <= self.simultaneous_bound)


def first_passage_diagnostics(
    hitting_times: ArrayLike,
    observed: ArrayLike,
    evaluation_times: ArrayLike,
    reference_cdf: ArrayLike | Callable[[Array], Array],
    /,
    *,
    horizon: float,
    confidence: float = 0.95,
) -> FirstPassageDiagnostics:
    """Check a common-horizon hitting-time sample without censoring fabrication."""

    hits = jnp.asarray(hitting_times, dtype=float).reshape((-1,))
    observed_values = jnp.asarray(observed, dtype=bool).reshape((-1,))
    if hits.shape != observed_values.shape or int(hits.shape[0]) == 0:
        raise ValueError("hitting_times and observed must be equal non-empty vectors.")
    terminal = float(horizon)
    if not np.isfinite(terminal):
        raise ValueError("horizon must be finite.")
    if bool(jnp.any(observed_values & (~jnp.isfinite(hits) | (hits > terminal)))):
        raise ValueError(
            "Observed hitting times must be finite and no later than horizon."
        )
    times = jnp.asarray(evaluation_times, dtype=float)
    if times.ndim != 1 or int(times.shape[0]) == 0:
        raise ValueError("evaluation_times must be a non-empty vector.")
    if bool(jnp.any(~jnp.isfinite(times))) or bool(jnp.any(jnp.diff(times) <= 0.0)):
        raise ValueError("evaluation_times must be finite and strictly increasing.")
    if bool(jnp.any(times > terminal)):
        raise ValueError("evaluation_times must not exceed horizon.")
    if callable(reference_cdf):
        reference_fn = cast(Callable[[Array], Array], reference_cdf)
        reference = jnp.asarray(reference_fn(times), dtype=float)
    else:
        reference = jnp.asarray(reference_cdf, dtype=float)
    if reference.shape != times.shape or bool(
        jnp.any(~jnp.isfinite(reference) | (reference < 0.0) | (reference > 1.0))
    ):
        raise ValueError("reference_cdf must return one valid probability per time.")
    empirical = jnp.mean(
        observed_values[:, None] & (hits[:, None] <= times[None, :]),
        axis=0,
    )
    deviation = jnp.abs(empirical - reference)
    level = _confidence(confidence)
    bound = jnp.sqrt(jnp.log(2.0 / (1.0 - level)) / (2.0 * float(hits.shape[0])))
    return FirstPassageDiagnostics(
        evaluation_times=times,
        empirical_cdf=empirical,
        reference_cdf=reference,
        absolute_error=deviation,
        max_cdf_deviation=jnp.max(deviation),
        simultaneous_bound=bound,
        observed_fraction=jnp.mean(observed_values),
        num_paths=int(hits.shape[0]),
        horizon=terminal,
        confidence=level,
    )


class ProcessValidationSplit(StrictModule):
    """Disjoint physical-case identities for train, calibration, and test data."""

    train_case_ids: tuple[str, ...] = eqx.field(static=True)
    calibration_case_ids: tuple[str, ...] = eqx.field(static=True)
    test_case_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        train_case_ids: Sequence[str],
        calibration_case_ids: Sequence[str],
        test_case_ids: Sequence[str],
        /,
    ):
        train = _case_ids(train_case_ids, name="train_case_ids")
        calibration = _case_ids(calibration_case_ids, name="calibration_case_ids")
        test = _case_ids(test_case_ids, name="test_case_ids")
        all_ids = train + calibration + test
        if len(set(all_ids)) != len(all_ids):
            raise ValueError(
                "Train, calibration, and test case identities must be disjoint."
            )
        self.train_case_ids = train
        self.calibration_case_ids = calibration
        self.test_case_ids = test

    def require_case_count(
        self, count: int, /, *, partition: Literal["calibration", "test"]
    ):
        expected = (
            len(self.calibration_case_ids)
            if partition == "calibration"
            else len(self.test_case_ids)
        )
        if int(count) != expected:
            raise ValueError(
                f"{partition} data contain {int(count)} cases, but the split declares "
                f"{expected}."
            )


def _case_ids(values: Sequence[str], /, *, name: str) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in values)
    if not resolved or any(not value for value in resolved):
        raise ValueError(f"{name} must contain non-empty identities.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate identities.")
    return resolved


def _case_count(values: ArrayLike | cx.Field, case_axis: int | str, /) -> int:
    if isinstance(values, cx.Field):
        if isinstance(case_axis, str):
            if case_axis not in values.named_shape:
                raise ValueError(f"Unknown case dimension {case_axis!r}.")
            return int(values.named_shape[case_axis])
        position = _axis(case_axis, values.data.ndim, name="case_axis")
        return int(values.data.shape[position])
    if isinstance(case_axis, str):
        raise TypeError("A string case_axis requires a coordax.Field.")
    array = jnp.asarray(values)
    position = _axis(case_axis, array.ndim, name="case_axis")
    return int(array.shape[position])


class HorizonScaleCalibrator(StrictModule):
    """Held-out Gaussian scale multipliers fitted independently by horizon."""

    scale_multiplier: Array
    horizons: Array
    split: ProcessValidationSplit

    @classmethod
    def fit(
        cls,
        location: ArrayLike,
        scale: ArrayLike,
        target: ArrayLike,
        horizons: ArrayLike,
        split: ProcessValidationSplit,
        /,
        *,
        case_axis: int = 0,
        horizon_axis: int = 1,
        mask: ArrayLike | None = None,
    ) -> HorizonScaleCalibrator:
        if not isinstance(split, ProcessValidationSplit):
            raise TypeError("split must be a ProcessValidationSplit.")
        location_values = jnp.asarray(location, dtype=float)
        scale_values = jnp.asarray(scale, dtype=float)
        target_values = jnp.asarray(target, dtype=float)
        if (
            location_values.shape != target_values.shape
            or scale_values.shape != target_values.shape
            or target_values.ndim < 2
        ):
            raise ValueError(
                "location, scale, and target must have equal rank-two-or-higher shapes."
            )
        case_position = _axis(case_axis, target_values.ndim, name="case_axis")
        horizon_position = _axis(horizon_axis, target_values.ndim, name="horizon_axis")
        if case_position == horizon_position:
            raise ValueError("case_axis and horizon_axis must be distinct.")
        split.require_case_count(
            int(target_values.shape[case_position]), partition="calibration"
        )
        permutation = (case_position, horizon_position) + tuple(
            index
            for index in range(target_values.ndim)
            if index not in (case_position, horizon_position)
        )
        location_values = jnp.transpose(location_values, permutation)
        scale_values = jnp.transpose(scale_values, permutation)
        target_values = jnp.transpose(target_values, permutation)
        active = jnp.ones_like(target_values, dtype=bool)
        if mask is not None:
            active = jnp.transpose(
                jnp.broadcast_to(jnp.asarray(mask, dtype=bool), jnp.shape(target)),
                permutation,
            )
        if bool(
            jnp.any(
                active
                & (
                    ~jnp.isfinite(location_values)
                    | ~jnp.isfinite(target_values)
                    | ~jnp.isfinite(scale_values)
                    | (scale_values <= 0.0)
                )
            )
        ):
            raise ValueError(
                "Active calibration values must be finite with positive scale."
            )
        standardized_squared = ((target_values - location_values) / scale_values) ** 2
        reduction_axes = (0,) + tuple(range(2, target_values.ndim))
        counts = jnp.sum(active, axis=reduction_axes)
        if bool(jnp.any(counts <= 0)):
            raise ValueError(
                "Every horizon requires at least one active calibration value."
            )
        multiplier = jnp.sqrt(
            jnp.sum(jnp.where(active, standardized_squared, 0.0), axis=reduction_axes)
            / counts
        )
        if bool(jnp.any(~jnp.isfinite(multiplier) | (multiplier <= 0.0))):
            raise ValueError("Calibration data imply invalid horizon scale multipliers.")
        horizon_values = jnp.asarray(horizons, dtype=float)
        if horizon_values.ndim != 1 or horizon_values.shape != multiplier.shape:
            raise ValueError("horizons must align with horizon_axis.")
        if bool(jnp.any(~jnp.isfinite(horizon_values))) or bool(
            jnp.any(jnp.diff(horizon_values) <= 0.0)
        ):
            raise ValueError("horizons must be finite and strictly increasing.")
        return cls(multiplier, horizon_values, split)

    def __init__(
        self,
        scale_multiplier: ArrayLike,
        horizons: ArrayLike,
        split: ProcessValidationSplit,
    ):
        multiplier = jnp.asarray(scale_multiplier, dtype=float)
        horizon_values = jnp.asarray(horizons, dtype=float)
        if multiplier.ndim != 1 or multiplier.shape != horizon_values.shape:
            raise ValueError("scale_multiplier and horizons must be equal vectors.")
        if bool(jnp.any(~jnp.isfinite(multiplier) | (multiplier <= 0.0))):
            raise ValueError("scale_multiplier must be finite and positive.")
        if not isinstance(split, ProcessValidationSplit):
            raise TypeError("split must be a ProcessValidationSplit.")
        self.scale_multiplier = multiplier
        self.horizons = horizon_values
        self.split = split

    def transform(self, scale: ArrayLike, /, *, horizon_axis: int = 1) -> Array:
        values = jnp.asarray(scale, dtype=float)
        position = _axis(horizon_axis, values.ndim, name="horizon_axis")
        if int(values.shape[position]) != int(self.scale_multiplier.shape[0]):
            raise ValueError("scale horizon axis does not match fitted horizons.")
        shape = [1] * values.ndim
        shape[position] = int(self.scale_multiplier.shape[0])
        return values * self.scale_multiplier.reshape(tuple(shape))


ConformalCalibrator = FunctionalConformal | NormalizedConformal | SplitConformal


class ProcessConformalCalibrator(StrictModule):
    """Case-split conformal calibration for whole trajectories or observables."""

    calibrator: ConformalCalibrator
    split: ProcessValidationSplit
    kind: ProcessConformalKind = eqx.field(static=True)
    observable_name: str | None = eqx.field(static=True)

    def __init__(
        self,
        calibrator: ConformalCalibrator,
        split: ProcessValidationSplit,
        /,
        *,
        kind: ProcessConformalKind,
        observable_name: str | None,
    ):
        if not isinstance(
            calibrator, (FunctionalConformal, NormalizedConformal, SplitConformal)
        ):
            raise TypeError("calibrator must be a supported conformal calibrator.")
        if not isinstance(split, ProcessValidationSplit):
            raise TypeError("split must be a ProcessValidationSplit.")
        if kind not in ("trajectory", "observable"):
            raise ValueError("kind must be 'trajectory' or 'observable'.")
        name = None if observable_name is None else str(observable_name)
        if kind == "trajectory" and name is not None:
            raise ValueError("Trajectory conformal calibration has no observable_name.")
        if kind == "observable" and not name:
            raise ValueError(
                "Observable conformal calibration requires an observable_name."
            )
        self.calibrator = calibrator
        self.split = split
        self.kind = kind
        self.observable_name = name

    @classmethod
    def calibrate_trajectory(
        cls,
        center: cx.Field | ArrayLike,
        target: cx.Field | ArrayLike,
        split: ProcessValidationSplit,
        /,
        *,
        alpha: float,
        case_axis: int | str = 0,
        time_axis: int | str = 1,
        scale: cx.Field | ArrayLike | None = None,
        min_scale: float = 1e-8,
        mask: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        score: Literal["max", "l2"] = "max",
    ) -> ProcessConformalCalibrator:
        if not isinstance(split, ProcessValidationSplit):
            raise TypeError("split must be a ProcessValidationSplit.")
        count = _case_count(target, case_axis)
        split.require_case_count(count, partition="calibration")
        if isinstance(target, cx.Field):
            if isinstance(time_axis, str):
                if time_axis not in target.named_shape:
                    raise ValueError(f"Unknown time dimension {time_axis!r}.")
                if time_axis == case_axis:
                    raise ValueError("case_axis and time_axis must be distinct.")
            elif not isinstance(case_axis, str) and _axis(
                time_axis, target.data.ndim, name="time_axis"
            ) == _axis(case_axis, target.data.ndim, name="case_axis"):
                raise ValueError("case_axis and time_axis must be distinct.")
        else:
            if isinstance(time_axis, str) or isinstance(case_axis, str):
                raise TypeError("String axes require coordax.Field inputs.")
            target_rank = jnp.asarray(target).ndim
            if _axis(time_axis, target_rank, name="time_axis") == _axis(
                case_axis, target_rank, name="case_axis"
            ):
                raise ValueError("case_axis and time_axis must be distinct.")
        calibrator = FunctionalConformal.calibrate(
            center,
            target,
            alpha=alpha,
            case_dim=case_axis,
            scale=scale,
            min_scale=min_scale,
            mask=mask,
            weights=weights,
            score=score,
        )
        return cls(
            calibrator,
            split,
            kind="trajectory",
            observable_name=None,
        )

    @classmethod
    def calibrate_observable(
        cls,
        center: cx.Field | ArrayLike,
        target: cx.Field | ArrayLike,
        split: ProcessValidationSplit,
        /,
        *,
        observable_name: str,
        alpha: float,
        case_axis: int | str = 0,
        scale: cx.Field | ArrayLike | None = None,
        min_scale: float = 1e-8,
        mask: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        score: Literal["max", "l2"] = "max",
    ) -> ProcessConformalCalibrator:
        if not isinstance(split, ProcessValidationSplit):
            raise TypeError("split must be a ProcessValidationSplit.")
        count = _case_count(target, case_axis)
        split.require_case_count(count, partition="calibration")
        rank = (
            target.data.ndim if isinstance(target, cx.Field) else jnp.asarray(target).ndim
        )
        if rank == 1:
            if scale is None:
                calibrator: ConformalCalibrator = SplitConformal.calibrate(
                    center,
                    target,
                    alpha=alpha,
                    case_dim=case_axis,
                    mask=mask,
                )
            else:
                calibrator = NormalizedConformal.calibrate(
                    center,
                    scale,
                    target,
                    alpha=alpha,
                    case_dim=case_axis,
                    min_scale=min_scale,
                    mask=mask,
                )
        else:
            calibrator = FunctionalConformal.calibrate(
                center,
                target,
                alpha=alpha,
                case_dim=case_axis,
                scale=scale,
                min_scale=min_scale,
                mask=mask,
                weights=weights,
                score=score,
            )
        return cls(
            calibrator,
            split,
            kind="observable",
            observable_name=observable_name,
        )

    def interval(
        self,
        center: cx.Field | ArrayLike,
        scale: cx.Field | ArrayLike | None = None,
        /,
    ) -> PredictionInterval:
        if isinstance(self.calibrator, SplitConformal):
            if scale is not None:
                raise ValueError("This conformal calibrator was fitted without scale.")
            return self.calibrator.interval(center)
        if isinstance(self.calibrator, NormalizedConformal):
            if scale is None:
                raise ValueError("This conformal calibrator requires scale.")
            return self.calibrator.interval(center, scale)
        return self.calibrator.interval(center, scale)


class ProcessConformalDiagnostics(StrictModule):
    """Held-out complete-case coverage for one process conformal calibrator."""

    covered: Array
    valid: Array
    empirical_coverage: Array
    standard_error: Array
    lower: Array
    upper: Array
    nominal_coverage: float = eqx.field(static=True)
    confidence: float = eqx.field(static=True)
    kind: ProcessConformalKind = eqx.field(static=True)
    observable_name: str | None = eqx.field(static=True)


def process_conformal_diagnostics(
    calibrator: ProcessConformalCalibrator,
    center: cx.Field | ArrayLike,
    target: cx.Field | ArrayLike,
    /,
    *,
    case_axis: int | str = 0,
    scale: cx.Field | ArrayLike | None = None,
    mask: ArrayLike | None = None,
    confidence: float = 0.95,
) -> ProcessConformalDiagnostics:
    """Evaluate simultaneous coverage on independent physical test cases."""

    if not isinstance(calibrator, ProcessConformalCalibrator):
        raise TypeError("calibrator must be a ProcessConformalCalibrator.")
    count = _case_count(target, case_axis)
    calibrator.split.require_case_count(count, partition="test")
    interval = calibrator.interval(center, scale)
    target_values = jnp.asarray(target.data if isinstance(target, cx.Field) else target)
    lower_values = jnp.asarray(interval.lower.data)
    upper_values = jnp.asarray(interval.upper.data)
    if (
        lower_values.shape != target_values.shape
        or upper_values.shape != target_values.shape
    ):
        raise ValueError("Conformal interval and test targets must have equal shapes.")
    if isinstance(case_axis, str):
        if not isinstance(target, cx.Field):
            raise TypeError("A string case_axis requires coordax.Field targets.")
        position = target.dims.index(case_axis)
    else:
        position = _axis(case_axis, target_values.ndim, name="case_axis")
    active = jnp.ones_like(target_values, dtype=bool)
    if mask is not None:
        active = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), target_values.shape)
    finite = (
        jnp.isfinite(target_values)
        & jnp.isfinite(lower_values)
        & jnp.isfinite(upper_values)
    )
    location_covered = (target_values >= lower_values) & (target_values <= upper_values)
    reduction_axes = tuple(
        index for index in range(target_values.ndim) if index != position
    )
    valid = jnp.all(finite | ~active, axis=reduction_axes) & jnp.any(
        active, axis=reduction_axes
    )
    covered = jnp.all(location_covered | ~active, axis=reduction_axes) & valid
    valid_count = jnp.sum(valid)
    if int(valid_count) < 2:
        raise ValueError("Conformal diagnostics require at least two valid test cases.")
    empirical = jnp.sum(covered) / valid_count
    standard_error = jnp.sqrt(empirical * (1.0 - empirical) / valid_count)
    level = _confidence(confidence)
    half_width = _normal_quantile(level) * standard_error
    return ProcessConformalDiagnostics(
        covered=covered,
        valid=valid,
        empirical_coverage=empirical,
        standard_error=standard_error,
        lower=jnp.maximum(empirical - half_width, 0.0),
        upper=jnp.minimum(empirical + half_width, 1.0),
        nominal_coverage=interval.nominal_coverage,
        confidence=level,
        kind=calibrator.kind,
        observable_name=calibrator.observable_name,
    )


class ProcessCalibrationReport(StrictModule):
    """Raw and calibrated held-out scores retained side by side."""

    raw_horizon: HorizonScoreDiagnostics
    calibrated_horizon: HorizonScoreDiagnostics
    raw_trajectory: TrajectoryScoreDiagnostics
    calibrated_trajectory: TrajectoryScoreDiagnostics
    raw_pointwise_coverage_error: Array
    calibrated_pointwise_coverage_error: Array
    raw_pointwise_coverage_error_upper: Array
    calibrated_pointwise_coverage_error_upper: Array
    split: ProcessValidationSplit
    confidence: float = eqx.field(static=True)


def process_calibration_report(
    raw_samples: ArrayLike,
    calibrated_samples: ArrayLike,
    targets: ArrayLike,
    horizons: ArrayLike,
    split: ProcessValidationSplit,
    /,
    *,
    sample_axis: int = 0,
    case_axis: int | None = 0,
    time_axis: int = 1,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    variogram_power: float = 0.5,
    max_lag: int | None = None,
    confidence: float = 0.95,
) -> ProcessCalibrationReport:
    """Compare raw and calibrated distributions only on the declared test cases."""

    if not isinstance(split, ProcessValidationSplit):
        raise TypeError("split must be a ProcessValidationSplit.")
    target_values = jnp.asarray(targets)
    if case_axis is None:
        if len(split.test_case_ids) != 1:
            raise ValueError("case_axis=None requires exactly one declared test case.")
    else:
        case_position = _axis(case_axis, target_values.ndim, name="case_axis")
        split.require_case_count(
            int(target_values.shape[case_position]), partition="test"
        )
    if jnp.shape(raw_samples) != jnp.shape(calibrated_samples):
        raise ValueError("Raw and calibrated sample arrays must have equal shapes.")
    raw_horizon = horizon_score_diagnostics(
        raw_samples,
        targets,
        horizons,
        sample_axis=sample_axis,
        case_axis=case_axis,
        horizon_axis=time_axis,
        mask=mask,
        weights=weights,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    calibrated_horizon = horizon_score_diagnostics(
        calibrated_samples,
        targets,
        horizons,
        sample_axis=sample_axis,
        case_axis=case_axis,
        horizon_axis=time_axis,
        mask=mask,
        weights=weights,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    raw_trajectory = trajectory_score_diagnostics(
        raw_samples,
        targets,
        sample_axis=sample_axis,
        case_axis=case_axis,
        time_axis=time_axis,
        mask=mask,
        weights=weights,
        variogram_power=variogram_power,
        max_lag=max_lag,
    )
    calibrated_trajectory = trajectory_score_diagnostics(
        calibrated_samples,
        targets,
        sample_axis=sample_axis,
        case_axis=case_axis,
        time_axis=time_axis,
        mask=mask,
        weights=weights,
        variogram_power=variogram_power,
        max_lag=max_lag,
    )
    nominal = float(upper_quantile) - float(lower_quantile)
    raw_error = jnp.abs(raw_horizon.pointwise_coverage - nominal)
    calibrated_error = jnp.abs(calibrated_horizon.pointwise_coverage - nominal)
    level = _confidence(confidence)
    raw_radius = jnp.sqrt(
        jnp.log(2.0 / (1.0 - level)) / (2.0 * jnp.maximum(raw_horizon.valid_cases, 1))
    )
    calibrated_radius = jnp.sqrt(
        jnp.log(2.0 / (1.0 - level))
        / (2.0 * jnp.maximum(calibrated_horizon.valid_cases, 1))
    )
    return ProcessCalibrationReport(
        raw_horizon=raw_horizon,
        calibrated_horizon=calibrated_horizon,
        raw_trajectory=raw_trajectory,
        calibrated_trajectory=calibrated_trajectory,
        raw_pointwise_coverage_error=raw_error,
        calibrated_pointwise_coverage_error=calibrated_error,
        raw_pointwise_coverage_error_upper=raw_error + raw_radius,
        calibrated_pointwise_coverage_error_upper=calibrated_error + calibrated_radius,
        split=split,
        confidence=level,
    )


class ProcessShiftEvaluationMatrix(StrictModule):
    """Seed-replicated evaluation over required stochastic distribution shifts."""

    raw_score: MonteCarloEstimate
    calibrated_score: MonteCarloEstimate
    raw_coverage: MonteCarloEstimate
    calibrated_coverage: MonteCarloEstimate
    raw_score_degradation: MonteCarloEstimate
    calibrated_score_degradation: MonteCarloEstimate
    raw_coverage_error: MonteCarloEstimate
    calibrated_coverage_error: MonteCarloEstimate
    paired_reference_excess: MonteCarloEstimate | None
    scenario_names: tuple[str, ...] = eqx.field(static=True)
    shift_kinds: tuple[ProcessShiftKind, ...] = eqx.field(static=True)
    seeds: tuple[int, ...] = eqx.field(static=True)
    baseline_index: int = eqx.field(static=True)
    nominal_coverage: float = eqx.field(static=True)
    required_shifts: tuple[ProcessShiftKind, ...] = eqx.field(static=True)

    @property
    def worst_calibrated_score_degradation_upper(self) -> Array:
        shifted = tuple(
            index
            for index in range(len(self.scenario_names))
            if index != self.baseline_index
        )
        return jnp.max(self.calibrated_score_degradation.upper[jnp.asarray(shifted)])


def process_shift_evaluation_matrix(
    raw_scores: ArrayLike,
    calibrated_scores: ArrayLike,
    raw_coverages: ArrayLike,
    calibrated_coverages: ArrayLike,
    /,
    *,
    scenario_names: Sequence[str],
    shift_kinds: Sequence[ProcessShiftKind],
    seeds: Sequence[int],
    baseline_index: int = 0,
    nominal_coverage: float = 0.9,
    paired_reference_scores: ArrayLike | None = None,
    required_shifts: Sequence[ProcessShiftKind] = (
        "rollout_horizon",
        "covariance",
        "initial_condition",
        "parameter_regime",
    ),
    confidence: float = 0.95,
) -> ProcessShiftEvaluationMatrix:
    """Aggregate seed-level shift outcomes without inventing unpaired confidence."""

    names = tuple(str(name) for name in scenario_names)
    kinds = tuple(str(kind) for kind in shift_kinds)
    seed_values = tuple(int(seed) for seed in seeds)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("scenario_names must be non-empty and unique.")
    valid_kinds = (
        "in_distribution",
        "rollout_horizon",
        "covariance",
        "initial_condition",
        "parameter_regime",
    )
    if len(kinds) != len(names) or any(kind not in valid_kinds for kind in kinds):
        raise ValueError("shift_kinds must contain one valid kind per scenario.")
    if len(seed_values) < 2 or len(set(seed_values)) != len(seed_values):
        raise ValueError("At least two unique independent seeds are required.")
    baseline = int(baseline_index)
    if not 0 <= baseline < len(names) or kinds[baseline] != "in_distribution":
        raise ValueError("baseline_index must select an in_distribution scenario.")
    required = tuple(str(kind) for kind in required_shifts)
    if len(set(required)) != len(required) or any(
        kind not in valid_kinds[1:] for kind in required
    ):
        raise ValueError("required_shifts must be unique supported shift kinds.")
    missing = tuple(kind for kind in required if kind not in kinds)
    if missing:
        raise ValueError(f"Shift evaluation is missing required scenarios {missing!r}.")
    nominal = float(nominal_coverage)
    if not 0.0 < nominal < 1.0:
        raise ValueError("nominal_coverage must lie strictly between zero and one.")

    shape = (len(seed_values), len(names))
    raw_score_values = jnp.asarray(raw_scores, dtype=float)
    calibrated_score_values = jnp.asarray(calibrated_scores, dtype=float)
    raw_coverage_values = jnp.asarray(raw_coverages, dtype=float)
    calibrated_coverage_values = jnp.asarray(calibrated_coverages, dtype=float)
    arrays = (
        raw_score_values,
        calibrated_score_values,
        raw_coverage_values,
        calibrated_coverage_values,
    )
    if any(value.shape != shape for value in arrays):
        raise ValueError(
            "Shift metric arrays must have shape (num_seeds, num_scenarios)."
        )
    if any(bool(jnp.any(~jnp.isfinite(value))) for value in arrays):
        raise ValueError("Shift metric arrays must be finite.")
    if bool(jnp.any(raw_score_values < 0.0)) or bool(
        jnp.any(calibrated_score_values < 0.0)
    ):
        raise ValueError("Shift scores must be non-negative.")
    if bool(
        jnp.any(
            (raw_coverage_values < 0.0)
            | (raw_coverage_values > 1.0)
            | (calibrated_coverage_values < 0.0)
            | (calibrated_coverage_values > 1.0)
        )
    ):
        raise ValueError("Shift coverages must lie between zero and one.")

    raw_baseline = raw_score_values[:, baseline, None]
    calibrated_baseline = calibrated_score_values[:, baseline, None]
    raw_degradation = (raw_score_values - raw_baseline) / jnp.maximum(
        jnp.abs(raw_baseline), 1e-12
    )
    calibrated_degradation = (
        calibrated_score_values - calibrated_baseline
    ) / jnp.maximum(jnp.abs(calibrated_baseline), 1e-12)
    paired_excess = None
    if paired_reference_scores is not None:
        reference = jnp.asarray(paired_reference_scores, dtype=float)
        if reference.shape != shape or bool(jnp.any(~jnp.isfinite(reference))):
            raise ValueError(
                "paired_reference_scores must be a finite shift metric matrix."
            )
        paired_excess = monte_carlo_estimate(
            calibrated_score_values - reference,
            confidence=confidence,
        )
    return ProcessShiftEvaluationMatrix(
        raw_score=monte_carlo_estimate(raw_score_values, confidence=confidence),
        calibrated_score=monte_carlo_estimate(
            calibrated_score_values, confidence=confidence
        ),
        raw_coverage=monte_carlo_estimate(raw_coverage_values, confidence=confidence),
        calibrated_coverage=monte_carlo_estimate(
            calibrated_coverage_values, confidence=confidence
        ),
        raw_score_degradation=monte_carlo_estimate(
            raw_degradation, confidence=confidence
        ),
        calibrated_score_degradation=monte_carlo_estimate(
            calibrated_degradation, confidence=confidence
        ),
        raw_coverage_error=monte_carlo_estimate(
            jnp.abs(raw_coverage_values - nominal), confidence=confidence
        ),
        calibrated_coverage_error=monte_carlo_estimate(
            jnp.abs(calibrated_coverage_values - nominal), confidence=confidence
        ),
        paired_reference_excess=paired_excess,
        scenario_names=names,
        shift_kinds=kinds,  # type: ignore[arg-type]
        seeds=seed_values,
        baseline_index=baseline,
        nominal_coverage=nominal,
        required_shifts=required,  # type: ignore[arg-type]
    )


class ProcessRetentionThresholds(StrictModule):
    """Caller-controlled statistical release limits for stochastic processes."""

    max_mean_relative_error: float = eqx.field(static=True)
    max_covariance_relative_error: float = eqx.field(static=True)
    max_semigroup_excess_upper: float = eqx.field(static=True)
    max_calibrated_coverage_error_upper: float = eqx.field(static=True)
    max_shift_score_degradation_upper: float = eqx.field(static=True)
    max_variance_remainder_ratio: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_mean_relative_error: float = 0.1,
        max_covariance_relative_error: float = 0.2,
        max_semigroup_excess_upper: float = 0.05,
        max_calibrated_coverage_error_upper: float = 0.1,
        max_shift_score_degradation_upper: float = 0.25,
        max_variance_remainder_ratio: float = 1e-6,
    ):
        values = {
            "max_mean_relative_error": max_mean_relative_error,
            "max_covariance_relative_error": max_covariance_relative_error,
            "max_semigroup_excess_upper": max_semigroup_excess_upper,
            "max_calibrated_coverage_error_upper": max_calibrated_coverage_error_upper,
            "max_shift_score_degradation_upper": max_shift_score_degradation_upper,
            "max_variance_remainder_ratio": max_variance_remainder_ratio,
        }
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in values.values()
        ):
            raise ValueError(
                "Process retention thresholds must be finite and non-negative."
            )
        for name, value in values.items():
            object.__setattr__(self, name, float(value))


class ProcessRetentionReport(StrictModule):
    """Auditable process promotion decision retaining every failed gate."""

    thresholds: ProcessRetentionThresholds
    passed: bool = eqx.field(static=True)
    failures: tuple[str, ...] = eqx.field(static=True)
    deterministic_replay: bool = eqx.field(static=True)
    stable_realization_ids: bool = eqx.field(static=True)
    rough_path_replay: bool = eqx.field(static=True)
    broken_reference_rejected: bool = eqx.field(static=True)
    raw_results_retained: bool = eqx.field(static=True)
    calibrated_results_retained: bool = eqx.field(static=True)
    numerical_refinement_paired: bool = eqx.field(static=True)
    uncertainty_sources: tuple[str, ...] = eqx.field(static=True)
    mean_relative_error: Array
    covariance_relative_error: Array
    semigroup_excess_upper: Array
    calibrated_coverage_error_upper: Array
    shift_score_degradation_upper: Array
    variance_remainder_ratio: Array

    def raise_for_failure(self) -> None:
        if not self.passed:
            raise RuntimeError(
                "Stochastic process retention gates failed: "
                + ", ".join(self.failures)
                + "."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "deterministic_replay": self.deterministic_replay,
            "stable_realization_ids": self.stable_realization_ids,
            "rough_path_replay": self.rough_path_replay,
            "broken_reference_rejected": self.broken_reference_rejected,
            "raw_results_retained": self.raw_results_retained,
            "calibrated_results_retained": self.calibrated_results_retained,
            "numerical_refinement_paired": self.numerical_refinement_paired,
            "uncertainty_sources": self.uncertainty_sources,
            "mean_relative_error": float(self.mean_relative_error),
            "covariance_relative_error": float(self.covariance_relative_error),
            "semigroup_excess_upper": float(self.semigroup_excess_upper),
            "calibrated_coverage_error_upper": float(
                self.calibrated_coverage_error_upper
            ),
            "shift_score_degradation_upper": float(self.shift_score_degradation_upper),
            "variance_remainder_ratio": float(self.variance_remainder_ratio),
        }


def process_retention_report(
    *,
    temporal: TemporalMomentDiagnostics,
    semigroup: SemigroupMonteCarloDiagnostics,
    calibration: ProcessCalibrationReport,
    shifts: ProcessShiftEvaluationMatrix,
    decomposition: PredictiveVarianceDecomposition,
    deterministic_replay: bool,
    stable_realization_ids: bool,
    rough_path_replay: bool,
    broken_reference_rejected: bool,
    raw_results_retained: bool,
    calibrated_results_retained: bool,
    uncertainty_sources: Sequence[UncertaintySource],
    numerical_refinement: PairedNumericalUncertainty | None = None,
    thresholds: ProcessRetentionThresholds | None = None,
) -> ProcessRetentionReport:
    """Gate process promotion on statistical evidence and provenance invariants."""

    if not isinstance(temporal, TemporalMomentDiagnostics):
        raise TypeError("temporal must be TemporalMomentDiagnostics.")
    if temporal.mean_relative_error is None or temporal.covariance_relative_error is None:
        raise ValueError(
            "Temporal retention requires analytic mean and covariance references."
        )
    if not isinstance(semigroup, SemigroupMonteCarloDiagnostics):
        raise TypeError("semigroup must be SemigroupMonteCarloDiagnostics.")
    if semigroup.excess is None:
        raise ValueError("Semigroup retention requires a paired reference law.")
    if not isinstance(calibration, ProcessCalibrationReport):
        raise TypeError("calibration must be a ProcessCalibrationReport.")
    if not isinstance(shifts, ProcessShiftEvaluationMatrix):
        raise TypeError("shifts must be a ProcessShiftEvaluationMatrix.")
    if not isinstance(decomposition, PredictiveVarianceDecomposition):
        raise TypeError("decomposition must be PredictiveVarianceDecomposition.")
    limits = thresholds or ProcessRetentionThresholds()
    if not isinstance(limits, ProcessRetentionThresholds):
        raise TypeError("thresholds must be ProcessRetentionThresholds.")
    sources = tuple(str(source) for source in uncertainty_sources)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("uncertainty_sources must be non-empty and unique.")
    invalid_sources = tuple(
        source for source in sources if source not in UNCERTAINTY_SOURCES
    )
    if invalid_sources:
        raise ValueError(f"Unknown uncertainty sources: {invalid_sources!r}.")
    numerical_paired = numerical_refinement is not None
    if "numerical" in sources and not numerical_paired:
        raise ValueError(
            "Numerical uncertainty retention requires paired refinement evidence."
        )
    if numerical_refinement is not None and not isinstance(
        numerical_refinement, PairedNumericalUncertainty
    ):
        raise TypeError(
            "numerical_refinement must be PairedNumericalUncertainty or None."
        )

    mean_error = jnp.asarray(temporal.mean_relative_error)
    covariance_error = jnp.asarray(temporal.covariance_relative_error)
    semigroup_upper = jnp.max(jnp.asarray(semigroup.excess.upper))
    calibration_upper = jnp.max(
        jnp.asarray(calibration.calibrated_pointwise_coverage_error_upper)
    )
    shift_upper = shifts.worst_calibrated_score_degradation_upper
    total = jnp.asarray(decomposition.total)
    remainder = jnp.asarray(decomposition.remainder)
    remainder_ratio = jnp.max(jnp.abs(remainder)) / jnp.maximum(
        jnp.max(jnp.abs(total)), 1e-12
    )

    failures: list[str] = []
    boolean_gates = (
        ("deterministic_replay", deterministic_replay),
        ("stable_realization_ids", stable_realization_ids),
        ("rough_path_replay", rough_path_replay),
        ("broken_reference_rejected", broken_reference_rejected),
        ("raw_results_retained", raw_results_retained),
        ("calibrated_results_retained", calibrated_results_retained),
    )
    failures.extend(name for name, passed in boolean_gates if not bool(passed))
    if bool(jnp.any(~jnp.isfinite(mean_error))) or bool(
        jnp.max(mean_error) > limits.max_mean_relative_error
    ):
        failures.append("mean_relative_error")
    if bool(jnp.any(~jnp.isfinite(covariance_error))) or bool(
        jnp.max(covariance_error) > limits.max_covariance_relative_error
    ):
        failures.append("covariance_relative_error")
    if not bool(jnp.isfinite(semigroup_upper)) or bool(
        semigroup_upper > limits.max_semigroup_excess_upper
    ):
        failures.append("semigroup_excess_upper")
    if not bool(jnp.isfinite(calibration_upper)) or bool(
        calibration_upper > limits.max_calibrated_coverage_error_upper
    ):
        failures.append("calibrated_coverage_error_upper")
    if not bool(jnp.isfinite(shift_upper)) or bool(
        shift_upper > limits.max_shift_score_degradation_upper
    ):
        failures.append("shift_score_degradation_upper")
    if not bool(jnp.isfinite(remainder_ratio)) or bool(
        remainder_ratio > limits.max_variance_remainder_ratio
    ):
        failures.append("variance_remainder_ratio")

    return ProcessRetentionReport(
        thresholds=limits,
        passed=not failures,
        failures=tuple(failures),
        deterministic_replay=bool(deterministic_replay),
        stable_realization_ids=bool(stable_realization_ids),
        rough_path_replay=bool(rough_path_replay),
        broken_reference_rejected=bool(broken_reference_rejected),
        raw_results_retained=bool(raw_results_retained),
        calibrated_results_retained=bool(calibrated_results_retained),
        numerical_refinement_paired=numerical_paired,
        uncertainty_sources=sources,
        mean_relative_error=jnp.max(mean_error),
        covariance_relative_error=jnp.max(covariance_error),
        semigroup_excess_upper=semigroup_upper,
        calibrated_coverage_error_upper=calibration_upper,
        shift_score_degradation_upper=shift_upper,
        variance_remainder_ratio=remainder_ratio,
    )


__all__ = [
    "first_passage_diagnostics",
    "FirstPassageDiagnostics",
    "HorizonScaleCalibrator",
    "jump_event_diagnostics",
    "JumpEventDiagnostics",
    "JumpEventSummary",
    "paired_refinement_uncertainty",
    "PairedNumericalUncertainty",
    "process_calibration_report",
    "ProcessCalibrationReport",
    "ProcessConformalCalibrator",
    "process_conformal_diagnostics",
    "ProcessConformalDiagnostics",
    "process_retention_report",
    "ProcessRetentionReport",
    "ProcessRetentionThresholds",
    "process_shift_evaluation_matrix",
    "ProcessShiftEvaluationMatrix",
    "ProcessShiftKind",
    "ProcessValidationSplit",
    "trajectory_score_diagnostics",
    "TrajectoryScoreDiagnostics",
]
