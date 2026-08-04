#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..stochastic._jump import AbstractJumpProcess, JumpEventBatch
from ..stochastic._martingale import MartingaleIncrements


def _confidence(value: float, /) -> float:
    resolved = float(value)
    if not 0.0 < resolved < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    return resolved


def _groups(labels: Sequence[str | None], /) -> tuple[tuple[int, ...], ...]:
    if any(label is None for label in labels):
        raise ValueError(
            "Confidence-aware martingale diagnostics require explicit independence labels."
        )
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        assert label is not None
        grouped.setdefault(label, []).append(index)
    return tuple(tuple(indices) for indices in grouped.values())


def _cluster_statistics(
    values: Array,
    valid: Array,
    groups: tuple[tuple[int, ...], ...],
    /,
) -> tuple[Array, Array, Array, Array]:
    """Aggregate ``(trajectory, item, event)`` arrays by independent cluster."""
    summaries = []
    cluster_valid = []
    for indices in groups:
        selected = jnp.asarray(indices, dtype=jnp.int32)
        selected_values = values[selected]
        selected_valid = valid[selected]
        count = jnp.sum(selected_valid, axis=0)
        total = jnp.sum(
            jnp.where(selected_valid[..., None], selected_values, 0.0), axis=0
        )
        summaries.append(total / jnp.maximum(count[..., None], 1))
        cluster_valid.append(count > 0)
    stacked = jnp.stack(summaries, axis=0)
    usable = jnp.stack(cluster_valid, axis=0)
    count = jnp.sum(usable, axis=0)
    mean = jnp.sum(jnp.where(usable[..., None], stacked, 0.0), axis=0) / jnp.maximum(
        count[..., None], 1
    )
    centered = jnp.where(usable[..., None], stacked - mean[None, ...], 0.0)
    variance = jnp.sum(centered**2, axis=0) / jnp.maximum(count[..., None] - 1, 1)
    standard_error = jnp.sqrt(variance / jnp.maximum(count[..., None], 1))
    standard_error = jnp.where(count[..., None] >= 2, standard_error, jnp.nan)
    return mean, standard_error, count, usable


class MartingaleDiagnostics(StrictModule):
    """Cluster-aware predictable-instrument martingale moment diagnostics."""

    moments: Array
    standard_errors: Array
    lower: Array
    upper: Array
    standardized: Array
    valid_fraction: Array
    independent_clusters: Array
    source_indices: Array
    target_indices: Array
    instrument_names: tuple[str, ...] = eqx.field(static=True)
    observable_shape: tuple[int, ...] = eqx.field(static=True)
    confidence: float = eqx.field(static=True)
    minimum_clusters: int = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        critical = float(jsp.special.ndtri(0.5 + 0.5 * self.confidence))
        enough = self.independent_clusters >= self.minimum_clusters
        finite = jnp.isfinite(self.standardized)
        return bool(jnp.all(enough)) and bool(
            jnp.all(jnp.where(finite, jnp.abs(self.standardized) <= critical, False))
        )


def martingale_diagnostics(
    residuals: MartingaleIncrements,
    instruments: Mapping[str, Callable[[Array, Array], ArrayLike]] | None = None,
    /,
    *,
    source_indices: ArrayLike | None = None,
    target_indices: ArrayLike | None = None,
    confidence: float = 0.95,
    minimum_clusters: int = 2,
) -> MartingaleDiagnostics:
    """Test E[h(X_s)(M_t-M_s)]=0 over explicit source/target pairs."""
    if not isinstance(residuals, MartingaleIncrements):
        raise TypeError("residuals must be MartingaleIncrements.")
    level = _confidence(confidence)
    minimum = int(minimum_clusters)
    if minimum < 2:
        raise ValueError("minimum_clusters must be at least two.")
    resolved = {"constant": lambda _state, _time: jnp.asarray(1.0)}
    if instruments is not None:
        resolved = dict(instruments)
        if not resolved or any(
            not isinstance(name, str) or not name or not callable(function)
            for name, function in resolved.items()
        ):
            raise ValueError("instruments must map non-empty names to callables.")

    num_times = residuals.trajectory.num_times
    if source_indices is None and target_indices is None:
        sources = jnp.arange(num_times - 1, dtype=jnp.int32)
        targets = sources + 1
    elif source_indices is None or target_indices is None:
        raise ValueError("source_indices and target_indices must be supplied together.")
    else:
        sources = jnp.asarray(source_indices, dtype=jnp.int32).reshape((-1,))
        targets = jnp.asarray(target_indices, dtype=jnp.int32).reshape((-1,))
    if sources.shape != targets.shape or sources.size == 0:
        raise ValueError(
            "source_indices and target_indices must be equal non-empty vectors."
        )
    if bool(
        jnp.any(sources < 0) | jnp.any(targets >= num_times) | jnp.any(targets <= sources)
    ):
        raise ValueError(
            "Every source/target pair must satisfy 0 <= source < target < num_times."
        )

    trajectory = residuals.trajectory
    leading_count = prod(trajectory.leading_shape) if trajectory.leading_shape else 1
    event_size = prod(residuals.observable_shape) if residuals.observable_shape else 1
    groups = _groups(trajectory.independence_ids)
    cumulative = residuals.cumulative.reshape((leading_count, num_times, event_size))
    differences = cumulative[:, targets] - cumulative[:, sources]
    pair_valid = jnp.stack(
        tuple(
            jnp.all(residuals.interval_valid[..., int(source) : int(target)], axis=-1)
            for source, target in zip(
                np.asarray(sources), np.asarray(targets), strict=True
            )
        ),
        axis=-1,
    ).reshape((leading_count, -1))

    state_shape = trajectory.state_shape
    states = trajectory.states.reshape((leading_count, num_times) + state_shape)[
        :, sources
    ]
    times = trajectory.times.reshape((leading_count, num_times))[:, sources]
    flat_states = states.reshape((-1,) + state_shape)
    flat_times = times.reshape((-1,))
    moments = []
    errors = []
    cluster_counts = []
    valid_fractions = []
    for function in resolved.values():
        instrument = jax.vmap(lambda state, time: jnp.asarray(function(state, time)))(
            flat_states, flat_times
        )
        if instrument.shape[1:] not in ((), residuals.observable_shape):
            raise ValueError(
                "Instruments must return scalars or arrays matching observable_shape."
            )
        instrument = instrument.reshape((leading_count, sources.size, -1))
        if instrument.shape[-1] == 1:
            instrument = jnp.broadcast_to(instrument, differences.shape)
        values = instrument * differences
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        valid = pair_valid & finite
        mean, standard_error, count, _ = _cluster_statistics(values, valid, groups)
        moments.append(mean)
        errors.append(standard_error)
        cluster_counts.append(count)
        valid_fractions.append(jnp.mean(valid, axis=0))

    moment_values = jnp.stack(moments, axis=0)
    standard_errors = jnp.stack(errors, axis=0)
    counts = jnp.stack(cluster_counts, axis=0)
    fractions = jnp.stack(valid_fractions, axis=0)
    quantile = jsp.special.ndtri(jnp.asarray(0.5 + 0.5 * level))
    standardized = jnp.where(
        standard_errors > 0.0,
        moment_values / standard_errors,
        jnp.where(jnp.abs(moment_values) <= 1e-14, 0.0, jnp.inf),
    )
    return MartingaleDiagnostics(
        moments=moment_values,
        standard_errors=standard_errors,
        lower=moment_values - quantile * standard_errors,
        upper=moment_values + quantile * standard_errors,
        standardized=standardized,
        valid_fraction=fractions,
        independent_clusters=counts,
        source_indices=sources,
        target_indices=targets,
        instrument_names=tuple(resolved),
        observable_shape=residuals.observable_shape,
        confidence=level,
        minimum_clusters=minimum,
    )


class QuadraticVariationDiagnostics(StrictModule):
    """Observed-versus-predictable total quadratic-covariation diagnostics."""

    observed: Array
    predicted: Array
    difference: Array
    standard_error: Array
    lower: Array
    upper: Array
    independent_clusters: Array
    valid_fraction: Array
    event_size: int = eqx.field(static=True)
    confidence: float = eqx.field(static=True)
    minimum_clusters: int = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(jnp.all(self.independent_clusters >= self.minimum_clusters)) and bool(
            jnp.all((self.lower <= 0.0) & (self.upper >= 0.0))
        )


def quadratic_variation_diagnostics(
    residuals: MartingaleIncrements,
    predicted_bracket_increments: ArrayLike,
    /,
    *,
    confidence: float = 0.95,
    minimum_clusters: int = 2,
) -> QuadraticVariationDiagnostics:
    """Compare realized quadratic covariation with declared predictable brackets."""
    level = _confidence(confidence)
    minimum = int(minimum_clusters)
    if minimum < 2:
        raise ValueError("minimum_clusters must be at least two.")
    event_size = prod(residuals.observable_shape) if residuals.observable_shape else 1
    expected = residuals.leading_shape + (residuals.num_intervals, event_size, event_size)
    predicted = jnp.asarray(predicted_bracket_increments)
    if predicted.shape != expected:
        raise ValueError(
            f"predicted_bracket_increments must have shape {expected}; got {predicted.shape}."
        )
    flat_increments = residuals.increments.reshape(
        (-1, residuals.num_intervals, event_size)
    )
    observed_interval = (
        flat_increments[..., :, :, None] * flat_increments[..., :, None, :]
    )
    predicted_interval = predicted.reshape(observed_interval.shape)
    valid = residuals.interval_valid.reshape((-1, residuals.num_intervals))
    finite = jnp.all(jnp.isfinite(predicted_interval), axis=(-1, -2))
    interval_valid = valid & finite
    observed_total = jnp.sum(
        jnp.where(interval_valid[..., None, None], observed_interval, 0.0), axis=1
    )
    predicted_total = jnp.sum(
        jnp.where(interval_valid[..., None, None], predicted_interval, 0.0), axis=1
    )
    path_valid = jnp.any(interval_valid, axis=1)
    differences = (observed_total - predicted_total).reshape((-1, 1, event_size**2))
    mean, standard_error, count, _ = _cluster_statistics(
        differences,
        path_valid[:, None],
        _groups(residuals.trajectory.independence_ids),
    )
    observed_mean, _, _, _ = _cluster_statistics(
        observed_total.reshape((-1, 1, event_size**2)),
        path_valid[:, None],
        _groups(residuals.trajectory.independence_ids),
    )
    predicted_mean, _, _, _ = _cluster_statistics(
        predicted_total.reshape((-1, 1, event_size**2)),
        path_valid[:, None],
        _groups(residuals.trajectory.independence_ids),
    )
    quantile = jsp.special.ndtri(jnp.asarray(0.5 + 0.5 * level))
    result_shape = (event_size, event_size)
    difference = mean[0].reshape(result_shape)
    error = standard_error[0].reshape(result_shape)
    return QuadraticVariationDiagnostics(
        observed=observed_mean[0].reshape(result_shape),
        predicted=predicted_mean[0].reshape(result_shape),
        difference=difference,
        standard_error=error,
        lower=difference - quantile * error,
        upper=difference + quantile * error,
        independent_clusters=count[0],
        valid_fraction=jnp.mean(path_valid),
        event_size=event_size,
        confidence=level,
        minimum_clusters=minimum,
    )


class JumpCompensatorDiagnostics(StrictModule):
    """Channel-count compensator diagnostics for finite-activity jump paths."""

    channel_counts: Array
    integrated_intensities: Array
    compensated_counts: Array
    compensated_mean: Array
    standard_error: Array
    lower: Array
    upper: Array
    successful_paths: Array
    independent_clusters: Array
    process_id: str = eqx.field(static=True)
    confidence: float = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(self.independent_clusters >= 2) and bool(
            jnp.all((self.lower <= 0.0) & (self.upper >= 0.0))
        )


def jump_compensator_diagnostics(
    events: JumpEventBatch,
    process: AbstractJumpProcess,
    initial_state: ArrayLike,
    /,
    *,
    t0: float,
    t1: float,
    args: object = None,
    independence_labels: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> JumpCompensatorDiagnostics:
    """Compare event counts with left-state integrated channel intensities."""
    if not isinstance(events, JumpEventBatch):
        raise TypeError("events must be a JumpEventBatch.")
    if not isinstance(process, AbstractJumpProcess):
        raise TypeError("process must implement AbstractJumpProcess.")
    if events.pre_states is None or events.post_states is None:
        raise ValueError("Jump compensator diagnostics require stored event states.")
    if events.state_shape != process.state_shape:
        raise ValueError("Event and process state shapes must agree.")
    start, end = float(t0), float(t1)
    if not end > start:
        raise ValueError("t1 must be greater than t0.")
    initial = jnp.broadcast_to(
        jnp.asarray(initial_state), events.batch_shape + events.state_shape
    )
    path_count = prod(events.batch_shape) if events.batch_shape else 1
    flat_initial = initial.reshape((path_count,) + events.state_shape)
    flat_times = events.times.reshape((path_count, events.max_events))
    flat_post = events.post_states.reshape(
        (path_count, events.max_events) + events.state_shape
    )
    counts = events.counts.reshape((path_count,))
    integrals = jnp.zeros((path_count, process.num_channels), dtype=float)
    for segment in range(events.max_events + 1):
        active = segment <= counts
        if segment == 0:
            segment_start = jnp.full((path_count,), start)
            states = flat_initial
        else:
            segment_start = jnp.where(
                segment - 1 < counts,
                flat_times[:, segment - 1],
                end,
            )
            states = flat_post[:, segment - 1]
        segment_end = (
            jnp.where(segment < counts, flat_times[:, segment], end)
            if segment < events.max_events
            else jnp.full((path_count,), end)
        )
        rates = jax.vmap(lambda time, state: process.intensities(time, state, args))(
            segment_start, states
        )
        rates = jnp.asarray(rates)
        if rates.shape != (path_count, process.num_channels):
            raise ValueError(
                "Process intensities returned an incompatible channel shape."
            )
        duration = jnp.maximum(segment_end - segment_start, 0.0)
        integrals = integrals + jnp.where(active[:, None], rates * duration[:, None], 0.0)
    one_hot = jax.nn.one_hot(events.channels, process.num_channels, dtype=float)
    channel_counts = jnp.sum(one_hot * events.valid[..., None], axis=-2).reshape(
        (path_count, process.num_channels)
    )
    compensated = channel_counts - integrals
    successful = events.successful.reshape((path_count,))
    level = _confidence(confidence)
    if independence_labels is None:
        labels = tuple(f"path:{index}" for index in range(path_count))
    else:
        labels = tuple(independence_labels)
        if len(labels) != path_count:
            raise ValueError("independence_labels must contain one label per event path.")
    mean, error, count, _ = _cluster_statistics(
        compensated[:, None, :], successful[:, None], _groups(labels)
    )
    quantile = jsp.special.ndtri(jnp.asarray(0.5 + 0.5 * level))
    return JumpCompensatorDiagnostics(
        channel_counts=channel_counts.reshape(
            events.batch_shape + (process.num_channels,)
        ),
        integrated_intensities=integrals.reshape(
            events.batch_shape + (process.num_channels,)
        ),
        compensated_counts=compensated.reshape(
            events.batch_shape + (process.num_channels,)
        ),
        compensated_mean=mean[0],
        standard_error=error[0],
        lower=mean[0] - quantile * error[0],
        upper=mean[0] + quantile * error[0],
        successful_paths=jnp.sum(successful),
        independent_clusters=count[0],
        process_id=process.process_id,
        confidence=level,
    )


class MartingaleValidationReport(StrictModule):
    """Aggregate acceptance decision without discarding underlying diagnostics."""

    moments: MartingaleDiagnostics
    quadratic_variation: QuadraticVariationDiagnostics | None

    @property
    def passed(self) -> bool:
        return self.moments.passed and (
            self.quadratic_variation is None or self.quadratic_variation.passed
        )


def martingale_validation_report(
    moments: MartingaleDiagnostics,
    quadratic_variation: QuadraticVariationDiagnostics | None = None,
    /,
) -> MartingaleValidationReport:
    if not isinstance(moments, MartingaleDiagnostics):
        raise TypeError("moments must be MartingaleDiagnostics.")
    if quadratic_variation is not None and not isinstance(
        quadratic_variation, QuadraticVariationDiagnostics
    ):
        raise TypeError(
            "quadratic_variation must be QuadraticVariationDiagnostics or None."
        )
    return MartingaleValidationReport(
        moments=moments, quadratic_variation=quadratic_variation
    )


__all__ = [
    "jump_compensator_diagnostics",
    "JumpCompensatorDiagnostics",
    "martingale_diagnostics",
    "MartingaleDiagnostics",
    "martingale_validation_report",
    "MartingaleValidationReport",
    "quadratic_variation_diagnostics",
    "QuadraticVariationDiagnostics",
]
