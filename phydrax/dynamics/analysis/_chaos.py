#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._evolution import AbstractEvolution
from .._grid import EvolutionGrid, IterationGrid, TimeGrid
from .._trajectory import TrajectoryData


DistanceMetric: TypeAlias = Literal["euclidean", "supremum"]
SurrogateMethod: TypeAlias = Literal["shuffle", "phase_randomized", "aaft"]
SurrogateAlternative: TypeAlias = Literal["greater", "less", "two_sided"]
ChaosUncertaintySource: TypeAlias = Literal[
    "initial_condition", "parameter", "noise", "numerics", "process", "other"
]

CHAOS_DIAGNOSTIC_SUCCESS = 0
CHAOS_DIAGNOSTIC_INSUFFICIENT_SAMPLES = 1
CHAOS_DIAGNOSTIC_NONFINITE = 2
CHAOS_DIAGNOSTIC_FIT_FAILED = 3
CHAOS_DIAGNOSTIC_EVOLUTION_FAILED = 4


class FiniteSizeGrowthResult(StrictModule):
    report_coordinates: Array
    growth_rates: Array
    separations: Array
    report_valid: Array
    average_growth_rates: Array
    final_reference_state: Array
    final_perturbed_states: Array
    initial_directions: Array
    valid: Array
    status: Array
    evolution: AbstractEvolution
    grid: EvolutionGrid
    perturbation_distance: float = eqx.field(static=True)
    rescale_interval: int = eqx.field(static=True)
    distance_geometry: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    evolution_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)


class RecurrenceQuantificationResult(StrictModule):
    recurrence: Array
    eligible: Array
    distances: Array
    recurrence_rate: Array
    determinism: Array
    laminarity: Array
    average_diagonal_length: Array
    longest_diagonal: Array
    diagonal_entropy: Array
    trapping_time: Array
    longest_vertical: Array
    divergence: Array
    diagonal_length_histogram: Array
    vertical_length_histogram: Array
    valid: Array
    status: Array
    radius: Array
    dataset_id: str = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    metric: DistanceMetric = eqx.field(static=True)
    theiler_window: int = eqx.field(static=True)
    minimum_diagonal_length: int = eqx.field(static=True)
    minimum_vertical_length: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class ZeroOneTestResult(StrictModule):
    statistic: Array
    frequency_statistics: Array
    frequencies: Array
    displacement: Array
    displacement_valid: Array
    used_sample_mask: Array
    segment_start: Array
    segment_end: Array
    median_absolute_deviation: Array
    valid: Array
    status: Array
    dataset_id: str = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    burn_in: int = eqx.field(static=True)
    fit_lag_start: int = eqx.field(static=True)
    fit_lag_end: int = eqx.field(static=True)
    num_frequencies: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class CorrelationDimensionResult(StrictModule):
    dimension: Array
    intercept: Array
    r_squared: Array
    correlation_sum: Array
    local_slope: Array
    fit_mask: Array
    eligible_pair_count: Array
    sample_mask: Array
    radii: Array
    valid: Array
    status: Array
    dataset_id: str = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    metric: DistanceMetric = eqx.field(static=True)
    theiler_window: int = eqx.field(static=True)
    fit_index_start: int = eqx.field(static=True)
    fit_index_end: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class SurrogateSignificanceResult(StrictModule):
    observed_statistic: Array
    surrogate_statistics: Array
    p_value: Array
    z_score: Array
    surrogate_quantiles: Array
    used_sample_mask: Array
    valid: Array
    status: Array
    method: SurrogateMethod = eqx.field(static=True)
    alternative: SurrogateAlternative = eqx.field(static=True)
    statistic_id: str = eqx.field(static=True)
    num_surrogates: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    segment_start: int = eqx.field(static=True)
    segment_end: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class ChaosUncertaintyResult(StrictModule):
    samples: Array
    sample_valid: Array
    sample_weights: Array
    mean: Array
    standard_deviation: Array
    quantiles: Array
    bootstrap_interval: Array
    bootstrap_means: Array
    effective_sample_size: Array
    source_variance: Array
    valid: Array
    status: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    metric_names: tuple[str, ...] = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    source_kinds: tuple[ChaosUncertaintySource, ...] = eqx.field(static=True)
    confidence: float = eqx.field(static=True)
    bootstrap_samples: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


def _distance_matrix(values: np.ndarray, metric: DistanceMetric, /) -> np.ndarray:
    difference = values[:, None, :] - values[None, :, :]
    if metric == "euclidean":
        return np.linalg.norm(difference, axis=-1)
    return np.max(np.abs(difference), axis=-1)


def _runs(values: np.ndarray, /) -> list[int]:
    padded = np.concatenate((np.asarray([False]), values, np.asarray([False])))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        int(changes[index + 1] - changes[index]) for index in range(0, len(changes), 2)
    ]


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    levels: np.ndarray,
    /,
) -> np.ndarray:
    result = np.empty((levels.size, values.shape[1]), dtype=float)
    for metric in range(values.shape[1]):
        order = np.argsort(values[:, metric])
        ordered_values = values[order, metric]
        ordered_weights = weights[order]
        cumulative = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / np.sum(
            ordered_weights
        )
        result[:, metric] = np.interp(
            levels,
            cumulative,
            ordered_values,
            left=ordered_values[0],
            right=ordered_values[-1],
        )
    return result


def _longest_segment(
    sample_valid: np.ndarray,
    transition_valid: np.ndarray | None,
    /,
) -> tuple[int, int]:
    best_start = 0
    best_end = 0
    start = 0
    size = int(sample_valid.size)
    while start < size:
        while start < size and not sample_valid[start]:
            start += 1
        end = start
        while end < size and sample_valid[end]:
            if (
                end > start
                and transition_valid is not None
                and not transition_valid[end - 1]
            ):
                break
            end += 1
        if end - start > best_end - best_start:
            best_start, best_end = start, end
        start = max(end, start + 1)
    return best_start, best_end


def finite_size_growth(
    evolution: AbstractEvolution,
    initial_state: ArrayLike,
    grid: EvolutionGrid,
    /,
    *,
    args: Any = None,
    directions: ArrayLike | None = None,
    num_directions: int | None = None,
    seed: int = 0,
    perturbation_distance: float = 1e-4,
    rescale_interval: int = 1,
) -> FiniteSizeGrowthResult:
    """Measure finite-amplitude growth with geometric separation and rescaling."""
    if not isinstance(evolution, AbstractEvolution):
        raise TypeError("evolution must be an AbstractEvolution.")
    if not isinstance(grid, (TimeGrid, IterationGrid)):
        raise TypeError("grid must be a TimeGrid or IterationGrid.")
    state = jnp.asarray(initial_state)
    if state.shape != evolution.state_layout.shape:
        raise ValueError("initial_state has the wrong shape.")
    distance = float(perturbation_distance)
    cadence = int(rescale_interval)
    if not np.isfinite(distance) or distance <= 0.0 or cadence < 1:
        raise ValueError("perturbation_distance and rescale_interval must be positive.")
    dimension = evolution.state_layout.size
    if directions is None:
        count = dimension if num_directions is None else int(num_directions)
        if count < 1:
            raise ValueError("num_directions must be positive.")
        vectors = jax.random.normal(
            jax.random.PRNGKey(int(seed)),
            (count,) + evolution.state_layout.shape,
            dtype=jnp.result_type(state, float),
        )
    else:
        vectors = jnp.asarray(directions)
        if vectors.ndim < 1 or vectors.shape[1:] != evolution.state_layout.shape:
            raise ValueError("directions must have shape (direction,) + state_shape.")
        count = int(vectors.shape[0])
        if num_directions is not None and int(num_directions) != count:
            raise ValueError("num_directions conflicts with supplied directions.")
    flat = vectors.reshape((count, -1))
    norms = jnp.linalg.norm(flat, axis=1)
    if bool(jnp.any(~jnp.isfinite(norms)) | jnp.any(norms == 0.0)):
        raise ValueError("directions must be finite and nonzero.")
    unit = (flat / norms[:, None]).reshape(vectors.shape)
    geometry = evolution.state_layout.geometry
    perturbed = jax.vmap(lambda vector: geometry.retract(state, distance * vector))(unit)
    report_steps = [
        index
        for index in range(1, grid.num_steps + 1)
        if index % cadence == 0 or index == grid.num_steps
    ]
    growth = jnp.full((count, len(report_steps)), jnp.nan)
    separations = jnp.full((count, len(report_steps)), jnp.nan)
    report_valid = jnp.zeros((count, len(report_steps)), dtype=bool)
    accumulated_log = jnp.zeros((count,), dtype=jnp.result_type(state, float))
    accumulated_time = jnp.asarray(0.0, dtype=grid.coordinates.dtype)
    interval_start = grid.coordinates[0]
    run_valid = jnp.ones((count,), dtype=bool)
    report_index = 0
    reference = state
    for step_index in range(grid.num_steps):
        source = grid.coordinates[step_index]
        target = grid.coordinates[step_index + 1]
        reference_step = evolution.advance(reference, source, target, args)
        next_reference = reference_step.final_state

        def advance_perturbed(value):
            step = evolution.advance(value, source, target, args)
            return step.final_state, step.valid

        next_perturbed, perturbation_valid = jax.vmap(advance_perturbed)(perturbed)
        run_valid = (
            run_valid
            & reference_step.valid
            & perturbation_valid
            & jnp.all(jnp.isfinite(next_perturbed).reshape((count, -1)), axis=1)
        )
        reference = next_reference
        perturbed = next_perturbed
        if step_index + 1 in report_steps:
            local = jax.vmap(lambda point: geometry.inverse_retract(reference, point))(
                perturbed
            )
            local_flat = local.reshape((count, -1))
            current_separation = jnp.linalg.norm(local_flat, axis=1)
            elapsed = target - interval_start
            finite = (
                run_valid
                & jnp.isfinite(current_separation)
                & (current_separation > 0.0)
                & jnp.isfinite(elapsed)
                & (elapsed > 0.0)
            )
            logarithmic_growth = jnp.log(current_separation / distance)
            growth = growth.at[:, report_index].set(
                jnp.where(finite, logarithmic_growth / elapsed, jnp.nan)
            )
            separations = separations.at[:, report_index].set(current_separation)
            report_valid = report_valid.at[:, report_index].set(finite)
            accumulated_log = accumulated_log + jnp.where(finite, logarithmic_growth, 0.0)
            accumulated_time = accumulated_time + elapsed
            safe_norm = jnp.where(finite, current_separation, 1.0)
            rescaled = (local_flat / safe_norm[:, None] * distance).reshape(local.shape)
            perturbed = jax.vmap(lambda vector: geometry.retract(reference, vector))(
                rescaled
            )
            interval_start = target
            report_index += 1
    direction_complete = jnp.all(report_valid, axis=1)
    average = jnp.where(
        direction_complete & (accumulated_time > 0.0),
        accumulated_log / accumulated_time,
        jnp.nan,
    )
    valid = jnp.all(direction_complete) & jnp.all(jnp.isfinite(average))
    status = jnp.where(
        valid,
        CHAOS_DIAGNOSTIC_SUCCESS,
        jnp.where(
            jnp.any(~run_valid),
            CHAOS_DIAGNOSTIC_EVOLUTION_FAILED,
            CHAOS_DIAGNOSTIC_NONFINITE,
        ),
    ).astype(jnp.int32)
    return FiniteSizeGrowthResult(
        report_coordinates=grid.coordinates[jnp.asarray(report_steps)],
        growth_rates=growth,
        separations=separations,
        report_valid=report_valid,
        average_growth_rates=average,
        final_reference_state=reference,
        final_perturbed_states=perturbed,
        initial_directions=unit,
        valid=valid,
        status=status,
        evolution=evolution,
        grid=grid,
        perturbation_distance=distance,
        rescale_interval=cadence,
        distance_geometry=(f"local-retraction-euclidean-norm:{geometry.geometry_id}"),
        method_id="finite-size-rescaled-separation",
        evolution_id=evolution.evolution_id,
        approximation_id=evolution.approximation_id,
    )


def recurrence_quantification(
    data: TrajectoryData,
    radius: ArrayLike,
    /,
    *,
    metric: DistanceMetric = "euclidean",
    theiler_window: int = 0,
    minimum_diagonal_length: int = 2,
    minimum_vertical_length: int = 2,
    max_samples: int = 4096,
) -> RecurrenceQuantificationResult:
    """Build recurrence masks and standard line-based RQA statistics per case."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    if metric not in ("euclidean", "supremum"):
        raise ValueError("Unsupported distance metric.")
    theiler = int(theiler_window)
    minimum_diagonal = int(minimum_diagonal_length)
    minimum_vertical = int(minimum_vertical_length)
    if theiler < 0 or minimum_diagonal < 1 or minimum_vertical < 1:
        raise ValueError("Theiler and line-length controls are invalid.")
    if data.capacity > int(max_samples):
        raise ValueError("Trajectory exceeds max_samples for dense recurrence analysis.")
    radius_value = np.asarray(radius, dtype=float)
    expected_radius_shape = data.case_shape
    if radius_value.shape == ():
        radius_value = np.broadcast_to(radius_value, expected_radius_shape or (1,))
        if not data.case_shape:
            radius_value = radius_value.reshape(())
    if radius_value.shape != expected_radius_shape:
        raise ValueError("radius must be scalar or have case_shape.")
    if np.any(~np.isfinite(radius_value)) or np.any(radius_value <= 0.0):
        raise ValueError("radius must be finite and positive.")
    case_count = data.num_cases
    samples = data.capacity
    states = np.asarray(data.states).reshape((case_count, samples, -1))
    sample_valid = np.asarray(data.sample_valid).reshape((case_count, samples))
    radii = np.asarray(radius_value).reshape((case_count,))
    recurrence = np.zeros((case_count, samples, samples), dtype=bool)
    eligible = np.zeros_like(recurrence)
    distances = np.full((case_count, samples, samples), np.nan)
    scalar_outputs = [np.full((case_count,), np.nan) for _ in range(10)]
    diagonal_histogram = np.zeros((case_count, samples + 1), dtype=np.int32)
    vertical_histogram = np.zeros((case_count, samples + 1), dtype=np.int32)
    valid_output = np.zeros((case_count,), dtype=bool)
    status = np.full((case_count,), CHAOS_DIAGNOSTIC_INSUFFICIENT_SAMPLES, dtype=np.int32)
    index = np.arange(samples)
    theiler_mask = np.abs(index[:, None] - index[None, :]) > theiler
    for case in range(case_count):
        distance_matrix = _distance_matrix(states[case], metric)
        pair_valid = sample_valid[case, :, None] & sample_valid[case, None, :]
        eligible_case = pair_valid & theiler_mask
        recurrence_case = eligible_case & (distance_matrix <= radii[case])
        distances[case] = np.where(pair_valid, distance_matrix, np.nan)
        eligible[case] = eligible_case
        recurrence[case] = recurrence_case
        recurrence_points = int(np.sum(recurrence_case))
        eligible_points = int(np.sum(eligible_case))
        diagonal_lengths: list[int] = []
        for offset in range(-samples + 1, samples):
            if abs(offset) <= theiler:
                continue
            diagonal_lengths.extend(_runs(np.diag(recurrence_case, k=offset)))
        vertical_lengths: list[int] = []
        for column in range(samples):
            vertical_lengths.extend(_runs(recurrence_case[:, column]))
        for length in diagonal_lengths:
            diagonal_histogram[case, length] += 1
        for length in vertical_lengths:
            vertical_histogram[case, length] += 1
        long_diagonal = [
            length for length in diagonal_lengths if length >= minimum_diagonal
        ]
        long_vertical = [
            length for length in vertical_lengths if length >= minimum_vertical
        ]
        diagonal_points = sum(long_diagonal)
        vertical_points = sum(long_vertical)
        recurrence_rate = (
            recurrence_points / eligible_points if eligible_points else np.nan
        )
        determinism = diagonal_points / recurrence_points if recurrence_points else np.nan
        laminarity = vertical_points / recurrence_points if recurrence_points else np.nan
        average_diagonal = np.mean(long_diagonal) if long_diagonal else np.nan
        longest_diagonal = max(long_diagonal, default=0)
        if long_diagonal:
            counts = np.bincount(long_diagonal)
            probabilities = counts[counts > 0] / len(long_diagonal)
            entropy = -np.sum(probabilities * np.log(probabilities))
        else:
            entropy = np.nan
        trapping = np.mean(long_vertical) if long_vertical else np.nan
        longest_vertical = max(long_vertical, default=0)
        divergence = 1.0 / longest_diagonal if longest_diagonal else np.nan
        values = (
            recurrence_rate,
            determinism,
            laminarity,
            average_diagonal,
            float(longest_diagonal),
            entropy,
            trapping,
            float(longest_vertical),
            divergence,
            float(recurrence_points),
        )
        for output, value in zip(scalar_outputs, values, strict=True):
            output[case] = value
        valid_output[case] = eligible_points > 0 and np.isfinite(recurrence_rate)
        status[case] = (
            CHAOS_DIAGNOSTIC_SUCCESS
            if valid_output[case]
            else CHAOS_DIAGNOSTIC_INSUFFICIENT_SAMPLES
        )
    output_shape = data.case_shape

    def shaped(values):
        return jnp.asarray(values).reshape(output_shape)

    return RecurrenceQuantificationResult(
        recurrence=jnp.asarray(recurrence).reshape(output_shape + (samples, samples)),
        eligible=jnp.asarray(eligible).reshape(output_shape + (samples, samples)),
        distances=jnp.asarray(distances).reshape(output_shape + (samples, samples)),
        recurrence_rate=shaped(scalar_outputs[0]),
        determinism=shaped(scalar_outputs[1]),
        laminarity=shaped(scalar_outputs[2]),
        average_diagonal_length=shaped(scalar_outputs[3]),
        longest_diagonal=shaped(scalar_outputs[4]),
        diagonal_entropy=shaped(scalar_outputs[5]),
        trapping_time=shaped(scalar_outputs[6]),
        longest_vertical=shaped(scalar_outputs[7]),
        divergence=shaped(scalar_outputs[8]),
        diagonal_length_histogram=jnp.asarray(diagonal_histogram).reshape(
            output_shape + (samples + 1,)
        ),
        vertical_length_histogram=jnp.asarray(vertical_histogram).reshape(
            output_shape + (samples + 1,)
        ),
        valid=shaped(valid_output),
        status=shaped(status),
        radius=jnp.asarray(radius_value),
        dataset_id=data.dataset_id,
        case_shape=data.case_shape,
        metric=metric,
        theiler_window=theiler,
        minimum_diagonal_length=minimum_diagonal,
        minimum_vertical_length=minimum_vertical,
        method_id="dense-threshold-recurrence-rqa",
    )


def _observable_values(
    data: TrajectoryData,
    observable: Callable[[Array, Array], Array] | None,
    component: int | None,
    /,
) -> Array:
    if observable is not None and component is not None:
        raise ValueError("Specify observable or component, not both.")
    if observable is None:
        selected = 0 if component is None else int(component)
        if selected < 0 or selected >= data.state_layout.size:
            raise ValueError("component is out of range.")
        return data.states.reshape(data.case_shape + (data.capacity, -1))[..., selected]
    if not callable(observable):
        raise TypeError("observable must be callable or None.")
    values = jax.vmap(
        lambda coordinate, state: observable(coordinate, state),
        in_axes=(0, 0),
    )(
        data.coordinates.reshape((-1,)),
        data.states.reshape((-1,) + data.state_layout.shape),
    )
    values = jnp.asarray(values)
    if values.shape != (data.num_cases * data.capacity,):
        raise ValueError("observable must return one scalar per sample.")
    return values.reshape(data.case_shape + (data.capacity,))


def zero_one_test(
    data: TrajectoryData,
    /,
    *,
    observable: Callable[[Array, Array], Array] | None = None,
    component: int | None = None,
    observable_id: str = "state-component",
    burn_in: int = 0,
    num_frequencies: int = 100,
    seed: int = 0,
    fit_lags: tuple[int, int] | None = None,
    minimum_samples: int = 100,
) -> ZeroOneTestResult:
    """Run the correlation form of the modified Gottwald--Melbourne 0--1 test."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    burn = int(burn_in)
    frequency_count = int(num_frequencies)
    minimum = int(minimum_samples)
    if burn < 0 or frequency_count < 1 or minimum < 10:
        raise ValueError("Invalid 0-1 test sampling controls.")
    if not isinstance(observable_id, str) or not observable_id:
        raise ValueError("observable_id must be non-empty.")
    values = np.asarray(_observable_values(data, observable, component))
    values = values.reshape((data.num_cases, data.capacity))
    sample_valid = np.asarray(data.sample_valid).reshape((data.num_cases, data.capacity))
    transition_valid = np.asarray(data.transition_valid).reshape(
        (data.num_cases, data.capacity - 1)
    )
    rng = np.random.default_rng(int(seed))
    frequencies = rng.uniform(np.pi / 5.0, 4.0 * np.pi / 5.0, frequency_count)
    maximum_lag_capacity = max(1, data.capacity // 10)
    displacement = np.full(
        (data.num_cases, frequency_count, maximum_lag_capacity), np.nan
    )
    displacement_valid = np.zeros_like(displacement, dtype=bool)
    frequency_statistics = np.full((data.num_cases, frequency_count), np.nan)
    statistic = np.full((data.num_cases,), np.nan)
    deviation = np.full((data.num_cases,), np.nan)
    used_mask = np.zeros((data.num_cases, data.capacity), dtype=bool)
    starts = np.zeros((data.num_cases,), dtype=np.int32)
    ends = np.zeros((data.num_cases,), dtype=np.int32)
    result_valid = np.zeros((data.num_cases,), dtype=bool)
    statuses = np.full(
        (data.num_cases,), CHAOS_DIAGNOSTIC_INSUFFICIENT_SAMPLES, dtype=np.int32
    )
    resolved_fit_start = 1 if fit_lags is None else int(fit_lags[0])
    resolved_fit_end = maximum_lag_capacity if fit_lags is None else int(fit_lags[1])
    if resolved_fit_start < 1 or resolved_fit_end <= resolved_fit_start:
        raise ValueError("fit_lags must be an increasing positive index interval.")
    for case in range(data.num_cases):
        candidate_valid = sample_valid[case].copy()
        candidate_valid[: min(burn, data.capacity)] = False
        start, end = _longest_segment(candidate_valid, transition_valid[case])
        starts[case], ends[case] = start, end
        used_mask[case, start:end] = True
        series = values[case, start:end]
        size = int(series.size)
        maximum_lag = min(maximum_lag_capacity, max(1, size // 10))
        fit_start = resolved_fit_start
        fit_end = min(resolved_fit_end, maximum_lag)
        if size < minimum or fit_end - fit_start < 3 or not np.all(np.isfinite(series)):
            continue
        time_index = np.arange(1, size + 1, dtype=float)
        lags = np.arange(1, maximum_lag + 1, dtype=float)
        mean = float(np.mean(series))
        for frequency_index, frequency in enumerate(frequencies):
            phase = time_index * frequency
            translation_p = np.cumsum(series * np.cos(phase))
            translation_q = np.cumsum(series * np.sin(phase))
            mean_square = np.asarray(
                [
                    np.mean(
                        (translation_p[lag:] - translation_p[:-lag]) ** 2
                        + (translation_q[lag:] - translation_q[:-lag]) ** 2
                    )
                    for lag in range(1, maximum_lag + 1)
                ]
            )
            oscillatory = (
                mean**2 * (1.0 - np.cos(lags * frequency)) / (1.0 - np.cos(frequency))
            )
            modified = mean_square - oscillatory
            displacement[case, frequency_index, :maximum_lag] = modified
            displacement_valid[case, frequency_index, :maximum_lag] = np.isfinite(
                modified
            )
            fit_values = modified[fit_start - 1 : fit_end]
            fit_indices = lags[fit_start - 1 : fit_end]
            if np.std(fit_values) > 0.0 and np.all(np.isfinite(fit_values)):
                frequency_statistics[case, frequency_index] = np.corrcoef(
                    fit_indices, fit_values
                )[0, 1]
        finite_statistics = frequency_statistics[case][
            np.isfinite(frequency_statistics[case])
        ]
        if finite_statistics.size >= max(3, frequency_count // 2):
            statistic[case] = np.median(finite_statistics)
            deviation[case] = np.median(np.abs(finite_statistics - statistic[case]))
            result_valid[case] = True
            statuses[case] = CHAOS_DIAGNOSTIC_SUCCESS
        else:
            statuses[case] = CHAOS_DIAGNOSTIC_FIT_FAILED
    output_shape = data.case_shape
    return ZeroOneTestResult(
        statistic=jnp.asarray(statistic).reshape(output_shape),
        frequency_statistics=jnp.asarray(frequency_statistics).reshape(
            output_shape + (frequency_count,)
        ),
        frequencies=jnp.asarray(frequencies),
        displacement=jnp.asarray(displacement).reshape(
            output_shape + (frequency_count, maximum_lag_capacity)
        ),
        displacement_valid=jnp.asarray(displacement_valid).reshape(
            output_shape + (frequency_count, maximum_lag_capacity)
        ),
        used_sample_mask=jnp.asarray(used_mask).reshape(output_shape + (data.capacity,)),
        segment_start=jnp.asarray(starts).reshape(output_shape),
        segment_end=jnp.asarray(ends).reshape(output_shape),
        median_absolute_deviation=jnp.asarray(deviation).reshape(output_shape),
        valid=jnp.asarray(result_valid).reshape(output_shape),
        status=jnp.asarray(statuses).reshape(output_shape),
        dataset_id=data.dataset_id,
        case_shape=data.case_shape,
        burn_in=burn,
        fit_lag_start=resolved_fit_start,
        fit_lag_end=resolved_fit_end,
        num_frequencies=frequency_count,
        seed=int(seed),
        observable_id=observable_id,
        method_id="modified-zero-one-correlation-median",
    )


def correlation_dimension(
    data: TrajectoryData,
    radii: ArrayLike,
    /,
    *,
    metric: DistanceMetric = "euclidean",
    theiler_window: int = 0,
    fit_indices: tuple[int, int] | None = None,
    max_samples: int = 4096,
) -> CorrelationDimensionResult:
    """Estimate Grassberger--Procaccia correlation dimension on declared radii."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    if metric not in ("euclidean", "supremum"):
        raise ValueError("Unsupported distance metric.")
    if data.capacity > int(max_samples):
        raise ValueError("Trajectory exceeds max_samples for pairwise distances.")
    radius_values = np.asarray(radii, dtype=float)
    if (
        radius_values.ndim != 1
        or radius_values.size < 4
        or np.any(~np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or np.any(np.diff(radius_values) <= 0.0)
    ):
        raise ValueError("radii must be a strictly increasing positive rank-1 array.")
    radius_count = int(radius_values.size)
    fit_start = 0 if fit_indices is None else int(fit_indices[0])
    fit_end = radius_count if fit_indices is None else int(fit_indices[1])
    if fit_start < 0 or fit_end > radius_count or fit_end - fit_start < 3:
        raise ValueError("fit_indices must select at least three radii.")
    theiler = int(theiler_window)
    if theiler < 0:
        raise ValueError("theiler_window must be nonnegative.")
    cases = data.num_cases
    samples = data.capacity
    states = np.asarray(data.states).reshape((cases, samples, -1))
    sample_valid = np.asarray(data.sample_valid).reshape((cases, samples))
    dimensions = np.full((cases,), np.nan)
    intercepts = np.full((cases,), np.nan)
    r_squared = np.full((cases,), np.nan)
    correlation_sum = np.full((cases, radius_count), np.nan)
    local_slope = np.full((cases, radius_count), np.nan)
    fit_mask = np.zeros((cases, radius_count), dtype=bool)
    pair_count = np.zeros((cases,), dtype=np.int64)
    result_valid = np.zeros((cases,), dtype=bool)
    statuses = np.full((cases,), CHAOS_DIAGNOSTIC_INSUFFICIENT_SAMPLES, dtype=np.int32)
    row, column = np.triu_indices(samples, k=1)
    separated = np.abs(row - column) > theiler
    for case in range(cases):
        distances = _distance_matrix(states[case], metric)[row, column]
        eligible = separated & sample_valid[case, row] & sample_valid[case, column]
        pair_distances = distances[eligible]
        pair_distances = pair_distances[np.isfinite(pair_distances)]
        pair_count[case] = pair_distances.size
        if pair_distances.size < 3:
            continue
        sums = np.asarray([np.mean(pair_distances <= radius) for radius in radius_values])
        correlation_sum[case] = sums
        finite = (sums > 0.0) & (sums < 1.0)
        selected = finite.copy()
        selected[:fit_start] = False
        selected[fit_end:] = False
        fit_mask[case] = selected
        if np.sum(selected) < 3:
            statuses[case] = CHAOS_DIAGNOSTIC_FIT_FAILED
            continue
        log_radius = np.log(radius_values)
        log_sum = np.full(sums.shape, np.nan)
        log_sum[finite] = np.log(sums[finite])
        coefficients = np.polyfit(log_radius[selected], log_sum[selected], 1)
        prediction = np.polyval(coefficients, log_radius[selected])
        residual = log_sum[selected] - prediction
        centered = log_sum[selected] - np.mean(log_sum[selected])
        total = np.sum(centered**2)
        dimensions[case] = coefficients[0]
        intercepts[case] = coefficients[1]
        r_squared[case] = 1.0 - np.sum(residual**2) / total if total > 0.0 else np.nan
        finite_indices = np.flatnonzero(finite)
        if finite_indices.size >= 2:
            local_slope[case, finite_indices] = np.gradient(
                log_sum[finite_indices], log_radius[finite_indices]
            )
        result_valid[case] = np.isfinite(dimensions[case])
        statuses[case] = (
            CHAOS_DIAGNOSTIC_SUCCESS
            if result_valid[case]
            else CHAOS_DIAGNOSTIC_FIT_FAILED
        )
    output_shape = data.case_shape
    return CorrelationDimensionResult(
        dimension=jnp.asarray(dimensions).reshape(output_shape),
        intercept=jnp.asarray(intercepts).reshape(output_shape),
        r_squared=jnp.asarray(r_squared).reshape(output_shape),
        correlation_sum=jnp.asarray(correlation_sum).reshape(
            output_shape + (radius_count,)
        ),
        local_slope=jnp.asarray(local_slope).reshape(output_shape + (radius_count,)),
        fit_mask=jnp.asarray(fit_mask).reshape(output_shape + (radius_count,)),
        eligible_pair_count=jnp.asarray(pair_count).reshape(output_shape),
        sample_mask=data.sample_valid,
        radii=jnp.asarray(radius_values),
        valid=jnp.asarray(result_valid).reshape(output_shape),
        status=jnp.asarray(statuses).reshape(output_shape),
        dataset_id=data.dataset_id,
        case_shape=data.case_shape,
        metric=metric,
        theiler_window=theiler,
        fit_index_start=fit_start,
        fit_index_end=fit_end,
        method_id="grassberger-procaccia-pair-count-log-linear-fit",
    )


def _phase_randomized(values: np.ndarray, rng: np.random.Generator, /) -> np.ndarray:
    spectrum = np.fft.rfft(values)
    randomized = spectrum.copy()
    stop = -1 if values.size % 2 == 0 else None
    count = randomized[1:stop].size
    randomized[1:stop] *= np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, count))
    return np.fft.irfft(randomized, n=values.size)


def _aaft(values: np.ndarray, rng: np.random.Generator, /) -> np.ndarray:
    gaussian = np.sort(rng.normal(size=values.size))
    ranks = np.argsort(np.argsort(values))
    gaussianized = gaussian[ranks]
    randomized = _phase_randomized(gaussianized, rng)
    target = np.sort(values)
    randomized_ranks = np.argsort(np.argsort(randomized))
    return target[randomized_ranks]


def surrogate_significance(
    values: ArrayLike,
    statistic: Callable[[Array], Array],
    /,
    *,
    statistic_id: str,
    sample_valid: ArrayLike | None = None,
    method: SurrogateMethod = "phase_randomized",
    alternative: SurrogateAlternative = "greater",
    num_surrogates: int = 199,
    seed: int = 0,
    minimum_samples: int = 32,
) -> SurrogateSignificanceResult:
    """Evaluate a scalar statistic against an explicit surrogate null protocol."""
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("values must be rank one.")
    if not callable(statistic):
        raise TypeError("statistic must be callable.")
    if not isinstance(statistic_id, str) or not statistic_id:
        raise ValueError("statistic_id must be non-empty.")
    if method not in ("shuffle", "phase_randomized", "aaft"):
        raise ValueError("Unsupported surrogate method.")
    if alternative not in ("greater", "less", "two_sided"):
        raise ValueError("Unsupported surrogate alternative.")
    count = int(num_surrogates)
    if count < 1:
        raise ValueError("num_surrogates must be positive.")
    mask = np.isfinite(series)
    if sample_valid is not None:
        supplied = np.asarray(sample_valid, dtype=bool)
        if supplied.shape != series.shape:
            raise ValueError("sample_valid must have the series shape.")
        mask &= supplied
    start, end = _longest_segment(mask, None)
    used = np.zeros(series.shape, dtype=bool)
    used[start:end] = True
    segment = series[start:end]
    surrogate_statistics = np.full((count,), np.nan)
    observed = np.nan
    valid = segment.size >= int(minimum_samples)
    if valid:
        observed = float(jnp.asarray(statistic(jnp.asarray(segment))))
        valid = np.isfinite(observed)
    rng = np.random.default_rng(int(seed))
    if valid:
        for index in range(count):
            if method == "shuffle":
                surrogate = rng.permutation(segment)
            elif method == "phase_randomized":
                surrogate = _phase_randomized(segment, rng)
            else:
                surrogate = _aaft(segment, rng)
            surrogate_statistics[index] = float(
                jnp.asarray(statistic(jnp.asarray(surrogate)))
            )
        valid = np.all(np.isfinite(surrogate_statistics))
    if valid:
        if alternative == "greater":
            extreme = np.sum(surrogate_statistics >= observed)
        elif alternative == "less":
            extreme = np.sum(surrogate_statistics <= observed)
        else:
            center = np.median(surrogate_statistics)
            extreme = np.sum(
                np.abs(surrogate_statistics - center) >= abs(observed - center)
            )
        p_value = (extreme + 1.0) / (count + 1.0)
        standard_deviation = np.std(surrogate_statistics, ddof=1) if count > 1 else 0.0
        z_score = (
            (observed - np.mean(surrogate_statistics)) / standard_deviation
            if standard_deviation > 0.0
            else np.nan
        )
        quantiles = np.quantile(surrogate_statistics, [0.025, 0.5, 0.975])
        status = CHAOS_DIAGNOSTIC_SUCCESS
    else:
        p_value = np.nan
        z_score = np.nan
        quantiles = np.full((3,), np.nan)
        status = (
            CHAOS_DIAGNOSTIC_INSUFFICIENT_SAMPLES
            if segment.size < int(minimum_samples)
            else CHAOS_DIAGNOSTIC_NONFINITE
        )
    return SurrogateSignificanceResult(
        observed_statistic=jnp.asarray(observed),
        surrogate_statistics=jnp.asarray(surrogate_statistics),
        p_value=jnp.asarray(p_value),
        z_score=jnp.asarray(z_score),
        surrogate_quantiles=jnp.asarray(quantiles),
        used_sample_mask=jnp.asarray(used),
        valid=jnp.asarray(valid),
        status=jnp.asarray(status, dtype=jnp.int32),
        method=method,
        alternative=alternative,
        statistic_id=statistic_id,
        num_surrogates=count,
        seed=int(seed),
        segment_start=start,
        segment_end=end,
        method_id=f"surrogate-significance:{method}:{alternative}:plus-one-pvalue",
    )


def summarize_chaos_uncertainty(
    samples: ArrayLike,
    /,
    *,
    metric_names: Sequence[str],
    case_axes: Sequence[str],
    source_kinds: Sequence[ChaosUncertaintySource],
    sample_valid: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    confidence: float = 0.95,
    bootstrap_samples: int = 999,
    seed: int = 0,
) -> ChaosUncertaintyResult:
    """Aggregate scalar diagnostics across declared uncertainty-source case axes."""
    values = np.asarray(samples, dtype=float)
    names = tuple(str(name) for name in metric_names)
    axes = tuple(str(name) for name in case_axes)
    sources = tuple(source_kinds)
    if values.ndim < 1 or values.shape[-1] != len(names) or not names:
        raise ValueError("samples must have shape case_shape + (num_metrics,).")
    case_shape = tuple(int(size) for size in values.shape[:-1])
    if (
        len(axes) != len(case_shape)
        or len(sources) != len(case_shape)
        or any(not name for name in names + axes)
        or len(set(names)) != len(names)
        or len(set(axes)) != len(axes)
    ):
        raise ValueError("Metric names and uncertainty axes must be unique and complete.")
    allowed_sources = {
        "initial_condition",
        "parameter",
        "noise",
        "numerics",
        "process",
        "other",
    }
    if any(source not in allowed_sources for source in sources):
        raise ValueError("Unsupported uncertainty source kind.")
    confidence_value = float(confidence)
    draws = int(bootstrap_samples)
    if not 0.0 < confidence_value < 1.0 or draws < 1:
        raise ValueError("confidence and bootstrap_samples are invalid.")
    valid = np.all(np.isfinite(values), axis=-1)
    if sample_valid is not None:
        supplied_valid = np.asarray(sample_valid, dtype=bool)
        if supplied_valid.shape != case_shape:
            raise ValueError("sample_valid must have case_shape.")
        valid &= supplied_valid
    sample_weights = (
        np.ones(case_shape, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if sample_weights.shape != case_shape:
        raise ValueError("weights must have case_shape.")
    if np.any(~np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
        raise ValueError("weights must be finite and nonnegative.")
    sample_weights = np.where(valid, sample_weights, 0.0)
    flat_values = values.reshape((-1, len(names)))
    flat_valid = valid.reshape((-1,))
    flat_weights = sample_weights.reshape((-1,))
    positive = flat_valid & (flat_weights > 0.0)
    total_weight = np.sum(flat_weights[positive])
    result_valid = bool(np.any(positive) and total_weight > 0.0)
    if result_valid:
        normalized = flat_weights[positive] / total_weight
        selected = flat_values[positive]
        mean = np.sum(normalized[:, None] * selected, axis=0)
        variance = np.sum(normalized[:, None] * (selected - mean) ** 2, axis=0)
        standard_deviation = np.sqrt(np.maximum(variance, 0.0))
        quantile_levels = np.asarray(
            [(1.0 - confidence_value) / 2.0, 0.5, (1.0 + confidence_value) / 2.0]
        )
        quantiles = _weighted_quantiles(selected, normalized, quantile_levels)
        effective = 1.0 / np.sum(normalized**2)
        rng = np.random.default_rng(int(seed))
        bootstrap = np.empty((draws, len(names)))
        for draw in range(draws):
            indices = rng.choice(
                selected.shape[0],
                size=selected.shape[0],
                replace=True,
                p=normalized,
            )
            bootstrap[draw] = np.mean(selected[indices], axis=0)
        bootstrap_interval = np.quantile(
            bootstrap,
            [(1.0 - confidence_value) / 2.0, (1.0 + confidence_value) / 2.0],
            axis=0,
        )
        status = CHAOS_DIAGNOSTIC_SUCCESS
    else:
        mean = np.full((len(names),), np.nan)
        standard_deviation = np.full((len(names),), np.nan)
        quantiles = np.full((3, len(names)), np.nan)
        effective = np.nan
        bootstrap = np.full((draws, len(names)), np.nan)
        bootstrap_interval = np.full((2, len(names)), np.nan)
        status = CHAOS_DIAGNOSTIC_INSUFFICIENT_SAMPLES
    source_variance = np.full((len(case_shape), len(names)), np.nan)
    if result_valid:
        for axis in range(len(case_shape)):
            moved_values = np.moveaxis(values, axis, 0).reshape(
                (case_shape[axis], -1, len(names))
            )
            moved_valid = np.moveaxis(valid, axis, 0).reshape((case_shape[axis], -1))
            moved_weights = np.moveaxis(sample_weights, axis, 0).reshape(
                (case_shape[axis], -1)
            )
            level_means = np.full((case_shape[axis], len(names)), np.nan)
            level_weights = np.zeros((case_shape[axis],))
            for level in range(case_shape[axis]):
                selected_level = moved_valid[level] & (moved_weights[level] > 0.0)
                level_weights[level] = np.sum(moved_weights[level, selected_level])
                if level_weights[level] > 0.0:
                    level_means[level] = np.average(
                        moved_values[level, selected_level],
                        axis=0,
                        weights=moved_weights[level, selected_level],
                    )
            level_valid = level_weights > 0.0
            if np.any(level_valid):
                level_probability = level_weights[level_valid] / np.sum(
                    level_weights[level_valid]
                )
                source_variance[axis] = np.sum(
                    level_probability[:, None] * (level_means[level_valid] - mean) ** 2,
                    axis=0,
                )
    return ChaosUncertaintyResult(
        samples=jnp.asarray(values),
        sample_valid=jnp.asarray(valid),
        sample_weights=jnp.asarray(sample_weights),
        mean=jnp.asarray(mean),
        standard_deviation=jnp.asarray(standard_deviation),
        quantiles=jnp.asarray(quantiles),
        bootstrap_interval=jnp.asarray(bootstrap_interval),
        bootstrap_means=jnp.asarray(bootstrap),
        effective_sample_size=jnp.asarray(effective),
        source_variance=jnp.asarray(source_variance),
        valid=jnp.asarray(result_valid),
        status=jnp.asarray(status, dtype=jnp.int32),
        case_shape=case_shape,
        metric_names=names,
        case_axes=axes,
        source_kinds=sources,
        confidence=confidence_value,
        bootstrap_samples=draws,
        seed=int(seed),
        method_id="weighted-case-ensemble-percentile-bootstrap",
    )


__all__ = [
    "CHAOS_DIAGNOSTIC_EVOLUTION_FAILED",
    "CHAOS_DIAGNOSTIC_FIT_FAILED",
    "CHAOS_DIAGNOSTIC_INSUFFICIENT_SAMPLES",
    "CHAOS_DIAGNOSTIC_NONFINITE",
    "CHAOS_DIAGNOSTIC_SUCCESS",
    "ChaosUncertaintyResult",
    "ChaosUncertaintySource",
    "CorrelationDimensionResult",
    "DistanceMetric",
    "FiniteSizeGrowthResult",
    "RecurrenceQuantificationResult",
    "SurrogateAlternative",
    "SurrogateMethod",
    "SurrogateSignificanceResult",
    "ZeroOneTestResult",
    "correlation_dimension",
    "finite_size_growth",
    "recurrence_quantification",
    "summarize_chaos_uncertainty",
    "surrogate_significance",
    "zero_one_test",
]
