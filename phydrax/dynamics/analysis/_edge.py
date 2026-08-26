#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from operator import index
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._geometry_precision import GeometryPrecisionPolicy
from ..._strict import StrictModule
from .._evolution import AbstractEvolution
from .._trajectory import TrajectoryData


RecurrenceSeedMetric: TypeAlias = Literal["euclidean", "supremum"]

EDGE_SUCCESS = 0
EDGE_INVALID_BRACKET = 1
EDGE_EVOLUTION_FAILED = 2
EDGE_NONFINITE = 3
EDGE_MAXIMUM_ITERATIONS = 4


class RecurrenceSeedCandidates(StrictModule):
    """Ordered near-recurrence pairs for periodic-orbit initialization."""

    source_indices: Array
    target_indices: Array
    source_coordinates: Array
    target_coordinates: Array
    periods: Array
    distances: Array
    source_states: Array
    valid: Array
    dataset_id: str = eqx.field(static=True)
    metric: RecurrenceSeedMetric = eqx.field(static=True)


def recurrence_seed_candidates(
    trajectory: TrajectoryData,
    count: int,
    /,
    *,
    minimum_separation: int = 1,
    metric: RecurrenceSeedMetric = "euclidean",
    maximum_pair_bytes: int = 512 * 1024**2,
) -> RecurrenceSeedCandidates:
    """Select the nearest valid, temporally separated pairs from one trajectory."""
    if not isinstance(trajectory, TrajectoryData):
        raise TypeError("trajectory must be TrajectoryData.")
    if trajectory.case_shape:
        raise ValueError("Recurrence seed extraction initially requires one trajectory.")
    if any(isinstance(value, bool) for value in (count, minimum_separation)):
        raise TypeError("count and minimum_separation must be integers.")
    selected_count = index(count)
    separation = index(minimum_separation)
    if selected_count < 1 or separation < 1:
        raise ValueError("count and minimum_separation must be positive.")
    if isinstance(maximum_pair_bytes, bool):
        raise TypeError("maximum_pair_bytes must be an integer.")
    maximum_bytes = index(maximum_pair_bytes)
    if maximum_bytes < 1:
        raise ValueError("maximum_pair_bytes must be positive.")
    if metric not in ("euclidean", "supremum"):
        raise ValueError("metric must be 'euclidean' or 'supremum'.")
    states = trajectory.states.reshape((trajectory.capacity, -1))
    available_pairs = max(trajectory.capacity - separation, 0)
    available_pairs = available_pairs * (available_pairs + 1) // 2
    if selected_count > available_pairs:
        raise ValueError("count exceeds the temporally separated pair capacity.")
    flattened_size = int(states.shape[-1])
    pair_count = trajectory.capacity**2
    real_itemsize = int(jnp.empty((), dtype=states.dtype).real.dtype.itemsize)
    required_bytes = pair_count * (
        flattened_size * int(states.dtype.itemsize) + 4 * real_itemsize + 16
    )
    if required_bytes > maximum_bytes:
        raise ValueError(
            f"Recurrence distances require {required_bytes} bytes; "
            f"maximum_pair_bytes={maximum_bytes}."
        )
    differences = states[:, None, :] - states[None, :, :]
    distances = (
        GeometryPrecisionPolicy().norm(differences, axis=-1)
        if metric == "euclidean"
        else jnp.max(jnp.abs(differences), axis=-1)
    )
    indices = jnp.arange(trajectory.capacity)
    admissible = (
        trajectory.sample_valid[:, None]
        & trajectory.sample_valid[None, :]
        & (indices[None, :] >= indices[:, None] + separation)
    )
    masked = jnp.where(admissible, distances, jnp.inf)
    flat_order = jnp.argsort(masked.reshape((-1,)))[:selected_count]
    sources = flat_order // trajectory.capacity
    targets = flat_order % trajectory.capacity
    selected_distances = masked[sources, targets]
    valid = jnp.isfinite(selected_distances)
    safe_sources = jnp.where(valid, sources, 0)
    safe_targets = jnp.where(valid, targets, 0)
    return RecurrenceSeedCandidates(
        source_indices=sources.astype(jnp.int32),
        target_indices=targets.astype(jnp.int32),
        source_coordinates=trajectory.coordinates[safe_sources],
        target_coordinates=trajectory.coordinates[safe_targets],
        periods=trajectory.coordinates[safe_targets]
        - trajectory.coordinates[safe_sources],
        distances=selected_distances,
        source_states=trajectory.states[safe_sources],
        valid=valid,
        dataset_id=trajectory.dataset_id,
        metric=metric,
    )


class EdgeTrackingProblem(StrictModule):
    """Finite-horizon basin-boundary bisection over one declared evolution."""

    evolution: AbstractEvolution
    classifier: Callable[[Array, Array, Any], ArrayLike]
    source_coordinate: Array
    target_coordinate: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        evolution: AbstractEvolution,
        classifier: Callable[[Array, Array, Any], ArrayLike],
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        /,
        *,
        problem_id: str | None = None,
    ):
        if not isinstance(evolution, AbstractEvolution):
            raise TypeError("evolution must be an AbstractEvolution.")
        if not callable(classifier):
            raise TypeError("classifier must be callable.")
        raw_source = jnp.asarray(source_coordinate)
        raw_target = jnp.asarray(target_coordinate)
        if jnp.iscomplexobj(raw_source) or jnp.iscomplexobj(raw_target):
            raise TypeError("Edge-tracking coordinates must be real.")
        source = raw_source.astype(float)
        target = raw_target.astype(float)
        if (
            source.shape != ()
            or target.shape != ()
            or not bool(jnp.isfinite(source) & jnp.isfinite(target) & (target > source))
        ):
            raise ValueError("Edge-tracking coordinates must be finite and increasing.")
        if problem_id is None:
            raise ValueError(
                "problem_id is required because classifier callables have no stable identity."
            )
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.evolution = evolution
        self.classifier = classifier
        self.source_coordinate = source
        self.target_coordinate = target
        self.problem_id = identifier


class EdgeTrackingResult(StrictModule):
    lower_state: Array
    upper_state: Array
    edge_state: Array
    lower_parameter: Array
    upper_parameter: Array
    classifier_values: Array
    bracket_widths: Array
    valid: Array
    converged: Array
    status: Array
    iterations: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)


def track_basin_edge(
    problem: EdgeTrackingProblem,
    lower_state: ArrayLike,
    upper_state: ArrayLike,
    /,
    *,
    iterations: int = 32,
    parameter_tolerance: float = 1e-8,
    args: Any = None,
) -> EdgeTrackingResult:
    """Bisect two opposite-outcome initial states over a fixed evolution horizon."""
    if not isinstance(problem, EdgeTrackingProblem):
        raise TypeError("problem must be an EdgeTrackingProblem.")
    if isinstance(iterations, bool):
        raise TypeError("iterations must be an integer.")
    steps = index(iterations)
    tolerance = float(parameter_tolerance)
    if steps < 1 or not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("iterations and parameter_tolerance must be positive.")
    lower = jnp.asarray(lower_state)
    upper = jnp.asarray(upper_state)
    expected = problem.evolution.state_layout.shape
    if lower.shape != expected or upper.shape != expected:
        raise ValueError(f"Edge bracket states must both have shape {expected}.")
    if lower.dtype != upper.dtype or not jnp.issubdtype(lower.dtype, jnp.inexact):
        raise TypeError("Edge bracket states must share one inexact dtype.")
    initial_state_finite = jnp.all(jnp.isfinite(lower)) & jnp.all(jnp.isfinite(upper))

    def classify(state):
        evolved = problem.evolution.advance(
            state,
            problem.source_coordinate,
            problem.target_coordinate,
            args,
        )
        value = jnp.asarray(
            problem.classifier(problem.target_coordinate, evolved.final_state, args)
        )
        value = value.reshape(())
        return value, evolved.valid & jnp.isfinite(value)

    lower_value, lower_valid = classify(lower)
    upper_value, upper_valid = classify(upper)
    reverse = (lower_value > 0.0) & (upper_value < 0.0)
    lower, upper = jax.lax.cond(
        reverse,
        lambda pair: (pair[1], pair[0]),
        lambda pair: pair,
        (lower, upper),
    )
    lower_value, upper_value = jax.lax.cond(
        reverse,
        lambda pair: (pair[1], pair[0]),
        lambda pair: pair,
        (lower_value, upper_value),
    )
    bracketed = (lower_value <= 0.0) & (upper_value >= 0.0)
    initial_evolution_valid = lower_valid & upper_valid
    initial_valid = initial_state_finite & initial_evolution_valid & bracketed
    classifier_history = jnp.full((steps,), jnp.nan, dtype=lower_value.dtype)
    width_history = jnp.full((steps,), jnp.nan, dtype=lower_value.dtype)

    def bisect(carry, index):
        (
            lower_current,
            upper_current,
            lower_parameter,
            upper_parameter,
            valid,
            values,
            widths,
        ) = carry
        midpoint = 0.5 * lower_current + 0.5 * upper_current
        middle_parameter = 0.5 * lower_parameter + 0.5 * upper_parameter
        middle_value, middle_valid = classify(midpoint)
        choose_lower = middle_value <= 0.0
        next_lower = jnp.where(choose_lower, midpoint, lower_current)
        next_upper = jnp.where(choose_lower, upper_current, midpoint)
        next_lower_parameter = jnp.where(choose_lower, middle_parameter, lower_parameter)
        next_upper_parameter = jnp.where(choose_lower, upper_parameter, middle_parameter)
        next_valid = valid & middle_valid
        values = values.at[index].set(middle_value)
        widths = widths.at[index].set(next_upper_parameter - next_lower_parameter)
        return (
            next_lower,
            next_upper,
            next_lower_parameter,
            next_upper_parameter,
            next_valid,
            values,
            widths,
        ), None

    initial = (
        lower,
        upper,
        jnp.asarray(0.0, dtype=lower_value.dtype),
        jnp.asarray(1.0, dtype=lower_value.dtype),
        initial_valid,
        classifier_history,
        width_history,
    )
    final, _ = jax.lax.scan(bisect, initial, jnp.arange(steps))
    finite = (
        jnp.all(jnp.isfinite(final[0]))
        & jnp.all(jnp.isfinite(final[1]))
        & jnp.all(jnp.isfinite(final[5]))
    )
    valid = final[4] & finite
    converged = valid & ((final[3] - final[2]) <= tolerance)
    status = jnp.where(
        ~finite | ~initial_state_finite,
        EDGE_NONFINITE,
        jnp.where(
            ~initial_evolution_valid,
            EDGE_EVOLUTION_FAILED,
            jnp.where(
                ~bracketed,
                EDGE_INVALID_BRACKET,
                jnp.where(
                    ~valid,
                    EDGE_EVOLUTION_FAILED,
                    jnp.where(converged, EDGE_SUCCESS, EDGE_MAXIMUM_ITERATIONS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return EdgeTrackingResult(
        lower_state=final[0],
        upper_state=final[1],
        edge_state=0.5 * final[0] + 0.5 * final[1],
        lower_parameter=final[2],
        upper_parameter=final[3],
        classifier_values=final[5],
        bracket_widths=final[6],
        valid=valid,
        converged=converged,
        status=status,
        iterations=steps,
        problem_id=problem.problem_id,
    )


__all__ = [
    "EDGE_EVOLUTION_FAILED",
    "EDGE_INVALID_BRACKET",
    "EDGE_MAXIMUM_ITERATIONS",
    "EDGE_NONFINITE",
    "EDGE_SUCCESS",
    "EdgeTrackingProblem",
    "EdgeTrackingResult",
    "RecurrenceSeedCandidates",
    "RecurrenceSeedMetric",
    "recurrence_seed_candidates",
    "track_basin_edge",
]
