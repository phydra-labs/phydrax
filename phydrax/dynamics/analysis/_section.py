#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from .._evolution import AbstractEvolution, EvolutionTrajectory
from .._layout import StateLayout
from .._trajectory import TrajectoryData


SectionDirection: TypeAlias = Literal["any", "positive", "negative"]
SectionRefinement: TypeAlias = Literal["interpolation", "evolution"]

SECTION_SUCCESS = 0
SECTION_REFINEMENT_FAILED = 1
SECTION_EVOLUTION_FAILED = 2
SECTION_NONFINITE = 3


class AbstractSection(StrictModule):
    """Scalar section function over one declared state layout."""

    state_layout: AbstractAttribute[StateLayout]
    section_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError

    def __call__(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        return self.evaluate(coordinate, state, args)


class AffineSection(AbstractSection):
    """An oriented affine hyperplane `normal @ state = offset`."""

    normal: Array
    offset: Array
    state_layout: StateLayout
    section_id: str = eqx.field(static=True)

    def __init__(
        self,
        normal: ArrayLike,
        offset: ArrayLike = 0.0,
        /,
        *,
        state_layout: StateLayout,
        section_id: str | None = None,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        normal_values = jnp.asarray(normal)
        if normal_values.shape != state_layout.shape:
            raise ValueError(
                f"normal must have state shape {state_layout.shape}; got {normal_values.shape}."
            )
        offset_value = jnp.asarray(offset)
        if offset_value.shape != ():
            raise ValueError("offset must be scalar.")
        if not bool(
            jnp.all(jnp.isfinite(normal_values))
            & jnp.isfinite(offset_value)
            & (jnp.linalg.norm(normal_values.reshape((-1,))) > 0.0)
        ):
            raise ValueError("normal and offset must be finite and normal nonzero.")
        identifier = (
            "affine-section:"
            + canonical_fingerprint(
                {
                    "normal": np.asarray(normal_values).tolist(),
                    "offset": float(offset_value),
                    "layout": state_layout.layout_id,
                }
            )
            if section_id is None
            else str(section_id)
        )
        if not identifier:
            raise ValueError("section_id must be non-empty.")
        self.normal = normal_values
        self.offset = offset_value
        self.state_layout = state_layout
        self.section_id = identifier

    def evaluate(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        del coordinate, args
        state_value = jnp.asarray(state)
        if state_value.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_value.shape}."
            )
        return (
            jnp.vdot(self.normal.reshape((-1,)), state_value.reshape((-1,))).real
            - self.offset
        )


class CallableSection(AbstractSection):
    """User-defined scalar section with explicit layout and stable identity."""

    function: Callable[[Array, Array, Any], Array]
    state_layout: StateLayout
    section_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array, Array, Any], Array],
        /,
        *,
        state_layout: StateLayout,
        section_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if not isinstance(section_id, str) or not section_id:
            raise ValueError("section_id must be a non-empty string.")
        self.function = function
        self.state_layout = state_layout
        self.section_id = section_id

    def evaluate(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        state_value = jnp.asarray(state)
        if state_value.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_value.shape}."
            )
        value = jnp.asarray(self.function(jnp.asarray(coordinate), state_value, args))
        if value.shape != ():
            raise ValueError("Section function must return one scalar.")
        return value


class SectionCrossings(StrictModule):
    """Fixed-capacity section events with refinement and overflow evidence."""

    coordinates: Array
    states: Array
    section_values: Array
    bracket_start: Array
    bracket_end: Array
    detected: Array
    valid: Array
    converged: Array
    iterations: Array
    status: Array
    count: Array
    detected_count: Array
    overflow: Array
    state_layout: StateLayout
    case_shape: tuple[int, ...] = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    direction: SectionDirection = eqx.field(static=True)
    refinement: SectionRefinement = eqx.field(static=True)
    section_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class SectionReturnMap(StrictModule):
    """Successive valid section states and physical return intervals."""

    source_coordinates: Array
    target_coordinates: Array
    source_states: Array
    target_states: Array
    return_intervals: Array
    valid: Array
    state_layout: StateLayout
    lag: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    section_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


def _crossed(left: float, right: float, direction: SectionDirection, /) -> bool:
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    positive = left < 0.0 and right >= 0.0
    negative = left > 0.0 and right <= 0.0
    if direction == "positive":
        return positive
    if direction == "negative":
        return negative
    return positive or negative


def _source_arrays(
    trajectory: EvolutionTrajectory | TrajectoryData,
    /,
) -> tuple[Array, Array, Array, Array, tuple[int, ...], StateLayout, str]:
    if isinstance(trajectory, EvolutionTrajectory):
        coordinates = trajectory.grid.coordinates[None, :]
        states = trajectory.states[None, ...]
        valid = trajectory.valid[None, :]
        transitions = valid[..., :-1] & valid[..., 1:]
        return (
            coordinates,
            states,
            valid,
            transitions,
            (),
            trajectory.state_layout,
            f"evolution:{trajectory.evolution_id}",
        )
    if isinstance(trajectory, TrajectoryData):
        return (
            trajectory.coordinates.reshape((trajectory.num_cases, trajectory.capacity)),
            trajectory.states.reshape(
                (trajectory.num_cases, trajectory.capacity)
                + trajectory.state_layout.shape
            ),
            trajectory.sample_valid.reshape((trajectory.num_cases, trajectory.capacity)),
            trajectory.transition_valid.reshape(
                (trajectory.num_cases, trajectory.capacity - 1)
            ),
            trajectory.case_shape,
            trajectory.state_layout,
            f"trajectory:{trajectory.dataset_id}",
        )
    raise TypeError("trajectory must be an EvolutionTrajectory or TrajectoryData.")


def _interpolated_state(
    left_coordinate: Array,
    right_coordinate: Array,
    left_state: Array,
    coordinate: Array,
    right_state: Array,
    /,
) -> Array:
    duration = right_coordinate - left_coordinate
    fraction = (coordinate - left_coordinate) / duration
    return (1.0 - fraction) * left_state + fraction * right_state


def _refine_crossing(
    *,
    section: AbstractSection,
    evolution: AbstractEvolution | None,
    refinement: SectionRefinement,
    left_coordinate: Array,
    right_coordinate: Array,
    left_state: Array,
    right_state: Array,
    left_value: Array,
    right_value: Array,
    args: Any,
    coordinate_tolerance: float,
    section_tolerance: float,
    max_iterations: int,
) -> tuple[Array, Array, Array, bool, int, int]:
    if abs(float(right_value)) <= section_tolerance:
        return right_coordinate, right_state, right_value, True, 0, SECTION_SUCCESS
    lower_coordinate = left_coordinate
    upper_coordinate = right_coordinate
    lower_value = left_value
    candidate_coordinate = 0.5 * (left_coordinate + right_coordinate)
    candidate_state = _interpolated_state(
        left_coordinate,
        right_coordinate,
        left_state,
        candidate_coordinate,
        right_state,
    )
    candidate_value = section.evaluate(candidate_coordinate, candidate_state, args)
    status = SECTION_REFINEMENT_FAILED
    for iteration in range(1, max_iterations + 1):
        candidate_coordinate = 0.5 * (lower_coordinate + upper_coordinate)
        if refinement == "evolution":
            if evolution is None:
                raise ValueError("evolution refinement requires an AbstractEvolution.")
            advanced = evolution.advance(
                left_state,
                left_coordinate,
                candidate_coordinate,
                args,
            )
            candidate_state = advanced.final_state
            if not bool(advanced.valid):
                return (
                    candidate_coordinate,
                    candidate_state,
                    jnp.asarray(jnp.nan),
                    False,
                    iteration,
                    SECTION_EVOLUTION_FAILED,
                )
        else:
            candidate_state = _interpolated_state(
                left_coordinate,
                right_coordinate,
                left_state,
                candidate_coordinate,
                right_state,
            )
        candidate_value = section.evaluate(candidate_coordinate, candidate_state, args)
        if not bool(jnp.isfinite(candidate_value)):
            return (
                candidate_coordinate,
                candidate_state,
                candidate_value,
                False,
                iteration,
                SECTION_NONFINITE,
            )
        converged = (
            abs(float(candidate_value)) <= section_tolerance
            or float(upper_coordinate - lower_coordinate) <= coordinate_tolerance
        )
        if converged:
            return (
                candidate_coordinate,
                candidate_state,
                candidate_value,
                True,
                iteration,
                SECTION_SUCCESS,
            )
        if (float(lower_value) < 0.0 and float(candidate_value) >= 0.0) or (
            float(lower_value) > 0.0 and float(candidate_value) <= 0.0
        ):
            upper_coordinate = candidate_coordinate
        else:
            lower_coordinate = candidate_coordinate
            lower_value = candidate_value
    return (
        candidate_coordinate,
        candidate_state,
        candidate_value,
        False,
        max_iterations,
        status,
    )


def find_section_crossings(
    trajectory: EvolutionTrajectory | TrajectoryData,
    section: AbstractSection,
    /,
    *,
    direction: SectionDirection = "any",
    refinement: SectionRefinement = "interpolation",
    evolution: AbstractEvolution | None = None,
    args: Any = None,
    max_crossings: int = 128,
    coordinate_tolerance: float = 1e-10,
    section_tolerance: float = 1e-9,
    max_iterations: int = 64,
) -> SectionCrossings:
    """Detect oriented brackets and refine roots into fixed-capacity event buffers."""
    if not isinstance(section, AbstractSection):
        raise TypeError("section must be an AbstractSection.")
    if direction not in ("any", "positive", "negative"):
        raise ValueError("direction must be 'any', 'positive', or 'negative'.")
    if refinement not in ("interpolation", "evolution"):
        raise ValueError("refinement must be 'interpolation' or 'evolution'.")
    if evolution is not None and not isinstance(evolution, AbstractEvolution):
        raise TypeError("evolution must be an AbstractEvolution or None.")
    if refinement == "evolution" and evolution is None:
        raise ValueError("evolution refinement requires an evolution.")
    capacity = int(max_crossings)
    iteration_limit = int(max_iterations)
    coordinate_tol = float(coordinate_tolerance)
    section_tol = float(section_tolerance)
    if capacity < 1 or iteration_limit < 1:
        raise ValueError("max_crossings and max_iterations must be positive.")
    if (
        not np.isfinite(coordinate_tol)
        or not np.isfinite(section_tol)
        or coordinate_tol <= 0.0
        or section_tol <= 0.0
    ):
        raise ValueError("Section tolerances must be finite and positive.")
    coordinates, states, sample_valid, transition_valid, case_shape, layout, source_id = (
        _source_arrays(trajectory)
    )
    if section.state_layout.layout_id != layout.layout_id:
        raise ValueError("Section and trajectory state layouts must match.")
    if evolution is not None and evolution.state_layout.layout_id != layout.layout_id:
        raise ValueError("Evolution and trajectory state layouts must match.")
    case_count, sample_count = coordinates.shape
    output_coordinates = []
    output_states = []
    output_values = []
    output_starts = []
    output_ends = []
    output_detected = []
    output_valid = []
    output_converged = []
    output_iterations = []
    output_status = []
    output_counts = []
    output_detected_counts = []
    output_overflow = []
    for case in range(case_count):
        crossing_coordinates = jnp.full((capacity,), jnp.nan)
        crossing_states = jnp.full((capacity,) + layout.shape, jnp.nan)
        crossing_values = jnp.full((capacity,), jnp.nan)
        bracket_starts = jnp.full((capacity,), jnp.nan)
        bracket_ends = jnp.full((capacity,), jnp.nan)
        detected = jnp.zeros((capacity,), dtype=bool)
        valid = jnp.zeros((capacity,), dtype=bool)
        converged = jnp.zeros((capacity,), dtype=bool)
        iterations = jnp.zeros((capacity,), dtype=jnp.int32)
        statuses = jnp.full((capacity,), SECTION_REFINEMENT_FAILED, dtype=jnp.int32)
        stored = 0
        detected_total = 0
        for interval in range(sample_count - 1):
            if not bool(
                sample_valid[case, interval]
                & sample_valid[case, interval + 1]
                & transition_valid[case, interval]
            ):
                continue
            left_coordinate = coordinates[case, interval]
            right_coordinate = coordinates[case, interval + 1]
            left_state = states[case, interval]
            right_state = states[case, interval + 1]
            left_value = section.evaluate(left_coordinate, left_state, args)
            right_value = section.evaluate(right_coordinate, right_state, args)
            if not _crossed(float(left_value), float(right_value), direction):
                continue
            detected_total += 1
            if stored >= capacity:
                continue
            (
                crossing_coordinate,
                crossing_state,
                crossing_value,
                crossing_converged,
                crossing_iterations,
                crossing_status,
            ) = _refine_crossing(
                section=section,
                evolution=evolution,
                refinement=refinement,
                left_coordinate=left_coordinate,
                right_coordinate=right_coordinate,
                left_state=left_state,
                right_state=right_state,
                left_value=left_value,
                right_value=right_value,
                args=args,
                coordinate_tolerance=coordinate_tol,
                section_tolerance=section_tol,
                max_iterations=iteration_limit,
            )
            finite = bool(
                jnp.isfinite(crossing_coordinate)
                & jnp.all(jnp.isfinite(crossing_state))
                & jnp.isfinite(crossing_value)
            )
            crossing_coordinates = crossing_coordinates.at[stored].set(
                crossing_coordinate
            )
            crossing_states = crossing_states.at[stored].set(crossing_state)
            crossing_values = crossing_values.at[stored].set(crossing_value)
            bracket_starts = bracket_starts.at[stored].set(left_coordinate)
            bracket_ends = bracket_ends.at[stored].set(right_coordinate)
            detected = detected.at[stored].set(True)
            converged = converged.at[stored].set(crossing_converged)
            valid = valid.at[stored].set(crossing_converged and finite)
            iterations = iterations.at[stored].set(crossing_iterations)
            statuses = statuses.at[stored].set(crossing_status)
            stored += 1
        output_coordinates.append(crossing_coordinates)
        output_states.append(crossing_states)
        output_values.append(crossing_values)
        output_starts.append(bracket_starts)
        output_ends.append(bracket_ends)
        output_detected.append(detected)
        output_valid.append(valid)
        output_converged.append(converged)
        output_iterations.append(iterations)
        output_status.append(statuses)
        output_counts.append(jnp.sum(valid).astype(jnp.int32))
        output_detected_counts.append(jnp.asarray(detected_total, dtype=jnp.int32))
        output_overflow.append(jnp.asarray(detected_total > capacity))
    result_case_shape = case_shape

    def restore(items: list[Array], tail: tuple[int, ...] = ()) -> Array:
        return jnp.stack(tuple(items)).reshape(result_case_shape + (capacity,) + tail)

    method_id = (
        f"section-crossings:direction={direction}:refinement={refinement}:"
        f"coordinate-tolerance={coordinate_tol:g}:section-tolerance={section_tol:g}:"
        f"max-iterations={iteration_limit}"
    )
    return SectionCrossings(
        coordinates=restore(output_coordinates),
        states=restore(output_states, layout.shape),
        section_values=restore(output_values),
        bracket_start=restore(output_starts),
        bracket_end=restore(output_ends),
        detected=restore(output_detected),
        valid=restore(output_valid),
        converged=restore(output_converged),
        iterations=restore(output_iterations),
        status=restore(output_status),
        count=jnp.stack(tuple(output_counts)).reshape(result_case_shape),
        detected_count=jnp.stack(tuple(output_detected_counts)).reshape(
            result_case_shape
        ),
        overflow=jnp.stack(tuple(output_overflow)).reshape(result_case_shape),
        state_layout=layout,
        case_shape=result_case_shape,
        capacity=capacity,
        direction=direction,
        refinement=refinement,
        section_id=section.section_id,
        source_id=source_id,
        method_id=method_id,
    )


def section_return_map(
    crossings: SectionCrossings,
    /,
    *,
    lag: int = 1,
) -> SectionReturnMap:
    """Pair ordered section events without crossing invalid intermediate events."""
    if not isinstance(crossings, SectionCrossings):
        raise TypeError("crossings must be SectionCrossings.")
    offset = int(lag)
    if offset < 1 or offset >= crossings.capacity:
        raise ValueError("lag must lie between one and crossings.capacity - 1.")
    count = crossings.capacity - offset
    valid = crossings.valid[..., :count] & crossings.valid[..., offset:]
    for intermediate in range(offset):
        valid = valid & crossings.valid[..., intermediate : intermediate + count]
    source_coordinates = crossings.coordinates[..., :count]
    target_coordinates = crossings.coordinates[..., offset:]
    source_index = (slice(None),) * len(crossings.case_shape) + (slice(0, count),)
    target_index = (slice(None),) * len(crossings.case_shape) + (slice(offset, None),)
    source_states = crossings.states[source_index]
    target_states = crossings.states[target_index]
    intervals = target_coordinates - source_coordinates
    valid = valid & jnp.isfinite(intervals) & (intervals > 0.0)
    return SectionReturnMap(
        source_coordinates=jnp.where(valid, source_coordinates, jnp.nan),
        target_coordinates=jnp.where(valid, target_coordinates, jnp.nan),
        source_states=jnp.where(
            valid.reshape(valid.shape + (1,) * len(crossings.state_layout.shape)),
            source_states,
            jnp.nan,
        ),
        target_states=jnp.where(
            valid.reshape(valid.shape + (1,) * len(crossings.state_layout.shape)),
            target_states,
            jnp.nan,
        ),
        return_intervals=jnp.where(valid, intervals, jnp.nan),
        valid=valid,
        state_layout=crossings.state_layout,
        lag=offset,
        case_shape=crossings.case_shape,
        section_id=crossings.section_id,
        source_id=crossings.source_id,
        method_id=f"section-return-map:lag={offset}:{crossings.method_id}",
    )


__all__ = [
    "AbstractSection",
    "AffineSection",
    "CallableSection",
    "SECTION_EVOLUTION_FAILED",
    "SECTION_NONFINITE",
    "SECTION_REFINEMENT_FAILED",
    "SECTION_SUCCESS",
    "SectionCrossings",
    "SectionDirection",
    "SectionRefinement",
    "SectionReturnMap",
    "find_section_crossings",
    "section_return_map",
]
