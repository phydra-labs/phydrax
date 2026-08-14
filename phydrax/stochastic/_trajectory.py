#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._frozendict import frozendict
from .._strict import StrictModule
from ._realization import (
    is_stochastic_realization,
    realization_independence_labels,
    realization_path_labels as _realization_path_ids,
    StochasticRealization,
)


TransitionWeighting = Literal["trajectory", "transition"]


def _names(values: Sequence[str], /, *, owner: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if any(not value for value in names) or len(set(names)) != len(names):
        raise ValueError(f"{owner} names must be non-empty and unique.")
    return names


def _broadcast_time_array(
    value: ArrayLike,
    /,
    *,
    leading_shape: tuple[int, ...],
    num_times: int,
    name: str,
) -> Array:
    array = jnp.asarray(value)
    if array.shape == (num_times,):
        return jnp.broadcast_to(array, leading_shape + (num_times,))
    if array.shape != leading_shape + (num_times,):
        raise ValueError(
            f"{name} must have shape {(num_times,)} or "
            f"{leading_shape + (num_times,)}; got {array.shape}."
        )
    return array


@dataclass(frozen=True)
class StochasticDriverSegmentReference:
    """Identity-preserving reference to one interval of one global driver path."""

    trajectory_id: str
    physical_case_id: str
    parameter_id: str | None
    realization_id: str | None
    coupling_id: str | None
    source_index: int
    target_index: int
    source_time: float
    target_time: float

    def __post_init__(self):
        if not self.trajectory_id or not self.physical_case_id:
            raise ValueError("Trajectory and physical-case IDs must be non-empty.")
        if int(self.source_index) < 0 or int(self.target_index) <= int(self.source_index):
            raise ValueError(
                "A driver segment requires 0 <= source_index < target_index."
            )
        if not float(self.target_time) > float(self.source_time):
            raise ValueError("A driver segment requires target_time > source_time.")


@dataclass(frozen=True, slots=True)
class _TrajectoryRecord:
    """Internal axis-explicit record shared by saved solver results."""

    times: ArrayLike
    states: ArrayLike
    state_shape: tuple[int, ...]
    realization_shape: tuple[int, ...] = ()
    valid: ArrayLike | None = None
    realizations: tuple[StochasticRealization | None, ...] | None = None
    case_shape: tuple[int, ...] = ()
    case_ids: tuple[str, ...] | None = None
    parameter_ids: tuple[str | None, ...] | None = None
    discretization_id: str | None = None
    basis_id: str | None = None
    approximation_id: str | None = None
    solver_name: str | None = None
    solver_id: str | None = None
    resolved_method: str | None = None
    state_geometry_id: str | None = None
    uncertainty_source: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        cases = tuple(int(size) for size in self.case_shape)
        realizations = tuple(int(size) for size in self.realization_shape)
        state = tuple(int(size) for size in self.state_shape)
        if any(size <= 0 for size in cases + realizations + state):
            raise ValueError("Trajectory record dimensions must be positive.")
        values = jnp.asarray(self.states)
        leading = cases + realizations
        expected_rank = len(leading) + 1 + len(state)
        if values.ndim != expected_rank:
            raise ValueError(
                "Trajectory record states must have rank "
                "case + realization + time + state."
            )
        if tuple(values.shape[: len(leading)]) != leading:
            raise ValueError(f"Trajectory record states must begin with shape {leading}.")
        if tuple(values.shape[len(leading) + 1 :]) != state:
            raise ValueError(
                f"Trajectory record states must end with state shape {state}."
            )
        num_times = int(values.shape[len(leading)])
        times = _broadcast_time_array(
            self.times,
            leading_shape=leading,
            num_times=num_times,
            name="times",
        )
        valid = (
            None
            if self.valid is None
            else _broadcast_time_array(
                jnp.asarray(self.valid, dtype=bool),
                leading_shape=leading,
                num_times=num_times,
                name="valid",
            )
        )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "states", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "case_shape", cases)
        object.__setattr__(self, "realization_shape", realizations)
        object.__setattr__(self, "state_shape", state)
        if self.realizations is not None:
            object.__setattr__(self, "realizations", tuple(self.realizations))

    def prepend(self, initial_time: ArrayLike, initial_state: ArrayLike, /):
        """Return a record with one finite initial state prepended."""
        leading = self.case_shape + self.realization_shape
        initial = jnp.asarray(initial_state)
        expected = leading + self.state_shape
        if initial.shape == self.state_shape:
            initial = jnp.broadcast_to(initial, expected)
        if initial.shape != expected:
            raise ValueError(
                f"initial_state must have shape {expected} or {self.state_shape}; "
                f"got {initial.shape}."
            )
        saved_times = jnp.asarray(self.times)
        saved_states = jnp.asarray(self.states)
        initial_times = jnp.broadcast_to(jnp.asarray(initial_time), leading)[..., None]
        if bool(jnp.any(initial_times[..., 0] >= saved_times[..., 0])):
            raise ValueError("initial_time must precede the first saved solution time.")
        time_axis = len(leading)
        states = jnp.concatenate(
            (jnp.expand_dims(initial, axis=time_axis), saved_states),
            axis=time_axis,
        )
        initial_valid = jnp.all(
            jnp.isfinite(initial),
            axis=tuple(range(len(leading), initial.ndim)),
        )[..., None]
        valid = (
            jnp.all(
                jnp.isfinite(saved_states),
                axis=tuple(range(time_axis + 1, saved_states.ndim)),
            )
            if self.valid is None
            else self.valid
        )
        return replace(
            self,
            times=jnp.concatenate((initial_times, saved_times), axis=-1),
            states=states,
            valid=jnp.concatenate((initial_valid, valid), axis=-1),
        )

    def to_stochastic_trajectory(
        self,
        /,
        *,
        case_axes: Sequence[str] = (),
        realization_axes: Sequence[str] = (),
        time_axis: str = "time",
        state_axes: Sequence[str] = ("state",),
    ) -> "StochasticTrajectory":
        """Lower the canonical saved-result record to the public trajectory."""
        metadata = {} if self.metadata is None else dict(self.metadata)
        for name, value in (
            ("solver_name", self.solver_name),
            ("solver_id", self.solver_id),
            ("resolved_method", self.resolved_method),
            ("state_geometry_id", self.state_geometry_id),
            ("uncertainty_source", self.uncertainty_source),
        ):
            if value is not None:
                metadata[name] = value
        return StochasticTrajectory(
            self.times,
            self.states,
            valid=self.valid,
            case_axes=case_axes,
            case_shape=self.case_shape,
            realization_axes=realization_axes,
            realization_shape=self.realization_shape,
            time_axis=time_axis,
            state_axes=state_axes,
            realizations=self.realizations,
            case_ids=self.case_ids,
            parameter_ids=self.parameter_ids,
            discretization_id=self.discretization_id,
            basis_id=self.basis_id,
            approximation_id=self.approximation_id,
            metadata=metadata,
        )


class StochasticTrajectory(StrictModule):
    """Axis-explicit stochastic trajectories with replay and coupling provenance.

    Arrays use ``case_shape + realization_shape + (time,) + state_shape``. Physical
    cases and stochastic realizations therefore remain distinct even when either is
    multidimensional. ``valid`` marks usable saved states; transition views additionally
    reject every interval containing an invalid saved state.
    """

    times: Array
    states: Array
    valid: Array
    realizations: tuple[StochasticRealization | None, ...]
    metadata: frozendict[str, Any]
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    realization_axes: tuple[str, ...] = eqx.field(static=True)
    realization_shape: tuple[int, ...] = eqx.field(static=True)
    time_axis: str = eqx.field(static=True)
    state_axes: tuple[str, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    parameter_ids: tuple[str | None, ...] = eqx.field(static=True)
    discretization_id: str | None = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)
    approximation_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        states: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        case_axes: Sequence[str] = (),
        case_shape: Sequence[int] = (),
        realization_axes: Sequence[str] = (),
        realization_shape: Sequence[int] | None = None,
        time_axis: str = "time",
        state_axes: Sequence[str] = ("state",),
        realizations: Sequence[StochasticRealization | None] | None = None,
        case_ids: Sequence[str] | None = None,
        parameter_ids: Sequence[str | None] | None = None,
        discretization_id: str | None = None,
        basis_id: str | None = None,
        approximation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        cases = tuple(int(size) for size in case_shape)
        if any(size <= 0 for size in cases):
            raise ValueError("case_shape dimensions must be positive.")
        case_names = _names(case_axes, owner="case_axes")
        if len(case_names) != len(cases):
            raise ValueError("case_axes and case_shape must have equal rank.")

        state_names = _names(state_axes, owner="state_axes")
        if not state_names:
            raise ValueError("state_axes must contain at least one axis.")
        array = jnp.asarray(states)
        fixed_rank = len(cases) + 1 + len(state_names)
        if realization_shape is None:
            realization_rank = array.ndim - fixed_rank
            if realization_rank < 0:
                raise ValueError("states has insufficient rank for its declared axes.")
            inferred_realizations = tuple(
                int(size)
                for size in array.shape[len(cases) : len(cases) + realization_rank]
            )
        else:
            inferred_realizations = tuple(int(size) for size in realization_shape)
        if any(size <= 0 for size in inferred_realizations):
            raise ValueError("realization_shape dimensions must be positive.")
        realization_names = _names(realization_axes, owner="realization_axes")
        if len(realization_names) != len(inferred_realizations):
            raise ValueError(
                "realization_axes and realization_shape must have equal rank."
            )
        if set(case_names) & set(realization_names):
            raise ValueError("Case and realization axis names must be disjoint.")
        time_name = str(time_axis)
        if not time_name or time_name in set(
            case_names + realization_names + state_names
        ):
            raise ValueError("time_axis must be non-empty and unique among all axes.")
        if set(state_names) & set(case_names + realization_names):
            raise ValueError("State axis names must be disjoint from leading axes.")

        leading = cases + inferred_realizations
        prefix_rank = len(leading)
        if array.ndim != prefix_rank + 1 + len(state_names):
            raise ValueError(
                "states must have rank case + realization + time + state; "
                f"got shape {array.shape}."
            )
        if tuple(int(size) for size in array.shape[:prefix_rank]) != leading:
            raise ValueError(
                f"states must begin with leading shape {leading}; got {array.shape}."
            )
        num_times = int(array.shape[prefix_rank])
        if num_times <= 0:
            raise ValueError("Trajectories require at least one saved time.")
        state_shape = tuple(int(size) for size in array.shape[prefix_rank + 1 :])
        time_values = _broadcast_time_array(
            times,
            leading_shape=leading,
            num_times=num_times,
            name="times",
        )
        if bool(jnp.any(~jnp.isfinite(time_values))):
            raise ValueError("Trajectory times must be finite.")
        if num_times > 1 and bool(jnp.any(jnp.diff(time_values, axis=-1) <= 0.0)):
            raise ValueError("Trajectory times must be strictly increasing.")
        valid_values = (
            jnp.all(jnp.isfinite(array), axis=tuple(range(prefix_rank + 1, array.ndim)))
            & jnp.isfinite(time_values)
            if valid is None
            else _broadcast_time_array(
                jnp.asarray(valid, dtype=bool),
                leading_shape=leading,
                num_times=num_times,
                name="valid",
            )
        )

        case_count = prod(cases) if cases else 1
        resolved_realizations = (
            (None,) * case_count if realizations is None else tuple(realizations)
        )
        if len(resolved_realizations) != case_count:
            raise ValueError("realizations must contain one entry per physical case.")
        for value in resolved_realizations:
            if value is not None and not is_stochastic_realization(value):
                raise TypeError(
                    "realizations must contain supported stochastic realizations or None."
                )
            if value is not None and value.sample_shape != inferred_realizations:
                raise ValueError(
                    "Each realization sample_shape must equal realization_shape."
                )
        ids = (
            tuple(f"case:{index}" for index in range(case_count))
            if case_ids is None
            else tuple(str(value) for value in case_ids)
        )
        if len(ids) != case_count or any(not value for value in ids):
            raise ValueError("case_ids must contain one non-empty ID per physical case.")
        if len(set(ids)) != len(ids):
            raise ValueError("case_ids must be unique.")
        parameters = (
            (None,) * case_count
            if parameter_ids is None
            else tuple(None if value is None else str(value) for value in parameter_ids)
        )
        if len(parameters) != case_count or any(value == "" for value in parameters):
            raise ValueError("parameter_ids must align with physical cases.")
        for name, value in (
            ("discretization_id", discretization_id),
            ("basis_id", basis_id),
            ("approximation_id", approximation_id),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be non-empty or None.")

        self.times = time_values
        self.states = array
        self.valid = jnp.asarray(valid_values, dtype=bool)
        self.realizations = resolved_realizations
        self.metadata = frozendict({} if metadata is None else metadata)
        self.case_axes = case_names
        self.case_shape = cases
        self.realization_axes = realization_names
        self.realization_shape = inferred_realizations
        self.time_axis = time_name
        self.state_axes = state_names
        self.state_shape = state_shape
        self.case_ids = ids
        self.parameter_ids = parameters
        self.discretization_id = discretization_id
        self.basis_id = basis_id
        self.approximation_id = approximation_id

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return self.case_shape + self.realization_shape

    @property
    def num_times(self) -> int:
        return int(self.states.shape[len(self.leading_shape)])

    @property
    def num_cases(self) -> int:
        return prod(self.case_shape) if self.case_shape else 1

    @property
    def num_realizations(self) -> int:
        return prod(self.realization_shape) if self.realization_shape else 1

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid, axis=-1)

    @property
    def marginal_valid(self) -> Array:
        """Per-saved-state eligibility for marginal empirical reductions."""
        return self.valid

    @property
    def path_valid(self) -> Array:
        """Per-path eligibility for complete-path empirical reductions."""
        return self.successful

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        out: list[str] = []
        for case_id, realization in zip(self.case_ids, self.realizations, strict=True):
            out.extend(
                _realization_path_ids(case_id, realization, self.realization_shape)
            )
        return tuple(out)

    @property
    def independence_ids(self) -> tuple[str | None, ...]:
        """Independent Monte Carlo cluster IDs aligned with ``trajectory_ids``."""
        out: list[str | None] = []
        for realization in self.realizations:
            out.extend(
                realization_independence_labels(realization, self.realization_shape)
            )
        return tuple(out)

    def to_predictive(self):
        """Convert realization axes into explicit process-uncertainty axes.

        Validity is reduced conservatively across every physical case and saved
        time: one realization is usable only when its complete trajectory is valid.
        """
        if not self.realization_axes:
            raise ValueError(
                "StochasticTrajectory.to_predictive requires a realization axis."
            )
        import coordax as cx

        from ..uq._predictive import PredictiveField, SampleAxis

        dims = (
            self.case_axes + self.realization_axes + (self.time_axis,) + self.state_axes
        )
        samples = cx.Field(self.states, dims=dims)
        time_position = len(self.case_shape) + len(self.realization_shape)
        reduction_axes = tuple(range(len(self.case_shape))) + (time_position,)
        valid = jnp.all(self.valid, axis=reduction_axes)
        return PredictiveField(
            samples,
            tuple(SampleAxis(axis, "process") for axis in self.realization_axes),
            valid=cx.Field(valid, dims=self.realization_axes),
        )

    @classmethod
    def from_solution(
        cls,
        solution: Any,
        /,
        *,
        initial_state: ArrayLike | None = None,
        initial_time: ArrayLike | None = None,
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] = ("state",),
        case_id: str = "case:0",
        parameter_id: str | None = None,
        discretization_id: str | None = None,
        basis_id: str | None = None,
        approximation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StochasticTrajectory":
        """Convert a differential solution without erasing process axes."""
        from ..solver._differential import DifferentialSolution

        if not isinstance(solution, DifferentialSolution):
            raise TypeError("from_solution requires a DifferentialSolution.")
        sample_shape = solution.sample_shape
        axes = (
            tuple(f"realization_{index}" for index in range(len(sample_shape)))
            if realization_axes is None
            else tuple(realization_axes)
        )
        record = _TrajectoryRecord(
            solution.times,
            solution.states,
            state_shape=tuple(solution.states.shape[len(sample_shape) + 1 :]),
            realization_shape=sample_shape,
            valid=solution.valid,
            realizations=(solution.realization,),
            case_ids=(case_id,),
            parameter_ids=(parameter_id,),
            discretization_id=discretization_id,
            basis_id=basis_id,
            approximation_id=approximation_id,
            solver_name=solution.solver_name,
            solver_id=solution.solver_id,
            resolved_method=solution.resolved_method,
            state_geometry_id=solution.state_geometry_id,
            uncertainty_source=(
                "process" if solution.realization is not None else "deterministic"
            ),
            metadata=metadata,
        )
        if initial_state is not None:
            if initial_time is None:
                if solution.realization is None:
                    raise ValueError(
                        "initial_time is required when a deterministic solution is prepended."
                    )
                initial_time = solution.realization.support[0]
            record = record.prepend(initial_time, initial_state)
        return record.to_stochastic_trajectory(
            realization_axes=axes,
            state_axes=state_axes,
        )

    @classmethod
    def stack_cases(
        cls,
        trajectories: Sequence["StochasticTrajectory"],
        /,
        *,
        case_axis: str = "case",
        case_ids: Sequence[str] | None = None,
        parameter_ids: Sequence[str | None] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StochasticTrajectory":
        """Stack scalar physical cases while preserving every global realization."""
        values = tuple(trajectories)
        if not values:
            raise ValueError("stack_cases requires at least one trajectory.")
        reference = values[0]
        if reference.case_shape:
            raise ValueError("stack_cases inputs must each represent one physical case.")
        for value in values:
            if not isinstance(value, StochasticTrajectory):
                raise TypeError(
                    "stack_cases inputs must be StochasticTrajectory objects."
                )
            signature = (
                value.realization_axes,
                value.realization_shape,
                value.time_axis,
                value.state_axes,
                value.state_shape,
                value.discretization_id,
                value.basis_id,
                value.approximation_id,
            )
            reference_signature = (
                reference.realization_axes,
                reference.realization_shape,
                reference.time_axis,
                reference.state_axes,
                reference.state_shape,
                reference.discretization_id,
                reference.basis_id,
                reference.approximation_id,
            )
            if signature != reference_signature:
                raise ValueError(
                    "Stacked trajectories must share all non-case semantics."
                )
            if not bool(jnp.allclose(value.times, reference.times)):
                raise ValueError("Stacked trajectories must share the same time grid.")
        return cls(
            jnp.stack(tuple(value.times for value in values)),
            jnp.stack(tuple(value.states for value in values)),
            valid=jnp.stack(tuple(value.valid for value in values)),
            case_axes=(case_axis,),
            case_shape=(len(values),),
            realization_axes=reference.realization_axes,
            realization_shape=reference.realization_shape,
            time_axis=reference.time_axis,
            state_axes=reference.state_axes,
            realizations=tuple(value.realizations[0] for value in values),
            case_ids=(
                tuple(value.case_ids[0] for value in values)
                if case_ids is None
                else case_ids
            ),
            parameter_ids=(
                tuple(value.parameter_ids[0] for value in values)
                if parameter_ids is None
                else parameter_ids
            ),
            discretization_id=reference.discretization_id,
            basis_id=reference.basis_id,
            approximation_id=reference.approximation_id,
            metadata=metadata,
        )

    def adjacent_transitions(self) -> "StochasticTransitionView":
        if self.num_times < 2:
            raise ValueError("Adjacent transitions require at least two saved times.")
        return StochasticTransitionView(
            self,
            jnp.arange(self.num_times - 1, dtype=jnp.int32),
            jnp.arange(1, self.num_times, dtype=jnp.int32),
        )

    def horizon_transitions(self, steps: int, /) -> "StochasticTransitionView":
        horizon = int(steps)
        if horizon <= 0 or horizon >= self.num_times:
            raise ValueError("steps must lie in [1, num_times - 1].")
        return StochasticTransitionView(
            self,
            jnp.arange(self.num_times - horizon, dtype=jnp.int32),
            jnp.arange(horizon, self.num_times, dtype=jnp.int32),
        )

    def terminal_transitions(self) -> "StochasticTransitionView":
        if self.num_times < 2:
            raise ValueError("Terminal transitions require at least two saved times.")
        return StochasticTransitionView(
            self,
            jnp.arange(self.num_times - 1, dtype=jnp.int32),
            jnp.full((self.num_times - 1,), self.num_times - 1, dtype=jnp.int32),
        )

    def transitions_at_times(
        self,
        source_times: ArrayLike,
        target_times: ArrayLike,
        /,
        *,
        atol: float = 1e-10,
    ) -> "StochasticTransitionView":
        shared = self.times.reshape((-1, self.num_times))
        reference = shared[0]
        if not bool(jnp.allclose(shared, reference[None, :], rtol=0.0, atol=atol)):
            raise ValueError("Explicit time pairs require one shared saved-time grid.")
        sources = jnp.asarray(source_times, dtype=float).reshape((-1,))
        targets = jnp.asarray(target_times, dtype=float).reshape((-1,))
        if sources.shape != targets.shape or int(sources.size) <= 0:
            raise ValueError(
                "source_times and target_times must be equal non-empty vectors."
            )
        source_distance = jnp.abs(sources[:, None] - reference[None, :])
        target_distance = jnp.abs(targets[:, None] - reference[None, :])
        source_indices = jnp.argmin(source_distance, axis=-1)
        target_indices = jnp.argmin(target_distance, axis=-1)
        if bool(jnp.any(jnp.min(source_distance, axis=-1) > float(atol))) or bool(
            jnp.any(jnp.min(target_distance, axis=-1) > float(atol))
        ):
            raise ValueError("Every explicit time must match a saved time within atol.")
        return StochasticTransitionView(self, source_indices, target_indices)


class StochasticTransitionView(StrictModule):
    """Lazy index view over trajectory states; overlapping windows are never stored."""

    trajectory: StochasticTrajectory
    source_indices: Array
    target_indices: Array

    def __init__(
        self,
        trajectory: StochasticTrajectory,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        /,
    ):
        if not isinstance(trajectory, StochasticTrajectory):
            raise TypeError("trajectory must be a StochasticTrajectory.")
        sources = jnp.asarray(source_indices, dtype=jnp.int32).reshape((-1,))
        targets = jnp.asarray(target_indices, dtype=jnp.int32).reshape((-1,))
        if sources.shape != targets.shape or int(sources.size) <= 0:
            raise ValueError("Transition indices must be equal non-empty vectors.")
        if bool(jnp.any(sources < 0)) or bool(jnp.any(targets >= trajectory.num_times)):
            raise ValueError("Transition indices lie outside the saved-time axis.")
        if bool(jnp.any(targets <= sources)):
            raise ValueError("Every transition requires source_index < target_index.")
        self.trajectory = trajectory
        self.source_indices = sources
        self.target_indices = targets

    @property
    def num_pairs(self) -> int:
        return int(self.source_indices.size)

    @property
    def transition_shape(self) -> tuple[int, ...]:
        return self.trajectory.leading_shape + (self.num_pairs,)

    @property
    def source_states(self) -> Array:
        axis = len(self.trajectory.leading_shape)
        return jnp.take(self.trajectory.states, self.source_indices, axis=axis)

    @property
    def target_states(self) -> Array:
        axis = len(self.trajectory.leading_shape)
        return jnp.take(self.trajectory.states, self.target_indices, axis=axis)

    @property
    def initial_states(self) -> Array:
        return self.source_states

    @property
    def final_states(self) -> Array:
        return self.target_states

    @property
    def source_times(self) -> Array:
        return jnp.take(self.trajectory.times, self.source_indices, axis=-1)

    @property
    def target_times(self) -> Array:
        return jnp.take(self.trajectory.times, self.target_indices, axis=-1)

    @property
    def durations(self) -> Array:
        return self.target_times - self.source_times

    @property
    def valid(self) -> Array:
        invalid = (~self.trajectory.valid).astype(jnp.int32)
        prefix = jnp.cumsum(invalid, axis=-1)
        target_invalid = jnp.take(prefix, self.target_indices, axis=-1)
        source_before = jnp.maximum(self.source_indices - 1, 0)
        before_invalid = jnp.take(prefix, source_before, axis=-1)
        before_invalid = jnp.where(
            self.source_indices == 0,
            jnp.zeros_like(before_invalid),
            before_invalid,
        )
        return (target_invalid - before_invalid) == 0

    @property
    def num_valid(self) -> int:
        return int(jnp.sum(self.valid))

    @property
    def duration(self) -> float:
        durations = self.durations
        first = durations.reshape((-1,))[0]
        if not bool(jnp.allclose(durations, first)):
            raise ValueError("This transition view does not have one common duration.")
        return float(first)

    @property
    def num_cases(self) -> int:
        return self.trajectory.num_cases

    @property
    def num_realizations(self) -> int:
        return self.trajectory.num_realizations

    @property
    def metadata(self) -> frozendict[str, Any]:
        return self.trajectory.metadata

    def sample_flat_indices(
        self,
        key: Key[Array, ""],
        num_samples: int,
        /,
        *,
        weighting: TransitionWeighting = "trajectory",
    ) -> Array:
        """Sample valid flattened pairs, balancing complete trajectories by default."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        mask = self.valid.reshape((-1, self.num_pairs))
        if not bool(jnp.any(mask)):
            raise ValueError("No valid transitions are available.")
        if weighting == "transition":
            candidates = jnp.flatnonzero(mask.reshape((-1,)), size=mask.size)
            candidate_mask = jnp.arange(mask.size) < jnp.sum(mask)
            probabilities = candidate_mask.astype(float) / jnp.sum(candidate_mask)
            return jr.choice(key, candidates, shape=(count,), p=probabilities)
        if weighting != "trajectory":
            raise ValueError("weighting must be 'trajectory' or 'transition'.")
        trajectory_key, pair_key = jr.split(key)
        counts = jnp.sum(mask, axis=-1)
        active = counts > 0
        probabilities = active.astype(float) / jnp.sum(active)
        trajectories = jr.choice(
            trajectory_key,
            mask.shape[0],
            shape=(count,),
            p=probabilities,
        )
        ranks = jnp.floor(jr.uniform(pair_key, (count,)) * counts[trajectories]).astype(
            jnp.int32
        )
        cumulative = jnp.cumsum(mask[trajectories], axis=-1)
        pairs = jnp.argmax(cumulative == (ranks[:, None] + 1), axis=-1)
        return trajectories * self.num_pairs + pairs

    def driver_segment_references(self) -> tuple[StochasticDriverSegmentReference, ...]:
        """Materialize small host-side interval provenance, never state windows."""
        times = np.asarray(self.trajectory.times).reshape((-1, self.trajectory.num_times))
        sources = np.asarray(self.source_indices)
        targets = np.asarray(self.target_indices)
        valid = np.asarray(self.valid).reshape((-1, self.num_pairs))
        trajectory_ids = self.trajectory.trajectory_ids
        path_count = self.trajectory.num_realizations
        out: list[StochasticDriverSegmentReference] = []
        for trajectory_index, trajectory_id in enumerate(trajectory_ids):
            case_index = trajectory_index // path_count
            realization = self.trajectory.realizations[case_index]
            realization_id = None if realization is None else realization.realization_id
            coupling_id = None if realization is None else realization.coupling_id
            for pair_index, (source, target) in enumerate(
                zip(sources, targets, strict=True)
            ):
                if not valid[trajectory_index, pair_index]:
                    continue
                out.append(
                    StochasticDriverSegmentReference(
                        trajectory_id,
                        self.trajectory.case_ids[case_index],
                        self.trajectory.parameter_ids[case_index],
                        realization_id,
                        coupling_id,
                        int(source),
                        int(target),
                        float(times[trajectory_index, source]),
                        float(times[trajectory_index, target]),
                    )
                )
        return tuple(out)

    def operator_dataset(
        self,
        /,
        *,
        source_axes: Sequence[Any],
        query_axes: Sequence[Any] | None = None,
        input_name: str = "state",
        target_name: str = "output",
        duration_name: str | None = "duration",
        source_time_name: str | None = None,
        case_axis: str = "transition",
    ):
        """Lower valid transitions to the canonical neural-operator dataset."""
        from ..nn.operator import OperatorCaseProvenance
        from ..nn.operator.training import operator_dataset_from_arrays

        axes = tuple(source_axes)
        targets_axes = axes if query_axes is None else tuple(query_axes)
        expected_shape = tuple(int(axis.size) for axis in axes)
        if expected_shape != self.trajectory.state_shape:
            raise ValueError(
                f"source_axes imply state shape {expected_shape}; "
                f"trajectory state shape is {self.trajectory.state_shape}."
            )
        target_shape = tuple(int(axis.size) for axis in targets_axes)
        if target_shape != self.trajectory.state_shape:
            raise ValueError("query_axes must match the trajectory state shape.")
        mask = self.valid.reshape((-1,))
        selected = jnp.flatnonzero(mask, size=int(jnp.sum(mask)))
        source = self.source_states.reshape((-1,) + self.trajectory.state_shape)[selected]
        target = self.target_states.reshape((-1,) + self.trajectory.state_shape)[selected]
        durations = self.durations.reshape((-1,))[selected]
        source_times = self.source_times.reshape((-1,))[selected]
        inputs: dict[str, Array] = {str(input_name): source}
        source_axis_map: dict[str, tuple[Any, ...]] = {str(input_name): axes}
        broadcast_shape = (int(selected.size),) + self.trajectory.state_shape
        if duration_name is not None:
            name = str(duration_name)
            inputs[name] = jnp.broadcast_to(
                durations.reshape((-1,) + (1,) * len(self.trajectory.state_shape)),
                broadcast_shape,
            )
            source_axis_map[name] = axes
        if source_time_name is not None:
            name = str(source_time_name)
            inputs[name] = jnp.broadcast_to(
                source_times.reshape((-1,) + (1,) * len(self.trajectory.state_shape)),
                broadcast_shape,
            )
            source_axis_map[name] = axes

        references = self.driver_segment_references()
        provenance: list[OperatorCaseProvenance] = []
        for index, reference in enumerate(references):
            identities = {
                "physical_case": reference.physical_case_id,
                "trajectory": reference.trajectory_id,
            }
            if reference.realization_id is not None:
                identities["realization"] = reference.realization_id
            if reference.coupling_id is not None:
                identities["coupling"] = reference.coupling_id
            if reference.parameter_id is not None:
                identities["parameters"] = reference.parameter_id
            provenance.append(
                OperatorCaseProvenance(
                    f"{reference.trajectory_id}:transition:{reference.source_index}:"
                    f"{reference.target_index}:{index}",
                    identities=identities,
                    order={
                        "source_time": reference.source_time,
                        "target_time": reference.target_time,
                    },
                )
            )
        if len(provenance) != int(selected.size):
            raise AssertionError(
                "Transition provenance and valid state selection diverged."
            )
        return operator_dataset_from_arrays(
            inputs,
            {str(target_name): target},
            source_axes=source_axis_map,
            query_axes=targets_axes,
            case_axis=str(case_axis),
            provenance=tuple(provenance),
        )


__all__ = [
    "StochasticDriverSegmentReference",
    "StochasticTrajectory",
    "StochasticTransitionView",
    "TransitionWeighting",
]
