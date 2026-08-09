#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._layout import InputLayout, StateLayout


CaseAxisRole: TypeAlias = Literal[
    "case", "dataset", "parameter", "process", "realization"
]
InputAlignment: TypeAlias = Literal["samples", "transitions"]


def _identifier(value: str | None, payload, prefix: str, /) -> str:
    if value is not None:
        if not isinstance(value, str) or not value:
            raise ValueError("Trajectory identifiers must be non-empty strings or None.")
        return value
    return f"{prefix}:{canonical_fingerprint(payload)}"


def _event_finite(values: Array, event_rank: int, /) -> Array:
    finite = jnp.isfinite(values)
    if event_rank:
        axes = tuple(range(finite.ndim - event_rank, finite.ndim))
        finite = jnp.all(finite, axis=axes)
    return finite


def _case_axes(
    shape: tuple[int, ...],
    names: Sequence[str] | None,
    roles: Sequence[CaseAxisRole] | None,
    /,
) -> tuple[tuple[str, ...], tuple[CaseAxisRole, ...]]:
    resolved_names = (
        tuple(f"case_{index}" for index in range(len(shape)))
        if names is None
        else tuple(str(name) for name in names)
    )
    resolved_roles = ("case",) * len(shape) if roles is None else tuple(roles)
    if (
        len(resolved_names) != len(shape)
        or any(not name for name in resolved_names)
        or len(set(resolved_names)) != len(resolved_names)
    ):
        raise ValueError("case_axes must uniquely name every case axis.")
    if len(resolved_roles) != len(shape) or any(
        role not in ("case", "dataset", "parameter", "process", "realization")
        for role in resolved_roles
    ):
        raise ValueError("case_axis_roles must assign one supported role per case axis.")
    return resolved_names, resolved_roles


class TrajectoryTransitions(StrictModule):
    """Static-capacity source/target pairs with explicit validity."""

    source_coordinates: Array
    target_coordinates: Array
    source_states: Array
    target_states: Array
    inputs: Array | None
    weights: Array
    valid: Array
    lag: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    input_shape: tuple[int, ...] | None = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)


class TrajectoryData(StrictModule):
    """Axis-explicit, padded trajectory observations for identification and analysis."""

    coordinates: Array
    states: Array
    sample_valid: Array
    transition_valid: Array
    reset_mask: Array
    weights: Array
    inputs: Array | None
    input_valid: Array | None
    derivatives: Array | None
    derivative_valid: Array | None
    state_layout: StateLayout
    input_layout: InputLayout | None
    input_alignment: InputAlignment | None = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_axis_roles: tuple[CaseAxisRole, ...] = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        states: ArrayLike,
        /,
        *,
        state_layout: StateLayout,
        sample_valid: ArrayLike | None = None,
        transition_valid: ArrayLike | None = None,
        reset_mask: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        inputs: ArrayLike | None = None,
        input_layout: InputLayout | None = None,
        input_valid: ArrayLike | None = None,
        input_alignment: InputAlignment = "transitions",
        derivatives: ArrayLike | None = None,
        derivative_valid: ArrayLike | None = None,
        case_axes: Sequence[str] | None = None,
        case_axis_roles: Sequence[CaseAxisRole] | None = None,
        coordinate_id: str = "time",
        source_id: str,
        dataset_id: str | None = None,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        coordinate_values = jnp.asarray(coordinates)
        if coordinate_values.ndim < 1 or int(coordinate_values.shape[-1]) < 2:
            raise ValueError(
                "coordinates must have shape case_shape + (capacity,) with capacity >= 2."
            )
        if jnp.issubdtype(coordinate_values.dtype, jnp.complexfloating):
            raise TypeError("Trajectory coordinates must be real-valued.")
        coordinate_values = coordinate_values.astype(
            jnp.result_type(coordinate_values, float)
        )
        cases = tuple(int(size) for size in coordinate_values.shape[:-1])
        if any(size <= 0 for size in cases):
            raise ValueError("Trajectory case dimensions must be positive.")
        capacity = int(coordinate_values.shape[-1])
        state_values = jnp.asarray(states)
        expected_states = cases + (capacity,) + state_layout.shape
        if state_values.shape != expected_states:
            raise ValueError(
                f"states must have shape {expected_states}; got {state_values.shape}."
            )
        if not jnp.issubdtype(state_values.dtype, jnp.inexact):
            state_values = state_values.astype(float)

        valid = (
            jnp.ones(cases + (capacity,), dtype=bool)
            if sample_valid is None
            else jnp.asarray(sample_valid, dtype=bool)
        )
        if valid.shape != cases + (capacity,):
            raise ValueError("sample_valid must have shape case_shape + (capacity,).")
        sample_finite = jnp.isfinite(coordinate_values) & _event_finite(
            state_values, len(state_layout.shape)
        )
        state_values = eqx.error_if(
            state_values,
            jnp.any(valid & ~sample_finite),
            "Every valid trajectory sample must have finite coordinate and state values.",
        )
        resets = (
            jnp.zeros(cases + (capacity - 1,), dtype=bool)
            if reset_mask is None
            else jnp.asarray(reset_mask, dtype=bool)
        )
        if resets.shape != cases + (capacity - 1,):
            raise ValueError("reset_mask must have shape case_shape + (capacity - 1,).")
        increasing = coordinate_values[..., 1:] > coordinate_values[..., :-1]
        adjacent_valid = valid[..., :-1] & valid[..., 1:] & increasing & ~resets
        transitions = (
            adjacent_valid
            if transition_valid is None
            else jnp.asarray(transition_valid, dtype=bool)
        )
        if transitions.shape != cases + (capacity - 1,):
            raise ValueError(
                "transition_valid must have shape case_shape + (capacity - 1,)."
            )
        state_values = eqx.error_if(
            state_values,
            jnp.any(transitions & ~adjacent_valid),
            "Valid transitions require finite ordered endpoint samples and no reset.",
        )

        sample_weights = (
            valid.astype(coordinate_values.dtype)
            if weights is None
            else jnp.asarray(weights, dtype=coordinate_values.dtype)
        )
        if sample_weights.shape != cases + (capacity,):
            raise ValueError("weights must have shape case_shape + (capacity,).")
        state_values = eqx.error_if(
            state_values,
            jnp.any(valid & (~jnp.isfinite(sample_weights) | (sample_weights < 0.0))),
            "Valid sample weights must be finite and nonnegative.",
        )
        sample_weights = jnp.where(valid, sample_weights, 0.0)

        if (inputs is None) != (input_layout is None):
            raise ValueError(
                "inputs and input_layout must either both be supplied or both absent."
            )
        if input_alignment not in ("samples", "transitions"):
            raise ValueError("input_alignment must be 'samples' or 'transitions'.")
        if inputs is None:
            input_values = None
            resolved_input_valid = None
            resolved_input_alignment = None
        else:
            resolved_input_alignment = input_alignment
            input_count = capacity if input_alignment == "samples" else capacity - 1
            input_values = jnp.asarray(inputs)
            expected_inputs = cases + (input_count,) + input_layout.shape
            if input_values.shape != expected_inputs:
                raise ValueError(
                    f"inputs must have shape {expected_inputs}; got {input_values.shape}."
                )
            if not jnp.issubdtype(input_values.dtype, jnp.inexact):
                input_values = input_values.astype(float)
            input_support = valid if input_alignment == "samples" else transitions
            resolved_input_valid = (
                input_support
                if input_valid is None
                else jnp.asarray(input_valid, dtype=bool)
            )
            if resolved_input_valid.shape != cases + (input_count,):
                raise ValueError(f"input_valid must have shape {cases + (input_count,)}.")
            input_finite = _event_finite(input_values, len(input_layout.shape))
            input_values = eqx.error_if(
                input_values,
                jnp.any(resolved_input_valid & (~input_support | ~input_finite)),
                "Valid inputs require valid supporting samples or transitions "
                "and finite input values.",
            )

        if derivatives is None:
            if derivative_valid is not None:
                raise ValueError("derivative_valid requires derivatives.")
            derivative_values = None
            resolved_derivative_valid = None
        else:
            derivative_values = jnp.asarray(derivatives)
            if derivative_values.shape != expected_states:
                raise ValueError(
                    f"derivatives must have shape {expected_states}; got {derivative_values.shape}."
                )
            if not jnp.issubdtype(derivative_values.dtype, jnp.inexact):
                derivative_values = derivative_values.astype(float)
            resolved_derivative_valid = (
                valid
                if derivative_valid is None
                else jnp.asarray(derivative_valid, dtype=bool)
            )
            if resolved_derivative_valid.shape != cases + (capacity,):
                raise ValueError(
                    "derivative_valid must have shape case_shape + (capacity,)."
                )
            derivative_finite = _event_finite(derivative_values, len(state_layout.shape))
            derivative_values = eqx.error_if(
                derivative_values,
                jnp.any(resolved_derivative_valid & (~valid | ~derivative_finite)),
                "Valid derivatives require valid samples and finite values.",
            )

        names, roles = _case_axes(cases, case_axes, case_axis_roles)
        if not isinstance(coordinate_id, str) or not coordinate_id:
            raise ValueError("coordinate_id must be a non-empty string.")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string.")
        resolved_dataset_id = _identifier(
            dataset_id,
            {
                "source": source_id,
                "coordinate": coordinate_id,
                "state_layout": state_layout.layout_id,
                "input_layout": None if input_layout is None else input_layout.layout_id,
                "input_alignment": resolved_input_alignment,
                "case_shape": list(cases),
                "capacity": capacity,
            },
            "trajectory-data",
        )
        self.coordinates = coordinate_values
        self.states = state_values
        self.sample_valid = valid
        self.transition_valid = transitions
        self.reset_mask = resets
        self.weights = sample_weights
        self.inputs = input_values
        self.input_valid = resolved_input_valid
        self.derivatives = derivative_values
        self.derivative_valid = resolved_derivative_valid
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.input_alignment = resolved_input_alignment
        self.case_shape = cases
        self.case_axes = names
        self.case_axis_roles = roles
        self.capacity = capacity
        self.coordinate_id = coordinate_id
        self.source_id = source_id
        self.dataset_id = resolved_dataset_id

    @property
    def num_cases(self) -> int:
        count = 1
        for size in self.case_shape:
            count *= size
        return count

    def transitions(self, lag: int = 1, /) -> TrajectoryTransitions:
        """Return source/target pairs without crossing invalid or reset transitions."""
        offset = int(lag)
        if offset < 1 or offset >= self.capacity:
            raise ValueError("lag must lie between one and capacity - 1.")
        pair_count = self.capacity - offset
        valid = self.sample_valid[..., :pair_count] & self.sample_valid[..., offset:]
        for intermediate in range(offset):
            valid = (
                valid
                & self.transition_valid[..., intermediate : intermediate + pair_count]
            )
        source_index = (slice(None),) * len(self.case_shape) + (slice(0, pair_count),)
        target_index = (slice(None),) * len(self.case_shape) + (slice(offset, None),)
        input_values = None if self.inputs is None else self.inputs[source_index]
        if self.input_valid is not None:
            valid = valid & self.input_valid[..., :pair_count]
        weights = jnp.sqrt(self.weights[..., :pair_count] * self.weights[..., offset:])
        return TrajectoryTransitions(
            source_coordinates=self.coordinates[..., :pair_count],
            target_coordinates=self.coordinates[..., offset:],
            source_states=self.states[source_index],
            target_states=self.states[target_index],
            inputs=input_values,
            weights=jnp.where(valid, weights, 0.0),
            valid=valid,
            lag=offset,
            case_shape=self.case_shape,
            state_shape=self.state_layout.shape,
            input_shape=None if self.input_layout is None else self.input_layout.shape,
            dataset_id=self.dataset_id,
        )

    def with_derivatives(
        self,
        derivatives: ArrayLike,
        derivative_valid: ArrayLike,
        /,
        *,
        source_id: str,
    ) -> TrajectoryData:
        """Return this dataset with one explicit derivative estimate attached."""
        return TrajectoryData(
            self.coordinates,
            self.states,
            state_layout=self.state_layout,
            sample_valid=self.sample_valid,
            transition_valid=self.transition_valid,
            reset_mask=self.reset_mask,
            weights=self.weights,
            inputs=self.inputs,
            input_layout=self.input_layout,
            input_valid=self.input_valid,
            input_alignment=(
                "transitions" if self.input_alignment is None else self.input_alignment
            ),
            derivatives=derivatives,
            derivative_valid=derivative_valid,
            case_axes=self.case_axes,
            case_axis_roles=self.case_axis_roles,
            coordinate_id=self.coordinate_id,
            source_id=source_id,
        )


__all__ = [
    "CaseAxisRole",
    "InputAlignment",
    "TrajectoryData",
    "TrajectoryTransitions",
]
