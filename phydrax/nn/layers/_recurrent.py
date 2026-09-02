# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from .._keys import EvalKey


AffineMode = Literal["elementwise", "matrix"]
AffineExecution = Literal["serial", "associative"]
RecurrentTimeDirection = Literal["forward", "backward"]


def _broadcast_case_mask(mask: Array, value: Array, /) -> Array:
    if value.ndim < mask.ndim or value.shape[: mask.ndim] != mask.shape:
        raise ValueError(
            "Recurrent state and output leaves must begin with the batch case shape."
        )
    return jnp.reshape(mask, mask.shape + (1,) * (value.ndim - mask.ndim))


def _tree_where(mask: Array, selected: Any, alternative: Any, /) -> Any:
    return jax.tree.map(
        lambda yes, no: jnp.where(_broadcast_case_mask(mask, yes), yes, no),
        selected,
        alternative,
    )


def _tree_zero_where_invalid(valid: Array, tree: Any, /) -> Any:
    return jax.tree.map(
        lambda value: jnp.where(
            _broadcast_case_mask(valid, value), value, jnp.zeros_like(value)
        ),
        tree,
    )


def _move_sequence_axis(tree: Any, source: int, destination: int, /) -> Any:
    return jax.tree.map(lambda value: jnp.moveaxis(value, source, destination), tree)


class RecurrentBatch(StrictModule):
    """Canonical packed sequence inputs with validity and reset semantics.

    Every input leaf begins with ``case_shape + (sequence_length,)``. ``valid``,
    ``reset``, and an optional physical ``time`` array have exactly that shape.
    ``time_direction`` declares whether continuation nodes are visited in
    increasing or decreasing physical time; node coordinates are never changed.
    Invalid steps are padding: recurrent execution preserves the previous state
    and emits a zero output. A valid reset step restarts from the cell's canonical
    initial state before evaluating that step.
    """

    inputs: Any
    valid: Array
    reset: Array
    time: Array | None
    case_shape: tuple[int, ...] = eqx.field(static=True)
    sequence_length: int = eqx.field(static=True)
    time_direction: RecurrentTimeDirection = eqx.field(static=True)

    def __init__(
        self,
        inputs: Any,
        valid: ArrayLike,
        /,
        *,
        reset: ArrayLike | None = None,
        time: ArrayLike | None = None,
        time_direction: RecurrentTimeDirection = "forward",
    ):
        valid_array = jnp.asarray(valid, dtype=bool)
        if valid_array.ndim < 1 or int(valid_array.shape[-1]) <= 0:
            raise ValueError("valid must contain a non-empty trailing sequence axis.")
        if time_direction not in ("forward", "backward"):
            raise ValueError("time_direction must be 'forward' or 'backward'.")
        case_shape = tuple(int(size) for size in valid_array.shape[:-1])
        sequence_length = int(valid_array.shape[-1])
        leaves = jax.tree.leaves(inputs)
        if not leaves:
            raise ValueError("inputs must contain at least one array leaf.")
        expected_prefix = case_shape + (sequence_length,)
        normalized_inputs = jax.tree.map(jnp.asarray, inputs)
        for leaf in jax.tree.leaves(normalized_inputs):
            if (
                leaf.ndim < len(expected_prefix)
                or leaf.shape[: len(expected_prefix)] != expected_prefix
            ):
                raise ValueError(
                    "Every recurrent input leaf must begin with "
                    f"case_shape + (sequence_length,) = {expected_prefix}; "
                    f"got {leaf.shape}."
                )

        if reset is None:
            reset_array = jnp.zeros_like(valid_array)
        else:
            reset_array = jnp.asarray(reset, dtype=bool)
            if reset_array.shape != valid_array.shape:
                raise ValueError("reset must have the same shape as valid.")
        valid_array = eqx.error_if(
            valid_array,
            jnp.any(reset_array & ~valid_array),
            "reset=True requires a valid recurrent sample.",
        )
        if sequence_length > 1:
            valid_after_padding = valid_array[..., 1:] & ~valid_array[..., :-1]
            valid_array = eqx.error_if(
                valid_array,
                jnp.any(valid_after_padding & ~reset_array[..., 1:]),
                "A valid recurrent sample after padding must declare reset=True.",
            )
        if time is None:
            time_array = None
        else:
            time_array = jnp.asarray(time)
            if time_array.shape != valid_array.shape:
                raise ValueError("time must have the same shape as valid.")
            if not jnp.issubdtype(time_array.dtype, jnp.floating):
                raise TypeError("time must have a real floating dtype.")
            time_array = eqx.error_if(
                time_array,
                jnp.any(valid_array & ~jnp.isfinite(time_array)),
                "Valid recurrent times must be finite.",
            )
            if sequence_length > 1:
                continuation = (
                    valid_array[..., :-1] & valid_array[..., 1:] & ~reset_array[..., 1:]
                )
                differences = time_array[..., 1:] - time_array[..., :-1]
                directed_differences = (
                    differences if time_direction == "forward" else -differences
                )
                direction_description = (
                    "non-decreasing" if time_direction == "forward" else "non-increasing"
                )
                time_array = eqx.error_if(
                    time_array,
                    jnp.any(continuation & (directed_differences < 0)),
                    "Continuation recurrent times must be "
                    f"{direction_description} in {time_direction} physical-time execution.",
                )

        self.inputs = normalized_inputs
        self.valid = valid_array
        self.reset = reset_array
        self.time = time_array
        self.case_shape = case_shape
        self.sequence_length = sequence_length
        self.time_direction = time_direction


class RecurrentTimeContext(StrictModule):
    """Last valid physical node carried between recurrent chunks."""

    time: Array
    has_time: Array
    direction: RecurrentTimeDirection = eqx.field(static=True)

    def __init__(
        self,
        time: ArrayLike,
        has_time: ArrayLike,
        /,
        *,
        direction: RecurrentTimeDirection,
    ):
        time_array = jnp.asarray(time)
        has_time_array = jnp.asarray(has_time, dtype=bool)
        if not jnp.issubdtype(time_array.dtype, jnp.floating):
            raise TypeError("Recurrent context time must have a real floating dtype.")
        if time_array.shape != has_time_array.shape:
            raise ValueError(
                "Recurrent context time and has_time must have equal shapes."
            )
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'.")
        time_array = eqx.error_if(
            time_array,
            jnp.any(has_time_array & ~jnp.isfinite(time_array)),
            "Valid recurrent context times must be finite.",
        )
        self.time = time_array
        self.has_time = has_time_array
        self.direction = direction


class RecurrentResult(StrictModule):
    """Post-step state trajectory, masked outputs, and final recurrent values."""

    states: Any
    outputs: Any
    final_state: Any
    final_output: Any
    final_context: RecurrentTimeContext | None = None


class AbstractRecurrentCell(StrictModule):
    """Stateful step contract consumed by :func:`run_recurrent`."""

    @abstractmethod
    def initial_state(self, case_shape: tuple[int, ...], /, *, dtype: Any) -> Any:
        """Return a state tree beginning with ``case_shape``."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        state: Any,
        inputs: Any,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[Any, Any]:
        """Return ``(next_state, output)`` for one packed sequence step."""
        raise NotImplementedError


class AbstractTimeAwareRecurrentCell(AbstractRecurrentCell):
    """Recurrent cell that consumes physical node coordinates and durations."""

    @abstractmethod
    def step_with_context(
        self,
        state: Any,
        inputs: Any,
        /,
        *,
        time: Array,
        interval: Array,
        key: EvalKey = None,
    ) -> tuple[Any, Any]:
        """Evaluate one coordinated step.

        ``interval`` is zero at a segment start and is the elapsed coordinate
        duration from the preceding valid continuation node.
        """
        raise NotImplementedError


class AbstractRecurrentOutputCell(AbstractRecurrentCell):
    """Recurrent cell whose observable output differs from its state."""

    @abstractmethod
    def output_from_state(self, state: Any, /) -> Any:
        """Return the observable output represented by a recurrent state."""
        raise NotImplementedError


class AbstractAssociativeRecurrence(StrictModule):
    """Declared associative summary algebra for parallel recurrent prefixes."""

    @abstractmethod
    def initial_state(self, case_shape: tuple[int, ...], /, *, dtype: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def identity(self, case_shape: tuple[int, ...], /, *, dtype: Any) -> Any:
        """Return the summary identity with leaves beginning in ``case_shape``."""
        raise NotImplementedError

    @abstractmethod
    def encode_step(
        self,
        inputs: Any,
        /,
        *,
        key: EvalKey = None,
    ) -> Any:
        """Encode one packed step into an associative summary."""
        raise NotImplementedError

    @abstractmethod
    def combine(self, left: Any, right: Any, /) -> Any:
        """Compose consecutive summaries; this operation must be associative."""
        raise NotImplementedError

    @abstractmethod
    def apply_prefix(self, state: Any, summary: Any, /) -> Any:
        """Apply one complete prefix summary to an entering state."""
        raise NotImplementedError

    def output_from_state(self, state: Any, /) -> Any:
        return state


class AbstractTimeAwareAssociativeRecurrence(AbstractAssociativeRecurrence):
    """Associative recurrence whose summaries depend on physical durations."""

    @abstractmethod
    def encode_step_with_context(
        self,
        inputs: Any,
        /,
        *,
        time: Array,
        interval: Array,
        key: EvalKey = None,
    ) -> Any:
        """Encode one coordinated step using the serial interval convention."""
        raise NotImplementedError


def _encode_associative_step(
    recurrence: AbstractAssociativeRecurrence,
    inputs: Any,
    time: Array,
    interval: Array,
    /,
    *,
    has_time: bool,
    key: EvalKey,
) -> Any:
    if has_time and isinstance(
        recurrence,
        AbstractTimeAwareAssociativeRecurrence,
    ):
        return recurrence.encode_step_with_context(
            inputs,
            time=time,
            interval=interval,
            key=key,
        )
    return recurrence.encode_step(inputs, key=key)


def run_associative_recurrence(
    recurrence: AbstractAssociativeRecurrence,
    batch: RecurrentBatch,
    /,
    *,
    initial_state: Any | None = None,
    reset_state: Any | None = None,
    initial_context: RecurrentTimeContext | None = None,
    key: EvalKey = None,
) -> RecurrentResult:
    """Execute a caller-declared associative recurrence with reset segmentation.

    ``initial_context`` carries the preceding valid physical node into a
    continuation chunk. Pass ``result.final_context`` together with
    ``result.final_state`` when splitting a time-aware sequence.
    """
    if not isinstance(recurrence, AbstractAssociativeRecurrence):
        raise TypeError("recurrence must be an AbstractAssociativeRecurrence.")
    if not isinstance(batch, RecurrentBatch):
        raise TypeError("batch must be a RecurrentBatch.")
    dtype = _input_dtype(batch)
    canonical = recurrence.initial_state(batch.case_shape, dtype=dtype)
    entering = canonical if initial_state is None else initial_state
    restart = canonical if reset_state is None else reset_state
    identity = recurrence.identity(batch.case_shape, dtype=dtype)
    sequence_axis = len(batch.case_shape)
    inputs = _move_sequence_axis(batch.inputs, sequence_axis, 0)
    valid = jnp.moveaxis(batch.valid, -1, 0)
    reset = jnp.moveaxis(batch.reset, -1, 0)

    times, intervals, has_time, final_context = _recurrent_time_context(
        batch,
        initial_context,
    )
    scan_times = jnp.moveaxis(times, -1, 0)
    scan_intervals = jnp.moveaxis(intervals, -1, 0)
    if key is None:
        summaries = jax.lax.map(
            lambda values: _encode_associative_step(
                recurrence,
                values[0],
                values[1],
                values[2],
                has_time=has_time,
                key=None,
            ),
            (inputs, scan_times, scan_intervals),
        )
    else:
        keys = jr.split(key, batch.sequence_length)
        summaries = jax.lax.map(
            lambda values: _encode_associative_step(
                recurrence,
                values[0],
                values[1],
                values[2],
                has_time=has_time,
                key=values[3],
            ),
            (inputs, scan_times, scan_intervals, keys),
        )
    summaries = jax.tree.map(
        lambda summary, unit: jnp.where(
            _broadcast_case_mask(valid, summary),
            summary,
            jnp.broadcast_to(unit, summary.shape),
        ),
        summaries,
        identity,
    )
    segment_start = reset & valid

    def segmented_combine(left, right):
        left_summary, left_reset = left
        right_summary, right_reset = right
        composed = recurrence.combine(left_summary, right_summary)
        selected = _tree_where(right_reset, right_summary, composed)
        return selected, left_reset | right_reset

    prefixes, contains_reset = jax.lax.associative_scan(
        segmented_combine,
        (summaries, segment_start),
        axis=0,
    )
    restart_scan = jax.tree.map(
        lambda value: jnp.broadcast_to(
            value,
            (batch.sequence_length,) + value.shape,
        ),
        restart,
    )
    entering_scan = jax.tree.map(
        lambda value: jnp.broadcast_to(
            value,
            (batch.sequence_length,) + value.shape,
        ),
        entering,
    )
    base_states = _tree_where(contains_reset, restart_scan, entering_scan)
    states_scan = jax.lax.map(
        lambda values: recurrence.apply_prefix(values[0], values[1]),
        (base_states, prefixes),
    )
    outputs_scan = jax.lax.map(recurrence.output_from_state, states_scan)
    outputs_scan = _tree_zero_where_invalid(valid, outputs_scan)
    states = _move_sequence_axis(states_scan, 0, sequence_axis)
    outputs = _move_sequence_axis(outputs_scan, 0, sequence_axis)
    final_state = jax.tree.map(lambda value: value[-1], states_scan)
    final_output = recurrence.output_from_state(final_state)
    return RecurrentResult(
        states,
        outputs,
        final_state,
        final_output,
        final_context=final_context,
    )


def _recurrent_output_from_state(
    cell: AbstractRecurrentCell,
    state: Any,
    /,
) -> Any:
    if isinstance(cell, AbstractRecurrentOutputCell):
        return cell.output_from_state(state)
    return state


def _input_dtype(batch: RecurrentBatch, /) -> jnp.dtype:
    leaves = jax.tree.leaves(batch.inputs)
    return jnp.result_type(*(leaf.dtype for leaf in leaves))


def _recurrent_time_context(
    batch: RecurrentBatch,
    initial_context: RecurrentTimeContext | None,
    /,
) -> tuple[Array, Array, bool, RecurrentTimeContext | None]:
    if batch.time is None:
        if initial_context is not None:
            raise ValueError("initial_context requires a recurrent batch with time.")
        zeros = jnp.zeros(batch.valid.shape, dtype=_input_dtype(batch))
        return zeros, zeros, False, None

    if initial_context is None:
        previous_time = jnp.zeros(batch.case_shape, dtype=batch.time.dtype)
        previous_has_time = jnp.zeros(batch.case_shape, dtype=bool)
    else:
        if not isinstance(initial_context, RecurrentTimeContext):
            raise TypeError("initial_context must be a RecurrentTimeContext or None.")
        if initial_context.direction != batch.time_direction:
            raise ValueError(
                "initial_context direction must match the recurrent batch time_direction."
            )
        if (
            initial_context.time.shape != batch.case_shape
            or initial_context.has_time.shape != batch.case_shape
        ):
            raise ValueError(
                "Recurrent context leaves must have the recurrent case shape "
                f"{batch.case_shape}."
            )
        previous_time = initial_context.time.astype(batch.time.dtype)
        previous_has_time = initial_context.has_time

    direction_sign = jnp.asarray(
        1.0 if batch.time_direction == "forward" else -1.0,
        dtype=batch.time.dtype,
    )
    times = jnp.where(batch.valid, batch.time, jnp.zeros_like(batch.time))
    first_continuation = batch.valid[..., 0] & ~batch.reset[..., 0] & previous_has_time
    first_difference = direction_sign * (batch.time[..., 0] - previous_time)
    times = eqx.error_if(
        times,
        jnp.any(first_continuation & (first_difference < 0)),
        "The first continuation time is inconsistent with the recurrent "
        f"{batch.time_direction} physical-time direction.",
    )
    first_interval = jnp.where(
        first_continuation,
        first_difference,
        jnp.zeros_like(first_difference),
    )
    if batch.sequence_length == 1:
        intervals = first_interval[..., None]
    else:
        continuation = (
            batch.valid[..., :-1] & batch.valid[..., 1:] & ~batch.reset[..., 1:]
        )
        differences = direction_sign * (batch.time[..., 1:] - batch.time[..., :-1])
        intervals = jnp.concatenate(
            (
                first_interval[..., None],
                jnp.where(continuation, differences, jnp.zeros_like(differences)),
            ),
            axis=-1,
        )

    scan_times = jnp.moveaxis(batch.time, -1, 0)
    scan_valid = jnp.moveaxis(batch.valid, -1, 0)

    def update_time(last_time: Array, values: tuple[Array, Array]):
        time, valid = values
        return jnp.where(valid, time, last_time), None

    final_time, _ = jax.lax.scan(
        update_time,
        previous_time,
        (scan_times, scan_valid),
    )
    final_has_time = previous_has_time | jnp.any(batch.valid, axis=-1)
    final_context = RecurrentTimeContext(
        final_time,
        final_has_time,
        direction=batch.time_direction,
    )
    return times, intervals, True, final_context


def _resolve_initial_state(
    cell: AbstractRecurrentCell,
    batch: RecurrentBatch,
    initial_state: Any | None,
    /,
) -> Any:
    state = (
        cell.initial_state(batch.case_shape, dtype=_input_dtype(batch))
        if initial_state is None
        else jax.tree.map(jnp.asarray, initial_state)
    )
    leaves = jax.tree.leaves(state)
    if not leaves:
        raise ValueError("initial_state must contain at least one array leaf.")
    for leaf in leaves:
        if (
            leaf.ndim < len(batch.case_shape)
            or leaf.shape[: len(batch.case_shape)] != batch.case_shape
        ):
            raise ValueError(
                "Every initial-state leaf must begin with the recurrent case shape "
                f"{batch.case_shape}; got {leaf.shape}."
            )
    return state


def run_recurrent(
    cell: AbstractRecurrentCell,
    batch: RecurrentBatch,
    /,
    *,
    initial_state: Any | None = None,
    reset_state: Any | None = None,
    initial_context: RecurrentTimeContext | None = None,
    key: EvalKey = None,
) -> RecurrentResult:
    """Execute a recurrent cell serially with explicit padding and reset rules.

    ``initial_state`` and ``initial_context`` are the streaming carry entering
    this chunk. Pass ``result.final_state`` and ``result.final_context`` into a
    continuation chunk so a time-aware cell receives the boundary interval.
    Reset steps use the cell's canonical initial state unless ``reset_state`` is
    supplied, so chunking cannot change packed-segment semantics.
    """
    if not isinstance(cell, AbstractRecurrentCell):
        raise TypeError("cell must be an AbstractRecurrentCell.")
    if not isinstance(batch, RecurrentBatch):
        raise TypeError("batch must be a RecurrentBatch.")
    canonical_state = _resolve_initial_state(cell, batch, None)
    state0 = (
        canonical_state
        if initial_state is None
        else _resolve_initial_state(cell, batch, initial_state)
    )
    restart_state = (
        canonical_state
        if reset_state is None
        else _resolve_initial_state(cell, batch, reset_state)
    )
    output0 = _recurrent_output_from_state(cell, state0)
    sequence_axis = len(batch.case_shape)
    scan_inputs = _move_sequence_axis(batch.inputs, sequence_axis, 0)
    scan_valid = jnp.moveaxis(batch.valid, -1, 0)
    scan_reset = jnp.moveaxis(batch.reset, -1, 0)
    times, intervals, has_time, final_context = _recurrent_time_context(
        batch,
        initial_context,
    )
    scan_times = jnp.moveaxis(times, -1, 0)
    scan_intervals = jnp.moveaxis(intervals, -1, 0)

    def evaluate_step(
        carry: tuple[Any, Any],
        step_inputs: tuple[Any, Array, Array, Array, Array],
        step_key: EvalKey,
    ):
        state, last_output = carry
        inputs, valid, reset, time, interval = step_inputs
        restarted = _tree_where(reset & valid, restart_state, state)
        safe_inputs = _tree_zero_where_invalid(valid, inputs)
        if has_time and isinstance(cell, AbstractTimeAwareRecurrentCell):
            proposed_state, output = cell.step_with_context(
                restarted,
                safe_inputs,
                time=time,
                interval=interval,
                key=step_key,
            )
        else:
            proposed_state, output = cell.step(
                restarted,
                safe_inputs,
                key=step_key,
            )
        next_state = _tree_where(valid, proposed_state, state)
        next_output = _tree_where(valid, output, last_output)
        masked_output = _tree_zero_where_invalid(valid, output)
        return (next_state, next_output), (next_state, masked_output)

    carry0 = (state0, output0)
    if key is None:

        def step_without_key(
            carry: tuple[Any, Any],
            step_inputs: tuple[Any, Array, Array, Array, Array],
        ):
            return evaluate_step(carry, step_inputs, None)

        (final_state, final_output), (scan_states, scan_outputs) = jax.lax.scan(
            step_without_key,
            carry0,
            (scan_inputs, scan_valid, scan_reset, scan_times, scan_intervals),
        )
    else:
        step_keys = jr.split(key, batch.sequence_length)

        def step_with_key(
            carry: tuple[Any, Any],
            step_inputs: tuple[Any, Array, Array, Array, Array, Array],
        ):
            inputs, valid, reset, time, interval, step_key = step_inputs
            return evaluate_step(
                carry,
                (inputs, valid, reset, time, interval),
                step_key,
            )

        (final_state, final_output), (scan_states, scan_outputs) = jax.lax.scan(
            step_with_key,
            carry0,
            (
                scan_inputs,
                scan_valid,
                scan_reset,
                scan_times,
                scan_intervals,
                step_keys,
            ),
        )

    states = _move_sequence_axis(scan_states, 0, sequence_axis)
    outputs = _move_sequence_axis(scan_outputs, 0, sequence_axis)
    return RecurrentResult(
        states=states,
        outputs=outputs,
        final_state=final_state,
        final_output=final_output,
        final_context=final_context,
    )


class AffineRecurrence(AbstractRecurrentCell):
    """Elementwise-diagonal or dense-matrix affine recurrent cell."""

    initial: Array
    mode: AffineMode = eqx.field(static=True)

    def __init__(
        self,
        initial_state: ArrayLike,
        /,
        *,
        mode: AffineMode = "elementwise",
    ):
        initial = jnp.asarray(initial_state)
        if initial.ndim < 1:
            raise ValueError("AffineRecurrence initial_state must have a state axis.")
        if mode not in ("elementwise", "matrix"):
            raise ValueError("mode must be 'elementwise' or 'matrix'.")
        if mode == "matrix" and initial.ndim != 1:
            raise ValueError("Matrix affine recurrence requires a vector initial state.")
        self.initial = initial
        self.mode = mode

    def initial_state(self, case_shape: tuple[int, ...], /, *, dtype: Any) -> Array:
        dtype_value = jnp.result_type(dtype, self.initial.dtype)
        return jnp.broadcast_to(
            self.initial.astype(dtype_value), case_shape + self.initial.shape
        )

    def apply_transition(self, transition: Array, state: Array, /) -> Array:
        if self.mode == "elementwise":
            state_rank = self.initial.ndim
            if (
                transition.shape[-state_rank:] != self.initial.shape
                or state.shape[-state_rank:] != self.initial.shape
            ):
                raise ValueError(
                    "Elementwise affine transitions and states must end with "
                    f"state shape {self.initial.shape}."
                )
            return transition * state
        state_size = int(self.initial.shape[0])
        if transition.shape[-2:] != (state_size, state_size) or state.shape[-1:] != (
            state_size,
        ):
            raise ValueError(
                "Matrix affine transitions and states must end with "
                f"({state_size}, {state_size}) and ({state_size},)."
            )
        return ein.contract("...ij,...j->...i", transition, state)

    def compose_transitions(
        self,
        earlier: tuple[Array, Array],
        later: tuple[Array, Array],
        /,
    ) -> tuple[Array, Array]:
        earlier_transition, earlier_addition = earlier
        later_transition, later_addition = later
        if self.mode == "elementwise":
            composed_transition = later_transition * earlier_transition
            propagated_addition = later_transition * earlier_addition
        else:
            composed_transition = ein.contract(
                "...ij,...jk->...ik", later_transition, earlier_transition
            )
            propagated_addition = ein.contract(
                "...ij,...j->...i", later_transition, earlier_addition
            )
        return composed_transition, propagated_addition + later_addition

    def identity_transition(self, reference: Array, /) -> Array:
        if self.mode == "elementwise":
            return jnp.ones_like(reference)
        size = int(reference.shape[-1])
        identity = jnp.eye(size, dtype=reference.dtype)
        return jnp.broadcast_to(identity, reference.shape)

    def step(
        self,
        state: Array,
        inputs: tuple[Array, Array],
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[Array, Array]:
        del key
        transition, addition = inputs
        next_state = self.apply_transition(transition, state) + addition
        return next_state, next_state


def run_affine_recurrence(
    recurrence: AffineRecurrence,
    batch: RecurrentBatch,
    /,
    *,
    initial_state: Array | None = None,
    reset_state: Array | None = None,
    execution: AffineExecution = "serial",
) -> RecurrentResult:
    """Execute affine steps serially or by associative prefix composition.

    ``initial_state`` is the streaming carry entering this chunk. Reset steps use
    the recurrence's canonical initial state unless ``reset_state`` is supplied.
    """
    if not isinstance(recurrence, AffineRecurrence):
        raise TypeError("recurrence must be an AffineRecurrence.")
    if not isinstance(batch, RecurrentBatch):
        raise TypeError("batch must be a RecurrentBatch.")
    if execution == "serial":
        return run_recurrent(
            recurrence,
            batch,
            initial_state=initial_state,
            reset_state=reset_state,
        )
    if execution != "associative":
        raise ValueError("execution must be 'serial' or 'associative'.")
    if not isinstance(batch.inputs, tuple) or len(batch.inputs) != 2:
        raise TypeError("Affine recurrent inputs must be (transition, addition).")

    state0 = _resolve_initial_state(recurrence, batch, initial_state)
    restart_state = (
        _resolve_initial_state(recurrence, batch, None)
        if reset_state is None
        else _resolve_initial_state(recurrence, batch, reset_state)
    )
    transition, addition = batch.inputs
    sequence_axis = len(batch.case_shape)
    transitions = jnp.moveaxis(transition, sequence_axis, 0)
    additions = jnp.moveaxis(addition, sequence_axis, 0)
    valid = jnp.moveaxis(batch.valid, -1, 0)
    reset = jnp.moveaxis(batch.reset, -1, 0)

    identity = recurrence.identity_transition(transitions)
    transition_mask = jnp.reshape(
        valid, valid.shape + (1,) * (transitions.ndim - valid.ndim)
    )
    addition_mask = jnp.reshape(valid, valid.shape + (1,) * (additions.ndim - valid.ndim))
    transitions = jnp.where(transition_mask, transitions, identity)
    additions = jnp.where(addition_mask, additions, jnp.zeros_like(additions))

    reset_active = valid & reset
    reset_transition_mask = jnp.reshape(
        reset_active,
        reset_active.shape + (1,) * (transitions.ndim - reset_active.ndim),
    )
    reset_addition = (
        recurrence.apply_transition(transitions, restart_state[None, ...]) + additions
    )
    reset_addition_mask = jnp.reshape(
        reset_active,
        reset_active.shape + (1,) * (additions.ndim - reset_active.ndim),
    )
    transitions = jnp.where(
        reset_transition_mask, jnp.zeros_like(transitions), transitions
    )
    additions = jnp.where(reset_addition_mask, reset_addition, additions)

    prefix_transition, prefix_addition = jax.lax.associative_scan(
        recurrence.compose_transitions, (transitions, additions), axis=0
    )
    scan_states = (
        recurrence.apply_transition(prefix_transition, state0[None, ...])
        + prefix_addition
    )
    masked_outputs = _tree_zero_where_invalid(valid, scan_states)
    states = jnp.moveaxis(scan_states, 0, sequence_axis)
    outputs = jnp.moveaxis(masked_outputs, 0, sequence_axis)
    final_state = jax.tree.map(lambda leaf: leaf[-1], scan_states)
    return RecurrentResult(
        states=states,
        outputs=outputs,
        final_state=final_state,
        final_output=final_state,
    )


__all__ = [
    "AbstractAssociativeRecurrence",
    "AbstractTimeAwareAssociativeRecurrence",
    "AbstractTimeAwareRecurrentCell",
    "AbstractRecurrentCell",
    "AbstractRecurrentOutputCell",
    "AffineRecurrence",
    "RecurrentBatch",
    "RecurrentTimeContext",
    "RecurrentTimeDirection",
    "RecurrentResult",
    "run_affine_recurrence",
    "run_associative_recurrence",
    "run_recurrent",
]
