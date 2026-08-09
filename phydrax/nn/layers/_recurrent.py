# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._keys import EvalKey


AffineMode = Literal["elementwise", "matrix"]
AffineExecution = Literal["serial", "associative"]


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

    def __init__(
        self,
        inputs: Any,
        valid: ArrayLike,
        /,
        *,
        reset: ArrayLike | None = None,
        time: ArrayLike | None = None,
    ):
        valid_array = jnp.asarray(valid, dtype=bool)
        if valid_array.ndim < 1 or int(valid_array.shape[-1]) <= 0:
            raise ValueError("valid must contain a non-empty trailing sequence axis.")
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

        self.inputs = normalized_inputs
        self.valid = valid_array
        self.reset = reset_array
        self.time = time_array
        self.case_shape = case_shape
        self.sequence_length = sequence_length


class RecurrentResult(StrictModule):
    """Post-step state trajectory, masked outputs, and final recurrent values."""

    states: Any
    outputs: Any
    final_state: Any
    final_output: Any


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


class AbstractRecurrentOutputCell(AbstractRecurrentCell):
    """Recurrent cell whose observable output differs from its state."""

    @abstractmethod
    def output_from_state(self, state: Any, /) -> Any:
        """Return the observable output represented by a recurrent state."""
        raise NotImplementedError


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
    key: EvalKey = None,
) -> RecurrentResult:
    """Execute a recurrent cell serially with explicit padding and reset rules.

    ``initial_state`` is the streaming carry entering this chunk. Reset steps use
    the cell's canonical initial state unless ``reset_state`` is supplied, so
    chunking cannot change packed-segment semantics.
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

    def evaluate_step(
        carry: tuple[Any, Any],
        step_inputs: tuple[Any, Array, Array],
        step_key: EvalKey,
    ):
        state, last_output = carry
        inputs, valid, reset = step_inputs
        restarted = _tree_where(reset & valid, restart_state, state)
        safe_inputs = _tree_zero_where_invalid(valid, inputs)
        proposed_state, output = cell.step(restarted, safe_inputs, key=step_key)
        next_state = _tree_where(valid, proposed_state, state)
        next_output = _tree_where(valid, output, last_output)
        masked_output = _tree_zero_where_invalid(valid, output)
        return (next_state, next_output), (next_state, masked_output)

    carry0 = (state0, output0)
    if key is None:

        def step_without_key(
            carry: tuple[Any, Any], step_inputs: tuple[Any, Array, Array]
        ):
            return evaluate_step(carry, step_inputs, None)

        (final_state, final_output), (scan_states, scan_outputs) = jax.lax.scan(
            step_without_key, carry0, (scan_inputs, scan_valid, scan_reset)
        )
    else:
        step_keys = jr.split(key, batch.sequence_length)

        def step_with_key(
            carry: tuple[Any, Any],
            step_inputs: tuple[Any, Array, Array, Array],
        ):
            inputs, valid, reset, step_key = step_inputs
            return evaluate_step(carry, (inputs, valid, reset), step_key)

        (final_state, final_output), (scan_states, scan_outputs) = jax.lax.scan(
            step_with_key, carry0, (scan_inputs, scan_valid, scan_reset, step_keys)
        )

    states = _move_sequence_axis(scan_states, 0, sequence_axis)
    outputs = _move_sequence_axis(scan_outputs, 0, sequence_axis)
    return RecurrentResult(
        states=states,
        outputs=outputs,
        final_state=final_state,
        final_output=final_output,
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
        return jnp.einsum("...ij,...j->...i", transition, state)

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
            composed_transition = jnp.einsum(
                "...ij,...jk->...ik", later_transition, earlier_transition
            )
            propagated_addition = jnp.einsum(
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
    "AbstractRecurrentCell",
    "AbstractRecurrentOutputCell",
    "AffineRecurrence",
    "RecurrentBatch",
    "RecurrentResult",
    "run_affine_recurrence",
    "run_recurrent",
]
