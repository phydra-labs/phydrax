#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._callable import _ensure_special_kwonly_args
from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._keys import EvalKey, split_eval_key
from ..layers import AbstractRecurrentCell, RecurrentBatch, RecurrentResult, run_recurrent
from ..layers._recurrent_cells import _recurrent_cell_output_width


RecurrentReturnMode = Literal["sequence", "final"]
BidirectionalMerge = Literal["concatenate", "sum", "mean"]


def _apply_readout(
    readout: Callable | None,
    values: Any,
    /,
    *,
    leading_ndim: int,
    key: EvalKey,
) -> Any:
    if readout is None:
        return values
    leaves = jax.tree.leaves(values)
    if not leaves:
        raise ValueError("Recurrent readout input must contain at least one array leaf.")
    leading_shape = tuple(jnp.asarray(leaves[0]).shape[:leading_ndim])
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if array.ndim < leading_ndim or array.shape[:leading_ndim] != leading_shape:
            raise ValueError(
                "Recurrent readout input leaves must share their leading case axes."
            )
    count = prod(leading_shape)
    flattened = jax.tree.map(
        lambda leaf: jnp.asarray(leaf).reshape(
            (count,) + jnp.asarray(leaf).shape[leading_ndim:]
        ),
        values,
    )
    if key is None:
        mapped = jax.vmap(lambda value: readout(value, key=None))(flattened)
    else:
        keys = jax.random.split(key, count)
        mapped = jax.vmap(lambda value, item_key: readout(value, key=item_key))(
            flattened, keys
        )
    return jax.tree.map(
        lambda leaf: jnp.asarray(leaf).reshape(
            leading_shape + jnp.asarray(leaf).shape[1:]
        ),
        mapped,
    )


def _mask_sequence_output(values: Any, valid: Array, /) -> Any:
    def _mask_leaf(leaf: Array) -> Array:
        array = jnp.asarray(leaf)
        if array.ndim < valid.ndim or array.shape[: valid.ndim] != valid.shape:
            raise ValueError(
                "Sequence readout leaves must begin with the recurrent case and "
                f"sequence shape {valid.shape}; got {array.shape}."
            )
        mask = valid.reshape(valid.shape + (1,) * (array.ndim - valid.ndim))
        return jnp.where(mask, array, jnp.zeros((), dtype=array.dtype))

    return jax.tree.map(_mask_leaf, values)


def _take_sequence_indices(tree: Any, indices: Array, /, *, sequence_axis: int) -> Any:
    def _take_leaf(leaf: Array) -> Array:
        array = jnp.asarray(leaf)
        index = indices.reshape(indices.shape + (1,) * (array.ndim - indices.ndim))
        index = jnp.broadcast_to(index, indices.shape + array.shape[sequence_axis + 1 :])
        return jnp.take_along_axis(array, index, axis=sequence_axis)

    return jax.tree.map(_take_leaf, tree)


def _segment_reverse_indices(batch: RecurrentBatch, /) -> tuple[Array, Array]:
    valid = batch.valid
    length = int(valid.shape[-1])
    positions = jnp.broadcast_to(jnp.arange(length), valid.shape)
    previous_invalid = jnp.concatenate(
        (jnp.ones(valid.shape[:-1] + (1,), dtype=bool), ~valid[..., :-1]),
        axis=-1,
    )
    starts = valid & (batch.reset | previous_invalid)
    next_boundary = jnp.concatenate(
        (
            (~valid[..., 1:]) | batch.reset[..., 1:],
            jnp.ones(valid.shape[:-1] + (1,), dtype=bool),
        ),
        axis=-1,
    )
    ends = valid & next_boundary
    start_indices = jax.lax.associative_scan(
        jnp.maximum,
        jnp.where(starts, positions, -jnp.ones_like(positions)),
        axis=-1,
    )
    reverse_end_candidates = jnp.flip(
        jnp.where(ends, positions, jnp.full_like(positions, length)),
        axis=-1,
    )
    end_indices = jnp.flip(
        jax.lax.associative_scan(jnp.minimum, reverse_end_candidates, axis=-1),
        axis=-1,
    )
    indices = jnp.where(valid, start_indices + end_indices - positions, positions)
    return indices, starts


def _reverse_recurrent_batch(batch: RecurrentBatch, /) -> tuple[RecurrentBatch, Array]:
    sequence_axis = len(batch.case_shape)
    indices, segment_starts = _segment_reverse_indices(batch)
    inputs = _take_sequence_indices(batch.inputs, indices, sequence_axis=sequence_axis)
    time = (
        None
        if batch.time is None
        else _take_sequence_indices(batch.time, indices, sequence_axis=sequence_axis)
    )
    return (
        RecurrentBatch(inputs, batch.valid, reset=segment_starts, time=time),
        indices,
    )


class RecurrentSequenceModel(StrictModule):
    """User-facing sequence model backed by the packed recurrent executor."""

    cell: AbstractRecurrentCell
    readout: Callable | None
    return_mode: RecurrentReturnMode = eqx.field(static=True)

    def __init__(
        self,
        cell: AbstractRecurrentCell,
        /,
        *,
        readout: Callable | None = None,
        return_mode: RecurrentReturnMode = "sequence",
    ):
        if not isinstance(cell, AbstractRecurrentCell):
            raise TypeError("cell must implement AbstractRecurrentCell.")
        if readout is not None and not callable(readout):
            raise TypeError("readout must be callable or None.")
        if return_mode not in ("sequence", "final"):
            raise ValueError("return_mode must be 'sequence' or 'final'.")
        self.cell = cell
        self.readout = None if readout is None else _ensure_special_kwonly_args(readout)
        self.return_mode = return_mode

    def evaluate_with_state(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: Any | None = None,
        key: EvalKey = DOC_KEY0,
    ) -> RecurrentResult:
        """Return the complete recurrent trajectory before applying the readout."""
        return run_recurrent(self.cell, batch, initial_state=initial_state, key=key)

    def __call__(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: Any | None = None,
        key: EvalKey = DOC_KEY0,
    ) -> Any:
        recurrent_key, readout_key = split_eval_key(key, 2)
        result = self.evaluate_with_state(
            batch,
            initial_state=initial_state,
            key=recurrent_key,
        )
        values = result.outputs if self.return_mode == "sequence" else result.final_output
        predictions = _apply_readout(
            self.readout,
            values,
            leading_ndim=batch.valid.ndim
            if self.return_mode == "sequence"
            else batch.valid.ndim - 1,
            key=readout_key,
        )
        if self.return_mode == "sequence":
            predictions = _mask_sequence_output(predictions, batch.valid)
        return predictions


class BidirectionalRecurrentSequenceModel(StrictModule):
    """Segment-aware forward/backward recurrent model with explicit merge policy."""

    forward_cell: AbstractRecurrentCell
    backward_cell: AbstractRecurrentCell
    readout: Callable | None
    merge: BidirectionalMerge = eqx.field(static=True)
    return_mode: RecurrentReturnMode = eqx.field(static=True)

    def __init__(
        self,
        forward_cell: AbstractRecurrentCell,
        backward_cell: AbstractRecurrentCell,
        /,
        *,
        readout: Callable | None = None,
        merge: BidirectionalMerge = "concatenate",
        return_mode: RecurrentReturnMode = "sequence",
    ):
        if not isinstance(forward_cell, AbstractRecurrentCell) or not isinstance(
            backward_cell, AbstractRecurrentCell
        ):
            raise TypeError("forward_cell and backward_cell must be recurrent cells.")
        if readout is not None and not callable(readout):
            raise TypeError("readout must be callable or None.")
        if merge not in ("concatenate", "sum", "mean"):
            raise ValueError("merge must be 'concatenate', 'sum', or 'mean'.")
        if return_mode not in ("sequence", "final"):
            raise ValueError("return_mode must be 'sequence' or 'final'.")
        if merge != "concatenate":
            forward_width = _recurrent_cell_output_width(forward_cell)
            backward_width = _recurrent_cell_output_width(backward_cell)
            if (
                forward_width is not None
                and backward_width is not None
                and forward_width != backward_width
            ):
                raise ValueError("sum and mean merges require equal cell output widths.")
        self.forward_cell = forward_cell
        self.backward_cell = backward_cell
        self.readout = None if readout is None else _ensure_special_kwonly_args(readout)
        self.merge = merge
        self.return_mode = return_mode

    def _merge(self, forward: Any, backward: Any, /) -> Array:
        forward_array = jnp.asarray(forward)
        backward_array = jnp.asarray(backward)
        if forward_array.shape[:-1] != backward_array.shape[:-1]:
            raise ValueError(
                "Forward and backward recurrent outputs have incompatible shapes."
            )
        if self.merge == "concatenate":
            return jnp.concatenate((forward_array, backward_array), axis=-1)
        if forward_array.shape != backward_array.shape:
            raise ValueError("sum and mean merges require exactly equal output shapes.")
        merged = forward_array + backward_array
        return merged if self.merge == "sum" else 0.5 * merged

    def __call__(
        self,
        batch: RecurrentBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Any:
        forward_key, backward_key, readout_key = split_eval_key(key, 3)
        forward_result = run_recurrent(self.forward_cell, batch, key=forward_key)
        backward_batch, reverse_indices = _reverse_recurrent_batch(batch)
        backward_result = run_recurrent(
            self.backward_cell,
            backward_batch,
            key=backward_key,
        )
        if self.return_mode == "sequence":
            sequence_axis = len(batch.case_shape)
            backward_values = _take_sequence_indices(
                backward_result.outputs,
                reverse_indices,
                sequence_axis=sequence_axis,
            )
            merged = self._merge(forward_result.outputs, backward_values)
        else:
            merged = self._merge(
                forward_result.final_output,
                backward_result.final_output,
            )
        predictions = _apply_readout(
            self.readout,
            merged,
            leading_ndim=batch.valid.ndim
            if self.return_mode == "sequence"
            else batch.valid.ndim - 1,
            key=readout_key,
        )
        if self.return_mode == "sequence":
            predictions = _mask_sequence_output(predictions, batch.valid)
        return predictions


__all__ = [
    "BidirectionalMerge",
    "BidirectionalRecurrentSequenceModel",
    "RecurrentReturnMode",
    "RecurrentSequenceModel",
]
