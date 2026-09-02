#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._keys import EvalKey
from ..layers import RecurrentBatch, RecurrentResult
from ..layers._weight_space_recurrence import (
    WeightSpaceExecution,
    WeightSpaceRecurrence,
    WeightSpaceState,
)
from ..parameters import ParameterSubspace


class FunctionalStateDecoder(StrictModule):
    """Reconstruct and evaluate one root function for every packed parameter state."""

    subspace: ParameterSubspace
    query_size: int = eqx.field(static=True)

    def __init__(self, subspace: ParameterSubspace, query_size: int, /):
        if not isinstance(subspace, ParameterSubspace):
            raise TypeError("subspace must be a ParameterSubspace.")
        if not callable(subspace.reconstruct(subspace.initial)):
            raise TypeError("ParameterSubspace must reconstruct a callable root model.")
        self.query_size = int(query_size)
        if self.query_size <= 0:
            raise ValueError("query_size must be positive.")
        self.subspace = subspace

    def reconstruct(self, vector: Array, /) -> PyTree[Any]:
        return self.subspace.reconstruct_vector(vector)

    def __call__(
        self,
        states: Array,
        queries: Array,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        parameter_states = jnp.asarray(states)
        if (
            parameter_states.ndim < 2
            or int(parameter_states.shape[-1]) != self.subspace.total_dimension
        ):
            raise ValueError(
                "states must have case and sequence axes followed by the selected "
                "parameter dimension."
            )
        case_shape = tuple(int(size) for size in parameter_states.shape[:-2])
        sequence_length = int(parameter_states.shape[-2])
        query_values = jnp.asarray(queries)
        if (
            self.query_size == 1
            and query_values.ndim >= 1
            and (query_values.ndim == 1 or query_values.shape[-1] != 1)
        ):
            query_values = query_values[..., None]
        if query_values.ndim < 2 or int(query_values.shape[-1]) != self.query_size:
            raise ValueError(
                f"queries must end in coordinate width {self.query_size}; got {query_values.shape}."
            )
        query_count = int(query_values.shape[-2])
        if query_values.shape[:-2] == ():
            query_values = jnp.broadcast_to(
                query_values,
                case_shape + (query_count, self.query_size),
            )
        elif query_values.shape[:-2] != case_shape:
            raise ValueError("queries must be shared or begin with the state case shape.")
        case_count = math.prod(case_shape) if case_shape else 1
        flat_states = parameter_states.reshape(
            (case_count, sequence_length, self.subspace.total_dimension)
        )
        flat_queries = query_values.reshape((case_count, query_count, self.query_size))

        def evaluate_parameter(vector: Array, query_set: Array) -> Array:
            model = self.subspace.reconstruct_vector(vector)
            return jax.vmap(lambda query: model(query, key=key))(query_set)

        def evaluate_case(state_sequence: Array, query_set: Array) -> Array:
            return jax.vmap(evaluate_parameter, in_axes=(0, None))(
                state_sequence,
                query_set,
            )

        decoded = jax.vmap(evaluate_case)(flat_states, flat_queries)
        return decoded.reshape(case_shape + decoded.shape[1:])


class WeightSpaceRecurrentModel(StrictModule):
    """Recurrent parameter-state model with coordinate-wise root-function decoding."""

    decoder: FunctionalStateDecoder
    recurrence: WeightSpaceRecurrence
    execution: WeightSpaceExecution = eqx.field(static=True)

    def __init__(
        self,
        subspace: ParameterSubspace,
        observation_size: int,
        query_size: int,
        /,
        *,
        execution: WeightSpaceExecution = "associative",
        input_mode: str = "difference",
        maximum_retention: float = 0.999,
        input_scale: float = 1e-2,
        dtype: Any | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if execution not in ("serial", "associative"):
            raise ValueError("execution must be 'serial' or 'associative'.")
        self.decoder = FunctionalStateDecoder(subspace, query_size)
        self.recurrence = WeightSpaceRecurrence(
            observation_size,
            subspace.total_dimension,
            input_mode=input_mode,
            maximum_retention=maximum_retention,
            input_scale=input_scale,
            dtype=subspace.pack().dtype if dtype is None else dtype,
            key=key,
        )
        self.execution = execution

    def parameter_trajectory(
        self,
        batch: RecurrentBatch,
        /,
        *,
        initial_state: WeightSpaceState | None = None,
        key: EvalKey = DOC_KEY0,
    ) -> RecurrentResult:
        return self.recurrence.evaluate_with_state(
            batch,
            self.decoder.subspace.pack(),
            initial_state=initial_state,
            execution=self.execution,
            key=key,
        )

    def __call__(
        self,
        batch: RecurrentBatch,
        queries: Array,
        /,
        *,
        initial_state: WeightSpaceState | None = None,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        trajectory = self.parameter_trajectory(
            batch,
            initial_state=initial_state,
            key=key,
        )
        return self.decoder(trajectory.states, queries, key=key)

    def evaluate_final(
        self,
        batch: RecurrentBatch,
        queries: Array,
        /,
        *,
        initial_state: WeightSpaceState | None = None,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        trajectory = self.parameter_trajectory(
            batch,
            initial_state=initial_state,
            key=key,
        )
        final = trajectory.final_output[..., None, :]
        return jnp.squeeze(
            self.decoder(final, queries, key=key),
            axis=len(batch.case_shape),
        )


__all__ = ["FunctionalStateDecoder", "WeightSpaceRecurrentModel"]
