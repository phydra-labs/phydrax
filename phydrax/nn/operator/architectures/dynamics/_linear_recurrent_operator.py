#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers import LinearRecurrentUnit, RecurrentBatch
from phydrax.nn.models import LinearRecurrentModel
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


def _contract_configuration(
    model: LinearRecurrentOperator,
) -> tuple[tuple[str, object], ...]:
    return (
        ("input_size", model.input_size),
        ("output_size", model.output_size),
        ("state_size", model.state_size),
        ("execution", model.execution),
        ("time_axis", model.time_axis),
        ("source_key", model.source_key),
        ("time_semantics", "ordered_samples"),
    )


class LinearRecurrentOperator(AbstractOperatorModel):
    """Causal LRU operator on coincident ordered sample sequences.

    The sample coordinates determine ordering only. The recurrence is discrete and
    does not claim invariance to physical-time reparameterization.
    """

    operator_architecture = "LinearRecurrentOperator"
    _operator_contract_configuration = staticmethod(_contract_configuration)

    model: LinearRecurrentModel
    input_size: int | Literal["scalar"]
    output_size: int | Literal["scalar"]
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    state_size: int
    execution: Literal["serial", "associative"]
    time_axis: str
    source_key: str | None

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] | None = None,
        state_size: int = 32,
        execution: Literal["serial", "associative"] = "associative",
        time_axis: str = "time",
        source_key: str | None = None,
        minimum_radius: float = 0.0,
        maximum_radius: float = 0.999,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        resolved_output = in_channels if out_channels is None else out_channels
        input_count = _get_size(in_channels)
        output_count = _get_size(resolved_output)
        if execution not in ("serial", "associative"):
            raise ValueError("execution must be 'serial' or 'associative'.")
        if not str(time_axis):
            raise ValueError("time_axis must be non-empty.")
        self.input_size = in_channels
        self.output_size = resolved_output
        self.in_size = in_channels
        self.out_size = resolved_output
        self.state_size = int(state_size)
        self.execution = execution
        self.time_axis = str(time_axis)
        self.source_key = source_key
        unit = LinearRecurrentUnit(
            input_count,
            self.state_size,
            output_size=output_count,
            min_radius=minimum_radius,
            max_initial_radius=maximum_radius,
            dtype=dtype,
            key=key,
        )
        self.model = LinearRecurrentModel(
            unit,
            execution=execution,
            return_mode="sequence",
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("source_key is required for multiple OperatorBatch inputs.")
        return next(iter(batch.inputs.values()))

    def _sequence_data(
        self,
        batch: OperatorBatch,
        /,
    ) -> tuple[RecurrentBatch, Array]:
        source = self._source(batch)
        query = batch.require_single_query()
        if source.values is None:
            raise ValueError("LinearRecurrentOperator source values cannot be None.")
        if source.sample_shape != query.sample_shape or len(source.sample_shape) != 1:
            raise ValueError("Source and query must share one coincident sequence axis.")
        if source.axes and (
            len(source.axes) != 1 or source.axes[0].name != self.time_axis
        ):
            raise ValueError(
                f"Source tensor grid requires one {self.time_axis!r} sample axis."
            )
        if query.axes and (len(query.axes) != 1 or query.axes[0].name != self.time_axis):
            raise ValueError(
                f"Query tensor grid requires one {self.time_axis!r} sample axis."
            )
        source_coordinates = source.coordinates_array(case_shape=batch.case_shape)
        query_coordinates = query.coordinates_array(case_shape=batch.case_shape)
        source_mask = source.mask_array(case_shape=batch.case_shape)
        query_mask = query.mask_array(case_shape=batch.case_shape)
        source_coordinates = eqx.error_if(
            source_coordinates,
            jnp.any(query_mask & ~source_mask),
            "Every valid query node requires a valid coincident source node.",
        )
        source_coordinates = eqx.error_if(
            source_coordinates,
            jnp.any(query_mask[..., None] & (source_coordinates != query_coordinates)),
            "Source and query sample coordinates must coincide.",
        )
        values = jnp.asarray(source.values)
        if self.input_size == "scalar":
            values = values[..., None]
        return RecurrentBatch(values, source_mask), query_mask

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        sequence, query_mask = self._sequence_data(batch)
        output = self.model(sequence, key=key)
        if self.output_size == "scalar":
            output = output[..., 0]
        mask = query_mask if self.output_size == "scalar" else query_mask[..., None]
        return jnp.where(mask, output, jnp.zeros_like(output))

    def __call__(
        self,
        inputs: OperatorBatch | RecurrentBatch,
        /,
        *,
        initial_state: Array | None = None,
        key: EvalKey = None,
    ) -> Array:
        if isinstance(inputs, OperatorBatch):
            if initial_state is not None:
                raise ValueError(
                    "OperatorBatch evaluation does not accept initial_state."
                )
            return self.__call_operator_batch__(inputs, key=key)
        if not isinstance(inputs, RecurrentBatch):
            raise TypeError("inputs must be an OperatorBatch or RecurrentBatch.")
        output = self.model(inputs, initial_state=initial_state, key=key)
        return output[..., 0] if self.output_size == "scalar" else output


__all__ = ["LinearRecurrentOperator"]
