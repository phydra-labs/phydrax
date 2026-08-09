#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax.nn._keys import EvalKey
from phydrax.nn.layers import RecurrentBatch
from phydrax.nn.models import WeightSpaceRecurrentModel
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel
from phydrax.nn.parameters import ParameterSubspace


WeightSpaceOutputSize = int | Literal["scalar"]


def _contract_configuration(
    model: WeightSpaceOperator,
) -> tuple[tuple[str, object], ...]:
    return (
        ("parameter_dimension", model.model.decoder.subspace.total_dimension),
        ("parameter_paths", model.model.decoder.subspace.leaf_paths),
        ("input_mode", model.model.recurrence.input_mode),
        ("execution", model.model.execution),
        ("source_key", model.source_key),
    )


class WeightSpaceOperator(AbstractOperatorModel):
    """Reconstruct a coordinate function from recurrent selected root-model weights."""

    operator_architecture = "WeightSpaceOperator"
    _operator_contract_configuration = staticmethod(_contract_configuration)

    in_size: int
    out_size: WeightSpaceOutputSize
    query_size: int
    source_key: str | None
    model: WeightSpaceRecurrentModel

    def __init__(
        self,
        subspace: ParameterSubspace,
        /,
        *,
        observation_size: int,
        query_size: int,
        out_channels: WeightSpaceOutputSize = "scalar",
        source_key: str | None = None,
        execution: str = "associative",
        input_mode: str = "difference",
        maximum_retention: float = 0.999,
        input_scale: float = 1e-2,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(subspace, ParameterSubspace):
            raise TypeError("subspace must be a ParameterSubspace.")
        self.in_size = int(observation_size)
        self.out_size = out_channels
        self.query_size = int(query_size)
        self.source_key = source_key
        if self.in_size <= 0 or self.query_size <= 0:
            raise ValueError("observation_size and query_size must be positive.")
        if out_channels != "scalar" and int(out_channels) <= 0:
            raise ValueError("out_channels must be 'scalar' or a positive integer.")
        self.model = WeightSpaceRecurrentModel(
            subspace,
            self.in_size,
            self.query_size,
            execution=execution,
            input_mode=input_mode,
            maximum_retention=maximum_retention,
            input_scale=input_scale,
            key=key,
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("source_key is required for multiple operator inputs.")
        return next(iter(batch.inputs.values()))

    def _observation_batch(
        self,
        source: FunctionSamples,
        case_shape: tuple[int, ...],
        /,
    ) -> RecurrentBatch:
        if source.values is None or len(source.sample_shape) != 1:
            raise ValueError(
                "WeightSpaceOperator requires one temporal source sample axis."
            )
        values = jnp.asarray(source.values)
        length = source.sample_shape[0]
        if self.in_size == 1 and values.shape == case_shape + (length,):
            values = values[..., None]
        expected = case_shape + (length, self.in_size)
        if values.shape != expected:
            raise ValueError(f"Source observations must have shape {expected}.")
        times = source.coordinates_array(case_shape=case_shape, flatten=True)
        if int(times.shape[-1]) != 1:
            raise ValueError(
                "Weight-space source coordinates must contain physical time only."
            )
        return RecurrentBatch(
            values,
            source.mask_array(case_shape=case_shape),
            time=times[..., 0],
        )

    def _validate_output_shape(self, output_shape: tuple[int, ...], /) -> None:
        expected = () if self.out_size == "scalar" else (int(self.out_size),)
        if output_shape != expected:
            raise ValueError(
                "Reconstructed root-model output shape does not match out_channels: "
                f"expected {expected}, got {output_shape}."
            )

    def evaluate_trajectory(
        self,
        observations: RecurrentBatch,
        queries: Array,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        """Decode the root function at every recurrent parameter state."""
        decoded = self.model(observations, queries, key=key)
        self._validate_output_shape(
            tuple(decoded.shape[len(observations.case_shape) + 2 :])
        )
        return decoded

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        source = self._source(batch)
        query = batch.require_single_query()
        observations = self._observation_batch(source, batch.case_shape)
        query_coordinates = query.coordinates_array(
            case_shape=batch.case_shape,
            flatten=True,
        )
        if int(query_coordinates.shape[-1]) != self.query_size:
            raise ValueError(f"Query coordinates must have dimension {self.query_size}.")
        decoded = self.model.evaluate_final(observations, query_coordinates, key=key)
        self._validate_output_shape(tuple(decoded.shape[len(batch.case_shape) + 1 :]))
        output_shape = decoded.shape[len(batch.case_shape) + 1 :]
        restored = decoded.reshape(batch.case_shape + query.sample_shape + output_shape)
        query_mask = query.mask_array(case_shape=batch.case_shape)
        mask = query_mask.reshape(
            batch.case_shape + query.sample_shape + (1,) * len(output_shape)
        )
        return jnp.where(mask, restored, jnp.zeros_like(restored))

    def __call__(
        self,
        x: tuple[RecurrentBatch, Array] | OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x, key=key)
        if (
            not isinstance(x, tuple)
            or len(x) != 2
            or not isinstance(x[0], RecurrentBatch)
        ):
            raise TypeError("WeightSpaceOperator requires (RecurrentBatch, queries).")
        return self.evaluate_trajectory(x[0], x[1], key=key)


__all__ = ["WeightSpaceOperator"]
