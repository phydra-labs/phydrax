#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..data import (
    FunctionSamples,
    OperatorAxis,
    OperatorBatch,
    OperatorFieldBatch,
    OperatorPrediction,
    OperatorTargetBatch,
)


DTypeName = Literal["float16", "bfloat16", "float32", "float64"]


def _dtype(name: DTypeName):
    return {
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
        "float64": jnp.float64,
    }[name]


def _cast_inexact(value: Any, dtype, /):
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        return array
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        target = jnp.complex128 if jnp.dtype(dtype).itemsize >= 8 else jnp.complex64
        return array.astype(target)
    return array.astype(dtype)


@dataclass(frozen=True)
class OperatorDTypePolicy:
    """Explicit parameter, forward-compute, and reduction precision."""

    parameter_dtype: DTypeName = "float32"
    compute_dtype: DTypeName = "float32"
    reduction_dtype: DTypeName = "float32"

    def __post_init__(self):
        for value in (
            self.parameter_dtype,
            self.compute_dtype,
            self.reduction_dtype,
        ):
            _dtype(value)

    def cast_model(self, model: Any, /) -> Any:
        dtype = _dtype(self.parameter_dtype)
        return jax.tree_util.tree_map(
            lambda leaf: (
                _cast_inexact(leaf, dtype) if isinstance(leaf, jax.Array) else leaf
            ),
            model,
        )

    def _samples(self, samples: FunctionSamples, /) -> FunctionSamples:
        dtype = _dtype(self.compute_dtype)
        values = None if samples.values is None else _cast_inexact(samples.values, dtype)
        axes = tuple(
            OperatorAxis(
                axis.name,
                _cast_inexact(axis.nodes, dtype),
                quadrature_weights=(
                    None
                    if axis.quadrature_weights is None
                    else _cast_inexact(axis.quadrature_weights, dtype)
                ),
                basis=axis.basis,
                periodic=axis.periodic,
            )
            for axis in samples.axes
        )
        topology = (
            None
            if samples.topology is None
            else jax.tree_util.tree_map(
                lambda leaf: (
                    _cast_inexact(leaf, dtype) if isinstance(leaf, jax.Array) else leaf
                ),
                samples.topology,
            )
        )
        return FunctionSamples(
            values=values,
            axes=axes,
            coordinates=(
                None
                if samples.coordinates is None
                else _cast_inexact(samples.coordinates, dtype)
            ),
            quadrature_weights=(
                None
                if samples.quadrature_weights is None
                else _cast_inexact(samples.quadrature_weights, dtype)
            ),
            mask=samples.mask,
            topology=topology,
        )

    def cast_batch(self, batch: OperatorBatch, /) -> OperatorBatch:
        return OperatorBatch(
            inputs={
                name: self._samples(samples) for name, samples in batch.inputs.items()
            },
            queries={
                name: self._samples(samples) for name, samples in batch.queries.items()
            },
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )

    def cast_targets(
        self,
        targets: OperatorTargetBatch,
        /,
    ) -> OperatorTargetBatch:
        return targets.map_values(
            lambda value: _cast_inexact(value, _dtype(self.compute_dtype))
        )

    def cast_prediction(
        self,
        prediction: OperatorPrediction,
        /,
    ) -> OperatorPrediction:
        return OperatorPrediction(
            {
                name: OperatorFieldBatch(
                    _cast_inexact(field.values, _dtype(self.compute_dtype)),
                    query_name=field.query_name,
                    spec=field.spec,
                )
                for name, field in prediction.fields.items()
            },
            prediction.queries,
            case_axes=prediction.case_axes,
            case_shape=prediction.case_shape,
        )

    def reduction(self, value: Any, /) -> Array:
        return _cast_inexact(value, _dtype(self.reduction_dtype))

    def to_dict(self) -> dict[str, str]:
        return {
            "parameter_dtype": self.parameter_dtype,
            "compute_dtype": self.compute_dtype,
            "reduction_dtype": self.reduction_dtype,
        }

    @classmethod
    def from_dict(cls, value: dict[str, str], /) -> "OperatorDTypePolicy":
        return cls(
            parameter_dtype=cast(DTypeName, value["parameter_dtype"]),
            compute_dtype=cast(DTypeName, value["compute_dtype"]),
            reduction_dtype=cast(DTypeName, value["reduction_dtype"]),
        )


__all__ = ["DTypeName", "OperatorDTypePolicy"]
