#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ._core import TensorTrainOperator, TTRoundingEvidence


class TensorTrainLinearCompressionEvidence(StrictModule):
    """Proven TT-SVD bound plus measured finite-matrix compression error."""

    rounding: TTRoundingEvidence
    measured_frobenius_error: Array
    bound_satisfied: bool = eqx.field(static=True)

    def __init__(
        self,
        rounding: TTRoundingEvidence,
        measured_frobenius_error: ArrayLike,
        /,
    ):
        measured = jnp.asarray(measured_frobenius_error)
        if measured.shape != ():
            raise ValueError("Linear compression error must be scalar.")
        self.rounding = rounding
        self.measured_frobenius_error = measured
        slack = (
            32
            * np.finfo(np.asarray(measured).dtype).eps
            * max(float(np.asarray(rounding.input_frobenius_norm)), 1.0)
        )
        self.bound_satisfied = bool(
            np.asarray(measured <= rounding.frobenius_error_bound + slack)
        )


class TensorTrainLinear(StrictModule):
    """Dense-feature linear layer backed by a genuine TT operator."""

    operator: TensorTrainOperator
    bias: Array
    compression_evidence: TensorTrainLinearCompressionEvidence
    input_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        operator: TensorTrainOperator,
        bias: ArrayLike | None,
        compression_evidence: TensorTrainLinearCompressionEvidence,
        /,
    ):
        output_size = prod(operator.output_mode_sizes)
        if bias is None:
            bias_array = jnp.zeros((output_size,), dtype=operator.dtype)
            use_bias = False
        else:
            bias_array = jnp.asarray(bias)
            if bias_array.shape != (output_size,):
                raise ValueError(
                    "TensorTrainLinear bias must match flattened output size."
                )
            use_bias = True
        self.operator = operator
        self.bias = bias_array
        self.compression_evidence = compression_evidence
        self.input_shape = operator.input_mode_sizes
        self.output_shape = operator.output_mode_sizes
        self.use_bias = use_bias

    @staticmethod
    def from_dense(
        weight: ArrayLike,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        /,
        *,
        bias: ArrayLike | None,
        max_ranks: int | Sequence[int],
        relative_tolerance: float,
        max_dense_entries: int,
    ) -> TensorTrainLinear:
        inputs = tuple(int(size) for size in input_shape)
        outputs = tuple(int(size) for size in output_shape)
        matrix = jnp.asarray(weight)
        if matrix.shape != (prod(outputs), prod(inputs)):
            raise ValueError(
                "Dense linear weight does not match input/output tensor shapes."
            )
        if matrix.size > int(max_dense_entries):
            raise ValueError(
                "Dense linear weight exceeds its explicit compression budget."
            )
        decomposition = TensorTrainOperator.from_dense(
            matrix,
            outputs,
            inputs,
            max_ranks=max_ranks,
            relative_tolerance=relative_tolerance,
        )
        reconstructed = decomposition.operator.to_matrix(max_entries=max_dense_entries)
        measured = jnp.sqrt(jnp.sum(jnp.abs(matrix - reconstructed) ** 2))
        evidence = TensorTrainLinearCompressionEvidence(decomposition.evidence, measured)
        return TensorTrainLinear(decomposition.operator, bias, evidence)

    def _forward_one(self, flattened: Array, /) -> Array:
        tensor = flattened.reshape(self.input_shape)
        order = len(self.input_shape)
        input_labels = list(range(order))
        output_labels = list(range(order, 2 * order))
        bond_labels = list(range(2 * order, 3 * order + 1))
        operands: list[object] = [tensor, input_labels]
        for axis, core in enumerate(self.operator.cores):
            operands.extend(
                [
                    core,
                    [
                        bond_labels[axis],
                        output_labels[axis],
                        input_labels[axis],
                        bond_labels[axis + 1],
                    ],
                ]
            )
        operands.append(output_labels)
        output = ein.contract(*operands)
        return output.reshape((-1,)) + self.bias

    def __call__(self, inputs: ArrayLike, /) -> Array:
        values = jnp.asarray(inputs)
        input_size = prod(self.input_shape)
        if values.ndim < 1 or values.shape[-1] != input_size:
            raise ValueError(
                "TensorTrainLinear expects flattened features on its last axis."
            )
        flat = values.reshape((-1, input_size))
        output = jax.vmap(self._forward_one)(flat)
        return output.reshape(values.shape[:-1] + (prod(self.output_shape),))


__all__ = ["TensorTrainLinear", "TensorTrainLinearCompressionEvidence"]
