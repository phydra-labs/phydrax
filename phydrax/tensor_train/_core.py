#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import prod, sqrt
from numbers import Number
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


def _positive_modes(mode_sizes: Sequence[int], /) -> tuple[int, ...]:
    modes = tuple(int(size) for size in mode_sizes)
    if not modes or any(size <= 0 for size in modes):
        raise ValueError("Tensor-train mode sizes must be a nonempty tuple of positives.")
    return modes


def _rank_caps(max_ranks: int | Sequence[int], order: int, /) -> tuple[int, ...]:
    if order <= 1:
        if isinstance(max_ranks, int):
            if max_ranks <= 0:
                raise ValueError("max_ranks must be positive.")
        elif tuple(max_ranks):
            raise ValueError("An order-one tensor has no TT cuts.")
        return ()
    if isinstance(max_ranks, int):
        caps = (int(max_ranks),) * (order - 1)
    else:
        caps = tuple(int(rank) for rank in max_ranks)
    if len(caps) != order - 1 or any(rank <= 0 for rank in caps):
        raise ValueError("max_ranks must provide one positive cap for every TT cut.")
    return caps


def _validate_relative_tolerance(relative_tolerance: float, /) -> float:
    value = float(relative_tolerance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative.")
    return value


def _selected_rank(singular_values: Array, cap: int, cut_tolerance: float, /) -> int:
    values = np.asarray(singular_values)
    maximum = min(int(cap), int(values.size))
    if cut_tolerance <= 0.0:
        return maximum
    squared = np.abs(values) ** 2
    for rank in range(1, maximum + 1):
        if float(np.sqrt(np.sum(squared[rank:]))) <= cut_tolerance:
            return rank
    return maximum


class TTRoundingEvidence(StrictModule):
    """Per-cut discarded singular mass and its rigorous TT Frobenius RSS bound."""

    per_cut_discarded_frobenius: Array
    frobenius_error_bound: Array
    relative_error_bound: Array
    input_frobenius_norm: Array
    output_ranks: tuple[int, ...] = eqx.field(static=True)
    max_ranks: tuple[int, ...] = eqx.field(static=True)
    requested_relative_tolerance: float = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    tolerance_met: bool = eqx.field(static=True)
    status: str = eqx.field(static=True)

    def __init__(
        self,
        per_cut_discarded_frobenius: ArrayLike,
        input_frobenius_norm: ArrayLike,
        output_ranks: Sequence[int],
        max_ranks: Sequence[int],
        requested_relative_tolerance: float,
        /,
    ):
        discarded = jnp.asarray(per_cut_discarded_frobenius)
        norm = jnp.asarray(input_frobenius_norm)
        if discarded.ndim != 1 or norm.shape != ():
            raise ValueError("Rounding evidence requires a cut vector and scalar norm.")
        ranks = tuple(int(rank) for rank in output_ranks)
        caps = tuple(int(rank) for rank in max_ranks)
        if discarded.shape != (len(ranks),) or len(caps) != len(ranks):
            raise ValueError("Rounding evidence cut counts must agree.")
        bound = jnp.sqrt(jnp.sum(discarded**2))
        safe_norm = jnp.where(norm > 0, norm, jnp.asarray(1, dtype=norm.dtype))
        relative_bound = jnp.where(norm > 0, bound / safe_norm, bound)
        tolerance = _validate_relative_tolerance(requested_relative_tolerance)
        dtype = np.asarray(norm).real.dtype
        slack = 32 * np.finfo(dtype).eps
        tolerance_met = bool(np.asarray(relative_bound <= tolerance + slack))
        self.per_cut_discarded_frobenius = discarded
        self.frobenius_error_bound = bound
        self.relative_error_bound = relative_bound
        self.input_frobenius_norm = norm
        self.output_ranks = ranks
        self.max_ranks = caps
        self.requested_relative_tolerance = tolerance
        self.exact = bool(np.all(np.asarray(discarded) == 0))
        self.tolerance_met = tolerance_met
        self.status = (
            "tolerance_met" if tolerance_met else "rank_cap_reached_before_tolerance"
        )


class TensorTrain(StrictModule):
    """Immutable tensor train with cores shaped ``(left_rank, mode, right_rank)``."""

    cores: tuple[Array, ...]
    mode_sizes: tuple[int, ...] = eqx.field(static=True)
    ranks: tuple[int, ...] = eqx.field(static=True)
    tensor_id: str = eqx.field(static=True)

    def __init__(self, cores: Sequence[ArrayLike], /):
        arrays = tuple(jnp.asarray(core) for core in cores)
        if not arrays or any(core.ndim != 3 for core in arrays):
            raise ValueError("TensorTrain requires nonempty rank-three cores.")
        if arrays[0].shape[0] != 1 or arrays[-1].shape[2] != 1:
            raise ValueError("TensorTrain boundary ranks must equal one.")
        for left, right in pairwise(arrays):
            if left.shape[2] != right.shape[0]:
                raise ValueError("Adjacent TensorTrain bond ranks must agree.")
        dtype = arrays[0].dtype
        if any(core.dtype != dtype for core in arrays):
            raise TypeError("All TensorTrain cores must have one dtype.")
        modes = _positive_modes(tuple(core.shape[1] for core in arrays))
        ranks = tuple(int(core.shape[2]) for core in arrays[:-1])
        self.cores = arrays
        self.mode_sizes = modes
        self.ranks = ranks
        self.tensor_id = canonical_fingerprint(
            {
                "kind": "tensor-train",
                "modes": modes,
                "ranks": ranks,
                "dtype": str(dtype),
            }
        )

    @property
    def order(self) -> int:
        return len(self.cores)

    @property
    def dtype(self):
        return self.cores[0].dtype

    @staticmethod
    def from_dense(
        tensor: ArrayLike,
        /,
        *,
        max_ranks: int | Sequence[int],
        relative_tolerance: float = 0.0,
    ) -> TensorTrainCompressionResult:
        """TT-SVD with explicit cut caps and complete truncation evidence."""
        return tt_svd(
            tensor,
            max_ranks=max_ranks,
            relative_tolerance=relative_tolerance,
        )

    def entry(self, index: Sequence[int], /) -> Array:
        position = tuple(int(value) for value in index)
        if len(position) != self.order:
            raise ValueError("A TensorTrain entry needs one index per mode.")
        if any(
            value < 0 or value >= size
            for value, size in zip(position, self.mode_sizes, strict=True)
        ):
            raise IndexError("TensorTrain entry index is outside its mode.")
        value = self.cores[0][0, position[0], :]
        for core, value_index in zip(self.cores[1:], position[1:], strict=True):
            value = ein.contract("a,ab->b", value, core[:, value_index, :])
        return value[0]

    def evaluate(self, indices: ArrayLike, /) -> Array:
        """Evaluate a statically shaped batch of integer multi-indices."""
        points = jnp.asarray(indices, dtype=jnp.int32)
        if points.ndim < 1 or points.shape[-1] != self.order:
            raise ValueError("indices must have trailing dimension equal to TT order.")
        limits = jnp.asarray(self.mode_sizes, dtype=jnp.int32)
        points = eqx.error_if(
            points,
            jnp.any((points < 0) | (points >= limits)),
            "TensorTrain evaluation index is outside its mode.",
        )
        flat = points.reshape((-1, self.order))

        def evaluate_one(point):
            value = self.cores[0][0, point[0], :]
            for axis, core in enumerate(self.cores[1:], start=1):
                value = ein.contract("a,ab->b", value, core[:, point[axis], :])
            return value[0]

        return jax.vmap(evaluate_one)(flat).reshape(points.shape[:-1])

    def to_dense(self, /, *, max_entries: int) -> Array:
        """Materialize only after an explicit entry budget check."""
        budget = int(max_entries)
        entries = prod(self.mode_sizes)
        if budget <= 0 or entries > budget:
            raise ValueError(
                f"Dense TensorTrain needs {entries} entries, exceeding budget {budget}."
            )
        data = self.cores[0][0, :, :]
        for core in self.cores[1:]:
            data = ein.contract("...a,aib->...ib", data, core)
        return data[..., 0]

    def inner(self, other: TensorTrain, /) -> Array:
        if self.mode_sizes != other.mode_sizes:
            raise ValueError("TensorTrain inner products require identical mode sizes.")
        environment = jnp.ones((1, 1), dtype=jnp.result_type(self.dtype, other.dtype))
        for left, right in zip(self.cores, other.cores, strict=True):
            environment = ein.contract(
                "ab,aic,bid->cd", environment, jnp.conj(left), right
            )
        return environment[0, 0]

    def squared_frobenius_norm(self) -> Array:
        return jnp.real(self.inner(self))

    def frobenius_norm(self) -> Array:
        return jnp.sqrt(jnp.maximum(self.squared_frobenius_norm(), 0))

    def conjugate(self) -> TensorTrain:
        return TensorTrain(tuple(jnp.conj(core) for core in self.cores))

    def __neg__(self) -> TensorTrain:
        cores = (-self.cores[0],) + self.cores[1:]
        return TensorTrain(cores)

    def __add__(self, other: TensorTrain) -> TensorTrain:
        if not isinstance(other, TensorTrain):
            raise TypeError("TensorTrain addition requires another TensorTrain.")
        if self.mode_sizes != other.mode_sizes:
            raise ValueError("TensorTrain addition requires identical mode sizes.")
        if self.order == 1:
            return TensorTrain((self.cores[0] + other.cores[0],))
        cores: list[Array] = [jnp.concatenate((self.cores[0], other.cores[0]), axis=2)]
        for left, right in zip(self.cores[1:-1], other.cores[1:-1], strict=True):
            upper = jnp.concatenate(
                (
                    left,
                    jnp.zeros(
                        (left.shape[0], left.shape[1], right.shape[2]), dtype=left.dtype
                    ),
                ),
                axis=2,
            )
            lower = jnp.concatenate(
                (
                    jnp.zeros(
                        (right.shape[0], right.shape[1], left.shape[2]), dtype=right.dtype
                    ),
                    right,
                ),
                axis=2,
            )
            cores.append(jnp.concatenate((upper, lower), axis=0))
        cores.append(jnp.concatenate((self.cores[-1], other.cores[-1]), axis=0))
        return TensorTrain(tuple(cores))

    def __sub__(self, other: TensorTrain) -> TensorTrain:
        return self + (-other)

    def hadamard(self, other: TensorTrain, /) -> TensorTrain:
        if self.mode_sizes != other.mode_sizes:
            raise ValueError("Hadamard products require identical mode sizes.")
        cores = []
        for left, right in zip(self.cores, other.cores, strict=True):
            product = ein.contract("aib,cid->acibd", left, right)
            cores.append(
                product.reshape(
                    (
                        left.shape[0] * right.shape[0],
                        left.shape[1],
                        left.shape[2] * right.shape[2],
                    )
                )
            )
        return TensorTrain(tuple(cores))

    def __mul__(self, other: Any) -> TensorTrain:
        if isinstance(other, TensorTrain):
            return self.hadamard(other)
        if isinstance(other, Number):
            return TensorTrain((self.cores[0] * other,) + self.cores[1:])
        scalar = jnp.asarray(other)
        if scalar.shape != ():
            raise TypeError(
                "TensorTrain multiplication requires a scalar or TensorTrain."
            )
        return TensorTrain((self.cores[0] * scalar,) + self.cores[1:])

    def __rmul__(self, other: Any) -> TensorTrain:
        return self * other

    def round(
        self,
        /,
        *,
        max_ranks: int | Sequence[int],
        relative_tolerance: float = 0.0,
    ) -> TensorTrainCompressionResult:
        return round_tensor_train(
            self,
            max_ranks=max_ranks,
            relative_tolerance=relative_tolerance,
        )


class TensorTrainCompressionResult(StrictModule):
    tensor: TensorTrain
    evidence: TTRoundingEvidence

    def __init__(self, tensor: TensorTrain, evidence: TTRoundingEvidence, /):
        self.tensor = tensor
        self.evidence = evidence


class TensorTrainOperator(StrictModule):
    """Immutable TT operator with cores ``(left_rank, output, input, right_rank)``."""

    cores: tuple[Array, ...]
    output_mode_sizes: tuple[int, ...] = eqx.field(static=True)
    input_mode_sizes: tuple[int, ...] = eqx.field(static=True)
    ranks: tuple[int, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(self, cores: Sequence[ArrayLike], /):
        arrays = tuple(jnp.asarray(core) for core in cores)
        if not arrays or any(core.ndim != 4 for core in arrays):
            raise ValueError("TensorTrainOperator requires nonempty rank-four cores.")
        if arrays[0].shape[0] != 1 or arrays[-1].shape[3] != 1:
            raise ValueError("TensorTrainOperator boundary ranks must equal one.")
        for left, right in pairwise(arrays):
            if left.shape[3] != right.shape[0]:
                raise ValueError("Adjacent TensorTrainOperator bond ranks must agree.")
        dtype = arrays[0].dtype
        if any(core.dtype != dtype for core in arrays):
            raise TypeError("All TensorTrainOperator cores must have one dtype.")
        outputs = _positive_modes(tuple(core.shape[1] for core in arrays))
        inputs = _positive_modes(tuple(core.shape[2] for core in arrays))
        ranks = tuple(int(core.shape[3]) for core in arrays[:-1])
        self.cores = arrays
        self.output_mode_sizes = outputs
        self.input_mode_sizes = inputs
        self.ranks = ranks
        self.operator_id = canonical_fingerprint(
            {
                "kind": "tensor-train-operator",
                "outputs": outputs,
                "inputs": inputs,
                "ranks": ranks,
                "dtype": str(dtype),
            }
        )

    @property
    def order(self) -> int:
        return len(self.cores)

    @property
    def dtype(self):
        return self.cores[0].dtype

    def adjoint(self) -> TensorTrainOperator:
        return TensorTrainOperator(
            tuple(jnp.swapaxes(jnp.conj(core), 1, 2) for core in self.cores)
        )

    def inner(self, other: TensorTrainOperator, /) -> Array:
        if (
            self.output_mode_sizes != other.output_mode_sizes
            or self.input_mode_sizes != other.input_mode_sizes
        ):
            raise ValueError("TT operator inner products require identical modes.")
        environment = jnp.ones((1, 1), dtype=jnp.result_type(self.dtype, other.dtype))
        for left, right in zip(self.cores, other.cores, strict=True):
            environment = ein.contract(
                "ab,aoic,boid->cd", environment, jnp.conj(left), right
            )
        return environment[0, 0]

    def frobenius_norm(self) -> Array:
        return jnp.sqrt(jnp.maximum(jnp.real(self.inner(self)), 0))

    @staticmethod
    def from_dense(
        matrix: ArrayLike,
        output_mode_sizes: Sequence[int],
        input_mode_sizes: Sequence[int],
        /,
        *,
        max_ranks: int | Sequence[int],
        relative_tolerance: float = 0.0,
    ) -> TensorTrainOperatorCompressionResult:
        outputs = _positive_modes(output_mode_sizes)
        inputs = _positive_modes(input_mode_sizes)
        if len(outputs) != len(inputs):
            raise ValueError("TT operator input and output orders must agree.")
        dense = jnp.asarray(matrix)
        if dense.shape == (prod(outputs), prod(inputs)):
            dense = dense.reshape(outputs + inputs)
        if dense.shape != outputs + inputs:
            raise ValueError("Dense operator shape does not match declared mode sizes.")
        order = len(outputs)
        permutation = tuple(
            axis
            for pair in zip(range(order), range(order, 2 * order), strict=True)
            for axis in pair
        )
        interleaved = jnp.transpose(dense, permutation)
        paired = interleaved.reshape(
            tuple(o * i for o, i in zip(outputs, inputs, strict=True))
        )
        decomposition = tt_svd(
            paired,
            max_ranks=max_ranks,
            relative_tolerance=relative_tolerance,
        )
        cores = tuple(
            core.reshape((core.shape[0], output, input_, core.shape[2]))
            for core, output, input_ in zip(
                decomposition.tensor.cores, outputs, inputs, strict=True
            )
        )
        return TensorTrainOperatorCompressionResult(
            TensorTrainOperator(cores), decomposition.evidence
        )

    def to_dense(self, /, *, max_entries: int) -> Array:
        entries = prod(self.output_mode_sizes) * prod(self.input_mode_sizes)
        budget = int(max_entries)
        if budget <= 0 or entries > budget:
            raise ValueError(
                f"Dense TT operator needs {entries} entries, exceeding budget {budget}."
            )
        data = self.cores[0][0, :, :, :]
        for core in self.cores[1:]:
            data = ein.contract("...a,aoib->...oib", data, core)
        interleaved = data[..., 0]
        order = self.order
        outputs = tuple(range(0, 2 * order, 2))
        inputs = tuple(range(1, 2 * order, 2))
        return jnp.transpose(interleaved, outputs + inputs)

    def to_matrix(self, /, *, max_entries: int) -> Array:
        return self.to_dense(max_entries=max_entries).reshape(
            (prod(self.output_mode_sizes), prod(self.input_mode_sizes))
        )

    def entry(
        self,
        output_index: Sequence[int],
        input_index: Sequence[int],
        /,
    ) -> Array:
        output = tuple(int(value) for value in output_index)
        input_ = tuple(int(value) for value in input_index)
        if len(output) != self.order or len(input_) != self.order:
            raise ValueError(
                "TT operator entries need one input and output index per mode."
            )
        if any(
            value < 0 or value >= size
            for value, size in zip(output, self.output_mode_sizes, strict=True)
        ) or any(
            value < 0 or value >= size
            for value, size in zip(input_, self.input_mode_sizes, strict=True)
        ):
            raise IndexError("TT operator entry index is outside its physical mode.")
        value = self.cores[0][0, output[0], input_[0], :]
        for core, out, inp in zip(self.cores[1:], output[1:], input_[1:], strict=True):
            value = ein.contract("a,ab->b", value, core[:, out, inp, :])
        return value[0]

    def evaluate(
        self,
        output_indices: ArrayLike,
        input_indices: ArrayLike,
        /,
    ) -> Array:
        """Evaluate a statically shaped batch of input/output multi-index pairs."""
        outputs = jnp.asarray(output_indices, dtype=jnp.int32)
        inputs = jnp.asarray(input_indices, dtype=jnp.int32)
        if (
            outputs.shape != inputs.shape
            or outputs.ndim < 1
            or outputs.shape[-1] != self.order
        ):
            raise ValueError("TT operator index batches must agree and match its order.")
        output_limits = jnp.asarray(self.output_mode_sizes, dtype=jnp.int32)
        input_limits = jnp.asarray(self.input_mode_sizes, dtype=jnp.int32)
        outputs = eqx.error_if(
            outputs,
            jnp.any((outputs < 0) | (outputs >= output_limits)),
            "TT operator output index is outside its mode.",
        )
        inputs = eqx.error_if(
            inputs,
            jnp.any((inputs < 0) | (inputs >= input_limits)),
            "TT operator input index is outside its mode.",
        )
        flat_outputs = outputs.reshape((-1, self.order))
        flat_inputs = inputs.reshape((-1, self.order))

        def evaluate_one(output, input_):
            value = self.cores[0][0, output[0], input_[0], :]
            for axis, core in enumerate(self.cores[1:], start=1):
                value = ein.contract(
                    "a,ab->b", value, core[:, output[axis], input_[axis], :]
                )
            return value[0]

        values = jax.vmap(evaluate_one)(flat_outputs, flat_inputs)
        return values.reshape(outputs.shape[:-1])

    def apply(self, tensor: TensorTrain, /) -> TensorTrain:
        """Apply exactly; this method never rounds or hides rank growth."""
        if self.input_mode_sizes != tensor.mode_sizes:
            raise ValueError("TT operator input modes do not match the tensor.")
        cores = []
        for operator_core, tensor_core in zip(self.cores, tensor.cores, strict=True):
            product = ein.contract("aoib,cid->acobd", operator_core, tensor_core)
            cores.append(
                product.reshape(
                    (
                        operator_core.shape[0] * tensor_core.shape[0],
                        operator_core.shape[1],
                        operator_core.shape[3] * tensor_core.shape[2],
                    )
                )
            )
        return TensorTrain(tuple(cores))

    def apply_and_round(
        self,
        tensor: TensorTrain,
        /,
        *,
        max_ranks: int | Sequence[int],
        relative_tolerance: float = 0.0,
    ) -> TensorTrainCompressionResult:
        return self.apply(tensor).round(
            max_ranks=max_ranks,
            relative_tolerance=relative_tolerance,
        )

    def round(
        self,
        /,
        *,
        max_ranks: int | Sequence[int],
        relative_tolerance: float = 0.0,
    ) -> TensorTrainOperatorCompressionResult:
        """Canonicalize and round paired physical modes with explicit evidence."""
        paired = TensorTrain(
            tuple(
                core.reshape(
                    (core.shape[0], core.shape[1] * core.shape[2], core.shape[3])
                )
                for core in self.cores
            )
        )
        rounded = paired.round(
            max_ranks=max_ranks,
            relative_tolerance=relative_tolerance,
        )
        cores = tuple(
            core.reshape((core.shape[0], output, input_, core.shape[2]))
            for core, output, input_ in zip(
                rounded.tensor.cores,
                self.output_mode_sizes,
                self.input_mode_sizes,
                strict=True,
            )
        )
        return TensorTrainOperatorCompressionResult(
            TensorTrainOperator(cores), rounded.evidence
        )

    def compose(self, other: TensorTrainOperator, /) -> TensorTrainOperator:
        """Return exact composition ``self(other(x))`` without compression."""
        if self.input_mode_sizes != other.output_mode_sizes:
            raise ValueError("Composed TT operator modes do not agree.")
        cores = []
        for left, right in zip(self.cores, other.cores, strict=True):
            product = ein.contract("aomb,cmid->acoibd", left, right)
            cores.append(
                product.reshape(
                    (
                        left.shape[0] * right.shape[0],
                        left.shape[1],
                        right.shape[2],
                        left.shape[3] * right.shape[3],
                    )
                )
            )
        return TensorTrainOperator(tuple(cores))

    def compose_and_round(
        self,
        other: TensorTrainOperator,
        /,
        *,
        max_ranks: int | Sequence[int],
        relative_tolerance: float = 0.0,
    ) -> TensorTrainOperatorCompressionResult:
        return self.compose(other).round(
            max_ranks=max_ranks,
            relative_tolerance=relative_tolerance,
        )

    def __neg__(self) -> TensorTrainOperator:
        return TensorTrainOperator((-self.cores[0],) + self.cores[1:])

    def __add__(self, other: TensorTrainOperator) -> TensorTrainOperator:
        if not isinstance(other, TensorTrainOperator):
            raise TypeError("TT operator addition requires another TT operator.")
        if (
            self.output_mode_sizes != other.output_mode_sizes
            or self.input_mode_sizes != other.input_mode_sizes
        ):
            raise ValueError("TT operator addition requires identical physical modes.")
        if self.order == 1:
            return TensorTrainOperator((self.cores[0] + other.cores[0],))
        cores: list[Array] = [jnp.concatenate((self.cores[0], other.cores[0]), axis=3)]
        for left, right in zip(self.cores[1:-1], other.cores[1:-1], strict=True):
            upper = jnp.concatenate(
                (
                    left,
                    jnp.zeros(
                        (left.shape[0], left.shape[1], left.shape[2], right.shape[3]),
                        dtype=left.dtype,
                    ),
                ),
                axis=3,
            )
            lower = jnp.concatenate(
                (
                    jnp.zeros(
                        (right.shape[0], right.shape[1], right.shape[2], left.shape[3]),
                        dtype=right.dtype,
                    ),
                    right,
                ),
                axis=3,
            )
            cores.append(jnp.concatenate((upper, lower), axis=0))
        cores.append(jnp.concatenate((self.cores[-1], other.cores[-1]), axis=0))
        return TensorTrainOperator(tuple(cores))

    def __sub__(self, other: TensorTrainOperator) -> TensorTrainOperator:
        return self + (-other)

    def __mul__(self, scalar: Any) -> TensorTrainOperator:
        if isinstance(scalar, Number):
            return TensorTrainOperator((self.cores[0] * scalar,) + self.cores[1:])
        value = jnp.asarray(scalar)
        if value.shape != ():
            raise TypeError("TT operator multiplication requires a scalar.")
        return TensorTrainOperator((self.cores[0] * value,) + self.cores[1:])

    def __rmul__(self, scalar: Any) -> TensorTrainOperator:
        return self * scalar


class TensorTrainOperatorCompressionResult(StrictModule):
    operator: TensorTrainOperator
    evidence: TTRoundingEvidence

    def __init__(
        self,
        operator: TensorTrainOperator,
        evidence: TTRoundingEvidence,
        /,
    ):
        self.operator = operator
        self.evidence = evidence


def tt_svd(
    tensor: ArrayLike,
    /,
    *,
    max_ranks: int | Sequence[int],
    relative_tolerance: float = 0.0,
) -> TensorTrainCompressionResult:
    """Compute a bounded TT-SVD and rigorous RSS Frobenius truncation bound."""
    dense = jnp.asarray(tensor)
    if dense.ndim < 1 or any(size <= 0 for size in dense.shape):
        raise ValueError("TT-SVD requires a nonempty tensor of order at least one.")
    if not np.issubdtype(np.dtype(dense.dtype), np.inexact):
        dense = dense.astype(jnp.float32)
    tolerance = _validate_relative_tolerance(relative_tolerance)
    caps = _rank_caps(max_ranks, dense.ndim)
    input_norm = jnp.sqrt(jnp.sum(jnp.abs(dense) ** 2))
    cut_tolerance = (
        tolerance * float(np.asarray(input_norm)) / sqrt(max(dense.ndim - 1, 1))
    )
    cores: list[Array] = []
    discarded: list[Array] = []
    unfolding = dense
    left_rank = 1
    for cut, mode_size in enumerate(dense.shape[:-1]):
        matrix = unfolding.reshape((left_rank * mode_size, -1))
        left, singular_values, right = jnp.linalg.svd(matrix, full_matrices=False)
        rank = _selected_rank(singular_values, caps[cut], cut_tolerance)
        tail = singular_values[rank:]
        discarded.append(jnp.sqrt(jnp.sum(jnp.abs(tail) ** 2)))
        cores.append(left[:, :rank].reshape((left_rank, mode_size, rank)))
        unfolding = singular_values[:rank, None] * right[:rank, :]
        left_rank = rank
    cores.append(unfolding.reshape((left_rank, dense.shape[-1], 1)))
    train = TensorTrain(tuple(cores))
    evidence = TTRoundingEvidence(
        jnp.stack(discarded) if discarded else jnp.zeros((0,), dtype=input_norm.dtype),
        input_norm,
        train.ranks,
        caps,
        tolerance,
    )
    return TensorTrainCompressionResult(train, evidence)


def _right_orthogonalize(tensor: TensorTrain, /) -> tuple[Array, ...]:
    cores = list(tensor.cores)
    for axis in range(tensor.order - 1, 0, -1):
        core = cores[axis]
        matrix = core.reshape((core.shape[0], core.shape[1] * core.shape[2]))
        orthogonal, triangular = jnp.linalg.qr(matrix.T, mode="reduced")
        cores[axis] = orthogonal.T.reshape(
            (orthogonal.shape[1], core.shape[1], core.shape[2])
        )
        cores[axis - 1] = ein.contract("aib,bc->aic", cores[axis - 1], triangular.T)
    return tuple(cores)


def round_tensor_train(
    tensor: TensorTrain,
    /,
    *,
    max_ranks: int | Sequence[int],
    relative_tolerance: float = 0.0,
) -> TensorTrainCompressionResult:
    """Canonical TT rounding with every discarded mode included in an RSS bound."""
    if not isinstance(tensor, TensorTrain):
        raise TypeError("round_tensor_train expects a TensorTrain.")
    if not np.issubdtype(np.dtype(tensor.dtype), np.inexact):
        tensor = TensorTrain(tuple(core.astype(jnp.float32) for core in tensor.cores))
    tolerance = _validate_relative_tolerance(relative_tolerance)
    caps = _rank_caps(max_ranks, tensor.order)
    input_norm = tensor.frobenius_norm()
    cut_tolerance = (
        tolerance * float(np.asarray(input_norm)) / sqrt(max(tensor.order - 1, 1))
    )
    cores = list(_right_orthogonalize(tensor))
    discarded: list[Array] = []
    for cut in range(tensor.order - 1):
        core = cores[cut]
        matrix = core.reshape((core.shape[0] * core.shape[1], core.shape[2]))
        left, singular_values, right = jnp.linalg.svd(matrix, full_matrices=False)
        rank = _selected_rank(singular_values, caps[cut], cut_tolerance)
        discarded.append(jnp.sqrt(jnp.sum(jnp.abs(singular_values[rank:]) ** 2)))
        cores[cut] = left[:, :rank].reshape((core.shape[0], core.shape[1], rank))
        transfer = singular_values[:rank, None] * right[:rank, :]
        cores[cut + 1] = ein.contract("ab,bic->aic", transfer, cores[cut + 1])
    rounded = TensorTrain(tuple(cores))
    evidence = TTRoundingEvidence(
        jnp.stack(discarded) if discarded else jnp.zeros((0,), dtype=input_norm.dtype),
        input_norm,
        rounded.ranks,
        caps,
        tolerance,
    )
    return TensorTrainCompressionResult(rounded, evidence)


__all__ = [
    "TTRoundingEvidence",
    "TensorTrain",
    "TensorTrainCompressionResult",
    "TensorTrainOperator",
    "TensorTrainOperatorCompressionResult",
    "round_tensor_train",
    "tt_svd",
]
