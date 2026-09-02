#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._canonical import _canonical_sweep
from ._core import MatrixProductOperator, MatrixProductState
from ._precision import TensorNetworkPrecisionPolicy
from ._split import TensorTruncationEvidence, truncated_svd


class ChainCompressionEvidence(StrictModule):
    """Ordered local loss records for one finite-chain compression."""

    truncations: tuple[TensorTruncationEvidence, ...]
    accumulated_discarded_weight: Array
    maximum_discarded_weight: Array
    valid: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        truncations: tuple[TensorTruncationEvidence, ...],
        /,
        *,
        precision_evidence: PrecisionEvidenceEnvelope,
        precision_policy_id: str,
        real_dtype,
    ):
        records = tuple(truncations)
        if any(not isinstance(record, TensorTruncationEvidence) for record in records):
            raise TypeError("truncations must contain TensorTruncationEvidence values.")
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        weights = (
            jnp.stack([record.discarded_weight for record in records])
            if records
            else jnp.zeros((0,), dtype=real_dtype)
        )
        self.truncations = records
        self.accumulated_discarded_weight = jnp.sum(weights)
        self.maximum_discarded_weight = (
            jnp.max(weights) if records else jnp.asarray(0.0, dtype=real_dtype)
        )
        self.valid = (
            jnp.all(jnp.stack([record.valid for record in records]))
            if records
            else jnp.asarray(True)
        )
        self.precision_evidence = precision_evidence
        self.precision_policy_id = str(precision_policy_id)


def _require_same_precision(left, right, /) -> TensorNetworkPrecisionPolicy:
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("Tensor-network precision policies must match.")
    return left.precision


def _compression_evidence(result, records, /) -> ChainCompressionEvidence:
    return ChainCompressionEvidence(
        tuple(records),
        precision_evidence=result.precision_evidence,
        precision_policy_id=result.precision.policy_id,
        real_dtype=result.tensors[0].real.dtype,
    )


def product_mpo(
    local_operators: ArrayLike,
    /,
    *,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> MatrixProductOperator:
    values = jnp.asarray(local_operators)
    if values.ndim != 3:
        raise ValueError("Product MPO inputs require shape (site, output, input).")
    return MatrixProductOperator(
        tuple(value[None, :, :, None] for value in values),
        precision=precision,
    )


def adjoint_mpo(operator: MatrixProductOperator, /) -> MatrixProductOperator:
    if not isinstance(operator, MatrixProductOperator):
        raise TypeError("operator must be a MatrixProductOperator.")
    tensors = tuple(jnp.swapaxes(jnp.conj(tensor), 1, 2) for tensor in operator.tensors)
    return MatrixProductOperator(tensors, precision=operator.precision)


def add_mpo(
    left: MatrixProductOperator,
    right: MatrixProductOperator,
    /,
) -> MatrixProductOperator:
    if not isinstance(left, MatrixProductOperator) or not isinstance(
        right, MatrixProductOperator
    ):
        raise TypeError("left and right must be MatrixProductOperator values.")
    precision = _require_same_precision(left, right)
    if (
        left.output_dimensions != right.output_dimensions
        or left.input_dimensions != right.input_dimensions
    ):
        raise ValueError("MPO dimensions must match for addition.")
    if left.site_count == 1:
        return MatrixProductOperator(
            (left.tensors[0] + right.tensors[0],), precision=precision
        )

    tensors: list[Array] = [jnp.concatenate((left.tensors[0], right.tensors[0]), axis=-1)]
    for first, second in zip(left.tensors[1:-1], right.tensors[1:-1], strict=True):
        shape = (
            first.shape[0] + second.shape[0],
            first.shape[1],
            first.shape[2],
            first.shape[3] + second.shape[3],
        )
        block = jnp.zeros(shape, dtype=jnp.result_type(first, second))
        block = block.at[: first.shape[0], :, :, : first.shape[3]].set(first)
        block = block.at[first.shape[0] :, :, :, first.shape[3] :].set(second)
        tensors.append(block)
    tensors.append(jnp.concatenate((left.tensors[-1], right.tensors[-1]), axis=0))
    return MatrixProductOperator(tuple(tensors), precision=precision)


def compress_mps(
    state: MatrixProductState,
    /,
    *,
    maximum_bond_dimension: int,
    normalize: bool = False,
) -> tuple[MatrixProductState, ChainCompressionEvidence]:
    if not isinstance(state, MatrixProductState):
        raise TypeError("state must be a MatrixProductState.")
    capacity = int(maximum_bond_dimension)
    if capacity < 1:
        raise ValueError("maximum_bond_dimension must be positive.")
    precision = state.precision
    tensors = list(_canonical_sweep(state.tensors, 0, precision))
    records = []
    for index in range(state.site_count - 1):
        tensor = tensors[index]
        matrix = tensor.reshape((-1, tensor.shape[-1]))
        left, right, evidence = truncated_svd(
            matrix,
            maximum_rank=capacity,
            absorb="right",
            precision=precision,
            evidence_source=tuple(tensors),
            evidence_children={"input-state": state.precision_evidence},
        )
        retained = evidence.retained_rank
        tensors[index] = left.reshape(tensor.shape[:-1] + (retained,))
        tensors[index + 1] = precision.storage(
            oe.contract(
                "ab,bpr->apr",
                precision.contraction(right),
                precision.contraction(tensors[index + 1]),
            )
        )
        records.append(evidence)
    result = MatrixProductState(tuple(tensors), precision=precision)
    if normalize:
        result = result.normalized()
    return result, _compression_evidence(result, records)


def compress_mpo(
    operator: MatrixProductOperator,
    /,
    *,
    maximum_bond_dimension: int,
) -> tuple[MatrixProductOperator, ChainCompressionEvidence]:
    if not isinstance(operator, MatrixProductOperator):
        raise TypeError("operator must be a MatrixProductOperator.")
    capacity = int(maximum_bond_dimension)
    if capacity < 1:
        raise ValueError("maximum_bond_dimension must be positive.")
    precision = operator.precision
    tensors = list(_canonical_sweep(operator.tensors, 0, precision))
    records = []
    for index in range(operator.site_count - 1):
        tensor = tensors[index]
        matrix = tensor.reshape((-1, tensor.shape[-1]))
        left, right, evidence = truncated_svd(
            matrix,
            maximum_rank=capacity,
            absorb="right",
            precision=precision,
            evidence_source=tuple(tensors),
            evidence_children={"input-operator": operator.precision_evidence},
        )
        retained = evidence.retained_rank
        tensors[index] = left.reshape(tensor.shape[:-1] + (retained,))
        tensors[index + 1] = precision.storage(
            oe.contract(
                "ab,boir->aoir",
                precision.contraction(right),
                precision.contraction(tensors[index + 1]),
            )
        )
        records.append(evidence)
    result = MatrixProductOperator(tuple(tensors), precision=precision)
    return result, _compression_evidence(result, records)


def apply_mpo(
    operator: MatrixProductOperator,
    state: MatrixProductState,
    /,
    *,
    maximum_bond_dimension: int,
    normalize: bool = False,
) -> tuple[MatrixProductState, ChainCompressionEvidence]:
    if not isinstance(operator, MatrixProductOperator) or not isinstance(
        state, MatrixProductState
    ):
        raise TypeError("operator and state must be MPO and MPS values.")
    precision = _require_same_precision(operator, state)
    if operator.site_count != state.site_count:
        raise ValueError("MPO and MPS site counts must match.")
    if operator.input_dimensions != state.physical_dimensions:
        raise ValueError("MPO input dimensions must match MPS physical dimensions.")
    tensors = []
    for op_tensor, state_tensor in zip(operator.tensors, state.tensors, strict=True):
        combined = oe.contract(
            "aoib,cid->acobd",
            precision.contraction(op_tensor),
            precision.contraction(state_tensor),
        )
        tensors.append(
            precision.storage(
                combined.reshape(
                    (
                        op_tensor.shape[0] * state_tensor.shape[0],
                        op_tensor.shape[1],
                        op_tensor.shape[-1] * state_tensor.shape[-1],
                    )
                )
            )
        )
    exact = MatrixProductState(tuple(tensors), precision=precision)
    return compress_mps(
        exact,
        maximum_bond_dimension=maximum_bond_dimension,
        normalize=normalize,
    )


def compose_mpo(
    left: MatrixProductOperator,
    right: MatrixProductOperator,
    /,
    *,
    maximum_bond_dimension: int,
) -> tuple[MatrixProductOperator, ChainCompressionEvidence]:
    if not isinstance(left, MatrixProductOperator) or not isinstance(
        right, MatrixProductOperator
    ):
        raise TypeError("left and right must be MatrixProductOperator values.")
    precision = _require_same_precision(left, right)
    if left.site_count != right.site_count:
        raise ValueError("MPO site counts must match for composition.")
    if left.input_dimensions != right.output_dimensions:
        raise ValueError("Left MPO inputs must match right MPO outputs.")
    tensors = []
    for first, second in zip(left.tensors, right.tensors, strict=True):
        combined = oe.contract(
            "aomb,cmid->acoibd",
            precision.contraction(first),
            precision.contraction(second),
        )
        tensors.append(
            precision.storage(
                combined.reshape(
                    (
                        first.shape[0] * second.shape[0],
                        first.shape[1],
                        second.shape[2],
                        first.shape[-1] * second.shape[-1],
                    )
                )
            )
        )
    exact = MatrixProductOperator(tuple(tensors), precision=precision)
    return compress_mpo(exact, maximum_bond_dimension=maximum_bond_dimension)


__all__ = [
    "ChainCompressionEvidence",
    "add_mpo",
    "adjoint_mpo",
    "apply_mpo",
    "compose_mpo",
    "compress_mpo",
    "compress_mps",
    "product_mpo",
]
