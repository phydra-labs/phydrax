#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
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


def scale_mpo(
    operator: MatrixProductOperator, coefficient: ArrayLike, /
) -> MatrixProductOperator:
    """Scale an MPO without changing its finite-chain structure."""
    if not isinstance(operator, MatrixProductOperator):
        raise TypeError("operator must be a MatrixProductOperator.")
    value = jnp.asarray(coefficient)
    if value.ndim != 0:
        raise ValueError("MPO coefficient must be scalar.")
    tensors = (operator.tensors[0] * value,) + operator.tensors[1:]
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


def apply_mpo_exact(
    operator: MatrixProductOperator,
    state: MatrixProductState,
    /,
) -> MatrixProductState:
    """Apply an MPO exactly, exposing the resulting product bond capacity."""
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
    return MatrixProductState(tuple(tensors), precision=precision)


def apply_mpo(
    operator: MatrixProductOperator,
    state: MatrixProductState,
    /,
    *,
    maximum_bond_dimension: int,
    normalize: bool = False,
) -> tuple[MatrixProductState, ChainCompressionEvidence]:
    exact = apply_mpo_exact(operator, state)
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


class VariationalCompressionPolicy(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    maximum_sweeps: int = eqx.field(static=True)
    gradient_step: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_tensor_elements: int = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_bond_dimension: int,
        maximum_sweeps: int = 8,
        gradient_step: float = 0.05,
        residual_tolerance: float = 1e-8,
        maximum_tensor_elements: int = 10_000_000,
    ):
        bond = int(maximum_bond_dimension)
        sweeps = int(maximum_sweeps)
        step = float(gradient_step)
        tolerance = float(residual_tolerance)
        elements = int(maximum_tensor_elements)
        if (
            bond < 1
            or sweeps < 1
            or not isfinite(step)
            or step <= 0.0
            or not isfinite(tolerance)
            or tolerance < 0.0
            or elements < 1
        ):
            raise ValueError("Variational compression policy values are invalid.")
        self.maximum_bond_dimension = bond
        self.maximum_sweeps = sweeps
        self.gradient_step = step
        self.residual_tolerance = tolerance
        self.maximum_tensor_elements = elements


class VariationalCompressionEvidence(StrictModule):
    objective_history: Array
    gradient_residual_history: Array
    discarded_weight_history: Array
    active_sweeps: Array
    converged: Array
    initial_compression: ChainCompressionEvidence


def _tuple_inner(left, right, /):
    environment = jnp.ones((1, 1), dtype=jnp.result_type(left[0], right[0]))
    for first, second in zip(left, right, strict=True):
        environment = oe.contract("ab,api,bpj->ij", environment, jnp.conj(first), second)
    return environment.reshape(())


def _compression_objective(candidate, target, target_norm, /):
    own = jnp.real(_tuple_inner(candidate, candidate))
    cross = jnp.real(_tuple_inner(candidate, target))
    return jnp.maximum(own - 2.0 * cross + target_norm, 0.0)


def variational_compress_mps(
    target: MatrixProductState,
    policy: VariationalCompressionPolicy,
    /,
) -> tuple[MatrixProductState, VariationalCompressionEvidence]:
    """Minimize the MPS distance by bounded projected tensor-gradient sweeps."""
    if not isinstance(target, MatrixProductState) or not isinstance(
        policy, VariationalCompressionPolicy
    ):
        raise TypeError("Variational compression requires an MPS and policy.")
    target_elements = sum(int(tensor.size) for tensor in target.tensors)
    if target_elements > policy.maximum_tensor_elements:
        raise MemoryError("Variational compression exceeds maximum_tensor_elements.")
    candidate, initial_evidence = compress_mps(
        target,
        maximum_bond_dimension=policy.maximum_bond_dimension,
        normalize=False,
    )
    real_dtype = target.tensors[0].real.dtype
    objectives = jnp.full((policy.maximum_sweeps + 1,), jnp.nan, dtype=real_dtype)
    residuals = jnp.full((policy.maximum_sweeps,), jnp.nan, dtype=real_dtype)
    discarded = jnp.full((policy.maximum_sweeps,), jnp.nan, dtype=real_dtype)
    active = jnp.zeros((policy.maximum_sweeps,), dtype=bool)
    target_values = target.precision.accumulation(target.tensors)
    target_norm = jnp.real(_tuple_inner(target_values, target_values))
    objective = _compression_objective(
        target.precision.accumulation(candidate.tensors),
        target_values,
        target_norm,
    )
    objectives = objectives.at[0].set(objective)
    converged = jnp.asarray(False)

    for sweep in range(policy.maximum_sweeps):
        candidate_values = target.precision.accumulation(candidate.tensors)
        gradient = jax.grad(_compression_objective)(
            candidate_values, target_values, target_norm
        )
        residual = jnp.sqrt(sum(jnp.real(jnp.vdot(value, value)) for value in gradient))
        updated = MatrixProductState(
            tuple(
                value - policy.gradient_step * derivative
                for value, derivative in zip(candidate.tensors, gradient, strict=True)
            ),
            precision=target.precision,
        )
        candidate, sweep_evidence = compress_mps(
            updated,
            maximum_bond_dimension=policy.maximum_bond_dimension,
            normalize=False,
        )
        objective = _compression_objective(
            target.precision.accumulation(candidate.tensors),
            target_values,
            target_norm,
        )
        objectives = objectives.at[sweep + 1].set(objective)
        residuals = residuals.at[sweep].set(residual)
        discarded = discarded.at[sweep].set(sweep_evidence.accumulated_discarded_weight)
        active = active.at[sweep].set(True)
        converged = residual <= policy.residual_tolerance
        if bool(converged):
            break
    evidence = VariationalCompressionEvidence(
        objectives,
        residuals,
        discarded,
        active,
        converged,
        initial_evidence,
    )
    return candidate, evidence


__all__ = [
    "ChainCompressionEvidence",
    "VariationalCompressionEvidence",
    "VariationalCompressionPolicy",
    "add_mpo",
    "adjoint_mpo",
    "apply_mpo",
    "apply_mpo_exact",
    "compose_mpo",
    "compress_mpo",
    "compress_mps",
    "product_mpo",
    "scale_mpo",
    "variational_compress_mps",
]
