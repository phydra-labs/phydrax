#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._core import MatrixProductState
from ._precision import TensorNetworkPrecisionPolicy


class TensorTruncationEvidence(StrictModule):
    retained_rank: int
    available_rank: int
    discarded_weight: Array
    valid: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        retained_rank: int,
        available_rank: int,
        discarded_weight: ArrayLike,
        /,
        precision_evidence: PrecisionEvidenceEnvelope,
        precision_policy_id: str,
    ):
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        self.retained_rank = int(retained_rank)
        self.available_rank = int(available_rank)
        self.discarded_weight = jnp.asarray(discarded_weight)
        self.valid = jnp.isfinite(self.discarded_weight) & (self.discarded_weight >= 0.0)
        self.precision_evidence = precision_evidence
        self.precision_policy_id = str(precision_policy_id)


def apply_two_site_gate(
    state: MatrixProductState,
    left_site: int,
    gate: ArrayLike,
    /,
    *,
    maximum_bond_dimension: int,
    normalize: bool = True,
) -> tuple[MatrixProductState, TensorTruncationEvidence]:
    if not isinstance(state, MatrixProductState):
        raise TypeError("state must be a MatrixProductState.")
    site = int(left_site)
    if not 0 <= site < state.site_count - 1:
        raise ValueError("Two-site gate index is out of range.")
    precision = state.precision
    left = precision.contraction(state.tensors[site])
    right = precision.contraction(state.tensors[site + 1])
    d_left = left.shape[1]
    d_right = right.shape[1]
    gate_ = precision.contraction(gate)
    if gate_.shape != (d_left, d_right, d_left, d_right):
        raise ValueError("Gate shape must be (out_left,out_right,in_left,in_right).")
    theta = jnp.tensordot(left, right, axes=(-1, 0))
    theta = jnp.einsum("abij,lijr->labr", gate_, theta)
    matrix = precision.factorization(
        theta.reshape((left.shape[0] * d_left, d_right * right.shape[-1]))
    )
    u, singular_values, vh = jnp.linalg.svd(matrix, full_matrices=False)
    available = singular_values.shape[0]
    retained = min(int(maximum_bond_dimension), available)
    discarded = precision.decision(
        precision.sum(jnp.abs(singular_values[retained:]) ** 2)
    )
    u = u[:, :retained]
    singular_values = singular_values[:retained]
    vh = vh[:retained, :]
    new_left = precision.storage(u.reshape((left.shape[0], d_left, retained)))
    new_right = precision.storage(
        (singular_values[:, None] * vh).reshape((retained, d_right, right.shape[-1]))
    )
    tensors = list(state.tensors)
    tensors[site] = new_left
    tensors[site + 1] = new_right
    result = MatrixProductState(
        tuple(tensors),
        precision=precision,
    )
    if normalize:
        result = result.normalized()
    evidence = precision.evidence_for(
        state.tensors,
        children={"input-state": state.precision_evidence},
        output_value=result.tensors,
    )
    return result, TensorTruncationEvidence(
        retained,
        available,
        discarded,
        evidence,
        precision.policy_id,
    )


def product_mps(
    local_states: ArrayLike,
    /,
    *,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> MatrixProductState:
    values = jnp.asarray(local_states)
    if values.ndim != 2:
        raise ValueError("Product MPS inputs require shape (site, physical).")
    return MatrixProductState(
        tuple(value[None, :, None] for value in values),
        precision=precision,
    )


__all__ = ["TensorTruncationEvidence", "apply_two_site_gate", "product_mps"]
