#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike

import phydrax.ein as ein

from ._core import MatrixProductState
from ._precision import TensorNetworkPrecisionPolicy
from ._split import TensorTruncationEvidence, truncated_svd


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
    if int(maximum_bond_dimension) < 1:
        raise ValueError("maximum_bond_dimension must be positive.")
    precision = state.precision
    left = precision.contraction(state.tensors[site])
    right = precision.contraction(state.tensors[site + 1])
    d_left = left.shape[1]
    d_right = right.shape[1]
    gate_ = precision.contraction(gate)
    if gate_.shape != (d_left, d_right, d_left, d_right):
        raise ValueError("Gate shape must be (out_left,out_right,in_left,in_right).")
    theta = ein.contract("lpi,iqr->lpqr", left, right)
    theta = ein.contract("abij,lijr->labr", gate_, theta)
    matrix = theta.reshape((left.shape[0] * d_left, d_right * right.shape[-1]))
    left_factor, right_factor, truncation = truncated_svd(
        matrix,
        maximum_rank=maximum_bond_dimension,
        absorb="right",
        precision=precision,
        evidence_source=state.tensors,
        evidence_children={"input-state": state.precision_evidence},
    )
    retained = truncation.retained_rank
    new_left = left_factor.reshape((left.shape[0], d_left, retained))
    new_right = right_factor.reshape((retained, d_right, right.shape[-1]))
    tensors = list(state.tensors)
    tensors[site] = new_left
    tensors[site + 1] = new_right
    result = MatrixProductState(
        tuple(tensors),
        precision=precision,
    )
    if normalize:
        result = result.normalized()
    return result, truncation


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
