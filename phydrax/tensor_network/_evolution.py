#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._core import MatrixProductState


class TensorTruncationEvidence(StrictModule):
    retained_rank: int
    available_rank: int
    discarded_weight: Array
    valid: Array

    def __init__(
        self,
        retained_rank: int,
        available_rank: int,
        discarded_weight: ArrayLike,
        /,
    ):
        self.retained_rank = int(retained_rank)
        self.available_rank = int(available_rank)
        self.discarded_weight = jnp.asarray(discarded_weight)
        self.valid = jnp.isfinite(self.discarded_weight) & (self.discarded_weight >= 0.0)


def apply_two_site_gate(
    state: MatrixProductState,
    left_site: int,
    gate: ArrayLike,
    /,
    *,
    maximum_bond_dimension: int,
) -> tuple[MatrixProductState, TensorTruncationEvidence]:
    if not isinstance(state, MatrixProductState):
        raise TypeError("state must be a MatrixProductState.")
    site = int(left_site)
    if not 0 <= site < state.site_count - 1:
        raise ValueError("Two-site gate index is out of range.")
    left = state.tensors[site]
    right = state.tensors[site + 1]
    d_left = left.shape[1]
    d_right = right.shape[1]
    gate_ = jnp.asarray(gate)
    if gate_.shape != (d_left, d_right, d_left, d_right):
        raise ValueError("Gate shape must be (out_left,out_right,in_left,in_right).")
    theta = jnp.tensordot(left, right, axes=(-1, 0))
    theta = jnp.einsum("abij,lijr->labr", gate_, theta)
    matrix = theta.reshape((left.shape[0] * d_left, d_right * right.shape[-1]))
    u, singular_values, vh = jnp.linalg.svd(matrix, full_matrices=False)
    available = singular_values.shape[0]
    retained = min(int(maximum_bond_dimension), available)
    discarded = jnp.sum(singular_values[retained:] ** 2)
    u = u[:, :retained]
    singular_values = singular_values[:retained]
    vh = vh[:retained, :]
    new_left = u.reshape((left.shape[0], d_left, retained))
    new_right = (singular_values[:, None] * vh).reshape(
        (retained, d_right, right.shape[-1])
    )
    tensors = list(state.tensors)
    tensors[site] = new_left
    tensors[site + 1] = new_right
    result = MatrixProductState(tuple(tensors)).normalized()
    return result, TensorTruncationEvidence(retained, available, discarded)


def product_mps(local_states: ArrayLike, /) -> MatrixProductState:
    values = jnp.asarray(local_states)
    if values.ndim != 2:
        raise ValueError("Product MPS inputs require shape (site, physical).")
    return MatrixProductState(tuple(value[None, :, None] for value in values))


__all__ = ["TensorTruncationEvidence", "apply_two_site_gate", "product_mps"]
