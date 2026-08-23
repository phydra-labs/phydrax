#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._core import MatrixProductState


class MPSCanonicalEvidence(StrictModule):
    left_residuals: Array
    right_residuals: Array
    center_norm: Array
    valid: Array
    center: int

    def __init__(
        self,
        left_residuals: ArrayLike,
        right_residuals: ArrayLike,
        center_norm: ArrayLike,
        /,
        *,
        center: int,
    ):
        self.left_residuals = jnp.asarray(left_residuals)
        self.right_residuals = jnp.asarray(right_residuals)
        self.center_norm = jnp.asarray(center_norm)
        self.valid = (
            jnp.all(jnp.isfinite(self.left_residuals))
            & jnp.all(jnp.isfinite(self.right_residuals))
            & jnp.isfinite(self.center_norm)
        )
        self.center = int(center)


def _evidence(tensors: tuple[Array, ...], center: int, /) -> MPSCanonicalEvidence:
    left = []
    right = []
    for index, tensor in enumerate(tensors):
        if index < center:
            matrix = tensor.reshape((-1, tensor.shape[-1]))
            left.append(
                jnp.max(jnp.abs(jnp.conj(matrix.T) @ matrix - jnp.eye(matrix.shape[-1])))
            )
        else:
            left.append(jnp.asarray(0.0))
        if index > center:
            matrix = tensor.reshape((tensor.shape[0], -1))
            right.append(
                jnp.max(jnp.abs(matrix @ jnp.conj(matrix.T) - jnp.eye(matrix.shape[0])))
            )
        else:
            right.append(jnp.asarray(0.0))
    return MPSCanonicalEvidence(
        jnp.stack(left),
        jnp.stack(right),
        jnp.linalg.norm(tensors[center]),
        center=center,
    )


def canonicalize_mps(
    state: MatrixProductState,
    /,
    *,
    center: int,
) -> tuple[MatrixProductState, MPSCanonicalEvidence]:
    if not isinstance(state, MatrixProductState):
        raise TypeError("state must be a MatrixProductState.")
    center_ = int(center)
    if not 0 <= center_ < state.site_count:
        raise ValueError("center is outside the MPS.")
    tensors = list(state.tensors)
    for index in range(center_):
        tensor = tensors[index]
        matrix = tensor.reshape((-1, tensor.shape[-1]))
        q, r = jnp.linalg.qr(matrix)
        rank = q.shape[-1]
        tensors[index] = q.reshape((tensor.shape[0], tensor.shape[1], rank))
        tensors[index + 1] = jnp.tensordot(r, tensors[index + 1], axes=(1, 0))
    for index in range(state.site_count - 1, center_, -1):
        tensor = tensors[index]
        matrix = tensor.reshape((tensor.shape[0], -1))
        q, r = jnp.linalg.qr(matrix.T)
        rank = q.shape[-1]
        tensors[index] = q.T.reshape((rank, tensor.shape[1], tensor.shape[2]))
        tensors[index - 1] = jnp.tensordot(tensors[index - 1], r.T, axes=(-1, 0))
    canonical = MatrixProductState(tuple(tensors)).normalized()
    return canonical, _evidence(canonical.tensors, center_)


__all__ = ["MPSCanonicalEvidence", "canonicalize_mps"]
