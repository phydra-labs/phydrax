#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._core import LocallyPurifiedDensity, MatrixProductState


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


class LPDOCanonicalEvidence(StrictModule):
    left_residuals: Array
    right_residuals: Array
    raw_trace: Array
    valid: Array
    center: int

    def __init__(
        self,
        left_residuals: ArrayLike,
        right_residuals: ArrayLike,
        raw_trace: ArrayLike,
        /,
        *,
        center: int,
    ):
        self.left_residuals = jnp.asarray(left_residuals)
        self.right_residuals = jnp.asarray(right_residuals)
        self.raw_trace = jnp.asarray(raw_trace)
        self.valid = (
            jnp.all(jnp.isfinite(self.left_residuals))
            & jnp.all(jnp.isfinite(self.right_residuals))
            & jnp.isfinite(self.raw_trace)
            & (self.raw_trace > 0.0)
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
    normalize: bool = True,
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
    canonical = MatrixProductState(tuple(tensors))
    if normalize:
        canonical = canonical.normalized()
    return canonical, _evidence(canonical.tensors, center_)


def canonicalize_lpdo(
    state: LocallyPurifiedDensity,
    /,
    *,
    center: int,
) -> tuple[LocallyPurifiedDensity, LPDOCanonicalEvidence]:
    if not isinstance(state, LocallyPurifiedDensity):
        raise TypeError("state must be a LocallyPurifiedDensity.")
    center_ = int(center)
    if not 0 <= center_ < state.site_count:
        raise ValueError("center is outside the LPDO.")
    tensors = list(state.tensors)
    for index in range(center_):
        tensor = tensors[index]
        matrix = tensor.reshape((-1, tensor.shape[-1]))
        q, r = jnp.linalg.qr(matrix)
        rank = q.shape[-1]
        tensors[index] = q.reshape(
            (tensor.shape[0], tensor.shape[1], tensor.shape[2], rank)
        )
        tensors[index + 1] = jnp.tensordot(r, tensors[index + 1], axes=(1, 0))
    for index in range(state.site_count - 1, center_, -1):
        tensor = tensors[index]
        matrix = tensor.reshape((tensor.shape[0], -1))
        q, r = jnp.linalg.qr(matrix.T)
        rank = q.shape[-1]
        tensors[index] = q.T.reshape(
            (rank, tensor.shape[1], tensor.shape[2], tensor.shape[3])
        )
        tensors[index - 1] = jnp.tensordot(tensors[index - 1], r.T, axes=(-1, 0))
    result = LocallyPurifiedDensity(tuple(tensors))
    left = []
    right = []
    for index, tensor in enumerate(result.tensors):
        left_matrix = tensor.reshape((-1, tensor.shape[-1]))
        right_matrix = tensor.reshape((tensor.shape[0], -1))
        left.append(
            jnp.max(
                jnp.abs(
                    jnp.conj(left_matrix.T) @ left_matrix - jnp.eye(left_matrix.shape[-1])
                )
            )
            if index < center_
            else jnp.asarray(0.0)
        )
        right.append(
            jnp.max(
                jnp.abs(
                    right_matrix @ jnp.conj(right_matrix.T)
                    - jnp.eye(right_matrix.shape[0])
                )
            )
            if index > center_
            else jnp.asarray(0.0)
        )
    return result, LPDOCanonicalEvidence(
        jnp.stack(left),
        jnp.stack(right),
        result.raw_trace(),
        center=center_,
    )


__all__ = [
    "LPDOCanonicalEvidence",
    "MPSCanonicalEvidence",
    "canonicalize_lpdo",
    "canonicalize_mps",
]
