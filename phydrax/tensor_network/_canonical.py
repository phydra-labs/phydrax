#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ._core import LocallyPurifiedDensity, MatrixProductState
from ._precision import TensorNetworkPrecisionPolicy


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


def _canonical_sweep(
    tensors: tuple[Array, ...],
    center: int,
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> tuple[Array, ...]:
    values = list(precision.factorization(tensors))
    for index in range(center):
        tensor = values[index]
        matrix = tensor.reshape((-1, tensor.shape[-1]))
        q, r = jnp.linalg.qr(matrix)
        rank = q.shape[-1]
        values[index] = q.reshape(tensor.shape[:-1] + (rank,))
        values[index + 1] = ein.contract("ab,b...->a...", r, values[index + 1])
    for index in range(len(values) - 1, center, -1):
        tensor = values[index]
        matrix = tensor.reshape((tensor.shape[0], -1))
        q, r = jnp.linalg.qr(matrix.T)
        rank = q.shape[-1]
        values[index] = q.T.reshape((rank,) + tensor.shape[1:])
        values[index - 1] = ein.contract("...a,ab->...b", values[index - 1], r.T)
    return tuple(precision.storage(values))


def _canonical_residuals(
    tensors: tuple[Array, ...],
    center: int,
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> tuple[Array, Array]:
    values = precision.accumulation(tensors)
    real_dtype = values[0].real.dtype
    left = []
    right = []
    for index, tensor in enumerate(values):
        if index < center:
            matrix = tensor.reshape((-1, tensor.shape[-1]))
            identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
            left.append(precision.norm(jnp.conj(matrix.T) @ matrix - identity))
        else:
            left.append(jnp.asarray(0.0, dtype=real_dtype))
        if index > center:
            matrix = tensor.reshape((tensor.shape[0], -1))
            identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
            right.append(precision.norm(matrix @ jnp.conj(matrix.T) - identity))
        else:
            right.append(jnp.asarray(0.0, dtype=real_dtype))
    return jnp.stack(left), jnp.stack(right)


def _mps_evidence(
    tensors: tuple[Array, ...],
    center: int,
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> MPSCanonicalEvidence:
    left, right = _canonical_residuals(tensors, center, precision)
    return MPSCanonicalEvidence(
        left,
        right,
        precision.norm(tensors[center]),
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
    tensors = _canonical_sweep(state.tensors, center_, state.precision)
    canonical = MatrixProductState(tensors, precision=state.precision)
    if normalize:
        norm = canonical.norm()
        norm = eqx.error_if(
            norm,
            ~jnp.isfinite(norm) | (norm <= 0.0),
            "MPS norm must be finite and positive.",
        )
        tensors = list(canonical.tensors)
        tensors[center_] = tensors[center_] / norm
        canonical = MatrixProductState(tuple(tensors), precision=state.precision)
    return canonical, _mps_evidence(canonical.tensors, center_, canonical.precision)


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
    tensors = _canonical_sweep(state.tensors, center_, state.precision)
    result = LocallyPurifiedDensity(tensors, precision=state.precision)
    left, right = _canonical_residuals(result.tensors, center_, result.precision)
    return result, LPDOCanonicalEvidence(
        left,
        right,
        result.raw_trace(),
        center=center_,
    )


__all__ = [
    "LPDOCanonicalEvidence",
    "MPSCanonicalEvidence",
    "canonicalize_lpdo",
    "canonicalize_mps",
]
