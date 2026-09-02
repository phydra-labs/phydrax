#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._core import TensorTrain, TensorTrainOperator


BoundaryKind = Literal["periodic", "dirichlet", "neumann"]


class BoundaryPolicy(StrictModule):
    """Explicit finite-grid boundary behavior for shifts and differences."""

    kind: BoundaryKind = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, kind: BoundaryKind, /):
        if kind not in ("periodic", "dirichlet", "neumann"):
            raise ValueError("Boundary kind must be periodic, dirichlet, or neumann.")
        self.kind = kind
        self.policy_id = canonical_fingerprint(
            {"kind": "tensor-train-boundary-policy", "boundary": kind}
        )


def kronecker_operator(factors: Sequence[ArrayLike], /) -> TensorTrainOperator:
    """Represent an exact Cartesian Kronecker product with TT rank one."""
    matrices = tuple(jnp.asarray(factor) for factor in factors)
    if not matrices or any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("Kronecker operator factors must be a nonempty matrix tuple.")
    dtype = jnp.result_type(*matrices)
    return TensorTrainOperator(
        tuple(matrix.astype(dtype)[None, :, :, None] for matrix in matrices)
    )


def identity_operator(
    mode_sizes: Sequence[int], /, *, dtype=jnp.float32
) -> TensorTrainOperator:
    modes = tuple(int(size) for size in mode_sizes)
    if not modes or any(size <= 0 for size in modes):
        raise ValueError("Identity operator modes must be nonempty and positive.")
    return kronecker_operator(tuple(jnp.eye(size, dtype=dtype) for size in modes))


def cartesian_identity(
    mode_sizes: Sequence[int], /, *, dtype=jnp.float32
) -> TensorTrainOperator:
    return identity_operator(mode_sizes, dtype=dtype)


def _shift_matrix(
    size: int,
    offset: int,
    boundary: BoundaryPolicy,
    dtype,
    /,
) -> Array:
    if boundary.kind == "periodic":
        return jnp.roll(jnp.eye(size, dtype=dtype), int(offset), axis=0)
    output = jnp.arange(size, dtype=jnp.int32)
    source = output - int(offset)
    if boundary.kind == "neumann":
        source = jnp.clip(source, 0, size - 1)
        return jnp.zeros((size, size), dtype=dtype).at[output, source].set(1)
    valid = (source >= 0) & (source < size)
    safe_source = jnp.clip(source, 0, size - 1)
    return (
        jnp.zeros((size, size), dtype=dtype)
        .at[output, safe_source]
        .set(valid.astype(dtype))
    )


def shift_operator(
    mode_sizes: Sequence[int],
    axis: int,
    /,
    *,
    offset: int,
    boundary: BoundaryPolicy,
    dtype=jnp.float32,
) -> TensorTrainOperator:
    modes = tuple(int(size) for size in mode_sizes)
    position = int(axis)
    if not modes or any(size <= 0 for size in modes):
        raise ValueError("Shift operator modes must be nonempty and positive.")
    if position < 0 or position >= len(modes):
        raise ValueError("Shift operator axis is outside the Cartesian shape.")
    factors = [jnp.eye(size, dtype=dtype) for size in modes]
    factors[position] = _shift_matrix(modes[position], int(offset), boundary, dtype)
    return kronecker_operator(tuple(factors))


def _negative_laplacian_factor(
    size: int,
    spacing: float,
    boundary: BoundaryPolicy,
    dtype,
    /,
) -> Array:
    if size < 2:
        raise ValueError("Cartesian Laplacian axes require at least two sites.")
    h = float(spacing)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("Cartesian spacing must be finite and positive.")
    matrix = 2 * jnp.eye(size, dtype=dtype)
    neighbor = jnp.arange(size - 1, dtype=jnp.int32)
    matrix = matrix.at[neighbor, neighbor + 1].set(-1)
    matrix = matrix.at[neighbor + 1, neighbor].set(-1)
    if boundary.kind == "periodic":
        matrix = matrix.at[0, -1].add(-1)
        matrix = matrix.at[-1, 0].add(-1)
    elif boundary.kind == "neumann":
        matrix = matrix.at[0, 0].set(1)
        matrix = matrix.at[-1, -1].set(1)
    return matrix / (h * h)


def laplacian_operator(
    mode_sizes: Sequence[int],
    /,
    *,
    spacing: float | Sequence[float],
    boundary: BoundaryPolicy | Sequence[BoundaryPolicy],
    dtype=jnp.float32,
) -> TensorTrainOperator:
    """Exact Kronecker-sum negative Cartesian Laplacian."""
    modes = tuple(int(size) for size in mode_sizes)
    if not modes or any(size <= 1 for size in modes):
        raise ValueError("Laplacian modes must all exceed one.")
    spacings = (
        (float(spacing),) * len(modes)
        if isinstance(spacing, (int, float))
        else tuple(float(value) for value in spacing)
    )
    boundaries = (
        (boundary,) * len(modes)
        if isinstance(boundary, BoundaryPolicy)
        else tuple(boundary)
    )
    if len(spacings) != len(modes) or len(boundaries) != len(modes):
        raise ValueError("Laplacian spacing and boundary policies must match its order.")
    terms = []
    for axis, (size, h, policy) in enumerate(
        zip(modes, spacings, boundaries, strict=True)
    ):
        factors = [jnp.eye(mode, dtype=dtype) for mode in modes]
        factors[axis] = _negative_laplacian_factor(size, h, policy, dtype)
        terms.append(kronecker_operator(tuple(factors)))
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


def diagonal_operator(diagonal: TensorTrain, /) -> TensorTrainOperator:
    """Lift one TT exactly to the diagonal of a TT operator."""
    cores = []
    for core in diagonal.cores:
        identity = jnp.eye(core.shape[1], dtype=core.dtype)
        cores.append(core[:, :, None, :] * identity[None, :, :, None])
    return TensorTrainOperator(tuple(cores))


__all__ = [
    "BoundaryKind",
    "BoundaryPolicy",
    "cartesian_identity",
    "diagonal_operator",
    "identity_operator",
    "kronecker_operator",
    "laplacian_operator",
    "shift_operator",
]
