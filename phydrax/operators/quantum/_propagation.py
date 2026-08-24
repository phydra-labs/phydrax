#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike


def _square_matrix(value: ArrayLike, name: str, /) -> Array:
    matrix = jnp.asarray(value)
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError(f"{name} must have square trailing matrix axes.")
    if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must use complex floating-point coordinates.")
    return matrix


def apply_unitary_to_state(unitary: ArrayLike, state: ArrayLike, /) -> Array:
    """Apply one unitary matrix or trajectory to state vectors."""
    matrix = _square_matrix(unitary, "unitary")
    vector = jnp.asarray(state)
    if vector.shape[-1:] != (matrix.shape[-1],):
        raise ValueError("State trailing dimension must match the unitary dimension.")
    return oe.contract("...ij,...j->...i", matrix, vector)


def conjugate_density(unitary: ArrayLike, density: ArrayLike, /) -> Array:
    """Apply ``rho -> U rho U†``."""
    matrix = _square_matrix(unitary, "unitary")
    density_ = _square_matrix(density, "density")
    if density_.shape[-2:] != matrix.shape[-2:]:
        raise ValueError("Density and unitary matrix dimensions must match.")
    adjoint = jnp.swapaxes(jnp.conj(matrix), -1, -2)
    return matrix @ density_ @ adjoint


def unitarity_residual(unitary: ArrayLike, /) -> Array:
    matrix = _square_matrix(unitary, "unitary")
    identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    adjoint = jnp.swapaxes(jnp.conj(matrix), -1, -2)
    return jnp.max(jnp.abs(adjoint @ matrix - identity), axis=(-2, -1))


def density_invariant_residuals(density: ArrayLike, /) -> tuple[Array, Array, Array]:
    matrix = _square_matrix(density, "density")
    adjoint = jnp.swapaxes(jnp.conj(matrix), -1, -2)
    hermitian = jnp.max(jnp.abs(matrix - adjoint), axis=(-2, -1))
    trace = jnp.abs(jnp.trace(matrix, axis1=-2, axis2=-1) - 1.0)
    minimum_eigenvalue = jnp.min(jnp.linalg.eigvalsh(0.5 * (matrix + adjoint)), axis=-1)
    return hermitian, trace, minimum_eigenvalue


__all__ = [
    "apply_unitary_to_state",
    "conjugate_density",
    "density_invariant_residuals",
    "unitarity_residual",
]
