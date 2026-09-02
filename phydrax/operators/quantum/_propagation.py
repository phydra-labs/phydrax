#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ._register import HilbertRegisterLayout


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
    return ein.contract("...ij,...j->...i", matrix, vector)


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


def _local_operation_inputs(
    layout: HilbertRegisterLayout,
    matrix: ArrayLike,
    targets: Sequence[str],
    state: ArrayLike,
    /,
) -> tuple[Array, tuple[int, ...], Array]:
    if not isinstance(layout, HilbertRegisterLayout):
        raise TypeError("layout must be a HilbertRegisterLayout.")
    operator = jnp.asarray(matrix)
    if (
        operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or not jnp.issubdtype(operator.dtype, jnp.complexfloating)
    ):
        raise ValueError("Local operator must have exact complex shape (dT, dT).")
    target_indices = layout.target_indices(targets)
    if operator.shape[0] != layout.target_dimension(targets):
        raise ValueError("Local operator dimension does not match its ordered targets.")
    vector = jnp.asarray(state)
    if (
        vector.ndim < 1
        or vector.shape[-1] != layout.dimension
        or not jnp.issubdtype(vector.dtype, jnp.complexfloating)
    ):
        raise ValueError("State must have complex shape (..., layout.dimension).")
    if vector.dtype != operator.dtype:
        raise TypeError("Local operator and state dtypes must match exactly.")
    return operator, target_indices, vector


def apply_local_operator_to_state(
    layout: HilbertRegisterLayout,
    operator: ArrayLike,
    targets: Sequence[str],
    state: ArrayLike,
    /,
) -> Array:
    """Apply one local linear operator without materializing its global embedding."""
    matrix, target_indices, vector = _local_operation_inputs(
        layout, operator, targets, state
    )
    batch_shape = vector.shape[:-1]
    batch_ndim = len(batch_shape)
    remaining_indices = tuple(
        index for index in range(layout.wire_count) if index not in target_indices
    )
    permutation = (
        tuple(range(batch_ndim))
        + tuple(batch_ndim + index for index in remaining_indices)
        + tuple(batch_ndim + index for index in target_indices)
    )
    reshaped = vector.reshape(batch_shape + layout.local_dimensions)
    ordered = jnp.transpose(reshaped, permutation)
    remaining_dimensions = tuple(
        layout.local_dimensions[index] for index in remaining_indices
    )
    target_dimensions = tuple(layout.local_dimensions[index] for index in target_indices)
    remaining_dimension = layout.dimension // layout.target_dimension(targets)
    grouped = ordered.reshape(batch_shape + (remaining_dimension, matrix.shape[-1]))
    transformed = ein.contract("ij,...rj->...ri", matrix, grouped)
    expanded = transformed.reshape(batch_shape + remaining_dimensions + target_dimensions)
    inverse = tuple(permutation.index(index) for index in range(len(permutation)))
    return jnp.transpose(expanded, inverse).reshape(vector.shape)


def apply_local_unitary_to_state(
    layout: HilbertRegisterLayout,
    unitary: ArrayLike,
    targets: Sequence[str],
    state: ArrayLike,
    /,
) -> Array:
    """Apply one local unitary without materializing its global embedding."""
    return apply_local_operator_to_state(layout, unitary, targets, state)


def conjugate_local_density(
    layout: HilbertRegisterLayout,
    unitary: ArrayLike,
    targets: Sequence[str],
    density: ArrayLike,
    /,
) -> Array:
    """Apply ``rho -> U rho U†`` through local ket and bra contractions."""
    matrix = jnp.asarray(unitary)
    value = jnp.asarray(density)
    if (
        value.ndim < 2
        or value.shape[-2:] != (layout.dimension, layout.dimension)
        or not jnp.issubdtype(value.dtype, jnp.complexfloating)
    ):
        raise ValueError(
            "Density must have complex shape (..., layout.dimension, layout.dimension)."
        )
    left = jnp.swapaxes(
        apply_local_unitary_to_state(
            layout, matrix, targets, jnp.swapaxes(value, -1, -2)
        ),
        -1,
        -2,
    )
    return apply_local_unitary_to_state(layout, jnp.conj(matrix), targets, left)


def apply_local_kraus_to_density(
    layout: HilbertRegisterLayout,
    kraus: ArrayLike,
    targets: Sequence[str],
    density: ArrayLike,
    /,
) -> Array:
    """Apply one local Kraus map without a global superoperator."""
    operators = jnp.asarray(kraus)
    if (
        operators.ndim != 3
        or operators.shape[0] < 1
        or operators.shape[1] != operators.shape[2]
        or not jnp.issubdtype(operators.dtype, jnp.complexfloating)
    ):
        raise ValueError("kraus must have exact complex shape (K, dT, dT).")
    value = jnp.asarray(density)
    if value.dtype != operators.dtype:
        raise TypeError("Kraus operators and density dtypes must match exactly.")

    def accumulate(index: int, current: Array) -> Array:
        return current + conjugate_local_density(layout, operators[index], targets, value)

    return jax.lax.fori_loop(0, operators.shape[0], accumulate, jnp.zeros_like(value))


def kraus_trace_preservation_residual(kraus: ArrayLike, /) -> Array:
    """Return ``max(abs(sum(K†K) - I))`` for one local Kraus stack."""
    operators = jnp.asarray(kraus)
    if (
        operators.ndim != 3
        or operators.shape[0] < 1
        or operators.shape[1] != operators.shape[2]
        or not jnp.issubdtype(operators.dtype, jnp.complexfloating)
    ):
        raise ValueError("kraus must have exact complex shape (K, dT, dT).")
    completeness = ein.contract("kai,kaj->ij", jnp.conj(operators), operators)
    identity = jnp.eye(operators.shape[-1], dtype=operators.dtype)
    return jnp.max(jnp.abs(completeness - identity))


__all__ = [
    "apply_local_operator_to_state",
    "apply_local_kraus_to_density",
    "apply_local_unitary_to_state",
    "apply_unitary_to_state",
    "conjugate_density",
    "conjugate_local_density",
    "density_invariant_residuals",
    "kraus_trace_preservation_residual",
    "unitarity_residual",
]
