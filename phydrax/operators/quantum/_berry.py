#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class QuantumGeometricTensorResult(StrictModule):
    state: Array
    tensor: Array
    metric: Array
    berry_curvature: Array
    berry_connection: Array
    normalization_residual: Array
    valid: Array

    def __init__(
        self,
        *,
        state: ArrayLike,
        tensor: ArrayLike,
        berry_connection: ArrayLike,
        normalization_residual: ArrayLike,
    ):
        self.state = jnp.asarray(state)
        self.tensor = jnp.asarray(tensor)
        self.metric = jnp.real(self.tensor)
        self.berry_curvature = 2.0 * jnp.imag(self.tensor)
        self.berry_connection = jnp.asarray(berry_connection)
        self.normalization_residual = jnp.asarray(normalization_residual)
        self.valid = (
            jnp.all(jnp.isfinite(self.state))
            & jnp.all(jnp.isfinite(self.tensor))
            & jnp.isfinite(self.normalization_residual)
        )


def quantum_geometric_tensor(
    state_function: Callable[[Array], Array],
    parameters: ArrayLike,
    /,
) -> QuantumGeometricTensorResult:
    """Return local Fubini–Study metric and Berry curvature."""
    if not callable(state_function):
        raise TypeError("state_function must be callable.")
    point = jnp.asarray(parameters)
    if point.ndim != 1:
        raise ValueError("Quantum geometric tensor requires vector parameters.")
    state = jnp.asarray(state_function(point))
    if state.ndim != 1 or not jnp.issubdtype(state.dtype, jnp.complexfloating):
        raise TypeError("state_function must return one complex state vector.")
    jacobian = jax.jacfwd(state_function)(point)
    overlap = jnp.einsum("s,si->i", jnp.conj(state), jacobian)
    horizontal = jacobian - state[:, None] * overlap[None, :]
    tensor = jnp.einsum("si,sj->ij", jnp.conj(horizontal), horizontal)
    connection = -jnp.imag(overlap)
    normalization = jnp.abs(jnp.vdot(state, state) - 1.0)
    return QuantumGeometricTensorResult(
        state=state,
        tensor=tensor,
        berry_connection=connection,
        normalization_residual=normalization,
    )


def berry_link(left: ArrayLike, right: ArrayLike, /) -> Array:
    """Return the normalized overlap link between two nonorthogonal rays."""
    left_ = jnp.asarray(left)
    right_ = jnp.asarray(right, dtype=left_.dtype)
    if left_.shape != right_.shape or left_.ndim != 1:
        raise ValueError("Berry-link states must be equal-shaped vectors.")
    overlap = jnp.vdot(left_, right_)
    magnitude = jnp.abs(overlap)
    return jnp.where(magnitude > 0.0, overlap / magnitude, jnp.nan + 0.0j)


def berry_loop_phase(states: ArrayLike, /) -> Array:
    """Return branch-safe phase of the closed product of overlap links."""
    values = jnp.asarray(states)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Berry loop requires at least two state vectors.")
    shifted = jnp.roll(values, -1, axis=0)
    links = jax.vmap(berry_link)(values, shifted)
    return jnp.angle(jnp.prod(links))


__all__ = [
    "QuantumGeometricTensorResult",
    "berry_link",
    "berry_loop_phase",
    "quantum_geometric_tensor",
]
