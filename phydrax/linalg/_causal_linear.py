#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from ._gaussian_chain import (
    associative_gaussian_filter,
    associative_gaussian_smoother,
)


def _validate_causal_arrays(
    transitions: Array,
    vectors: Array,
    /,
    *,
    vector_name: str,
) -> tuple[Array, Array]:
    matrices = jnp.asarray(transitions)
    values = jnp.asarray(vectors)
    if matrices.ndim != 3 or matrices.shape[-2] != matrices.shape[-1]:
        raise ValueError("transitions must have shape (time, state, state).")
    if values.ndim != 2 or values.shape != matrices.shape[:1] + matrices.shape[-1:]:
        raise ValueError(
            f"{vector_name} must have shape (time, state) matching transitions."
        )
    if matrices.shape[0] < 1:
        raise ValueError("A causal linear system requires at least one time step.")
    if not jnp.issubdtype(matrices.dtype, jnp.inexact) or not jnp.issubdtype(
        values.dtype, jnp.inexact
    ):
        raise TypeError("Causal linear systems require inexact arrays.")
    dtype = jnp.result_type(matrices, values)
    return matrices.astype(dtype), values.astype(dtype)


def _compose_affine(
    earlier: tuple[Array, Array],
    later: tuple[Array, Array],
    /,
) -> tuple[Array, Array]:
    earlier_transition, earlier_offset = earlier
    later_transition, later_offset = later
    transition = ein.contract("...ij,...jk->...ik", later_transition, earlier_transition)
    offset = (
        ein.contract("...ij,...j->...i", later_transition, earlier_offset) + later_offset
    )
    return transition, offset


def associative_affine_solve(
    transitions: Array,
    offsets: Array,
    /,
) -> Array:
    """Solve ``x[t] = A[t] x[t - 1] + b[t]`` by associative composition.

    The first transition is applied to an implicit zero predecessor. Callers
    represent a fixed initial boundary by setting ``transitions[0]`` to zero and
    placing the complete first value in ``offsets[0]``.
    """

    matrices, vectors = _validate_causal_arrays(
        transitions,
        offsets,
        vector_name="offsets",
    )
    _, prefixes = jax.lax.associative_scan(
        _compose_affine,
        (matrices, vectors),
        axis=0,
    )
    return prefixes


def associative_transpose_solve(
    transitions: Array,
    right_hand_side: Array,
    /,
) -> Array:
    """Solve the transpose of a unit block-bidiagonal causal system.

    ``transitions[t]`` is the derivative mapping state ``t - 1`` to state
    ``t``. The represented forward operator has identity diagonal blocks and
    lower blocks ``-transitions[t]``.
    """

    matrices, right = _validate_causal_arrays(
        transitions,
        right_hand_side,
        vector_name="right_hand_side",
    )
    zero = jnp.zeros_like(matrices[:1])
    reverse_transitions = jnp.concatenate(
        (zero, jnp.swapaxes(matrices[1:][::-1], -1, -2)),
        axis=0,
    )
    reverse_solution = associative_affine_solve(
        reverse_transitions,
        right[::-1],
    )
    return reverse_solution[::-1]


def causal_linearized_residual(
    transitions: Array,
    residuals: Array,
    step: Array,
    /,
) -> Array:
    """Evaluate ``r + J step`` without assembling the temporal Jacobian."""

    matrices, residual = _validate_causal_arrays(
        transitions,
        residuals,
        vector_name="residuals",
    )
    direction = jnp.asarray(step, dtype=residual.dtype)
    if direction.shape != residual.shape:
        raise ValueError("step must have the same shape as residuals.")
    predecessor = jnp.concatenate((jnp.zeros_like(direction[:1]), direction[:-1]), axis=0)
    propagated = ein.contract("tij,tj->ti", matrices, predecessor)
    return residual + direction - propagated


def _damped_causal_least_squares(
    transitions: Array,
    residuals: Array,
    damping: Array,
    /,
) -> Array:
    num_steps, state_size = residuals.shape
    dtype = residuals.dtype
    identity = jnp.eye(state_size, dtype=dtype)
    process_covariances = (
        jnp.broadcast_to(
            identity,
            (num_steps, state_size, state_size),
        )
        .at[0]
        .set(jnp.zeros_like(identity))
    )
    filter_transitions = transitions.at[0].set(identity)
    filter_offsets = (-residuals).at[0].set(jnp.zeros_like(residuals[0]))
    observation_matrices = jnp.broadcast_to(
        identity,
        (num_steps, state_size, state_size),
    )
    observation_offsets = jnp.zeros((num_steps, state_size), dtype=dtype)
    observation_covariances = jnp.broadcast_to(
        identity / damping,
        (num_steps, state_size, state_size),
    )
    observations = jnp.zeros((num_steps, state_size), dtype=dtype)
    masks = jnp.ones((num_steps, state_size), dtype=bool)
    initial_mean = -residuals[0]
    initial_covariance = identity

    filtered_means, filtered_covariances = associative_gaussian_filter(
        initial_mean[None, :],
        initial_covariance[None, :, :],
        filter_transitions[:, None, :, :],
        filter_offsets[:, None, :],
        process_covariances[:, None, :, :],
        observation_matrices[:, None, :, :],
        observation_offsets[:, None, :],
        observation_covariances[:, None, :, :],
        observations[:, None, :],
        masks[:, None, :],
    )
    filtered_means = filtered_means[:, 0]
    filtered_covariances = filtered_covariances[:, 0]

    predecessor_means = jnp.concatenate(
        (initial_mean[None, :], filtered_means[:-1]), axis=0
    )
    predecessor_covariances = jnp.concatenate(
        (initial_covariance[None, :, :], filtered_covariances[:-1]),
        axis=0,
    )
    predicted_means = (
        ein.contract("tij,tj->ti", filter_transitions, predecessor_means) + filter_offsets
    )
    predicted_covariances = (
        filter_transitions
        @ predecessor_covariances
        @ jnp.swapaxes(filter_transitions, -1, -2)
        + process_covariances
    )
    smoothed_means, _, _ = associative_gaussian_smoother(
        filtered_means[:, None, :],
        filtered_covariances[:, None, :, :],
        predicted_means[:, None, :],
        predicted_covariances[:, None, :, :],
        filter_transitions[:, None, :, :],
        jnp.ones((num_steps, 1), dtype=bool),
    )
    return smoothed_means[:, 0]


def solve_causal_least_squares(
    transitions: Array,
    residuals: Array,
    damping: Array,
    /,
) -> Array:
    """Solve a scalar-damped causal linearized least-squares subproblem."""

    matrices, residual = _validate_causal_arrays(
        transitions,
        residuals,
        vector_name="residuals",
    )
    damping_value = jnp.asarray(damping, dtype=residual.real.dtype)
    if damping_value.shape != ():
        raise ValueError("damping must be scalar.")
    damping_value = eqx.error_if(
        damping_value,
        (~jnp.isfinite(damping_value)) | (damping_value < 0.0),
        "damping must be finite and nonnegative.",
    )
    direct_transitions = matrices.at[0].set(jnp.zeros_like(matrices[0]))
    return jax.lax.cond(
        damping_value > 0.0,
        lambda _: _damped_causal_least_squares(
            direct_transitions,
            residual,
            damping_value,
        ),
        lambda _: associative_affine_solve(direct_transitions, -residual),
        operand=None,
    )


__all__ = [
    "associative_affine_solve",
    "associative_transpose_solve",
    "causal_linearized_residual",
    "solve_causal_least_squares",
]
