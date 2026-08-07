#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule


class GaussianFilterElement(StrictModule):
    """Covariance-form affine Gaussian factor used by temporal prefix scans."""

    transition: Array
    offset: Array
    covariance: Array
    information_vector: Array
    information_matrix: Array
    element_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        transition: Array,
        offset: Array,
        covariance: Array,
        information_vector: Array,
        information_matrix: Array,
        /,
        *,
        element_id: str = "gaussian-filter-element",
    ):
        if not isinstance(element_id, str) or not element_id:
            raise ValueError("element_id must be a non-empty string.")
        self.transition = transition
        self.offset = offset
        self.covariance = covariance
        self.information_vector = information_vector
        self.information_matrix = information_matrix
        self.element_id = element_id
        self.resolved_method = "covariance-form-associative-scan"


def _solve(matrix: Array, right: Array, /) -> Array:
    if right.ndim == matrix.ndim - 1:
        return jnp.linalg.solve(matrix, right[..., None])[..., 0]
    return jnp.linalg.solve(matrix, right)


def _matvec(matrix: Array, vector: Array, /) -> Array:
    return jnp.einsum("...ij,...j->...i", matrix, vector)


def combine_gaussian_filter_elements(
    left: GaussianFilterElement,
    right: GaussianFilterElement,
    /,
) -> GaussianFilterElement:
    """Compose adjacent covariance-form filtering elements associatively."""
    size = left.transition.shape[-1]
    identity = jnp.eye(size, dtype=left.transition.dtype)
    left_covariance_right_information = (
        identity + left.covariance @ right.information_matrix
    )
    right_information_left_covariance = (
        identity + right.information_matrix @ left.covariance
    )

    propagated_transition = _solve(
        left_covariance_right_information, left.transition
    )
    shifted_offset = _solve(
        left_covariance_right_information,
        left.offset + _matvec(left.covariance, right.information_vector),
    )
    propagated_covariance = _solve(
        left_covariance_right_information, left.covariance
    )
    backward_vector = _solve(
        right_information_left_covariance,
        right.information_vector - _matvec(right.information_matrix, left.offset),
    )
    backward_matrix = _solve(
        right_information_left_covariance,
        right.information_matrix @ left.transition,
    )

    transition = right.transition @ propagated_transition
    offset = _matvec(right.transition, shifted_offset) + right.offset
    covariance = (
        right.transition @ propagated_covariance @ jnp.swapaxes(right.transition, -1, -2)
        + right.covariance
    )
    information_vector = (
        _matvec(jnp.swapaxes(left.transition, -1, -2), backward_vector)
        + left.information_vector
    )
    information_matrix = (
        jnp.swapaxes(left.transition, -1, -2) @ backward_matrix
        + left.information_matrix
    )
    return GaussianFilterElement(
        transition,
        offset,
        covariance,
        information_vector,
        information_matrix,
        element_id=left.element_id,
    )


def _observation_conditioned_elements(
    transitions: Array,
    offsets: Array,
    process_covariances: Array,
    observation_matrices: Array,
    observation_offsets: Array,
    observation_covariances: Array,
    observations: Array,
    masks: Array,
    /,
    *,
    covariance_regularization: float,
) -> GaussianFilterElement:
    state_size = transitions.shape[-1]
    observation_size = observation_matrices.shape[-2]
    dtype = transitions.dtype
    active_float = masks.astype(dtype)
    effective_matrix = observation_matrices * active_float[..., :, None]
    observation_identity = jnp.eye(observation_size, dtype=dtype)
    effective_covariance = (
        observation_covariances
        * active_float[..., :, None]
        * active_float[..., None, :]
        + observation_identity * (1.0 - active_float[..., :, None])
        + covariance_regularization
        * observation_identity
        * active_float[..., :, None]
    )
    predicted_residual = observations - observation_offsets - jnp.einsum(
        "...ij,...j->...i", effective_matrix, offsets
    )
    innovation_covariance = (
        effective_matrix
        @ process_covariances
        @ jnp.swapaxes(effective_matrix, -1, -2)
        + effective_covariance
    )
    cross_covariance = process_covariances @ jnp.swapaxes(effective_matrix, -1, -2)
    gain = jnp.swapaxes(
        _solve(innovation_covariance, jnp.swapaxes(cross_covariance, -1, -2)),
        -1,
        -2,
    )
    identity = jnp.eye(state_size, dtype=dtype)
    update_operator = identity - gain @ effective_matrix
    transition = update_operator @ transitions
    offset = _matvec(update_operator, offsets) + _matvec(
        gain, observations - observation_offsets
    )
    covariance = (
        update_operator
        @ process_covariances
        @ jnp.swapaxes(update_operator, -1, -2)
        + gain @ effective_covariance @ jnp.swapaxes(gain, -1, -2)
    )
    transition_observation = effective_matrix @ transitions
    solved_residual = _solve(innovation_covariance, predicted_residual[..., None])[..., 0]
    solved_transition = _solve(innovation_covariance, transition_observation)
    information_vector = jnp.einsum(
        "...ji,...j->...i", transition_observation, solved_residual
    )
    information_matrix = (
        jnp.swapaxes(transition_observation, -1, -2) @ solved_transition
    )
    return GaussianFilterElement(
        transition,
        offset,
        covariance,
        information_vector,
        information_matrix,
    )


def associative_gaussian_filter(
    initial_mean: Array,
    initial_covariance: Array,
    transitions: Array,
    offsets: Array,
    process_covariances: Array,
    observation_matrices: Array,
    observation_offsets: Array,
    observation_covariances: Array,
    observations: Array,
    masks: Array,
    /,
    *,
    covariance_regularization: float = 0.0,
) -> tuple[Array, Array]:
    """Filter a time-major batch through a logarithmic-depth associative scan."""
    local = _observation_conditioned_elements(
        transitions,
        offsets,
        process_covariances,
        observation_matrices,
        observation_offsets,
        observation_covariances,
        observations,
        masks,
        covariance_regularization=covariance_regularization,
    )
    state_size = transitions.shape[-1]
    case_count = transitions.shape[1]
    dtype = transitions.dtype
    initial = GaussianFilterElement(
        jnp.zeros((1, case_count, state_size, state_size), dtype=dtype),
        initial_mean[None, ...],
        initial_covariance[None, ...],
        jnp.zeros((1, case_count, state_size), dtype=dtype),
        jnp.zeros((1, case_count, state_size, state_size), dtype=dtype),
    )
    elements = GaussianFilterElement(
        jnp.concatenate((initial.transition, local.transition), axis=0),
        jnp.concatenate((initial.offset, local.offset), axis=0),
        jnp.concatenate((initial.covariance, local.covariance), axis=0),
        jnp.concatenate(
            (initial.information_vector, local.information_vector), axis=0
        ),
        jnp.concatenate(
            (initial.information_matrix, local.information_matrix), axis=0
        ),
    )
    prefixes = jax.lax.associative_scan(
        combine_gaussian_filter_elements, elements, axis=0
    )
    covariances = prefixes.covariance[1:]
    covariances = 0.5 * (covariances + jnp.swapaxes(covariances, -1, -2))
    return prefixes.offset[1:], covariances


def associative_gaussian_smoother(
    filtered_means: Array,
    filtered_covariances: Array,
    predicted_means: Array,
    predicted_covariances: Array,
    transitions: Array,
    valid: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Apply the RTS affine conditionals with one reversed associative scan."""
    num_steps, case_count, state_size = filtered_means.shape
    if num_steps == 1:
        gains = jnp.zeros((0, case_count, state_size, state_size), filtered_means.dtype)
        return filtered_means, filtered_covariances, gains
    cross = filtered_covariances[:-1] @ jnp.swapaxes(transitions[1:], -1, -2)
    gains = jnp.swapaxes(
        _solve(predicted_covariances[1:], jnp.swapaxes(cross, -1, -2)),
        -1,
        -2,
    )
    pair_valid = valid[:-1] & valid[1:]
    gains = jnp.where(pair_valid[..., None, None], gains, 0.0)
    conditional_offsets = filtered_means[:-1] - jnp.einsum(
        "...ij,...j->...i", gains, predicted_means[1:]
    )
    conditional_covariances = filtered_covariances[:-1] - (
        gains
        @ predicted_covariances[1:]
        @ jnp.swapaxes(gains, -1, -2)
    )
    conditional_covariances = 0.5 * (
        conditional_covariances + jnp.swapaxes(conditional_covariances, -1, -2)
    )
    transition = jnp.where(
        pair_valid[..., None, None], gains, jnp.zeros_like(gains)
    )
    offset = jnp.where(
        pair_valid[..., None], conditional_offsets, filtered_means[:-1]
    )
    covariance = jnp.where(
        pair_valid[..., None, None],
        conditional_covariances,
        filtered_covariances[:-1],
    )
    terminal_transition = jnp.zeros((1, case_count, state_size, state_size), filtered_means.dtype)
    reverse_transition = jnp.concatenate(
        (terminal_transition, transition[::-1]), axis=0
    )
    reverse_offset = jnp.concatenate(
        (filtered_means[-1:], offset[::-1]), axis=0
    )
    reverse_covariance = jnp.concatenate(
        (filtered_covariances[-1:], covariance[::-1]), axis=0
    )
    zero_vector = jnp.zeros_like(reverse_offset)
    zero_matrix = jnp.zeros_like(reverse_transition)
    elements = GaussianFilterElement(
        reverse_transition,
        reverse_offset,
        reverse_covariance,
        zero_vector,
        zero_matrix,
    )
    prefixes = jax.lax.associative_scan(
        combine_gaussian_filter_elements, elements, axis=0
    )
    smoothed_covariances = prefixes.covariance[::-1]
    smoothed_covariances = 0.5 * (
        smoothed_covariances + jnp.swapaxes(smoothed_covariances, -1, -2)
    )
    return prefixes.offset[::-1], smoothed_covariances, gains


def associative_affine_values(
    transitions: Array,
    offsets: Array,
    /,
) -> Array:
    """Evaluate a reversed affine recursion for many keyed path samples."""
    zero_covariance = jnp.zeros_like(transitions)
    zero_vector = jnp.zeros_like(offsets)
    elements = GaussianFilterElement(
        transitions,
        offsets,
        zero_covariance,
        zero_vector,
        zero_covariance,
    )
    prefixes = jax.lax.associative_scan(
        combine_gaussian_filter_elements, elements, axis=0
    )
    return prefixes.offset


__all__ = [
    "associative_gaussian_filter",
    "associative_gaussian_smoother",
    "combine_gaussian_filter_elements",
    "GaussianFilterElement",
]
