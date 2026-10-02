#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from .eigen._spectral_derivatives import (
    covariance_square_root_tangent,
    selected_covariance_perturbations,
    singular_cross_block_responses,
)


@final
class SingularSubspaceResponse(StrictModule):
    """Compact, fixed-role tangent lanes; neither array is a primal fit cache."""

    frame_correction: Array
    covariance_correction: Array

    def __init__(self, frame_correction: Array, covariance_correction: Array) -> None:
        if frame_correction.ndim != 2 or covariance_correction.ndim != 2:
            raise ValueError("Singular response arrays must be matrices.")
        rank = frame_correction.shape[1]
        if covariance_correction.shape != (rank, rank):
            raise ValueError(
                "Covariance response must have the selected square core shape."
            )
        self.frame_correction = eqx.error_if(
            frame_correction,
            jnp.any(frame_correction != 0),
            "A singular response frame correction must be identically zero in the primal.",
        )
        self.covariance_correction = eqx.error_if(
            covariance_correction,
            jnp.any(covariance_correction != 0),
            "A singular response covariance correction must be identically zero in the primal.",
        )


def make_subspace_response(
    frame_response: Array, core_response: Array, /
) -> SingularSubspaceResponse:
    """Erase primals while preserving their attached first-order response."""
    return SingularSubspaceResponse(
        frame_response - jax.lax.stop_gradient(frame_response),
        core_response - jax.lax.stop_gradient(core_response),
    )


def row_phase_evidence(rows: Array, /) -> tuple[Array, Array]:
    """Magnitude and uniqueness margin of each canonical row pivot."""
    magnitudes = jnp.abs(rows)
    largest = jnp.max(magnitudes, axis=-1)
    pivots = jnp.argmax(magnitudes, axis=-1)
    indices = jnp.arange(rows.shape[-1])
    indices = indices.reshape((1,) * (rows.ndim - 1) + (rows.shape[-1],))
    runner_up = jnp.max(jnp.where(indices == pivots[..., None], 0, magnitudes), axis=-1)
    return largest, largest - runner_up


def canonicalize_rows(rows: Array, /) -> Array:
    """Choose a nonnegative real largest-magnitude pivot, with first-index ties."""
    indices = jnp.argmax(jnp.abs(rows), axis=-1)
    pivots = jnp.take_along_axis(rows, indices[..., None], axis=-1)[..., 0]
    magnitudes = jnp.abs(pivots)
    denominator = jnp.where(magnitudes > 0, magnitudes, 1).astype(pivots.dtype)
    phases = jnp.where(magnitudes > 0, pivots / denominator, 1)
    return rows * jnp.conj(phases)[..., None]


def canonicalize_singular_triplets(
    left_vectors: Array, singular_values: Array, right_vectors: Array, /
) -> tuple[Array, Array, Array]:
    """Rotate both column frames by the same right-pivot phase."""
    rows = jnp.conj(right_vectors.T)
    indices = jnp.argmax(jnp.abs(rows), axis=-1)
    pivots = jnp.take_along_axis(rows, indices[:, None], axis=-1)[:, 0]
    magnitude = jnp.abs(pivots)
    denominator = jnp.where(magnitude > 0, magnitude, 1).astype(pivots.dtype)
    phases = jnp.where(magnitude > 0, pivots / denominator, 1)
    return (
        left_vectors * phases[None, :],
        singular_values,
        right_vectors * phases[None, :],
    )


def _selected_blocks(
    matrix_tangent: Array,
    left_vectors: Array,
    singular_values: Array,
    right_vectors: Array,
    selected_indices: Array,
    cross_mask: Array,
    /,
) -> tuple[Array, Array, Array]:
    selected_left = left_vectors[:, selected_indices]
    selected_right = right_vectors[:, selected_indices]
    selected_values = singular_values[selected_indices]
    forward_images = matrix_tangent @ selected_right
    adjoint_images = jnp.conj(matrix_tangent.T) @ selected_left
    forward_block = jnp.conj(left_vectors.T) @ forward_images
    adjoint_block = jnp.conj(right_vectors.T) @ adjoint_images
    left_block, right_block = singular_cross_block_responses(
        forward_block, adjoint_block, singular_values, selected_values, cross_mask
    )
    reciprocal = 1 / jnp.where(selected_values > 0, selected_values, 1)
    left_null = forward_images - left_vectors @ forward_block
    right_null = adjoint_images - right_vectors @ adjoint_block
    left_tangent = (
        left_vectors @ left_block
        + left_null * reciprocal.astype(left_null.dtype)[None, :]
    )
    right_tangent = (
        right_vectors @ right_block
        + right_null * reciprocal.astype(right_null.dtype)[None, :]
    )
    return left_tangent, right_tangent, forward_block[selected_indices, :]


@jax.custom_jvp
def selected_singular_responses(
    matrix: Array,
    left_vectors: Array,
    singular_values: Array,
    right_vectors: Array,
    selected_indices: Array,
    differentiation_valid: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Attach horizontal frame and complete covariance-core tangents to a thin SVD.

    The caller owns decomposition and numerical admission. Factors have no AD
    dependence here; only the rectangular matrix drives this mathematical rule.
    Passing a compressed matrix therefore composes the finite range algorithm.
    """
    del matrix, differentiation_valid
    selected_values = singular_values[selected_indices]
    core = jnp.diag(selected_values**2).astype(right_vectors.dtype)
    return (
        left_vectors[:, selected_indices],
        right_vectors[:, selected_indices],
        core,
        core,
    )


@selected_singular_responses.defjvp
def _selected_singular_responses_jvp(
    primals: tuple[Array, Array, Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array, Array, Array],
) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array]]:
    matrix, left, values, right, indices, valid = primals
    matrix_tangent = eqx.error_if(
        tangents[0], ~valid, "Singular projector differentiation is not admitted."
    )
    membership = jnp.any(jnp.arange(values.shape[0])[:, None] == indices[None, :], axis=1)
    cross = ~membership[:, None]
    left_tangent, right_tangent, selected_block = _selected_blocks(
        matrix_tangent, left, values, right, indices, cross
    )
    # A projector onto an entire side is identity, even with selected zero modes.
    if indices.shape[0] == left.shape[0]:
        left_tangent = jnp.zeros_like(left[:, indices])
    if indices.shape[0] == right.shape[0]:
        right_tangent = jnp.zeros_like(right[:, indices])
    left_core, right_core = selected_covariance_perturbations(
        selected_block, values[indices]
    )
    primal = selected_singular_responses(matrix, left, values, right, indices, valid)
    return primal, (left_tangent, right_tangent, left_core, right_core)


@jax.custom_jvp
def singular_value_response(
    matrix: Array,
    left_vectors: Array,
    singular_values: Array,
    right_vectors: Array,
    selected_indices: Array,
    differentiation_valid: Array,
    /,
) -> Array:
    """Individually isolated singular-value response, without a pivot condition."""
    del matrix, left_vectors, right_vectors, differentiation_valid
    return singular_values[selected_indices]


@singular_value_response.defjvp
def _singular_value_response_jvp(
    primals: tuple[Array, Array, Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array, Array, Array],
) -> tuple[Array, Array]:
    matrix, left, values, right, indices, valid = primals
    matrix_tangent = eqx.error_if(
        tangents[0], ~valid, "Singular-value differentiation is not admitted."
    )
    block = jnp.conj(left[:, indices].T) @ matrix_tangent @ right[:, indices]
    return singular_value_response(matrix, left, values, right, indices, valid), jnp.real(
        jnp.diag(block)
    )


@jax.custom_jvp
def attach_selected_triplet_derivative(
    matrix: Array,
    left_vectors: Array,
    singular_values: Array,
    right_vectors: Array,
    selected_indices: Array,
    differentiation_valid: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Canonical retained triplets, with no discarded/discarded denominators."""
    del matrix, differentiation_valid
    left, values, right = canonicalize_singular_triplets(
        left_vectors, singular_values, right_vectors
    )
    return left[:, selected_indices], values[selected_indices], right[:, selected_indices]


@attach_selected_triplet_derivative.defjvp
def _attach_selected_triplet_derivative_jvp(
    primals: tuple[Array, Array, Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array, Array, Array],
) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array]]:
    matrix, left, values, right, indices, valid = primals
    left, values, right = canonicalize_singular_triplets(left, values, right)
    matrix_tangent = eqx.error_if(
        tangents[0], ~valid, "Singular-basis differentiation is not admitted."
    )
    cross = jnp.arange(values.shape[0])[:, None] != indices[None, :]
    left_tangent, right_tangent, block = _selected_blocks(
        matrix_tangent, left, values, right, indices, cross
    )
    selected_left, selected_right = left[:, indices], right[:, indices]
    selected_values = values[indices]
    if jnp.issubdtype(matrix.dtype, jnp.complexfloating):
        left_tangent = (
            left_tangent
            + selected_left * (1j * jnp.imag(jnp.diag(block)) / selected_values)[None, :]
        )
        pivots = jnp.argmax(jnp.abs(selected_right), axis=0)
        pivot_values = selected_right[pivots, jnp.arange(indices.shape[0])]
        pivot_tangent = right_tangent[pivots, jnp.arange(indices.shape[0])]
        connection = -jnp.imag(pivot_tangent / pivot_values)
        left_tangent = left_tangent + selected_left * (1j * connection)[None, :]
        right_tangent = right_tangent + selected_right * (1j * connection)[None, :]
    primal = (selected_left, selected_values, selected_right)
    return primal, (left_tangent, jnp.real(jnp.diag(block)), right_tangent)


@jax.custom_jvp
def _covariance_factor(
    frame: Array,
    singular_values: Array,
    frame_correction: Array,
    core_correction: Array,
    /,
) -> Array:
    del frame_correction, core_correction
    return frame * singular_values.astype(frame.dtype)[None, :]


@_covariance_factor.defjvp
def _covariance_factor_jvp(
    primals: tuple[Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    frame, values, frame_correction, core_correction = primals
    frame_tangent, values_tangent, horizontal, core = tangents
    checked_values = eqx.error_if(
        values,
        jnp.any(values <= 0),
        "Covariance-factor differentiation requires positive retained values.",
    )
    tangent = (
        frame_tangent * values.astype(frame.dtype)[None, :]
        + frame * values_tangent.astype(frame.dtype)[None, :]
    )
    tangent = tangent + covariance_square_root_tangent(
        frame, checked_values, horizontal, core
    )
    return _covariance_factor(frame, values, frame_correction, core_correction), tangent


def covariance_factor(
    frame: Array, singular_values: Array, response: SingularSubspaceResponse, /
) -> Array:
    """Attach the smooth positive covariance factor used by paired pseudo rows."""
    return _covariance_factor(
        frame, singular_values, response.frame_correction, response.covariance_correction
    )


def projector_action(
    frame: Array, response: SingularSubspaceResponse, coordinates: Array, /
) -> Array:
    """Apply the current Euclidean projector and its compact horizontal response."""
    current = frame + response.frame_correction
    return current @ (jnp.conj(current.T) @ coordinates)


def covariance_action(
    frame: Array,
    singular_values: Array,
    response: SingularSubspaceResponse,
    coordinates: Array,
    /,
) -> Array:
    """Apply selected covariance without retaining an ambient square matrix."""
    current = frame + response.frame_correction
    coefficients = jnp.conj(current.T) @ coordinates
    energy = (singular_values**2).astype(coefficients.dtype)
    diagonal = energy if coefficients.ndim == 1 else energy[:, None]
    core_images = diagonal * coefficients + response.covariance_correction @ coefficients
    return current @ core_images


__all__ = [
    "SingularSubspaceResponse",
    "attach_selected_triplet_derivative",
    "canonicalize_rows",
    "canonicalize_singular_triplets",
    "covariance_factor",
    "covariance_action",
    "make_subspace_response",
    "row_phase_evidence",
    "projector_action",
    "selected_singular_responses",
    "singular_value_response",
]
