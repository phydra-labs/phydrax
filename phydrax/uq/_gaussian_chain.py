#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod
from typing import Literal, Sequence, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    RHSLayout,
    solve,
)


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
    solve_dtype = jnp.result_type(matrix, right)
    matrix = matrix.astype(solve_dtype)
    right = right.astype(solve_dtype)
    result = solve(
        LinearSystem(DenseLinearOperator(matrix)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
        rhs_layout=(RHSLayout((right.shape[-1],)) if right.ndim == matrix.ndim else None),
    )
    return result.value


def _matvec(matrix: Array, vector: Array, /) -> Array:
    return oe.contract("...ij,...j->...i", matrix, vector)


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

    propagated_transition = _solve(left_covariance_right_information, left.transition)
    shifted_offset = _solve(
        left_covariance_right_information,
        left.offset + _matvec(left.covariance, right.information_vector),
    )
    propagated_covariance = _solve(left_covariance_right_information, left.covariance)
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
        jnp.swapaxes(left.transition, -1, -2) @ backward_matrix + left.information_matrix
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
        observation_covariances * active_float[..., :, None] * active_float[..., None, :]
        + observation_identity * (1.0 - active_float[..., :, None])
        + covariance_regularization * observation_identity * active_float[..., :, None]
    )
    predicted_residual = (
        observations
        - observation_offsets
        - oe.contract("...ij,...j->...i", effective_matrix, offsets)
    )
    innovation_covariance = (
        effective_matrix @ process_covariances @ jnp.swapaxes(effective_matrix, -1, -2)
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
    covariance = update_operator @ process_covariances @ jnp.swapaxes(
        update_operator, -1, -2
    ) + gain @ effective_covariance @ jnp.swapaxes(gain, -1, -2)
    transition_observation = effective_matrix @ transitions
    solved_residual = _solve(innovation_covariance, predicted_residual[..., None])[..., 0]
    solved_transition = _solve(innovation_covariance, transition_observation)
    information_vector = oe.contract(
        "...ji,...j->...i", transition_observation, solved_residual
    )
    information_matrix = jnp.swapaxes(transition_observation, -1, -2) @ solved_transition
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
        jnp.concatenate((initial.information_vector, local.information_vector), axis=0),
        jnp.concatenate((initial.information_matrix, local.information_matrix), axis=0),
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
    conditional_offsets = filtered_means[:-1] - oe.contract(
        "...ij,...j->...i", gains, predicted_means[1:]
    )
    conditional_covariances = filtered_covariances[:-1] - (
        gains @ predicted_covariances[1:] @ jnp.swapaxes(gains, -1, -2)
    )
    conditional_covariances = 0.5 * (
        conditional_covariances + jnp.swapaxes(conditional_covariances, -1, -2)
    )
    transition = jnp.where(pair_valid[..., None, None], gains, jnp.zeros_like(gains))
    offset = jnp.where(pair_valid[..., None], conditional_offsets, filtered_means[:-1])
    covariance = jnp.where(
        pair_valid[..., None, None],
        conditional_covariances,
        filtered_covariances[:-1],
    )
    terminal_transition = jnp.zeros(
        (1, case_count, state_size, state_size), filtered_means.dtype
    )
    reverse_transition = jnp.concatenate((terminal_transition, transition[::-1]), axis=0)
    reverse_offset = jnp.concatenate((filtered_means[-1:], offset[::-1]), axis=0)
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


def associative_freeze(
    initial: Array,
    values: Array,
    accepted: Array,
    /,
) -> Array:
    """Retain the most recent accepted value with an associative prefix scan."""
    flags = jnp.concatenate(
        (jnp.ones((1,) + accepted.shape[1:], dtype=bool), accepted), axis=0
    )
    seeded_values = jnp.concatenate((initial[None, ...], values), axis=0)

    def select_latest(left, right):
        left_flag, left_value = left
        right_flag, right_value = right
        selector = right_flag.reshape(
            right_flag.shape + (1,) * (right_value.ndim - right_flag.ndim)
        )
        return left_flag | right_flag, jnp.where(selector, right_value, left_value)

    _, prefixes = jax.lax.associative_scan(select_latest, (flags, seeded_values), axis=0)
    return prefixes[1:]


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


GaussianMarkovStatus: TypeAlias = Literal[0, 1, 2, 3, 4]
GaussianMarkovExecutionMethod: TypeAlias = Literal["sequential", "parallel", "auto"]

GAUSSIAN_MARKOV_SUCCESS: GaussianMarkovStatus = 0
GAUSSIAN_MARKOV_NONFINITE: GaussianMarkovStatus = 1
GAUSSIAN_MARKOV_NON_HERMITIAN: GaussianMarkovStatus = 2
GAUSSIAN_MARKOV_NOT_POSITIVE_DEFINITE: GaussianMarkovStatus = 3
GAUSSIAN_MARKOV_INVALID_NODE_MASK: GaussianMarkovStatus = 4


def gaussian_markov_status_name(value: int, /) -> str:
    """Return the stable name of one Gaussian Markov status code."""
    names = (
        "success",
        "nonfinite",
        "non_hermitian",
        "not_positive_definite",
        "invalid_node_mask",
    )
    code = int(value)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown Gaussian Markov status code {code}.")
    return names[code]


def _identifier(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _rank_tolerance(value: float, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError("rank_tolerance must be finite and nonnegative.")
    return resolved


def _real_dtype(*arrays: Array, owner: str) -> jnp.dtype:
    dtype = jnp.result_type(*arrays)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(dtype, jnp.float32)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise TypeError(f"{owner} supports real arrays only.")
    return dtype


class GaussianInformationElement(StrictModule):
    """Two-endpoint Gaussian information potential for associative elimination."""

    left_precision: Array
    right_precision: Array
    transition_precision: Array
    left_information: Array
    right_information: Array
    log_scale: Array
    valid: Array
    status: Array
    element_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        left_precision: Array,
        right_precision: Array,
        transition_precision: Array,
        left_information: Array,
        right_information: Array,
        log_scale: Array,
        valid: Array,
        status: Array,
        /,
        *,
        element_id: str = "gaussian-information-element",
    ):
        self.left_precision = left_precision
        self.right_precision = right_precision
        self.transition_precision = transition_precision
        self.left_information = left_information
        self.right_information = right_information
        self.log_scale = log_scale
        self.valid = valid
        self.status = status
        self.element_id = _identifier(element_id, owner="element_id")
        self.resolved_method = "information-form-associative-elimination"


class GaussianMarkovInformation(StrictModule):
    """Block-tridiagonal natural coordinates of a real Gaussian Markov law."""

    diagonal_precision: Array
    transition_precision: Array
    information_vector: Array
    node_valid: Array
    information_id: str = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        diagonal_precision: ArrayLike,
        transition_precision: ArrayLike,
        information_vector: ArrayLike,
        /,
        *,
        node_valid: ArrayLike | None = None,
        information_id: str = "gaussian-markov-information",
        rank_tolerance: float = 0.0,
    ):
        diagonal = jnp.asarray(diagonal_precision)
        transition = jnp.asarray(transition_precision)
        vector = jnp.asarray(information_vector)
        if diagonal.ndim < 3 or diagonal.shape[-1] != diagonal.shape[-2]:
            raise ValueError(
                "diagonal_precision must have shape (..., node, state, state)."
            )
        node_count = int(diagonal.shape[-3])
        state_size = int(diagonal.shape[-1])
        if node_count < 1 or state_size < 1:
            raise ValueError(
                "Gaussian Markov node and state dimensions must be positive."
            )
        expected_transition = diagonal.shape[:-3] + (
            max(node_count - 1, 0),
            state_size,
            state_size,
        )
        expected_vector = diagonal.shape[:-3] + (node_count, state_size)
        if transition.shape != expected_transition:
            raise ValueError(
                f"transition_precision must have shape {expected_transition}; "
                f"got {transition.shape}."
            )
        if vector.shape != expected_vector:
            raise ValueError(
                f"information_vector must have shape {expected_vector}; "
                f"got {vector.shape}."
            )
        expected_valid = diagonal.shape[:-2]
        valid = (
            jnp.ones(expected_valid, dtype=bool)
            if node_valid is None
            else jnp.asarray(node_valid, dtype=bool)
        )
        if valid.shape != expected_valid:
            raise ValueError(
                f"node_valid must have shape {expected_valid}; got {valid.shape}."
            )
        dtype = _real_dtype(
            diagonal, transition, vector, owner="GaussianMarkovInformation"
        )
        self.diagonal_precision = diagonal.astype(dtype)
        self.transition_precision = transition.astype(dtype)
        self.information_vector = vector.astype(dtype)
        self.node_valid = valid
        self.information_id = _identifier(information_id, owner="information_id")
        self.rank_tolerance = _rank_tolerance(rank_tolerance)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.diagonal_precision.shape[:-3])

    @property
    def num_nodes(self) -> int:
        return int(self.diagonal_precision.shape[-3])

    @property
    def state_size(self) -> int:
        return int(self.diagonal_precision.shape[-1])


class GaussianMarkovLogNormalizerResult(StrictModule):
    """Log normalizer and untouched information-factorization diagnostics."""

    value: Array
    valid: Array
    status: Array
    information: GaussianMarkovInformation
    execution_method: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


class GaussianMarkovMoments(StrictModule):
    """Mean coordinates and marginal diagnostics of a Gaussian Markov law."""

    means: Array
    second_moments: Array
    transition_second_moments: Array
    node_valid: Array
    log_normalizer: Array
    valid: Array
    status: Array
    moments_id: str = eqx.field(static=True)
    information_id: str = eqx.field(static=True)
    execution_method: str = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        means: Array,
        second_moments: Array,
        transition_second_moments: Array,
        node_valid: Array,
        log_normalizer: Array,
        valid: Array,
        status: Array,
        /,
        *,
        moments_id: str = "gaussian-markov-moments",
        information_id: str = "gaussian-markov-information",
        execution_method: str = "provided",
        rank_tolerance: float = 0.0,
    ):
        means_ = jnp.asarray(means)
        second_ = jnp.asarray(second_moments)
        transition_ = jnp.asarray(transition_second_moments)
        if means_.ndim < 2:
            raise ValueError("means must have shape (..., node, state).")
        node_count = int(means_.shape[-2])
        state_size = int(means_.shape[-1])
        expected_second = means_.shape + (state_size,)
        expected_transition = means_.shape[:-2] + (
            max(node_count - 1, 0),
            state_size,
            state_size,
        )
        if second_.shape != expected_second:
            raise ValueError(
                f"second_moments must have shape {expected_second}; got {second_.shape}."
            )
        if transition_.shape != expected_transition:
            raise ValueError(
                "transition_second_moments must have shape "
                f"{expected_transition}; got {transition_.shape}."
            )
        expected_node_valid = means_.shape[:-1]
        node_valid_ = jnp.asarray(node_valid, dtype=bool)
        if node_valid_.shape != expected_node_valid:
            raise ValueError(
                f"node_valid must have shape {expected_node_valid}; "
                f"got {node_valid_.shape}."
            )
        batch_shape = means_.shape[:-2]
        if jnp.shape(log_normalizer) != batch_shape:
            raise ValueError(
                f"log_normalizer must have shape {batch_shape}; "
                f"got {jnp.shape(log_normalizer)}."
            )
        if jnp.shape(valid) != batch_shape or jnp.shape(status) != batch_shape:
            raise ValueError(
                "valid and status must have the Gaussian Markov batch shape."
            )
        dtype = _real_dtype(means_, second_, transition_, owner="GaussianMarkovMoments")
        self.means = means_.astype(dtype)
        self.second_moments = second_.astype(dtype)
        self.transition_second_moments = transition_.astype(dtype)
        self.node_valid = node_valid_
        self.log_normalizer = jnp.asarray(log_normalizer, dtype=dtype)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.moments_id = _identifier(moments_id, owner="moments_id")
        self.information_id = _identifier(information_id, owner="information_id")
        self.execution_method = _identifier(execution_method, owner="execution_method")
        self.rank_tolerance = _rank_tolerance(rank_tolerance)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.means.shape[:-2])

    @property
    def num_nodes(self) -> int:
        return int(self.means.shape[-2])

    @property
    def state_size(self) -> int:
        return int(self.means.shape[-1])

    @property
    def covariances(self) -> Array:
        outer = self.means[..., :, :, None] * self.means[..., :, None, :]
        covariance = self.second_moments - outer
        covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
        return jnp.where(self.node_valid[..., None, None], covariance, 0.0)

    @property
    def transition_cross_covariances(self) -> Array:
        outer = self.means[..., :-1, :, None] * self.means[..., 1:, None, :]
        cross = self.transition_second_moments - outer
        edge_valid = self.node_valid[..., :-1] & self.node_valid[..., 1:]
        return jnp.where(edge_valid[..., None, None], cross, 0.0)

    @property
    def successful(self) -> Array:
        return self.valid


def _symmetrize(matrix: Array, /) -> Array:
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _matrix_finite(matrix: Array, /) -> Array:
    return jnp.all(jnp.isfinite(matrix), axis=(-2, -1))


def _vector_finite(vector: Array, /) -> Array:
    return jnp.all(jnp.isfinite(vector), axis=-1)


def _matrix_hermitian(matrix: Array, /) -> Array:
    dtype = matrix.dtype
    scale = jnp.maximum(jnp.max(jnp.abs(matrix), axis=(-2, -1)), 1.0)
    tolerance = 64.0 * jnp.finfo(dtype).eps * scale
    error = jnp.max(jnp.abs(matrix - jnp.swapaxes(matrix, -1, -2)), axis=(-2, -1))
    return error <= tolerance


def _canonical_information_arrays(
    diagonal_precision: Array,
    transition_precision: Array,
    information_vector: Array,
    node_valid: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    state_size = diagonal_precision.shape[-1]
    identity = jnp.eye(state_size, dtype=diagonal_precision.dtype)
    node_mask = node_valid[..., :, None, None]
    edge_valid = node_valid[..., :-1] & node_valid[..., 1:]
    diagonal_active = jnp.where(node_mask, diagonal_precision, 0.0)
    vector_active = jnp.where(node_valid[..., :, None], information_vector, 0.0)
    transition_active = jnp.where(
        edge_valid[..., :, None, None], transition_precision, 0.0
    )
    prefix_valid = node_valid[..., 0] & jnp.all(
        ~node_valid[..., 1:] | node_valid[..., :-1], axis=-1
    )
    finite = (
        jnp.all(~node_valid | _matrix_finite(diagonal_active), axis=-1)
        & jnp.all(~node_valid | _vector_finite(vector_active), axis=-1)
        & jnp.all(~edge_valid | _matrix_finite(transition_active), axis=-1)
    )
    hermitian = jnp.all(~node_valid | _matrix_hermitian(diagonal_active), axis=-1)
    status = jnp.where(
        ~prefix_valid,
        GAUSSIAN_MARKOV_INVALID_NODE_MASK,
        jnp.where(
            ~finite,
            GAUSSIAN_MARKOV_NONFINITE,
            jnp.where(
                ~hermitian,
                GAUSSIAN_MARKOV_NON_HERMITIAN,
                GAUSSIAN_MARKOV_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    valid = status == GAUSSIAN_MARKOV_SUCCESS
    diagonal = jnp.where(node_mask, _symmetrize(diagonal_precision), identity)
    vector = jnp.where(node_valid[..., :, None], information_vector, 0.0)
    transition = jnp.where(edge_valid[..., :, None, None], transition_precision, 0.0)
    return diagonal, transition, vector, valid, status


def _cholesky_solve(
    matrix: Array,
    right: Array,
    /,
    *,
    rank_tolerance: float,
) -> tuple[Array, Array, Array, Array]:
    vector_right = right.ndim == matrix.ndim - 1
    right_matrix = right[..., None] if vector_right else right
    factor = jnp.linalg.cholesky(matrix)
    diagonal = jnp.diagonal(factor, axis1=-2, axis2=-1)
    matrix_finite = _matrix_finite(matrix)
    factor_finite = _matrix_finite(factor)
    scale = jnp.maximum(jnp.max(jnp.abs(diagonal), axis=-1), 1.0)
    full_rank = factor_finite & jnp.all(
        diagonal > rank_tolerance * scale[..., None], axis=-1
    )
    valid = matrix_finite & full_rank
    identity = jnp.broadcast_to(
        jnp.eye(matrix.shape[-1], dtype=matrix.dtype), matrix.shape
    )
    safe_factor = jnp.where(valid[..., None, None], factor, identity)
    lower = jsp.linalg.solve_triangular(safe_factor, right_matrix, lower=True)
    solution = jsp.linalg.solve_triangular(
        jnp.swapaxes(safe_factor, -1, -2), lower, lower=False
    )
    log_determinant = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(safe_factor, axis1=-2, axis2=-1)), axis=-1
    )
    status = jnp.where(
        ~matrix_finite,
        GAUSSIAN_MARKOV_NONFINITE,
        jnp.where(
            ~full_rank,
            GAUSSIAN_MARKOV_NOT_POSITIVE_DEFINITE,
            GAUSSIAN_MARKOV_SUCCESS,
        ),
    ).astype(jnp.int32)
    if vector_right:
        solution = solution[..., 0]
    return solution, log_determinant, valid, status


def combine_gaussian_information_elements(
    left: GaussianInformationElement,
    right: GaussianInformationElement,
    /,
    *,
    rank_tolerance: float = 0.0,
) -> GaussianInformationElement:
    """Compose adjacent information potentials by eliminating their shared node."""
    tolerance = rank_tolerance
    shared_precision = _symmetrize(left.right_precision + right.left_precision)
    shared_information = left.right_information + right.left_information
    state_size = shared_precision.shape[-1]
    right_sides = jnp.concatenate(
        (
            jnp.swapaxes(left.transition_precision, -1, -2),
            right.transition_precision,
            shared_information[..., None],
        ),
        axis=-1,
    )
    solved, log_determinant, pivot_valid, pivot_status = _cholesky_solve(
        shared_precision,
        right_sides,
        rank_tolerance=tolerance,
    )
    solved_left = solved[..., :state_size]
    solved_right = solved[..., state_size : 2 * state_size]
    solved_information = solved[..., 2 * state_size]
    left_precision = _symmetrize(
        left.left_precision - left.transition_precision @ solved_left
    )
    right_precision = _symmetrize(
        right.right_precision
        - jnp.swapaxes(right.transition_precision, -1, -2) @ solved_right
    )
    transition_precision = -(left.transition_precision @ solved_right)
    left_information = left.left_information - _matvec(
        left.transition_precision, solved_information
    )
    right_information = right.right_information - _matvec(
        jnp.swapaxes(right.transition_precision, -1, -2), solved_information
    )
    dtype = shared_precision.dtype
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=dtype))
    log_scale = (
        left.log_scale
        + right.log_scale
        + 0.5 * jnp.sum(shared_information * solved_information, axis=-1)
        + 0.5 * state_size * log_two_pi
        - 0.5 * log_determinant
    )
    incoming_valid = left.valid & right.valid
    incoming_status = jnp.where(left.valid, right.status, left.status)
    valid = incoming_valid & pivot_valid
    status = jnp.where(incoming_valid, pivot_status, incoming_status)
    return GaussianInformationElement(
        left_precision,
        right_precision,
        transition_precision,
        left_information,
        right_information,
        log_scale,
        valid,
        status,
        element_id=left.element_id,
    )


def _information_elements(
    diagonal_precision: Array,
    transition_precision: Array,
    information_vector: Array,
    /,
    *,
    information_id: str,
) -> GaussianInformationElement:
    edge_count = transition_precision.shape[-3]
    left_precision = diagonal_precision[..., :-1, :, :]
    right_precision = jnp.zeros_like(left_precision)
    right_precision = right_precision.at[..., edge_count - 1, :, :].set(
        diagonal_precision[..., -1, :, :]
    )
    left_information = information_vector[..., :-1, :]
    right_information = jnp.zeros_like(left_information)
    right_information = right_information.at[..., edge_count - 1, :].set(
        information_vector[..., -1, :]
    )
    batch_edge_shape = left_information.shape[:-1]
    valid = jnp.ones(batch_edge_shape, dtype=bool)
    status = jnp.zeros(batch_edge_shape, dtype=jnp.int32)
    log_scale = jnp.zeros(batch_edge_shape, dtype=diagonal_precision.dtype)
    return GaussianInformationElement(
        jnp.moveaxis(left_precision, -3, 0),
        jnp.moveaxis(right_precision, -3, 0),
        jnp.moveaxis(transition_precision, -3, 0),
        jnp.moveaxis(left_information, -2, 0),
        jnp.moveaxis(right_information, -2, 0),
        jnp.moveaxis(log_scale, -1, 0),
        jnp.moveaxis(valid, -1, 0),
        jnp.moveaxis(status, -1, 0),
        element_id=information_id,
    )


def _take_information_element(
    elements: GaussianInformationElement,
    index: int,
    /,
) -> GaussianInformationElement:
    return GaussianInformationElement(
        elements.left_precision[index],
        elements.right_precision[index],
        elements.transition_precision[index],
        elements.left_information[index],
        elements.right_information[index],
        elements.log_scale[index],
        elements.valid[index],
        elements.status[index],
        element_id=elements.element_id,
    )


def _reduce_information_elements(
    elements: GaussianInformationElement,
    /,
    *,
    method: Literal["sequential", "parallel"],
    rank_tolerance: float,
) -> GaussianInformationElement:
    combine = lambda left, right: combine_gaussian_information_elements(
        left,
        right,
        rank_tolerance=rank_tolerance,
    )
    edge_count = elements.left_precision.shape[0]
    if method == "parallel":
        prefixes = jax.lax.associative_scan(combine, elements, axis=0)
        return _take_information_element(prefixes, edge_count - 1)
    initial = _take_information_element(elements, 0)
    if edge_count == 1:
        return initial

    def scan_step(carry, item):
        combined = combine(carry, item)
        return combined, None

    remaining = GaussianInformationElement(
        elements.left_precision[1:],
        elements.right_precision[1:],
        elements.transition_precision[1:],
        elements.left_information[1:],
        elements.right_information[1:],
        elements.log_scale[1:],
        elements.valid[1:],
        elements.status[1:],
        element_id=elements.element_id,
    )
    final, _ = jax.lax.scan(scan_step, initial, remaining)
    return final


def _integrate_information_element(
    element: GaussianInformationElement,
    /,
    *,
    rank_tolerance: float,
) -> tuple[Array, Array, Array]:
    top = jnp.concatenate((element.left_precision, element.transition_precision), axis=-1)
    bottom = jnp.concatenate(
        (
            jnp.swapaxes(element.transition_precision, -1, -2),
            element.right_precision,
        ),
        axis=-1,
    )
    precision = _symmetrize(jnp.concatenate((top, bottom), axis=-2))
    information = jnp.concatenate(
        (element.left_information, element.right_information), axis=-1
    )
    solved, log_determinant, factor_valid, factor_status = _cholesky_solve(
        precision,
        information,
        rank_tolerance=rank_tolerance,
    )
    event_size = precision.shape[-1]
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=precision.dtype))
    value = (
        element.log_scale
        + 0.5 * jnp.sum(information * solved, axis=-1)
        + 0.5 * event_size * log_two_pi
        - 0.5 * log_determinant
    )
    valid = element.valid & factor_valid
    status = jnp.where(element.valid, factor_status, element.status)
    return value, valid, status


def _resolve_gaussian_markov_method(
    num_nodes: int,
    state_size: int,
    method: GaussianMarkovExecutionMethod,
    /,
) -> Literal["sequential", "parallel"]:
    if method not in ("sequential", "parallel", "auto"):
        raise ValueError("method must be 'sequential', 'parallel', or 'auto'.")
    if method != "auto":
        return method
    return "parallel" if num_nodes >= 64 and state_size <= 32 else "sequential"


def _gaussian_markov_log_normalizer_arrays(
    diagonal_precision: Array,
    transition_precision: Array,
    information_vector: Array,
    node_valid: Array,
    /,
    *,
    method: Literal["sequential", "parallel"],
    rank_tolerance: float,
    information_id: str,
) -> tuple[Array, Array, Array]:
    diagonal, transition, vector, input_valid, input_status = (
        _canonical_information_arrays(
            diagonal_precision,
            transition_precision,
            information_vector,
            node_valid,
        )
    )
    node_count = diagonal.shape[-3]
    state_size = diagonal.shape[-1]
    if node_count == 1:
        solved, log_determinant, factor_valid, factor_status = _cholesky_solve(
            diagonal[..., 0, :, :],
            vector[..., 0, :],
            rank_tolerance=rank_tolerance,
        )
        log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=diagonal.dtype))
        value = (
            0.5 * jnp.sum(vector[..., 0, :] * solved, axis=-1)
            + 0.5 * state_size * log_two_pi
            - 0.5 * log_determinant
        )
        valid = input_valid & factor_valid
        status = jnp.where(input_valid, factor_status, input_status)
    else:
        elements = _information_elements(
            diagonal,
            transition,
            vector,
            information_id=information_id,
        )
        final = _reduce_information_elements(
            elements,
            method=method,
            rank_tolerance=rank_tolerance,
        )
        value, factor_valid, factor_status = _integrate_information_element(
            final,
            rank_tolerance=rank_tolerance,
        )
        valid = input_valid & factor_valid
        status = jnp.where(input_valid, factor_status, input_status)
    inactive_count = jnp.sum(~node_valid, axis=-1, dtype=diagonal.dtype)
    padding_log_normalizer = (
        0.5
        * inactive_count
        * state_size
        * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=diagonal.dtype))
    )
    value = value - padding_log_normalizer
    return jnp.where(valid, value, -jnp.inf), valid, status


def gaussian_markov_log_normalizer(
    information: GaussianMarkovInformation,
    /,
    *,
    method: GaussianMarkovExecutionMethod = "auto",
) -> GaussianMarkovLogNormalizerResult:
    """Evaluate a block-tridiagonal Gaussian log normalizer without densifying."""
    if not isinstance(information, GaussianMarkovInformation):
        raise TypeError("information must be a GaussianMarkovInformation.")
    resolved = _resolve_gaussian_markov_method(
        information.num_nodes, information.state_size, method
    )
    value, valid, status = _gaussian_markov_log_normalizer_arrays(
        information.diagonal_precision,
        information.transition_precision,
        information.information_vector,
        information.node_valid,
        method=resolved,
        rank_tolerance=information.rank_tolerance,
        information_id=information.information_id,
    )
    return GaussianMarkovLogNormalizerResult(
        value,
        valid,
        status,
        information,
        execution_method=resolved,
    )


def gaussian_markov_moments(
    information: GaussianMarkovInformation,
    /,
    *,
    method: GaussianMarkovExecutionMethod = "auto",
    moments_id: str = "gaussian-markov-moments",
) -> GaussianMarkovMoments:
    """Convert Gaussian Markov natural coordinates to sufficient statistics."""
    if not isinstance(information, GaussianMarkovInformation):
        raise TypeError("information must be a GaussianMarkovInformation.")
    resolved = _resolve_gaussian_markov_method(
        information.num_nodes, information.state_size, method
    )
    batch_shape = information.batch_shape
    case_count = prod(batch_shape) if batch_shape else 1
    node_count = information.num_nodes
    state_size = information.state_size
    diagonal = information.diagonal_precision.reshape(
        (case_count, node_count, state_size, state_size)
    )
    transition = information.transition_precision.reshape(
        (case_count, max(node_count - 1, 0), state_size, state_size)
    )
    vector = information.information_vector.reshape((case_count, node_count, state_size))
    node_valid = information.node_valid.reshape((case_count, node_count))

    def objective(diagonal_case, transition_case, vector_case, valid_case):
        value, valid, status = _gaussian_markov_log_normalizer_arrays(
            diagonal_case,
            transition_case,
            vector_case,
            valid_case,
            method=resolved,
            rank_tolerance=information.rank_tolerance,
            information_id=information.information_id,
        )
        return value, (valid, status)

    differentiated = jax.value_and_grad(
        objective,
        argnums=(0, 1, 2),
        has_aux=True,
    )
    (values_and_aux, gradients) = jax.vmap(differentiated)(
        diagonal,
        transition,
        vector,
        node_valid,
    )
    log_normalizer, (valid, status) = values_and_aux
    diagonal_gradient, transition_gradient, vector_gradient = gradients
    means = vector_gradient
    second_moments = _symmetrize(-2.0 * diagonal_gradient)
    transition_second_moments = -transition_gradient
    means = jnp.where(node_valid[..., None], means, 0.0)
    second_moments = jnp.where(node_valid[..., None, None], second_moments, 0.0)
    edge_valid = node_valid[..., :-1] & node_valid[..., 1:]
    transition_second_moments = jnp.where(
        edge_valid[..., None, None], transition_second_moments, 0.0
    )
    case_valid = valid[:, None]
    means = jnp.where(case_valid[..., None], means, jnp.nan)
    second_moments = jnp.where(case_valid[..., None, None], second_moments, jnp.nan)
    transition_second_moments = jnp.where(
        case_valid[..., None, None], transition_second_moments, jnp.nan
    )
    means = jnp.where(node_valid[..., None], means, 0.0)
    second_moments = jnp.where(node_valid[..., None, None], second_moments, 0.0)
    transition_second_moments = jnp.where(
        edge_valid[..., None, None], transition_second_moments, 0.0
    )
    return GaussianMarkovMoments(
        means.reshape(batch_shape + (node_count, state_size)),
        second_moments.reshape(batch_shape + (node_count, state_size, state_size)),
        transition_second_moments.reshape(
            batch_shape + (max(node_count - 1, 0), state_size, state_size)
        ),
        information.node_valid,
        log_normalizer.reshape(batch_shape),
        valid.reshape(batch_shape),
        status.reshape(batch_shape),
        moments_id=moments_id,
        information_id=information.information_id,
        execution_method=resolved,
        rank_tolerance=information.rank_tolerance,
    )


def gaussian_markov_moments_from_marginals(
    means: ArrayLike,
    covariances: ArrayLike,
    transition_cross_covariances: ArrayLike,
    /,
    *,
    node_valid: ArrayLike | None = None,
    moments_id: str = "gaussian-markov-moments",
    information_id: str = "gaussian-markov-information",
    rank_tolerance: float = 0.0,
) -> GaussianMarkovMoments:
    """Construct sufficient statistics from source-target Gaussian marginals."""
    means_ = jnp.asarray(means)
    covariance_ = jnp.asarray(covariances)
    cross_ = jnp.asarray(transition_cross_covariances)
    if means_.ndim < 2:
        raise ValueError("means must have shape (..., node, state).")
    node_count = means_.shape[-2]
    state_size = means_.shape[-1]
    expected_covariance = means_.shape + (state_size,)
    expected_cross = means_.shape[:-2] + (
        max(node_count - 1, 0),
        state_size,
        state_size,
    )
    if covariance_.shape != expected_covariance:
        raise ValueError(
            f"covariances must have shape {expected_covariance}; got {covariance_.shape}."
        )
    if cross_.shape != expected_cross:
        raise ValueError(
            f"transition_cross_covariances must have shape {expected_cross}; "
            f"got {cross_.shape}."
        )
    valid_nodes = (
        jnp.ones(means_.shape[:-1], dtype=bool)
        if node_valid is None
        else jnp.asarray(node_valid, dtype=bool)
    )
    second = covariance_ + means_[..., :, :, None] * means_[..., :, None, :]
    transition_second = cross_ + means_[..., :-1, :, None] * means_[..., 1:, None, :]
    batch_shape = means_.shape[:-2]
    finite = (
        jnp.all(jnp.isfinite(means_), axis=(-2, -1))
        & jnp.all(jnp.isfinite(covariance_), axis=(-3, -2, -1))
        & jnp.all(jnp.isfinite(cross_), axis=(-3, -2, -1))
    )
    status = jnp.where(finite, GAUSSIAN_MARKOV_SUCCESS, GAUSSIAN_MARKOV_NONFINITE).astype(
        jnp.int32
    )
    return GaussianMarkovMoments(
        means_,
        second,
        transition_second,
        valid_nodes,
        jnp.full(batch_shape, jnp.nan, dtype=means_.dtype),
        finite,
        status,
        moments_id=moments_id,
        information_id=information_id,
        execution_method="provided-marginals",
        rank_tolerance=rank_tolerance,
    )


def gaussian_markov_information_from_moments(
    moments: GaussianMarkovMoments,
    /,
    *,
    information_id: str | None = None,
) -> GaussianMarkovInformation:
    """Convert full-rank Markov marginals to block information coordinates."""
    if not isinstance(moments, GaussianMarkovMoments):
        raise TypeError("moments must be GaussianMarkovMoments.")
    means = moments.means
    covariances = moments.covariances
    cross = moments.transition_cross_covariances
    node_valid = moments.node_valid
    state_size = moments.state_size
    node_count = moments.num_nodes
    dtype = means.dtype
    identity = jnp.eye(state_size, dtype=dtype)
    initial_covariance = jnp.where(
        node_valid[..., 0, None, None], covariances[..., 0, :, :], identity
    )
    initial_precision = jnp.linalg.solve(initial_covariance, identity)
    initial_information = _matvec(initial_precision, means[..., 0, :])
    diagonal = jnp.zeros_like(covariances)
    vector = jnp.zeros_like(means)
    diagonal = diagonal.at[..., 0, :, :].set(initial_precision)
    vector = vector.at[..., 0, :].set(initial_information)
    if node_count > 1:
        edge_valid = node_valid[..., :-1] & node_valid[..., 1:]
        source_covariance = jnp.where(
            edge_valid[..., None, None], covariances[..., :-1, :, :], identity
        )
        solved_cross = jnp.linalg.solve(source_covariance, cross)
        transition_matrix = jnp.swapaxes(solved_cross, -1, -2)
        conditional_offset = means[..., 1:, :] - _matvec(
            transition_matrix, means[..., :-1, :]
        )
        conditional_covariance = _symmetrize(
            covariances[..., 1:, :, :] - transition_matrix @ cross
        )
        conditional_covariance = jnp.where(
            edge_valid[..., None, None], conditional_covariance, identity
        )
        conditional_precision = jnp.linalg.solve(conditional_covariance, identity)
        precision_transition = conditional_precision @ transition_matrix
        transition_precision = -jnp.swapaxes(precision_transition, -1, -2)
        source_precision = (
            jnp.swapaxes(transition_matrix, -1, -2)
            @ conditional_precision
            @ transition_matrix
        )
        source_information = -_matvec(
            jnp.swapaxes(transition_matrix, -1, -2),
            _matvec(conditional_precision, conditional_offset),
        )
        target_information = _matvec(conditional_precision, conditional_offset)
        source_precision = jnp.where(edge_valid[..., None, None], source_precision, 0.0)
        conditional_precision = jnp.where(
            edge_valid[..., None, None], conditional_precision, 0.0
        )
        transition_precision = jnp.where(
            edge_valid[..., None, None], transition_precision, 0.0
        )
        source_information = jnp.where(edge_valid[..., None], source_information, 0.0)
        target_information = jnp.where(edge_valid[..., None], target_information, 0.0)
        diagonal = diagonal.at[..., :-1, :, :].add(source_precision)
        diagonal = diagonal.at[..., 1:, :, :].add(conditional_precision)
        vector = vector.at[..., :-1, :].add(source_information)
        vector = vector.at[..., 1:, :].add(target_information)
    else:
        transition_precision = jnp.zeros(
            moments.batch_shape + (0, state_size, state_size), dtype=dtype
        )
    diagonal = jnp.where(node_valid[..., None, None], _symmetrize(diagonal), identity)
    vector = jnp.where(node_valid[..., None], vector, 0.0)
    return GaussianMarkovInformation(
        diagonal,
        transition_precision,
        vector,
        node_valid=node_valid,
        information_id=(
            moments.information_id if information_id is None else information_id
        ),
        rank_tolerance=moments.rank_tolerance,
    )


def sample_gaussian_markov(
    key: Key[Array, ""],
    moments: GaussianMarkovMoments,
    /,
    *,
    sample_shape: Sequence[int] = (),
) -> Array:
    """Draw coherent forward-conditional paths from Gaussian Markov moments."""
    if not isinstance(moments, GaussianMarkovMoments):
        raise TypeError("moments must be GaussianMarkovMoments.")
    samples = tuple(int(size) for size in sample_shape)
    if any(size <= 0 for size in samples):
        raise ValueError("sample_shape dimensions must be positive.")
    sample_count = prod(samples) if samples else 1
    batch_shape = moments.batch_shape
    case_count = prod(batch_shape) if batch_shape else 1
    node_count = moments.num_nodes
    state_size = moments.state_size
    means = moments.means.reshape((case_count, node_count, state_size))
    covariances = moments.covariances.reshape(
        (case_count, node_count, state_size, state_size)
    )
    cross = moments.transition_cross_covariances.reshape(
        (case_count, max(node_count - 1, 0), state_size, state_size)
    )
    node_valid = moments.node_valid.reshape((case_count, node_count))
    member_indices = jnp.arange(sample_count, dtype=jnp.uint32)
    case_indices = jnp.arange(case_count, dtype=jnp.uint32)
    node_indices = jnp.arange(node_count, dtype=jnp.uint32)

    def member_keys(member):
        member_key = jr.fold_in(key, member)

        def case_keys(case):
            case_key = jr.fold_in(member_key, case)
            return jax.vmap(lambda node: jr.fold_in(case_key, node))(node_indices)

        return jax.vmap(case_keys)(case_indices)

    keys = jax.vmap(member_keys)(member_indices)
    normals = jax.vmap(
        lambda member_keys_: jax.vmap(
            lambda case_keys_: jax.vmap(
                lambda draw_key: jr.normal(draw_key, (state_size,), dtype=means.dtype)
            )(case_keys_)
        )(member_keys_)
    )(keys)
    initial_factor = jnp.linalg.cholesky(covariances[:, 0])
    initial = means[None, :, 0, :] + oe.contract(
        "cij,scj->sci", initial_factor, normals[:, :, 0, :]
    )
    initial = jnp.where(node_valid[None, :, 0, None], initial, 0.0)
    if node_count == 1:
        paths = initial[:, :, None, :]
    else:
        edge_valid = node_valid[:, :-1] & node_valid[:, 1:]
        source_covariance = jnp.where(
            edge_valid[..., None, None],
            covariances[:, :-1],
            jnp.eye(state_size, dtype=means.dtype),
        )
        transition = jnp.swapaxes(jnp.linalg.solve(source_covariance, cross), -1, -2)
        offset = means[:, 1:] - _matvec(transition, means[:, :-1])
        conditional_covariance = _symmetrize(covariances[:, 1:] - transition @ cross)
        conditional_covariance = jnp.where(
            edge_valid[..., None, None],
            conditional_covariance,
            jnp.eye(state_size, dtype=means.dtype),
        )
        conditional_factor = jnp.linalg.cholesky(conditional_covariance)

        def sample_step(previous, inputs):
            transition_, offset_, factor_, normal_, active_ = inputs
            value = (
                oe.contract("cij,scj->sci", transition_, previous)
                + offset_[None, ...]
                + oe.contract("cij,scj->sci", factor_, normal_)
            )
            value = jnp.where(active_[None, :, None], value, 0.0)
            return value, value

        _, remaining = jax.lax.scan(
            sample_step,
            initial,
            (
                jnp.moveaxis(transition, 1, 0),
                jnp.moveaxis(offset, 1, 0),
                jnp.moveaxis(conditional_factor, 1, 0),
                jnp.moveaxis(normals[:, :, 1:, :], 2, 0),
                jnp.moveaxis(edge_valid, 1, 0),
            ),
        )
        paths = jnp.concatenate(
            (initial[:, :, None, :], jnp.moveaxis(remaining, 0, 2)), axis=2
        )
    output = paths.reshape(samples + batch_shape + (node_count, state_size))
    if samples:
        return output
    return output.reshape(batch_shape + (node_count, state_size))


__all__ = [
    "associative_freeze",
    "associative_gaussian_filter",
    "associative_gaussian_smoother",
    "combine_gaussian_filter_elements",
    "combine_gaussian_information_elements",
    "GAUSSIAN_MARKOV_INVALID_NODE_MASK",
    "GAUSSIAN_MARKOV_NONFINITE",
    "GAUSSIAN_MARKOV_NON_HERMITIAN",
    "GAUSSIAN_MARKOV_NOT_POSITIVE_DEFINITE",
    "GAUSSIAN_MARKOV_SUCCESS",
    "gaussian_markov_information_from_moments",
    "gaussian_markov_log_normalizer",
    "gaussian_markov_moments",
    "gaussian_markov_moments_from_marginals",
    "gaussian_markov_status_name",
    "GaussianFilterElement",
    "GaussianInformationElement",
    "GaussianMarkovExecutionMethod",
    "GaussianMarkovInformation",
    "GaussianMarkovLogNormalizerResult",
    "GaussianMarkovMoments",
    "GaussianMarkovStatus",
    "sample_gaussian_markov",
]
