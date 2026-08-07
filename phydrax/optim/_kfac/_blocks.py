#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._types import (
    AffineBlockSpec,
    AffineFactorObservation,
    BlockCurvatureObservation,
    BlockCurvatureState,
    DenseFactorState,
    KronFactorState,
    ParameterLayout,
    UncoveredBlockSpec,
)


def initialize_block_state(
    layout: ParameterLayout,
    /,
    *,
    num_terms: int,
    dtype,
) -> BlockCurvatureState:
    """Create uninitialized per-term factor state for a static block layout."""

    affine = tuple(
        tuple(
            KronFactorState(
                jnp.zeros((block.input_size, block.input_size), dtype=dtype),
                jnp.zeros((block.output_size, block.output_size), dtype=dtype),
                jnp.asarray(False),
            )
            for _ in range(int(num_terms))
        )
        for block in layout.affine_blocks
    )
    if layout.uncovered_block is None:
        uncovered: tuple[DenseFactorState, ...] = ()
    else:
        size = layout.uncovered_block.parameter_count
        value_shape = (
            (size, size) if layout.uncovered_block.approximation == "exact" else (size,)
        )
        uncovered = tuple(
            DenseFactorState(
                jnp.zeros(value_shape, dtype=dtype),
                jnp.asarray(False),
            )
            for _ in range(int(num_terms))
        )
    return BlockCurvatureState(affine, uncovered)


def estimate_kron_factors_from_chunks(
    jacobian_chunks: Iterable[Array],
    block: AffineBlockSpec,
    /,
    *,
    approximation: Literal["expand", "reduce"],
) -> tuple[Array, Array]:
    """Estimate PSD Kronecker factors without joining residual-Jacobian chunks."""

    if approximation not in ("expand", "reduce"):
        raise ValueError("approximation must be either 'expand' or 'reduce'.")
    activation_sum = None
    sensitivity_sum = None
    target_trace = None
    count = 0
    for jacobian_chunk in jacobian_chunks:
        jacobian = jnp.asarray(jacobian_chunk)
        if activation_sum is None:
            dtype = jacobian.dtype
            activation_sum = jnp.zeros((block.input_size, block.input_size), dtype=dtype)
            sensitivity_sum = jnp.zeros(
                (block.output_size, block.output_size), dtype=dtype
            )
            target_trace = jnp.asarray(0.0, dtype=dtype)
        matrices = jacobian.reshape(
            (int(jacobian.shape[0]), block.output_size, block.input_size)
        )
        left, singular_values, right = jnp.linalg.svd(matrices, full_matrices=False)
        if approximation == "reduce":
            left = left[..., :1]
            singular_values = singular_values[..., :1]
            right = right[..., :1, :]
        root = jnp.sqrt(jnp.maximum(singular_values, 0.0))
        sensitivities = jnp.swapaxes(left * root[:, None, :], -1, -2).reshape(
            (-1, block.output_size)
        )
        activations = (right * root[:, :, None]).reshape((-1, block.input_size))
        activation_sum = activation_sum + activations.T @ activations
        sensitivity_sum = sensitivity_sum + sensitivities.T @ sensitivities
        target_trace = target_trace + jnp.sum(jnp.square(jacobian))
        count += int(activations.shape[0])
    if activation_sum is None or sensitivity_sum is None or target_trace is None:
        raise ValueError("KFAC factor estimation requires at least one Jacobian chunk.")
    activation = activation_sum / float(max(count, 1))
    estimated_trace = jnp.trace(activation) * jnp.trace(sensitivity_sum)
    scale = jnp.where(
        estimated_trace > 0.0,
        target_trace / estimated_trace,
        jnp.asarray(1.0, dtype=dtype),
    )
    sensitivity = scale * sensitivity_sum
    activation = 0.5 * (activation + activation.T)
    sensitivity = 0.5 * (sensitivity + sensitivity.T)
    return activation, sensitivity


def estimate_kron_factors(
    jacobian: Array,
    block: AffineBlockSpec,
    /,
    *,
    approximation: Literal["expand", "reduce"],
) -> tuple[Array, Array]:
    """Estimate PSD Kronecker factors from exact residual Jacobian rows."""

    return estimate_kron_factors_from_chunks(
        (jacobian,),
        block,
        approximation=approximation,
    )


def _ema(old: Array, observed: Array, initialized: Array, decay: float) -> Array:
    updated = float(decay) * old + (1.0 - float(decay)) * observed
    return jnp.where(initialized, updated, observed)


def update_block_state_from_observations(
    state: BlockCurvatureState,
    observations: tuple[BlockCurvatureObservation, ...],
    /,
    *,
    factor_decay: float,
    term_indices: tuple[int, ...] | None = None,
) -> BlockCurvatureState:
    """Update selected per-term factors from block-local observations."""

    if term_indices is None:
        term_indices = tuple(range(len(observations)))
    if len(term_indices) != len(observations):
        raise ValueError("term_indices must align one-to-one with observations.")

    new_affine: list[tuple[KronFactorState, ...]] = []
    for block_index, term_states in enumerate(state.affine):
        updated_terms = list(term_states)
        for term_index, observation in zip(term_indices, observations, strict=True):
            old = term_states[term_index]
            observed = observation.affine[block_index]
            updated_terms[term_index] = KronFactorState(
                _ema(
                    old.activation,
                    observed.activation,
                    old.initialized,
                    factor_decay,
                ),
                _ema(
                    old.sensitivity,
                    observed.sensitivity,
                    old.initialized,
                    factor_decay,
                ),
                jnp.asarray(True),
            )
        new_affine.append(tuple(updated_terms))

    if not state.uncovered:
        new_uncovered: tuple[DenseFactorState, ...] = ()
    else:
        updated_uncovered = list(state.uncovered)
        for term_index, observation in zip(term_indices, observations, strict=True):
            if observation.uncovered is None:
                raise ValueError("KFAC observation is missing uncovered curvature.")
            old = state.uncovered[term_index]
            updated_uncovered[term_index] = DenseFactorState(
                _ema(
                    old.value,
                    observation.uncovered,
                    old.initialized,
                    factor_decay,
                ),
                jnp.asarray(True),
            )
        new_uncovered = tuple(updated_uncovered)
    return BlockCurvatureState(tuple(new_affine), new_uncovered)


def update_block_state(
    state: BlockCurvatureState,
    layout: ParameterLayout,
    term_jacobians: tuple[Array, ...],
    /,
    *,
    approximation: Literal["expand", "reduce"],
    factor_decay: float,
    term_indices: tuple[int, ...] | None = None,
) -> BlockCurvatureState:
    """Update selected term contributions from dense Jacobian test oracles."""

    observations: list[BlockCurvatureObservation] = []
    for jacobian in term_jacobians:
        affine = tuple(
            AffineFactorObservation(
                *estimate_kron_factors(
                    jnp.take(
                        jacobian,
                        jnp.asarray(block.indices, dtype=jnp.int32),
                        axis=1,
                    ),
                    block,
                    approximation=approximation,
                )
            )
            for block in layout.affine_blocks
        )
        uncovered_spec = layout.uncovered_block
        if uncovered_spec is None:
            uncovered = None
        else:
            selected = jnp.take(
                jacobian,
                jnp.asarray(uncovered_spec.indices, dtype=jnp.int32),
                axis=1,
            )
            if uncovered_spec.approximation == "exact":
                uncovered = selected.T @ selected
                uncovered = 0.5 * (uncovered + uncovered.T)
            else:
                uncovered = jnp.sum(jnp.square(selected), axis=0)
        observations.append(BlockCurvatureObservation(affine, uncovered))
    return update_block_state_from_observations(
        state,
        tuple(observations),
        factor_decay=factor_decay,
        term_indices=term_indices,
    )


def kron_matvec(
    factors: tuple[KronFactorState, ...],
    vector: Array,
    block: AffineBlockSpec,
    /,
    *,
    damping: float,
) -> Array:
    """Apply a sum of Kronecker products plus isotropic damping."""

    matrix = jnp.asarray(vector).reshape((block.output_size, block.input_size))
    result = float(damping) * matrix
    for factor in factors:
        result = result + factor.sensitivity @ matrix @ factor.activation.T
    return result.reshape((-1,))


def kron_diagonal(
    factors: tuple[KronFactorState, ...],
    block: AffineBlockSpec,
    /,
    *,
    damping: float,
) -> Array:
    diagonal = jnp.full(
        (block.output_size, block.input_size),
        float(damping),
        dtype=factors[0].activation.dtype,
    )
    for factor in factors:
        diagonal = diagonal + jnp.outer(
            jnp.diag(factor.sensitivity),
            jnp.diag(factor.activation),
        )
    return diagonal.reshape((-1,))


def kron_dense_matrix(
    factors: tuple[KronFactorState, ...],
    /,
    *,
    damping: float = 0.0,
) -> Array:
    """Materialize one small block for tests and numerical diagnostics only."""

    output_size = int(factors[0].sensitivity.shape[0])
    input_size = int(factors[0].activation.shape[0])
    dense = jnp.eye(output_size * input_size, dtype=factors[0].activation.dtype)
    dense = float(damping) * dense
    for factor in factors:
        dense = dense + jnp.kron(factor.sensitivity, factor.activation)
    return dense


def preconditioned_conjugate_gradient(
    matvec,
    rhs: Array,
    preconditioner_diagonal: Array,
    /,
    *,
    max_steps: int,
    relative_tolerance: float,
) -> tuple[Array, Array, Array]:
    """Solve a symmetric positive-definite block with diagonal-preconditioned CG."""

    rhs = jnp.asarray(rhs)
    if int(max_steps) <= 0:
        raise ValueError("cg_max_steps must be positive.")
    if float(relative_tolerance) <= 0.0:
        raise ValueError("cg_relative_tolerance must be positive.")
    diagonal = jnp.asarray(
        eqx.error_if(
            preconditioner_diagonal,
            jnp.any(~jnp.isfinite(preconditioner_diagonal))
            | jnp.any(preconditioner_diagonal <= 0.0),
            "KFAC PCG requires a finite positive preconditioner diagonal.",
        )
    )
    rhs_norm = jnp.linalg.norm(rhs)
    x0 = jnp.zeros_like(rhs)
    residual0 = rhs
    z0 = residual0 / diagonal
    direction0 = z0
    rz0 = jnp.vdot(residual0, z0).real
    threshold = float(relative_tolerance) * rhs_norm

    def cond(carry):
        step, _x, residual, _z, _direction, _rz = carry
        return (step < int(max_steps)) & (jnp.linalg.norm(residual) > threshold)

    def body(carry):
        step, x, residual, z, direction, rz = carry
        product = matvec(direction)
        denominator = jnp.vdot(direction, product).real
        denominator = eqx.error_if(
            denominator,
            (~jnp.isfinite(denominator)) | (denominator <= 0.0),
            "KFAC PCG encountered a nonpositive curvature denominator.",
        )
        alpha = rz / denominator
        new_x = x + alpha * direction
        new_residual = residual - alpha * product
        new_z = new_residual / diagonal
        new_rz = jnp.vdot(new_residual, new_z).real
        beta = jnp.where(rz > 0.0, new_rz / rz, 0.0)
        new_direction = new_z + beta * direction
        return step + 1, new_x, new_residual, new_z, new_direction, new_rz

    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        x0,
        residual0,
        z0,
        direction0,
        rz0,
    )
    step, solution, residual, _z, _direction, _rz = jax.lax.while_loop(
        cond,
        body,
        initial,
    )
    relative_residual = jnp.where(
        rhs_norm > 0.0,
        jnp.linalg.norm(residual) / rhs_norm,
        0.0,
    )
    solution = jnp.where(rhs_norm > 0.0, solution, jnp.zeros_like(solution))
    return solution, step, relative_residual


def _uncovered_direction(
    factors: tuple[DenseFactorState, ...],
    gradient: Array,
    spec: UncoveredBlockSpec,
    /,
    *,
    damping: float,
) -> Array:
    if spec.approximation == "exact":
        curvature = sum(
            (factor.value for factor in factors),
            jnp.zeros_like(factors[0].value),
        )
        curvature = curvature + float(damping) * jnp.eye(
            spec.parameter_count,
            dtype=curvature.dtype,
        )
        return jnp.linalg.solve(curvature, gradient)
    diagonal = sum(
        (factor.value for factor in factors),
        jnp.zeros_like(factors[0].value),
    )
    diagonal = diagonal + float(damping)
    diagonal = eqx.error_if(
        diagonal,
        jnp.any(~jnp.isfinite(diagonal)) | jnp.any(diagonal <= 0.0),
        "KFAC diagonal fallback requires finite positive curvature.",
    )
    return gradient / diagonal


def solve_block_direction(
    state: BlockCurvatureState,
    layout: ParameterLayout,
    gradient: Array,
    /,
    *,
    damping: float,
    cg_max_steps: int,
    cg_relative_tolerance: float,
) -> tuple[Array, Array, Array]:
    """Solve every independent block and scatter into the flat parameter order."""

    gradient = jnp.asarray(gradient)
    direction = jnp.zeros_like(gradient)
    max_iterations = jnp.asarray(0, dtype=jnp.int32)
    max_relative_residual = jnp.asarray(0.0, dtype=gradient.real.dtype)
    for block, factors in zip(layout.affine_blocks, state.affine, strict=True):
        indices = jnp.asarray(block.indices, dtype=jnp.int32)
        block_gradient = jnp.take(gradient, indices)
        block_direction, iterations, relative_residual = (
            preconditioned_conjugate_gradient(
                lambda vector: kron_matvec(
                    factors,
                    vector,
                    block,
                    damping=damping,
                ),
                block_gradient,
                kron_diagonal(factors, block, damping=damping),
                max_steps=cg_max_steps,
                relative_tolerance=cg_relative_tolerance,
            )
        )
        direction = direction.at[indices].set(block_direction)
        max_iterations = jnp.maximum(max_iterations, iterations)
        max_relative_residual = jnp.maximum(
            max_relative_residual,
            relative_residual,
        )

    if layout.uncovered_block is not None:
        indices = jnp.asarray(layout.uncovered_block.indices, dtype=jnp.int32)
        uncovered_gradient = jnp.take(gradient, indices)
        uncovered_direction = _uncovered_direction(
            state.uncovered,
            uncovered_gradient,
            layout.uncovered_block,
            damping=damping,
        )
        direction = direction.at[indices].set(uncovered_direction)
    return direction, max_iterations, max_relative_residual


__all__ = [
    "AffineFactorObservation",
    "BlockCurvatureObservation",
    "BlockCurvatureState",
    "DenseFactorState",
    "KronFactorState",
    "estimate_kron_factors",
    "estimate_kron_factors_from_chunks",
    "initialize_block_state",
    "kron_dense_matrix",
    "kron_diagonal",
    "kron_matvec",
    "preconditioned_conjugate_gradient",
    "solve_block_direction",
    "update_block_state",
    "update_block_state_from_observations",
]
