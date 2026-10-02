#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._operators import AbstractLinearOperator
from .._spaces import _coordinate_pairing_matrix, ArraySpace
from ._problems import Eigenproblem, EigenproblemLike


def projector_from_selection(
    vectors: Array,
    inverse_basis: Array,
    selected_mask: Array,
    /,
) -> Array:
    """Construct the basis-invariant spectral projector for a fixed selection."""
    weights = jnp.asarray(selected_mask, dtype=vectors.dtype)
    return (vectors * weights) @ inverse_basis


def density_from_projector(projector: Array, paired_metric: Array, /) -> Array:
    """Return the contravariant kernel D satisfying P = D G."""
    return jnp.swapaxes(
        jnp.linalg.solve(
            jnp.swapaxes(paired_metric, -1, -2),
            jnp.swapaxes(projector, -1, -2),
        ),
        -1,
        -2,
    )


def projector_tangent(
    problem: EigenproblemLike,
    problem_tangent: EigenproblemLike | None,
    eigenvalues: Array,
    eigenvectors: Array,
    inverse_basis: Array,
    selected_mask: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Evaluate the exact isolated-cluster projector tangent."""
    perturbation, paired_metric_tangent = perturbation_in_eigenbasis(
        problem,
        problem_tangent,
        eigenvalues,
        eigenvectors,
    )
    derivative_in_basis = isolated_projector_divided_difference(
        eigenvalues, perturbation, selected_mask
    )
    derivative = eigenvectors @ derivative_in_basis @ inverse_basis
    return derivative, derivative_in_basis, paired_metric_tangent


def density_tangent(
    projector: Array,
    projector_derivative: Array,
    density: Array,
    paired_metric: Array,
    paired_metric_tangent: Array,
    /,
) -> Array:
    """Differentiate D = P G⁻¹ without forming an inverse."""
    del projector
    right_hand_side = projector_derivative - density @ paired_metric_tangent
    return jnp.swapaxes(
        jnp.linalg.solve(
            jnp.swapaxes(paired_metric, -1, -2),
            jnp.swapaxes(right_hand_side, -1, -2),
        ),
        -1,
        -2,
    )


def perturbation_in_eigenbasis(
    problem: EigenproblemLike,
    problem_tangent: EigenproblemLike | None,
    eigenvalues: Array,
    eigenvectors: Array,
    /,
) -> tuple[Array, Array]:
    """Return V⁻¹(dT)V and dG for T = B⁻¹A and G = R B."""

    def operator_and_metric_images(
        current_problem: EigenproblemLike,
    ) -> tuple[Array, Array, Array]:
        operator_images = _operator_coordinate_columns(
            current_problem.operator,
            eigenvectors,
        )
        if isinstance(current_problem, Eigenproblem):
            metric_images = eigenvectors
        else:
            metric_images = _operator_coordinate_columns(
                current_problem.metric_operator,
                eigenvectors,
            )
        return operator_images, metric_images, _paired_metric(current_problem)

    (
        (operator_images, metric_images, paired_metric),
        (operator_tangent, metric_images_tangent, paired_metric_tangent),
    ) = eqx.filter_jvp(
        operator_and_metric_images,
        (problem,),
        (problem_tangent,),
    )
    if operator_tangent is None:
        operator_tangent = jnp.zeros_like(operator_images)
    if metric_images_tangent is None:
        metric_images_tangent = jnp.zeros_like(metric_images)
    if paired_metric_tangent is None:
        paired_metric_tangent = jnp.zeros_like(paired_metric)
    space = problem.operator.source
    pairing = _coordinate_pairing_matrix(space)
    residual_tangent = (
        operator_tangent - metric_images_tangent * eigenvalues[..., None, :]
    )
    perturbation = (
        jnp.conj(jnp.swapaxes(eigenvectors, -1, -2)) @ pairing @ residual_tangent
    )
    return perturbation, paired_metric_tangent


def projector_derivative_residuals(
    eigenvalues: Array,
    eigenvectors: Array,
    inverse_basis: Array,
    selected_mask: Array,
    projector: Array,
    derivative: Array,
    derivative_in_basis: Array,
    perturbation_in_basis: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Return cross-block, commutator, and projector-tangent residuals."""
    selected = jnp.asarray(selected_mask, dtype=eigenvectors.dtype)
    membership_difference = selected[:, None] - selected[None, :]
    eigenvalue_difference = eigenvalues[..., :, None].astype(
        eigenvectors.dtype
    ) - eigenvalues[..., None, :].astype(eigenvectors.dtype)
    cross_residual = jnp.linalg.norm(
        eigenvalue_difference * derivative_in_basis
        - membership_difference * perturbation_in_basis,
        axis=(-2, -1),
    )
    spectral_operator = (
        eigenvectors * eigenvalues.astype(eigenvectors.dtype)[..., None, :]
    ) @ inverse_basis
    perturbation = eigenvectors @ perturbation_in_basis @ inverse_basis
    commutator_residual = jnp.linalg.norm(
        spectral_operator @ derivative
        - derivative @ spectral_operator
        - projector @ perturbation
        + perturbation @ projector,
        axis=(-2, -1),
    )
    tangent_residual = jnp.linalg.norm(
        projector @ derivative + derivative @ projector - derivative,
        axis=(-2, -1),
    )
    return cross_residual, commutator_residual, tangent_residual


@eqx.filter_custom_jvp
def attach_projector_derivative(
    problem: EigenproblemLike,
    projector: Array,
    eigenvalues: Array,
    eigenvectors: Array,
    inverse_basis: Array,
    selected_mask: Array,
    differentiation_valid: Array,
    /,
) -> Array:
    """Attach the mathematical first-order projector derivative to a stopped value."""
    del (
        problem,
        eigenvalues,
        eigenvectors,
        inverse_basis,
        selected_mask,
        differentiation_valid,
    )
    return projector


@attach_projector_derivative.def_jvp
def _attach_projector_derivative_jvp(
    primals: tuple[
        EigenproblemLike,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ],
    tangents: tuple[
        EigenproblemLike | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
    ],
) -> tuple[Array, Array]:
    (
        problem,
        projector,
        eigenvalues,
        eigenvectors,
        inverse_basis,
        selected_mask,
        differentiation_valid,
    ) = primals
    problem_tangent, _, _, _, _, _, _ = tangents
    derivative, _, _ = projector_tangent(
        problem,
        problem_tangent,
        eigenvalues,
        eigenvectors,
        inverse_basis,
        selected_mask,
    )
    derivative = jnp.where(
        differentiation_valid[..., None, None],
        derivative,
        0,
    )
    return projector, derivative


@eqx.filter_custom_jvp
def attach_density_derivative(
    problem: EigenproblemLike,
    density: Array,
    projector: Array,
    paired_metric: Array,
    eigenvalues: Array,
    eigenvectors: Array,
    inverse_basis: Array,
    selected_mask: Array,
    differentiation_valid: Array,
    /,
) -> Array:
    """Attach the mathematical first-order density-kernel derivative."""
    del (
        problem,
        projector,
        paired_metric,
        eigenvalues,
        eigenvectors,
        inverse_basis,
        selected_mask,
        differentiation_valid,
    )
    return density


@attach_density_derivative.def_jvp
def _attach_density_derivative_jvp(
    primals: tuple[
        EigenproblemLike,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ],
    tangents: tuple[
        EigenproblemLike | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
    ],
) -> tuple[Array, Array]:
    (
        problem,
        density,
        projector,
        paired_metric,
        eigenvalues,
        eigenvectors,
        inverse_basis,
        selected_mask,
        differentiation_valid,
    ) = primals
    problem_tangent, _, _, _, _, _, _, _, _ = tangents
    projector_derivative, _, paired_metric_tangent = projector_tangent(
        problem,
        problem_tangent,
        eigenvalues,
        eigenvectors,
        inverse_basis,
        selected_mask,
    )
    derivative = density_tangent(
        projector,
        projector_derivative,
        density,
        paired_metric,
        paired_metric_tangent,
    )
    derivative = jnp.where(
        differentiation_valid[..., None, None],
        derivative,
        0,
    )
    return density, derivative


def _paired_metric(problem: EigenproblemLike, /) -> Array:
    space = problem.operator.source
    pairing = _coordinate_pairing_matrix(space)
    if isinstance(problem, Eigenproblem):
        return jnp.broadcast_to(
            pairing,
            problem.batch_shape + pairing.shape,
        )
    identity = jnp.broadcast_to(
        jnp.eye(space.size, dtype=pairing.dtype),
        problem.batch_shape + (space.size, space.size),
    )
    metric = _operator_coordinate_columns(problem.metric_operator, identity)
    return pairing @ metric


def _operator_coordinate_columns(
    operator: AbstractLinearOperator, block: Array, /
) -> Array:
    space = operator.source
    if operator.batch_shape:
        if not isinstance(space, ArraySpace):
            raise TypeError("Batched spectral actions require an ArraySpace.")
        width = block.shape[-1]
        structured = block.reshape(operator.batch_shape + space.shape + (width,))
        images = operator.mv(structured)
        return jnp.asarray(images).reshape(operator.batch_shape + (space.size, width))

    def apply(column: Array) -> Array:
        return space.flatten(operator.mv(space.unflatten(column)))

    return jax.vmap(apply, in_axes=1, out_axes=1)(block)


def isolated_projector_divided_difference(
    eigenvalues: Array, perturbation: Array, selected_mask: Array, /
) -> Array:
    """Cross-membership divided difference; internal repetitions never divide."""
    selected = selected_mask.astype(perturbation.dtype)
    difference = selected[..., :, None] - selected[..., None, :]
    cross = difference != 0
    gaps = eigenvalues[..., :, None] - eigenvalues[..., None, :]
    return jnp.where(
        cross,
        difference * perturbation / jnp.where(cross, gaps, 1).astype(perturbation.dtype),
        0,
    )


def singular_cross_block_responses(
    forward_block: Array,
    adjoint_block: Array,
    singular_values: Array,
    selected_values: Array,
    cross_mask: Array,
    /,
) -> tuple[Array, Array]:
    """Thin left/right responses using only admitted cross-cluster gaps."""
    ambient_values = singular_values.astype(forward_block.dtype)[:, None]
    retained_values = selected_values.astype(forward_block.dtype)[None, :]
    gaps = selected_values[None, :] ** 2 - singular_values[:, None] ** 2
    denominator = jnp.where(cross_mask, gaps, 1).astype(forward_block.dtype)
    left = (
        forward_block * retained_values + adjoint_block * ambient_values
    ) / denominator
    right = (
        forward_block * ambient_values + adjoint_block * retained_values
    ) / denominator
    return jnp.where(cross_mask, left, 0), jnp.where(cross_mask, right, 0)


def selected_covariance_perturbations(
    perturbation: Array, singular_values: Array, /
) -> tuple[Array, Array]:
    """Return selected left/right covariance cores, including internal repeats."""
    adjoint = jnp.conj(perturbation.T)
    values = singular_values.astype(perturbation.dtype)
    left = perturbation * values[None, :] + values[:, None] * adjoint
    right = values[:, None] * perturbation + adjoint * values[None, :]
    return left, right


def covariance_square_root_tangent(
    frame: Array,
    singular_values: Array,
    horizontal_tangent: Array,
    core_tangent: Array,
    /,
) -> Array:
    """Smooth factor response for positive selected covariance, not an eigengauge."""
    sums = singular_values[:, None] + singular_values[None, :]
    return horizontal_tangent * singular_values.astype(horizontal_tangent.dtype)[
        None, :
    ] + frame @ (core_tangent / sums.astype(core_tangent.dtype))


__all__ = [
    "attach_density_derivative",
    "attach_projector_derivative",
    "density_from_projector",
    "density_tangent",
    "perturbation_in_eigenbasis",
    "projector_derivative_residuals",
    "projector_from_selection",
    "projector_tangent",
    "covariance_square_root_tangent",
    "isolated_projector_divided_difference",
    "selected_covariance_perturbations",
    "singular_cross_block_responses",
]
