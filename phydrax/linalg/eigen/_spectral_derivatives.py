#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._spaces import _coordinate_pairing_matrix
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
    problem_tangent: Any,
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
    selected = jnp.asarray(selected_mask, dtype=eigenvectors.dtype)
    membership_difference = selected[:, None] - selected[None, :]
    eigenvalue_difference = eigenvalues[..., :, None].astype(
        eigenvectors.dtype
    ) - eigenvalues[..., None, :].astype(eigenvectors.dtype)
    cross_block = membership_difference != 0
    safe_difference = jnp.where(cross_block, eigenvalue_difference, 1)
    derivative_in_basis = jnp.where(
        cross_block,
        membership_difference * perturbation / safe_difference,
        0,
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
    problem_tangent: Any,
    eigenvalues: Array,
    eigenvectors: Array,
    /,
) -> tuple[Array, Array]:
    """Return V⁻¹(dT)V and dG for T = B⁻¹A and G = R B."""

    def operator_and_metric_images(current_problem):
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
    /,
) -> Array:
    """Attach the mathematical first-order projector derivative to a stopped value."""
    del problem, eigenvalues, eigenvectors, inverse_basis, selected_mask
    return projector


@attach_projector_derivative.def_jvp
def _attach_projector_derivative_jvp(primals, tangents):
    problem, projector, eigenvalues, eigenvectors, inverse_basis, selected_mask = primals
    problem_tangent, _, _, _, _, _ = tangents
    derivative, _, _ = projector_tangent(
        problem,
        problem_tangent,
        eigenvalues,
        eigenvectors,
        inverse_basis,
        selected_mask,
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
    )
    return density


@attach_density_derivative.def_jvp
def _attach_density_derivative_jvp(primals, tangents):
    (
        problem,
        density,
        projector,
        paired_metric,
        eigenvalues,
        eigenvectors,
        inverse_basis,
        selected_mask,
    ) = primals
    problem_tangent, _, _, _, _, _, _, _ = tangents
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


def _operator_coordinate_columns(operator: Any, block: Array, /) -> Array:
    space = operator.source
    if operator.batch_shape:
        width = block.shape[-1]
        structured = block.reshape(operator.batch_shape + space.shape + (width,))
        images = operator.mv(structured)
        return jnp.asarray(images).reshape(operator.batch_shape + (space.size, width))

    def apply(column):
        return space.flatten(operator.mv(space.unflatten(column)))

    return jax.vmap(apply, in_axes=1, out_axes=1)(block)


__all__ = [
    "attach_density_derivative",
    "attach_projector_derivative",
    "density_from_projector",
    "density_tangent",
    "perturbation_in_eigenbasis",
    "projector_derivative_residuals",
    "projector_from_selection",
    "projector_tangent",
]
