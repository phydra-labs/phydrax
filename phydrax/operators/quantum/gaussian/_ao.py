#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pointwise contracted Gaussian AO values and spatial derivatives."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ....ein import contract
from ._basis import PreparedGaussianBasis


def cartesian_ao_values(
    basis: PreparedGaussianBasis,
    nuclear_positions: ArrayLike,
    points: ArrayLike,
    /,
) -> Array:
    positions = jnp.asarray(nuclear_positions)
    points_ = jnp.asarray(points, dtype=positions.dtype)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("nuclear_positions must have shape (N, 3).")
    if points_.ndim != 2 or points_.shape[1] != 3:
        raise ValueError("AO evaluation points must have shape (P, 3).")
    count = basis.cartesian_basis_function_count
    primitive_count = int(basis.exponents.shape[1])
    values = jnp.zeros((points_.shape[0], count), dtype=positions.dtype)
    for function in range(count):
        displacement = points_ - positions[basis.center_indices[function]][None, :]
        angular = basis.angular_tuples[function]
        polynomial = (
            displacement[:, 0] ** angular[0]
            * displacement[:, 1] ** angular[1]
            * displacement[:, 2] ** angular[2]
        )
        radius_squared = jnp.sum(displacement * displacement, axis=1)
        radial = jnp.zeros((points_.shape[0],), dtype=positions.dtype)
        for primitive in range(primitive_count):
            radial = radial + jnp.where(
                basis.primitive_mask[function, primitive],
                basis.normalized_coefficients[function, primitive]
                * jnp.exp(-basis.exponents[function, primitive] * radius_squared),
                0.0,
            )
        values = values.at[:, function].set(polynomial * radial)
    return values


def ao_values(
    basis: PreparedGaussianBasis,
    nuclear_positions: ArrayLike,
    points: ArrayLike,
    /,
) -> Array:
    cartesian = cartesian_ao_values(basis, nuclear_positions, points)
    return contract("pc,ca->pa", cartesian, basis.transformation)


def ao_gradients(
    basis: PreparedGaussianBasis,
    nuclear_positions: ArrayLike,
    points: ArrayLike,
    /,
) -> Array:
    positions = jnp.asarray(nuclear_positions)
    points_ = jnp.asarray(points, dtype=positions.dtype)

    def at_point(point):
        return ao_values(basis, positions, point[None, :])[0]

    return jax.vmap(jax.jacfwd(at_point))(points_)


def ao_hessians(
    basis: PreparedGaussianBasis,
    nuclear_positions: ArrayLike,
    points: ArrayLike,
    /,
) -> Array:
    positions = jnp.asarray(nuclear_positions)
    points_ = jnp.asarray(points, dtype=positions.dtype)

    def at_point(point):
        return ao_values(basis, positions, point[None, :])[0]

    return jax.vmap(jax.jacfwd(jax.jacfwd(at_point)))(points_)


class AOEvaluation(eqx.Module):
    values: Array
    gradients: Array | None
    hessians: Array | None


def evaluate_ao(
    basis: PreparedGaussianBasis,
    nuclear_positions: ArrayLike,
    points: ArrayLike,
    /,
    *,
    derivative_order: int = 0,
) -> AOEvaluation:
    order = int(derivative_order)
    if order not in (0, 1, 2):
        raise ValueError("AO derivative_order must be zero, one, or two.")
    values = ao_values(basis, nuclear_positions, points)
    gradients = None if order == 0 else ao_gradients(basis, nuclear_positions, points)
    hessians = None if order < 2 else ao_hessians(basis, nuclear_positions, points)
    return AOEvaluation(values, gradients, hessians)


__all__ = [
    "AOEvaluation",
    "ao_gradients",
    "ao_hessians",
    "ao_values",
    "cartesian_ao_values",
    "evaluate_ao",
]
