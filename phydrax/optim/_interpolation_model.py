#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule


def quadratic_basis(step: Any, /) -> Array:
    value = jnp.asarray(step)
    if value.ndim != 1:
        raise ValueError("Quadratic interpolation step must be one vector.")
    products = []
    for row in range(value.size):
        for column in range(row, value.size):
            factor = 0.5 if row == column else 1.0
            products.append(factor * value[row] * value[column])
    return jnp.concatenate(
        [
            jnp.ones((1,), dtype=value.dtype),
            value,
            jnp.stack(products),
        ]
    )


class QuadraticResidualModel(StrictModule):
    """Residual-wise quadratic interpolation model in scaled trust coordinates."""

    coefficients: Array
    center: Array
    radius: Array
    condition_estimate: Array
    residual_dimension: int = eqx.field(static=True)
    interpolation_rank: Array
    parameter_dimension: int = eqx.field(static=True)

    def residual(self, parameters: Any, /) -> Array:
        coordinates = jnp.asarray(parameters)
        scaled = (coordinates - self.center) / self.radius
        return self.coefficients @ quadratic_basis(scaled)

    def jacobian(self, parameters: Any, /) -> Array:
        return jax.jacfwd(self.residual)(jnp.asarray(parameters))

    def objective(self, parameters: Any, /) -> Array:
        residual = self.residual(parameters)
        return 0.5 * jnp.real(jnp.vdot(residual, residual))


class QuadraticScalarModel(StrictModule):
    """Scalar quadratic interpolation model in scaled trust coordinates."""

    coefficients: Array
    center: Array
    radius: Array
    condition_estimate: Array
    parameter_dimension: int = eqx.field(static=True)

    def value(self, parameters: Any, /) -> Array:
        coordinates = jnp.asarray(parameters)
        scaled = (coordinates - self.center) / self.radius
        return jnp.real(jnp.vdot(self.coefficients, quadratic_basis(scaled)))

    def gradient(self, parameters: Any, /) -> Array:
        return jax.grad(self.value)(jnp.asarray(parameters))

    def hessian(self, parameters: Any, /) -> Array:
        return jax.hessian(self.value)(jnp.asarray(parameters))


class InterpolationSet(StrictModule):
    points: Array
    residuals: Array
    center: Array
    radius: Array
    evaluations: Array

    def __init__(
        self,
        points: Any,
        residuals: Any,
        center: Any,
        radius: Any,
        /,
        *,
        evaluations: Any,
    ):
        points_ = jnp.asarray(points)
        residuals_ = jnp.asarray(residuals)
        center_ = jnp.asarray(center)
        radius_ = jnp.asarray(radius)
        if points_.ndim != 2 or residuals_.ndim != 2:
            raise ValueError("Interpolation points and residuals must be matrices.")
        if points_.shape[0] != residuals_.shape[0]:
            raise ValueError("Interpolation point and residual counts must match.")
        if center_.shape != (points_.shape[1],):
            raise ValueError("Interpolation center shape is incompatible.")
        if radius_.shape != ():
            raise ValueError("Interpolation radius must be scalar.")
        self.points = points_
        self.residuals = residuals_
        self.center = center_
        self.radius = radius_
        self.evaluations = jnp.asarray(evaluations, dtype=jnp.int32)


def fit_quadratic_residual_model(
    interpolation: InterpolationSet,
    /,
    *,
    regularization: float = 1e-12,
) -> QuadraticResidualModel:
    if not isinstance(interpolation, InterpolationSet):
        raise TypeError("interpolation must be InterpolationSet.")
    scaled = (interpolation.points - interpolation.center[None, :]) / interpolation.radius
    design = jax.vmap(quadratic_basis)(scaled)
    regularization_ = float(regularization)
    gram = jnp.conj(design.T) @ design + regularization_ * jnp.eye(
        design.shape[1], dtype=design.dtype
    )
    right = jnp.conj(design.T) @ interpolation.residuals
    coefficients = jnp.linalg.solve(gram, right).T
    singular_values = jnp.linalg.svd(design, compute_uv=False)
    condition = singular_values[0] / jnp.maximum(
        singular_values[-1],
        1e-30,
    )
    rank = jnp.linalg.matrix_rank(design)
    return QuadraticResidualModel(
        coefficients=coefficients,
        center=interpolation.center,
        radius=interpolation.radius,
        condition_estimate=condition,
        interpolation_rank=rank,
        residual_dimension=interpolation.residuals.shape[1],
        parameter_dimension=interpolation.points.shape[1],
    )


def fit_quadratic_scalar_model(
    points: Any,
    values: Any,
    center: Any,
    radius: Any,
    /,
    *,
    regularization: float = 1e-12,
) -> QuadraticScalarModel:
    points_ = jnp.asarray(points)
    values_ = jnp.asarray(values)
    center_ = jnp.asarray(center)
    radius_ = jnp.asarray(radius)
    if points_.ndim != 2 or values_.shape != (points_.shape[0],):
        raise ValueError("Scalar interpolation shapes are incompatible.")
    scaled = (points_ - center_[None, :]) / radius_
    design = jax.vmap(quadratic_basis)(scaled)
    if design.shape[0] < design.shape[1]:
        linear_design = design[:, : 1 + points_.shape[1]]
        linear_coefficients = jnp.linalg.lstsq(
            linear_design,
            values_,
            rcond=None,
        )[0]
        coefficients = jnp.concatenate(
            [
                linear_coefficients,
                jnp.zeros(
                    (design.shape[1] - linear_coefficients.size,),
                    dtype=linear_coefficients.dtype,
                ),
            ]
        )
        singular_values = jnp.linalg.svd(
            linear_design,
            compute_uv=False,
        )
    else:
        gram = jnp.conj(design.T) @ design + regularization * jnp.eye(
            design.shape[1],
            dtype=design.dtype,
        )
        coefficients = jnp.linalg.solve(
            gram,
            jnp.conj(design.T) @ values_,
        )
        singular_values = jnp.linalg.svd(design, compute_uv=False)
    condition = singular_values[0] / jnp.maximum(
        singular_values[-1],
        1e-30,
    )
    return QuadraticScalarModel(
        coefficients,
        center_,
        radius_,
        condition,
        parameter_dimension=points_.shape[1],
    )


def coordinate_interpolation_points(center: Any, radius: Any, /) -> Array:
    center_ = jnp.asarray(center)
    radius_ = jnp.asarray(radius)
    identity = jnp.eye(center_.size, dtype=center_.dtype)
    return jnp.concatenate(
        [
            center_[None, :],
            center_[None, :] + radius_ * identity,
            center_[None, :] - radius_ * identity,
        ],
        axis=0,
    )


__all__ = [
    "InterpolationSet",
    "QuadraticResidualModel",
    "coordinate_interpolation_points",
    "fit_quadratic_residual_model",
    "quadratic_basis",
    "QuadraticScalarModel",
    "fit_quadratic_scalar_model",
]
