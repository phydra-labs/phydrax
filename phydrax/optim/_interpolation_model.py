#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    prepare as prepare_linear,
    solve as solve_linear,
    solve_many as solve_linear_many,
)


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
    precision: NonlinearPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    linear_plan_id: str = eqx.field(static=True)

    def residual(self, parameters: Any, /) -> Array:
        coordinates = jnp.asarray(parameters)
        scaled = (coordinates - self.center) / self.radius
        return self.precision.residual(self.coefficients @ quadratic_basis(scaled))

    def jacobian(self, parameters: Any, /) -> Array:
        return jax.jacfwd(self.residual)(jnp.asarray(parameters))

    def objective(self, parameters: Any, /) -> Array:
        residual = self.precision.accumulation(self.residual(parameters))
        return self.precision.decision(
            0.5 * jnp.real(jnp.sum(jnp.conj(residual) * residual))
        )


class QuadraticScalarModel(StrictModule):
    """Scalar quadratic interpolation model in scaled trust coordinates."""

    coefficients: Array
    center: Array
    radius: Array
    condition_estimate: Array
    parameter_dimension: int = eqx.field(static=True)
    precision: NonlinearPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    linear_plan_id: str = eqx.field(static=True)

    def value(self, parameters: Any, /) -> Array:
        coordinates = jnp.asarray(parameters)
        scaled = (coordinates - self.center) / self.radius
        coefficients = self.precision.accumulation(self.coefficients)
        basis = self.precision.accumulation(quadratic_basis(scaled))
        return self.precision.decision(jnp.real(jnp.sum(jnp.conj(coefficients) * basis)))

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


def _fit_design(
    design: Array,
    values: Array,
    regularization: float,
    precision: NonlinearPrecisionPolicy,
    linear: LinearSolvePolicy | None,
    /,
):
    linear_ = (
        LinearSolvePolicy(DenseSVD(damping=regularization**0.5))
        if linear is None
        else linear
    )
    if not isinstance(linear_, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    design_ = precision.accumulation(design)
    values_ = precision.accumulation(values)
    prepared = prepare_linear(
        LeastSquaresProblem(DenseLinearOperator(design_)),
        precision.bind_linear(linear_),
    )
    return (
        solve_linear_many(prepared, values_)
        if values_.ndim == 2
        else solve_linear(prepared, values_)
    )


def fit_quadratic_residual_model(
    interpolation: InterpolationSet,
    /,
    *,
    regularization: float = 1e-12,
    linear: LinearSolvePolicy | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> QuadraticResidualModel:
    if not isinstance(interpolation, InterpolationSet):
        raise TypeError("interpolation must be InterpolationSet.")
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    points = precision_.state(interpolation.points)
    residuals = precision_.residual(interpolation.residuals)
    precision_.validate_trees(points, residuals)
    scaled = (points - interpolation.center[None, :]) / interpolation.radius
    design = jax.vmap(quadratic_basis)(scaled)
    regularization_ = float(regularization)
    if regularization_ < 0.0 or not jnp.isfinite(regularization_):
        raise ValueError("regularization must be finite and non-negative.")
    result = _fit_design(
        design,
        residuals,
        regularization_,
        precision_,
        linear,
    )
    coefficients = precision_.direction(result.value.T)
    condition = precision_.decision(jnp.max(result.diagnostics.condition_estimate))
    rank = jnp.min(result.diagnostics.rank)
    return QuadraticResidualModel(
        coefficients=coefficients,
        center=interpolation.center,
        radius=interpolation.radius,
        condition_estimate=condition,
        interpolation_rank=rank,
        residual_dimension=interpolation.residuals.shape[1],
        parameter_dimension=interpolation.points.shape[1],
        precision=precision_,
        precision_evidence=precision_.evidence_for(
            points,
            residuals,
            output_value=coefficients,
        ),
        linear_plan_id=result.provenance.plan_id,
    )


def fit_quadratic_scalar_model(
    points: Any,
    values: Any,
    center: Any,
    radius: Any,
    /,
    *,
    regularization: float = 1e-12,
    linear: LinearSolvePolicy | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> QuadraticScalarModel:
    points_ = jnp.asarray(points)
    values_ = jnp.asarray(values)
    center_ = jnp.asarray(center)
    radius_ = jnp.asarray(radius)
    if points_.ndim != 2 or values_.shape != (points_.shape[0],):
        raise ValueError("Scalar interpolation shapes are incompatible.")
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    points_ = precision_.state(points_)
    values_ = precision_.residual(values_)
    precision_.validate_trees(points_, values_)
    scaled = (points_ - center_[None, :]) / radius_
    design = jax.vmap(quadratic_basis)(scaled)
    if design.shape[0] < design.shape[1]:
        fitted_design = design[:, : 1 + points_.shape[1]]
        result = _fit_design(
            fitted_design,
            values_,
            regularization,
            precision_,
            linear,
        )
        fitted = precision_.direction(result.value)
        coefficients = jnp.concatenate(
            [
                fitted,
                jnp.zeros(
                    (design.shape[1] - fitted.size,),
                    dtype=fitted.dtype,
                ),
            ]
        )
    else:
        result = _fit_design(
            design,
            values_,
            regularization,
            precision_,
            linear,
        )
        coefficients = precision_.direction(result.value)
    condition = precision_.decision(jnp.max(result.diagnostics.condition_estimate))
    return QuadraticScalarModel(
        coefficients=coefficients,
        center=center_,
        radius=radius_,
        condition_estimate=condition,
        parameter_dimension=points_.shape[1],
        precision=precision_,
        precision_evidence=precision_.evidence_for(
            points_,
            values_,
            output_value=coefficients,
        ),
        linear_plan_id=result.provenance.plan_id,
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
