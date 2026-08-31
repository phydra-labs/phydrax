#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import (
    DenseLinearOperator,
    LinearSystem,
    OperatorProperties,
    prepare,
    solve_many,
)
from ....optim import (
    Bounds,
    MinimizationResult,
    minimize,
    OptimizationTermination,
    ProjectedLBFGS,
)


def _spd_inverse(matrix: Array, /) -> Array:
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "verified",
            },
        ),
    )
    prepared = prepare(LinearSystem(operator))
    return solve_many(prepared, jnp.eye(matrix.shape[0], dtype=matrix.dtype)).value


class StructuralObservationModel(StrictModule, NonTrainableState):
    prediction: Callable = eqx.field(static=True)
    observed: Array
    covariance: Array
    discrepancy_covariance: Array
    precision: Array
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        prediction: Callable,
        observed: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        discrepancy_covariance: ArrayLike | None = None,
        observation_id: str,
    ):
        if not callable(prediction):
            raise TypeError("Structural prediction must be callable.")
        observed_ = jnp.asarray(observed)
        covariance_ = jnp.asarray(covariance, dtype=observed_.dtype)
        discrepancy = (
            jnp.zeros_like(covariance_)
            if discrepancy_covariance is None
            else jnp.asarray(discrepancy_covariance, dtype=observed_.dtype)
        )
        if (
            covariance_.shape != (observed_.size, observed_.size)
            or discrepancy.shape != covariance_.shape
        ):
            raise ValueError("Observation covariance shapes are invalid.")
        total = covariance_ + discrepancy
        if bool(jnp.any(jnp.linalg.eigvalsh(total) <= 0.0)):
            raise ValueError("Total observation covariance must be positive definite.")
        self.prediction = prediction
        self.observed = observed_
        self.covariance = covariance_
        self.discrepancy_covariance = discrepancy
        self.precision = _spd_inverse(total)
        self.observation_id = str(observation_id)

    def residual(self, parameters: ArrayLike, args: Any = None, /) -> Array:
        predicted = jnp.asarray(self.prediction(jnp.asarray(parameters), args))
        if predicted.shape != self.observed.shape:
            raise ValueError("Prediction and observation shapes do not match.")
        return predicted - self.observed


class StructuralCalibrationProblem(StrictModule, NonTrainableState):
    observations: tuple[StructuralObservationModel, ...]
    prior_mean: Array
    prior_precision: Array
    bounds: Bounds | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        observations: tuple[StructuralObservationModel, ...],
        prior_mean: ArrayLike,
        prior_covariance: ArrayLike,
        /,
        *,
        bounds: Bounds | None = None,
        problem_id: str = "structural-calibration",
    ):
        if not observations:
            raise ValueError("Calibration requires at least one observation model.")
        mean = jnp.asarray(prior_mean)
        covariance = jnp.asarray(prior_covariance, dtype=mean.dtype)
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("Prior covariance has the wrong shape.")
        self.observations = observations
        self.prior_mean = mean
        self.prior_precision = _spd_inverse(covariance)
        self.bounds = bounds
        self.problem_id = str(problem_id)

    def negative_log_posterior(self, parameters: Array, args: Any = None, /) -> Array:
        delta = parameters - self.prior_mean
        value = 0.5 * delta @ self.prior_precision @ delta
        for observation in self.observations:
            residual = observation.residual(parameters, args).reshape((-1,))
            value = value + 0.5 * residual @ observation.precision @ residual
        return value


class StructuralCalibrationResult(StrictModule):
    optimization: MinimizationResult
    posterior_hessian: Array
    posterior_covariance: Array
    hessian_eigenvalues: Array
    identifiable: Array
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.optimization.successful & self.identifiable


def calibrate_structural_map(
    problem: StructuralCalibrationProblem,
    initial_parameters: ArrayLike,
    /,
    *,
    args: Any = None,
    termination: OptimizationTermination | None = None,
) -> StructuralCalibrationResult:
    solved = minimize(
        lambda parameters, arguments: problem.negative_log_posterior(
            parameters, arguments
        ),
        jnp.asarray(initial_parameters),
        bounds=problem.bounds,
        method=ProjectedLBFGS(),
        termination=(
            OptimizationTermination(maximum_steps=300)
            if termination is None
            else termination
        ),
        args=args,
    )
    hessian = jax.hessian(problem.negative_log_posterior)(solved.parameters, args)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = jnp.linalg.eigvalsh(hessian)
    identifiable = jnp.min(eigenvalues) > 1.0e-10 * jnp.max(eigenvalues)
    covariance = _spd_inverse(hessian)
    return StructuralCalibrationResult(
        solved,
        hessian,
        covariance,
        eigenvalues,
        identifiable,
        problem.problem_id,
    )


__all__ = [
    "StructuralCalibrationProblem",
    "StructuralCalibrationResult",
    "StructuralObservationModel",
    "calibrate_structural_map",
]
