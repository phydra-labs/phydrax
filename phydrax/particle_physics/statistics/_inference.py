#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ...optim import (
    Bounds,
    minimize,
    OptimizationStatus,
    OptimizationTermination,
    ProjectedLBFGS,
)
from ._binned import BinnedStatisticalModel, evaluate_binned_model


class BinnedFitResult(StrictModule, NonTrainableState):
    parameters: Array
    covariance: Array
    negative_log_likelihood: Array
    gradient_norm: Array
    optimizer_status: Array
    covariance_valid: Array
    valid: Array
    model_id: str = eqx.field(static=True)


def fit_binned_model(
    model: BinnedStatisticalModel,
    initial_parameters: ArrayLike | None = None,
    /,
    *,
    maximum_steps: int = 512,
    absolute_optimality: float = 1.0e-8,
) -> BinnedFitResult:
    """Fit a bounded binned model with the native projected L-BFGS route."""
    if not isinstance(model, BinnedStatisticalModel):
        raise TypeError("model must be BinnedStatisticalModel.")
    initial = (
        model.initial_parameters
        if initial_parameters is None
        else jnp.asarray(initial_parameters, dtype=model.nominal_samples.dtype)
    )
    if initial.shape != model.initial_parameters.shape:
        raise ValueError("initial_parameters must align with model parameters.")
    if not bool(evaluate_binned_model(model, initial).valid):
        raise ValueError("Initial statistical parameters do not define a valid model.")

    def objective(parameters, _):
        evaluated = evaluate_binned_model(model, parameters)
        finite_penalty = jnp.asarray(1.0e30, dtype=parameters.dtype)
        return jnp.where(evaluated.valid, -evaluated.total_log_likelihood, finite_penalty)

    optimized = minimize(
        objective,
        initial,
        bounds=Bounds(model.lower_bounds, model.upper_bounds),
        method=ProjectedLBFGS(),
        termination=OptimizationTermination(
            absolute_optimality=float(absolute_optimality),
            relative_optimality=0.0,
            maximum_steps=int(maximum_steps),
        ),
    )
    hessian = jax.hessian(objective)(optimized.parameters, None)
    covariance_result = solve(
        LinearSystem(DenseLinearOperator(hessian)),
        jnp.eye(hessian.shape[0], dtype=hessian.dtype),
        policy=LinearSolvePolicy(DenseLU()),
    )
    gradient = jax.grad(objective)(optimized.parameters, None)
    covariance_valid = jnp.all(covariance_result.status == 0) & jnp.all(
        jnp.isfinite(covariance_result.value)
    )
    valid = (
        (optimized.status == int(OptimizationStatus.SUCCESS))
        & covariance_valid
        & evaluate_binned_model(model, optimized.parameters).valid
    )
    return BinnedFitResult(
        optimized.parameters,
        covariance_result.value,
        optimized.objective,
        jnp.linalg.norm(gradient),
        jnp.asarray(optimized.status, dtype=jnp.int32),
        covariance_valid,
        valid,
        model.model_id,
    )


__all__ = ["BinnedFitResult", "fit_binned_model"]
