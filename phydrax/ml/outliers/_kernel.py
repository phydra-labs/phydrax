#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel, ModelBinding
from ...kernels import AbstractPositiveDefiniteKernel, SquaredExponentialKernel
from .._batch import MLBatch
from .._contracts import AbstractRecipe, FitResult, GradientContract, ML_NONCONVERGED
from ._common import (
    _BLOCKWISE_BINDING,
    _case_count,
    _fit_arrays,
    _fit_status,
    _prepare_queries,
    _restore_scores,
    _score_bounds,
    OutlierDiagnostics,
)


def _project_capped_simplex(values: Array, caps: Array) -> Array:
    """Euclidean projection onto sum(alpha)=1 with per-coordinate upper bounds."""
    lower = jnp.min(values - caps)
    upper = jnp.max(values)

    def search(_iteration, bounds):
        low, high = bounds
        multiplier = 0.5 * (low + high)
        mass = jnp.sum(jnp.clip(values - multiplier, 0.0, caps))
        low = jnp.where(mass > 1.0, multiplier, low)
        high = jnp.where(mass > 1.0, high, multiplier)
        return low, high

    lower, upper = jax.lax.fori_loop(0, 64, search, (lower, upper))
    return jnp.clip(values - 0.5 * (lower + upper), 0.0, caps)


def _fit_ocsvm_one(
    gram: Array,
    weights: Array,
    active: Array,
    nu: float,
    iterations: int,
    learning_rate: float,
) -> tuple[Array, Array, Array, Array, Array]:
    total_weight = jnp.sum(weights)
    caps = weights / jnp.maximum(float(nu) * total_weight, jnp.finfo(float).tiny)
    alpha = _project_capped_simplex(
        weights / jnp.maximum(total_weight, jnp.finfo(float).tiny), caps
    )
    lipschitz_bound = jnp.maximum(
        jnp.max(jnp.sum(jnp.abs(gram), axis=-1)), jnp.asarray(1.0, dtype=gram.dtype)
    )
    step_size = float(learning_rate) / lipschitz_bound

    def step(_iteration, current):
        gradient = gram @ current
        return _project_capped_simplex(current - step_size * gradient, caps)

    alpha = jax.lax.fori_loop(0, iterations, step, alpha)
    decision = gram @ alpha
    epsilon = 10.0 * jnp.finfo(alpha.dtype).eps
    free = active & (alpha > epsilon) & (alpha < caps - epsilon)
    free_mass = jnp.sum(free)
    rho = jnp.where(
        free_mass > 0,
        jnp.sum(jnp.where(free, decision, 0.0)) / jnp.maximum(free_mass, 1),
        jnp.sum(alpha * decision),
    )
    projected = _project_capped_simplex(alpha - step_size * decision, caps)
    residual = jnp.linalg.norm(projected - alpha)
    objective = 0.5 * jnp.real(alpha @ (gram @ alpha))
    return alpha, rho, decision, residual, objective


class OneClassSVMModel(AbstractArrayModel):
    """Native kernel one-class SVM with smooth novelty scores and hard prediction."""

    training_features: Array
    dual_coefficients: Array
    rho: Array
    active: Array
    kernel: AbstractPositiveDefiniteKernel
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        training_features: ArrayLike,
        dual_coefficients: ArrayLike,
        rho: ArrayLike,
        active: ArrayLike,
        kernel: AbstractPositiveDefiniteKernel,
        *,
        case_shape: tuple[int, ...],
    ):
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be a native AbstractPositiveDefiniteKernel.")
        train = jnp.asarray(training_features)
        self.training_features = train
        self.dual_coefficients = jnp.asarray(dual_coefficients)
        self.rho = jnp.asarray(rho)
        self.active = jnp.asarray(active, dtype=bool)
        self.kernel = kernel
        self.case_shape = tuple(case_shape)
        self.in_size = int(train.shape[-1])
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        if jnp.issubdtype(queries.dtype, jnp.complexfloating):
            raise TypeError(
                "Native positive-definite kernels currently require real features."
            )
        cases = _case_count(self.case_shape)
        train = self.training_features.reshape(
            (cases,) + self.training_features.shape[-2:]
        )
        coefficients = self.dual_coefficients.reshape(
            (cases, self.dual_coefficients.shape[-1])
        )
        rho = self.rho.reshape((cases,))

        def score_one(query, train_, coefficients_, rho_):
            cross_gram = self.kernel.matrix(query, train_)
            return rho_ - cross_gram @ coefficients_

        scores = jax.vmap(score_one)(queries, train, coefficients, rho)
        return _restore_scores(
            scores, case_shape=self.case_shape, query_shape=query_shape
        )

    def predict(self, x: Any, /) -> Array:
        """Return hard anomaly indicators where the signed novelty score is positive."""
        return jax.lax.stop_gradient(self(x) > 0.0)

    def smooth_membership(self, x: Any, /, *, temperature: ArrayLike = 1.0) -> Array:
        return jax.nn.sigmoid(
            self(x) / jnp.maximum(jnp.asarray(temperature), jnp.finfo(float).tiny)
        )


class OneClassSVMRecipe(AbstractRecipe):
    """Fixed-iteration projected native-kernel one-class SVM dual fit."""

    kernel: AbstractPositiveDefiniteKernel
    nu: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractPositiveDefiniteKernel | None = None,
        *,
        nu: float = 0.1,
        iterations: int = 250,
        learning_rate: float = 0.1,
        tolerance: float = 1e-5,
    ):
        kernel_ = SquaredExponentialKernel() if kernel is None else kernel
        if not isinstance(kernel_, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be a native AbstractPositiveDefiniteKernel.")
        if not 0.0 < float(nu) <= 1.0:
            raise ValueError("nu must lie in (0, 1].")
        if int(iterations) <= 0 or float(learning_rate) <= 0.0 or float(tolerance) <= 0.0:
            raise ValueError("iterations, learning_rate, and tolerance must be positive.")
        self.kernel = kernel_
        self.nu = float(nu)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.tolerance = float(tolerance)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, active = _fit_arrays(batch)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError(
                "Native positive-definite kernels currently require real features."
            )
        cases = _case_count(batch.case_shape)
        flat_x = x.reshape((cases, batch.sample_count, batch.feature_count))
        gram = jax.vmap(lambda x_: self.kernel.matrix(x_, x_))(flat_x)
        outputs = jax.vmap(
            lambda gram_, weights_, active_: _fit_ocsvm_one(
                gram_,
                weights_,
                active_,
                self.nu,
                self.iterations,
                self.learning_rate,
            )
        )(
            gram,
            weights.reshape((cases, batch.sample_count)),
            active.reshape((cases, batch.sample_count)),
        )
        coefficients, rho, decision, residual, objective = outputs
        coefficients = coefficients.reshape(batch.case_shape + (batch.sample_count,))
        rho = rho.reshape(batch.case_shape)
        scores = (rho.reshape((cases, 1)) - decision).reshape(
            batch.case_shape + (batch.sample_count,)
        )
        residual = residual.reshape(batch.case_shape)
        objective = objective.reshape(batch.case_shape)
        minimum, maximum = _score_bounds(scores, active)
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(coefficients), axis=-1) & jnp.isfinite(rho)
        enough = effective >= 2
        converged = residual <= self.tolerance
        valid = finite & enough & converged
        status = _fit_status(finite, enough)
        status = jnp.where(finite & enough & ~converged, ML_NONCONVERGED, status).astype(
            jnp.int32
        )
        diagnostics = OutlierDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.iterations,
            effective_samples=effective,
            threshold=jnp.zeros_like(rho),
            score_minimum=minimum,
            score_maximum=maximum,
            rank=-1,
            condition=jnp.nan,
            converged=converged,
            method="one-class-svm-native-kernel",
        )
        model = OneClassSVMModel(
            x,
            coefficients,
            rho,
            active,
            self.kernel,
            case_shape=batch.case_shape,
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("support_partition", "predict", "valid", "status"),
            conditions=(
                "kernel is differentiable at evaluated inputs",
                "capped-simplex active set and support partition are held fixed",
                "fixed projected-gradient iteration count",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="one-class-svm-native-kernel",
            gradient_contract=contract,
        )


__all__ = ["OneClassSVMModel", "OneClassSVMRecipe"]
