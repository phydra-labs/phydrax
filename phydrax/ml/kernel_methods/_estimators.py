#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel
from ..._model._binding import ModelBinding
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import solve_weighted_least_squares
from .._soft_discrete import temperature_sigmoid
from .._sparse_features import SparseFeatures
from ._utils import (
    case_kernel_matrix,
    finite_array,
    flatten_targets,
    query_kernel_matrix,
    validate_kernel,
    validated_weights,
)


def _size(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _apply_coefficients(
    matrix: Array, coefficients: Array, intercept: Array, case_shape: tuple[int, ...]
) -> Array:
    if not case_shape:
        return matrix @ coefficients + intercept
    cases = _size(case_shape)
    query_shape = matrix.shape[len(case_shape) : -1]
    q = _size(tuple(int(s) for s in query_shape)) if query_shape else 1
    m = matrix.reshape((cases, q, matrix.shape[-1]))
    c = coefficients.reshape((cases, coefficients.shape[-2], coefficients.shape[-1]))
    b = intercept.reshape((cases, intercept.shape[-1]))
    out = jax.vmap(lambda a, d, e: a @ d + e)(m, c, b)
    return out.reshape(case_shape + query_shape + (coefficients.shape[-1],))


class AbstractKernelLinearModel(AbstractArrayModel):
    """Smooth kernel expansion shared by ridge, LS-SVM, SVC and SVR."""

    support: Array
    coefficients: Array
    intercept: Array
    support_mask: Array
    kernel: Any
    feature_count: int = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Any = eqx.field(static=True)

    def __init__(
        self,
        support: Array,
        coefficients: Array,
        intercept: Array,
        support_mask: Array,
        kernel: Any,
        feature_count: int,
        output_shape: tuple[int, ...],
        case_shape: tuple[int, ...],
        method: str,
    ):
        self.support = support
        self.coefficients = coefficients
        self.intercept = intercept
        self.support_mask = support_mask
        self.kernel = kernel
        self.feature_count = int(feature_count)
        self.output_shape = tuple(int(size) for size in output_shape)
        self.case_shape = tuple(int(size) for size in case_shape)
        self.method = str(method)
        self.in_size = self.feature_count
        self.out_size = "scalar" if not self.output_shape else self.output_shape

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        matrix, query_shape = query_kernel_matrix(
            self.kernel, jnp.asarray(x), self.support, self.case_shape
        )
        support_mask = self.support_mask.reshape(
            self.case_shape + (1,) * len(query_shape) + (self.support.shape[-2],)
        )
        matrix = matrix * support_mask
        result = _apply_coefficients(
            matrix, self.coefficients, self.intercept, self.case_shape
        )
        if not self.output_shape:
            return result[..., 0]
        return result.reshape(result.shape[:-1] + self.output_shape)


class KernelRidgeModel(AbstractKernelLinearModel):
    """Fitted kernel-ridge expansion."""


class LeastSquaresSVMModel(AbstractKernelLinearModel):
    """Least-squares SVM decision function with explicit hard/smooth views."""

    def decision_function(self, x: ArrayLike, /) -> Array:
        return self(x)

    def predict(self, x: ArrayLike, /) -> Array:
        return jnp.where(self(x) >= 0.0, 1, -1).astype(jnp.int32)

    def probabilities(self, x: ArrayLike, /, *, temperature: ArrayLike = 1.0) -> Array:
        positive = temperature_sigmoid(self(x), temperature=temperature)
        return jnp.stack((1.0 - positive, positive), axis=-1)


class SupportVectorClassifierModel(AbstractKernelLinearModel):
    """Smooth SVC decision function; ``predict`` is the separate hard output."""

    def decision_function(self, x: ArrayLike, /) -> Array:
        return self(x)

    def predict(self, x: ArrayLike, /) -> Array:
        return jnp.where(self(x) >= 0.0, 1, -1).astype(jnp.int32)

    def probabilities(self, x: ArrayLike, /, *, temperature: ArrayLike = 1.0) -> Array:
        p = temperature_sigmoid(self(x), temperature=temperature)
        return jnp.stack((1.0 - p, p), axis=-1)


class SupportVectorRegressorModel(AbstractKernelLinearModel):
    """Kernel epsilon-insensitive regression expansion."""


class OneClassSVMModel(AbstractKernelLinearModel):
    """Smooth one-class score; ``predict`` exposes the hard inlier decision."""

    def score_samples(self, x: ArrayLike, /) -> Array:
        return self(x)

    def predict(self, x: ArrayLike, /) -> Array:
        return jnp.where(self(x) >= 0.0, 1, -1).astype(jnp.int32)

    def inlier_probability(
        self, x: ArrayLike, /, *, temperature: ArrayLike = 1.0
    ) -> Array:
        return temperature_sigmoid(self(x), temperature=temperature)


class KernelRidgeRecipe(AbstractRecipe):
    kernel: Any
    alpha: Array
    fit_intercept: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: Any,
        /,
        *,
        alpha: ArrayLike = 1.0,
        fit_intercept: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.kernel = validate_kernel(kernel)
        alpha_ = jnp.asarray(alpha, dtype=float)
        if alpha_.ndim != 0:
            raise ValueError("alpha must be scalar.")
        self.alpha = eqx.error_if(
            alpha_,
            (~jnp.isfinite(alpha_)) | (alpha_ < 0.0),
            "alpha must be finite and nonnegative.",
        )
        self.fit_intercept = bool(fit_intercept)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        target = batch.require_targets()
        x = batch.dense_features()
        feature_valid = jnp.all(finite_array(x), axis=-1)
        x = jnp.where(finite_array(x), x, 0.0)
        sample_shape = batch.case_shape + (batch.sample_count,)
        y, output_shape = flatten_targets(target, sample_shape)
        weights = validated_weights(batch.effective_weight(self.weight_policy))
        if batch.target_mask is not None:
            weights = weights * jnp.all(
                batch.target_mask.reshape(sample_shape + (-1,)), axis=-1
            )
        weights = weights * feature_valid * jnp.all(finite_array(y), axis=-1)
        gram = case_kernel_matrix(self.kernel, x, x, batch.case_shape)
        gram = gram * (weights > 0)[..., None, :]
        solved = solve_weighted_least_squares(
            gram, y, weights, ridge=self.alpha, fit_intercept=self.fit_intercept
        )
        model = KernelRidgeModel(
            support=x,
            coefficients=solved.coefficients.reshape(
                batch.case_shape + (batch.sample_count, -1)
            ),
            intercept=solved.intercept.reshape(batch.case_shape + (-1,)),
            support_mask=weights > 0.0,
            kernel=self.kernel,
            feature_count=batch.feature_count,
            output_shape=output_shape,
            case_shape=batch.case_shape,
            method="kernel-ridge",
        )
        effective = jnp.sum(weights > 0.0, axis=-1)
        valid = solved.valid & (effective > 0)
        status = jnp.where(effective > 0, solved.status, ML_INSUFFICIENT_DATA)
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.sum(solved.residual_sum_squares, axis=-1)
            if solved.residual_sum_squares.ndim > len(batch.case_shape)
            else solved.residual_sum_squares,
            effective_samples=effective,
            rank=solved.rank,
            condition=solved.condition,
            method="kernel-ridge-augmented-svd",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="kernel-ridge",
            gradient_contract=GradientContract.direct(
                conditions=("Fixed support capacity and rank branch.",)
            ),
        )


class LeastSquaresSVMRecipe(AbstractRecipe):
    """Binary LS-SVM fitted as a regularized kernel least-squares system."""

    kernel: Any
    alpha: Array
    fit_intercept: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: Any,
        /,
        *,
        alpha: ArrayLike = 1.0,
        fit_intercept: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        configured = KernelRidgeRecipe(
            kernel,
            alpha=alpha,
            fit_intercept=fit_intercept,
            weight_policy=weight_policy,
        )
        self.kernel = configured.kernel
        self.alpha = configured.alpha
        self.fit_intercept = configured.fit_intercept
        self.weight_policy = configured.weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        target = batch.require_targets()
        if target.shape != batch.case_shape + (batch.sample_count,):
            raise ValueError(
                "LeastSquaresSVMRecipe requires one scalar label per sample."
            )
        labels = jnp.where(target > 0, 1.0, -1.0)
        binary = MLBatch(
            batch.features,
            labels,
            feature_mask=None
            if isinstance(batch.features, SparseFeatures)
            else batch.feature_mask,
            target_mask=batch.target_mask,
            sample_mask=batch.sample_mask,
            sample_weight=batch.sample_weight,
            measure_weight=batch.measure_weight,
            feature_schema=batch.feature_schema,
            target_schema=batch.target_schema,
        )
        result = KernelRidgeRecipe(
            self.kernel,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            weight_policy=self.weight_policy,
        ).fit_batch(binary, key=key)
        raw = result.as_trainable()
        if jnp.iscomplexobj(raw.coefficients) or jnp.iscomplexobj(raw.intercept):
            raise TypeError("Least-squares SVM requires a real-valued kernel expansion.")
        model = LeastSquaresSVMModel(
            support=raw.support,
            coefficients=raw.coefficients,
            intercept=raw.intercept,
            support_mask=raw.support_mask,
            kernel=raw.kernel,
            feature_count=raw.feature_count,
            output_shape=raw.output_shape,
            case_shape=raw.case_shape,
            method="least-squares-svm",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="direct",
            nondifferentiable_outputs=("predict",),
            conditions=("Binary labels are fixed discrete data.",),
        )
        return FitResult(
            model,
            result.diagnostics,
            valid=result.valid,
            status=result.status,
            method="least-squares-svm",
            gradient_contract=contract,
        )


def _binary_training(batch: MLBatch, policy: WeightPolicy) -> tuple[Array, Array, Array]:
    target = batch.require_targets()
    if target.shape != batch.case_shape + (batch.sample_count,):
        raise ValueError("Binary kernel machines require one scalar target per sample.")
    x = batch.dense_features()
    feature_valid = jnp.all(finite_array(x), axis=-1)
    x = jnp.where(finite_array(x), x, 0.0)
    target_valid = jnp.isfinite(target)
    labels = jnp.where(target > 0, 1.0, -1.0)
    weights = validated_weights(batch.effective_weight(policy))
    if batch.target_mask is not None:
        weights = weights * batch.target_mask
    weights = weights * feature_valid * target_valid
    return x, labels, weights


def _case_optimizer_inputs(batch: MLBatch, x: Array, *values: Array) -> tuple[Array, ...]:
    cases = _size(batch.case_shape)
    return (
        x.reshape((cases, batch.sample_count, batch.feature_count)),
        *(v.reshape((cases,) + v.shape[len(batch.case_shape) :]) for v in values),
    )


class SupportVectorClassifierRecipe(AbstractRecipe):
    kernel: Any
    c: Array
    iterations: int = eqx.field(static=True)
    learning_rate: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: Any,
        /,
        *,
        c: ArrayLike = 1.0,
        iterations: int = 200,
        learning_rate: ArrayLike = 0.05,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.kernel = validate_kernel(kernel)
        self.c = jnp.asarray(c, dtype=float)
        self.learning_rate = jnp.asarray(learning_rate, dtype=float)
        if self.c.ndim != 0 or self.learning_rate.ndim != 0 or iterations <= 0:
            raise ValueError(
                "c and learning_rate must be scalar and iterations positive."
            )
        self.c = eqx.error_if(
            self.c,
            (~jnp.isfinite(self.c)) | (self.c <= 0),
            "c must be positive and finite.",
        )
        self.learning_rate = eqx.error_if(
            self.learning_rate,
            (~jnp.isfinite(self.learning_rate)) | (self.learning_rate <= 0),
            "learning_rate must be positive and finite.",
        )
        self.iterations = int(iterations)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, labels, weights = _binary_training(batch, self.weight_policy)
        gram = case_kernel_matrix(self.kernel, x, x, batch.case_shape)
        if jnp.iscomplexobj(gram):
            raise TypeError("SVC requires a real-valued kernel.")
        cases = _size(batch.case_shape)
        k = gram.reshape((cases, batch.sample_count, batch.sample_count))
        y = labels.reshape((cases, batch.sample_count))
        w = weights.reshape((cases, batch.sample_count))
        c = self.c
        lr = self.learning_rate

        def solve_one(ki, yi, wi):
            cap = c * wi
            q = (yi[:, None] * ki) * yi[None, :]

            def step(_, alpha):
                proposal = jnp.clip(alpha + lr * (wi - q @ alpha), 0.0, cap)
                denom = jnp.sum(wi) + jnp.finfo(wi.dtype).tiny
                proposal = jnp.clip(
                    proposal - yi * (jnp.sum(proposal * yi) / denom) * wi, 0.0, cap
                )
                return proposal

            alpha = jax.lax.fori_loop(0, self.iterations, step, jnp.zeros_like(wi))
            coefficient = alpha * yi
            active = (alpha > 1e-6) & (alpha < cap - 1e-6) & (wi > 0)
            residual = yi - ki @ coefficient
            intercept = jnp.sum(jnp.where(active, residual, 0.0)) / jnp.maximum(
                jnp.sum(active), 1
            )
            objective = 0.5 * alpha @ (q @ alpha) - jnp.sum(wi * alpha)
            return coefficient, intercept, objective

        coef, intercept, objective = jax.vmap(solve_one)(k, y, w)
        coef = coef.reshape(batch.case_shape + (batch.sample_count, 1))
        intercept = intercept.reshape(batch.case_shape + (1,))
        effective = jnp.sum(weights > 0, axis=-1)
        finite = jnp.all(finite_array(coef), axis=(-2, -1)) & jnp.isfinite(
            intercept[..., 0]
        )
        valid = finite & (effective >= 2)
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(effective >= 2, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        )
        model = SupportVectorClassifierModel(
            support=x,
            coefficients=coef,
            intercept=intercept,
            support_mask=weights > 0,
            kernel=self.kernel,
            feature_count=batch.feature_count,
            output_shape=(),
            case_shape=batch.case_shape,
            method="svc",
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            objective=objective.reshape(batch.case_shape),
            iterations=self.iterations,
            effective_samples=effective,
            method="projected-dual-svc",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("predict",),
            conditions=("Fixed projected-optimization path and binary labels.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="svc",
            gradient_contract=contract,
        )


class SupportVectorRegressorRecipe(AbstractRecipe):
    kernel: Any
    c: Array
    epsilon: Array
    iterations: int = eqx.field(static=True)
    learning_rate: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: Any,
        /,
        *,
        c: ArrayLike = 1.0,
        epsilon: ArrayLike = 0.1,
        iterations: int = 200,
        learning_rate: ArrayLike = 0.02,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.kernel = validate_kernel(kernel)
        self.c = jnp.asarray(c, dtype=float)
        self.epsilon = jnp.asarray(epsilon, dtype=float)
        self.learning_rate = jnp.asarray(learning_rate, dtype=float)
        if (
            any(v.ndim != 0 for v in (self.c, self.epsilon, self.learning_rate))
            or iterations <= 0
        ):
            raise ValueError(
                "SVR hyperparameters must be scalar and iterations positive."
            )
        if bool(
            jnp.any(jnp.asarray([self.c <= 0, self.epsilon < 0, self.learning_rate <= 0]))
        ):
            raise ValueError("c and learning_rate must be positive; epsilon nonnegative.")
        self.iterations = int(iterations)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        target = batch.require_targets()
        if target.shape != batch.case_shape + (batch.sample_count,):
            raise ValueError("SupportVectorRegressorRecipe requires scalar targets.")
        x = batch.dense_features()
        feature_valid = jnp.all(finite_array(x), axis=-1)
        x = jnp.where(finite_array(x), x, 0.0)
        target_valid = finite_array(target)
        target = jnp.where(target_valid, target, 0.0)
        weights = validated_weights(batch.effective_weight(self.weight_policy))
        if batch.target_mask is not None:
            weights = weights * batch.target_mask
        weights = weights * feature_valid * target_valid
        gram = case_kernel_matrix(self.kernel, x, x, batch.case_shape)
        active_pair = (weights > 0)[..., :, None] & (weights > 0)[..., None, :]
        gram = jnp.where(active_pair, gram, 0.0)
        if jnp.iscomplexobj(target) or jnp.iscomplexobj(gram):
            raise TypeError(
                "Epsilon-insensitive SVR requires real targets and kernel values."
            )
        cases = _size(batch.case_shape)
        k = gram.reshape((cases, batch.sample_count, batch.sample_count))
        y = target.reshape((cases, batch.sample_count))
        w = weights.reshape((cases, batch.sample_count))

        def solve_one(ki, yi, wi):
            denom = jnp.sum(wi) + jnp.finfo(wi.dtype).tiny
            mean = jnp.sum(wi * yi) / denom
            beta0 = jnp.zeros_like(yi)

            def step(_, state):
                beta, intercept = state
                residual = ki @ beta + intercept - yi
                loss_grad = (
                    jnp.where(jnp.abs(residual) > self.epsilon, jnp.sign(residual), 0.0)
                    * wi
                )
                beta = beta - self.learning_rate * (
                    ki @ beta + self.c * (ki @ loss_grad)
                ) / jnp.maximum(ki.shape[0], 1)
                intercept = (
                    intercept - self.learning_rate * self.c * jnp.sum(loss_grad) / denom
                )
                return beta, intercept

            beta, intercept = jax.lax.fori_loop(0, self.iterations, step, (beta0, mean))
            residual = ki @ beta + intercept - yi
            objective = 0.5 * beta @ (ki @ beta) + self.c * jnp.sum(
                wi * jnp.maximum(jnp.abs(residual) - self.epsilon, 0.0)
            )
            return beta, intercept, objective

        coef, intercept, objective = jax.vmap(solve_one)(k, y, w)
        coef = coef.reshape(batch.case_shape + (batch.sample_count, 1))
        intercept = intercept.reshape(batch.case_shape + (1,))
        effective = jnp.sum(weights > 0, axis=-1)
        finite = jnp.all(finite_array(coef), axis=(-2, -1)) & finite_array(
            intercept[..., 0]
        )
        valid = finite & (effective > 0)
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(effective > 0, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        )
        model = SupportVectorRegressorModel(
            support=x,
            coefficients=coef,
            intercept=intercept,
            support_mask=weights > 0,
            kernel=self.kernel,
            feature_count=batch.feature_count,
            output_shape=(),
            case_shape=batch.case_shape,
            method="svr",
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            objective=objective.reshape(batch.case_shape),
            iterations=self.iterations,
            effective_samples=effective,
            method="unrolled-epsilon-svr",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="svr",
            gradient_contract=GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="conditional",
                fit_targets="almost-everywhere",
                fit_weights="almost-everywhere",
                fit_hyperparameters="almost-everywhere",
                fit_mode="unrolled",
                conditions=("Away from epsilon-tube and active-mask boundaries.",),
            ),
        )


class OneClassSVMRecipe(AbstractRecipe):
    kernel: Any
    nu: Array
    iterations: int = eqx.field(static=True)
    learning_rate: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: Any,
        /,
        *,
        nu: ArrayLike = 0.5,
        iterations: int = 200,
        learning_rate: ArrayLike = 0.05,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.kernel = validate_kernel(kernel)
        self.nu = jnp.asarray(nu, dtype=float)
        self.learning_rate = jnp.asarray(learning_rate, dtype=float)
        if self.nu.ndim != 0 or self.learning_rate.ndim != 0 or iterations <= 0:
            raise ValueError(
                "nu and learning_rate must be scalar and iterations positive."
            )
        if bool((self.nu <= 0) | (self.nu > 1) | (self.learning_rate <= 0)):
            raise ValueError("nu must lie in (0, 1] and learning_rate must be positive.")
        self.iterations = int(iterations)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x = batch.dense_features()
        feature_valid = jnp.all(finite_array(x), axis=-1)
        x = jnp.where(finite_array(x), x, 0.0)
        weights = (
            validated_weights(batch.effective_weight(self.weight_policy)) * feature_valid
        )
        gram = case_kernel_matrix(self.kernel, x, x, batch.case_shape)
        if jnp.iscomplexobj(gram):
            raise TypeError("One-class SVM requires a real-valued kernel.")
        cases = _size(batch.case_shape)
        k = gram.reshape((cases, batch.sample_count, batch.sample_count))
        w = weights.reshape((cases, batch.sample_count))

        def solve_one(ki, wi):
            total = jnp.sum(wi)
            normalized = wi / jnp.maximum(total, jnp.finfo(wi.dtype).tiny)
            cap = normalized / self.nu
            alpha0 = normalized

            def project(v):
                clipped = jnp.clip(v, 0.0, cap)
                return clipped / jnp.maximum(jnp.sum(clipped), jnp.finfo(v.dtype).tiny)

            alpha = jax.lax.fori_loop(
                0,
                self.iterations,
                lambda _, a: project(a - self.learning_rate * (ki @ a)),
                alpha0,
            )
            scores = ki @ alpha
            active = (alpha > 1e-6) & (alpha < cap - 1e-6) & (wi > 0)
            rho = jnp.sum(jnp.where(active, scores, 0.0)) / jnp.maximum(
                jnp.sum(active), 1
            )
            return alpha, -rho, 0.5 * alpha @ scores

        coef, intercept, objective = jax.vmap(solve_one)(k, w)
        coef = coef.reshape(batch.case_shape + (batch.sample_count, 1))
        intercept = intercept.reshape(batch.case_shape + (1,))
        effective = jnp.sum(weights > 0, axis=-1)
        finite = jnp.all(finite_array(coef), axis=(-2, -1)) & finite_array(
            intercept[..., 0]
        )
        valid = finite & (effective > 0)
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(effective > 0, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        )
        model = OneClassSVMModel(
            support=x,
            coefficients=coef,
            intercept=intercept,
            support_mask=weights > 0,
            kernel=self.kernel,
            feature_count=batch.feature_count,
            output_shape=(),
            case_shape=batch.case_shape,
            method="one-class-svm",
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            objective=objective.reshape(batch.case_shape),
            iterations=self.iterations,
            effective_samples=effective,
            method="projected-one-class-svm",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("predict",),
            conditions=("Fixed active-set branch.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="one-class-svm",
            gradient_contract=contract,
        )


__all__ = [
    "AbstractKernelLinearModel",
    "KernelRidgeModel",
    "KernelRidgeRecipe",
    "LeastSquaresSVMModel",
    "LeastSquaresSVMRecipe",
    "OneClassSVMModel",
    "OneClassSVMRecipe",
    "SupportVectorClassifierModel",
    "SupportVectorClassifierRecipe",
    "SupportVectorRegressorModel",
    "SupportVectorRegressorRecipe",
]
