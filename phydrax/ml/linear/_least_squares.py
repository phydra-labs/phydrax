#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...linalg import RankPolicy
from ...linalg._dense_pseudoinverse import (
    apply_pseudoinverse,
    factor_pseudoinverse,
)
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    GradientContract,
    ML_NONFINITE,
    ML_RANK_DEFICIENT,
    ML_SUCCESS,
)
from .._numerics import solve_weighted_least_squares
from ._base import (
    AbstractLinearRegressorModel,
    design_matmul,
    design_transpose_matmul,
    parameter_dtype,
    prepare_supervised,
    PreparedBatch,
    restore_case_shape,
    weighted_feature_gram,
)


class OLSModel(AbstractLinearRegressorModel):
    """Fitted weighted ordinary least-squares model."""


class RidgeModel(AbstractLinearRegressorModel):
    """Fitted isotropic Tikhonov (ridge) model."""


class TikhonovModel(AbstractLinearRegressorModel):
    """Fitted general Tikhonov model."""


def _validated_rcond(value: float | None, /) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("rcond must be finite and positive.")
    return result


def _direct_contract() -> GradientContract:
    return GradientContract.direct(
        conditions=(
            "Masks and sparse index structure are fixed.",
            "The retained singular subspace is locally constant.",
            "Every fitted augmented design has full column rank.",
        )
    )


def _aggregate_direct(
    prepared: PreparedBatch,
    coefficients: Array,
    intercept: Array,
    /,
    *,
    objective: Array,
    rank: Array,
    condition: Array,
    solver_valid: Array,
    solver_status: Array,
    method: str,
    model_type: type[AbstractLinearRegressorModel],
    gradient_contract: GradientContract,
) -> FitResult:
    parameter_finite = (
        jnp.all(jnp.isfinite(jnp.real(coefficients)), axis=(1, 2))
        & jnp.all(jnp.isfinite(jnp.imag(coefficients)), axis=(1, 2))
        & jnp.all(jnp.isfinite(jnp.real(intercept)), axis=1)
        & jnp.all(jnp.isfinite(jnp.imag(intercept)), axis=1)
    )
    output_valid = jnp.all(solver_valid, axis=-1)
    valid = prepared.data_valid & output_valid & parameter_finite
    status = jnp.max(solver_status, axis=-1)
    status = jnp.where(~prepared.data_valid, prepared.data_status, status)
    status = jnp.where(~parameter_finite, ML_NONFINITE, status).astype(jnp.int32)
    valid_cases = restore_case_shape(prepared, valid)
    status_cases = restore_case_shape(prepared, status)
    diagnostics = FitDiagnostics(
        valid=valid_cases,
        status=status_cases,
        objective=restore_case_shape(prepared, objective),
        iterations=restore_case_shape(prepared, jnp.asarray(0)),
        effective_samples=restore_case_shape(prepared, prepared.effective_samples),
        rank=restore_case_shape(prepared, jnp.min(rank, axis=-1)),
        condition=restore_case_shape(prepared, jnp.max(condition, axis=-1)),
        method=method,
    )
    model = model_type(
        coefficients,
        intercept,
        case_shape=prepared.case_shape,
        target_shape=prepared.target_shape,
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid_cases,
        status=status_cases,
        method=method,
        gradient_contract=gradient_contract,
    )


def _dense_isotropic_solve(
    prepared: PreparedBatch,
    /,
    *,
    ridge: Array,
    fit_intercept: bool,
    regularize_intercept: bool,
    rcond: float | None,
    method: str,
    model_type: type[AbstractLinearRegressorModel],
) -> FitResult:
    design = prepared.design.dense
    assert design is not None
    dtype = jnp.result_type(parameter_dtype(prepared), ridge)
    design = design.astype(dtype)
    cases, samples, features = design.shape
    outputs = prepared.outputs
    expanded_design = jnp.broadcast_to(
        design[:, None, :, :], (cases, outputs, samples, features)
    )
    targets = jnp.swapaxes(prepared.targets, 1, 2)
    weights = jnp.swapaxes(prepared.weights, 1, 2)
    solved = solve_weighted_least_squares(
        expanded_design,
        targets,
        weights,
        ridge=ridge,
        fit_intercept=fit_intercept,
        regularize_intercept=regularize_intercept,
        rcond=rcond,
    )
    coefficients = jnp.swapaxes(solved.coefficients, -1, -2).astype(dtype)
    intercept = solved.intercept.astype(dtype)
    objective = jnp.sum(solved.residual_sum_squares, axis=-1)
    if jnp.ndim(ridge) == 0:
        objective = objective + ridge * jnp.sum(jnp.abs(coefficients) ** 2, axis=(1, 2))
        if regularize_intercept:
            objective = objective + ridge * jnp.sum(jnp.abs(intercept) ** 2, axis=1)
    return _aggregate_direct(
        prepared,
        coefficients,
        intercept,
        objective=objective,
        rank=solved.rank,
        condition=solved.condition,
        solver_valid=solved.valid,
        solver_status=solved.status,
        method=method,
        model_type=model_type,
        gradient_contract=_direct_contract(),
    )


def _normal_solve(
    prepared: PreparedBatch,
    /,
    *,
    penalty_gram: Array,
    intercept_penalty: Array,
    fit_intercept: bool,
    rcond: float | None,
    method: str,
    model_type: type[AbstractLinearRegressorModel],
) -> FitResult:
    design = prepared.design
    weights = prepared.weights
    targets = prepared.targets
    cases = targets.shape[0]
    features = design.features
    outputs = prepared.outputs
    weighted_target = weights * targets
    rhs_features = design_transpose_matmul(design, weighted_target)
    rhs_intercept = jnp.sum(weighted_target, axis=1)

    feature_gram = weighted_feature_gram(design, weights)
    penalty = jnp.broadcast_to(penalty_gram, (cases, outputs, features, features))
    feature_gram = feature_gram + penalty

    parameter_count = features + int(fit_intercept)
    gram = jnp.zeros(
        (cases, outputs, parameter_count, parameter_count),
        dtype=jnp.result_type(feature_gram, targets),
    )
    gram = gram.at[..., :features, :features].set(feature_gram)
    rhs = jnp.zeros((cases, outputs, parameter_count), dtype=gram.dtype)
    rhs = rhs.at[..., :features].set(jnp.swapaxes(rhs_features, 1, 2))
    if fit_intercept:
        cross = design_transpose_matmul(design, weights)
        cross = jnp.swapaxes(cross, 1, 2)
        gram = gram.at[..., :features, -1].set(cross)
        gram = gram.at[..., -1, :features].set(jnp.conj(cross))
        gram = gram.at[..., -1, -1].set(jnp.sum(weights, axis=1) + intercept_penalty)
        rhs = rhs.at[..., -1].set(rhs_intercept)

    cutoff = (
        max(prepared.design.samples, parameter_count)
        * jnp.finfo(jnp.real(gram).dtype).eps
        if rcond is None
        else float(rcond)
    )
    gram_cutoff = cutoff * cutoff
    factors = factor_pseudoinverse(
        gram,
        RankPolicy(relative_cutoff=gram_cutoff),
        hermitian=True,
    )
    singular = factors.singular_values
    rank = factors.rank
    condition = jnp.sqrt(factors.condition_estimate)
    parameters = apply_pseudoinverse(factors, rhs)
    coefficients = jnp.swapaxes(parameters[..., :features], 1, 2)
    intercept = (
        parameters[..., -1]
        if fit_intercept
        else jnp.zeros((cases, outputs), dtype=parameters.dtype)
    )
    prediction = design_matmul(design, coefficients) + intercept[:, None, :]
    residual = prediction - targets
    rss = jnp.sum(weights * jnp.real(residual * jnp.conj(residual)), axis=(1, 2))
    penalty_value = jnp.real(
        ein.contract(
            "cfo,cofg,cgo->c",
            jnp.conj(coefficients),
            penalty,
            coefficients,
        )
    )
    if fit_intercept:
        penalty_value = penalty_value + jnp.sum(
            intercept_penalty * jnp.abs(intercept) ** 2, axis=-1
        )
    objective = rss + penalty_value
    finite = jnp.all(jnp.isfinite(singular), axis=-1)
    full_rank = rank == parameter_count
    solver_valid = finite & full_rank
    solver_status = jnp.where(
        ~finite,
        ML_NONFINITE,
        jnp.where(full_rank, ML_SUCCESS, ML_RANK_DEFICIENT),
    ).astype(jnp.int32)
    return _aggregate_direct(
        prepared,
        coefficients,
        intercept,
        objective=objective,
        rank=rank,
        condition=condition,
        solver_valid=solver_valid,
        solver_status=solver_status,
        method=method,
        model_type=model_type,
        gradient_contract=_direct_contract(),
    )


class OLSRecipe(AbstractRecipe):
    """Weighted ordinary least squares with an augmented-SVD dense solve."""

    fit_intercept: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    rcond: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        weight_policy: WeightPolicy = "statistical",
        rcond: float | None = None,
    ):
        self.fit_intercept = bool(fit_intercept)
        self.weight_policy = weight_policy
        self.rcond = _validated_rcond(rcond)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        prepared = prepare_supervised(batch, weight_policy=self.weight_policy)
        if not prepared.design.sparse:
            return _dense_isotropic_solve(
                prepared,
                ridge=jnp.asarray(0.0),
                fit_intercept=self.fit_intercept,
                regularize_intercept=False,
                rcond=self.rcond,
                method="weighted-ols-augmented-svd",
                model_type=OLSModel,
            )
        zero = jnp.zeros(
            (prepared.design.features, prepared.design.features),
            dtype=prepared.targets.dtype,
        )
        return _normal_solve(
            prepared,
            penalty_gram=zero,
            intercept_penalty=jnp.asarray(0.0),
            fit_intercept=self.fit_intercept,
            rcond=self.rcond,
            method="weighted-sparse-ols-normal-svd",
            model_type=OLSModel,
        )


class RidgeRecipe(AbstractRecipe):
    """Weighted ridge regression, including an explicit intercept penalty policy."""

    alpha: Array
    fit_intercept: bool = eqx.field(static=True)
    regularize_intercept: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    rcond: float | None = eqx.field(static=True)

    def __init__(
        self,
        alpha: ArrayLike = 1.0,
        *,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
        weight_policy: WeightPolicy = "statistical",
        rcond: float | None = None,
    ):
        alpha_ = jnp.asarray(alpha)
        if alpha_.weak_type:
            alpha_ = alpha_.astype(jnp.float32)
        if alpha_.ndim != 0:
            raise ValueError("alpha must be a finite non-negative scalar.")
        self.alpha = eqx.error_if(
            alpha_,
            ~jnp.isfinite(alpha_) | (alpha_ < 0.0),
            "alpha must be a finite non-negative scalar.",
        )
        self.fit_intercept = bool(fit_intercept)
        self.regularize_intercept = bool(regularize_intercept)
        self.weight_policy = weight_policy
        self.rcond = _validated_rcond(rcond)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        prepared = prepare_supervised(batch, weight_policy=self.weight_policy)
        alpha = self.alpha
        if not prepared.design.sparse:
            return _dense_isotropic_solve(
                prepared,
                ridge=alpha,
                fit_intercept=self.fit_intercept,
                regularize_intercept=self.regularize_intercept,
                rcond=self.rcond,
                method="weighted-ridge-augmented-svd",
                model_type=RidgeModel,
            )
        penalty = alpha * jnp.eye(prepared.design.features, dtype=prepared.targets.dtype)
        return _normal_solve(
            prepared,
            penalty_gram=penalty,
            intercept_penalty=alpha if self.regularize_intercept else jnp.asarray(0.0),
            fit_intercept=self.fit_intercept,
            rcond=self.rcond,
            method="weighted-sparse-ridge-normal-svd",
            model_type=RidgeModel,
        )


class TikhonovRecipe(AbstractRecipe):
    r"""General weighted Tikhonov regression with penalty ``strength * ||L beta||²``."""

    penalty: Array
    strength: Array
    fit_intercept: bool = eqx.field(static=True)
    regularize_intercept: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    rcond: float | None = eqx.field(static=True)

    def __init__(
        self,
        penalty: ArrayLike,
        *,
        strength: ArrayLike = 1.0,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
        weight_policy: WeightPolicy = "statistical",
        rcond: float | None = None,
    ):
        value = jnp.asarray(penalty)
        if value.ndim not in {1, 2}:
            raise ValueError(
                "penalty must be a diagonal vector or a two-dimensional L matrix."
            )
        self.penalty = value
        strength_ = jnp.asarray(strength)
        if strength_.weak_type:
            strength_ = strength_.astype(jnp.float32)
        if strength_.ndim != 0:
            raise ValueError("strength must be a finite non-negative scalar.")
        self.strength = eqx.error_if(
            strength_,
            ~jnp.isfinite(strength_) | (strength_ < 0.0),
            "strength must be a finite non-negative scalar.",
        )
        self.fit_intercept = bool(fit_intercept)
        self.regularize_intercept = bool(regularize_intercept)
        self.weight_policy = weight_policy
        self.rcond = _validated_rcond(rcond)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        prepared = prepare_supervised(batch, weight_policy=self.weight_policy)
        features = prepared.design.features
        if int(self.penalty.shape[-1]) != features:
            raise ValueError("The Tikhonov penalty final dimension must match features.")
        operator = jnp.diag(self.penalty) if self.penalty.ndim == 1 else self.penalty
        strength = self.strength
        gram = strength * (jnp.conj(operator).T @ operator)
        return _normal_solve(
            prepared,
            penalty_gram=gram,
            intercept_penalty=strength if self.regularize_intercept else jnp.asarray(0.0),
            fit_intercept=self.fit_intercept,
            rcond=self.rcond,
            method="weighted-general-tikhonov-normal-svd",
            model_type=TikhonovModel,
        )


__all__ = [
    "OLSRecipe",
    "OLSModel",
    "RidgeRecipe",
    "RidgeModel",
    "TikhonovRecipe",
    "TikhonovModel",
]
