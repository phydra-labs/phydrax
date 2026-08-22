#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...optim import (
    ConvexDifferentiationPolicy,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexTermination,
    DensePrimalDualQP,
    QuadraticProgram,
    solve_quadratic_program,
    solve_quadratic_program_primal,
)
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)
from ._base import (
    AbstractLinearRegressorModel,
    design_matmul,
    design_row_norm_bound,
    design_transpose_matmul,
    iterative_fit,
    parameter_dtype,
    prepare_supervised,
    PreparedBatch,
    restore_case_shape,
    unrolled_contract,
    weighted_rank_condition,
)
from ._least_squares import _normal_solve


class HuberModel(AbstractLinearRegressorModel):
    """Fitted smooth-loss robust linear model without hard sample selection."""


class QuantileModel(AbstractLinearRegressorModel):
    """Fitted conditional-quantile linear model."""


class RANSACModel(AbstractLinearRegressorModel):
    """Fitted exact hard-consensus linear model."""


class TheilSenModel(AbstractLinearRegressorModel):
    """Fitted hard subset-median linear model."""


class RobustDiagnostics(StrictModule):
    """Fit diagnostics plus exact robust-selection evidence."""

    common: FitDiagnostics
    inlier_mask: Array
    selected_subset: Array
    subset_scores: Array

    def __init__(
        self,
        *,
        common: FitDiagnostics,
        inlier_mask: Array,
        selected_subset: Array,
        subset_scores: Array,
    ):
        self.common = common
        self.inlier_mask = jnp.asarray(inlier_mask, dtype=bool)
        self.selected_subset = jnp.asarray(selected_subset, dtype=bool)
        self.subset_scores = jax.lax.stop_gradient(jnp.asarray(subset_scores))

    @property
    def valid(self) -> Array:
        return self.common.valid

    @property
    def status(self) -> Array:
        return self.common.status

    @property
    def objective(self) -> Array:
        return self.common.objective

    @property
    def iterations(self) -> Array:
        return self.common.iterations

    @property
    def effective_samples(self) -> Array:
        return self.common.effective_samples

    @property
    def rank(self) -> Array:
        return self.common.rank

    @property
    def condition(self) -> Array:
        return self.common.condition

    @property
    def method(self) -> str:
        return self.common.method


def _scalar(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    result = jnp.asarray(value)
    if result.weak_type:
        result = result.astype(jnp.float32)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    invalid = ~jnp.isfinite(result) | ((result <= 0.0) if positive else (result < 0.0))
    qualifier = "positive" if positive else "non-negative"
    return eqx.error_if(result, invalid, f"{name} must be {qualifier}.")


def _fit_robust_loss(recipe, batch: MLBatch, /, *, family: str) -> FitResult:
    prepared = prepare_supervised(
        batch, weight_policy=recipe.weight_policy, require_real=True
    )
    cases = prepared.targets.shape[0]
    features = prepared.design.features
    outputs = prepared.outputs
    family_parameter = recipe.delta if family == "huber" else recipe.quantile
    dtype = jnp.result_type(
        parameter_dtype(prepared), family_parameter, recipe.l2_strength
    )
    if recipe.learning_rate is not None:
        dtype = jnp.result_type(dtype, recipe.learning_rate)
    coefficients = jnp.zeros((cases, features, outputs), dtype=dtype)
    mass = jnp.sum(prepared.weights, axis=1)
    intercept = (
        jnp.where(
            mass > 0.0,
            jnp.sum(prepared.weights * prepared.targets, axis=1)
            / jnp.maximum(mass, jnp.finfo(mass.dtype).tiny),
            0.0,
        )
        if recipe.fit_intercept
        else jnp.zeros((cases, outputs), dtype=dtype)
    )
    intercept = intercept.astype(dtype)
    if recipe.learning_rate is None:
        bound = design_row_norm_bound(prepared.design) + float(recipe.fit_intercept)
        lipschitz = (
            jnp.max(jnp.sum(prepared.weights * bound[..., None], axis=1))
            + recipe.l2_strength
        )
        learning_rate = 1.0 / jnp.maximum(lipschitz, jnp.finfo(dtype).tiny)
    else:
        learning_rate = recipe.learning_rate

    def loss_derivative(residual):
        if family == "huber":
            magnitude = jnp.abs(residual)
            loss = jnp.where(
                magnitude <= recipe.delta,
                0.5 * residual * residual,
                recipe.delta * (magnitude - 0.5 * recipe.delta),
            )
            derivative = jnp.where(
                magnitude <= recipe.delta,
                residual,
                recipe.delta * jnp.sign(residual),
            )
            return loss, derivative
        loss = jnp.where(
            residual >= 0.0,
            (1.0 - recipe.quantile) * residual,
            -recipe.quantile * residual,
        )
        derivative = jnp.where(residual >= 0.0, 1.0 - recipe.quantile, -recipe.quantile)
        return loss, derivative

    def objective(beta, bias):
        residual = (
            design_matmul(prepared.design, beta) + bias[:, None, :] - prepared.targets
        )
        loss, _ = loss_derivative(residual)
        return jnp.sum(
            prepared.weights * loss, axis=(1, 2)
        ) + 0.5 * recipe.l2_strength * jnp.sum(beta * beta, axis=(1, 2))

    def step(state, iteration):
        del iteration
        beta, bias = state
        residual = (
            design_matmul(prepared.design, beta) + bias[:, None, :] - prepared.targets
        )
        _, derivative = loss_derivative(residual)
        weighted = prepared.weights * derivative
        beta_candidate = beta - learning_rate * (
            design_transpose_matmul(prepared.design, weighted) + recipe.l2_strength * beta
        )
        bias_candidate = (
            bias - learning_rate * jnp.sum(weighted, axis=1)
            if recipe.fit_intercept
            else bias
        )
        change = jnp.maximum(
            jnp.max(jnp.abs(beta_candidate - beta)),
            jnp.max(jnp.abs(bias_candidate - bias)),
        )
        return (
            (
                beta_candidate,
                bias_candidate,
            ),
            jnp.sum(objective(beta_candidate, bias_candidate)),
            change,
        )

    model_type = HuberModel if family == "huber" else QuantileModel
    return iterative_fit(
        prepared,
        step=step,
        initial=(coefficients, intercept),
        max_iterations=recipe.max_iterations,
        tolerance=recipe.tolerance,
        method=f"weighted-{family}-fixed-subgradient",
        objective=objective,
        model_factory=lambda beta, bias: model_type(
            beta,
            bias,
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        ),
        gradient_contract=unrolled_contract(nonsmooth=True),
    )


class HuberRegressorRecipe(AbstractRecipe):
    """Weighted Huber M-estimator; unlike RANSAC it performs no hard selection."""

    delta: Array
    l2_strength: Array
    learning_rate: Array | None
    fit_intercept: bool = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        delta: ArrayLike = 1.0,
        *,
        l2_strength: ArrayLike = 0.0,
        fit_intercept: bool = True,
        learning_rate: ArrayLike | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.delta = _scalar(delta, "delta", positive=True)
        self.l2_strength = _scalar(l2_strength, "l2_strength")
        self.learning_rate = (
            None
            if learning_rate is None
            else _scalar(learning_rate, "learning_rate", positive=True)
        )
        self.fit_intercept = bool(fit_intercept)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_robust_loss(self, batch, family="huber")


class QuantileRegressorRecipe(AbstractRecipe):
    """Weighted quantile regression via fixed subgradients or the native dense QP."""

    quantile: Array
    l2_strength: Array
    learning_rate: Array | None
    fit_intercept: bool = eqx.field(static=True)
    solver: Literal["fixed-subgradient", "dense-qp"] = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    max_dense_dimension: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        quantile: ArrayLike = 0.5,
        *,
        l2_strength: ArrayLike = 0.0,
        fit_intercept: bool = True,
        solver: Literal["fixed-subgradient", "dense-qp"] = "dense-qp",
        learning_rate: ArrayLike | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
        max_dense_dimension: int = 512,
    ):
        quantile_ = jnp.asarray(quantile)
        if quantile_.weak_type:
            quantile_ = quantile_.astype(jnp.float32)
        if quantile_.ndim != 0:
            raise ValueError("quantile must be a scalar strictly between zero and one.")
        quantile_ = eqx.error_if(
            quantile_,
            ~jnp.isfinite(quantile_) | (quantile_ <= 0.0) | (quantile_ >= 1.0),
            "quantile must be a scalar strictly between zero and one.",
        )
        if solver not in {"fixed-subgradient", "dense-qp"}:
            raise ValueError("solver must be 'fixed-subgradient' or 'dense-qp'.")
        if int(max_dense_dimension) <= 0:
            raise ValueError("max_dense_dimension must be positive.")
        self.quantile = quantile_
        self.l2_strength = _scalar(l2_strength, "l2_strength")
        self.learning_rate = (
            None
            if learning_rate is None
            else _scalar(learning_rate, "learning_rate", positive=True)
        )
        self.fit_intercept = bool(fit_intercept)
        self.solver = solver
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy
        self.max_dense_dimension = int(max_dense_dimension)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if self.solver == "fixed-subgradient":
            return _fit_robust_loss(self, batch, family="quantile")
        return self._fit_qp(batch)

    def _fit_qp(self, batch: MLBatch) -> FitResult:
        prepared = prepare_supervised(
            batch, weight_policy=self.weight_policy, require_real=True
        )
        if prepared.design.sparse:
            raise TypeError(
                "dense-qp quantile fitting requires dense features; use fixed-subgradient for SparseFeatures."
            )
        design = prepared.design.dense
        assert design is not None
        dtype = jnp.result_type(
            parameter_dtype(prepared), self.quantile, self.l2_strength
        )
        design = design.astype(dtype)
        cases, samples, features = design.shape
        outputs = prepared.outputs
        parameters = features + int(self.fit_intercept)
        variables = parameters + 2 * samples
        augmented = (
            jnp.concatenate(
                (design, jnp.ones((cases, samples, 1), dtype=design.dtype)), axis=-1
            )
            if self.fit_intercept
            else design
        )
        augmented = jnp.broadcast_to(
            augmented[:, None, :, :], (cases, outputs, samples, parameters)
        )
        identity = jnp.broadcast_to(
            jnp.eye(samples, dtype=design.dtype), (cases, outputs, samples, samples)
        )
        active = jnp.swapaxes(prepared.weights, 1, 2) > 0.0
        equality_active = jnp.concatenate((augmented, identity, -identity), axis=-1)
        equality_inactive = jnp.zeros_like(equality_active)
        equality_inactive = equality_inactive.at[
            ..., parameters : parameters + samples
        ].set(identity)
        equality = jnp.where(active[..., None], equality_active, equality_inactive)
        equality_rhs = jnp.where(active, jnp.swapaxes(prepared.targets, 1, 2), 0.0)
        quadratic = jnp.zeros((cases, outputs, variables, variables), dtype=design.dtype)
        quadratic = quadratic.at[..., :features, :features].set(
            self.l2_strength * jnp.eye(features, dtype=design.dtype)
        )
        weights = jnp.swapaxes(prepared.weights, 1, 2)
        linear = jnp.zeros((cases, outputs, variables), dtype=dtype)
        linear = linear.at[..., parameters : parameters + samples].set(
            jnp.where(active, self.quantile * weights, 1.0)
        )
        linear = linear.at[..., parameters + samples :].set(
            jnp.where(active, (1.0 - self.quantile) * weights, 1.0)
        )
        inequalities = jnp.zeros(
            (cases, outputs, 2 * samples, variables), dtype=design.dtype
        )
        inequalities = inequalities.at[
            ..., :samples, parameters : parameters + samples
        ].set(-jnp.eye(samples, dtype=design.dtype))
        inequalities = inequalities.at[..., samples:, parameters + samples :].set(
            -jnp.eye(samples, dtype=design.dtype)
        )
        inequality_rhs = jnp.zeros((cases, outputs, 2 * samples), dtype=design.dtype)
        # An inactive row already fixes its positive residual to zero by equality.
        # Keep the corresponding lower bound strictly slack so the implicit KKT
        # system does not contain the same constraint twice.
        inequality_rhs = inequality_rhs.at[..., :samples].set(jnp.where(active, 0.0, 1.0))
        problem = QuadraticProgram(
            quadratic,
            linear,
            equality_matrix=equality,
            equality_rhs=equality_rhs,
            inequality_matrix=inequalities,
            inequality_rhs=inequality_rhs,
        )
        policy = ConvexSolvePolicy(
            DensePrimalDualQP(max_kkt_dimension=self.max_dense_dimension),
            termination=ConvexTermination(
                absolute=self.tolerance,
                maximum_steps=self.max_iterations,
            ),
        )
        audited = solve_quadratic_program(problem, policy=policy)
        differentiable = solve_quadratic_program_primal(
            problem,
            policy=policy,
            differentiation=ConvexDifferentiationPolicy("active-set-kkt"),
        )
        primal = differentiable
        coefficients = jnp.swapaxes(primal[..., :features], 1, 2)
        intercept = (
            primal[..., features]
            if self.fit_intercept
            else jnp.zeros((cases, outputs), dtype=primal.dtype)
        )
        qp_status = audited.status
        mapped = jnp.where(
            (qp_status == int(ConvexProgramStatus.NONFINITE_INPUT))
            | (qp_status == int(ConvexProgramStatus.NONFINITE_OUTPUT)),
            ML_NONFINITE,
            jnp.where(
                qp_status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE),
                ML_INFEASIBLE,
                jnp.where(
                    qp_status == int(ConvexProgramStatus.OPTIMAL),
                    ML_SUCCESS,
                    ML_NONCONVERGED,
                ),
            ),
        )
        status = jnp.max(mapped, axis=-1).astype(jnp.int32)
        parameter_finite = jnp.all(jnp.isfinite(coefficients), axis=(1, 2)) & jnp.all(
            jnp.isfinite(intercept), axis=1
        )
        status = jnp.where(~prepared.data_valid, prepared.data_status, status)
        status = jnp.where(~parameter_finite, ML_NONFINITE, status)
        valid = prepared.data_valid & jnp.all(audited.valid, axis=-1) & parameter_finite
        objective = jnp.sum(audited.objective, axis=-1)
        rank, condition = weighted_rank_condition(prepared.design, prepared.weights)
        valid_cases = restore_case_shape(prepared, valid)
        status_cases = restore_case_shape(prepared, status)
        diagnostics = FitDiagnostics(
            valid=valid_cases,
            status=status_cases,
            objective=restore_case_shape(prepared, objective),
            iterations=restore_case_shape(prepared, jnp.max(audited.iterations, axis=-1)),
            effective_samples=restore_case_shape(prepared, prepared.effective_samples),
            rank=restore_case_shape(prepared, rank),
            condition=restore_case_shape(prepared, condition),
            method="weighted-quantile-native-dense-qp",
        )
        model = QuantileModel(
            coefficients,
            intercept,
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="implicit",
            conditions=(
                "The dense QP active set is locally constant.",
                "Masks and active inequality identities are fixed.",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid_cases,
            status=status_cases,
            method="weighted-quantile-native-dense-qp",
            gradient_contract=contract,
        )


def _subset_prepared(prepared: PreparedBatch, weights: Array) -> PreparedBatch:
    mass = jnp.sum(weights, axis=1)
    squared = jnp.sum(weights * weights, axis=1)
    effective = jnp.min(jnp.where(squared > 0.0, mass * mass / squared, 0.0), axis=-1)
    enough = jnp.all(mass > 0.0, axis=-1)
    data_status = jnp.where(
        prepared.data_valid,
        jnp.where(enough, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        prepared.data_status,
    ).astype(jnp.int32)
    return PreparedBatch(
        design=prepared.design,
        targets=prepared.targets,
        weights=weights,
        data_valid=data_status == ML_SUCCESS,
        data_status=data_status,
        effective_samples=effective,
        case_shape=prepared.case_shape,
        target_shape=prepared.target_shape,
        outputs=prepared.outputs,
    )


def _hard_contract(method: str) -> GradientContract:
    return GradientContract(
        prediction_inputs="smooth",
        prediction_parameters="smooth",
        fit_features="none",
        fit_targets="none",
        fit_weights="none",
        fit_hyperparameters="none",
        fit_mode="stopped",
        nondifferentiable_outputs=(
            "selected_subset",
            "inlier_mask",
            "subset_scores",
        ),
        conditions=(f"{method} subset and order-statistic choices are discrete.",),
    )


class RANSACRegressorRecipe(AbstractRecipe):
    """Exact random-sample consensus with hard inlier selection and an OLS refit."""

    residual_threshold: float = eqx.field(static=True)
    min_samples: int | None = eqx.field(static=True)
    num_trials: int = eqx.field(static=True)
    fit_intercept: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual_threshold: float = 1.0,
        min_samples: int | None = None,
        num_trials: int = 64,
        fit_intercept: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        threshold = float(residual_threshold)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("residual_threshold must be finite and positive.")
        if int(num_trials) <= 0:
            raise ValueError("num_trials must be positive.")
        self.residual_threshold = threshold
        self.min_samples = None if min_samples is None else int(min_samples)
        self.num_trials = int(num_trials)
        self.fit_intercept = bool(fit_intercept)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("RANSAC fitting requires an explicit JAX key.")
        prepared = prepare_supervised(batch, weight_policy=self.weight_policy)
        cases, samples, _ = prepared.targets.shape
        subset_size = self.min_samples or (
            prepared.design.features + int(self.fit_intercept)
        )
        if subset_size <= 0 or subset_size > samples:
            raise ValueError("min_samples must be in [1, sample_count].")
        active = jnp.any(prepared.weights > 0.0, axis=-1)
        zero = jnp.zeros(
            (prepared.design.features, prepared.design.features),
            dtype=prepared.targets.dtype,
        )
        keys = jax.random.split(key, self.num_trials)
        inlier_candidates = []
        subsets = []
        scores = []
        for trial_key in keys:
            random_score = jax.random.uniform(trial_key, (cases, samples))
            random_score = jnp.where(active, random_score, jnp.inf)
            indices = jnp.argsort(random_score, axis=-1)[:, :subset_size]
            subset = (
                jnp.zeros((cases, samples), dtype=bool)
                .at[jnp.arange(cases)[:, None], indices]
                .set(True)
            )
            subset_weights = prepared.weights * subset[..., None]
            candidate = _normal_solve(
                _subset_prepared(prepared, subset_weights),
                penalty_gram=zero,
                intercept_penalty=jnp.asarray(0.0),
                fit_intercept=self.fit_intercept,
                rcond=None,
                method="ransac-subset-ols",
                model_type=RANSACModel,
            )
            model = candidate.as_trainable(RANSACModel)
            beta = model.coefficients.reshape(
                (cases, prepared.design.features, prepared.outputs)
            )
            bias = model.intercept.reshape((cases, prepared.outputs))
            residual = (
                design_matmul(prepared.design, beta) + bias[:, None, :] - prepared.targets
            )
            inlier = active & jnp.all(
                (prepared.weights == 0.0)
                | (jnp.abs(residual) <= self.residual_threshold),
                axis=-1,
            )
            mass = jnp.sum(
                jnp.where(inlier[..., None], prepared.weights, 0.0), axis=(1, 2)
            )
            loss = jnp.sum(prepared.weights * jnp.abs(residual), axis=(1, 2))
            score = mass - jnp.finfo(float).eps * loss
            score = jnp.where(candidate.valid, score, -jnp.inf)
            inlier_candidates.append(inlier)
            subsets.append(subset)
            scores.append(score)
        score_array = jnp.stack(scores, axis=-1)
        choice = jnp.argmax(score_array, axis=-1)
        inlier_array = jnp.stack(inlier_candidates, axis=1)
        subset_array = jnp.stack(subsets, axis=1)
        row = jnp.arange(cases)
        selected_inlier = inlier_array[row, choice]
        selected_subset = subset_array[row, choice]
        refit = _normal_solve(
            _subset_prepared(prepared, prepared.weights * selected_inlier[..., None]),
            penalty_gram=zero,
            intercept_penalty=jnp.asarray(0.0),
            fit_intercept=self.fit_intercept,
            rcond=None,
            method="ransac-hard-consensus-refit",
            model_type=RANSACModel,
        )
        common = jax.tree_util.tree_map(jax.lax.stop_gradient, refit.diagnostics)
        diagnostics = RobustDiagnostics(
            common=common,
            inlier_mask=selected_inlier.reshape(prepared.case_shape + (samples,)),
            selected_subset=selected_subset.reshape(prepared.case_shape + (samples,)),
            subset_scores=score_array.reshape(prepared.case_shape + (self.num_trials,)),
        )
        refit_model = refit.as_trainable(RANSACModel)
        model = RANSACModel(
            jax.lax.stop_gradient(refit_model.coefficients),
            jax.lax.stop_gradient(refit_model.intercept),
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        )
        return FitResult(
            model,
            diagnostics,
            valid=refit.valid,
            status=refit.status,
            method="ransac-hard-consensus-refit",
            gradient_contract=_hard_contract("RANSAC"),
        )


class TheilSenRegressorRecipe(AbstractRecipe):
    """Random fixed-capacity Theil-Sen generalization using coordinatewise subset medians."""

    subset_size: int | None = eqx.field(static=True)
    num_subsets: int = eqx.field(static=True)
    fit_intercept: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        subset_size: int | None = None,
        num_subsets: int = 128,
        fit_intercept: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(num_subsets) <= 0:
            raise ValueError("num_subsets must be positive.")
        self.subset_size = None if subset_size is None else int(subset_size)
        self.num_subsets = int(num_subsets)
        self.fit_intercept = bool(fit_intercept)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("Theil-Sen fitting requires an explicit JAX key.")
        prepared = prepare_supervised(
            batch, weight_policy=self.weight_policy, require_real=True
        )
        cases, samples, _ = prepared.targets.shape
        subset_size = self.subset_size or (
            prepared.design.features + int(self.fit_intercept)
        )
        if subset_size <= 0 or subset_size > samples:
            raise ValueError("subset_size must be in [1, sample_count].")
        active = jnp.any(prepared.weights > 0.0, axis=-1)
        zero = jnp.zeros(
            (prepared.design.features, prepared.design.features),
            dtype=prepared.targets.dtype,
        )
        keys = jax.random.split(key, self.num_subsets)
        coefficients = []
        intercepts = []
        valid_subsets = []
        subset_masks = []
        for subset_key in keys:
            score = jnp.where(
                active,
                jax.random.uniform(subset_key, (cases, samples)),
                jnp.inf,
            )
            indices = jnp.argsort(score, axis=-1)[:, :subset_size]
            mask = (
                jnp.zeros((cases, samples), dtype=bool)
                .at[jnp.arange(cases)[:, None], indices]
                .set(True)
            )
            candidate = _normal_solve(
                _subset_prepared(prepared, prepared.weights * mask[..., None]),
                penalty_gram=zero,
                intercept_penalty=jnp.asarray(0.0),
                fit_intercept=self.fit_intercept,
                rcond=None,
                method="theil-sen-subset-ols",
                model_type=TheilSenModel,
            )
            model = candidate.as_trainable(TheilSenModel)
            coefficients.append(
                model.coefficients.reshape(
                    (cases, prepared.design.features, prepared.outputs)
                )
            )
            intercepts.append(model.intercept.reshape((cases, prepared.outputs)))
            valid_subsets.append(candidate.valid.reshape((cases,)))
            subset_masks.append(mask)
        beta_candidates = jnp.stack(coefficients, axis=1)
        bias_candidates = jnp.stack(intercepts, axis=1)
        subset_valid = jnp.stack(valid_subsets, axis=1)
        beta_candidates = jnp.where(
            subset_valid[..., None, None], beta_candidates, jnp.nan
        )
        bias_candidates = jnp.where(subset_valid[..., None], bias_candidates, jnp.nan)
        beta = jnp.nanmedian(beta_candidates, axis=1)
        bias = jnp.nanmedian(bias_candidates, axis=1)
        prediction = design_matmul(prepared.design, beta) + bias[:, None, :]
        residual = prediction - prepared.targets
        objective = jnp.sum(prepared.weights * jnp.abs(residual), axis=(1, 2))
        finite = jnp.all(jnp.isfinite(beta), axis=(1, 2)) & jnp.all(
            jnp.isfinite(bias), axis=1
        )
        any_subset = jnp.any(subset_valid, axis=1)
        valid = prepared.data_valid & any_subset & finite
        status = jnp.where(
            ~prepared.data_valid,
            prepared.data_status,
            jnp.where(
                ~finite,
                ML_NONFINITE,
                jnp.where(any_subset, ML_SUCCESS, ML_INSUFFICIENT_DATA),
            ),
        ).astype(jnp.int32)
        rank, condition = weighted_rank_condition(prepared.design, prepared.weights)
        valid_cases = restore_case_shape(prepared, valid)
        status_cases = restore_case_shape(prepared, status)
        common = FitDiagnostics(
            valid=valid_cases,
            status=status_cases,
            objective=restore_case_shape(prepared, objective),
            iterations=restore_case_shape(prepared, jnp.asarray(self.num_subsets)),
            effective_samples=restore_case_shape(prepared, prepared.effective_samples),
            rank=restore_case_shape(prepared, rank),
            condition=restore_case_shape(prepared, condition),
            method="theil-sen-hard-subset-median",
        )
        common = jax.tree_util.tree_map(jax.lax.stop_gradient, common)
        subset_stack = jnp.stack(subset_masks, axis=1)
        diagnostics = RobustDiagnostics(
            common=common,
            inlier_mask=jnp.zeros(prepared.case_shape + (samples,), dtype=bool),
            selected_subset=subset_stack.reshape(
                prepared.case_shape + (self.num_subsets, samples)
            ),
            subset_scores=subset_valid.reshape(prepared.case_shape + (self.num_subsets,)),
        )
        model = TheilSenModel(
            jax.lax.stop_gradient(beta),
            jax.lax.stop_gradient(bias),
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid_cases,
            status=status_cases,
            method="theil-sen-hard-subset-median",
            gradient_contract=_hard_contract("Theil-Sen"),
        )


__all__ = [
    "HuberModel",
    "HuberRegressorRecipe",
    "QuantileModel",
    "QuantileRegressorRecipe",
    "RANSACModel",
    "RANSACRegressorRecipe",
    "RobustDiagnostics",
    "TheilSenModel",
    "TheilSenRegressorRecipe",
]
