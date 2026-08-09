#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    FitDiagnostics,
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import run_fixed_iterations
from .._schema import TargetSchema
from .._sparse_features import SparseFeatures


def _product(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 1


def _finite(value: Array) -> Array:
    return jnp.isfinite(jnp.real(value)) & jnp.isfinite(jnp.imag(value))


class Design(StrictModule):
    """Case-flattened dense or fixed-width sparse design operator."""

    dense: Array | None
    values: Array | None
    indices: Array | None
    entry_valid: Array | None
    row_valid: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    samples: int = eqx.field(static=True)
    features: int = eqx.field(static=True)
    sparse: bool = eqx.field(static=True)

    def __init__(self, batch: MLBatch):
        cases = _product(batch.case_shape)
        if isinstance(batch.features, SparseFeatures):
            sparse = batch.features
            raw = sparse.values.reshape((cases, batch.sample_count, sparse.row_width))
            valid = sparse.columns.valid.reshape(raw.shape)
            finite = _finite(raw)
            self.dense = None
            self.values = jnp.where(valid & finite, raw, 0)
            self.indices = sparse.columns.source_indices.reshape(raw.shape)
            self.entry_valid = valid
            self.row_valid = jnp.all((~valid) | finite, axis=-1)
            self.sparse = True
        else:
            raw = jnp.asarray(batch.features).reshape(
                (cases, batch.sample_count, batch.feature_count)
            )
            mask = batch.feature_mask.reshape(raw.shape)
            finite = _finite(raw)
            self.dense = jnp.where(mask & finite, raw, 0)
            self.values = None
            self.indices = None
            self.entry_valid = None
            self.row_valid = jnp.all((~mask) | finite, axis=-1)
            self.sparse = False
        self.case_shape = batch.case_shape
        self.samples = batch.sample_count
        self.features = batch.feature_count


class PreparedBatch(StrictModule):
    design: Design
    targets: Array
    weights: Array
    data_valid: Array
    data_status: Array
    effective_samples: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    outputs: int = eqx.field(static=True)


def prepare_supervised(
    batch: MLBatch,
    /,
    *,
    weight_policy: WeightPolicy,
    require_real: bool = False,
) -> PreparedBatch:
    target = batch.require_targets()
    if require_real and jnp.issubdtype(target.dtype, jnp.complexfloating):
        raise TypeError("This linear family requires real-valued targets.")
    design = Design(batch)
    if require_real:
        design_dtype = (
            design.values.dtype if design.sparse else design.dense.dtype  # type: ignore[union-attr]
        )
        if jnp.issubdtype(design_dtype, jnp.complexfloating):
            raise TypeError("This linear family requires real-valued features.")
    feature_dtype = (
        design.values.dtype if design.sparse else design.dense.dtype  # type: ignore[union-attr]
    )
    data_dtype = jnp.result_type(feature_dtype, target.dtype, jnp.float32)
    weight_dtype = jnp.real(jnp.empty((), dtype=data_dtype)).dtype
    cases = _product(batch.case_shape)
    target_shape = batch.target_shape or ()
    outputs = _product(target_shape)
    raw_y = target.reshape((cases, batch.sample_count, outputs))
    mask = batch.target_mask
    assert mask is not None
    target_mask = mask.reshape(raw_y.shape)
    y_finite = _finite(raw_y)
    y = jnp.where(target_mask & y_finite, raw_y, 0)

    raw_weight = (
        batch.effective_weight(weight_policy)
        .astype(weight_dtype)
        .reshape((cases, batch.sample_count))
    )
    weight_finite = jnp.isfinite(raw_weight)
    weight_nonnegative = raw_weight >= 0.0
    safe_base = jnp.where(weight_finite & weight_nonnegative, raw_weight, 0.0)
    active = target_mask & y_finite & design.row_valid[..., None]
    weights = jnp.where(active, safe_base[..., None], 0.0)

    weight_values_finite = jnp.all(weight_finite, axis=-1)
    weights_feasible = jnp.all(weight_nonnegative, axis=-1)
    feature_valid = jnp.all(design.row_valid | (safe_base == 0.0), axis=-1)
    target_valid = jnp.all(
        ((~target_mask) | y_finite) | (safe_base[..., None] == 0.0), axis=(1, 2)
    )
    mass = jnp.sum(weights, axis=1)
    enough = jnp.all(mass > 0.0, axis=-1)
    data_status = jnp.where(
        ~(weight_values_finite & feature_valid & target_valid),
        ML_NONFINITE,
        jnp.where(
            ~weights_feasible,
            ML_INFEASIBLE,
            jnp.where(enough, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        ),
    ).astype(jnp.int32)
    data_valid = data_status == ML_SUCCESS
    squared_mass = jnp.sum(weights * weights, axis=1)
    effective = jnp.min(
        jnp.where(squared_mass > 0.0, mass * mass / squared_mass, 0.0), axis=-1
    )
    return PreparedBatch(
        design=design,
        targets=y,
        weights=weights,
        data_valid=data_valid,
        data_status=data_status,
        effective_samples=effective,
        case_shape=batch.case_shape,
        target_shape=target_shape,
        outputs=outputs,
    )


def parameter_dtype(prepared: PreparedBatch, /):
    """Return the lossless floating/complex dtype shared by data and weights."""
    design = prepared.design
    feature_dtype = (
        design.values.dtype if design.sparse else design.dense.dtype  # type: ignore[union-attr]
    )
    return jnp.result_type(
        feature_dtype,
        prepared.targets.dtype,
        prepared.weights.dtype,
        jnp.float32,
    )


def restore_case_shape(prepared: PreparedBatch, value: Array, /) -> Array:
    """Restore one case-flattened scalar diagnostic to the public case shape."""
    flat = jnp.asarray(value)
    if flat.ndim == 0:
        flat = jnp.broadcast_to(flat, (prepared.targets.shape[0],))
    return flat.reshape(prepared.case_shape)


def design_matmul(design: Design, coefficients: Array, /) -> Array:
    """Apply a case-flattened design to ``(case, feature, output)`` weights."""
    if not design.sparse:
        assert design.dense is not None
        return jnp.einsum("cnf,cfo->cno", design.dense, coefficients)
    assert design.values is not None
    assert design.indices is not None
    assert design.entry_valid is not None

    def one(values, indices, valid, beta):
        gathered = beta[indices]
        return jnp.sum(
            jnp.where(valid[..., None], values[..., None] * gathered, 0), axis=1
        )

    return jax.vmap(one)(design.values, design.indices, design.entry_valid, coefficients)


def design_transpose_matmul(design: Design, values: Array, /) -> Array:
    """Apply the conjugate transpose to ``(case, sample, output)`` values."""
    if not design.sparse:
        assert design.dense is not None
        return jnp.einsum("cnf,cno->cfo", jnp.conj(design.dense), values)
    assert design.values is not None
    assert design.indices is not None
    assert design.entry_valid is not None

    def one(entries, indices, valid, residual):
        updates = jnp.where(
            valid[..., None], jnp.conj(entries)[..., None] * residual[:, None, :], 0
        )
        return (
            jnp.zeros((design.features, residual.shape[-1]), dtype=updates.dtype)
            .at[indices]
            .add(updates)
        )

    return jax.vmap(one)(design.values, design.indices, design.entry_valid, values)


def design_row_norm_bound(design: Design, /) -> Array:
    """Squared row-norm upper bound, exact for dense rows and safe with sparse duplicates."""
    if not design.sparse:
        assert design.dense is not None
        return jnp.sum(jnp.abs(design.dense) ** 2, axis=-1)
    assert design.values is not None
    assert design.entry_valid is not None
    absolute = jnp.where(design.entry_valid, jnp.abs(design.values), 0.0)
    return jnp.sum(absolute, axis=-1) ** 2


def weighted_feature_gram(design: Design, weights: Array, /) -> Array:
    """Return exact per-output ``Xᴴ W X`` matrices without densifying sparse rows."""
    if not design.sparse:
        assert design.dense is not None
        return jnp.einsum(
            "cnf,cno,cng->cofg",
            jnp.conj(design.dense),
            weights,
            design.dense,
        )
    assert design.values is not None
    assert design.indices is not None
    assert design.entry_valid is not None

    def one_case(entries, indices, valid, case_weights):
        entries = jnp.where(valid, entries, 0)

        def one_output(weight):
            updates = (
                jnp.conj(entries)[:, :, None]
                * entries[:, None, :]
                * weight[:, None, None]
            )
            rows = jnp.broadcast_to(indices[:, :, None], updates.shape)
            columns = jnp.broadcast_to(indices[:, None, :], updates.shape)
            keep = valid[:, :, None] & valid[:, None, :]
            return (
                jnp.zeros((design.features, design.features), dtype=updates.dtype)
                .at[rows, columns]
                .add(jnp.where(keep, updates, 0))
            )

        return jax.vmap(one_output, in_axes=1, out_axes=0)(case_weights)

    return jax.vmap(one_case)(design.values, design.indices, design.entry_valid, weights)


def weighted_rank_condition(design: Design, weights: Array, /) -> tuple[Array, Array]:
    """Diagnose the minimum output rank and worst output condition of a design."""
    gram = weighted_feature_gram(design, weights)
    singular = jnp.linalg.svd(gram, compute_uv=False)
    largest = jnp.max(singular, axis=-1)
    design_cutoff = (
        max(design.samples, design.features) * jnp.finfo(jnp.real(gram).dtype).eps
    )
    retained = singular > largest[..., None] * design_cutoff**2
    rank = jnp.sum(retained, axis=-1, dtype=jnp.int32)
    smallest = jnp.min(jnp.where(retained, singular, jnp.inf), axis=-1)
    condition = jnp.where(
        rank > 0,
        jnp.sqrt(largest / jnp.maximum(smallest, jnp.finfo(jnp.real(gram).dtype).tiny)),
        jnp.inf,
    )
    return jnp.min(rank, axis=-1), jnp.max(condition, axis=-1)


def _reshape_coefficients(
    coefficients: Array,
    case_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> Array:
    return coefficients.reshape(case_shape + (coefficients.shape[-2],) + target_shape)


def _reshape_intercept(
    intercept: Array, case_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> Array:
    return intercept.reshape(case_shape + target_shape)


def linear_prediction(
    x: Any,
    coefficients: Array,
    intercept: Array,
    /,
    *,
    case_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> Array:
    """Evaluate case-aware dense or fixed-width sparse linear predictions."""
    features = int(coefficients.shape[len(case_shape)])
    outputs = _product(target_shape)
    cases = _product(case_shape)
    beta = coefficients.reshape((cases, features, outputs))
    bias = intercept.reshape((cases, outputs))

    if isinstance(x, SparseFeatures):
        if x.feature_count != features or x.case_shape != case_shape:
            raise ValueError(
                "Sparse prediction input must match fitted case and feature shapes."
            )
        raw = x.values.reshape((cases, x.sample_count, x.row_width))
        valid = x.columns.valid.reshape(raw.shape)
        entries = jnp.where(valid, raw, 0)
        indices = x.columns.source_indices.reshape(raw.shape)

        def one(case_entries, case_indices, case_valid, case_beta, case_bias):
            gathered = case_beta[case_indices]
            return (
                jnp.sum(
                    jnp.where(
                        case_valid[..., None],
                        case_entries[..., None] * gathered,
                        0,
                    ),
                    axis=1,
                )
                + case_bias
            )

        result = jax.vmap(one)(entries, indices, valid, beta, bias)
        return result.reshape(case_shape + (x.sample_count,) + target_shape)

    values = jnp.asarray(x)
    if values.ndim < 1 or int(values.shape[-1]) != features:
        raise ValueError(f"Expected final feature dimension {features}.")
    if tuple(int(size) for size in values.shape[: len(case_shape)]) != case_shape:
        raise ValueError("Prediction input must begin with the fitted case shape.")
    sample_shape = tuple(int(size) for size in values.shape[len(case_shape) : -1])
    values_cases = values.reshape((cases,) + sample_shape + (features,))
    result = jax.vmap(lambda a, b, c: jnp.einsum("...f,fo->...o", a, b) + c)(
        values_cases, beta, bias
    )
    return result.reshape(case_shape + sample_shape + target_shape)


class AbstractLinearModel(AbstractArrayModel):
    """Shared immutable affine state for linear model families."""

    coefficients: Array
    intercept: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        coefficients: Array,
        intercept: Array,
        /,
        *,
        case_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
    ):
        self.coefficients = _reshape_coefficients(
            jnp.asarray(coefficients), case_shape, target_shape
        )
        self.intercept = _reshape_intercept(
            jnp.asarray(intercept), case_shape, target_shape
        )
        self.case_shape = tuple(case_shape)
        self.target_shape = tuple(target_shape)
        self.in_size = int(self.coefficients.shape[len(case_shape)])
        self.out_size = target_shape if target_shape else "scalar"

    def linear_predictor(self, x: Any, /) -> Array:
        return linear_prediction(
            x,
            self.coefficients,
            self.intercept,
            case_shape=self.case_shape,
            target_shape=self.target_shape,
        )


class AbstractLinearRegressorModel(AbstractLinearModel):
    """Shared executable contract for affine regression models."""

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.linear_predictor(x)


class LinearRegressorModel(AbstractLinearRegressorModel):
    """Immutable generic affine multi-output regressor."""


class AbstractLinearScoreClassifierModel(AbstractLinearModel):
    """Shared binary linear classifier state and hard-label operations."""

    labels: Array

    def __init__(
        self,
        coefficients: Array,
        intercept: Array,
        labels: Array,
        /,
        *,
        case_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
    ):
        super().__init__(
            coefficients,
            intercept,
            case_shape=case_shape,
            target_shape=target_shape,
        )
        self.labels = jnp.asarray(labels)

    def decision_function(self, x: Any, /) -> Array:
        return self.linear_predictor(x)

    def predict_indices(self, x: Any, /) -> Array:
        return (self.decision_function(x) >= 0.0).astype(jnp.int32)

    def predict(self, x: Any, /) -> Array:
        return self.labels[self.predict_indices(x)]


class LinearScoreClassifierModel(AbstractLinearScoreClassifierModel):
    """Binary linear classifier whose call is its differentiable decision score."""

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.decision_function(x)


class LogisticClassifierModel(AbstractLinearScoreClassifierModel):
    """Binary/multilabel logistic model; calls return smooth positive-class probabilities."""

    def positive_probability(self, x: Any, /) -> Array:
        return jax.nn.sigmoid(self.decision_function(x))

    def predict_proba(self, x: Any, /) -> Array:
        positive = self.positive_probability(x)
        return jnp.stack((1.0 - positive, positive), axis=-1)

    def predict_log_proba(self, x: Any, /) -> Array:
        score = self.decision_function(x)
        return jnp.stack((jax.nn.log_sigmoid(-score), jax.nn.log_sigmoid(score)), axis=-1)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.positive_probability(x)


class MultinomialLogisticModel(AbstractArrayModel):
    """Identified multiclass softmax model with explicit hard prediction methods."""

    coefficients: Array
    intercept: Array
    labels: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        coefficients: Array,
        intercept: Array,
        labels: Array,
        /,
        *,
        case_shape: tuple[int, ...],
    ):
        self.coefficients = jnp.asarray(coefficients).reshape(
            case_shape + coefficients.shape[-2:]
        )
        self.intercept = jnp.asarray(intercept).reshape(
            case_shape + (coefficients.shape[-1],)
        )
        self.labels = jnp.asarray(labels)
        self.case_shape = tuple(case_shape)
        self.in_size = int(coefficients.shape[-2])
        self.out_size = int(coefficients.shape[-1])

    def decision_function(self, x: Any, /) -> Array:
        return linear_prediction(
            x,
            self.coefficients,
            self.intercept,
            case_shape=self.case_shape,
            target_shape=(self.out_size,),
        )

    def predict_log_proba(self, x: Any, /) -> Array:
        return jax.nn.log_softmax(self.decision_function(x), axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        return jax.nn.softmax(self.decision_function(x), axis=-1)

    def predict_indices(self, x: Any, /) -> Array:
        return jnp.argmax(self.decision_function(x), axis=-1)

    def predict(self, x: Any, /) -> Array:
        return self.labels[self.predict_indices(x)]

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class AbstractGeneralizedLinearModel(AbstractLinearModel):
    """Shared affine predictor with a declared smooth inverse link."""

    inverse_link: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: Array,
        intercept: Array,
        /,
        *,
        case_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        inverse_link: str,
    ):
        super().__init__(
            coefficients,
            intercept,
            case_shape=case_shape,
            target_shape=target_shape,
        )
        if inverse_link not in {"identity", "exp"}:
            raise ValueError(f"Unsupported inverse link {inverse_link!r}.")
        self.inverse_link = inverse_link

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        eta = self.linear_predictor(x)
        return eta if self.inverse_link == "identity" else jnp.exp(eta)


class GeneralizedLinearModel(AbstractGeneralizedLinearModel):
    """Generic generalized linear model."""


def binary_targets(
    prepared: PreparedBatch, schema: TargetSchema, /
) -> tuple[Array, Array, Array]:
    labels = jnp.asarray(schema.class_labels if schema.class_labels else (0, 1))
    if labels.shape != (2,):
        raise ValueError("Binary classification requires exactly two class labels.")
    raw = prepared.targets
    is_negative = raw == labels[0]
    is_positive = raw == labels[1]
    valid = jnp.all((is_negative | is_positive) | (prepared.weights == 0.0), axis=(1, 2))
    encoded = jnp.where(is_positive, 1.0, 0.0).astype(
        jnp.result_type(raw.dtype, jnp.float32)
    )
    return encoded, labels, valid


def multinomial_targets(
    prepared: PreparedBatch,
    schema: TargetSchema,
    /,
    *,
    num_classes: int | None,
) -> tuple[Array, Array, Array]:
    if prepared.target_shape:
        raise ValueError("Multinomial targets must be scalar class labels per sample.")
    if schema.class_labels:
        labels = jnp.asarray(schema.class_labels)
        classes = len(schema.class_labels)
        if num_classes is not None and int(num_classes) != classes:
            raise ValueError("num_classes conflicts with target_schema.class_labels.")
    else:
        if num_classes is None or int(num_classes) < 2:
            raise ValueError(
                "Multinomial fitting requires num_classes or target_schema.class_labels."
            )
        classes = int(num_classes)
        labels = jnp.arange(classes, dtype=jnp.int32)
    raw = prepared.targets[..., 0]
    matches = raw[..., None] == labels
    active = prepared.weights[..., 0] > 0.0
    valid = jnp.all(jnp.any(matches, axis=-1) | (~active), axis=-1)
    encoded = jnp.argmax(matches, axis=-1).astype(jnp.int32)
    return encoded, labels, valid


def iterative_fit(
    prepared: PreparedBatch,
    /,
    *,
    step,
    initial: tuple[Array, Array],
    max_iterations: int,
    tolerance: float,
    method: str,
    objective,
    model_factory,
    gradient_contract: GradientContract,
    extra_valid: Array | bool = True,
) -> FitResult:
    """Run a fixed differentiable optimization and package common diagnostics."""
    iteration = run_fixed_iterations(
        initial,
        step,
        max_iterations=max_iterations,
        tolerance=tolerance,
        method=method,
    )
    coefficients, intercept = iteration.value
    value = objective(coefficients, intercept)
    parameter_finite = jnp.all(_finite(coefficients), axis=(1, 2)) & jnp.all(
        _finite(intercept), axis=1
    )
    extra = jnp.broadcast_to(jnp.asarray(extra_valid, dtype=bool), parameter_finite.shape)
    valid = (
        prepared.data_valid
        & extra
        & parameter_finite
        & iteration.finite
        & iteration.converged
    )
    status = jnp.where(
        ~prepared.data_valid,
        prepared.data_status,
        jnp.where(
            ~extra,
            ML_INFEASIBLE,
            jnp.where(
                ~parameter_finite | ~iteration.finite,
                ML_NONFINITE,
                jnp.where(iteration.converged, ML_SUCCESS, ML_NONCONVERGED),
            ),
        ),
    ).astype(jnp.int32)
    rank, condition = weighted_rank_condition(prepared.design, prepared.weights)
    valid_cases = restore_case_shape(prepared, valid)
    status_cases = restore_case_shape(prepared, status)
    diagnostics = FitDiagnostics(
        valid=valid_cases,
        status=status_cases,
        objective=restore_case_shape(prepared, value),
        iterations=restore_case_shape(prepared, iteration.iterations),
        effective_samples=restore_case_shape(prepared, prepared.effective_samples),
        rank=restore_case_shape(prepared, rank),
        condition=restore_case_shape(prepared, condition),
        method=method,
    )
    model = model_factory(coefficients, intercept)
    return FitResult(
        model,
        diagnostics,
        valid=valid_cases,
        status=status_cases,
        method=method,
        gradient_contract=gradient_contract,
    )


def unrolled_contract(
    *,
    prediction_inputs: str = "smooth",
    nonsmooth: bool = False,
    fit_targets: str | None = None,
    hard_outputs: tuple[str, ...] = (),
) -> GradientContract:
    level = "almost-everywhere" if nonsmooth else "smooth"
    return GradientContract(
        prediction_inputs=prediction_inputs,  # type: ignore[arg-type]
        prediction_parameters=level,
        fit_features=level,
        fit_targets=level if fit_targets is None else fit_targets,  # type: ignore[arg-type]
        fit_weights="conditional",
        fit_hyperparameters=level,
        fit_mode="unrolled",
        nondifferentiable_outputs=hard_outputs,
        conditions=(
            "Masks, sparse index structure, and iteration count are fixed.",
            "Reported validity and convergence statuses are nondifferentiable.",
        ),
    )


__all__ = [
    "AbstractGeneralizedLinearModel",
    "AbstractLinearRegressorModel",
    "AbstractLinearScoreClassifierModel",
    "Design",
    "GeneralizedLinearModel",
    "LinearRegressorModel",
    "LinearScoreClassifierModel",
    "LogisticClassifierModel",
    "MultinomialLogisticModel",
    "PreparedBatch",
    "binary_targets",
    "design_matmul",
    "design_row_norm_bound",
    "design_transpose_matmul",
    "iterative_fit",
    "linear_prediction",
    "multinomial_targets",
    "prepare_supervised",
    "parameter_dtype",
    "unrolled_contract",
    "weighted_feature_gram",
    "weighted_rank_condition",
]
