#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._model import AbstractArrayModel
from .._batch import MLBatch
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._schema import FeatureSchema, TargetSchema
from ._hard import _initial_score, _prepare_batch, TreeFitDiagnostics
from ._representation import (
    _tree_output_shape,
    apply_objective,
    ObjectiveTransform,
    TreeEnsemble,
)


TemperatureSchedule: TypeAlias = Literal["constant", "linear", "geometric"]
SoftObjective: TypeAlias = Literal["squared_error", "logistic", "softmax"]

_SOFT_CONTRACT = GradientContract(
    prediction_inputs="smooth",
    prediction_parameters="smooth",
    fit_features="conditional",
    fit_targets="conditional",
    fit_weights="conditional",
    fit_hyperparameters="conditional",
    fit_mode="unrolled",
    nondifferentiable_outputs=("hardened structure", "hardened feature choices"),
    conditions=(
        "All scheduled temperatures are finite and strictly positive.",
        "Hardening creates a disconnected hard model; no straight-through gradient is used.",
    ),
)


def _soft_tree_values(
    points: Array,
    feature_logits: Array,
    threshold: Array,
    missing_left_logit: Array,
    leaf_value: Array,
    temperature: Array,
    depth: int,
) -> Array:
    """Evaluate every complete soft tree for a matrix of points."""
    temperature = eqx.error_if(
        temperature,
        ~jnp.isfinite(temperature) | (temperature <= 0.0),
        "Soft-tree temperature must be finite and strictly positive.",
    )
    selection = jax.nn.softmax(feature_logits, axis=-1)
    finite = jnp.isfinite(points)
    safe_points = jnp.where(finite, points, 0.0)
    feature_gate = jax.nn.sigmoid(
        (threshold[None, ...] - safe_points[:, None, None, :]) / temperature
    )
    missing_probability = jax.nn.sigmoid(missing_left_logit)[None, ..., None]
    feature_gate = jnp.where(finite[:, None, None, :], feature_gate, missing_probability)
    left_probability = jnp.sum(selection[None, ...] * feature_gate, axis=-1)
    probability = jnp.ones(
        (points.shape[0], feature_logits.shape[0], 1), dtype=leaf_value.real.dtype
    )
    for level in range(depth):
        start = 2**level - 1
        stop = 2 ** (level + 1) - 1
        gate = left_probability[:, :, start:stop]
        probability = jnp.stack(
            (probability * gate, probability * (1.0 - gate)), axis=-1
        ).reshape((points.shape[0], feature_logits.shape[0], -1))
    return ein.contract("ptl,tlo->pto", probability, leaf_value)


def _soft_predict_case(
    points: Array,
    feature_logits: Array,
    threshold: Array,
    missing_left_logit: Array,
    leaf_value: Array,
    tree_weight: Array,
    base_score: Array,
    temperature: Array,
    depth: int,
) -> tuple[Array, Array]:
    trees = _soft_tree_values(
        points,
        feature_logits,
        threshold,
        missing_left_logit,
        leaf_value,
        temperature,
        depth,
    )
    return base_score + jnp.sum(trees * tree_weight[None, :, None], axis=1), trees


class _AbstractSoftTree(AbstractArrayModel):
    """Shared differentiable complete-tree representation for the three soft families."""

    feature_logits: Array
    threshold: Array
    missing_left_logit: Array
    leaf_value: Array
    tree_weight: Array
    base_score: Array
    temperature: Array
    feature_schema: FeatureSchema = eqx.field(static=True)
    target_schema: TargetSchema = eqx.field(static=True)
    objective_transform: ObjectiveTransform = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        feature_logits: ArrayLike,
        threshold: ArrayLike,
        missing_left_logit: ArrayLike,
        leaf_value: ArrayLike,
        tree_weight: ArrayLike,
        base_score: ArrayLike,
        temperature: ArrayLike,
        feature_schema: FeatureSchema,
        target_schema: TargetSchema | None = None,
        objective_transform: ObjectiveTransform = "identity",
        case_shape: tuple[int, ...] = (),
        out_size: int | tuple[int, ...] | Literal["scalar"] | None = None,
    ):
        case_shape_ = tuple(int(size) for size in case_shape)
        logits = jnp.asarray(feature_logits)
        if not jnp.issubdtype(logits.dtype, jnp.inexact):
            logits = logits.astype(jnp.float32)
        if logits.ndim != len(case_shape_) + 3:
            raise ValueError(
                "feature_logits must have shape case_shape + (tree, internal_node, feature)."
            )
        if tuple(logits.shape[: len(case_shape_)]) != case_shape_:
            raise ValueError("feature_logits does not begin with case_shape.")
        tree_count, internal_count, feature_count = map(int, logits.shape[-3:])
        depth = int(math.log2(internal_count + 1))
        if internal_count != 2**depth - 1 or depth <= 0:
            raise ValueError(
                "Soft trees require a complete positive-depth internal structure."
            )
        if len(feature_schema.names) != feature_count:
            raise ValueError("feature_schema must align with feature logits.")
        threshold_ = jnp.asarray(threshold, dtype=logits.dtype)
        if threshold_.shape != logits.shape:
            raise ValueError("threshold must match feature_logits.")
        missing_ = jnp.asarray(missing_left_logit, dtype=logits.dtype)
        expected_missing = case_shape_ + (tree_count, internal_count)
        if missing_.shape != expected_missing:
            raise ValueError(
                "missing_left_logit must have shape case_shape + (tree, internal_node)."
            )
        leaf_ = jnp.asarray(leaf_value)
        if not jnp.issubdtype(leaf_.dtype, jnp.inexact):
            leaf_ = leaf_.astype(jnp.float32)
        leaf_count = 2**depth
        if leaf_.ndim != len(case_shape_) + 3 or leaf_.shape[-3:-1] != (
            tree_count,
            leaf_count,
        ):
            raise ValueError(
                "leaf_value must have shape case_shape + (tree, leaf, output)."
            )
        output_count = int(leaf_.shape[-1])
        inferred_out = "scalar" if output_count == 1 else output_count
        out_size_ = inferred_out if out_size is None else out_size
        expected_outputs = (
            1 if out_size_ == "scalar" else math.prod(_tree_output_shape(out_size_))
        )
        if expected_outputs != output_count:
            raise ValueError("out_size does not match leaf outputs.")
        tree_weight_ = jnp.asarray(tree_weight, dtype=leaf_.real.dtype)
        if tree_weight_.shape != case_shape_ + (tree_count,):
            raise ValueError("tree_weight must have shape case_shape + (tree,).")
        base_ = jnp.asarray(base_score, dtype=leaf_.dtype)
        if base_.shape != case_shape_ + (output_count,):
            raise ValueError("base_score must have shape case_shape + (output,).")
        temperature_ = jnp.asarray(temperature, dtype=logits.real.dtype)
        if temperature_.shape not in {(), case_shape_}:
            raise ValueError("temperature must be scalar or have case_shape.")
        if isinstance(temperature, (int, float)) and (
            not math.isfinite(float(temperature)) or float(temperature) <= 0.0
        ):
            raise ValueError("temperature must be finite and positive.")
        if objective_transform not in {"identity", "sigmoid", "softmax"}:
            raise ValueError(
                "Soft trees support identity, sigmoid, and softmax transforms."
            )
        if objective_transform == "sigmoid" and output_count != 1:
            raise ValueError("A sigmoid soft-tree objective requires one raw output.")
        if objective_transform == "softmax" and output_count < 2:
            raise ValueError(
                "A softmax soft-tree objective requires at least two outputs."
            )
        self.feature_logits = logits
        self.threshold = threshold_
        self.missing_left_logit = missing_
        self.leaf_value = leaf_
        self.tree_weight = tree_weight_
        self.base_score = base_
        self.temperature = temperature_
        self.feature_schema = feature_schema
        self.target_schema = TargetSchema() if target_schema is None else target_schema
        self.objective_transform = objective_transform
        self.case_shape = case_shape_
        self.depth = depth
        self.in_size = feature_count
        self.out_size = out_size_

    @property
    def class_schema(self) -> TargetSchema:
        """Return the class/target vocabulary and semantics carried by the model."""
        return self.target_schema

    @property
    def tree_count(self) -> int:
        return int(self.feature_logits.shape[-3])

    @property
    def internal_node_count(self) -> int:
        return int(self.feature_logits.shape[-2])

    @property
    def leaf_count(self) -> int:
        return int(self.leaf_value.shape[-2])

    @property
    def output_count(self) -> int:
        return int(self.leaf_value.shape[-1])

    def _evaluate(self, x: Any, /) -> tuple[Array, Array, tuple[int, ...]]:
        values = jnp.asarray(x)
        if values.shape[-1:] != (self.in_size,):
            raise ValueError(f"Expected final feature axis of size {self.in_size}.")
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Sigmoid tree gates require real features.")
        parameters_finite = (
            jnp.all(jnp.isfinite(self.feature_logits))
            & jnp.all(jnp.isfinite(self.threshold))
            & jnp.all(jnp.isfinite(self.missing_left_logit))
            & jnp.all(jnp.isfinite(self.leaf_value))
            & jnp.all(jnp.isfinite(self.tree_weight))
            & jnp.all(jnp.isfinite(self.base_score))
        )
        values = eqx.error_if(
            values, ~parameters_finite, "Soft-tree parameters must all be finite."
        )
        case_count = math.prod(self.case_shape) if self.case_shape else 1
        arrays = (
            self.feature_logits.reshape((case_count,) + self.feature_logits.shape[-3:]),
            self.threshold.reshape((case_count,) + self.threshold.shape[-3:]),
            self.missing_left_logit.reshape(
                (case_count,) + self.missing_left_logit.shape[-2:]
            ),
            self.leaf_value.reshape((case_count,) + self.leaf_value.shape[-3:]),
            self.tree_weight.reshape((case_count, self.tree_count)),
            self.base_score.reshape((case_count, self.output_count)),
            jnp.broadcast_to(self.temperature, self.case_shape or ()).reshape(
                (case_count,)
            ),
        )
        if self.case_shape:
            if tuple(values.shape[: len(self.case_shape)]) != self.case_shape:
                raise ValueError(
                    "Case-dependent soft trees require inputs beginning with case_shape."
                )
            point_shape = tuple(values.shape[len(self.case_shape) : -1])
            points = values.reshape((case_count, -1, self.in_size))
            raw, trees = jax.vmap(
                lambda pts, *params: _soft_predict_case(pts, *params, depth=self.depth)
            )(points, *arrays)
            lead_shape = self.case_shape + point_shape
        else:
            point_shape = tuple(values.shape[:-1])
            points = values.reshape((-1, self.in_size))
            raw, trees = _soft_predict_case(
                points, *(array[0] for array in arrays), depth=self.depth
            )
            lead_shape = point_shape
        return raw, trees, lead_shape

    def predict_raw(self, x: Any, /) -> Array:
        raw, _, lead_shape = self._evaluate(x)
        if self.out_size == "scalar":
            return raw.reshape(lead_shape + (1,))[..., 0]
        return raw.reshape(lead_shape + _tree_output_shape(self.out_size))

    def predict_trees(self, x: Any, /) -> Array:
        _, trees, lead_shape = self._evaluate(x)
        return trees.reshape(
            lead_shape + (self.tree_count,) + _tree_output_shape(self.out_size)
        )

    def feature_probabilities(self, /) -> Array:
        """Return differentiable feature-selection probabilities for every gate."""
        return jax.nn.softmax(self.feature_logits, axis=-1)

    def harden(self, /) -> TreeEnsemble:
        """Create an independent hard axis-aligned model with no straight-through path."""
        logits = jax.lax.stop_gradient(self.feature_logits)
        thresholds = jax.lax.stop_gradient(self.threshold)
        leaves = jax.lax.stop_gradient(self.leaf_value)
        missing = jax.lax.stop_gradient(self.missing_left_logit)
        tree_weight = jax.lax.stop_gradient(self.tree_weight)
        base = jax.lax.stop_gradient(self.base_score)
        feature = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        selected_threshold = jnp.take_along_axis(thresholds, feature[..., None], axis=-1)[
            ..., 0
        ]
        node_capacity = 2 ** (self.depth + 1) - 1
        internal = self.internal_node_count
        shape = self.case_shape + (self.tree_count, node_capacity)
        feature_nodes = (
            jnp.full(shape, -1, dtype=jnp.int32).at[..., :internal].set(feature)
        )
        threshold_nodes = (
            jnp.zeros(shape, dtype=thresholds.dtype)
            .at[..., :internal]
            .set(selected_threshold)
        )
        indices = jnp.arange(node_capacity, dtype=jnp.int32)
        left = jnp.broadcast_to(2 * indices + 1, shape)
        right = jnp.broadcast_to(2 * indices + 2, shape)
        child_valid = indices < internal
        left = jnp.where(child_valid, left, -1)
        right = jnp.where(child_valid, right, -1)
        default_left = jnp.zeros(shape, dtype=bool).at[..., :internal].set(missing >= 0.0)
        leaf_nodes = (
            jnp.zeros(shape + (self.output_count,), dtype=leaves.dtype)
            .at[..., internal:, :]
            .set(leaves)
        )
        return TreeEnsemble(
            feature_index=feature_nodes,
            threshold=threshold_nodes,
            left_child=left,
            right_child=right,
            default_left=default_left,
            leaf_value=leaf_nodes,
            node_mask=jnp.ones(shape, dtype=bool),
            leaf_mask=jnp.broadcast_to(indices >= internal, shape),
            tree_mask=jnp.ones(self.case_shape + (self.tree_count,), dtype=bool),
            tree_weight=tree_weight,
            base_score=base,
            feature_schema=self.feature_schema,
            target_schema=self.target_schema,
            objective_transform=self.objective_transform,
            case_shape=self.case_shape,
            out_size=self.out_size,
            max_steps=self.depth + 1,
            capacity_exhausted=False,
        )

    def predict_labels(self, x: Any, /, *, threshold: float = 0.5) -> Array:
        """Return nondifferentiable class indices from smooth class probabilities."""
        prediction = self(x)
        if self.objective_transform == "sigmoid":
            return (prediction >= threshold).astype(jnp.int32)
        if self.objective_transform == "softmax":
            return jnp.argmax(prediction, axis=-1).astype(jnp.int32)
        raise ValueError("Class labels require sigmoid or softmax soft-tree outputs.")

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return apply_objective(self.predict_raw(x), self.objective_transform)


class SoftDecisionTree(_AbstractSoftTree):
    """One differentiable decision tree with probabilistic routing gates."""


class SoftRandomForest(_AbstractSoftTree):
    """Averaged differentiable trees trained jointly from randomized initialization."""


class SoftGradientBoostedTrees(_AbstractSoftTree):
    """Differentiable additive soft-tree model with learned leaf increments."""


def _temperature_at(
    step: Array,
    iterations: int,
    initial: Array,
    final: Array,
    schedule: TemperatureSchedule,
) -> Array:
    fraction = step.astype(jnp.asarray(initial).dtype) / max(iterations - 1, 1)
    if schedule == "constant":
        return jnp.asarray(initial)
    if schedule == "linear":
        return initial + fraction * (final - initial)
    if schedule == "geometric":
        return initial * jnp.exp(fraction * jnp.log(final / initial))
    raise ValueError(f"Unsupported temperature schedule {schedule!r}.")


def _fit_soft_case(
    x: Array,
    y: Array,
    weight: Array,
    sample_mask: Array,
    *,
    key: Array,
    tree_count: int,
    depth: int,
    iterations: int,
    learning_rate: Array,
    objective: SoftObjective,
    initial_temperature: Array,
    final_temperature: Array,
    temperature_schedule: TemperatureSchedule,
    sparsity: Array,
    tree_weight: Array,
) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
    learning_rate = eqx.error_if(
        learning_rate,
        ~jnp.isfinite(learning_rate) | (learning_rate <= 0.0),
        "Soft-tree learning_rate must be finite and positive.",
    )
    initial_temperature = eqx.error_if(
        initial_temperature,
        ~jnp.isfinite(initial_temperature) | (initial_temperature <= 0.0),
        "Soft-tree initial_temperature must be finite and positive.",
    )
    final_temperature = eqx.error_if(
        final_temperature,
        ~jnp.isfinite(final_temperature) | (final_temperature <= 0.0),
        "Soft-tree final_temperature must be finite and positive.",
    )
    sparsity = eqx.error_if(
        sparsity,
        ~jnp.isfinite(sparsity) | (sparsity < 0.0),
        "Soft-tree sparsity must be finite and nonnegative.",
    )
    tree_weight = eqx.error_if(
        tree_weight,
        jnp.any(~jnp.isfinite(tree_weight) | (tree_weight <= 0.0)),
        "Soft-tree ensemble weights must be finite and positive.",
    )
    internal = 2**depth - 1
    leaves = 2**depth
    feature_count = int(x.shape[-1])
    output_count = int(y.shape[-1])
    logits_key, threshold_key, leaf_key = jax.random.split(key, 3)
    logits = 0.05 * jax.random.normal(
        logits_key, (tree_count, internal, feature_count), dtype=x.dtype
    )
    finite = jnp.isfinite(x)
    feature_weight = finite * weight[:, None]
    feature_denominator = jnp.maximum(
        jnp.sum(feature_weight, axis=0), jnp.finfo(weight.dtype).tiny
    )
    feature_mean = (
        jnp.sum(jnp.where(finite, x, 0.0) * weight[:, None], axis=0) / feature_denominator
    )
    feature_scale = jnp.sqrt(
        jnp.sum(jnp.where(finite, (x - feature_mean) ** 2, 0.0) * weight[:, None], axis=0)
        / feature_denominator
    )
    thresholds = feature_mean + 0.1 * jnp.maximum(
        feature_scale, 1e-3
    ) * jax.random.normal(
        threshold_key, (tree_count, internal, feature_count), dtype=x.dtype
    )
    leaf_values = 0.01 * jax.random.normal(
        leaf_key, (tree_count, leaves, output_count), dtype=y.dtype
    )
    missing_logits = jnp.zeros((tree_count, internal), dtype=x.dtype)
    base = _initial_score(objective, y, weight)
    parameters = (logits, thresholds, missing_logits, leaf_values, base)
    normalized_weight = jnp.where(sample_mask, weight, 0.0)
    denominator = jnp.maximum(
        jnp.sum(normalized_weight), jnp.finfo(normalized_weight.dtype).tiny
    )

    def loss_function(params, temperature):
        logits_, thresholds_, missing_, leaves_, base_ = params
        raw, _ = _soft_predict_case(
            x,
            logits_,
            thresholds_,
            missing_,
            leaves_,
            tree_weight,
            base_,
            temperature,
            depth,
        )
        if objective == "squared_error":
            residual = raw - y
            data_loss = (
                0.5
                * jnp.sum(
                    normalized_weight[:, None] * jnp.real(residual * jnp.conj(residual))
                )
                / denominator
            )
        elif objective == "logistic":
            data_loss = (
                -jnp.sum(
                    normalized_weight[:, None]
                    * (y * jax.nn.log_sigmoid(raw) + (1.0 - y) * jax.nn.log_sigmoid(-raw))
                )
                / denominator
            )
        elif objective == "softmax":
            data_loss = (
                -jnp.sum(
                    normalized_weight[:, None] * y * jax.nn.log_softmax(raw, axis=-1)
                )
                / denominator
            )
        else:
            raise ValueError(f"Unsupported soft objective {objective!r}.")
        selection = jax.nn.softmax(logits_, axis=-1)
        sparsity_penalty = jnp.mean(selection * (1.0 - selection))
        return data_loss + sparsity * sparsity_penalty

    def update(params, step):
        temperature = _temperature_at(
            step,
            iterations,
            initial_temperature,
            final_temperature,
            temperature_schedule,
        )
        loss, gradients = jax.value_and_grad(loss_function)(params, temperature)
        updated = jax.tree.map(
            lambda value, gradient: value - learning_rate * gradient, params, gradients
        )
        return updated, loss

    parameters, losses = jax.lax.scan(update, parameters, jnp.arange(iterations))
    return parameters, losses


class _AbstractSoftTreeRecipe(AbstractRecipe):
    tree_count: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    learning_rate: Array
    objective: SoftObjective = eqx.field(static=True)
    initial_temperature: Array
    final_temperature: Array
    temperature_schedule: TemperatureSchedule = eqx.field(static=True)
    sparsity: Array
    ensemble_kind: Literal["tree", "forest", "boosted"] = eqx.field(static=True)
    tree_learning_rate: Array
    num_classes: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        tree_count: int,
        ensemble_kind: Literal["tree", "forest", "boosted"],
        depth: int = 3,
        iterations: int = 200,
        learning_rate: ArrayLike = 1e-2,
        objective: SoftObjective = "squared_error",
        initial_temperature: ArrayLike = 1.0,
        final_temperature: ArrayLike = 0.1,
        temperature_schedule: TemperatureSchedule = "geometric",
        sparsity: ArrayLike = 0.0,
        tree_learning_rate: ArrayLike = 0.1,
        num_classes: int | None = None,
    ):
        if tree_count <= 0 or depth <= 0 or iterations <= 0:
            raise ValueError("Soft tree count, depth, and iterations must be positive.")
        if isinstance(learning_rate, (int, float)) and learning_rate <= 0.0:
            raise ValueError("Soft-tree learning_rate must be positive.")
        if isinstance(tree_learning_rate, (int, float)) and tree_learning_rate <= 0.0:
            raise ValueError("Soft-tree tree_learning_rate must be positive.")
        for name, value in (
            ("initial_temperature", initial_temperature),
            ("final_temperature", final_temperature),
        ):
            if isinstance(value, (int, float)) and (
                not math.isfinite(float(value)) or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and strictly positive.")
        if temperature_schedule not in {"constant", "linear", "geometric"}:
            raise ValueError("Unsupported temperature schedule.")
        if (
            temperature_schedule == "constant"
            and isinstance(initial_temperature, (int, float))
            and isinstance(final_temperature, (int, float))
            and initial_temperature != final_temperature
        ):
            raise ValueError("A constant schedule requires equal temperature endpoints.")
        if objective not in {"squared_error", "logistic", "softmax"}:
            raise ValueError("Unsupported soft-tree objective.")
        if isinstance(sparsity, (int, float)) and sparsity < 0.0:
            raise ValueError("sparsity must be nonnegative.")
        self.tree_count = int(tree_count)
        self.depth = int(depth)
        self.iterations = int(iterations)
        self.learning_rate = jnp.asarray(learning_rate)
        self.objective = objective
        self.initial_temperature = jnp.asarray(initial_temperature)
        self.final_temperature = jnp.asarray(final_temperature)
        self.temperature_schedule = temperature_schedule
        self.sparsity = jnp.asarray(sparsity)
        self.ensemble_kind = ensemble_kind
        self.tree_learning_rate = jnp.asarray(tree_learning_rate)
        self.num_classes = num_classes

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("Soft tree fitting requires an explicit JAX key.")
        if any(kind in {"categorical", "boolean"} for kind in batch.feature_schema.kinds):
            raise ValueError(
                "Soft sigmoid gates support continuous and ordinal schemas only; "
                "categorical relaxation must be performed explicitly before fitting."
            )
        classification = self.objective in {"logistic", "softmax"}
        x, y, weight, sample_mask, output_count, output_shape, target_schema = (
            _prepare_batch(
                batch, classification=classification, num_classes=self.num_classes
            )
        )
        if self.objective == "logistic":
            if output_count != 2:
                raise ValueError("Soft logistic trees require two classes.")
            y = y[..., 1:]
            output_count = 1
            output_shape = ()
        if (
            jnp.issubdtype(y.dtype, jnp.complexfloating)
            and self.objective != "squared_error"
        ):
            raise TypeError("Only soft squared-error trees support complex targets.")
        case_count = int(x.shape[0])
        case_keys = jax.random.split(key, case_count)
        if self.ensemble_kind == "forest":
            tree_weight_single = jnp.full(
                (self.tree_count,), 1.0 / self.tree_count, dtype=x.dtype
            )
        elif self.ensemble_kind == "boosted":
            tree_weight_single = jnp.full(
                (self.tree_count,), self.tree_learning_rate, dtype=x.dtype
            )
        else:
            tree_weight_single = jnp.ones((1,), dtype=x.dtype)
        parameter_cases = []
        final_losses = []
        for case in range(case_count):
            parameters, losses = _fit_soft_case(
                x[case],
                y[case],
                weight[case],
                sample_mask[case],
                key=case_keys[case],
                tree_count=self.tree_count,
                depth=self.depth,
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                objective=self.objective,
                initial_temperature=self.initial_temperature,
                final_temperature=self.final_temperature,
                temperature_schedule=self.temperature_schedule,
                sparsity=self.sparsity,
                tree_weight=tree_weight_single,
            )
            parameter_cases.append(parameters)
            final_losses.append(losses[-1])
        stacked = [
            jnp.stack([case[index] for case in parameter_cases]) for index in range(5)
        ]
        logits, thresholds, missing, leaves, base = stacked
        logits = logits.reshape(batch.case_shape + logits.shape[1:])
        thresholds = thresholds.reshape(batch.case_shape + thresholds.shape[1:])
        missing = missing.reshape(batch.case_shape + missing.shape[1:])
        leaves = leaves.reshape(batch.case_shape + leaves.shape[1:])
        base = base.reshape(batch.case_shape + (output_count,))
        tree_weight = jnp.broadcast_to(
            tree_weight_single, batch.case_shape + (self.tree_count,)
        )
        transform: ObjectiveTransform = (
            "sigmoid"
            if self.objective == "logistic"
            else ("softmax" if self.objective == "softmax" else "identity")
        )
        out_size = (
            output_shape
            if output_shape
            else (output_count if output_count > 1 else "scalar")
        )
        model_type = {
            "tree": SoftDecisionTree,
            "forest": SoftRandomForest,
            "boosted": SoftGradientBoostedTrees,
        }[self.ensemble_kind]
        model = model_type(
            feature_logits=logits,
            threshold=thresholds,
            missing_left_logit=missing,
            leaf_value=leaves,
            tree_weight=tree_weight,
            base_score=base,
            temperature=jnp.asarray(self.final_temperature, dtype=x.dtype),
            feature_schema=batch.feature_schema,
            target_schema=target_schema,
            objective_transform=transform,
            case_shape=batch.case_shape,
            out_size=out_size,
        )
        finite = jnp.stack([jnp.isfinite(loss) for loss in final_losses]).reshape(
            batch.case_shape
        )
        effective = jnp.sum(sample_mask & (weight > 0.0), axis=-1).reshape(
            batch.case_shape
        )
        enough = effective > 0
        valid = finite & enough
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(enough, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        )
        diagnostics = TreeFitDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.stack(final_losses).reshape(batch.case_shape),
            iterations=jnp.full(batch.case_shape, self.iterations),
            effective_samples=effective,
            trees_built=jnp.full(batch.case_shape, self.tree_count),
            nodes_used=jnp.full(
                batch.case_shape, self.tree_count * (2 ** (self.depth + 1) - 1)
            ),
            leaves_used=jnp.full(batch.case_shape, self.tree_count * 2**self.depth),
            capacity_exhausted=jnp.zeros(batch.case_shape, dtype=bool),
            converged=valid,
            method=f"soft_{self.ensemble_kind}",
            split_search="relaxed",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method=f"soft_{self.ensemble_kind}",
            gradient_contract=_SOFT_CONTRACT,
        )


class SoftDecisionTreeRecipe(_AbstractSoftTreeRecipe):
    def __init__(self, **kwargs):
        super().__init__(tree_count=1, ensemble_kind="tree", **kwargs)


class SoftRandomForestRecipe(_AbstractSoftTreeRecipe):
    def __init__(self, *, n_estimators: int = 32, **kwargs):
        super().__init__(tree_count=n_estimators, ensemble_kind="forest", **kwargs)


class SoftGradientBoostedTreesRecipe(_AbstractSoftTreeRecipe):
    def __init__(self, *, n_estimators: int = 32, **kwargs):
        super().__init__(tree_count=n_estimators, ensemble_kind="boosted", **kwargs)


__all__ = [
    "SoftDecisionTree",
    "SoftDecisionTreeRecipe",
    "SoftGradientBoostedTrees",
    "SoftGradientBoostedTreesRecipe",
    "SoftObjective",
    "SoftRandomForest",
    "SoftRandomForestRecipe",
    "TemperatureSchedule",
]
