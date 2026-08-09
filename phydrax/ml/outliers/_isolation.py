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
from .._batch import MLBatch
from .._contracts import AbstractRecipe, FitResult, GradientContract
from ._common import (
    _BLOCKWISE_BINDING,
    _case_count,
    _fit_arrays,
    _fit_status,
    _prepare_queries,
    _restore_scores,
    _score_bounds,
    _weighted_threshold,
    OutlierDiagnostics,
)


_EULER_GAMMA = 0.5772156649015329


def _average_path_adjustment(mass: Array) -> Array:
    mass = jnp.asarray(mass)
    safe = jnp.maximum(mass, 2.0)
    approximation = 2.0 * (jnp.log(safe - 1.0) + _EULER_GAMMA) - 2.0 * (safe - 1.0) / safe
    return jnp.where(mass <= 1.0, 0.0, jnp.where(mass == 2.0, 1.0, approximation))


def _build_tree_one(
    x: Array,
    weights: Array,
    active: Array,
    key: Array,
    max_depth: int,
) -> tuple[Array, Array, Array, Array]:
    feature_count = int(x.shape[-1])
    node_count = 2 ** (max_depth + 1) - 1
    internal_count = 2**max_depth - 1
    membership = jnp.zeros((node_count, x.shape[0]), dtype=bool).at[0].set(active)
    features = jnp.zeros((node_count,), dtype=jnp.int32)
    thresholds = jnp.zeros((node_count,), dtype=x.dtype)
    splittable = jnp.zeros((node_count,), dtype=bool)
    keys = jax.random.split(key, internal_count * 2)

    for node in range(internal_count):
        members = membership[node]
        feature = jax.random.randint(
            keys[2 * node], (), 0, feature_count, dtype=jnp.int32
        )
        values = x[:, feature]
        minimum = jnp.min(jnp.where(members, values, jnp.inf))
        maximum = jnp.max(jnp.where(members, values, -jnp.inf))
        fraction = jax.random.uniform(keys[2 * node + 1], (), dtype=x.dtype)
        threshold = minimum + fraction * (maximum - minimum)
        can_split = (
            (jnp.sum(members, dtype=jnp.int32) > 1)
            & jnp.isfinite(minimum)
            & jnp.isfinite(maximum)
            & (maximum > minimum)
        )
        left_members = members & (values < threshold)
        right_members = members & ~left_members
        can_split = can_split & jnp.any(left_members) & jnp.any(right_members)
        left = 2 * node + 1
        right = left + 1
        membership = membership.at[left].set(jnp.where(can_split, left_members, False))
        membership = membership.at[right].set(jnp.where(can_split, right_members, False))
        features = features.at[node].set(feature)
        thresholds = thresholds.at[node].set(jnp.where(can_split, threshold, 0.0))
        splittable = splittable.at[node].set(can_split)

    active_count = jnp.sum(active)
    mean_weight = jnp.sum(weights) / jnp.maximum(active_count, 1)
    normalized_weights = weights / jnp.maximum(mean_weight, jnp.finfo(float).tiny)
    leaf_mass = jnp.sum(membership * normalized_weights[None, :], axis=-1)
    return features, thresholds, splittable, leaf_mass


def _hard_tree_path(
    point: Array,
    features: Array,
    thresholds: Array,
    splittable: Array,
    leaf_mass: Array,
    max_depth: int,
) -> Array:
    def descend(_depth, state):
        node, depth, done = state
        split = splittable[node] & ~done
        go_left = point[features[node]] < thresholds[node]
        child = jnp.where(go_left, 2 * node + 1, 2 * node + 2)
        node = jnp.where(split, child, node)
        depth = depth + split.astype(jnp.int32)
        done = done | ~splittable[node]
        return node, depth, done

    node, depth, _done = jax.lax.fori_loop(
        0,
        max_depth,
        descend,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        ),
    )
    return depth.astype(point.dtype) + _average_path_adjustment(leaf_mass[node])


def _hard_forest_scores_one(
    queries: Array,
    features: Array,
    thresholds: Array,
    splittable: Array,
    leaf_mass: Array,
    normalization: Array,
    max_depth: int,
) -> Array:
    def score_point(point):
        paths = jax.vmap(
            lambda f_, t_, s_, m_: _hard_tree_path(point, f_, t_, s_, m_, max_depth)
        )(features, thresholds, splittable, leaf_mass)
        return jnp.exp2(
            -jnp.mean(paths) / jnp.maximum(normalization, jnp.finfo(float).tiny)
        )

    return jax.vmap(score_point)(queries)


def _smooth_tree_path(
    point: Array,
    features: Array,
    thresholds: Array,
    splittable: Array,
    leaf_mass: Array,
    max_depth: int,
    temperature: float,
) -> Array:
    node_count = int(features.shape[0])
    internal_count = 2**max_depth - 1
    probabilities = jnp.zeros((node_count,), dtype=point.dtype).at[0].set(1.0)
    expected = jnp.asarray(0.0, dtype=point.dtype)
    for node in range(internal_count):
        probability = probabilities[node]
        depth = (node + 1).bit_length() - 1
        split = splittable[node].astype(point.dtype)
        terminal_path = depth + _average_path_adjustment(leaf_mass[node])
        expected = expected + probability * (1.0 - split) * terminal_path
        gate = jax.nn.sigmoid(
            (thresholds[node] - point[features[node]]) / float(temperature)
        )
        left = 2 * node + 1
        right = left + 1
        probabilities = probabilities.at[left].add(probability * split * gate)
        probabilities = probabilities.at[right].add(probability * split * (1.0 - gate))
    leaf_start = internal_count
    expected = expected + jnp.sum(
        probabilities[leaf_start:]
        * (max_depth + _average_path_adjustment(leaf_mass[leaf_start:]))
    )
    return expected


def _smooth_forest_scores_one(
    queries: Array,
    features: Array,
    thresholds: Array,
    splittable: Array,
    leaf_mass: Array,
    normalization: Array,
    max_depth: int,
    temperature: float,
) -> Array:
    def score_point(point):
        paths = jax.vmap(
            lambda f_, t_, s_, m_: _smooth_tree_path(
                point, f_, t_, s_, m_, max_depth, temperature
            )
        )(features, thresholds, splittable, leaf_mass)
        return jnp.exp2(
            -jnp.mean(paths) / jnp.maximum(normalization, jnp.finfo(float).tiny)
        )

    return jax.vmap(score_point)(queries)


class IsolationForestModel(AbstractArrayModel):
    """Exact hard isolation forest; splits, paths, and predictions are nondifferentiable."""

    feature_indices: Array
    thresholds: Array
    splittable: Array
    leaf_mass: Array
    normalization: Array
    threshold: Array
    max_depth: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        feature_indices: ArrayLike,
        thresholds: ArrayLike,
        splittable: ArrayLike,
        leaf_mass: ArrayLike,
        normalization: ArrayLike,
        threshold: ArrayLike,
        *,
        max_depth: int,
        feature_count: int,
        case_shape: tuple[int, ...],
    ):
        self.feature_indices = jnp.asarray(feature_indices, dtype=jnp.int32)
        self.thresholds = jnp.asarray(thresholds)
        self.splittable = jnp.asarray(splittable, dtype=bool)
        self.leaf_mass = jnp.asarray(leaf_mass)
        self.normalization = jnp.asarray(normalization)
        self.threshold = jnp.asarray(threshold)
        self.max_depth = int(max_depth)
        self.case_shape = tuple(case_shape)
        self.in_size = int(feature_count)
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        if jnp.issubdtype(queries.dtype, jnp.complexfloating):
            raise TypeError(
                "Isolation forest split ordering is undefined for complex features."
            )
        queries = queries.astype(self.thresholds.dtype)
        cases = _case_count(self.case_shape)
        tree_shape = self.feature_indices.shape[-2:]
        scores = jax.vmap(
            lambda q_, f_, t_, s_, m_, n_: _hard_forest_scores_one(
                q_, f_, t_, s_, m_, n_, self.max_depth
            )
        )(
            queries,
            self.feature_indices.reshape((cases,) + tree_shape),
            self.thresholds.reshape((cases,) + tree_shape),
            self.splittable.reshape((cases,) + tree_shape),
            self.leaf_mass.reshape((cases,) + tree_shape),
            self.normalization.reshape((cases,)),
        )
        return jax.lax.stop_gradient(
            _restore_scores(scores, case_shape=self.case_shape, query_shape=query_shape)
        )

    def predict(self, x: Any, /) -> Array:
        scores = self(x)
        threshold = self.threshold.reshape(
            self.case_shape + (1,) * (scores.ndim - len(self.case_shape))
        )
        return jax.lax.stop_gradient(scores > threshold)

    def relaxed(self, *, temperature: float = 0.1) -> "SmoothIsolationForestModel":
        """Return a distinct smooth-routing model; no straight-through estimator is used."""
        return SmoothIsolationForestModel(
            self.feature_indices,
            self.thresholds,
            self.splittable,
            self.leaf_mass,
            self.normalization,
            self.threshold,
            max_depth=self.max_depth,
            feature_count=self.in_size,
            temperature=temperature,
            case_shape=self.case_shape,
        )


class SmoothIsolationForestModel(AbstractArrayModel):
    """Differentiable sigmoid-routing relaxation of a fitted isolation forest."""

    feature_indices: Array
    thresholds: Array
    splittable: Array
    leaf_mass: Array
    normalization: Array
    threshold: Array
    temperature: float = eqx.field(static=True)
    max_depth: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        feature_indices: ArrayLike,
        thresholds: ArrayLike,
        splittable: ArrayLike,
        leaf_mass: ArrayLike,
        normalization: ArrayLike,
        threshold: ArrayLike,
        *,
        max_depth: int,
        feature_count: int,
        temperature: float,
        case_shape: tuple[int, ...],
    ):
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive.")
        self.feature_indices = jnp.asarray(feature_indices, dtype=jnp.int32)
        self.thresholds = jnp.asarray(thresholds)
        self.splittable = jnp.asarray(splittable, dtype=bool)
        self.leaf_mass = jnp.asarray(leaf_mass)
        self.normalization = jnp.asarray(normalization)
        self.threshold = jnp.asarray(threshold)
        self.temperature = float(temperature)
        self.max_depth = int(max_depth)
        self.case_shape = tuple(case_shape)
        self.in_size = int(feature_count)
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        if jnp.issubdtype(queries.dtype, jnp.complexfloating):
            raise TypeError("Smooth isolation routing requires real features.")
        queries = queries.astype(self.thresholds.dtype)
        cases = _case_count(self.case_shape)
        tree_shape = self.feature_indices.shape[-2:]
        scores = jax.vmap(
            lambda q_, f_, t_, s_, m_, n_: _smooth_forest_scores_one(
                q_, f_, t_, s_, m_, n_, self.max_depth, self.temperature
            )
        )(
            queries,
            self.feature_indices.reshape((cases,) + tree_shape),
            self.thresholds.reshape((cases,) + tree_shape),
            self.splittable.reshape((cases,) + tree_shape),
            self.leaf_mass.reshape((cases,) + tree_shape),
            self.normalization.reshape((cases,)),
        )
        return _restore_scores(
            scores, case_shape=self.case_shape, query_shape=query_shape
        )

    def smooth_membership(self, x: Any, /, *, temperature: ArrayLike = 1.0) -> Array:
        scores = self(x)
        threshold = self.threshold.reshape(
            self.case_shape + (1,) * (scores.ndim - len(self.case_shape))
        )
        temperature_ = jnp.asarray(temperature)
        temperature_ = eqx.error_if(
            temperature_,
            jnp.any(~jnp.isfinite(temperature_) | (temperature_ <= 0.0)),
            "temperature must be finite and positive.",
        )
        return jax.nn.sigmoid((scores - threshold) / temperature_)


class IsolationForestRecipe(AbstractRecipe):
    """Native weighted isolation forest with explicit key and fixed tree capacity."""

    n_estimators: int = eqx.field(static=True)
    max_depth: int = eqx.field(static=True)
    contamination: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 8,
        contamination: float = 0.1,
    ):
        if int(n_estimators) <= 0:
            raise ValueError("n_estimators must be positive.")
        if int(max_depth) <= 0 or int(max_depth) > 12:
            raise ValueError("max_depth must lie in [1, 12] to bound tree capacity.")
        if not 0.0 < float(contamination) < 0.5:
            raise ValueError("contamination must lie in (0, 0.5).")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.contamination = float(contamination)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("IsolationForestRecipe requires an explicit JAX key.")
        x, weights, active = _fit_arrays(batch)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError(
                "Isolation forest split ordering is undefined for complex features."
            )
        cases = _case_count(batch.case_shape)
        case_keys = jax.random.split(key, cases)
        tree_keys = jax.vmap(lambda key_: jax.random.split(key_, self.n_estimators))(
            case_keys
        )
        flat_x = x.reshape((cases, batch.sample_count, batch.feature_count))
        flat_weights = weights.reshape((cases, batch.sample_count))
        flat_active = active.reshape((cases, batch.sample_count))

        def build_case(x_, weights_, active_, keys_):
            return jax.vmap(
                lambda key_: _build_tree_one(x_, weights_, active_, key_, self.max_depth)
            )(keys_)

        features, thresholds, splittable, leaf_mass = jax.vmap(build_case)(
            flat_x, flat_weights, flat_active, tree_keys
        )
        effective = jnp.sum(flat_active, axis=-1)
        normalization = _average_path_adjustment(effective)
        training_scores = jax.vmap(
            lambda q_, f_, t_, s_, m_, n_: _hard_forest_scores_one(
                q_, f_, t_, s_, m_, n_, self.max_depth
            )
        )(flat_x, features, thresholds, splittable, leaf_mass, normalization)
        training_scores = training_scores.reshape(
            batch.case_shape + (batch.sample_count,)
        )
        threshold = _weighted_threshold(training_scores, weights, self.contamination)
        minimum, maximum = _score_bounds(training_scores, active)
        features = features.reshape(batch.case_shape + features.shape[-2:])
        thresholds = thresholds.reshape(batch.case_shape + thresholds.shape[-2:])
        splittable = splittable.reshape(batch.case_shape + splittable.shape[-2:])
        leaf_mass = leaf_mass.reshape(batch.case_shape + leaf_mass.shape[-2:])
        normalization = normalization.reshape(batch.case_shape)
        effective = effective.reshape(batch.case_shape)
        finite = jnp.all(jnp.isfinite(thresholds), axis=(-2, -1)) & jnp.isfinite(
            threshold
        )
        enough = effective >= 2
        valid = finite & enough
        status = _fit_status(finite, enough)
        diagnostics = OutlierDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.sum(jnp.where(active, weights * training_scores, 0.0), axis=-1)
            / jnp.maximum(jnp.sum(weights, axis=-1), jnp.finfo(float).tiny),
            iterations=self.max_depth,
            effective_samples=effective,
            threshold=threshold,
            score_minimum=minimum,
            score_maximum=maximum,
            rank=-1,
            condition=jnp.nan,
            converged=True,
            method="isolation-forest",
        )
        model = IsolationForestModel(
            features,
            thresholds,
            splittable,
            leaf_mass,
            normalization,
            threshold,
            max_depth=self.max_depth,
            feature_count=batch.feature_count,
            case_shape=batch.case_shape,
        )
        contract = GradientContract(
            prediction_inputs="none",
            prediction_parameters="none",
            fit_features="none",
            fit_targets="none",
            fit_weights="none",
            fit_hyperparameters="none",
            fit_mode="stopped",
            nondifferentiable_outputs=(
                "tree_topology",
                "split_features",
                "hard_paths",
                "predict",
                "threshold",
                "valid",
                "status",
            ),
            conditions=(
                "relaxed() returns a distinct sigmoid-routing model with smooth input scores",
                "tree capacity is exactly 2^(max_depth+1)-1 nodes per estimator",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="isolation-forest",
            gradient_contract=contract,
        )


__all__ = [
    "IsolationForestModel",
    "IsolationForestRecipe",
    "SmoothIsolationForestModel",
]
