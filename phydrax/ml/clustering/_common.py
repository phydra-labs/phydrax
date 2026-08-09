#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._numerics import MetricName


ClusterInitialization: TypeAlias = Literal["random", "first", "k-means++"]
EmptyClusterPolicy: TypeAlias = Literal["retain", "reseed", "error"]


def real_dtype(dtype: jnp.dtype) -> jnp.dtype:
    return jnp.empty((), dtype=dtype).real.dtype


def validated_scalar(value: Any, name: str, /, *, allow_zero: bool) -> Array:
    scalar = jnp.asarray(value)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    relation = "nonnegative" if allow_zero else "positive"
    invalid = ~jnp.isfinite(scalar) | (scalar < 0.0 if allow_zero else scalar <= 0.0)
    return eqx.error_if(scalar, invalid, f"{name} must be finite and {relation}.")


def nonnegative_scalar(value: Any, name: str, /) -> Array:
    return validated_scalar(value, name, allow_zero=True)


def positive_scalar(value: Any, name: str, /) -> Array:
    return validated_scalar(value, name, allow_zero=False)


def active_data(
    batch: MLBatch, policy: WeightPolicy
) -> tuple[Array, Array, Array, Array]:
    x = batch.dense_features()
    raw = batch.effective_weight(policy)
    complete = jnp.all(batch.feature_mask, axis=-1)
    finite_x = jnp.all(jnp.isfinite(x), axis=-1)
    weights_ok = jnp.isfinite(raw) & (raw >= 0.0)
    active = complete & finite_x & weights_ok & (raw > 0.0)
    w = jnp.where(active, raw, 0.0).astype(real_dtype(x.dtype))
    x = jnp.where(active[..., None], x, 0)
    invalid = jnp.any(batch.sample_mask & (~weights_ok | (complete & ~finite_x)), axis=-1)
    return x, w, active, invalid


def effective_sample_count(weights: Array, /) -> Array:
    total = jnp.sum(weights, axis=-1)
    squared = jnp.sum(weights * weights, axis=-1)
    return jnp.where(squared > 0.0, total * total / squared, 0.0)


def distances_to_centers(
    values: Array, centers: Array, metric: MetricName, case_shape: tuple[int, ...]
) -> Array:
    minimum_rank = len(case_shape) + 1
    if values.ndim < minimum_rank:
        raise ValueError("case axes and the final feature axis must be distinct.")
    if (
        values.shape[: len(case_shape)] != case_shape
        or values.shape[-1] != centers.shape[-1]
    ):
        raise ValueError("input must have shape case + sample_shape + (feature,).")
    sample_ndim = values.ndim - minimum_rank
    centers_ = centers.reshape(case_shape + (1,) * sample_ndim + centers.shape[-2:])
    difference = values[..., None, :] - centers_
    squared = jnp.real(jnp.sum(jnp.conj(difference) * difference, axis=-1))
    if metric == "squared-euclidean":
        return jnp.maximum(squared, 0.0)
    if metric == "euclidean":
        return jnp.sqrt(jnp.maximum(squared, 0.0))
    if metric == "manhattan":
        return jnp.sum(jnp.abs(difference), axis=-1)
    if metric == "cosine":
        numerator = jnp.real(jnp.sum(jnp.conj(values[..., None, :]) * centers_, axis=-1))
        left_norm = jnp.sqrt(jnp.real(jnp.sum(jnp.conj(values) * values, axis=-1)))[
            ..., None
        ]
        right_norm = jnp.sqrt(jnp.real(jnp.sum(jnp.conj(centers_) * centers_, axis=-1)))
        similarity = numerator / jnp.maximum(
            left_norm * right_norm, jnp.finfo(real_dtype(values.dtype)).tiny
        )
        return jnp.maximum(1.0 - similarity, 0.0)
    raise ValueError(f"unsupported metric {metric!r}.")


def pairwise_distances(values: Array, metric: MetricName) -> Array:
    difference = values[..., :, None, :] - values[..., None, :, :]
    squared = jnp.real(jnp.sum(jnp.conj(difference) * difference, axis=-1))
    if metric == "squared-euclidean":
        return jnp.maximum(squared, 0.0)
    if metric == "euclidean":
        return jnp.sqrt(jnp.maximum(squared, 0.0))
    if metric == "manhattan":
        return jnp.sum(jnp.abs(difference), axis=-1)
    if metric == "cosine":
        norms = jnp.sqrt(jnp.real(jnp.sum(jnp.conj(values) * values, axis=-1)))
        similarity = jnp.real(
            values @ jnp.conj(jnp.swapaxes(values, -1, -2))
        ) / jnp.maximum(
            norms[..., :, None] * norms[..., None, :],
            jnp.finfo(real_dtype(values.dtype)).tiny,
        )
        return jnp.maximum(1.0 - similarity, 0.0)
    raise ValueError(f"unsupported metric {metric!r}.")


def stable_top_indices(scores: Array, count: int, /) -> Array:
    """Return descending-score indices with lower indices winning exact ties."""
    return jnp.argsort(-jnp.asarray(scores), axis=-1, stable=True)[..., :count]


def initialize_centers(
    x: Array,
    w: Array,
    cluster_count: int,
    initialization: ClusterInitialization,
    key: Any,
) -> Array:
    case_shape = x.shape[:-2]
    n, p = x.shape[-2:]
    case_count = 1
    for size in case_shape:
        case_count *= size
    flat_x = x.reshape((case_count, n, p))
    flat_w = w.reshape((case_count, n))
    if initialization not in ("random", "first", "k-means++"):
        raise ValueError(f"unsupported initialization {initialization!r}.")
    if initialization == "first":
        order = jnp.argsort(
            jnp.where(flat_w > 0.0, jnp.arange(n), n), axis=-1, stable=True
        )
        indices = order[:, :cluster_count]
        centers = jnp.take_along_axis(flat_x, indices[..., None], axis=-2)
    else:
        if key is None:
            raise ValueError(
                f"{initialization} initialization requires an explicit JAX key."
            )
        keys = jax.random.split(key, case_count)

        def choose(values, weights, case_key):
            first_key, current_key = jax.random.split(case_key)
            logits = jnp.where(weights > 0.0, jnp.log(weights), -jnp.inf)
            first = jax.random.categorical(first_key, logits).astype(jnp.int32)
            indices = jnp.zeros((cluster_count,), dtype=jnp.int32).at[0].set(first)

            def add_center(i, state):
                indices, current_key = state
                current_key, draw_key = jax.random.split(current_key)
                chosen = values[indices]
                difference = values[:, None, :] - chosen[None, :, :]
                distance = jnp.min(
                    jnp.real(jnp.sum(jnp.conj(difference) * difference, axis=-1))
                    + jnp.where(jnp.arange(cluster_count) < i, 0.0, jnp.inf),
                    axis=-1,
                )
                scores = weights if initialization == "random" else weights * distance
                scores = jnp.where(scores > 0.0, scores, weights)
                selected = jax.random.categorical(
                    draw_key,
                    jnp.where(scores > 0.0, jnp.log(scores), -jnp.inf),
                ).astype(jnp.int32)
                return indices.at[i].set(selected), current_key

            indices, _ = jax.lax.fori_loop(
                1, cluster_count, add_center, (indices, current_key)
            )
            return values[indices]

        centers = jax.vmap(choose)(flat_x, flat_w, keys)
    return centers.reshape(case_shape + (cluster_count, p))


class ClusterDiagnostics(StrictModule):
    valid: Array
    status: Array
    objective: Array
    iterations: Array
    effective_samples: Array
    cluster_mass: Array
    active_clusters: Array
    empty_clusters_seen: Array
    converged: Array
    degeneracy: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        objective: Any,
        iterations: Any,
        effective_samples: Any,
        cluster_mass: Any,
        active_clusters: Any,
        empty_clusters_seen: Any,
        converged: Any,
        degeneracy: Any = False,
        method: str,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.cluster_mass = jnp.asarray(cluster_mass)
        self.active_clusters = jnp.asarray(active_clusters, dtype=bool)
        self.empty_clusters_seen = jnp.asarray(empty_clusters_seen, dtype=bool)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.degeneracy = jnp.asarray(degeneracy, dtype=bool)
        self.method = str(method)


class HardClusterModel(AbstractArrayModel):
    """Terminal nondifferentiable nearest-representative assignment."""

    centers: Array
    active_clusters: Array
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    metric: MetricName = eqx.field(static=True)
    method: str = eqx.field(static=True)

    def __init__(
        self,
        centers: Array,
        active_clusters: Array,
        /,
        *,
        metric: MetricName = "squared-euclidean",
        method: str,
    ):
        self.centers = jnp.asarray(centers)
        self.active_clusters = jnp.asarray(active_clusters, dtype=bool)
        self.in_size = self.centers.shape[-1]
        self.out_size = "scalar"
        self.case_shape = self.centers.shape[:-2]
        self.metric = metric
        self.method = str(method)

    def distances(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        distances = distances_to_centers(
            values, self.centers, self.metric, self.case_shape
        )
        sample_ndim = distances.ndim - len(self.case_shape) - 1
        active = self.active_clusters.reshape(
            self.case_shape + (1,) * sample_ndim + (self.centers.shape[-2],)
        )
        return jnp.where(active, distances, jnp.inf)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return jax.lax.stop_gradient(
            jnp.argmin(self.distances(x), axis=-1).astype(jnp.int32)
        )


class SoftClusterModel(AbstractArrayModel):
    """Differentiable temperature-relaxed nearest-center responsibilities."""

    centers: Array
    active_clusters: Array
    temperature: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    metric: MetricName = eqx.field(static=True)
    method: str = eqx.field(static=True)

    def __init__(
        self,
        centers: Array,
        active_clusters: Array,
        temperature: Any,
        /,
        *,
        metric: MetricName = "squared-euclidean",
        method: str,
    ):
        self.centers = jnp.asarray(centers)
        self.active_clusters = jnp.asarray(active_clusters, dtype=bool)
        self.temperature = positive_scalar(
            jnp.asarray(temperature, dtype=real_dtype(self.centers.dtype)), "temperature"
        )
        self.in_size = self.centers.shape[-1]
        self.out_size = self.centers.shape[-2]
        self.case_shape = self.centers.shape[:-2]
        self.metric = metric
        self.method = str(method)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = jnp.asarray(x)
        distances = distances_to_centers(
            values, self.centers, self.metric, self.case_shape
        )
        sample_ndim = distances.ndim - len(self.case_shape) - 1
        active = self.active_clusters.reshape(
            self.case_shape + (1,) * sample_ndim + (self.out_size,)
        )
        logits = jnp.where(active, -distances / self.temperature, -jnp.inf)
        probabilities = jax.nn.softmax(logits, axis=-1)
        return jnp.where(jnp.isfinite(probabilities), probabilities, 0.0)

    def hard_labels(self, x: Any, /) -> Array:
        return jax.lax.stop_gradient(jnp.argmax(self(x), axis=-1).astype(jnp.int32))


__all__ = [
    "ClusterDiagnostics",
    "ClusterInitialization",
    "EmptyClusterPolicy",
    "HardClusterModel",
    "SoftClusterModel",
    "stable_top_indices",
    "active_data",
    "effective_sample_count",
    "distances_to_centers",
    "initialize_centers",
    "pairwise_distances",
    "nonnegative_scalar",
    "positive_scalar",
    "real_dtype",
    "validated_scalar",
]
