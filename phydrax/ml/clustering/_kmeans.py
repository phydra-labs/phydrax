#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import MetricName
from ._common import (
    active_data,
    ClusterDiagnostics,
    ClusterInitialization,
    distances_to_centers,
    effective_sample_count,
    EmptyClusterPolicy,
    HardClusterModel,
    initialize_centers,
    nonnegative_scalar,
    pairwise_distances,
    positive_scalar,
    real_dtype,
    SoftClusterModel,
    stable_top_indices,
)


def _validate_cluster_policies(
    initialization: ClusterInitialization, empty_policy: EmptyClusterPolicy
) -> None:
    if initialization not in ("random", "first", "k-means++"):
        raise ValueError("unsupported cluster initialization.")
    if empty_policy not in ("retain", "reseed", "error"):
        raise ValueError("unsupported empty-cluster policy.")


def _update_centers(
    x: Array,
    w: Array,
    assignment: Array,
    centers: Array,
    cluster_count: int,
    empty_policy: EmptyClusterPolicy,
) -> tuple[Array, Array, Array]:
    membership = jax.nn.one_hot(assignment, cluster_count, dtype=w.dtype)
    weighted = w[..., :, None] * membership
    mass = jnp.sum(weighted, axis=-2)
    proposed = (
        jnp.einsum("...nk,...nf->...kf", weighted, x)
        / jnp.maximum(mass, jnp.finfo(w.dtype).tiny)[..., :, None]
    )
    empty = mass <= jnp.finfo(w.dtype).eps * jnp.maximum(
        jnp.sum(w, axis=-1, keepdims=True), 1.0
    )
    if empty_policy == "reseed":
        distances = distances_to_centers(x, centers, "squared-euclidean", x.shape[:-2])
        farthest = jnp.min(distances, axis=-1)
        farthest = jnp.where(w > 0.0, farthest, -jnp.inf)
        candidates = stable_top_indices(farthest, cluster_count)
        replacements = jnp.take_along_axis(x, candidates[..., :, None], axis=-2)
        proposed = jnp.where(empty[..., :, None], replacements, proposed)
    else:
        proposed = jnp.where(empty[..., :, None], centers, proposed)
    return proposed, mass, empty


def _fit_kmeans(
    batch: MLBatch,
    *,
    cluster_count: int,
    max_iterations: int,
    tolerance: Array | float,
    initialization: ClusterInitialization,
    empty_policy: EmptyClusterPolicy,
    weight_policy: WeightPolicy,
    key: Any,
    temperature: Array | None,
) -> tuple[Array, ...]:
    if cluster_count > batch.sample_count:
        raise ValueError("cluster_count cannot exceed fixed sample capacity.")
    x, w, active, invalid = active_data(batch, weight_policy)
    centers = initialize_centers(x, w, cluster_count, initialization, key)
    case_shape = batch.case_shape
    delta = jnp.full(case_shape, jnp.inf, dtype=w.dtype)
    empty_seen = jnp.zeros(case_shape, dtype=bool)

    def step(_, state):
        centers, delta, empty_seen = state
        distances = distances_to_centers(x, centers, "squared-euclidean", case_shape)
        if temperature is None:
            assignment = jnp.argmin(distances, axis=-1)
            next_centers, _, empty = _update_centers(
                x, w, assignment, centers, cluster_count, empty_policy
            )
        else:
            responsibility = jax.nn.softmax(-distances / temperature, axis=-1)
            weighted = w[..., :, None] * responsibility
            mass = jnp.sum(weighted, axis=-2)
            next_centers = (
                jnp.einsum("...nk,...nf->...kf", weighted, x)
                / jnp.maximum(mass, jnp.finfo(w.dtype).tiny)[..., :, None]
            )
            empty = mass <= jnp.finfo(w.dtype).eps * jnp.maximum(
                jnp.sum(w, axis=-1, keepdims=True), 1.0
            )
            if empty_policy == "reseed":
                farthest = jnp.where(w > 0.0, jnp.min(distances, axis=-1), -jnp.inf)
                candidates = stable_top_indices(farthest, cluster_count)
                replacements = jnp.take_along_axis(x, candidates[..., :, None], axis=-2)
                next_centers = jnp.where(empty[..., :, None], replacements, next_centers)
            else:
                next_centers = jnp.where(empty[..., :, None], centers, next_centers)
        next_delta = jnp.max(jnp.linalg.norm(next_centers - centers, axis=-1), axis=-1)
        return next_centers, next_delta, empty_seen | jnp.any(empty, axis=-1)

    centers, delta, empty_seen = jax.lax.fori_loop(
        0, max_iterations, step, (centers, delta, empty_seen)
    )
    distances = distances_to_centers(x, centers, "squared-euclidean", case_shape)
    if temperature is None:
        assignment = jnp.argmin(distances, axis=-1)
        probability = jax.nn.one_hot(assignment, cluster_count, dtype=w.dtype)
    else:
        probability = jax.nn.softmax(-distances / temperature, axis=-1)
        assignment = jnp.argmax(probability, axis=-1)
    cluster_mass = jnp.sum(w[..., :, None] * probability, axis=-2)
    active_clusters = cluster_mass > jnp.finfo(w.dtype).eps * jnp.maximum(
        jnp.sum(w, axis=-1, keepdims=True), 1.0
    )
    objective = jnp.sum(
        w * jnp.sum(probability * distances, axis=-1), axis=-1
    ) / jnp.maximum(jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny)
    enough = jnp.sum(active, axis=-1) >= cluster_count
    finite = jnp.isfinite(objective) & jnp.all(jnp.isfinite(centers), axis=(-2, -1))
    converged = delta <= tolerance
    empty_error = empty_seen & (empty_policy == "error")
    valid = enough & finite & converged & ~invalid & ~empty_error
    status = jnp.where(
        invalid | ~finite,
        ML_NONFINITE,
        jnp.where(
            ~enough | empty_error,
            ML_INSUFFICIENT_DATA,
            jnp.where(~converged, ML_NONCONVERGED, ML_SUCCESS),
        ),
    )
    return (
        centers,
        probability,
        cluster_mass,
        active_clusters,
        objective,
        delta,
        empty_seen,
        converged,
        valid,
        status,
        effective_sample_count(w),
    )


def _fit_result(
    batch: MLBatch,
    values: tuple[Array, ...],
    *,
    iterations: int,
    method: str,
    soft_temperature: Array | None,
) -> FitResult:
    centers, _, mass, active, objective, _, empty, converged, valid, status, effective = (
        values
    )
    if soft_temperature is None:
        model = HardClusterModel(centers, active, method=method)
        contract = GradientContract(
            prediction_inputs="none",
            prediction_parameters="none",
            fit_mode="stopped",
            nondifferentiable_outputs=("labels", "assignments"),
            conditions=("deterministic lowest-index tie breaking",),
        )
    else:
        model = SoftClusterModel(centers, active, soft_temperature, method=method)
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("hard_labels",),
            conditions=(
                "positive temperature",
                "fixed active mask",
                "fixed initialization indices",
            ),
        )
    diagnostics = ClusterDiagnostics(
        valid=valid,
        status=status,
        objective=objective,
        iterations=iterations,
        effective_samples=effective,
        cluster_mass=mass,
        active_clusters=active,
        empty_clusters_seen=empty,
        converged=converged,
        method=method,
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        gradient_contract=contract,
    )


class KMeans(AbstractRecipe):
    cluster_count: int = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    initialization: ClusterInitialization = eqx.field(static=True)
    empty_policy: EmptyClusterPolicy = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        cluster_count: int,
        /,
        *,
        max_iterations: int = 64,
        tolerance: float = 1e-4,
        initialization: ClusterInitialization = "k-means++",
        empty_policy: EmptyClusterPolicy = "reseed",
        weight_policy: WeightPolicy = "statistical",
    ):
        if cluster_count <= 0 or max_iterations <= 0 or tolerance < 0.0:
            raise ValueError("invalid k-means configuration.")
        _validate_cluster_policies(initialization, empty_policy)
        self.cluster_count = int(cluster_count)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.initialization = initialization
        self.empty_policy = empty_policy
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        values = _fit_kmeans(
            batch,
            cluster_count=self.cluster_count,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            initialization=self.initialization,
            empty_policy=self.empty_policy,
            weight_policy=self.weight_policy,
            key=key,
            temperature=None,
        )
        return _fit_result(
            batch,
            values,
            iterations=self.max_iterations,
            method="k-means",
            soft_temperature=None,
        )


class SoftKMeans(AbstractRecipe):
    cluster_count: int = eqx.field(static=True)
    temperature: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: Array
    initialization: ClusterInitialization = eqx.field(static=True)
    empty_policy: EmptyClusterPolicy = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        cluster_count: int,
        /,
        *,
        temperature: float = 1.0,
        max_iterations: int = 64,
        tolerance: float = 1e-4,
        initialization: ClusterInitialization = "k-means++",
        empty_policy: EmptyClusterPolicy = "reseed",
        weight_policy: WeightPolicy = "statistical",
    ):
        if cluster_count <= 0 or max_iterations <= 0:
            raise ValueError("cluster_count and max_iterations must be positive.")
        _validate_cluster_policies(initialization, empty_policy)
        self.cluster_count = int(cluster_count)
        self.temperature = positive_scalar(temperature, "temperature")
        self.max_iterations = int(max_iterations)
        self.tolerance = nonnegative_scalar(tolerance, "tolerance")
        self.initialization = initialization
        self.empty_policy = empty_policy
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        values = _fit_kmeans(
            batch,
            cluster_count=self.cluster_count,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            initialization=self.initialization,
            empty_policy=self.empty_policy,
            weight_policy=self.weight_policy,
            key=key,
            temperature=self.temperature,
        )
        return _fit_result(
            batch,
            values,
            iterations=self.max_iterations,
            method="soft-k-means",
            soft_temperature=self.temperature,
        )


class KMedoids(AbstractRecipe):
    cluster_count: int = eqx.field(static=True)
    metric: MetricName = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    initialization: ClusterInitialization = eqx.field(static=True)
    empty_policy: EmptyClusterPolicy = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        cluster_count: int,
        /,
        *,
        metric: MetricName = "euclidean",
        max_iterations: int = 64,
        initialization: ClusterInitialization = "k-means++",
        empty_policy: EmptyClusterPolicy = "reseed",
        weight_policy: WeightPolicy = "statistical",
    ):
        if cluster_count <= 0 or max_iterations <= 0:
            raise ValueError("invalid k-medoids configuration.")
        _validate_cluster_policies(initialization, empty_policy)
        if metric not in ("euclidean", "squared-euclidean", "manhattan", "cosine"):
            raise ValueError("unsupported k-medoids metric.")
        self.cluster_count = int(cluster_count)
        self.metric = metric
        self.max_iterations = int(max_iterations)
        self.initialization = initialization
        self.empty_policy = empty_policy
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if self.cluster_count > batch.sample_count:
            raise ValueError("cluster_count cannot exceed sample capacity.")
        x, w, active_samples, invalid = active_data(batch, self.weight_policy)
        medoids = initialize_centers(x, w, self.cluster_count, self.initialization, key)
        pairwise = pairwise_distances(x, self.metric)
        changed = jnp.ones(batch.case_shape, dtype=bool)
        empty_seen = jnp.zeros(batch.case_shape, dtype=bool)

        def step(_, state):
            medoids, changed, empty_seen = state
            distances = distances_to_centers(x, medoids, self.metric, batch.case_shape)
            labels = jnp.argmin(distances, axis=-1)
            membership = (
                jax.nn.one_hot(labels, self.cluster_count, dtype=w.dtype)
                * w[..., :, None]
            )
            mass = jnp.sum(membership, axis=-2)
            costs = jnp.einsum("...ik,...ij->...kj", membership, pairwise)
            candidate_valid = jnp.swapaxes(membership > 0.0, -1, -2)
            indices = jnp.argmin(jnp.where(candidate_valid, costs, jnp.inf), axis=-1)
            proposed = jnp.take_along_axis(x, indices[..., :, None], axis=-2)
            empty = mass <= 0.0
            if self.empty_policy == "reseed":
                farthest = jnp.where(w > 0.0, jnp.min(distances, axis=-1), -jnp.inf)
                replacements = stable_top_indices(farthest, self.cluster_count)
                replacement_values = jnp.take_along_axis(
                    x, replacements[..., :, None], axis=-2
                )
                proposed = jnp.where(empty[..., :, None], replacement_values, proposed)
            else:
                proposed = jnp.where(empty[..., :, None], medoids, proposed)
            changed = jnp.any(jnp.any(proposed != medoids, axis=-1), axis=-1)
            return proposed, changed, empty_seen | jnp.any(empty, axis=-1)

        medoids, changed, empty_seen = jax.lax.fori_loop(
            0, self.max_iterations, step, (medoids, changed, empty_seen)
        )
        distances = distances_to_centers(x, medoids, self.metric, batch.case_shape)
        labels = jnp.argmin(distances, axis=-1)
        membership = jax.nn.one_hot(labels, self.cluster_count, dtype=w.dtype)
        mass = jnp.sum(w[..., :, None] * membership, axis=-2)
        active = mass > 0.0
        objective = jnp.sum(w * jnp.min(distances, axis=-1), axis=-1) / jnp.maximum(
            jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny
        )
        enough = jnp.sum(active_samples, axis=-1) >= self.cluster_count
        finite = jnp.isfinite(objective)
        converged = ~changed
        empty_error = empty_seen & (self.empty_policy == "error")
        valid = enough & finite & converged & ~invalid & ~empty_error
        status = jnp.where(
            invalid | ~finite,
            ML_NONFINITE,
            jnp.where(
                ~enough | empty_error,
                ML_INSUFFICIENT_DATA,
                jnp.where(~converged, ML_NONCONVERGED, ML_SUCCESS),
            ),
        )
        model = HardClusterModel(medoids, active, metric=self.metric, method="k-medoids")
        diagnostics = ClusterDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.max_iterations,
            effective_samples=effective_sample_count(w),
            cluster_mass=mass,
            active_clusters=active,
            empty_clusters_seen=empty_seen,
            converged=converged,
            method="k-medoids",
        )
        contract = GradientContract(
            prediction_inputs="none",
            prediction_parameters="none",
            fit_mode="stopped",
            nondifferentiable_outputs=("labels", "medoid indices"),
            conditions=("deterministic lowest-index tie breaking",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="k-medoids",
            gradient_contract=contract,
        )


class StreamingKMeans(StrictModule):
    """Immutable fixed-capacity online k-means state."""

    centers: Array
    cluster_mass: Array
    updates: Array

    def __init__(
        self,
        centers: ArrayLike,
        cluster_mass: ArrayLike | None = None,
        updates: ArrayLike = 0,
    ):
        centers_ = jnp.asarray(centers)
        if centers_.ndim < 2:
            raise ValueError("centers must have shape case + (cluster, feature).")
        self.centers = centers_
        self.cluster_mass = (
            jnp.zeros(centers_.shape[:-1], dtype=real_dtype(centers_.dtype))
            if cluster_mass is None
            else jnp.asarray(cluster_mass, dtype=real_dtype(centers_.dtype))
        )
        self.updates = jnp.asarray(updates, dtype=jnp.int32)

    def update(
        self,
        values: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        temperature: float | None = None,
    ) -> "StreamingKMeans":
        x = jnp.asarray(values, dtype=self.centers.dtype)
        case_shape = self.centers.shape[:-2]
        if x.shape[:-2] != case_shape or x.shape[-1] != self.centers.shape[-1]:
            raise ValueError("values must have shape case + (sample, feature).")
        w = (
            jnp.ones(x.shape[:-1], dtype=self.cluster_mass.dtype)
            if weights is None
            else jnp.broadcast_to(
                jnp.asarray(weights, dtype=self.cluster_mass.dtype), x.shape[:-1]
            )
        )
        active = jnp.isfinite(w) & (w >= 0.0) & jnp.all(jnp.isfinite(x), axis=-1)
        if mask is not None:
            active &= jnp.broadcast_to(jnp.asarray(mask, dtype=bool), x.shape[:-1])
        w = jnp.where(active, w, 0.0)
        x = jnp.where(active[..., None], x, 0)
        distances = distances_to_centers(x, self.centers, "squared-euclidean", case_shape)
        if temperature is None:
            responsibility = jax.nn.one_hot(
                jnp.argmin(distances, axis=-1), self.centers.shape[-2], dtype=w.dtype
            )
        else:
            temperature_ = positive_scalar(temperature, "temperature")
            responsibility = jax.nn.softmax(-distances / temperature_, axis=-1)
        weighted = w[..., :, None] * responsibility
        batch_mass = jnp.sum(weighted, axis=-2)
        total_mass = self.cluster_mass + batch_mass
        numerator = self.cluster_mass[..., :, None] * self.centers + jnp.einsum(
            "...nk,...nf->...kf", weighted, x
        )
        centers = jnp.where(
            total_mass[..., :, None] > 0.0,
            numerator / jnp.maximum(total_mass, jnp.finfo(w.dtype).tiny)[..., :, None],
            self.centers,
        )
        return StreamingKMeans(centers, total_mass, self.updates + 1)

    def model(self, /, *, temperature: float | None = None):
        active = self.cluster_mass > 0.0
        if temperature is None:
            return HardClusterModel(self.centers, active, method="streaming-k-means")
        return SoftClusterModel(
            self.centers, active, temperature, method="streaming-soft-k-means"
        )


class MiniBatchKMeans(AbstractRecipe):
    cluster_count: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    initialization: ClusterInitialization = eqx.field(static=True)
    empty_policy: EmptyClusterPolicy = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        cluster_count: int,
        /,
        *,
        batch_size: int = 32,
        max_iterations: int = 64,
        initialization: ClusterInitialization = "k-means++",
        empty_policy: EmptyClusterPolicy = "retain",
        weight_policy: WeightPolicy = "statistical",
    ):
        if cluster_count <= 0 or batch_size <= 0 or max_iterations <= 0:
            raise ValueError("invalid mini-batch k-means configuration.")
        _validate_cluster_policies(initialization, empty_policy)
        self.cluster_count = int(cluster_count)
        self.batch_size = int(batch_size)
        self.max_iterations = int(max_iterations)
        self.initialization = initialization
        self.empty_policy = empty_policy
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("mini-batch k-means requires an explicit JAX key.")
        if self.cluster_count > batch.sample_count:
            raise ValueError("cluster_count cannot exceed sample capacity.")
        x, w, active_samples, invalid = active_data(batch, self.weight_policy)
        init_key, draw_key = jax.random.split(key)
        centers = initialize_centers(
            x, w, self.cluster_count, self.initialization, init_key
        )
        case_count = 1
        for size in batch.case_shape:
            case_count *= size
        split_keys = jax.random.split(draw_key, case_count * self.max_iterations)
        keys = split_keys.reshape(
            (case_count, self.max_iterations) + split_keys.shape[1:]
        )
        flat_w = w.reshape((case_count, batch.sample_count))

        def draws(case_weights, case_keys):
            logits = jnp.where(case_weights > 0.0, jnp.log(case_weights), -jnp.inf)
            return jax.vmap(
                lambda sample_key: jax.random.categorical(
                    sample_key, logits, shape=(self.batch_size,)
                )
            )(case_keys)

        indices = jax.vmap(draws)(flat_w, keys).reshape(
            batch.case_shape + (self.max_iterations, self.batch_size)
        )
        mass = jnp.zeros(batch.case_shape + (self.cluster_count,), dtype=w.dtype)

        def step(i, state):
            centers, mass = state
            selected = indices[..., i, :]
            batch_x = jnp.take_along_axis(x, selected[..., :, None], axis=-2)
            batch_w = jnp.take_along_axis(w, selected, axis=-1)
            distances = distances_to_centers(
                batch_x, centers, "squared-euclidean", batch.case_shape
            )
            assignment = jnp.argmin(distances, axis=-1)
            membership = jax.nn.one_hot(assignment, self.cluster_count, dtype=w.dtype)
            weighted = batch_w[..., :, None] * membership
            increment = jnp.sum(weighted, axis=-2)
            total = mass + increment
            numerator = mass[..., :, None] * centers + jnp.einsum(
                "...bk,...bf->...kf", weighted, batch_x
            )
            centers = jnp.where(
                total[..., :, None] > 0.0,
                numerator / jnp.maximum(total, jnp.finfo(w.dtype).tiny)[..., :, None],
                centers,
            )
            return centers, total

        centers, mass = jax.lax.fori_loop(0, self.max_iterations, step, (centers, mass))
        distances = distances_to_centers(
            x, centers, "squared-euclidean", batch.case_shape
        )
        labels = jnp.argmin(distances, axis=-1)
        membership = jax.nn.one_hot(labels, self.cluster_count, dtype=w.dtype)
        final_mass = jnp.sum(w[..., :, None] * membership, axis=-2)
        active = final_mass > 0.0
        empty_seen = ~active
        if self.empty_policy == "reseed":
            farthest = jnp.where(w > 0.0, jnp.min(distances, axis=-1), -jnp.inf)
            replacements = stable_top_indices(farthest, self.cluster_count)
            replacement_values = jnp.take_along_axis(
                x, replacements[..., :, None], axis=-2
            )
            centers = jnp.where(empty_seen[..., :, None], replacement_values, centers)
            distances = distances_to_centers(
                x, centers, "squared-euclidean", batch.case_shape
            )
            labels = jnp.argmin(distances, axis=-1)
            membership = jax.nn.one_hot(labels, self.cluster_count, dtype=w.dtype)
            final_mass = jnp.sum(w[..., :, None] * membership, axis=-2)
            active = final_mass > 0.0
        objective = jnp.sum(w * jnp.min(distances, axis=-1), axis=-1) / jnp.maximum(
            jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny
        )
        enough = jnp.sum(active_samples, axis=-1) >= self.cluster_count
        empty_error = jnp.any(empty_seen, axis=-1) & (self.empty_policy == "error")
        finite = jnp.isfinite(objective)
        valid = enough & finite & ~invalid & ~empty_error
        status = jnp.where(
            invalid | ~finite,
            ML_NONFINITE,
            jnp.where(~enough | empty_error, ML_INSUFFICIENT_DATA, ML_SUCCESS),
        )
        model = HardClusterModel(centers, active, method="mini-batch-k-means")
        diagnostics = ClusterDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.max_iterations,
            effective_samples=effective_sample_count(w),
            cluster_mass=final_mass,
            active_clusters=active,
            empty_clusters_seen=jnp.any(empty_seen, axis=-1),
            converged=jnp.ones_like(valid),
            method="mini-batch-k-means",
        )
        contract = GradientContract(
            prediction_inputs="none",
            prediction_parameters="none",
            fit_mode="stopped",
            nondifferentiable_outputs=("labels", "sampled mini-batches"),
            conditions=("explicit random key", "fixed iteration count"),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="mini-batch-k-means",
            gradient_contract=contract,
        )


__all__ = ["KMeans", "KMedoids", "MiniBatchKMeans", "SoftKMeans", "StreamingKMeans"]
