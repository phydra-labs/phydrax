#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from ._common import (
    active_data,
    ClusterDiagnostics,
    distances_to_centers,
    effective_sample_count,
    HardClusterModel,
    nonnegative_scalar,
    pairwise_distances,
    positive_scalar,
    SoftClusterModel,
)


AgglomerativeLinkage: TypeAlias = Literal["ward", "centroid"]


def _deterministic_embedding_kmeans(
    embedding: Array, w: Array, cluster_count: int, iterations: int
) -> tuple[Array, Array, Array]:
    n = embedding.shape[-2]
    order = jnp.argsort(jnp.where(w > 0.0, jnp.arange(n), n), axis=-1)
    indices = order[..., :cluster_count]
    centers = jnp.take_along_axis(embedding, indices[..., :, None], axis=-2)

    def step(_, centers):
        distances = distances_to_centers(
            embedding, centers, "squared-euclidean", embedding.shape[:-2]
        )
        labels = jnp.argmin(distances, axis=-1)
        membership = (
            jax.nn.one_hot(labels, cluster_count, dtype=w.dtype) * w[..., :, None]
        )
        mass = jnp.sum(membership, axis=-2)
        proposed = (
            oe.contract("...nk,...nf->...kf", membership, embedding)
            / jnp.maximum(mass, jnp.finfo(w.dtype).tiny)[..., :, None]
        )
        return jnp.where((mass > 0.0)[..., :, None], proposed, centers)

    centers = jax.lax.fori_loop(0, iterations, step, centers)
    distances = distances_to_centers(
        embedding, centers, "squared-euclidean", embedding.shape[:-2]
    )
    labels = jnp.argmin(distances, axis=-1)
    membership = jax.nn.one_hot(labels, cluster_count, dtype=w.dtype) * w[..., :, None]
    mass = jnp.sum(membership, axis=-2)
    return centers, labels, mass


class SpectralClustering(AbstractRecipe):
    cluster_count: int = eqx.field(static=True)
    gamma: Array
    temperature: Array
    kmeans_iterations: int = eqx.field(static=True)
    eigenvalue_tolerance: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        cluster_count: int,
        /,
        *,
        gamma: float = 1.0,
        temperature: float = 1.0,
        kmeans_iterations: int = 32,
        eigenvalue_tolerance: float = 1e-7,
        weight_policy: WeightPolicy = "statistical",
    ):
        if cluster_count <= 0 or kmeans_iterations <= 0:
            raise ValueError("cluster_count and kmeans_iterations must be positive.")
        self.cluster_count = int(cluster_count)
        self.gamma = positive_scalar(gamma, "gamma")
        self.temperature = positive_scalar(temperature, "temperature")
        self.kmeans_iterations = int(kmeans_iterations)
        self.eigenvalue_tolerance = nonnegative_scalar(
            eigenvalue_tolerance, "eigenvalue_tolerance"
        )
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if self.cluster_count > batch.sample_count:
            raise ValueError("cluster_count cannot exceed sample capacity.")
        x, w, active, invalid = active_data(batch, self.weight_policy)
        n = batch.sample_count
        distance = pairwise_distances(x, "squared-euclidean")
        adjacency = jnp.exp(-self.gamma * distance) * jnp.sqrt(
            w[..., :, None] * w[..., None, :]
        )
        pair_active = active[..., :, None] & active[..., None, :]
        adjacency = jnp.where(pair_active & ~jnp.eye(n, dtype=bool), adjacency, 0.0)
        degree = jnp.sum(adjacency, axis=-1)
        inverse_root = jnp.where(degree > 0.0, jax.lax.rsqrt(degree), 0.0)
        normalized = inverse_root[..., :, None] * adjacency * inverse_root[..., None, :]
        laplacian = jnp.eye(n, dtype=adjacency.dtype) - normalized
        laplacian = jnp.where(pair_active, laplacian, 0.0)
        laplacian = laplacian.at[..., jnp.arange(n), jnp.arange(n)].set(
            jnp.where(active, jnp.diagonal(laplacian, axis1=-2, axis2=-1), 2.0)
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(laplacian)
        embedding = eigenvectors[..., :, : self.cluster_count]
        pivot = jnp.argmax(jnp.abs(embedding), axis=-2)
        phase_value = jnp.take_along_axis(
            embedding, pivot[..., None, :], axis=-2
        ).squeeze(-2)
        phase = jnp.where(
            jnp.abs(phase_value) > 0.0, phase_value / jnp.abs(phase_value), 1.0
        )
        embedding = embedding * jnp.conj(phase)[..., None, :]
        embedding = embedding / jnp.maximum(
            jnp.linalg.norm(embedding, axis=-1, keepdims=True), jnp.finfo(w.dtype).tiny
        )
        _, labels, cluster_mass = _deterministic_embedding_kmeans(
            embedding, w, self.cluster_count, self.kmeans_iterations
        )
        membership = (
            jax.nn.one_hot(labels, self.cluster_count, dtype=w.dtype) * w[..., :, None]
        )
        centers = (
            oe.contract("...nk,...nf->...kf", membership, x)
            / jnp.maximum(cluster_mass, jnp.finfo(w.dtype).tiny)[..., :, None]
        )
        active_clusters = cluster_mass > 0.0
        sample_distance = distances_to_centers(
            x, centers, "squared-euclidean", batch.case_shape
        )
        objective = jnp.sum(w * jnp.min(sample_distance, axis=-1), axis=-1) / jnp.maximum(
            jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny
        )
        component_count = jnp.sum(
            eigenvalues < self.eigenvalue_tolerance, axis=-1
        ) + jnp.sum(active & (degree <= 0.0), axis=-1)
        disconnected = component_count > 1
        component_overflow = component_count > self.cluster_count
        enough = jnp.sum(active, axis=-1) >= self.cluster_count
        finite = jnp.isfinite(objective) & jnp.all(jnp.isfinite(embedding), axis=(-2, -1))
        valid = (
            enough
            & finite
            & ~invalid
            & ~component_overflow
            & jnp.all(active_clusters, axis=-1)
        )
        status = jnp.where(
            invalid | ~finite,
            ML_NONFINITE,
            jnp.where(
                ~enough | component_overflow | ~jnp.all(active_clusters, axis=-1),
                ML_INSUFFICIENT_DATA,
                ML_SUCCESS,
            ),
        )
        model = SoftClusterModel(
            centers, active_clusters, self.temperature, method="spectral-clustering"
        )
        diagnostics = ClusterDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.kmeans_iterations,
            effective_samples=effective_sample_count(w),
            cluster_mass=cluster_mass,
            active_clusters=active_clusters,
            empty_clusters_seen=~jnp.all(active_clusters, axis=-1),
            converged=jnp.ones_like(valid),
            degeneracy=disconnected,
            method="spectral-clustering",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="spectral",
            nondifferentiable_outputs=("hard_labels", "eigenvector ordering"),
            conditions=(
                "separated retained eigenspace",
                "fixed graph support",
                "fixed k-means assignments",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="spectral-clustering",
            gradient_contract=contract,
        )


def _agglomerate_one(
    x: Array, w: Array, cluster_count: int, linkage: AgglomerativeLinkage
) -> tuple[Array, Array, Array, Array]:
    n = x.shape[-2]
    centers = x
    mass = w
    active = w > 0.0
    roots = jnp.arange(n, dtype=jnp.int32)

    def merge_step(_, state):
        centers, mass, active, roots = state
        difference = centers[:, None, :] - centers[None, :, :]
        distance = jnp.real(jnp.sum(jnp.conj(difference) * difference, axis=-1))
        if linkage == "ward":
            distance = (
                distance
                * (mass[:, None] * mass[None, :])
                / jnp.maximum(mass[:, None] + mass[None, :], jnp.finfo(w.dtype).tiny)
            )
        candidates = (
            active[:, None] & active[None, :] & jnp.triu(jnp.ones((n, n), dtype=bool), 1)
        )
        should_merge = jnp.sum(active) > cluster_count
        # Row-major flattening makes the first equal-distance pair lexicographic by (left, right).
        flat = jnp.argmin(jnp.where(candidates, distance, jnp.inf).reshape((-1,))).astype(
            roots.dtype
        )
        sample_count = jnp.asarray(n, dtype=roots.dtype)
        left = flat // sample_count
        right = flat % sample_count
        total = mass[left] + mass[right]
        merged = (
            mass[left] * centers[left] + mass[right] * centers[right]
        ) / jnp.maximum(total, jnp.finfo(w.dtype).tiny)
        centers = jax.lax.cond(
            should_merge, lambda c: c.at[left].set(merged), lambda c: c, centers
        )
        mass = jax.lax.cond(
            should_merge,
            lambda m: m.at[left].set(total).at[right].set(0.0),
            lambda m: m,
            mass,
        )
        active = jax.lax.cond(
            should_merge, lambda a: a.at[right].set(False), lambda a: a, active
        )
        roots = jax.lax.cond(
            should_merge, lambda r: jnp.where(r == right, left, r), lambda r: r, roots
        )
        return centers, mass, active, roots

    centers, mass, active, roots = jax.lax.fori_loop(
        0, n, merge_step, (centers, mass, active, roots)
    )
    indices = jnp.arange(n, dtype=roots.dtype)
    active_indices = jnp.argsort(
        jnp.where(active, indices, jnp.asarray(n, dtype=roots.dtype)), stable=True
    )[:cluster_count].astype(roots.dtype)
    final_centers = centers[active_indices]
    final_mass = mass[active_indices]
    labels = jnp.argmax(roots[:, None] == active_indices[None, :], axis=-1).astype(
        jnp.int32
    )
    labels = jnp.where(w > 0.0, labels, -1)
    return final_centers, final_mass, labels, jnp.sum(active)


class AgglomerativeClustering(AbstractRecipe):
    cluster_count: int = eqx.field(static=True)
    linkage: AgglomerativeLinkage = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        cluster_count: int,
        /,
        *,
        linkage: AgglomerativeLinkage = "ward",
        weight_policy: WeightPolicy = "statistical",
    ):
        if cluster_count <= 0 or linkage not in ("ward", "centroid"):
            raise ValueError("invalid agglomerative clustering configuration.")
        self.cluster_count = int(cluster_count)
        self.linkage = linkage
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if self.cluster_count > batch.sample_count:
            raise ValueError("cluster_count cannot exceed sample capacity.")
        x, w, active_samples, invalid = active_data(batch, self.weight_policy)
        n, p = x.shape[-2:]
        case_count = 1
        for size in batch.case_shape:
            case_count *= size
        centers, mass, labels, final_count = jax.vmap(
            lambda values, weights: _agglomerate_one(
                values, weights, self.cluster_count, self.linkage
            )
        )(x.reshape((case_count, n, p)), w.reshape((case_count, n)))
        centers = centers.reshape(batch.case_shape + (self.cluster_count, p))
        mass = mass.reshape(batch.case_shape + (self.cluster_count,))
        labels = labels.reshape(batch.case_shape + (n,))
        final_count = final_count.reshape(batch.case_shape)
        active_clusters = mass > 0.0
        distance = distances_to_centers(x, centers, "squared-euclidean", batch.case_shape)
        assigned_distance = jnp.take_along_axis(
            distance, jnp.maximum(labels, 0)[..., None], axis=-1
        ).squeeze(-1)
        objective = jnp.sum(w * assigned_distance, axis=-1) / jnp.maximum(
            jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny
        )
        enough = jnp.sum(active_samples, axis=-1) >= self.cluster_count
        finite = jnp.isfinite(objective) & jnp.all(jnp.isfinite(centers), axis=(-2, -1))
        valid = enough & finite & ~invalid & (final_count == self.cluster_count)
        status = jnp.where(
            invalid | ~finite,
            ML_NONFINITE,
            jnp.where(
                ~enough | (final_count != self.cluster_count),
                ML_INSUFFICIENT_DATA,
                ML_SUCCESS,
            ),
        )
        model = HardClusterModel(
            centers, active_clusters, method="agglomerative-clustering"
        )
        diagnostics = ClusterDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=batch.sample_count,
            effective_samples=effective_sample_count(w),
            cluster_mass=mass,
            active_clusters=active_clusters,
            empty_clusters_seen=~jnp.all(active_clusters, axis=-1),
            converged=final_count == self.cluster_count,
            method="agglomerative-clustering",
        )
        contract = GradientContract(
            prediction_inputs="none",
            prediction_parameters="none",
            fit_mode="stopped",
            nondifferentiable_outputs=("labels", "merge tree"),
            conditions=("deterministic lexicographic merge ties",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="agglomerative-clustering",
            gradient_contract=contract,
        )


__all__ = ["AgglomerativeClustering", "AgglomerativeLinkage", "SpectralClustering"]
