#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from ..._model import AbstractArrayModel
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_CAPACITY_EXHAUSTED,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from ._common import (
    active_data,
    ClusterDiagnostics,
    distances_to_centers,
    effective_sample_count,
    pairwise_distances,
    positive_scalar,
    real_dtype,
)


class DensityClusterModel(AbstractArrayModel):
    """Fixed-capacity core-point model with hard radius labels and smooth memberships."""

    core_points: Array
    core_labels: Array
    core_active: Array
    cluster_active: Array
    radius: Array
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    cluster_capacity: int = eqx.field(static=True)
    method: str = eqx.field(static=True)

    def __init__(
        self,
        core_points: Array,
        core_labels: Array,
        core_active: Array,
        cluster_active: Array,
        radius: Any,
        /,
        *,
        method: str,
    ):
        self.core_points = jnp.asarray(core_points)
        self.core_labels = jnp.asarray(core_labels, dtype=jnp.int32)
        self.core_active = jnp.asarray(core_active, dtype=bool)
        self.cluster_active = jnp.asarray(cluster_active, dtype=bool)
        self.radius = jnp.asarray(radius, dtype=real_dtype(self.core_points.dtype))
        self.in_size = self.core_points.shape[-1]
        self.out_size = "scalar"
        self.case_shape = self.core_points.shape[:-2]
        self.cluster_capacity = self.cluster_active.shape[-1]
        self.method = str(method)

    def core_distances(self, x: Any, /) -> Array:
        distance = distances_to_centers(
            jnp.asarray(x), self.core_points, "euclidean", self.case_shape
        )
        sample_ndim = distance.ndim - len(self.case_shape) - 1
        active = self.core_active.reshape(
            self.case_shape + (1,) * sample_ndim + (self.core_points.shape[-2],)
        )
        return jnp.where(active, distance, jnp.inf)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        distance = self.core_distances(x)
        nearest = jnp.argmin(distance, axis=-1)
        sample_ndim = distance.ndim - len(self.case_shape) - 1
        labels = self.core_labels.reshape(
            self.case_shape + (1,) * sample_ndim + (self.core_points.shape[-2],)
        )
        label = jnp.take_along_axis(labels, nearest[..., None], axis=-1).squeeze(-1)
        minimum = jnp.min(distance, axis=-1)
        return jax.lax.stop_gradient(
            jnp.where(minimum <= self.radius, label, -1).astype(jnp.int32)
        )

    def soft_membership(self, x: Any, /, *, temperature: float = 1.0) -> Array:
        temperature_ = positive_scalar(temperature, "temperature")
        distance = self.core_distances(x)
        sample_ndim = distance.ndim - len(self.case_shape) - 1
        labels = self.core_labels.reshape(
            self.case_shape + (1,) * sample_ndim + (self.core_points.shape[-2],)
        )
        active_core = self.core_active.reshape(
            self.case_shape + (1,) * sample_ndim + (self.core_points.shape[-2],)
        )
        membership = jnp.exp(-0.5 * jnp.square(distance / temperature_)) * active_core
        one_hot = jax.nn.one_hot(
            jnp.maximum(labels, 0), self.cluster_capacity, dtype=membership.dtype
        )
        cluster_score = ein.contract("...q,...qk->...k", membership, one_hot)
        active_cluster = self.cluster_active.reshape(
            self.case_shape + (1,) * sample_ndim + (self.cluster_capacity,)
        )
        cluster_score = jnp.where(active_cluster, cluster_score, 0.0)
        total = jnp.sum(cluster_score, axis=-1, keepdims=True)
        return jnp.where(total > 0.0, cluster_score / total, 0.0)


def _density_one(
    x: Array,
    w: Array,
    radius: float,
    minimum_mass: float,
    cluster_capacity: int,
    core_capacity: int,
) -> tuple[Array, ...]:
    n, p = x.shape
    active = w > 0.0
    distance = pairwise_distances(x, "euclidean")
    adjacency = (distance <= radius) & active[:, None] & active[None, :]
    neighborhood_mass = adjacency @ w
    core = active & (neighborhood_mass >= minimum_mass)
    sentinel = jnp.asarray(n, dtype=jnp.int32)
    root = jnp.where(core, jnp.arange(n, dtype=jnp.int32), sentinel)
    core_edges = adjacency & core[:, None] & core[None, :]

    def propagate(_, labels):
        candidate = jnp.where(core_edges, labels[None, :], sentinel)
        return jnp.where(core, jnp.minimum(labels, jnp.min(candidate, axis=-1)), sentinel)

    root = jax.lax.fori_loop(0, n, propagate, root)
    adjacent_root = jnp.min(
        jnp.where(adjacency & core[None, :], root[None, :], sentinel), axis=-1
    )
    root = jnp.where(
        core,
        root,
        jnp.where(active & (adjacent_root < sentinel), adjacent_root, sentinel),
    )
    is_root = core & (root == jnp.arange(n, dtype=jnp.int32))
    cluster_count = jnp.sum(is_root)
    cluster_label = jnp.sum(
        is_root[None, :] & (jnp.arange(n)[None, :] < root[:, None]), axis=-1
    ).astype(jnp.int32)
    labels = jnp.where(root < sentinel, cluster_label, -1)
    within_capacity = labels < cluster_capacity
    labels = jnp.where(within_capacity, labels, -1)
    membership = (
        jax.nn.one_hot(jnp.maximum(labels, 0), cluster_capacity, dtype=w.dtype)
        * ((labels >= 0) * w)[:, None]
    )
    cluster_mass = jnp.sum(membership, axis=0)
    centers = (
        membership.T @ x / jnp.maximum(cluster_mass, jnp.finfo(w.dtype).tiny)[:, None]
    )
    core_order = jnp.argsort(jnp.where(core & within_capacity, jnp.arange(n), n))[
        :core_capacity
    ]
    core_active = (core_order < n) & core[core_order] & within_capacity[core_order]
    safe_order = jnp.minimum(core_order, n - 1)
    core_points = x[safe_order]
    core_labels = jnp.where(core_active, labels[safe_order], -1)
    return (
        centers,
        cluster_mass,
        labels,
        core_points,
        core_labels,
        core_active,
        cluster_count,
        jnp.sum(core),
        jnp.sum(active),
    )


def _fit_density(
    batch: MLBatch,
    *,
    radius: float,
    minimum_mass: float,
    cluster_capacity: int,
    core_capacity: int,
    weight_policy: WeightPolicy,
    method: str,
) -> FitResult:
    if cluster_capacity > batch.sample_count or core_capacity > batch.sample_count:
        raise ValueError("cluster and core capacities cannot exceed sample capacity.")
    x, w, active, invalid = active_data(batch, weight_policy)
    n, p = x.shape[-2:]
    case_count = 1
    for size in batch.case_shape:
        case_count *= size
    outputs = jax.vmap(
        lambda values, weights: _density_one(
            values, weights, radius, minimum_mass, cluster_capacity, core_capacity
        )
    )(x.reshape((case_count, n, p)), w.reshape((case_count, n)))
    (
        centers,
        mass,
        labels,
        core_points,
        core_labels,
        core_active,
        cluster_count,
        core_count,
        active_count,
    ) = outputs
    centers = centers.reshape(batch.case_shape + (cluster_capacity, p))
    mass = mass.reshape(batch.case_shape + (cluster_capacity,))
    labels = labels.reshape(batch.case_shape + (n,))
    core_points = core_points.reshape(batch.case_shape + (core_capacity, p))
    core_labels = core_labels.reshape(batch.case_shape + (core_capacity,))
    core_active = core_active.reshape(batch.case_shape + (core_capacity,))
    cluster_count = cluster_count.reshape(batch.case_shape)
    core_count = core_count.reshape(batch.case_shape)
    active_count = active_count.reshape(batch.case_shape)
    active_clusters = mass > 0.0
    exhausted = (cluster_count > cluster_capacity) | (core_count > core_capacity)
    finite = jnp.all(jnp.isfinite(centers), axis=(-2, -1)) & jnp.all(
        jnp.isfinite(core_points), axis=(-2, -1)
    )
    enough = (active_count > 0) & (core_count > 0)
    valid = enough & finite & ~invalid & ~exhausted
    status = jnp.where(
        invalid | ~finite,
        ML_NONFINITE,
        jnp.where(
            ~enough,
            ML_INSUFFICIENT_DATA,
            jnp.where(exhausted, ML_CAPACITY_EXHAUSTED, ML_SUCCESS),
        ),
    )
    assigned = labels >= 0
    objective = 1.0 - jnp.sum(jnp.where(assigned, w, 0.0), axis=-1) / jnp.maximum(
        jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny
    )
    model = DensityClusterModel(
        core_points, core_labels, core_active, active_clusters, radius, method=method
    )
    diagnostics = ClusterDiagnostics(
        valid=valid,
        status=status,
        objective=objective,
        iterations=batch.sample_count,
        effective_samples=effective_sample_count(w),
        cluster_mass=mass,
        active_clusters=active_clusters,
        empty_clusters_seen=False,
        converged=jnp.ones_like(valid),
        degeneracy=exhausted | (core_count == 0),
        method=method,
    )
    contract = GradientContract(
        prediction_inputs="conditional",
        prediction_parameters="conditional",
        fit_mode="stopped",
        nondifferentiable_outputs=("labels", "core mask", "connected components"),
        conditions=(
            "soft_membership is smooth away from zero normalization",
            "hard radius labels are terminal",
            "fixed capacities",
        ),
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        gradient_contract=contract,
    )


class DBSCAN(AbstractRecipe):
    cluster_capacity: int = eqx.field(static=True)
    core_capacity: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    minimum_samples: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        cluster_capacity: int,
        core_capacity: int,
        /,
        *,
        radius: float = 0.5,
        minimum_samples: float = 5.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        if (
            cluster_capacity <= 0
            or core_capacity <= 0
            or radius <= 0.0
            or minimum_samples <= 0.0
        ):
            raise ValueError("invalid DBSCAN configuration.")
        self.cluster_capacity = int(cluster_capacity)
        self.core_capacity = int(core_capacity)
        self.radius = float(radius)
        self.minimum_samples = float(minimum_samples)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_density(
            batch,
            radius=self.radius,
            minimum_mass=self.minimum_samples,
            cluster_capacity=self.cluster_capacity,
            core_capacity=self.core_capacity,
            weight_policy=self.weight_policy,
            method="dbscan",
        )


class ConnectivityClustering(AbstractRecipe):
    cluster_capacity: int = eqx.field(static=True)
    representative_capacity: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        cluster_capacity: int,
        representative_capacity: int,
        /,
        *,
        radius: float = 1.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        if cluster_capacity <= 0 or representative_capacity <= 0 or radius <= 0.0:
            raise ValueError("invalid connectivity clustering configuration.")
        self.cluster_capacity = int(cluster_capacity)
        self.representative_capacity = int(representative_capacity)
        self.radius = float(radius)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_density(
            batch,
            radius=self.radius,
            minimum_mass=0.0,
            cluster_capacity=self.cluster_capacity,
            core_capacity=self.representative_capacity,
            weight_policy=self.weight_policy,
            method="connectivity-clustering",
        )


__all__ = ["ConnectivityClustering", "DBSCAN", "DensityClusterModel"]
