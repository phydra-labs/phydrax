#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

import phydrax.ein as ein

from ..._differentiation import (
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from ..._trainable import fixed_field
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    ML_CAPACITY_EXHAUSTED,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
    prediction_fit_contract,
)
from .._numerics._pairwise import pair_block_size
from .._schema import AbstractFittedModel
from ..neighbors._utils import blocked_case_top_k, gather_support
from ._common import (
    active_data,
    ClusterDiagnostics,
    distances_to_centers,
    effective_sample_count,
    pairwise_distances,
    positive_scalar,
    real_dtype,
)


# Radius labels of the nearest core point are locally constant and stopped.
_RADIUS_LABEL_CONTRACT = DerivativeContract(
    (
        SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.NONE),
        SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, GradientLevel.NONE),
    ),
    route=DerivativeRoute.DIRECT,
    regularity=DerivativeRegularity.piecewise_polynomial(continuity=-1, degree_bound=0),
    nondifferentiable_outputs=("labels",),
)


class DensityClusterModel(AbstractFittedModel):
    """Fixed-capacity core-point model with hard radius labels and smooth memberships."""

    core_points: Array = fixed_field()
    core_labels: Array
    core_active: Array
    cluster_active: Array
    radius: Array = fixed_field()
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
    ) -> None:
        self.core_points = jnp.asarray(core_points)
        self.core_labels = jnp.asarray(core_labels, dtype=jnp.int32)
        self.core_active = jnp.asarray(core_active, dtype=jnp.bool_)
        self.cluster_active = jnp.asarray(cluster_active, dtype=jnp.bool_)
        self.radius = jnp.asarray(radius, dtype=real_dtype(self.core_points.dtype))
        self.in_size = self.core_points.shape[-1]
        self.out_size = "scalar"
        self.case_shape = self.core_points.shape[:-2]
        self.cluster_capacity = self.cluster_active.shape[-1]
        self.method = str(method)

    def _prediction_contract(self) -> DerivativeContract:
        return _RADIUS_LABEL_CONTRACT

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
        indices, distances = blocked_case_top_k(
            jnp.asarray(x),
            self.core_points,
            self.core_active,
            self.case_shape,
            "euclidean",
            1,
        )
        label = gather_support(self.core_labels[..., None], indices, self.case_shape)[
            ..., 0, 0
        ]
        minimum = distances[..., 0]
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


def _radius_reduce(
    x: Array,
    active: Array,
    radius: Array | float,
    values: Array,
    /,
    *,
    minimum: bool,
    block_size: int = 64,
) -> Array:
    n = x.shape[0]
    b = block_size
    padded = ((n + b - 1) // b) * b
    points = jnp.pad(x, ((0, padded - n), (0, 0)))
    mask = jnp.pad(active, ((0, padded - n),))
    fill = n if minimum else 0
    source = jnp.pad(values, ((0, padded - n),), constant_values=fill)
    output = jnp.full((padded,), fill, dtype=values.dtype)

    def row_step(i: Array, output: Array) -> Array:
        query = jax.lax.dynamic_slice_in_dim(points, i * b, b)
        query_mask = jax.lax.dynamic_slice_in_dim(mask, i * b, b)

        def column_step(j: Array, total: Array) -> Array:
            support = jax.lax.dynamic_slice_in_dim(points, j * b, b)
            support_mask = jax.lax.dynamic_slice_in_dim(mask, j * b, b)
            value = jax.lax.dynamic_slice_in_dim(source, j * b, b)
            difference = query[:, None, :] - support[None, :, :]
            distance = jnp.sqrt(
                jnp.maximum(
                    jnp.real(jnp.sum(jnp.conj(difference) * difference, axis=-1)), 0.0
                )
            )
            adjacency = (distance <= radius) & query_mask[:, None] & support_mask[None, :]
            if minimum:
                return jnp.minimum(
                    total, jnp.min(jnp.where(adjacency, value[None, :], n), axis=-1)
                )
            return total + jnp.where(adjacency, value[None, :], 0.0).sum(axis=-1)

        reduced = jax.lax.fori_loop(
            0, padded // b, column_step, jnp.full((b,), fill, dtype=values.dtype)
        )
        return jax.lax.dynamic_update_slice_in_dim(output, reduced, i * b, axis=0)

    return jax.lax.fori_loop(0, padded // b, row_step, output)[:n]


def _density_one(
    x: Array,
    w: Array,
    radius: float,
    minimum_mass: float,
    cluster_capacity: int,
    core_capacity: int,
    workspace_bytes: int = 4 * 1024 * 1024,
) -> tuple[Array, ...]:
    n, p = x.shape
    active = w > 0.0
    b = pair_block_size(
        n, n, p, jnp.dtype(jnp.result_type(x, w)).itemsize, workspace_bytes
    )
    adjacency = None
    if b >= n:
        adjacency = (
            (pairwise_distances(x, "euclidean") <= radius)
            & active[:, None]
            & active[None, :]
        )
        neighborhood_mass = adjacency @ w
    else:
        neighborhood_mass = _radius_reduce(
            x, active, radius, w, minimum=False, block_size=b
        )
    core = active & (neighborhood_mass >= minimum_mass)
    sentinel = jnp.asarray(n, dtype=jnp.int32)
    root = jnp.where(core, jnp.arange(n, dtype=jnp.int32), sentinel)
    core_edges = None if adjacency is None else adjacency & core[:, None] & core[None, :]

    def propagate(_: Array, labels: Array) -> Array:
        if core_edges is None:
            candidate = _radius_reduce(
                x, core, radius, labels, minimum=True, block_size=b
            )
        else:
            candidate = jnp.min(jnp.where(core_edges, labels[None, :], sentinel), axis=-1)
        return jnp.where(core, jnp.minimum(labels, candidate), sentinel)

    root = jax.lax.fori_loop(0, n, propagate, root)
    adjacent_root = (
        _radius_reduce(
            x, active, radius, jnp.where(core, root, sentinel), minimum=True, block_size=b
        )
        if adjacency is None
        else jnp.min(
            jnp.where(adjacency & core[None, :], root[None, :], sentinel), axis=-1
        )
    )
    root = jnp.where(
        core,
        root,
        jnp.where(active & (adjacent_root < sentinel), adjacent_root, sentinel),
    )
    is_root = core & (root == jnp.arange(n, dtype=jnp.int32))
    cluster_count = jnp.sum(is_root)
    root_prefix = jnp.cumsum(is_root.astype(jnp.int32)) - is_root.astype(jnp.int32)
    cluster_label = root_prefix[jnp.minimum(root, n - 1)]
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
    workspace_bytes: int,
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
            values,
            weights,
            radius,
            minimum_mass,
            cluster_capacity,
            core_capacity,
            workspace_bytes // case_count,
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
    contract = prediction_fit_contract(
        model._prediction_contract(),
        route=DerivativeRoute.STOPPED,
        nondifferentiable_outputs=("core mask", "connected components"),
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
        derivative_contract=contract,
    )


class DBSCAN(AbstractRecipe):
    cluster_capacity: int = eqx.field(static=True)
    core_capacity: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    minimum_samples: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        cluster_capacity: int,
        core_capacity: int,
        /,
        *,
        radius: float = 0.5,
        minimum_samples: float = 5.0,
        weight_policy: WeightPolicy = "statistical",
        workspace_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if workspace_bytes <= 0:
            raise ValueError("workspace_bytes must be positive.")
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
        self.workspace_bytes = workspace_bytes

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
            workspace_bytes=self.workspace_bytes,
        )


class ConnectivityClustering(AbstractRecipe):
    cluster_capacity: int = eqx.field(static=True)
    representative_capacity: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        cluster_capacity: int,
        representative_capacity: int,
        /,
        *,
        radius: float = 1.0,
        weight_policy: WeightPolicy = "statistical",
        workspace_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if workspace_bytes <= 0:
            raise ValueError("workspace_bytes must be positive.")
        if cluster_capacity <= 0 or representative_capacity <= 0 or radius <= 0.0:
            raise ValueError("invalid connectivity clustering configuration.")
        self.cluster_capacity = int(cluster_capacity)
        self.representative_capacity = int(representative_capacity)
        self.radius = float(radius)
        self.weight_policy = weight_policy
        self.workspace_bytes = workspace_bytes

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
            workspace_bytes=self.workspace_bytes,
        )


__all__ = ["ConnectivityClustering", "DBSCAN", "DensityClusterModel"]
