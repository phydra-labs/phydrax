#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._differentiation import (
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from ..._model._array import value_derivative_contract
from ..._sampling._addressing import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import fixed_field
from ...sparse import EdgeRelation, RelationAccumulation, route_reduce
from ...typing import parse, PRNGKey
from .._batch import MLBatch, WeightPolicy
from .._contracts import prediction_fit_contract
from .._numerics import MetricName
from .._schema import AbstractFittedModel
from ..neighbors._utils import blocked_case_top_k, distance_softmax_regularity


ClusterInitialization: TypeAlias = Literal["random", "first", "k-means++"]
EmptyClusterPolicy: TypeAlias = Literal["retain", "reseed", "error"]
SpectralEigenMethod: TypeAlias = Literal["dense-eigh", "restarted-lanczos"]
MiniBatchBatchLaw: TypeAlias = Literal["quality", "throughput"]
ClusterRevival: TypeAlias = Literal["split", "off"]

EMA_MINI_BATCH_METHOD = "ema-mini-batch-k-means"

# Default bound on transient pair workspace for assignment and distance tiles.
# Below about 64 MiB, per-tile loop overhead made Lloyd iterations and
# predictions slower than the unbounded dense route they replace.
WORKSPACE_BYTES = 64 * 1024 * 1024
_RANDOM_ADDRESS = SampleAddress(
    "phydrax.ml.clustering", "center-initialization", role="random-rows"
)
_PLUS_PLUS_ADDRESS = SampleAddress(
    "phydrax.ml.clustering", "center-initialization", role="k-means++"
)

# Hard labels are locally constant between assignment boundaries and stopped.
_HARD_LABEL_CONTRACT = DerivativeContract(
    (SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, GradientLevel.NONE),),
    route=DerivativeRoute.DIRECT,
    regularity=DerivativeRegularity.piecewise_polynomial(continuity=-1, degree_bound=0),
    nondifferentiable_outputs=("labels",),
)


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


def _center_distances(values: Array, centers: Array, metric: MetricName) -> Array:
    """Distances of case + (row, feature) values to case + (center, feature) centers."""
    difference = values[..., :, None, :] - centers[..., None, :, :]
    squared = jnp.real(jnp.sum(jnp.conj(difference) * difference, axis=-1))
    match metric:
        case "squared-euclidean":
            return jnp.maximum(squared, 0.0)
        case "euclidean":
            return jnp.sqrt(jnp.maximum(squared, 0.0))
        case "manhattan":
            return jnp.sum(jnp.abs(difference), axis=-1)
        case "cosine":
            numerator = jnp.real(
                jnp.sum(
                    jnp.conj(values[..., :, None, :]) * centers[..., None, :, :], axis=-1
                )
            )
            left_norm = jnp.sqrt(jnp.real(jnp.sum(jnp.conj(values) * values, axis=-1)))
            right_norm = jnp.sqrt(jnp.real(jnp.sum(jnp.conj(centers) * centers, axis=-1)))
            similarity = numerator / jnp.maximum(
                left_norm[..., :, None] * right_norm[..., None, :],
                jnp.finfo(real_dtype(values.dtype)).tiny,
            )
            return jnp.maximum(1.0 - similarity, 0.0)
        case _:
            raise ValueError(f"unsupported metric {metric!r}.")


def distances_to_centers(
    values: Array,
    centers: Array,
    metric: MetricName,
    case_shape: tuple[int, ...],
    *,
    workspace_bytes: int = WORKSPACE_BYTES,
) -> Array:
    """Exact difference-form distances, streamed over sample rows within a workspace.

    Row chunks are rematerialized under reverse-mode differentiation so the
    pair-feature differences of only one chunk are ever live.
    """
    minimum_rank = len(case_shape) + 1
    if values.ndim < minimum_rank:
        raise ValueError("case axes and the final feature axis must be distinct.")
    if (
        values.shape[: len(case_shape)] != case_shape
        or values.shape[-1] != centers.shape[-1]
    ):
        raise ValueError("input must have shape case + sample_shape + (feature,).")
    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be positive.")
    sample_shape = values.shape[len(case_shape) : -1]
    samples = prod(sample_shape)
    clusters, features = centers.shape[-2:]
    flat = values.reshape(case_shape + (samples, features))
    row_bytes = (
        prod(case_shape)
        * clusters
        * (features + 3)
        * jnp.dtype(jnp.result_type(values, centers)).itemsize
    )
    rows = max(1, workspace_bytes // max(row_bytes, 1))
    if rows >= samples:
        distances = _center_distances(flat, centers, metric)
    else:
        chunks = -(-samples // rows)
        rows = -(-samples // chunks)
        axis = len(case_shape)
        padded = jnp.pad(flat, ((0, 0),) * axis + ((0, chunks * rows - samples), (0, 0)))
        blocks = jnp.moveaxis(
            padded.reshape(case_shape + (chunks, rows, features)), axis, 0
        )
        distances = jax.lax.map(
            jax.checkpoint(lambda block: _center_distances(block, centers, metric)),
            blocks,
        )
        distances = jnp.moveaxis(distances, 0, axis).reshape(
            case_shape + (chunks * rows, clusters)
        )[..., :samples, :]
    return distances.reshape(case_shape + sample_shape + (clusters,))


def cluster_sums(
    payload: tuple[Array, ...],
    labels: Array,
    valid: Array,
    cluster_count: int,
    /,
    *,
    accumulation: RelationAccumulation,
) -> tuple[Array, ...]:
    """Per-cluster sums of case + (row,) + event payloads on the sparse route substrate.

    Labels change every call, so each case reduces one unprepared edge relation
    from rows to clusters; ``accumulation`` selects the route summation order.
    """
    case_shape = labels.shape[:-1]
    n = labels.shape[-1]
    cases = prod(case_shape)
    sources = np.arange(n, dtype=np.int32)

    def reduce(
        case_labels: Array, case_valid: Array, case_payload: tuple[Array, ...]
    ) -> tuple[Array, ...]:
        relation = EdgeRelation(
            sources,
            case_labels,
            source_size=n,
            target_size=cluster_count,
            valid=case_valid,
        )
        return route_reduce(relation, case_payload, accumulation=accumulation)

    flat = tuple(
        leaf.reshape((cases, n) + leaf.shape[len(case_shape) + 1 :]) for leaf in payload
    )
    reduced = jax.vmap(reduce)(
        labels.reshape((cases, n)), valid.reshape((cases, n)), flat
    )
    return tuple(leaf.reshape(case_shape + leaf.shape[1:]) for leaf in reduced)


def nearest_centers(
    values: Array,
    centers: Array,
    active_clusters: Array,
    case_shape: tuple[int, ...],
    /,
    *,
    count: int = 1,
    workspace_bytes: int = WORKSPACE_BYTES,
) -> tuple[Array, Array]:
    """Nearest active centers in squared Euclidean distance, lowest index on ties."""
    return blocked_case_top_k(
        values,
        centers,
        active_clusters,
        case_shape,
        "squared-euclidean",
        count,
        workspace_bytes=workspace_bytes,
    )


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


def _random_rows(values: Array, weights: Array, count: int, key: PRNGKey) -> Array:
    """Distinct rows drawn without replacement in proportion to their weights."""
    noise = jax.random.gumbel(key, weights.shape, dtype=weights.dtype)
    scores = jnp.where(weights > 0.0, jnp.log(weights) + noise, -jnp.inf)
    _, indices = jax.lax.top_k(scores, count)
    return values[indices]


def _plus_plus_rows(
    values: Array, weights: Array, count: int, root: PRNGKey, case: Array
) -> Array:
    """Weighted k-means++ seeding with a running nearest-seed distance."""

    def squared(index: Array) -> Array:
        difference = values - values[index]
        return jnp.real(jnp.sum(jnp.conj(difference) * difference, axis=-1))

    def draw(scores: Array, step: Array | int) -> Array:
        logits = jnp.where(scores > 0.0, jnp.log(scores), -jnp.inf)
        key = derive_key(root, _PLUS_PLUS_ADDRESS, case, step)
        return jax.random.categorical(key, logits).astype(jnp.int32)

    first = draw(weights, 0)
    indices = jnp.zeros((count,), dtype=jnp.int32).at[0].set(first)

    def add(i: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
        indices, nearest = state
        scores = weights * nearest
        # Only when every weighted row coincides with a seed must a seed repeat.
        scores = jnp.where(jnp.sum(scores) > 0.0, scores, weights)
        selected = draw(scores, i)
        return indices.at[i].set(selected), jnp.minimum(nearest, squared(selected))

    indices, _ = jax.lax.fori_loop(1, count, add, (indices, squared(first)))
    return values[indices]


def initialize_centers(
    x: Array,
    w: Array,
    cluster_count: int,
    initialization: ClusterInitialization,
    key: Any,
) -> Array:
    case_shape = x.shape[:-2]
    n, p = x.shape[-2:]
    case_count = prod(case_shape)
    flat_x = x.reshape((case_count, n, p))
    flat_w = w.reshape((case_count, n))
    initialization = parse(initialization, ClusterInitialization, "initialization")
    if initialization != "first" and key is None:
        raise ValueError(f"{initialization} initialization requires an explicit JAX key.")
    cases = jnp.arange(case_count, dtype=jnp.uint32)
    match initialization:
        case "first":
            order = jnp.argsort(
                jnp.where(flat_w > 0.0, jnp.arange(n), n), axis=-1, stable=True
            )
            indices = order[:, :cluster_count]
            centers = jnp.take_along_axis(flat_x, indices[..., None], axis=-2)
        case "random":
            root = parse(key, PRNGKey, "key")
            centers = jax.vmap(
                lambda values, weights, case: _random_rows(
                    values,
                    weights,
                    cluster_count,
                    derive_key(root, _RANDOM_ADDRESS, case),
                )
            )(flat_x, flat_w, cases)
        case "k-means++":
            root = parse(key, PRNGKey, "key")
            centers = jax.vmap(
                lambda values, weights, case: _plus_plus_rows(
                    values, weights, cluster_count, root, case
                )
            )(flat_x, flat_w, cases)
        case _:
            assert_never(initialization)
    return centers.reshape(case_shape + (cluster_count, p))


class MiniBatchClusterEvidence(StrictModule):
    """Resolved schedule and per-case evidence of an EMA mini-batch k-means fit.

    ``batch_size``, ``batch_law`` (``None`` for an explicit batch size),
    ``epochs`` and ``epoch_fractions`` are static schedule identity. The arrays
    carry the fitted case axes: the starting step, the number of updates, each
    epoch's assignment-time objective over its processed rows, the number of
    revival splits, the closing gain of the exact last-epoch means, the
    last-epoch imbalance, and whether the schedule lies inside the domain the
    scaling laws were fitted on.
    """

    initial_step: Array
    updates: Array
    epoch_objectives: Array
    revivals: Array
    closing_gain: Array
    imbalance: Array
    in_fitted_domain: Array
    batch_size: int = eqx.field(static=True)
    batch_law: MiniBatchBatchLaw | None = eqx.field(static=True)
    epochs: float = eqx.field(static=True)
    epoch_fractions: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_step: Array,
        updates: Array,
        epoch_objectives: Array,
        revivals: Array,
        closing_gain: Array,
        imbalance: Array,
        in_fitted_domain: Array,
        batch_size: int,
        batch_law: MiniBatchBatchLaw | None,
        epochs: float,
        epoch_fractions: tuple[float, ...],
    ) -> None:
        if batch_size <= 0 or not epochs > 0.0 or not epoch_fractions:
            raise ValueError(
                "Mini-batch schedule identity must be positive and nonempty."
            )
        case_shape = jnp.shape(updates)
        per_case = (initial_step, revivals, closing_gain, imbalance, in_fitted_domain)
        if any(jnp.shape(value) != case_shape for value in per_case) or jnp.shape(
            epoch_objectives
        ) != case_shape + (len(epoch_fractions),):
            raise ValueError("Mini-batch evidence must share the fitted case axes.")
        self.initial_step = jnp.asarray(initial_step)
        self.updates = jnp.asarray(updates, dtype=jnp.int32)
        self.epoch_objectives = jnp.asarray(epoch_objectives)
        self.revivals = jnp.asarray(revivals, dtype=jnp.int32)
        self.closing_gain = jnp.asarray(closing_gain)
        self.imbalance = jnp.asarray(imbalance)
        self.in_fitted_domain = jnp.asarray(in_fitted_domain, dtype=jnp.bool_)
        self.batch_size = int(batch_size)
        self.batch_law = (
            None
            if batch_law is None
            else parse(batch_law, MiniBatchBatchLaw, "batch_law")
        )
        self.epochs = float(epochs)
        self.epoch_fractions = tuple(float(value) for value in epoch_fractions)


class ClusterDiagnostics(StrictModule):
    """Consumer fit evidence, including spectral arrays and static method identity.

    Spectral fits require matching ``eigenvalues`` and ``eigen_residuals`` arrays
    with leading fitted case axes, plus static ``eigen_method`` metadata.
    Non-spectral fits require all three fields to be ``None``. EMA mini-batch
    k-means fits, and only they, carry ``minibatch`` schedule evidence. These
    numerical leaves are part of the diagnostics' PyTree and persistence
    template.
    """

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
    eigenvalues: Array | None
    eigen_residuals: Array | None
    eigen_method: SpectralEigenMethod | None = eqx.field(static=True)
    minibatch: MiniBatchClusterEvidence | None

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
        eigenvalues: Array | None = None,
        eigen_residuals: Array | None = None,
        eigen_method: SpectralEigenMethod | None = None,
        minibatch: MiniBatchClusterEvidence | None = None,
    ) -> None:
        spectral = method == "spectral-clustering"
        present = (
            eigenvalues is not None,
            eigen_residuals is not None,
            eigen_method is not None,
        )
        if (spectral and not all(present)) or (not spectral and any(present)):
            raise ValueError(
                "Spectral evidence must be present only for spectral clustering."
            )
        if eigenvalues is not None and eigen_residuals is not None:
            if eigenvalues.shape != eigen_residuals.shape:
                raise ValueError("Eigenvalues and residuals must have identical shapes.")
            if eigenvalues.shape[:-1] != jnp.shape(valid):
                raise ValueError("Spectral evidence must match the fitted case axes.")
        if (method == EMA_MINI_BATCH_METHOD) != (minibatch is not None):
            raise ValueError(
                "Mini-batch evidence must be present only for EMA mini-batch k-means."
            )
        if minibatch is not None and minibatch.updates.shape != jnp.shape(valid):
            raise ValueError("Mini-batch evidence must match the fitted case axes.")
        parsed_eigen_method = (
            None
            if eigen_method is None
            else parse(eigen_method, SpectralEigenMethod, "eigen_method")
        )
        self.eigenvalues = eigenvalues
        self.eigen_residuals = eigen_residuals
        self.eigen_method = parsed_eigen_method
        self.minibatch = minibatch
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.cluster_mass = jnp.asarray(cluster_mass)
        self.active_clusters = jnp.asarray(active_clusters, dtype=jnp.bool_)
        self.empty_clusters_seen = jnp.asarray(empty_clusters_seen, dtype=jnp.bool_)
        self.converged = jnp.asarray(converged, dtype=jnp.bool_)
        self.degeneracy = jnp.asarray(degeneracy, dtype=jnp.bool_)
        self.method = str(method)


class HardClusterModel(AbstractFittedModel):
    """Terminal nondifferentiable nearest-representative assignment."""

    centers: Array = fixed_field()
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
    ) -> None:
        self.centers = jnp.asarray(centers)
        self.active_clusters = jnp.asarray(active_clusters, dtype=jnp.bool_)
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

    def _prediction_contract(self) -> DerivativeContract:
        return _HARD_LABEL_CONTRACT

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        indices, _ = blocked_case_top_k(
            jnp.asarray(x),
            self.centers,
            self.active_clusters,
            self.case_shape,
            self.metric,
            1,
            workspace_bytes=WORKSPACE_BYTES,
        )
        return jax.lax.stop_gradient(indices[..., 0])


class SoftClusterModel(AbstractFittedModel):
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
    ) -> None:
        self.centers = jnp.asarray(centers)
        self.active_clusters = jnp.asarray(active_clusters, dtype=jnp.bool_)
        self.temperature = positive_scalar(
            jnp.asarray(temperature, dtype=real_dtype(self.centers.dtype)), "temperature"
        )
        self.in_size = self.centers.shape[-1]
        self.out_size = self.centers.shape[-2]
        self.case_shape = self.centers.shape[:-2]
        self.metric = metric
        self.method = str(method)

    def _prediction_contract(self) -> DerivativeContract:
        return prediction_fit_contract(
            value_derivative_contract(distance_softmax_regularity(self.metric)),
            route=DerivativeRoute.DIRECT,
            nondifferentiable_outputs=("hard_labels",),
        )

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
    "ClusterRevival",
    "EMA_MINI_BATCH_METHOD",
    "EmptyClusterPolicy",
    "HardClusterModel",
    "MiniBatchBatchLaw",
    "MiniBatchClusterEvidence",
    "SoftClusterModel",
    "stable_top_indices",
    "active_data",
    "cluster_sums",
    "effective_sample_count",
    "distances_to_centers",
    "initialize_centers",
    "nearest_centers",
    "WORKSPACE_BYTES",
    "pairwise_distances",
    "nonnegative_scalar",
    "positive_scalar",
    "real_dtype",
    "validated_scalar",
]
