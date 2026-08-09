#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import ModelBinding
from ..._strict import StrictModule
from ...sparse import RowRelation
from .._batch import MLBatch
from .._contracts import (
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import pairwise_distances


_BLOCKWISE_BINDING = ModelBinding.blockwise("flat", pass_key=True)


class NeighborhoodGraph(StrictModule):
    """Fixed-width hard nearest-neighbor topology with conditional edge lengths."""

    relation: RowRelation
    distances: Array
    adjacency: Array
    active: Array
    components: Array
    minimum_degree: Array
    maximum_degree: Array
    metric: str = eqx.field(static=True)
    topology_gradient: str = eqx.field(static=True)

    def __init__(
        self,
        relation: RowRelation,
        distances: ArrayLike,
        adjacency: ArrayLike,
        active: ArrayLike,
        components: ArrayLike,
        minimum_degree: ArrayLike,
        maximum_degree: ArrayLike,
        *,
        metric: str,
    ):
        self.relation = relation
        self.distances = jnp.asarray(distances)
        self.adjacency = jnp.asarray(adjacency, dtype=bool)
        self.active = jnp.asarray(active, dtype=bool)
        self.components = jnp.asarray(components, dtype=jnp.int32)
        self.minimum_degree = jnp.asarray(minimum_degree, dtype=jnp.int32)
        self.maximum_degree = jnp.asarray(maximum_degree, dtype=jnp.int32)
        self.metric = str(metric)
        self.topology_gradient = (
            "none; selected edge lengths are conditional on fixed topology"
        )


class ManifoldDiagnostics(StrictModule):
    """Connectivity, convergence, and spectral diagnostics for an embedding fit."""

    valid: Array
    status: Array
    objective: Array
    iterations: Array
    effective_samples: Array
    residual: Array
    eigenvalues: Array
    connected_components: Array
    minimum_degree: Array
    maximum_degree: Array
    converged: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        objective: Any = jnp.nan,
        iterations: Any = 0,
        effective_samples: Any = 0,
        residual: Any = jnp.nan,
        eigenvalues: Any = jnp.asarray([]),
        connected_components: Any = 1,
        minimum_degree: Any = 0,
        maximum_degree: Any = 0,
        converged: Any = True,
        method: str,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.residual = jnp.asarray(residual)
        self.eigenvalues = jnp.asarray(eigenvalues)
        self.connected_components = jnp.asarray(connected_components, dtype=jnp.int32)
        self.minimum_degree = jnp.asarray(minimum_degree, dtype=jnp.int32)
        self.maximum_degree = jnp.asarray(maximum_degree, dtype=jnp.int32)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.method = str(method)


def _case_count(case_shape: tuple[int, ...]) -> int:
    return prod(case_shape) if case_shape else 1


def _fit_arrays(batch: MLBatch) -> tuple[Array, Array, Array]:
    x = batch.dense_features()
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.float32)
    weights = batch.effective_weight("statistical").astype(x.real.dtype)
    weights = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights) | (weights < 0.0)),
        "Manifold sample weights must be finite and nonnegative.",
    )
    feature_valid = jnp.all(batch.feature_mask, axis=-1)
    finite = jnp.all(jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x)), axis=-1)
    active = (
        batch.sample_mask
        & feature_valid
        & finite
        & jnp.isfinite(weights)
        & (weights > 0.0)
    )
    safe_x = jnp.where(active[..., None], x, 0)
    safe_weights = jnp.where(active, weights, 0.0)
    return safe_x, safe_weights, active


def _prepare_queries(
    value: ArrayLike,
    *,
    case_shape: tuple[int, ...],
    feature_count: int,
) -> tuple[Array, tuple[int, ...], bool]:
    x = jnp.asarray(value)
    if x.ndim == 0 or int(x.shape[-1]) != int(feature_count):
        raise ValueError(f"Input must end in feature axis of size {feature_count}.")
    point = x.ndim == 1
    if case_shape and tuple(int(s) for s in x.shape[: len(case_shape)]) == case_shape:
        query_shape = tuple(int(s) for s in x.shape[len(case_shape) : -1])
        shaped = x.reshape(
            (
                _case_count(case_shape),
                prod(query_shape) if query_shape else 1,
                feature_count,
            )
        )
    else:
        query_shape = tuple(int(s) for s in x.shape[:-1])
        broadcast = jnp.broadcast_to(x, case_shape + x.shape)
        shaped = broadcast.reshape(
            (
                _case_count(case_shape),
                prod(query_shape) if query_shape else 1,
                feature_count,
            )
        )
    return shaped, query_shape, point


def _restore_queries(
    values: Array,
    *,
    case_shape: tuple[int, ...],
    query_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> Array:
    return values.reshape(case_shape + query_shape + output_shape)


def _euclidean_from_squared(squared_distances: ArrayLike) -> Array:
    """Take an exact-zero-preserving square root with a finite zero cotangent."""
    squared = jnp.asarray(squared_distances)
    positive = squared > 0.0
    safe_squared = jnp.where(positive, squared, jnp.ones_like(squared))
    return jnp.where(positive, jnp.sqrt(safe_squared), jnp.zeros_like(squared))


def _euclidean_distances(left: ArrayLike, right: ArrayLike | None = None) -> Array:
    squared = pairwise_distances(left, right, metric="squared-euclidean")
    return _euclidean_from_squared(squared)


def _connectivity_one(adjacency: Array, active: Array) -> tuple[Array, Array, Array]:
    n = int(active.shape[0])
    labels = jnp.where(active, jnp.arange(n, dtype=jnp.int32), n)

    def propagate(_iteration, current):
        candidates = jnp.where(adjacency, current[None, :], n)
        neighbor_min = jnp.min(candidates, axis=-1)
        return jnp.where(active, jnp.minimum(current, neighbor_min), n)

    labels = jax.lax.fori_loop(0, n, propagate, labels)
    roots = jnp.arange(n, dtype=jnp.int32)
    components = jnp.sum(
        jnp.any((labels[:, None] == roots[None, :]) & active[:, None], axis=0),
        dtype=jnp.int32,
    )
    degree = jnp.sum(adjacency, axis=-1, dtype=jnp.int32)
    minimum = jnp.min(jnp.where(active, degree, n), initial=n)
    minimum = jnp.where(jnp.any(active), minimum, 0)
    maximum = jnp.max(jnp.where(active, degree, 0), initial=0)
    return components, minimum, maximum


def build_neighbor_graph(
    features: ArrayLike,
    active: ArrayLike,
    /,
    *,
    n_neighbors: int,
    metric: str = "euclidean",
) -> NeighborhoodGraph:
    """Construct a hard k-NN relation; only edge distances retain conditional gradients."""
    x = jnp.asarray(features)
    included = jnp.asarray(active, dtype=bool)
    if x.ndim < 2 or included.shape != x.shape[:-1]:
        raise ValueError("features and active must end in (sample, feature) and sample.")
    n = int(x.shape[-2])
    k = int(n_neighbors)
    if k <= 0 or k >= n:
        raise ValueError(f"n_neighbors must lie in [1, {n - 1}].")
    distances = (
        _euclidean_distances(x)
        if metric == "euclidean"
        else pairwise_distances(x, metric=metric)
    )
    eye = jnp.eye(n, dtype=bool)
    eligible = included[..., :, None] & included[..., None, :] & ~eye
    ranked = jnp.where(eligible, distances, jnp.inf)
    _negative, indices = jax.lax.top_k(-ranked, k)
    indices = jax.lax.stop_gradient(indices.astype(jnp.int32))
    selected = jnp.take_along_axis(distances, indices, axis=-1)
    neighbor_active = jnp.take_along_axis(
        jnp.broadcast_to(included[..., None, :], distances.shape), indices, axis=-1
    )
    row_valid = included[..., :, None] & neighbor_active & jnp.isfinite(selected)
    selected = jnp.where(row_valid, selected, 0.0)
    one_hot = jax.nn.one_hot(indices, n, dtype=bool)
    directed = jnp.any(one_hot & row_valid[..., None], axis=-2)
    adjacency = directed | jnp.swapaxes(directed, -1, -2)
    case_shape = tuple(int(s) for s in x.shape[:-2])
    flat_adjacency = adjacency.reshape((_case_count(case_shape), n, n))
    flat_active = included.reshape((_case_count(case_shape), n))
    components, minimum, maximum = jax.vmap(_connectivity_one)(
        flat_adjacency, flat_active
    )
    relation = RowRelation(
        indices,
        source_size=n,
        valid=row_valid,
        case_shape=case_shape,
    )
    return NeighborhoodGraph(
        relation,
        selected,
        adjacency,
        included,
        components.reshape(case_shape),
        minimum.reshape(case_shape),
        maximum.reshape(case_shape),
        metric=metric,
    )


def _stable_hermitian_eigh(
    matrix: Array, *, protected: ArrayLike | None = None
) -> tuple[Array, Array]:
    """Diagonalize Hermitian matrices after a machine-resolution spectral tie break."""
    hermitian = 0.5 * (matrix + jnp.conj(jnp.swapaxes(matrix, -1, -2)))
    n = int(hermitian.shape[-1])
    real_dtype = hermitian.real.dtype
    magnitude = jnp.maximum(
        jnp.linalg.norm(hermitian, axis=(-2, -1), keepdims=True),
        jnp.ones((), dtype=real_dtype),
    )
    spacing = jnp.finfo(real_dtype).eps * magnitude
    offsets = jnp.arange(n, dtype=real_dtype)
    identity = jnp.eye(n, dtype=hermitian.dtype)
    tie_breaker = identity * offsets
    if protected is not None:
        vector = jnp.asarray(protected, dtype=hermitian.dtype)
        norm = jnp.sqrt(jnp.sum(jnp.real(vector * jnp.conj(vector)), axis=-1))
        unit = vector / jnp.maximum(norm, jnp.finfo(real_dtype).tiny)[..., None]
        protected_projector = unit[..., :, None] * jnp.conj(unit[..., None, :])
        complement = identity - protected_projector
        tie_breaker = (
            complement @ tie_breaker @ complement - float(n) * protected_projector
        )
    return jnp.linalg.eigh(hermitian + spacing * tie_breaker)


def _canonicalize_columns(vectors: Array) -> Array:
    pivots = jnp.argmax(jnp.abs(vectors), axis=-2)
    values = jnp.take_along_axis(vectors, pivots[None, :], axis=-2)[0]
    magnitude = jnp.abs(values)
    phase = jnp.where(magnitude > 0.0, values / magnitude, jnp.ones_like(values))
    return vectors * jnp.conj(phase)[None, :]


def _spectral_coordinates(
    matrix: Array,
    dimensions: int,
    *,
    smallest: bool,
    skip: int = 0,
    protected: ArrayLike | None = None,
) -> tuple[Array, Array]:
    eigenvalues, eigenvectors = _stable_hermitian_eigh(matrix, protected=protected)
    if smallest:
        selected_values = eigenvalues[skip : skip + dimensions]
        selected_vectors = eigenvectors[:, skip : skip + dimensions]
    else:
        selected_values = eigenvalues[-dimensions:][::-1]
        selected_vectors = eigenvectors[:, -dimensions:][:, ::-1]
    return selected_values, _canonicalize_columns(selected_vectors)


def _classical_mds_one(
    squared_distances: Array,
    weights: Array,
    dimensions: int,
) -> tuple[Array, Array, Array, Array, Array]:
    total = jnp.sum(weights)
    probabilities = weights / jnp.maximum(total, jnp.finfo(weights.dtype).tiny)
    safe_distances = jnp.where(
        (weights[:, None] > 0.0) & (weights[None, :] > 0.0), squared_distances, 0.0
    )
    row_mean = safe_distances @ probabilities
    grand_mean = probabilities @ row_mean
    gram = -0.5 * (safe_distances - row_mean[:, None] - row_mean[None, :] + grand_mean)
    inactive = weights <= 0.0
    gram = jnp.where(inactive[:, None] | inactive[None, :], 0.0, gram)
    eigenvalues, eigenvectors = _spectral_coordinates(gram, dimensions, smallest=False)
    positive = jnp.maximum(jnp.real(eigenvalues), 0.0)
    coordinates = eigenvectors * jnp.sqrt(positive)[None, :]
    coordinates = jnp.where((weights > 0.0)[:, None], coordinates, 0.0)
    reconstruction = coordinates @ jnp.conj(coordinates).T
    residual = jnp.linalg.norm(gram - reconstruction) / jnp.maximum(
        jnp.linalg.norm(gram), jnp.finfo(gram.real.dtype).tiny
    )
    return coordinates, eigenvalues, row_mean, jnp.asarray(grand_mean), residual


def _classical_transform_one(
    squared_distances: Array,
    weights: Array,
    training_row_mean: Array,
    grand_mean: Array,
    coordinates: Array,
    eigenvalues: Array,
) -> Array:
    probabilities = weights / jnp.maximum(jnp.sum(weights), jnp.finfo(weights.dtype).tiny)
    query_mean = squared_distances @ probabilities
    centered = -0.5 * (
        squared_distances - query_mean[:, None] - training_row_mean[None, :] + grand_mean
    )
    floor = jnp.finfo(eigenvalues.dtype).eps
    inverse_scale = 1.0 / jnp.maximum(
        jnp.sqrt(jnp.maximum(jnp.real(eigenvalues), 0.0)), floor
    )
    basis = coordinates * inverse_scale[None, :]
    return (centered @ basis) * inverse_scale[None, :]


def _fit_status(
    finite: Array,
    enough: Array,
    connected: Array | None = None,
) -> Array:
    status = jnp.where(finite, ML_SUCCESS, ML_NONFINITE)
    status = jnp.where(finite & ~enough, ML_INSUFFICIENT_DATA, status)
    if connected is not None:
        status = jnp.where(finite & enough & ~connected, ML_INFEASIBLE, status)
    return status.astype(jnp.int32)


__all__ = ["ManifoldDiagnostics", "NeighborhoodGraph", "build_neighbor_graph"]
