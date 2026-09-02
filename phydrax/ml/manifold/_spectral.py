#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._model import AbstractArrayModel, ModelBinding
from .._batch import MLBatch
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_NONCONVERGED,
)
from .._numerics import pairwise_distances
from ._common import (
    _BLOCKWISE_BINDING,
    _canonicalize_columns,
    _case_count,
    _classical_mds_one,
    _classical_transform_one,
    _euclidean_distances,
    _euclidean_from_squared,
    _fit_arrays,
    _fit_status,
    _prepare_queries,
    _restore_queries,
    _stable_hermitian_eigh,
    build_neighbor_graph,
    ManifoldDiagnostics,
)


class SpectralEmbeddingModel(AbstractArrayModel):
    """Normalized-graph eigenmap with conditional Nyström extension."""

    training_features: Array
    eigenvectors: Array
    eigenvalues: Array
    degrees: Array
    training_weights: Array
    active: Array
    bandwidth: float = eqx.field(static=True)
    n_neighbors: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        training_features: ArrayLike,
        eigenvectors: ArrayLike,
        eigenvalues: ArrayLike,
        degrees: ArrayLike,
        training_weights: ArrayLike,
        active: ArrayLike,
        *,
        bandwidth: float,
        n_neighbors: int,
        case_shape: tuple[int, ...],
    ):
        train = jnp.asarray(training_features)
        vectors = jnp.asarray(eigenvectors)
        self.training_features = train
        self.eigenvectors = vectors
        self.eigenvalues = jnp.asarray(eigenvalues)
        self.degrees = jnp.asarray(degrees)
        self.training_weights = jnp.asarray(training_weights)
        self.active = jnp.asarray(active, dtype=bool)
        self.bandwidth = float(bandwidth)
        self.n_neighbors = int(n_neighbors)
        self.case_shape = tuple(case_shape)
        self.in_size = int(train.shape[-1])
        self.out_size = int(vectors.shape[-1])

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape, _point = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        cases = _case_count(self.case_shape)
        train = self.training_features.reshape(
            (cases,) + self.training_features.shape[-2:]
        )
        vectors = self.eigenvectors.reshape((cases,) + self.eigenvectors.shape[-2:])
        values = self.eigenvalues.reshape((cases, self.out_size))
        degrees = self.degrees.reshape((cases, self.degrees.shape[-1]))
        weights = self.training_weights.reshape((cases, self.training_weights.shape[-1]))
        active = self.active.reshape((cases, self.active.shape[-1]))

        def transform_one(query, train_, vectors_, values_, degrees_, weights_, active_):
            distances = _euclidean_distances(query, train_)
            ranked = jnp.where(active_[None, :], distances, jnp.inf)
            _negative, indices = jax.lax.top_k(-ranked, self.n_neighbors)
            indices = jax.lax.stop_gradient(indices.astype(jnp.int32))
            selected = jnp.take_along_axis(distances, indices, axis=-1)
            affinities = jnp.exp(-0.5 * (selected / self.bandwidth) ** 2) * jnp.sqrt(
                weights_[indices]
            )
            query_degree = jnp.sum(affinities, axis=-1)
            product = query_degree[:, None] * degrees_[indices]
            normalized = affinities / jnp.sqrt(
                jnp.maximum(product, jnp.finfo(product.dtype).tiny)
            )
            projected = ein.contract("qk,qkd->qd", normalized, vectors_[indices])
            return (
                projected
                / jnp.maximum(jnp.abs(values_), jnp.finfo(values_.dtype).eps)[None, :]
            )

        result = jax.vmap(transform_one)(
            queries, train, vectors, values, degrees, weights, active
        )
        return _restore_queries(
            result,
            case_shape=self.case_shape,
            query_shape=query_shape,
            output_shape=(self.out_size,),
        )


def _spectral_one(
    distances: Array,
    indices: Array,
    route_valid: Array,
    active: Array,
    sample_weights: Array,
    dimensions: int,
    bandwidth: float,
) -> tuple[Array, Array, Array]:
    n = int(active.shape[0])
    affinities = jnp.exp(-0.5 * (distances / bandwidth) ** 2)
    affinities = jnp.where(route_valid, affinities, 0.0)
    rows = jnp.arange(n, dtype=jnp.int32)[:, None]
    matrix = jnp.zeros((n, n), dtype=affinities.dtype).at[rows, indices].max(affinities)
    matrix = jnp.maximum(matrix, matrix.T)
    matrix = matrix * jnp.sqrt(sample_weights[:, None] * sample_weights[None, :])
    degree = jnp.sum(matrix, axis=-1)
    degree_product = degree[:, None] * degree[None, :]
    normalized = matrix / jnp.sqrt(
        jnp.maximum(degree_product, jnp.finfo(degree_product.dtype).tiny)
    )
    normalized = jnp.where(active[:, None] & active[None, :], normalized, 0.0)
    eigenvalues, eigenvectors = _stable_hermitian_eigh(normalized)
    selected_values = eigenvalues[-(dimensions + 1) : -1][::-1]
    selected_vectors = eigenvectors[:, -(dimensions + 1) : -1][:, ::-1]
    selected_vectors = _canonicalize_columns(selected_vectors)
    selected_vectors = jnp.where(active[:, None], selected_vectors, 0.0)
    return selected_vectors, selected_values, degree


class SpectralEmbeddingRecipe(AbstractRecipe):
    """Hard k-NN heat graph followed by a normalized differentiable eigensolve."""

    n_components: int = eqx.field(static=True)
    n_neighbors: int = eqx.field(static=True)
    bandwidth: float = eqx.field(static=True)

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 10,
        bandwidth: float = 1.0,
    ):
        if int(n_components) <= 0 or int(n_neighbors) <= 0:
            raise ValueError("n_components and n_neighbors must be positive.")
        if float(bandwidth) <= 0.0:
            raise ValueError("bandwidth must be positive.")
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.bandwidth = float(bandwidth)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        n = batch.sample_count
        if self.n_components + 1 >= n or self.n_neighbors >= n:
            raise ValueError(
                "Spectral embedding dimensions and neighbors must be smaller than samples."
            )
        x, weights, active = _fit_arrays(batch)
        graph = build_neighbor_graph(x, active, n_neighbors=self.n_neighbors)
        cases = _case_count(batch.case_shape)
        vectors, eigenvalues, degrees = jax.vmap(
            lambda d_, i_, r_, a_, w_: _spectral_one(
                d_, i_, r_, a_, w_, self.n_components, self.bandwidth
            )
        )(
            graph.distances.reshape((cases, n, self.n_neighbors)),
            graph.relation.source_indices.reshape((cases, n, self.n_neighbors)),
            graph.relation.valid.reshape((cases, n, self.n_neighbors)),
            active.reshape((cases, n)),
            weights.reshape((cases, n)),
        )
        vectors = vectors.reshape(batch.case_shape + (n, self.n_components))
        eigenvalues = eigenvalues.reshape(batch.case_shape + (self.n_components,))
        degrees = degrees.reshape(batch.case_shape + (n,))
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(vectors), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(eigenvalues), axis=-1
        )
        enough = effective >= max(self.n_neighbors + 1, self.n_components + 2)
        connected = graph.components == 1
        valid = finite & enough & connected
        status = _fit_status(finite, enough, connected)
        diagnostics = ManifoldDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.sum(eigenvalues, axis=-1),
            iterations=1,
            effective_samples=effective,
            residual=1.0 - jnp.min(eigenvalues, axis=-1),
            eigenvalues=eigenvalues,
            connected_components=graph.components,
            minimum_degree=graph.minimum_degree,
            maximum_degree=graph.maximum_degree,
            converged=True,
            method="spectral-embedding",
        )
        model = SpectralEmbeddingModel(
            x,
            vectors,
            eigenvalues,
            degrees,
            weights,
            active,
            bandwidth=self.bandwidth,
            n_neighbors=self.n_neighbors,
            case_shape=batch.case_shape,
        )
        contract = GradientContract(
            prediction_inputs="conditional",
            prediction_parameters="conditional",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="spectral",
            nondifferentiable_outputs=(
                "neighbor_indices",
                "connectivity",
                "valid",
                "status",
            ),
            conditions=("k-NN topology is held fixed", "retained eigenvalues are simple"),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="spectral-embedding",
            gradient_contract=contract,
        )


MDSMethod = Literal["classical", "smacof"]


class MultidimensionalScalingModel(AbstractArrayModel):
    """Metric MDS coordinates; only classical MDS has a Gower transform."""

    training_features: Array
    training_embedding: Array
    training_weights: Array
    training_row_mean: Array
    grand_mean: Array
    eigenvalues: Array
    method: str = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        training_features: ArrayLike,
        training_embedding: ArrayLike,
        training_weights: ArrayLike,
        training_row_mean: ArrayLike,
        grand_mean: ArrayLike,
        eigenvalues: ArrayLike,
        *,
        method: MDSMethod,
        case_shape: tuple[int, ...],
    ):
        train = jnp.asarray(training_features)
        embedding = jnp.asarray(training_embedding)
        self.training_features = train
        self.training_embedding = embedding
        self.training_weights = jnp.asarray(training_weights)
        self.training_row_mean = jnp.asarray(training_row_mean)
        self.grand_mean = jnp.asarray(grand_mean)
        self.eigenvalues = jnp.asarray(eigenvalues)
        self.method = str(method)
        self.case_shape = tuple(case_shape)
        self.in_size = int(train.shape[-1])
        self.out_size = int(embedding.shape[-1])

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        if self.method != "classical":
            raise ValueError(
                "SMACOF MDS is transductive; an out-of-sample transform is undefined."
            )
        queries, query_shape, _point = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        cases = _case_count(self.case_shape)
        train = self.training_features.reshape(
            (cases,) + self.training_features.shape[-2:]
        )
        embedding = self.training_embedding.reshape(
            (cases,) + self.training_embedding.shape[-2:]
        )
        weights = self.training_weights.reshape((cases, self.training_weights.shape[-1]))
        row_mean = self.training_row_mean.reshape(
            (cases, self.training_row_mean.shape[-1])
        )
        grand = self.grand_mean.reshape((cases,))
        values = self.eigenvalues.reshape((cases, self.out_size))

        def transform_one(query, train_, weights_, mean_, grand_, embedding_, values_):
            squared = pairwise_distances(query, train_, metric="squared-euclidean")
            return _classical_transform_one(
                squared, weights_, mean_, grand_, embedding_, values_
            )

        transformed = jax.vmap(transform_one)(
            queries, train, weights, row_mean, grand, embedding, values
        )
        return _restore_queries(
            transformed,
            case_shape=self.case_shape,
            query_shape=query_shape,
            output_shape=(self.out_size,),
        )


def _smacof_one(
    distances: Array,
    weights: Array,
    initial: Array,
    iterations: int,
) -> tuple[Array, Array, Array]:
    pair_weights = jnp.sqrt(weights[:, None] * weights[None, :])
    pair_weights = pair_weights * (
        1.0 - jnp.eye(weights.shape[0], dtype=pair_weights.dtype)
    )

    def step(_iteration, state):
        current, _previous_delta = state
        embedded = _euclidean_distances(current)
        ratio = jnp.where(
            embedded > 0.0,
            distances / jnp.maximum(embedded, jnp.finfo(embedded.dtype).eps),
            0.0,
        )
        off_diagonal = -pair_weights * ratio
        b_matrix = off_diagonal - jnp.diag(jnp.sum(off_diagonal, axis=-1))
        denominator = jnp.maximum(
            jnp.sum(pair_weights, axis=-1), jnp.finfo(pair_weights.dtype).tiny
        )
        updated = (b_matrix @ current) / denominator[:, None]
        updated = updated - jnp.sum(weights[:, None] * updated, axis=0) / jnp.maximum(
            jnp.sum(weights), jnp.finfo(weights.dtype).tiny
        )
        updated = jnp.where(weights[:, None] > 0.0, updated, 0.0)
        delta = jnp.linalg.norm(updated - current) / jnp.maximum(
            jnp.linalg.norm(current), jnp.finfo(current.real.dtype).tiny
        )
        return updated, delta

    coordinates, delta = jax.lax.fori_loop(
        0, iterations, step, (initial, jnp.asarray(jnp.inf, dtype=initial.real.dtype))
    )
    embedded = _euclidean_distances(coordinates)
    stress = 0.5 * jnp.sum(pair_weights * (distances - embedded) ** 2)
    return coordinates, stress, delta


class MultidimensionalScalingRecipe(AbstractRecipe):
    """Weighted metric classical MDS or fixed-iteration SMACOF."""

    n_components: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        n_components: int = 2,
        *,
        method: MDSMethod = "classical",
        iterations: int = 100,
        tolerance: float = 1e-5,
    ):
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        if method not in ("classical", "smacof"):
            raise ValueError("Only metric classical and SMACOF MDS are supported.")
        if int(iterations) <= 0 or float(tolerance) <= 0.0:
            raise ValueError("iterations and tolerance must be positive.")
        self.n_components = int(n_components)
        self.method = str(method)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        n = batch.sample_count
        if self.n_components >= n:
            raise ValueError("MDS requires n_components < samples.")
        x, weights, active = _fit_arrays(batch)
        squared = pairwise_distances(x, metric="squared-euclidean")
        distances = _euclidean_from_squared(squared)
        cases = _case_count(batch.case_shape)
        classical = jax.vmap(
            lambda d_, w_: _classical_mds_one(d_, w_, self.n_components)
        )(squared.reshape((cases, n, n)), weights.reshape((cases, n)))
        coordinates, eigenvalues, row_mean, grand_mean, spectral_residual = classical
        if self.method == "smacof":
            coordinates, objective, residual = jax.vmap(
                lambda d_, w_, y_: _smacof_one(d_, w_, y_, self.iterations)
            )(
                distances.reshape((cases, n, n)),
                weights.reshape((cases, n)),
                coordinates,
            )
        else:
            objective = spectral_residual
            residual = spectral_residual
        coordinates = coordinates.reshape(batch.case_shape + (n, self.n_components))
        eigenvalues = eigenvalues.reshape(batch.case_shape + (self.n_components,))
        row_mean = row_mean.reshape(batch.case_shape + (n,))
        grand_mean = grand_mean.reshape(batch.case_shape)
        objective = objective.reshape(batch.case_shape)
        residual = residual.reshape(batch.case_shape)
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(jnp.real(coordinates)), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(jnp.imag(coordinates)), axis=(-2, -1)
        )
        enough = effective >= self.n_components + 1
        converged = (
            jnp.ones_like(finite)
            if self.method == "classical"
            else residual <= self.tolerance
        )
        valid = finite & enough & converged
        status = _fit_status(finite, enough)
        status = jnp.where(finite & enough & ~converged, ML_NONCONVERGED, status).astype(
            jnp.int32
        )
        diagnostics = ManifoldDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=1 if self.method == "classical" else self.iterations,
            effective_samples=effective,
            residual=residual,
            eigenvalues=eigenvalues,
            connected_components=1,
            minimum_degree=0,
            maximum_degree=0,
            converged=converged,
            method=f"mds-{self.method}",
        )
        model = MultidimensionalScalingModel(
            x,
            coordinates,
            weights,
            row_mean,
            grand_mean,
            eigenvalues,
            method=self.method,
            case_shape=batch.case_shape,
        )
        transform_supported = self.method == "classical"
        contract = GradientContract(
            prediction_inputs="smooth" if transform_supported else "none",
            prediction_parameters="smooth" if transform_supported else "none",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="spectral" if transform_supported else "unrolled",
            nondifferentiable_outputs=("valid", "status"),
            conditions=("retained eigenspaces are simple", "SMACOF is transductive only"),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method=f"mds-{self.method}",
            gradient_contract=contract,
        )


class IsomapModel(AbstractArrayModel):
    """Geodesic landmark embedding with hard-neighbor out-of-sample extension."""

    training_features: Array
    training_embedding: Array
    geodesic_distances: Array
    training_weights: Array
    training_row_mean: Array
    grand_mean: Array
    eigenvalues: Array
    active: Array
    n_neighbors: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        training_features: ArrayLike,
        training_embedding: ArrayLike,
        geodesic_distances: ArrayLike,
        training_weights: ArrayLike,
        training_row_mean: ArrayLike,
        grand_mean: ArrayLike,
        eigenvalues: ArrayLike,
        active: ArrayLike,
        *,
        n_neighbors: int,
        case_shape: tuple[int, ...],
    ):
        train = jnp.asarray(training_features)
        embedding = jnp.asarray(training_embedding)
        self.training_features = train
        self.training_embedding = embedding
        self.geodesic_distances = jnp.asarray(geodesic_distances)
        self.training_weights = jnp.asarray(training_weights)
        self.training_row_mean = jnp.asarray(training_row_mean)
        self.grand_mean = jnp.asarray(grand_mean)
        self.eigenvalues = jnp.asarray(eigenvalues)
        self.active = jnp.asarray(active, dtype=bool)
        self.n_neighbors = int(n_neighbors)
        self.case_shape = tuple(case_shape)
        self.in_size = int(train.shape[-1])
        self.out_size = int(embedding.shape[-1])

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape, _point = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        cases = _case_count(self.case_shape)
        train = self.training_features.reshape(
            (cases,) + self.training_features.shape[-2:]
        )
        embedding = self.training_embedding.reshape(
            (cases,) + self.training_embedding.shape[-2:]
        )
        geodesic = self.geodesic_distances.reshape(
            (cases,) + self.geodesic_distances.shape[-2:]
        )
        weights = self.training_weights.reshape((cases, self.training_weights.shape[-1]))
        row_mean = self.training_row_mean.reshape(
            (cases, self.training_row_mean.shape[-1])
        )
        grand = self.grand_mean.reshape((cases,))
        values = self.eigenvalues.reshape((cases, self.out_size))
        active = self.active.reshape((cases, self.active.shape[-1]))

        def transform_one(
            query,
            train_,
            embedding_,
            geodesic_,
            weights_,
            mean_,
            grand_,
            values_,
            active_,
        ):
            direct = _euclidean_distances(query, train_)
            ranked = jnp.where(active_[None, :], direct, jnp.inf)
            _negative, indices = jax.lax.top_k(-ranked, self.n_neighbors)
            indices = jax.lax.stop_gradient(indices.astype(jnp.int32))
            selected = jnp.take_along_axis(direct, indices, axis=-1)
            query_geodesic = jnp.min(selected[:, :, None] + geodesic_[indices], axis=1)
            return _classical_transform_one(
                query_geodesic * query_geodesic,
                weights_,
                mean_,
                grand_,
                embedding_,
                values_,
            )

        transformed = jax.vmap(transform_one)(
            queries, train, embedding, geodesic, weights, row_mean, grand, values, active
        )
        return _restore_queries(
            transformed,
            case_shape=self.case_shape,
            query_shape=query_shape,
            output_shape=(self.out_size,),
        )


def _geodesic_one(
    edge_distances: Array,
    indices: Array,
    route_valid: Array,
    active: Array,
) -> Array:
    n = int(active.shape[0])
    rows = jnp.arange(n, dtype=jnp.int32)[:, None]
    graph = jnp.full((n, n), jnp.inf, dtype=edge_distances.dtype)
    graph = graph.at[rows, indices].min(jnp.where(route_valid, edge_distances, jnp.inf))
    graph = jnp.minimum(graph, graph.T)
    graph = graph.at[jnp.diag_indices(n)].set(jnp.where(active, 0.0, jnp.inf))

    def relax(k, current):
        return jnp.minimum(current, current[:, k, None] + current[k, None, :])

    return jax.lax.fori_loop(0, n, relax, graph)


class IsomapRecipe(AbstractRecipe):
    """Capacity-bounded hard k-NN geodesics followed by classical MDS."""

    n_components: int = eqx.field(static=True)
    n_neighbors: int = eqx.field(static=True)
    max_samples: int = eqx.field(static=True)

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 8,
        max_samples: int = 2048,
    ):
        if int(n_components) <= 0 or int(n_neighbors) <= 0 or int(max_samples) <= 0:
            raise ValueError(
                "n_components, n_neighbors, and max_samples must be positive."
            )
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.max_samples = int(max_samples)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        n = batch.sample_count
        if n > self.max_samples:
            raise ValueError(
                f"Isomap sample capacity exceeded: {n} > max_samples={self.max_samples}."
            )
        if self.n_components >= n or self.n_neighbors >= n:
            raise ValueError(
                "Isomap dimensions and neighbors must be smaller than samples."
            )
        x, weights, active = _fit_arrays(batch)
        graph = build_neighbor_graph(x, active, n_neighbors=self.n_neighbors)
        cases = _case_count(batch.case_shape)
        geodesic = jax.vmap(_geodesic_one)(
            graph.distances.reshape((cases, n, self.n_neighbors)),
            graph.relation.source_indices.reshape((cases, n, self.n_neighbors)),
            graph.relation.valid.reshape((cases, n, self.n_neighbors)),
            active.reshape((cases, n)),
        )
        safe_geodesic = jnp.where(jnp.isfinite(geodesic), geodesic, 0.0)
        embedding, eigenvalues, row_mean, grand_mean, residual = jax.vmap(
            lambda d_, w_: _classical_mds_one(d_ * d_, w_, self.n_components)
        )(safe_geodesic, weights.reshape((cases, n)))
        embedding = embedding.reshape(batch.case_shape + (n, self.n_components))
        eigenvalues = eigenvalues.reshape(batch.case_shape + (self.n_components,))
        geodesic = geodesic.reshape(batch.case_shape + (n, n))
        row_mean = row_mean.reshape(batch.case_shape + (n,))
        grand_mean = grand_mean.reshape(batch.case_shape)
        residual = residual.reshape(batch.case_shape)
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(embedding), axis=(-2, -1))
        enough = effective >= max(self.n_neighbors + 1, self.n_components + 1)
        connected = graph.components == 1
        valid = finite & enough & connected
        status = _fit_status(finite, enough, connected)
        diagnostics = ManifoldDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.sum(jnp.maximum(-eigenvalues, 0.0), axis=-1),
            iterations=n,
            effective_samples=effective,
            residual=residual,
            eigenvalues=eigenvalues,
            connected_components=graph.components,
            minimum_degree=graph.minimum_degree,
            maximum_degree=graph.maximum_degree,
            converged=connected,
            method="isomap",
        )
        model = IsomapModel(
            x,
            embedding,
            geodesic,
            weights,
            row_mean,
            grand_mean,
            eigenvalues,
            active,
            n_neighbors=self.n_neighbors,
            case_shape=batch.case_shape,
        )
        contract = GradientContract(
            prediction_inputs="conditional",
            prediction_parameters="conditional",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="spectral",
            nondifferentiable_outputs=(
                "neighbor_indices",
                "shortest_path_topology",
                "connectivity",
                "valid",
                "status",
            ),
            conditions=(
                "nearest-neighbor and shortest-path choices are held fixed",
                "retained MDS eigenspaces are simple",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="isomap",
            gradient_contract=contract,
        )


__all__ = [
    "IsomapModel",
    "IsomapRecipe",
    "MDSMethod",
    "MultidimensionalScalingModel",
    "MultidimensionalScalingRecipe",
    "SpectralEmbeddingModel",
    "SpectralEmbeddingRecipe",
]
