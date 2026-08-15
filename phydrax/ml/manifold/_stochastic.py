#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel, ModelBinding
from .._batch import MLBatch
from .._contracts import AbstractRecipe, FitResult, GradientContract, ML_NONCONVERGED
from .._numerics import pairwise_distances
from ._common import (
    _BLOCKWISE_BINDING,
    _case_count,
    _euclidean_distances,
    _fit_arrays,
    _fit_status,
    _prepare_queries,
    _restore_queries,
    build_neighbor_graph,
    ManifoldDiagnostics,
)


def _tsne_squared_distances(values: Array) -> Array:
    """Smooth Hermitian squared distances with an exact constant diagonal."""
    x = jnp.asarray(values)
    norms = jnp.sum(jnp.real(x * jnp.conj(x)), axis=-1, keepdims=True)
    cross = jnp.real(x @ jnp.conj(jnp.swapaxes(x, -1, -2)))
    squared = norms + jnp.swapaxes(norms, -1, -2) - 2.0 * cross
    squared = 0.5 * (squared + jnp.swapaxes(squared, -1, -2))
    diagonal = jnp.eye(x.shape[-2], dtype=bool)
    return jnp.where(diagonal, jnp.zeros_like(squared), squared)


def _perplexity_probabilities_one(
    squared_distances: Array,
    weights: Array,
    active: Array,
    perplexity: float,
) -> Array:
    n = int(active.shape[0])
    eligible = active[:, None] & active[None, :] & ~jnp.eye(n, dtype=bool)
    tiny = jnp.finfo(weights.dtype).tiny
    log_weights = jnp.log(jnp.maximum(weights, tiny))
    target_entropy = jnp.log(float(perplexity))

    def row_probabilities(distance_row, eligible_row):
        def entropy(beta):
            logits = -beta * distance_row + log_weights
            logits = jnp.where(
                eligible_row,
                logits,
                jnp.asarray(jnp.finfo(logits.dtype).min, dtype=logits.dtype),
            )
            probabilities = jax.nn.softmax(logits)
            probabilities = jnp.where(eligible_row, probabilities, 0.0)
            value = -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, tiny)))
            return value, probabilities

        def search(_iteration, bounds):
            lower, upper = bounds
            beta = 0.5 * (lower + upper)
            value, _probabilities = entropy(beta)
            lower = jnp.where(value > target_entropy, beta, lower)
            upper = jnp.where(value > target_entropy, upper, beta)
            return lower, upper

        lower, upper = jax.lax.fori_loop(
            0,
            64,
            search,
            (
                jnp.asarray(0.0, dtype=squared_distances.dtype),
                jnp.asarray(1e6, dtype=squared_distances.dtype),
            ),
        )
        _entropy, probabilities = entropy(0.5 * (lower + upper))
        return probabilities

    conditional = jax.vmap(row_probabilities)(squared_distances, eligible)
    joint = conditional * weights[:, None] + conditional.T * weights[None, :]
    joint = jnp.where(eligible, joint, 0.0)
    return joint / jnp.maximum(jnp.sum(joint), jnp.finfo(joint.dtype).tiny)


def _tsne_loss(embedding: Array, probabilities: Array, active: Array) -> Array:
    squared = _tsne_squared_distances(embedding)
    eligible = active[:, None] & active[None, :] & ~jnp.eye(active.shape[0], dtype=bool)
    numerator = jnp.where(eligible, 1.0 / (1.0 + squared), 0.0)
    tiny = jnp.finfo(probabilities.dtype).tiny
    q = numerator / jnp.maximum(jnp.sum(numerator), tiny)
    safe_probabilities = jnp.where(eligible, probabilities, jnp.ones_like(probabilities))
    safe_q = jnp.where(eligible, q, jnp.ones_like(q))
    terms = probabilities * (
        jnp.log(jnp.maximum(safe_probabilities, tiny))
        - jnp.log(jnp.maximum(safe_q, tiny))
    )
    return jnp.sum(jnp.where(eligible, terms, 0.0))


def _optimize_tsne_one(
    probabilities: Array,
    weights: Array,
    active: Array,
    key: Array,
    dimensions: int,
    iterations: int,
    learning_rate: float,
    momentum: float,
) -> tuple[Array, Array, Array]:
    dtype = probabilities.dtype
    initial = 1e-4 * jax.random.normal(key, (active.shape[0], dimensions), dtype=dtype)
    initial = jnp.where(active[:, None], initial, 0.0)

    def step(_iteration, state):
        embedding, velocity = state
        _value, gradient = jax.value_and_grad(_tsne_loss)(
            embedding, probabilities, active
        )
        velocity = momentum * velocity - learning_rate * gradient
        embedding = embedding + velocity
        center = jnp.sum(weights[:, None] * embedding, axis=0) / jnp.maximum(
            jnp.sum(weights), jnp.finfo(weights.dtype).tiny
        )
        embedding = jnp.where(active[:, None], embedding - center, 0.0)
        return embedding, velocity

    embedding, _velocity = jax.lax.fori_loop(
        0, iterations, step, (initial, jnp.zeros_like(initial))
    )
    objective, gradient = jax.value_and_grad(_tsne_loss)(embedding, probabilities, active)
    gradient_norm = jnp.linalg.norm(gradient)
    return embedding, objective, gradient_norm


class TSNEModel(AbstractArrayModel):
    """Transductive t-SNE coordinates; no mathematically defined transform is claimed."""

    embedding: Array
    training_features: Array
    active: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        training_features: ArrayLike,
        embedding: ArrayLike,
        active: ArrayLike,
        *,
        case_shape: tuple[int, ...],
    ):
        train = jnp.asarray(training_features)
        coordinates = jnp.asarray(embedding)
        self.embedding = coordinates
        self.training_features = train
        self.active = jnp.asarray(active, dtype=bool)
        self.case_shape = tuple(case_shape)
        self.in_size = int(train.shape[-1])
        self.out_size = int(coordinates.shape[-1])

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        raise ValueError(
            "t-SNE is transductive; this fitted model exposes embedding but no transform."
        )


class TSNERecipe(AbstractRecipe):
    """Exact weighted t-SNE with explicit random initialization and fixed iterations."""

    n_components: int = eqx.field(static=True)
    perplexity: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    max_samples: int = eqx.field(static=True)

    def __init__(
        self,
        n_components: int = 2,
        *,
        perplexity: float = 30.0,
        iterations: int = 500,
        learning_rate: float = 100.0,
        momentum: float = 0.8,
        tolerance: float = 1e-4,
        max_samples: int = 4096,
    ):
        if int(n_components) <= 0 or float(perplexity) <= 1.0:
            raise ValueError(
                "n_components must be positive and perplexity must exceed one."
            )
        if int(iterations) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("iterations and learning_rate must be positive.")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must lie in [0, 1).")
        if float(tolerance) <= 0.0 or int(max_samples) <= 0:
            raise ValueError("tolerance and max_samples must be positive.")
        self.n_components = int(n_components)
        self.perplexity = float(perplexity)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.tolerance = float(tolerance)
        self.max_samples = int(max_samples)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("TSNERecipe requires an explicit JAX key.")
        n = batch.sample_count
        if n > self.max_samples:
            raise ValueError(f"t-SNE sample capacity exceeded: {n} > {self.max_samples}.")
        if self.perplexity >= n:
            raise ValueError("perplexity must be smaller than sample_count.")
        x, weights, active = _fit_arrays(batch)
        squared = _tsne_squared_distances(x)
        cases = _case_count(batch.case_shape)
        flat_weights = weights.reshape((cases, n))
        flat_active = active.reshape((cases, n))
        probabilities = jax.vmap(
            lambda d_, w_, a_: _perplexity_probabilities_one(d_, w_, a_, self.perplexity)
        )(squared.reshape((cases, n, n)), flat_weights, flat_active)
        keys = jax.random.split(key, cases)
        embedding, objective, residual = jax.vmap(
            lambda p_, w_, a_, k_: _optimize_tsne_one(
                p_,
                w_,
                a_,
                k_,
                self.n_components,
                self.iterations,
                self.learning_rate,
                self.momentum,
            )
        )(probabilities, flat_weights, flat_active, keys)
        embedding = embedding.reshape(batch.case_shape + (n, self.n_components))
        objective = objective.reshape(batch.case_shape)
        residual = residual.reshape(batch.case_shape)
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(embedding), axis=(-2, -1)) & jnp.isfinite(objective)
        enough = effective > self.perplexity
        converged = residual <= self.tolerance
        valid = finite & enough & converged
        status = _fit_status(finite, enough)
        status = jnp.where(finite & enough & ~converged, ML_NONCONVERGED, status).astype(
            jnp.int32
        )
        diagnostics = ManifoldDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.iterations,
            effective_samples=effective,
            residual=residual,
            eigenvalues=jnp.empty(batch.case_shape + (0,)),
            connected_components=1,
            minimum_degree=n - 1,
            maximum_degree=n - 1,
            converged=converged,
            method="tsne-exact",
        )
        model = TSNEModel(x, embedding, active, case_shape=batch.case_shape)
        contract = GradientContract(
            prediction_inputs="none",
            prediction_parameters="none",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("valid", "status"),
            conditions=(
                "fixed explicit initialization key and iteration count",
                "perplexity bandwidth bisection branch decisions are held fixed",
                "fitted t-SNE is transductive only",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="tsne-exact",
            gradient_contract=contract,
        )


def _fuzzy_graph_one(
    distances: Array,
    indices: Array,
    route_valid: Array,
    active: Array,
    weights: Array,
    n_neighbors: int,
) -> tuple[Array, Array, Array]:
    n = int(active.shape[0])
    positive = route_valid & (distances > 0.0)
    rho = jnp.min(jnp.where(positive, distances, jnp.inf), axis=-1)
    rho = jnp.where(jnp.isfinite(rho), rho, 0.0)
    shifted = jnp.maximum(distances - rho[:, None], 0.0)
    target = jnp.log2(float(n_neighbors))

    def solve_sigma(values, valid):
        def search(_iteration, bounds):
            lower, upper = bounds
            sigma = 0.5 * (lower + upper)
            mass = jnp.sum(jnp.where(valid, jnp.exp(-values / sigma), 0.0))
            lower = jnp.where(mass < target, sigma, lower)
            upper = jnp.where(mass < target, upper, sigma)
            return lower, upper

        lower, upper = jax.lax.fori_loop(
            0,
            64,
            search,
            (
                jnp.asarray(jnp.finfo(values.dtype).eps),
                jnp.asarray(1e6, dtype=values.dtype),
            ),
        )
        return 0.5 * (lower + upper)

    sigma = jax.vmap(solve_sigma)(shifted, route_valid)
    directed_values = jnp.where(route_valid, jnp.exp(-shifted / sigma[:, None]), 0.0)
    rows = jnp.arange(n, dtype=jnp.int32)[:, None]
    directed = (
        jnp.zeros((n, n), dtype=distances.dtype).at[rows, indices].max(directed_values)
    )
    fuzzy = directed + directed.T - directed * directed.T
    normalized_weights = weights / jnp.maximum(
        jnp.max(weights), jnp.finfo(weights.dtype).tiny
    )
    fuzzy = fuzzy * jnp.sqrt(normalized_weights[:, None] * normalized_weights[None, :])
    fuzzy = jnp.where(active[:, None] & active[None, :], fuzzy, 0.0)
    return fuzzy, rho, sigma


def _umap_loss(
    embedding: Array,
    fuzzy: Array,
    active: Array,
    min_dist: float,
    repulsion: float,
) -> Array:
    squared = pairwise_distances(embedding, metric="squared-euclidean")
    scale = max(float(min_dist), 1e-3)
    q = 1.0 / (1.0 + squared / (scale * scale))
    eligible = active[:, None] & active[None, :] & ~jnp.eye(active.shape[0], dtype=bool)
    epsilon = jnp.finfo(q.dtype).eps
    q = jnp.clip(q, epsilon, 1.0 - epsilon)
    loss = -fuzzy * jnp.log(q) - float(repulsion) * (1.0 - fuzzy) * jnp.log1p(-q)
    return jnp.sum(jnp.where(eligible, loss, 0.0)) / jnp.maximum(jnp.sum(eligible), 1)


def _optimize_umap_one(
    fuzzy: Array,
    weights: Array,
    active: Array,
    key: Array,
    dimensions: int,
    iterations: int,
    learning_rate: float,
    min_dist: float,
    repulsion: float,
) -> tuple[Array, Array, Array]:
    initial = 1e-3 * jax.random.normal(
        key, (active.shape[0], dimensions), dtype=fuzzy.dtype
    )
    initial = jnp.where(active[:, None], initial, 0.0)

    def step(iteration, embedding):
        _value, gradient = jax.value_and_grad(_umap_loss)(
            embedding, fuzzy, active, min_dist, repulsion
        )
        rate = learning_rate * (1.0 - iteration / float(iterations))
        updated = embedding - rate * gradient
        center = jnp.sum(weights[:, None] * updated, axis=0) / jnp.maximum(
            jnp.sum(weights), jnp.finfo(weights.dtype).tiny
        )
        return jnp.where(active[:, None], updated - center, 0.0)

    embedding = jax.lax.fori_loop(0, iterations, step, initial)
    objective, gradient = jax.value_and_grad(_umap_loss)(
        embedding, fuzzy, active, min_dist, repulsion
    )
    return embedding, objective, jnp.linalg.norm(gradient)


class FuzzyGraphEmbeddingModel(AbstractArrayModel):
    """UMAP-like embedding with a conditional fuzzy barycentric transform."""

    training_features: Array
    embedding: Array
    active: Array
    n_neighbors: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        training_features: ArrayLike,
        embedding: ArrayLike,
        active: ArrayLike,
        *,
        n_neighbors: int,
        case_shape: tuple[int, ...],
    ):
        train = jnp.asarray(training_features)
        coordinates = jnp.asarray(embedding)
        self.training_features = train
        self.embedding = coordinates
        self.active = jnp.asarray(active, dtype=bool)
        self.n_neighbors = int(n_neighbors)
        self.case_shape = tuple(case_shape)
        self.in_size = int(train.shape[-1])
        self.out_size = int(coordinates.shape[-1])

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape, _point = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        cases = _case_count(self.case_shape)
        train = self.training_features.reshape(
            (cases,) + self.training_features.shape[-2:]
        )
        embedding = self.embedding.reshape((cases,) + self.embedding.shape[-2:])
        active = self.active.reshape((cases, self.active.shape[-1]))

        def transform_one(query, train_, embedding_, active_):
            distances = _euclidean_distances(query, train_)
            ranked = jnp.where(active_[None, :], distances, jnp.inf)
            _negative, indices = jax.lax.top_k(-ranked, self.n_neighbors)
            indices = jax.lax.stop_gradient(indices.astype(jnp.int32))
            selected = jnp.take_along_axis(distances, indices, axis=-1)
            rho = jnp.min(jnp.where(selected > 0.0, selected, jnp.inf), axis=-1)
            rho = jnp.where(jnp.isfinite(rho), rho, 0.0)
            scale = jnp.maximum(
                jnp.mean(selected, axis=-1), jnp.finfo(selected.dtype).eps
            )
            membership = jnp.exp(
                -jnp.maximum(selected - rho[:, None], 0.0) / scale[:, None]
            )
            membership = membership / jnp.maximum(
                jnp.sum(membership, axis=-1, keepdims=True),
                jnp.finfo(membership.dtype).tiny,
            )
            return oe.contract("qk,qkd->qd", membership, embedding_[indices])

        result = jax.vmap(transform_one)(queries, train, embedding, active)
        return _restore_queries(
            result,
            case_shape=self.case_shape,
            query_shape=query_shape,
            output_shape=(self.out_size,),
        )


class FuzzyGraphEmbeddingRecipe(AbstractRecipe):
    """UMAP-like fuzzy k-NN graph with explicit hard topology and smooth layout."""

    n_components: int = eqx.field(static=True)
    n_neighbors: int = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    min_dist: float = eqx.field(static=True)
    repulsion: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    max_samples: int = eqx.field(static=True)

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 15,
        iterations: int = 300,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        repulsion: float = 1.0,
        tolerance: float = 1e-4,
        max_samples: int = 4096,
    ):
        values = (n_components, n_neighbors, iterations, max_samples)
        if any(int(value) <= 0 for value in values):
            raise ValueError(
                "component, neighbor, iteration, and capacity counts must be positive."
            )
        if float(learning_rate) <= 0.0 or float(min_dist) < 0.0:
            raise ValueError("learning_rate must be positive and min_dist nonnegative.")
        if float(repulsion) < 0.0 or float(tolerance) <= 0.0:
            raise ValueError("repulsion must be nonnegative and tolerance positive.")
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.min_dist = float(min_dist)
        self.repulsion = float(repulsion)
        self.tolerance = float(tolerance)
        self.max_samples = int(max_samples)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("FuzzyGraphEmbeddingRecipe requires an explicit JAX key.")
        n = batch.sample_count
        if n > self.max_samples:
            raise ValueError(
                f"Fuzzy embedding sample capacity exceeded: {n} > {self.max_samples}."
            )
        if self.n_neighbors >= n:
            raise ValueError("n_neighbors must be smaller than sample_count.")
        x, weights, active = _fit_arrays(batch)
        graph = build_neighbor_graph(x, active, n_neighbors=self.n_neighbors)
        cases = _case_count(batch.case_shape)
        fuzzy, _rho, _sigma = jax.vmap(
            lambda d_, i_, r_, a_, w_: _fuzzy_graph_one(
                d_, i_, r_, a_, w_, self.n_neighbors
            )
        )(
            graph.distances.reshape((cases, n, self.n_neighbors)),
            graph.relation.source_indices.reshape((cases, n, self.n_neighbors)),
            graph.relation.valid.reshape((cases, n, self.n_neighbors)),
            active.reshape((cases, n)),
            weights.reshape((cases, n)),
        )
        keys = jax.random.split(key, cases)
        embedding, objective, residual = jax.vmap(
            lambda f_, w_, a_, k_: _optimize_umap_one(
                f_,
                w_,
                a_,
                k_,
                self.n_components,
                self.iterations,
                self.learning_rate,
                self.min_dist,
                self.repulsion,
            )
        )(fuzzy, weights.reshape((cases, n)), active.reshape((cases, n)), keys)
        embedding = embedding.reshape(batch.case_shape + (n, self.n_components))
        objective = objective.reshape(batch.case_shape)
        residual = residual.reshape(batch.case_shape)
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(embedding), axis=(-2, -1)) & jnp.isfinite(objective)
        enough = effective >= self.n_neighbors + 1
        connected = graph.components == 1
        converged = residual <= self.tolerance
        valid = finite & enough & connected & converged
        status = _fit_status(finite, enough, connected)
        status = jnp.where(
            finite & enough & connected & ~converged, ML_NONCONVERGED, status
        ).astype(jnp.int32)
        diagnostics = ManifoldDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.iterations,
            effective_samples=effective,
            residual=residual,
            eigenvalues=jnp.empty(batch.case_shape + (0,)),
            connected_components=graph.components,
            minimum_degree=graph.minimum_degree,
            maximum_degree=graph.maximum_degree,
            converged=converged,
            method="fuzzy-graph-embedding",
        )
        model = FuzzyGraphEmbeddingModel(
            x,
            embedding,
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
            fit_mode="unrolled",
            nondifferentiable_outputs=(
                "neighbor_indices",
                "connectivity",
                "valid",
                "status",
            ),
            conditions=(
                "hard k-NN topology is held fixed",
                "fixed explicit initialization key and iteration count",
                "transform is fuzzy barycentric, not refitting the training topology",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="fuzzy-graph-embedding",
            gradient_contract=contract,
        )


__all__ = [
    "FuzzyGraphEmbeddingModel",
    "FuzzyGraphEmbeddingRecipe",
    "TSNEModel",
    "TSNERecipe",
]
