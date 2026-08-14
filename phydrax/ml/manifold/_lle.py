#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar, Literal

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
    _euclidean_distances,
    _fit_arrays,
    _fit_status,
    _prepare_queries,
    _restore_queries,
    _spectral_coordinates,
    _stable_hermitian_eigh,
    build_neighbor_graph,
    ManifoldDiagnostics,
)


LLEVariant = Literal["standard", "modified", "hessian", "ltsa"]


def _reconstruction_weights_one(
    x: Array,
    indices: Array,
    active: Array,
    regularization: float,
) -> Array:
    neighbors = x[indices]
    centered = neighbors - x[:, None, :]
    gram = jnp.einsum("nki,nli->nkl", centered, jnp.conj(centered))
    k = int(indices.shape[-1])
    trace = jnp.real(jnp.trace(gram, axis1=-2, axis2=-1))
    ridge = float(regularization) * jnp.maximum(trace / k, jnp.finfo(x.real.dtype).eps)
    gram = gram + ridge[:, None, None] * jnp.eye(k, dtype=gram.dtype)
    ones = jnp.ones((x.shape[0], k, 1), dtype=gram.dtype)
    local = jnp.linalg.solve(gram, ones)[..., 0]
    normalization = jnp.sum(local, axis=-1, keepdims=True)
    local = local / jnp.where(
        jnp.abs(normalization) > jnp.finfo(x.real.dtype).tiny,
        normalization,
        jnp.ones_like(normalization),
    )
    neighbor_active = active[indices]
    local = jnp.where((active[:, None] & neighbor_active), local, 0.0)
    return local


def _scatter_local(indices: Array, local_matrices: Array, active: Array) -> Array:
    n = int(indices.shape[0])
    selectors = jax.nn.one_hot(indices, n, dtype=local_matrices.dtype)
    contributions = jax.vmap(lambda e, local: jnp.conj(e).T @ local @ e)(
        selectors, local_matrices
    )
    return jnp.sum(contributions * active[:, None, None], axis=0)


def _lle_alignment_one(
    x: Array,
    weights: Array,
    active: Array,
    indices: Array,
    dimensions: int,
    regularization: float,
    variant: LLEVariant,
) -> tuple[Array, Array, Array]:
    n = int(x.shape[0])
    k = int(indices.shape[-1])
    local_weights = _reconstruction_weights_one(x, indices, active, regularization)
    rows = jnp.arange(n, dtype=jnp.int32)[:, None]
    reconstruction = (
        jnp.zeros((n, n), dtype=local_weights.dtype).at[rows, indices].add(local_weights)
    )
    residual_map = jnp.eye(n, dtype=reconstruction.dtype) - reconstruction
    alignment = jnp.conj(residual_map).T @ (weights[:, None] * residual_map)

    if variant != "standard":
        neighbors = x[indices]
        centered = neighbors - jnp.mean(neighbors, axis=1, keepdims=True)
    if variant == "modified":
        nullity = max(1, k - dimensions - 1)
        _evals, local_vectors = _stable_hermitian_eigh(
            jnp.einsum("nki,nli->nkl", centered, jnp.conj(centered))
        )
        null_basis = local_vectors[:, :, :nullity]
        ones = jnp.ones((k, 1), dtype=null_basis.dtype) / jnp.sqrt(float(k))
        coefficient = jnp.einsum("ki,nkj->nij", jnp.conj(ones), null_basis)
        null_basis = null_basis - ones[None, :, :] * coefficient
        projector = null_basis @ jnp.conj(jnp.swapaxes(null_basis, -1, -2))
        alignment = alignment + _scatter_local(indices, projector, weights)
    elif variant == "hessian":
        u, _singular, _vh = jnp.linalg.svd(centered, full_matrices=False)
        tangent = u[:, :, :dimensions]
        columns = [jnp.ones((n, k, 1), dtype=tangent.dtype), tangent]
        quadratic = []
        for left in range(dimensions):
            for right in range(left, dimensions):
                factor = jnp.sqrt(2.0) if left != right else 1.0
                quadratic.append(
                    (factor * tangent[:, :, left] * tangent[:, :, right])[:, :, None]
                )
        design = jnp.concatenate(columns + quadratic, axis=-1)
        orthogonal, _r = jnp.linalg.qr(design, mode="reduced")
        hessian = orthogonal[:, :, 1 + dimensions :]
        projector = hessian @ jnp.conj(jnp.swapaxes(hessian, -1, -2))
        alignment = _scatter_local(indices, projector, weights)
    elif variant == "ltsa":
        u, _singular, _vh = jnp.linalg.svd(centered, full_matrices=False)
        tangent = u[:, :, :dimensions]
        design = jnp.concatenate(
            [jnp.ones((n, k, 1), dtype=tangent.dtype) / jnp.sqrt(float(k)), tangent],
            axis=-1,
        )
        orthogonal, _r = jnp.linalg.qr(design, mode="reduced")
        projector = jnp.eye(k, dtype=orthogonal.dtype)[
            None, :, :
        ] - orthogonal @ jnp.conj(jnp.swapaxes(orthogonal, -1, -2))
        alignment = _scatter_local(indices, projector, weights)

    inactive = ~active
    alignment = jnp.where(inactive[:, None] | inactive[None, :], 0.0, alignment)
    alignment = alignment + jnp.diag(inactive.astype(alignment.real.dtype))
    eigenvalues, eigenvectors = _spectral_coordinates(
        alignment,
        dimensions,
        smallest=True,
        skip=1,
        protected=active.astype(alignment.dtype),
    )
    scale = jnp.sqrt(jnp.maximum(jnp.sum(active), 1))
    embedding = eigenvectors * scale
    embedding = jnp.where(active[:, None], embedding, 0.0)
    reconstructed = jnp.einsum("nk,nkd->nd", local_weights, embedding[indices])
    residual = jnp.sqrt(
        jnp.sum(
            weights[:, None]
            * jnp.real((embedding - reconstructed) * jnp.conj(embedding - reconstructed))
        )
        / jnp.maximum(jnp.sum(weights), jnp.finfo(weights.dtype).tiny)
    )
    return embedding, eigenvalues, residual


class LocallyLinearEmbeddingModel(AbstractArrayModel):
    """Fitted LLE embedding with conditional barycentric out-of-sample extension."""

    training_features: Array
    training_embedding: Array
    active: Array
    regularization: float = eqx.field(static=True)
    n_neighbors: int = eqx.field(static=True)
    variant: str = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    transform_supported: bool = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        training_features: ArrayLike,
        training_embedding: ArrayLike,
        active: ArrayLike,
        *,
        regularization: float,
        n_neighbors: int,
        variant: LLEVariant,
        case_shape: tuple[int, ...],
    ):
        x = jnp.asarray(training_features)
        embedding = jnp.asarray(training_embedding)
        self.training_features = x
        self.training_embedding = embedding
        self.active = jnp.asarray(active, dtype=bool)
        self.regularization = float(regularization)
        self.n_neighbors = int(n_neighbors)
        self.variant = str(variant)
        self.case_shape = tuple(case_shape)
        self.transform_supported = variant in ("standard", "modified")
        self.in_size = int(x.shape[-1])
        self.out_size = int(embedding.shape[-1])

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        if not self.transform_supported:
            raise ValueError(
                f"Out-of-sample transform is not mathematically defined for LLE variant {self.variant!r}."
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
        active = self.active.reshape((cases, self.active.shape[-1]))

        def transform_one(query, train_, embedding_, active_):
            distances = _euclidean_distances(query, train_)
            ranked = jnp.where(active_[None, :], distances, jnp.inf)
            _negative, indices = jax.lax.top_k(-ranked, self.n_neighbors)
            indices = jax.lax.stop_gradient(indices.astype(jnp.int32))
            neighbors = train_[indices]
            centered = neighbors - query[:, None, :]
            gram = jnp.einsum("qki,qli->qkl", centered, jnp.conj(centered))
            trace = jnp.real(jnp.trace(gram, axis1=-2, axis2=-1))
            ridge = self.regularization * jnp.maximum(
                trace / self.n_neighbors, jnp.finfo(query.real.dtype).eps
            )
            gram = gram + ridge[:, None, None] * jnp.eye(
                self.n_neighbors, dtype=gram.dtype
            )
            weights = jnp.linalg.solve(
                gram, jnp.ones((query.shape[0], self.n_neighbors, 1), dtype=gram.dtype)
            )[..., 0]
            normalization = jnp.sum(weights, axis=-1, keepdims=True)
            weights = weights / jnp.where(
                jnp.abs(normalization) > jnp.finfo(query.real.dtype).tiny,
                normalization,
                jnp.ones_like(normalization),
            )
            return jnp.einsum("qk,qkd->qd", weights, embedding_[indices])

        transformed = jax.vmap(transform_one)(queries, train, embedding, active)
        return _restore_queries(
            transformed,
            case_shape=self.case_shape,
            query_shape=query_shape,
            output_shape=(self.out_size,),
        )


class LocallyLinearEmbeddingRecipe(AbstractRecipe):
    """Hard-neighborhood LLE, MLLE, Hessian-LLE, or LTSA spectral embedding."""

    n_components: int = eqx.field(static=True)
    n_neighbors: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    variant: LLEVariant = eqx.field(static=True)

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 8,
        regularization: float = 1e-3,
        variant: LLEVariant = "standard",
    ):
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        if int(n_neighbors) <= 0:
            raise ValueError("n_neighbors must be positive.")
        if float(regularization) <= 0.0:
            raise ValueError("regularization must be positive.")
        if variant not in ("standard", "modified", "hessian", "ltsa"):
            raise ValueError(f"Unsupported LLE variant {variant!r}.")
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.regularization = float(regularization)
        self.variant = variant

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        n = batch.sample_count
        d = self.n_components
        k = self.n_neighbors
        if d >= n or k >= n:
            raise ValueError(
                "LLE requires n_components < samples and n_neighbors < samples."
            )
        if self.variant == "modified" and k <= d + 1:
            raise ValueError("Modified LLE requires n_neighbors > n_components + 1.")
        if self.variant in ("hessian", "ltsa") and d > batch.feature_count:
            raise ValueError("Tangent-space LLE dimensions cannot exceed feature_count.")
        if self.variant == "hessian" and k < 1 + d + d * (d + 1) // 2:
            raise ValueError(
                "Hessian LLE requires enough neighbors for its tangent Hessian basis."
            )
        if self.variant == "ltsa" and k <= d:
            raise ValueError("LTSA requires n_neighbors > n_components.")
        x, weights, active = _fit_arrays(batch)
        graph = build_neighbor_graph(x, active, n_neighbors=k)
        cases = _case_count(batch.case_shape)
        flat_x = x.reshape((cases, n, batch.feature_count))
        flat_weights = weights.reshape((cases, n))
        flat_active = active.reshape((cases, n))
        flat_indices = graph.relation.source_indices.reshape((cases, n, k))
        embedding, eigenvalues, residual = jax.vmap(
            lambda x_, w_, a_, i_: _lle_alignment_one(
                x_, w_, a_, i_, d, self.regularization, self.variant
            )
        )(flat_x, flat_weights, flat_active, flat_indices)
        embedding = embedding.reshape(batch.case_shape + (n, d))
        eigenvalues = eigenvalues.reshape(batch.case_shape + (d,))
        residual = residual.reshape(batch.case_shape)
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(jnp.real(embedding)), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(jnp.imag(embedding)), axis=(-2, -1)
        )
        enough = effective >= max(k + 1, d + 2)
        connected = graph.components == 1
        valid = finite & enough & connected
        status = _fit_status(finite, enough, connected)
        diagnostics = ManifoldDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.sum(jnp.real(eigenvalues), axis=-1),
            iterations=1,
            effective_samples=effective,
            residual=residual,
            eigenvalues=eigenvalues,
            connected_components=graph.components,
            minimum_degree=graph.minimum_degree,
            maximum_degree=graph.maximum_degree,
            converged=True,
            method=f"lle-{self.variant}",
        )
        model = LocallyLinearEmbeddingModel(
            x,
            embedding,
            active,
            regularization=self.regularization,
            n_neighbors=k,
            variant=self.variant,
            case_shape=batch.case_shape,
        )
        transform_supported = self.variant in ("standard", "modified")
        contract = GradientContract(
            prediction_inputs="conditional" if transform_supported else "none",
            prediction_parameters="conditional" if transform_supported else "none",
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
            conditions=(
                "neighbor topology is held fixed",
                "retained eigenspaces are simple",
                "Hessian-LLE and LTSA are transductive only",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method=f"lle-{self.variant}",
            gradient_contract=contract,
        )


__all__ = ["LLEVariant", "LocallyLinearEmbeddingModel", "LocallyLinearEmbeddingRecipe"]
