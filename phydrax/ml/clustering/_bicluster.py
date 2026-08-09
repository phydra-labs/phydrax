#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from ._common import active_data, distances_to_centers, positive_scalar, real_dtype
from ._spectral import _deterministic_embedding_kmeans


class BiclusterDiagnostics(StrictModule):
    valid: Array
    status: Array
    objective: Array
    iterations: Array
    row_labels: Array
    column_labels: Array
    row_mass: Array
    column_mass: Array
    row_active: Array
    column_active: Array
    degeneracy: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        objective: Any,
        iterations: int,
        row_labels: Any,
        column_labels: Any,
        row_mass: Any,
        column_mass: Any,
        row_active: Any,
        column_active: Any,
        degeneracy: Any,
        method: str,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.row_labels = jax.lax.stop_gradient(jnp.asarray(row_labels, dtype=jnp.int32))
        self.column_labels = jax.lax.stop_gradient(
            jnp.asarray(column_labels, dtype=jnp.int32)
        )
        self.row_mass = jnp.asarray(row_mass)
        self.column_mass = jnp.asarray(column_mass)
        self.row_active = jnp.asarray(row_active, dtype=bool)
        self.column_active = jnp.asarray(column_active, dtype=bool)
        self.degeneracy = jnp.asarray(degeneracy, dtype=bool)
        self.method = str(method)


class BiclusterModel(AbstractArrayModel):
    """Blockwise row transform plus immutable terminal column partition."""

    row_centers: Array
    row_active: Array
    column_labels: Array
    column_active: Array
    temperature: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)

    def __init__(
        self,
        row_centers: Array,
        row_active: Array,
        column_labels: Array,
        column_active: Array,
        temperature: Any,
        /,
        *,
        method: str,
    ):
        self.row_centers = jnp.asarray(row_centers)
        self.row_active = jnp.asarray(row_active, dtype=bool)
        self.column_labels = jax.lax.stop_gradient(
            jnp.asarray(column_labels, dtype=jnp.int32)
        )
        self.column_active = jnp.asarray(column_active, dtype=bool)
        self.temperature = positive_scalar(
            jnp.asarray(temperature, dtype=real_dtype(self.row_centers.dtype)),
            "temperature",
        )
        self.in_size = self.row_centers.shape[-1]
        self.out_size = self.row_centers.shape[-2]
        self.case_shape = self.row_centers.shape[:-2]
        self.method = str(method)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        distance = distances_to_centers(
            jnp.asarray(x), self.row_centers, "squared-euclidean", self.case_shape
        )
        sample_ndim = distance.ndim - len(self.case_shape) - 1
        active = self.row_active.reshape(
            self.case_shape + (1,) * sample_ndim + (self.out_size,)
        )
        probability = jax.nn.softmax(
            jnp.where(active, -distance / self.temperature, -jnp.inf), axis=-1
        )
        return jnp.where(jnp.isfinite(probability), probability, 0.0)

    def hard_row_labels(self, x: Any, /) -> Array:
        return jax.lax.stop_gradient(jnp.argmax(self(x), axis=-1).astype(jnp.int32))


def _row_centers(x: Array, w: Array, labels: Array, count: int) -> tuple[Array, Array]:
    membership = jax.nn.one_hot(labels, count, dtype=w.dtype) * w[..., :, None]
    mass = jnp.sum(membership, axis=-2)
    centers = (
        jnp.einsum("...nk,...nf->...kf", membership, x)
        / jnp.maximum(mass, jnp.finfo(w.dtype).tiny)[..., :, None]
    )
    return centers, mass


def _column_mass(labels: Array, count: int, dtype: jnp.dtype) -> Array:
    return jnp.sum(jax.nn.one_hot(labels, count, dtype=dtype), axis=-2)


def _finish(
    batch: MLBatch,
    x: Array,
    w: Array,
    invalid: Array,
    row_labels: Array,
    column_labels: Array,
    row_count: int,
    column_count: int,
    iterations: int,
    temperature: Array,
    method: str,
    infeasible: Array | bool = False,
) -> FitResult:
    row_centers, row_mass = _row_centers(x, w, row_labels, row_count)
    column_mass = _column_mass(column_labels, column_count, w.dtype)
    row_active = row_mass > 0.0
    column_active = column_mass > 0.0
    distance = distances_to_centers(x, row_centers, "squared-euclidean", batch.case_shape)
    objective = jnp.sum(
        w * jnp.take_along_axis(distance, row_labels[..., None], axis=-1).squeeze(-1),
        axis=-1,
    ) / jnp.maximum(jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny)
    reported_row_labels = jnp.where(w > 0.0, row_labels, -1)
    enough = (jnp.sum(w > 0.0, axis=-1) >= row_count) & (
        batch.feature_count >= column_count
    )
    finite = jnp.isfinite(objective) & jnp.all(jnp.isfinite(row_centers), axis=(-2, -1))
    nonempty = jnp.all(row_active, axis=-1) & jnp.all(column_active, axis=-1)
    infeasible_ = jnp.asarray(infeasible, dtype=bool)
    valid = enough & finite & nonempty & ~invalid & ~infeasible_
    status = jnp.where(
        invalid | ~finite,
        ML_NONFINITE,
        jnp.where(
            infeasible_,
            ML_INFEASIBLE,
            jnp.where(~enough | ~nonempty, ML_INSUFFICIENT_DATA, ML_SUCCESS),
        ),
    )
    model = BiclusterModel(
        row_centers, row_active, column_labels, column_active, temperature, method=method
    )
    diagnostics = BiclusterDiagnostics(
        valid=valid,
        status=status,
        objective=objective,
        iterations=iterations,
        row_labels=reported_row_labels,
        column_labels=column_labels,
        row_mass=row_mass,
        column_mass=column_mass,
        row_active=row_active,
        column_active=column_active,
        degeneracy=~nonempty | infeasible_,
        method=method,
    )
    contract = GradientContract(
        prediction_inputs="smooth",
        prediction_parameters="smooth",
        fit_features="conditional",
        fit_weights="conditional",
        fit_hyperparameters="conditional",
        fit_mode="spectral" if "spectral" in method else "unrolled",
        nondifferentiable_outputs=(
            "hard_row_labels",
            "column_labels",
            "block assignments",
        ),
        conditions=(
            "fixed row and column partitions",
            "separated spectral subspaces when applicable",
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


class SpectralBiclustering(AbstractRecipe):
    row_clusters: int = eqx.field(static=True)
    column_clusters: int = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    temperature: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        row_clusters: int,
        column_clusters: int,
        /,
        *,
        max_iterations: int = 32,
        temperature: float = 1.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        if row_clusters <= 0 or column_clusters <= 0 or max_iterations <= 0:
            raise ValueError(
                "row_clusters, column_clusters, and max_iterations must be positive."
            )
        self.row_clusters = int(row_clusters)
        self.column_clusters = int(column_clusters)
        self.max_iterations = int(max_iterations)
        self.temperature = positive_scalar(temperature, "temperature")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if (
            self.row_clusters > batch.sample_count
            or self.column_clusters > batch.feature_count
        ):
            raise ValueError("bicluster counts exceed fixed matrix dimensions.")
        x, w, _, invalid = active_data(batch, self.weight_policy)
        active_rows = w > 0.0
        row_mean = jnp.mean(x, axis=-1, keepdims=True)
        column_mean = (
            jnp.einsum("...n,...nf->...f", w, x)
            / jnp.maximum(jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny)[..., None]
        )
        overall_mean = (
            jnp.sum(w[..., :, None] * x, axis=(-2, -1), keepdims=True)
            / jnp.maximum(
                jnp.sum(w, axis=-1) * batch.feature_count, jnp.finfo(w.dtype).tiny
            )[..., None, None]
        )
        checkerboard_residual = jnp.where(
            active_rows[..., :, None],
            x - row_mean - column_mean[..., None, :] + overall_mean,
            0.0,
        )
        left, _, right_h = jnp.linalg.svd(checkerboard_residual, full_matrices=False)
        row_embedding = left[..., :, : self.row_clusters]
        column_embedding = jnp.swapaxes(right_h[..., : self.column_clusters, :], -1, -2)
        _, row_labels, _ = _deterministic_embedding_kmeans(
            row_embedding, w, self.row_clusters, self.max_iterations
        )
        column_weights = jnp.where(
            jnp.sum(w, axis=-1, keepdims=True) > 0.0,
            jnp.ones(batch.case_shape + (batch.feature_count,), dtype=w.dtype),
            0.0,
        )
        _, column_labels, _ = _deterministic_embedding_kmeans(
            column_embedding, column_weights, self.column_clusters, self.max_iterations
        )
        return _finish(
            batch,
            x,
            w,
            invalid,
            row_labels,
            column_labels,
            self.row_clusters,
            self.column_clusters,
            self.max_iterations,
            self.temperature,
            "spectral-biclustering",
        )


class SpectralCoclustering(AbstractRecipe):
    row_clusters: int = eqx.field(static=True)
    column_clusters: int = eqx.field(static=True)
    kmeans_iterations: int = eqx.field(static=True)
    temperature: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        row_clusters: int,
        column_clusters: int,
        /,
        *,
        kmeans_iterations: int = 32,
        temperature: float = 1.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        if row_clusters <= 0 or column_clusters <= 0 or kmeans_iterations <= 0:
            raise ValueError(
                "row_clusters, column_clusters, and kmeans_iterations must be positive."
            )
        self.row_clusters = int(row_clusters)
        self.column_clusters = int(column_clusters)
        self.kmeans_iterations = int(kmeans_iterations)
        self.temperature = positive_scalar(temperature, "temperature")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if (
            self.row_clusters > batch.sample_count
            or self.column_clusters > batch.feature_count
        ):
            raise ValueError("co-cluster counts exceed fixed matrix dimensions.")
        x, w, _, invalid = active_data(batch, self.weight_policy)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise ValueError("spectral co-clustering requires real nonnegative features.")
        infeasible = jnp.any((x < 0.0) & (w[..., :, None] > 0.0), axis=(-2, -1))
        nonnegative = jnp.maximum(x, 0.0) * (w > 0.0)[..., :, None]
        row_degree = jnp.sum(nonnegative, axis=-1)
        column_degree = jnp.sum(nonnegative, axis=-2)
        normalized = (
            nonnegative
            / jnp.sqrt(jnp.maximum(row_degree, jnp.finfo(w.dtype).tiny))[..., :, None]
        )
        normalized = (
            normalized
            / jnp.sqrt(jnp.maximum(column_degree, jnp.finfo(w.dtype).tiny))[..., None, :]
        )
        left, _, right_h = jnp.linalg.svd(normalized, full_matrices=False)
        row_embedding = left[..., :, : self.row_clusters]
        column_embedding = jnp.swapaxes(right_h[..., : self.column_clusters, :], -1, -2)
        _, row_labels, _ = _deterministic_embedding_kmeans(
            row_embedding, w, self.row_clusters, self.kmeans_iterations
        )
        column_weights = jnp.where(column_degree > 0.0, 1.0, 0.0).astype(w.dtype)
        _, column_labels, _ = _deterministic_embedding_kmeans(
            column_embedding, column_weights, self.column_clusters, self.kmeans_iterations
        )
        return _finish(
            batch,
            x,
            w,
            invalid,
            row_labels,
            column_labels,
            self.row_clusters,
            self.column_clusters,
            self.kmeans_iterations,
            self.temperature,
            "spectral-coclustering",
            infeasible=infeasible,
        )


__all__ = [
    "BiclusterDiagnostics",
    "BiclusterModel",
    "SpectralBiclustering",
    "SpectralCoclustering",
]
