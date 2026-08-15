#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

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
    ML_CAPACITY_EXHAUSTED,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)
from ._common import (
    active_data,
    ClusterDiagnostics,
    ClusterInitialization,
    distances_to_centers,
    effective_sample_count,
    initialize_centers,
    nonnegative_scalar,
    pairwise_distances,
    positive_scalar,
    SoftClusterModel,
    stable_top_indices,
)


class MeanShift(AbstractRecipe):
    center_capacity: int = eqx.field(static=True)
    bandwidth: Array
    merge_tolerance: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: Array
    initialization: ClusterInitialization = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        center_capacity: int,
        /,
        *,
        bandwidth: float = 1.0,
        merge_tolerance: float | None = None,
        max_iterations: int = 64,
        tolerance: float = 1e-4,
        initialization: ClusterInitialization = "first",
        weight_policy: WeightPolicy = "statistical",
    ):
        resolved_merge_tolerance = (
            0.5 * jnp.asarray(bandwidth) if merge_tolerance is None else merge_tolerance
        )
        if center_capacity <= 0 or max_iterations <= 0:
            raise ValueError("center_capacity and max_iterations must be positive.")
        if initialization not in ("random", "first", "k-means++"):
            raise ValueError("unsupported mean-shift initialization.")
        self.center_capacity = int(center_capacity)
        self.bandwidth = positive_scalar(bandwidth, "bandwidth")
        self.merge_tolerance = positive_scalar(
            resolved_merge_tolerance, "merge_tolerance"
        )
        self.max_iterations = int(max_iterations)
        self.tolerance = nonnegative_scalar(tolerance, "tolerance")
        self.initialization = initialization
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if self.center_capacity > batch.sample_count:
            raise ValueError("center_capacity cannot exceed sample capacity.")
        x, w, active_samples, invalid = active_data(batch, self.weight_policy)
        centers = initialize_centers(x, w, self.center_capacity, self.initialization, key)
        delta = jnp.full(batch.case_shape, jnp.inf, dtype=w.dtype)

        def step(_, state):
            centers, delta = state
            distance = distances_to_centers(
                x, centers, "squared-euclidean", batch.case_shape
            )
            kernel = (
                jnp.exp(-0.5 * distance / (self.bandwidth * self.bandwidth))
                * w[..., :, None]
            )
            mass = jnp.sum(kernel, axis=-2)
            proposed = (
                oe.contract("...nk,...nf->...kf", kernel, x)
                / jnp.maximum(mass, jnp.finfo(w.dtype).tiny)[..., :, None]
            )
            proposed = jnp.where((mass > 0.0)[..., :, None], proposed, centers)
            delta = jnp.max(jnp.linalg.norm(proposed - centers, axis=-1), axis=-1)
            return proposed, delta

        centers, delta = jax.lax.fori_loop(0, self.max_iterations, step, (centers, delta))
        center_distance = pairwise_distances(centers, "euclidean")
        earlier = (
            jnp.arange(self.center_capacity)[:, None]
            > jnp.arange(self.center_capacity)[None, :]
        )
        duplicate = jnp.any((center_distance <= self.merge_tolerance) & earlier, axis=-1)
        active_centers = ~duplicate
        sample_distance = distances_to_centers(
            x, centers, "squared-euclidean", batch.case_shape
        )
        logits = jnp.where(
            active_centers[..., None, :],
            -0.5 * sample_distance / (self.bandwidth * self.bandwidth),
            -jnp.inf,
        )
        responsibility = jax.nn.softmax(logits, axis=-1)
        cluster_mass = jnp.sum(w[..., :, None] * responsibility, axis=-2)
        active_centers &= cluster_mass > 0.0
        covered = (
            jnp.any(
                (sample_distance <= self.bandwidth * self.bandwidth)
                & active_centers[..., None, :],
                axis=-1,
            )
            | ~active_samples
        )
        exhausted = jnp.any(~covered, axis=-1)
        objective = jnp.sum(
            w * jnp.sum(responsibility * sample_distance, axis=-1), axis=-1
        ) / jnp.maximum(jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny)
        converged = delta <= self.tolerance
        enough = jnp.any(active_samples, axis=-1)
        finite = jnp.isfinite(objective) & jnp.all(jnp.isfinite(centers), axis=(-2, -1))
        valid = enough & finite & converged & ~invalid & ~exhausted
        status = jnp.where(
            invalid | ~finite,
            ML_NONFINITE,
            jnp.where(
                ~enough,
                ML_INSUFFICIENT_DATA,
                jnp.where(
                    exhausted,
                    ML_CAPACITY_EXHAUSTED,
                    jnp.where(~converged, ML_NONCONVERGED, ML_SUCCESS),
                ),
            ),
        )
        model = SoftClusterModel(
            centers, active_centers, self.bandwidth * self.bandwidth, method="mean-shift"
        )
        diagnostics = ClusterDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.max_iterations,
            effective_samples=effective_sample_count(w),
            cluster_mass=cluster_mass,
            active_clusters=active_centers,
            empty_clusters_seen=False,
            converged=converged,
            degeneracy=exhausted,
            method="mean-shift",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("hard_labels", "merged mode mask"),
            conditions=("fixed seed and merge ordering", "positive bandwidth"),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="mean-shift",
            gradient_contract=contract,
        )


class AffinityPropagation(AbstractRecipe):
    exemplar_capacity: int = eqx.field(static=True)
    preference: Array
    preference_is_data_driven: bool = eqx.field(static=True)
    damping: Array
    temperature: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        exemplar_capacity: int,
        /,
        *,
        preference: float | None = None,
        damping: float = 0.5,
        temperature: float = 1.0,
        max_iterations: int = 128,
        tolerance: float = 1e-4,
        weight_policy: WeightPolicy = "statistical",
    ):
        if exemplar_capacity <= 0 or max_iterations <= 0:
            raise ValueError("exemplar_capacity and max_iterations must be positive.")
        self.exemplar_capacity = int(exemplar_capacity)
        self.preference_is_data_driven = preference is None
        preference_ = jnp.asarray(0.0 if preference is None else preference)
        if preference_.ndim != 0:
            raise ValueError("preference must be a scalar.")
        self.preference = eqx.error_if(
            preference_, ~jnp.isfinite(preference_), "preference must be finite."
        )
        damping_ = jnp.asarray(damping)
        if damping_.ndim != 0:
            raise ValueError("damping must be a scalar.")
        self.damping = eqx.error_if(
            damping_,
            ~jnp.isfinite(damping_) | (damping_ < 0.0) | (damping_ >= 1.0),
            "damping must be finite in [0, 1).",
        )
        self.temperature = positive_scalar(temperature, "temperature")
        self.max_iterations = int(max_iterations)
        self.tolerance = nonnegative_scalar(tolerance, "tolerance")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if self.exemplar_capacity > batch.sample_count:
            raise ValueError("exemplar_capacity cannot exceed sample capacity.")
        x, w, active, invalid = active_data(batch, self.weight_policy)
        n = batch.sample_count
        distance = pairwise_distances(x, "squared-euclidean")
        pair_active = active[..., :, None] & active[..., None, :]
        pair_weight = jnp.sqrt(w[..., :, None] * w[..., None, :])
        similarity = jnp.where(
            pair_active, -distance * pair_weight, -jnp.asarray(1e30, dtype=w.dtype)
        )
        if self.preference_is_data_driven:
            preference = jnp.nanmedian(
                jnp.where(pair_active & ~jnp.eye(n, dtype=bool), similarity, jnp.nan),
                axis=(-2, -1),
            )
            preference = jnp.where(jnp.isfinite(preference), preference, 0.0)
        else:
            preference = jnp.full(batch.case_shape, self.preference, dtype=w.dtype)
        diagonal = jnp.arange(n)
        similarity = similarity.at[..., diagonal, diagonal].set(
            jnp.where(active, preference[..., None], -jnp.asarray(1e30, dtype=w.dtype))
        )
        availability = jnp.zeros_like(similarity)
        responsibility = jnp.zeros_like(similarity)
        delta = jnp.full(batch.case_shape, jnp.inf, dtype=w.dtype)
        identity = jnp.eye(n, dtype=bool)

        def step(_, state):
            availability, responsibility, delta = state
            combined = availability + similarity
            best_index = jnp.argmax(combined, axis=-1)
            best = jnp.max(combined, axis=-1)
            without_best = jnp.where(
                jax.nn.one_hot(best_index, n, dtype=bool), -jnp.inf, combined
            )
            second = jnp.max(without_best, axis=-1)
            second = jnp.where(jnp.isfinite(second), second, 0.0)
            excluded = jnp.where(
                jax.nn.one_hot(best_index, n, dtype=bool),
                second[..., None],
                best[..., None],
            )
            proposed_r = similarity - excluded
            positive = jnp.maximum(proposed_r, 0.0)
            positive_off = jnp.where(identity, 0.0, positive)
            column_sum = jnp.sum(positive_off, axis=-2)
            r_diag = jnp.diagonal(proposed_r, axis1=-2, axis2=-1)
            proposed_a = jnp.minimum(
                0.0, r_diag[..., None, :] + column_sum[..., None, :] - positive_off
            )
            proposed_a = jnp.where(identity, column_sum[..., None, :], proposed_a)
            proposed_a = jnp.where(pair_active, proposed_a, 0.0)
            proposed_r = jnp.where(pair_active, proposed_r, 0.0)
            next_a = self.damping * availability + (1.0 - self.damping) * proposed_a
            next_r = self.damping * responsibility + (1.0 - self.damping) * proposed_r
            delta = jnp.maximum(
                jnp.max(jnp.abs(next_a - availability), axis=(-2, -1)),
                jnp.max(jnp.abs(next_r - responsibility), axis=(-2, -1)),
            )
            return next_a, next_r, delta

        availability, responsibility, delta = jax.lax.fori_loop(
            0, self.max_iterations, step, (availability, responsibility, delta)
        )
        exemplar_score = jnp.diagonal(availability + responsibility, axis1=-2, axis2=-1)
        exemplar_score = jnp.where(active, exemplar_score, -jnp.inf)
        indices = stable_top_indices(exemplar_score, self.exemplar_capacity)
        selected_score = jnp.take_along_axis(exemplar_score, indices, axis=-1)
        active_exemplars = selected_score > 0.0
        none = ~jnp.any(active_exemplars, axis=-1)
        active_exemplars = active_exemplars.at[..., 0].set(
            active_exemplars[..., 0] | none
        )
        centers = jnp.take_along_axis(x, indices[..., :, None], axis=-2)
        sample_distance = distances_to_centers(
            x, centers, "squared-euclidean", batch.case_shape
        )
        assignment_probability = jax.nn.softmax(
            jnp.where(
                active_exemplars[..., None, :],
                -sample_distance / self.temperature,
                -jnp.inf,
            ),
            axis=-1,
        )
        cluster_mass = jnp.sum(w[..., :, None] * assignment_probability, axis=-2)
        positive_count = jnp.sum((exemplar_score > 0.0) & active, axis=-1)
        exhausted = positive_count > self.exemplar_capacity
        converged = delta <= self.tolerance
        enough = jnp.any(active, axis=-1)
        finite = jnp.all(jnp.isfinite(centers), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(jnp.where(pair_active, responsibility, 0.0)), axis=(-2, -1)
        )
        objective = -jnp.sum(w * jnp.max(similarity, axis=-1), axis=-1) / jnp.maximum(
            jnp.sum(w, axis=-1), jnp.finfo(w.dtype).tiny
        )
        valid = enough & finite & converged & ~invalid & ~exhausted
        status = jnp.where(
            invalid | ~finite,
            ML_NONFINITE,
            jnp.where(
                ~enough,
                ML_INSUFFICIENT_DATA,
                jnp.where(
                    exhausted,
                    ML_CAPACITY_EXHAUSTED,
                    jnp.where(~converged, ML_NONCONVERGED, ML_SUCCESS),
                ),
            ),
        )
        model = SoftClusterModel(
            centers, active_exemplars, self.temperature, method="affinity-propagation"
        )
        diagnostics = ClusterDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.max_iterations,
            effective_samples=effective_sample_count(w),
            cluster_mass=cluster_mass,
            active_clusters=active_exemplars,
            empty_clusters_seen=False,
            converged=converged,
            degeneracy=exhausted,
            method="affinity-propagation",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("hard_labels", "exemplar selection"),
            conditions=("fixed message iterations", "fixed exemplar top-k ordering"),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="affinity-propagation",
            gradient_contract=contract,
        )


__all__ = ["AffinityPropagation", "MeanShift"]
