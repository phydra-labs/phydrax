#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import (
    ML_CAPACITY_EXHAUSTED,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_SUCCESS,
    MLBatch,
)
from phydrax.ml.clustering import (
    AffinityPropagation,
    AgglomerativeClustering,
    BiclusterModel,
    ConnectivityClustering,
    DBSCAN,
    DensityClusterModel,
    MeanShift,
    SpectralBiclustering,
    SpectralClustering,
    SpectralCoclustering,
)


def test_dbscan_finds_weighted_components_and_exposes_hard_and_soft_routes():
    features = jnp.array([[0.0], [5.0], [0.1], [5.1], [100.0]])
    batch = MLBatch(
        features,
        sample_mask=jnp.array([True, True, True, True, False]),
        sample_weight=jnp.array([1.0, 2.0, 1.0, 2.0, 100.0]),
    )
    result = DBSCAN(2, 4, radius=0.2, minimum_samples=2.0).fit_batch(batch)
    model = result.as_trainable()

    assert result.status == ML_SUCCESS
    assert isinstance(model, DensityClusterModel)
    assert jnp.array_equal(model(features[:4]), jnp.array([0, 1, 0, 1]))
    assert jnp.allclose(result.diagnostics.cluster_mass, jnp.array([2.0, 4.0]))
    membership = model.soft_membership(features[:4], temperature=0.25)
    assert membership.shape == (4, 2)
    assert jnp.allclose(jnp.sum(membership, axis=-1), 1.0)
    gradient = jax.grad(lambda point: model.soft_membership(point, temperature=0.25)[0])(
        jnp.array([0.05])
    )
    assert jnp.all(jnp.isfinite(gradient))
    assert result.gradient_contract.fit_mode == "stopped"
    assert "hard radius labels are terminal" in result.gradient_contract.conditions


def test_connectivity_clustering_keeps_isolated_points_but_reports_capacity():
    features = jnp.array([[0.0], [0.1], [3.0]])
    successful = ConnectivityClustering(2, 3, radius=0.2).fit_batch(MLBatch(features))
    exhausted = ConnectivityClustering(2, 3, radius=0.01).fit_batch(MLBatch(features))

    assert successful.status == ML_SUCCESS
    assert jnp.array_equal(successful.as_trainable()(features), jnp.array([0, 0, 1]))
    assert exhausted.status == ML_CAPACITY_EXHAUSTED
    assert exhausted.diagnostics.degeneracy
    assert not exhausted.valid


def test_dbscan_reports_no_core_points_and_rejects_fixed_capacity_overflow():
    no_core = DBSCAN(2, 3, radius=0.1, minimum_samples=2.0).fit_batch(
        MLBatch(jnp.array([[0.0], [2.0], [4.0]]))
    )

    assert no_core.status == ML_INSUFFICIENT_DATA
    assert no_core.diagnostics.degeneracy

    with pytest.raises(ValueError, match="capacities"):
        DBSCAN(4, 3).fit_batch(MLBatch(jnp.ones((3, 1))))


def test_mean_shift_returns_smooth_modes_and_declared_unrolled_gradients():
    features = jnp.array([[-2.0], [2.0], [-2.1], [2.1]])
    weights = jnp.array([1.0, 1.0, 1.5, 1.5])
    recipe = MeanShift(
        2,
        bandwidth=0.5,
        merge_tolerance=0.1,
        max_iterations=5,
        tolerance=1e6,
        initialization="first",
    )
    result = recipe.fit_batch(MLBatch(features, sample_weight=weights))
    model = result.as_trainable()
    probability = model(features)

    feature_gradient = jax.grad(
        lambda values: recipe.fit_batch(
            MLBatch(values, sample_weight=weights)
        ).as_trainable()(jnp.array([0.2]))[0]
    )(features)
    weight_gradient = jax.grad(
        lambda value: recipe.fit_batch(
            MLBatch(features, sample_weight=value)
        ).as_trainable()(jnp.array([0.2]))[0]
    )(weights)
    bandwidth_gradient = jax.grad(
        lambda bandwidth: (
            MeanShift(
                2,
                bandwidth=bandwidth,
                merge_tolerance=0.1,
                max_iterations=5,
                tolerance=1e6,
                initialization="first",
            )
            .fit_batch(MLBatch(features, sample_weight=weights))
            .as_trainable()(jnp.array([0.2]))[0]
        )
    )(jnp.asarray(0.5))

    assert result.status == ML_SUCCESS
    assert jnp.all(result.diagnostics.active_clusters)
    assert jnp.allclose(jnp.sum(probability, axis=-1), 1.0)
    assert result.gradient_contract.fit_mode == "unrolled"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))
    assert jnp.isfinite(bandwidth_gradient)


def test_affinity_propagation_has_a_deterministic_fallback_exemplar():
    features = jnp.array([[-1.0], [0.0], [1.0]])
    result = AffinityPropagation(
        2,
        preference=-100.0,
        max_iterations=3,
        tolerance=1e6,
        temperature=0.75,
    ).fit_batch(MLBatch(features))
    model = result.as_trainable()

    assert result.status == ML_SUCCESS
    assert jnp.sum(model.active_clusters) >= 1
    assert model.active_clusters[0]
    assert jnp.allclose(jnp.sum(model(features), axis=-1), 1.0)
    assert result.gradient_contract.fit_mode == "unrolled"
    assert "fixed exemplar top-k ordering" in result.gradient_contract.conditions


def test_spectral_clustering_is_jittable_vmappable_and_reports_disconnection():
    features = jnp.array([[-2.0], [2.0], [-1.8], [1.8]])
    result = SpectralClustering(
        2, gamma=0.5, temperature=0.4, kmeans_iterations=8
    ).fit_batch(MLBatch(features))
    model = result.as_trainable()
    probes = jnp.array([[-1.9], [1.9]])

    assert result.status == ML_SUCCESS
    assert jnp.all(result.diagnostics.active_clusters)
    assert jax.jit(model)(probes).shape == (2, 2)
    assert jax.vmap(model)(probes).shape == (2, 2)
    assert jnp.all(
        jnp.isfinite(jax.grad(lambda point: model(point)[0])(jnp.array([0.1])))
    )
    assert result.gradient_contract.fit_mode == "spectral"

    disconnected = SpectralClustering(1, gamma=1.0, kmeans_iterations=2).fit_batch(
        MLBatch(jnp.array([[0.0], [100.0]]))
    )
    assert disconnected.status == ML_INSUFFICIENT_DATA
    assert disconnected.diagnostics.degeneracy


def test_agglomerative_clustering_uses_lexicographic_merge_ties():
    features = jnp.array([[0.0], [4.0], [2.0]])
    result = AgglomerativeClustering(2, linkage="centroid").fit_batch(MLBatch(features))
    model = result.as_trainable()

    assert result.status == ML_SUCCESS
    assert jnp.allclose(model.centers, jnp.array([[1.0], [4.0]]))
    assert jnp.array_equal(model(features), jnp.array([0, 1, 0]))
    assert result.gradient_contract.fit_mode == "stopped"
    assert result.gradient_contract.conditions == (
        "deterministic lexicographic merge ties",
    )


def _checkerboard():
    return jnp.array(
        [
            [5.0, 1.0, 4.8, 1.2],
            [1.0, 5.0, 1.2, 4.8],
            [4.5, 1.1, 4.2, 1.0],
            [1.1, 4.5, 1.0, 4.2],
        ]
    )


@pytest.mark.parametrize(
    "recipe",
    [
        SpectralBiclustering(2, 2, max_iterations=8, temperature=0.5),
        SpectralCoclustering(2, 2, kmeans_iterations=8, temperature=0.5),
    ],
)
def test_biclustering_families_return_fixed_column_partitions_and_smooth_rows(recipe):
    features = _checkerboard()
    result = recipe.fit_batch(MLBatch(features))
    model = result.as_trainable()

    assert result.status == ML_SUCCESS
    assert isinstance(model, BiclusterModel)
    assert result.diagnostics.row_labels.shape == (4,)
    assert result.diagnostics.column_labels.shape == (4,)
    assert jnp.all(result.diagnostics.row_active)
    assert jnp.all(result.diagnostics.column_active)
    assert jnp.allclose(jnp.sum(model(features), axis=-1), 1.0)
    assert model.column_labels.shape == (4,)
    assert jnp.all(jnp.isfinite(jax.grad(lambda row: model(row)[0])(features[0])))
    assert result.gradient_contract.prediction_inputs == "smooth"
    assert "column_labels" in result.gradient_contract.nondifferentiable_outputs


def test_coclustering_rejects_complex_and_reports_negative_data_infeasible():
    complex_features = _checkerboard().astype(jnp.complex64) * (1.0 + 0.2j)
    with pytest.raises(ValueError, match="real nonnegative"):
        SpectralCoclustering(2, 2).fit_batch(MLBatch(complex_features))

    negative = _checkerboard().at[0, 0].set(-1.0)
    result = SpectralCoclustering(2, 2).fit_batch(MLBatch(negative))
    assert result.status == ML_INFEASIBLE
    assert result.diagnostics.degeneracy
