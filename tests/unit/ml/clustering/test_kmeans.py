#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import (
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
    MLBatch,
)
from phydrax.ml.clustering import (
    HardClusterModel,
    KMeans,
    KMedoids,
    MiniBatchKMeans,
    SoftClusterModel,
    SoftKMeans,
    StreamingKMeans,
)


def test_kmeans_preserves_case_sample_feature_and_ignored_target_axes():
    features = jnp.array(
        [
            [[-3.0], [3.0], [-2.0], [2.0]],
            [[10.0], [20.0], [11.0], [19.0]],
        ]
    )
    targets = jnp.arange(16.0).reshape(2, 4, 2)
    recipe = KMeans(2, initialization="first", tolerance=1e-6)
    result = recipe.fit_batch(MLBatch(features, targets))
    model = result.as_trainable()

    assert result.status.shape == (2,)
    assert jnp.all(result.status == ML_SUCCESS)
    assert model.centers.shape == (2, 2, 1)
    assert jnp.allclose(
        model.centers,
        jnp.array([[[-2.5], [2.5]], [[10.5], [19.5]]]),
        atol=1e-5,
    )
    expected = jnp.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=jnp.int32)
    assert jnp.array_equal(result.model(features), expected)
    assert jnp.array_equal(model(features), expected)
    assert recipe.cluster_count == 2


def test_kmeans_product_weights_sample_and_feature_masks_route_to_the_fit():
    features = jnp.array([[0.0], [10.0], [100.0], [999.0]])
    batch = MLBatch(
        features,
        jnp.arange(4.0),
        feature_mask=jnp.array([[True], [True], [True], [False]]),
        sample_mask=jnp.array([True, True, False, True]),
        sample_weight=jnp.array([1.0, 3.0, 99.0, 7.0]),
        measure_weight=jnp.array([2.0, 1.0, 5.0, 11.0]),
    )
    product = KMeans(1, initialization="first", weight_policy="product").fit_batch(batch)
    statistical = KMeans(
        1, initialization="first", weight_policy="statistical"
    ).fit_batch(batch)

    assert product.status == ML_SUCCESS
    assert jnp.allclose(product.as_trainable().centers[0, 0], 6.0, atol=1e-6)
    assert jnp.allclose(statistical.as_trainable().centers[0, 0], 7.5, atol=1e-6)
    assert jnp.allclose(product.diagnostics.cluster_mass, jnp.array([5.0]))
    assert jnp.allclose(product.diagnostics.effective_samples, 25.0 / 13.0)


def test_hard_and_soft_cluster_models_have_exact_ties_and_distinct_gradients():
    centers = jnp.array([[-1.0], [1.0]])
    active = jnp.array([True, True])
    hard = HardClusterModel(centers, active, method="test-hard")
    soft = SoftClusterModel(centers, active, 0.5, method="test-soft")
    midpoint = jnp.array([0.0])

    assert hard(midpoint) == 0
    assert jnp.array_equal(hard(jnp.array([[-0.2], [0.0], [0.2]])), jnp.array([0, 0, 1]))
    assert jnp.allclose(soft(midpoint), jnp.array([0.5, 0.5]), atol=1e-7)
    assert soft.hard_labels(midpoint) == 0
    assert jnp.array_equal(
        jax.grad(lambda point: hard(point).astype(jnp.float32))(midpoint),
        jnp.zeros_like(midpoint),
    )
    soft_gradient = jax.grad(lambda point: soft(point)[0])(midpoint)
    assert jnp.all(jnp.isfinite(soft_gradient))
    assert jnp.any(soft_gradient != 0.0)
    assert jnp.allclose(jax.jit(soft)(midpoint), soft(midpoint))
    assert jax.vmap(soft)(jnp.array([[-0.5], [0.5]])).shape == (2, 2)


def test_soft_kmeans_exercises_declared_prediction_and_fit_gradients():
    features = jnp.array([[-2.0], [2.0], [-1.5], [1.5]])
    weights = jnp.array([1.0, 1.2, 0.9, 1.1])
    point = jnp.array([0.25])
    recipe = SoftKMeans(
        2,
        temperature=0.8,
        max_iterations=5,
        tolerance=1e6,
        initialization="first",
    )
    result = recipe.fit_batch(MLBatch(features, sample_weight=weights))
    model = result.as_trainable()

    feature_gradient = jax.grad(
        lambda values: recipe.fit_batch(
            MLBatch(values, sample_weight=weights)
        ).as_trainable()(point)[0]
    )(features)
    weight_gradient = jax.grad(
        lambda value: recipe.fit_batch(
            MLBatch(features, sample_weight=value)
        ).as_trainable()(point)[0]
    )(weights)
    parameter_gradient = jax.grad(
        lambda centers: SoftClusterModel(
            centers, model.active_clusters, model.temperature, method="parameter-gradient"
        )(point)[0]
    )(model.centers)
    temperature_gradient = jax.grad(
        lambda temperature: (
            SoftKMeans(
                2,
                temperature=temperature,
                max_iterations=5,
                tolerance=1e6,
                initialization="first",
            )
            .fit_batch(MLBatch(features, sample_weight=weights))
            .as_trainable()(point)[0]
        )
    )(jnp.asarray(0.8))

    assert result.status == ML_SUCCESS
    assert result.gradient_contract.prediction_inputs == "smooth"
    assert result.gradient_contract.prediction_parameters == "smooth"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"
    assert result.gradient_contract.fit_mode == "unrolled"
    assert jnp.all(jnp.isfinite(jax.grad(lambda value: model(value)[0])(point)))
    assert jnp.all(jnp.isfinite(parameter_gradient))
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))
    assert jnp.isfinite(temperature_gradient)


def test_kmedoids_returns_observations_and_uses_deterministic_manhattan_ties():
    features = jnp.array([[0.0], [9.0], [2.0], [10.0]])
    result = KMedoids(
        2, metric="manhattan", initialization="first", max_iterations=16
    ).fit_batch(MLBatch(features))
    model = result.as_trainable()

    assert result.status == ML_SUCCESS
    assert jnp.allclose(model.centers, jnp.array([[0.0], [9.0]]))
    assert jnp.array_equal(model(features), jnp.array([0, 1, 0, 1]))
    assert jnp.all(
        jnp.any(model.centers[:, None, :] == features[None, :, :], axis=(1, 2))
    )
    assert result.gradient_contract.fit_mode == "stopped"
    assert result.gradient_contract.prediction_inputs == "none"


def test_minibatch_kmeans_requires_a_key_and_replays_it_exactly():
    features = jnp.array([[-3.0], [3.0], [-2.0], [2.0], [-1.0], [1.0]])
    recipe = MiniBatchKMeans(
        2,
        batch_size=3,
        max_iterations=6,
        initialization="random",
        empty_policy="reseed",
    )

    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(MLBatch(features))

    key = jax.random.key(17)
    first = recipe.fit_batch(MLBatch(features), key=key)
    second = recipe.fit_batch(MLBatch(features), key=key)

    assert first.status == ML_SUCCESS
    assert jnp.array_equal(first.as_trainable().centers, second.as_trainable().centers)
    assert jnp.array_equal(first.model(features), second.model(features))
    assert first.gradient_contract.fit_mode == "stopped"
    assert "explicit random key" in first.gradient_contract.conditions


def test_streaming_kmeans_updates_immutably_with_hard_and_soft_models():
    state = StreamingKMeans(jnp.array([[0.0], [10.0]]))
    updated = state.update(
        jnp.array([[2.0], [8.0], [100.0]]),
        weights=jnp.array([1.0, 1.0, 50.0]),
        mask=jnp.array([True, True, False]),
    )

    assert jnp.array_equal(state.centers, jnp.array([[0.0], [10.0]]))
    assert jnp.array_equal(state.cluster_mass, jnp.zeros(2))
    assert jnp.allclose(updated.centers, jnp.array([[2.0], [8.0]]))
    assert jnp.allclose(updated.cluster_mass, jnp.ones(2))
    assert updated.updates == 1
    assert jnp.array_equal(updated.model()(jnp.array([[1.0], [9.0]])), jnp.array([0, 1]))
    probability = updated.model(temperature=0.5)(jnp.array([[1.0], [9.0]]))
    assert probability.shape == (2, 2)
    assert jnp.allclose(jnp.sum(probability, axis=-1), 1.0)


def test_kmeans_supports_complex_features_with_real_distances_and_centers():
    features = jnp.array([[-2.0 + 1.0j], [2.0 - 1.0j], [-1.0 + 1.0j], [1.0 - 1.0j]])
    result = KMeans(2, initialization="first").fit_batch(MLBatch(features))
    model = result.as_trainable()

    assert result.status == ML_SUCCESS
    assert jnp.issubdtype(model.centers.dtype, jnp.complexfloating)
    assert jnp.allclose(model.centers, jnp.array([[-1.5 + 1.0j], [1.5 - 1.0j]]))
    assert jnp.array_equal(model(features), jnp.array([0, 1, 0, 1]))


def test_kmeans_reports_empty_underfull_nonfinite_and_nonconverged_cases():
    constant = KMeans(
        2, initialization="first", empty_policy="error", max_iterations=3
    ).fit_batch(MLBatch(jnp.ones((3, 1))))
    empty = KMeans(1, initialization="first").fit_batch(
        MLBatch(jnp.arange(3.0)[:, None], sample_mask=jnp.zeros(3, dtype=bool))
    )
    singleton = KMeans(1, initialization="first").fit_batch(MLBatch(jnp.array([[4.0]])))
    nonfinite = KMeans(1, initialization="first").fit_batch(
        MLBatch(jnp.array([[0.0], [jnp.nan], [1.0]]))
    )
    nonconverged = KMeans(
        2, initialization="first", max_iterations=1, tolerance=0.0
    ).fit_batch(MLBatch(jnp.array([[0.0], [10.0], [1.0], [9.0]])))

    assert constant.status == ML_INSUFFICIENT_DATA
    assert constant.diagnostics.empty_clusters_seen
    assert empty.status == ML_INSUFFICIENT_DATA
    assert singleton.status == ML_SUCCESS
    assert nonfinite.status == ML_NONFINITE
    assert nonconverged.status == ML_NONCONVERGED
    assert not nonconverged.diagnostics.converged

    with pytest.raises(ValueError, match="sample capacity"):
        KMeans(4, initialization="first").fit_batch(MLBatch(jnp.ones((3, 1))))


def test_case_bound_models_reject_missing_or_wrong_case_axes():
    cases = jnp.array(
        [
            [[-2.0], [2.0], [-1.0], [1.0]],
            [[8.0], [12.0], [9.0], [11.0]],
        ]
    )
    model = KMeans(2, initialization="first").fit_batch(MLBatch(cases)).as_trainable()

    with pytest.raises(ValueError, match="case"):
        model(jnp.array([0.0]))
    with pytest.raises(ValueError, match="case"):
        model(jnp.zeros((3, 1)))
