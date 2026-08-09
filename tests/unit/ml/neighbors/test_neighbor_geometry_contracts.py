#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.ml import MLBatch
from phydrax.ml.neighbors import (
    KernelDensityRecipe,
    LocalOutlierFactorRecipe,
    MahalanobisMetricRecipe,
    NearestCentroidRecipe,
    NeighborhoodComponentsAnalysisRecipe,
)


def _cluster_data():
    return jnp.array(
        [
            [-1.2, -0.8],
            [-0.9, -1.1],
            [-0.6, -0.7],
            [0.7, 0.8],
            [1.0, 1.2],
            [1.3, 0.7],
        ]
    )


def test_kernel_density_preserves_measure_weights_masks_chunking_and_gradients():
    features = _cluster_data()
    measure = jnp.array([1.0, 2.0, 1.0, 0.5, 1.5, 2.0])
    result = KernelDensityRecipe(0.45, weight_policy="measure").fit_batch(
        MLBatch(
            features,
            measure_weight=measure,
            sample_mask=jnp.array([True, True, False, True, True, True]),
        )
    )
    model = result.as_trainable()
    query = jnp.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])

    assert model(query).shape == (3,)
    assert model.score_samples(query).shape == (3,)
    assert jnp.allclose(jnp.exp(model.score_samples(query)), model(query))
    assert jnp.allclose(model(query), model.predict_chunked(query, chunk_size=1))
    assert result.diagnostics.effective_samples == 5
    assert result.gradient_contract.fit_mode == "direct"
    assert jax.jit(model)(query).shape == (3,)

    input_gradient = jax.grad(lambda point: model(point))(query[0])
    bandwidth_gradient = jax.grad(
        lambda bandwidth: (
            KernelDensityRecipe(bandwidth)
            .fit_batch(MLBatch(features))
            .as_trainable()(query[0])
        )
    )(jnp.array(0.45))
    feature_gradient = jax.grad(
        lambda value: (
            KernelDensityRecipe(0.45).fit_batch(MLBatch(value)).as_trainable()(query[0])
        )
    )(features)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.isfinite(bandwidth_gradient)
    assert jnp.all(jnp.isfinite(feature_gradient))


def test_local_outlier_factor_uses_chunked_weighted_geometry_and_hard_output():
    inliers = _cluster_data()
    features = jnp.concatenate((inliers, jnp.array([[5.0, 5.0]])), axis=0)
    weights = jnp.array([1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5])
    result = LocalOutlierFactorRecipe(2, metric="euclidean", chunk_size=2).fit_batch(
        MLBatch(features, sample_weight=weights)
    )
    model = result.as_trainable()
    scores = model.score_samples(features)

    assert scores.shape == (7,)
    assert scores[-1] > jnp.median(scores[:-1])
    assert model.predict(features, threshold=1.5).dtype == jnp.int32
    assert jnp.allclose(scores, model.predict_chunked(features, chunk_size=3))
    assert result.diagnostics.method == "chunked-local-outlier-factor"
    assert "neighbor_indices" in result.gradient_contract.nondifferentiable_outputs
    assert "predict" in result.gradient_contract.nondifferentiable_outputs


def test_nearest_centroid_probabilities_cases_masks_and_hard_contract():
    features = _cluster_data()
    labels = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    cases = jnp.stack((features, features + jnp.array([0.1, -0.2])), axis=0)
    case_labels = jnp.stack((labels, labels), axis=0)
    result = NearestCentroidRecipe(class_count=2, temperature=0.3).fit_batch(
        MLBatch(
            cases,
            case_labels,
            target_mask=jnp.array([True, True, False, True, True, True]),
            sample_weight=jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 1.0]),
        )
    )
    model = result.as_trainable()
    probability = model.probabilities(cases[:, :2])

    assert probability.shape == (2, 2, 2)
    assert jnp.allclose(jnp.sum(probability, axis=-1), 1.0)
    assert model.predict(cases[:, :2]).dtype == jnp.int32
    assert "predict" in result.gradient_contract.nondifferentiable_outputs
    assert jnp.all(
        jnp.isfinite(
            jax.grad(lambda point: jnp.sum(model.probabilities(point) ** 2))(cases[:, 0])
        )
    )


def test_nca_embedding_geometry_unrolled_gradients_jit_and_vmap():
    features = _cluster_data()
    labels = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    recipe = NeighborhoodComponentsAnalysisRecipe(
        component_count=2,
        iterations=3,
        learning_rate=0.01,
        temperature=0.8,
        ridge=1e-3,
    )
    result = recipe.fit_batch(MLBatch(features, labels))
    model = result.as_trainable()
    embedded = model(features)

    assert embedded.shape == features.shape
    assert model.metric_matrix.shape == (2, 2)
    assert jnp.all(jnp.linalg.eigvalsh(model.metric_matrix) >= -1e-6)
    assert jax.jit(model)(features).shape == features.shape
    assert jax.vmap(model)(features).shape == features.shape
    assert result.diagnostics.iterations == 3
    assert result.gradient_contract.fit_mode == "unrolled"

    prediction_gradient = jax.grad(lambda point: jnp.sum(model(point) ** 2))(features[0])
    feature_gradient = jax.grad(
        lambda value: jnp.sum(
            NeighborhoodComponentsAnalysisRecipe(iterations=2)
            .fit_batch(MLBatch(value, labels))
            .as_trainable()(features[:1])
            ** 2
        )
    )(features)
    assert jnp.all(jnp.isfinite(prediction_gradient))
    assert jnp.all(jnp.isfinite(feature_gradient))


def test_mahalanobis_unsupervised_and_supervised_whitening_geometry():
    features = _cluster_data()
    labels = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    unsupervised = MahalanobisMetricRecipe(ridge=1e-3).fit_batch(MLBatch(features))
    supervised = MahalanobisMetricRecipe(ridge=1e-3, component_count=1).fit_batch(
        MLBatch(features, labels, sample_weight=jnp.arange(1.0, 7.0))
    )
    unsupervised_model = unsupervised.as_trainable()
    supervised_model = supervised.as_trainable()

    assert unsupervised_model(features).shape == (6, 2)
    assert supervised_model(features).shape == (6, 1)
    assert unsupervised_model.metric_matrix.shape == (2, 2)
    assert jnp.all(jnp.linalg.eigvalsh(unsupervised_model.metric_matrix) > 0.0)
    assert supervised.diagnostics.rank == 2
    assert supervised.gradient_contract.fit_mode == "spectral"
    assert jnp.all(
        jnp.isfinite(
            jax.grad(lambda point: supervised_model.squared_distance(point, features[0]))(
                features[-1]
            )
        )
    )
