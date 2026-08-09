#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import ML_SUCCESS, MLBatch
from phydrax.ml.clustering import (
    AffinityPropagation,
    DensityClusterModel,
    SpectralBiclustering,
    SpectralClustering,
    SpectralCoclustering,
)


_CLUSTER_DATA = jnp.array(
    [
        [-2.3, -0.4],
        [2.1, 0.7],
        [-1.7, 0.2],
        [2.8, 1.4],
        [-0.6, 1.8],
        [1.2, -1.3],
    ]
)
_CLUSTER_WEIGHT = jnp.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.3])
_BLOCK_DATA = jnp.array(
    [
        [5.0, 1.0, 4.8, 1.2],
        [1.0, 5.0, 1.2, 4.8],
        [4.5, 1.1, 4.2, 1.0],
        [1.1, 4.5, 1.0, 4.2],
    ]
)
_BLOCK_WEIGHT = jnp.array([1.0, 1.2, 0.9, 1.1])


@pytest.mark.parametrize(
    "recipe, features, weights, point",
    [
        (
            AffinityPropagation(
                2,
                preference=-100.0,
                temperature=0.7,
                max_iterations=3,
                tolerance=1e6,
            ),
            _CLUSTER_DATA,
            _CLUSTER_WEIGHT,
            jnp.array([0.2, -0.1]),
        ),
        (
            SpectralClustering(
                2,
                gamma=0.4,
                temperature=0.7,
                kmeans_iterations=8,
            ),
            _CLUSTER_DATA,
            _CLUSTER_WEIGHT,
            jnp.array([0.2, -0.1]),
        ),
        (
            SpectralBiclustering(2, 2, max_iterations=8, temperature=0.7),
            _BLOCK_DATA,
            _BLOCK_WEIGHT,
            _BLOCK_DATA[0],
        ),
        (
            SpectralCoclustering(2, 2, kmeans_iterations=8, temperature=0.7),
            _BLOCK_DATA,
            _BLOCK_WEIGHT,
            _BLOCK_DATA[0],
        ),
    ],
)
def test_each_smooth_clustering_family_honors_declared_fit_gradients(
    recipe, features, weights, point
):
    def feature_loss(values):
        return recipe.fit_batch(MLBatch(values, sample_weight=weights)).as_trainable()(
            point
        )[0]

    def weight_loss(value):
        return recipe.fit_batch(MLBatch(features, sample_weight=value)).as_trainable()(
            point
        )[0]

    result = recipe.fit_batch(MLBatch(features, sample_weight=weights))
    feature_gradient = jax.grad(feature_loss)(features)
    weight_gradient = jax.grad(weight_loss)(weights)

    assert result.status == ML_SUCCESS
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))


@pytest.mark.parametrize(
    "factory, features, weights, point",
    [
        (
            lambda temperature: AffinityPropagation(
                2,
                preference=-100.0,
                temperature=temperature,
                max_iterations=3,
                tolerance=1e6,
            ),
            _CLUSTER_DATA,
            _CLUSTER_WEIGHT,
            jnp.array([0.2, -0.1]),
        ),
        (
            lambda temperature: SpectralClustering(
                2,
                gamma=0.4,
                temperature=temperature,
                kmeans_iterations=8,
            ),
            _CLUSTER_DATA,
            _CLUSTER_WEIGHT,
            jnp.array([0.2, -0.1]),
        ),
        (
            lambda temperature: SpectralBiclustering(
                2, 2, max_iterations=8, temperature=temperature
            ),
            _BLOCK_DATA,
            _BLOCK_WEIGHT,
            _BLOCK_DATA[0],
        ),
        (
            lambda temperature: SpectralCoclustering(
                2, 2, kmeans_iterations=8, temperature=temperature
            ),
            _BLOCK_DATA,
            _BLOCK_WEIGHT,
            _BLOCK_DATA[0],
        ),
    ],
)
def test_each_smooth_clustering_family_honors_declared_hyperparameter_gradient(
    factory, features, weights, point
):
    gradient = jax.grad(
        lambda temperature: (
            factory(temperature)
            .fit_batch(MLBatch(features, sample_weight=weights))
            .as_trainable()(point)[0]
        )
    )(jnp.asarray(0.7))

    assert jnp.isfinite(gradient)


def test_density_soft_membership_has_declared_parameter_gradient():
    core_points = jnp.array([[0.0], [0.2], [3.0], [3.2]])
    core_labels = jnp.array([0, 0, 1, 1])
    core_active = jnp.ones(4, dtype=bool)
    cluster_active = jnp.ones(2, dtype=bool)
    point = jnp.array([0.1])

    def loss(points):
        model = DensityClusterModel(
            points,
            core_labels,
            core_active,
            cluster_active,
            0.4,
            method="density-parameter-gradient",
        )
        return model.soft_membership(point, temperature=0.3)[0]

    gradient = jax.grad(loss)(core_points)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0.0)
