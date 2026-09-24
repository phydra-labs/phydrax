#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._trainable import ArrayRole, partition_parameters, require_parameter_roles
from phydrax.ml import MLBatch
from phydrax.ml.neighbors import KNeighborsClassifierRecipe, KNeighborsRegressorRecipe
from phydrax.ml.preprocessing import StandardScaler
from phydrax.ml.tree import DecisionTreeRegressor


def _features():
    return jnp.array(
        [
            [-1.3, -0.7],
            [-0.9, -1.2],
            [-0.5, -0.6],
            [0.6, 0.7],
            [1.0, 1.3],
            [1.4, 0.6],
        ]
    )


def _targets():
    return jnp.array([-1.0, -1.5, -0.4, 0.8, 1.6, 1.1])


def _roles(model):
    resolution = require_parameter_roles(model, context=type(model).__name__)
    return dict(zip(resolution.paths, resolution.roles, strict=True))


def _parameter_leaves(model):
    return jax.tree_util.tree_leaves(partition_parameters(model)[0])


def test_fitted_ridge_is_frozen_until_explicitly_unfrozen():
    result = phx.ml.fit(phx.ml.linear.RidgeRecipe(1e-3), _features(), _targets())

    assert set(_roles(result.model).values()) == {ArrayRole.FIXED}
    roles = _roles(result.as_trainable())
    assert roles[".coefficients"] is ArrayRole.PARAMETER
    assert roles[".intercept"] is ArrayRole.PARAMETER


def test_fitted_knn_training_set_never_trains():
    batch = MLBatch(_features(), _targets())
    regressor = KNeighborsRegressorRecipe(2).fit_batch(batch).as_trainable()
    labels = jnp.array([0, 0, 0, 1, 1, 1])
    classifier = (
        KNeighborsClassifierRecipe(2, class_count=2)
        .fit_batch(MLBatch(_features(), labels))
        .as_trainable()
    )

    for model in (regressor, classifier):
        roles = _roles(model)
        assert roles[".support"] is ArrayRole.FIXED
        assert _parameter_leaves(model) == []
    assert _roles(regressor)[".targets"] is ArrayRole.FIXED


def test_fitted_hard_tree_trains_leaf_values_not_thresholds():
    model = (
        DecisionTreeRegressor(max_depth=2)
        .fit_batch(MLBatch(_features(), _targets()))
        .as_trainable()
    )
    roles = _roles(model)

    assert roles[".threshold"] is ArrayRole.FIXED
    assert roles[".leaf_value"] is ArrayRole.PARAMETER


def test_fitted_scaler_statistics_never_train():
    model = StandardScaler().fit_batch(MLBatch(_features())).as_trainable()

    _roles(model)
    assert _parameter_leaves(model) == []
