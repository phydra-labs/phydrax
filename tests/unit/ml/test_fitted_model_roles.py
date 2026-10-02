#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._trainable import ArrayRole, partition_parameters, require_parameter_roles
from phydrax.ml import MLBatch
from phydrax.ml.decomposition import PCA, SubspaceModel
from phydrax.ml.neighbors import KNeighborsClassifierRecipe, KNeighborsRegressorRecipe
from phydrax.ml.preprocessing import StandardScaler
from phydrax.ml.tree import DecisionTreeRegressor


def _features() -> Any:
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


def _targets() -> Any:
    return jnp.array([-1.0, -1.5, -0.4, 0.8, 1.6, 1.1])


def _roles(model: Any) -> Any:
    resolution = require_parameter_roles(model, context=type(model).__name__)
    return dict(zip(resolution.paths, resolution.roles, strict=True))


def _parameter_leaves(model: Any) -> Any:
    return jax.tree_util.tree_leaves(partition_parameters(model)[0])


def test_fitted_contracts() -> None:
    result = phx.ml.fit(phx.ml.linear.RidgeRecipe(1e-3), _features(), _targets())

    assert set(_roles(result.model).values()) == {ArrayRole.FIXED}
    roles = _roles(result.as_trainable())
    assert roles[".coefficients"] is ArrayRole.PARAMETER
    assert roles[".intercept"] is ArrayRole.PARAMETER
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
    model = (
        DecisionTreeRegressor(max_depth=2)
        .fit_batch(MLBatch(_features(), _targets()))
        .as_trainable()
    )
    roles = _roles(model)

    assert roles[".threshold"] is ArrayRole.FIXED
    assert roles[".leaf_value"] is ArrayRole.PARAMETER
    model = StandardScaler().fit_batch(MLBatch(_features())).as_trainable()

    _roles(model)
    assert _parameter_leaves(model) == []


def test_projector_response_carriers_stay_fixed_during_current_basis_update() -> None:
    samples = jnp.array(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    model = phx.ml.fit(PCA(2, differentiate="projector"), samples).as_trainable(
        SubspaceModel
    )
    parameters, model_state, fixed = partition_parameters(model)
    roles = _roles(model)
    assert roles[".weighted_components"] is ArrayRole.PARAMETER
    assert roles[".frame_correction"] is ArrayRole.FIXED
    assert roles[".covariance_correction"] is ArrayRole.FIXED
    assert jnp.array_equal(model.frame_correction, jnp.zeros_like(model.frame_correction))
    assert jnp.array_equal(
        model.covariance_correction, jnp.zeros_like(model.covariance_correction)
    )
    updated_parameters = jax.tree_util.tree_map(lambda value: value + 0.07, parameters)
    updated = eqx.combine(updated_parameters, model_state, fixed)
    assert jnp.array_equal(updated.frame_correction, model.frame_correction)
    assert jnp.array_equal(updated.covariance_correction, model.covariance_correction)
    probe = jnp.array([[0.7, -0.3, 1.1], [-0.4, 0.9, 0.2]])
    cotangent = jnp.array([[1.2, 0.2, -0.4], [0.6, -0.7, 0.8]])
    assert jnp.allclose(
        updated.project(probe),
        updated.inverse_transform(updated.transform(probe)),
        rtol=1e-12,
        atol=1e-12,
    )
    gradient = eqx.filter_grad(
        lambda current: jnp.sum(
            eqx.combine(current, model_state, fixed).project(probe) * cotangent
        )
    )(updated_parameters)
    direction = jnp.array([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]])
    step = 1e-5
    plus = eqx.tree_at(
        lambda current: current.weighted_components,
        updated,
        updated.weighted_components + step * direction,
    )
    minus = eqx.tree_at(
        lambda current: current.weighted_components,
        updated,
        updated.weighted_components - step * direction,
    )
    finite_difference = jnp.sum((plus.project(probe) - minus.project(probe)) * cotangent)
    finite_difference = finite_difference / (2 * step)
    derivative = jnp.sum(gradient.weighted_components * direction)
    assert jnp.abs(derivative) > 1e-3
    assert jnp.allclose(derivative, finite_difference, rtol=1e-7, atol=1e-9)
