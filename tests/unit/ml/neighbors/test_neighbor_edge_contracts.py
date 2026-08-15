#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax._model import FrozenModel
from phydrax.ml import (
    fit,
    ML_CAPACITY_EXHAUSTED,
    ML_INSUFFICIENT_DATA,
    MLBatch,
    SparseFeatures,
)
from phydrax.ml.neighbors import (
    ExactNeighborRegressorModel,
    KernelDensityRecipe,
    KernelNeighborsClassifierRecipe,
    KernelNeighborsRegressorRecipe,
    KNeighborsClassifierRecipe,
    KNeighborsRegressorRecipe,
    LocalOutlierFactorRecipe,
    MahalanobisMetricRecipe,
    NearestCentroidRecipe,
    NeighborhoodComponentsAnalysisRecipe,
    RadiusNeighborsClassifierRecipe,
    RadiusNeighborsRegressorRecipe,
)


def _cluster_data():
    features = jnp.array(
        [
            [-1.3, -0.7],
            [-0.9, -1.2],
            [-0.5, -0.6],
            [0.6, 0.7],
            [1.0, 1.3],
            [1.4, 0.6],
        ]
    )
    labels = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    targets = 0.6 * features[:, 0] - 0.4 * features[:, 1]
    weights = jnp.array([0.8, 1.1, 0.9, 1.2, 0.7, 1.0])
    return features, labels, targets, weights


def _assert_finite(values):
    if isinstance(values, tuple):
        assert all(jnp.all(jnp.isfinite(value)) for value in values)
    else:
        assert jnp.all(jnp.isfinite(values))


def _assert_prediction_parameter_gradient(model, query):
    gradient = eqx.filter_grad(
        lambda current: jnp.sum(jnp.square(jnp.real(current(query))))
    )(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(gradient, eqx.is_inexact_array))
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_exact_neighbors_select_unmasked_geometry_preserve_target_axes_and_freeze():
    features = jnp.array([[0.0], [2.0], [5.0]])
    targets = jnp.array([[0.0, 10.0], [20.0, 30.0], [50.0, 60.0]])
    recipe = KNeighborsRegressorRecipe(1, metric="euclidean")
    result = fit(
        recipe,
        features,
        targets,
        sample_mask=jnp.array([True, False, True]),
    )
    model = result.as_trainable()
    assert isinstance(model, ExactNeighborRegressorModel)
    query = jnp.array([[[1.9], [4.8]]])
    indices, distances = model.neighbor_indices(query)

    assert isinstance(result.model, FrozenModel)
    assert recipe.neighbor_count == 1
    assert indices.shape == (1, 2, 1)
    assert jnp.array_equal(indices[..., 0], jnp.array([[0, 2]]))
    assert jnp.allclose(distances[..., 0], jnp.array([[1.9, 0.2]]), atol=1e-6)
    assert jnp.allclose(model(query), jnp.array([[[0.0, 10.0], [50.0, 60.0]]]))
    assert result.gradient_contract.prediction_inputs == "almost-everywhere"
    assert result.gradient_contract.prediction_parameters == "almost-everywhere"
    _assert_finite(jax.grad(lambda point: jnp.sum(model(point)))(jnp.array([1.7])))
    _assert_prediction_parameter_gradient(model, jnp.array([[1.7], [4.6]]))


def test_exact_classifier_weights_labels_and_weight_policy_are_observable():
    features = jnp.array([[0.0], [2.0], [5.0]])
    labels = jnp.array([0, 1, 3], dtype=jnp.int32)
    sample_weight = jnp.array([1.0, 3.0, 7.0])
    measure_weight = jnp.array([2.0, 0.5, 11.0])
    result = KNeighborsClassifierRecipe(
        2, class_count=2, weight_policy="product"
    ).fit_batch(
        MLBatch(
            features,
            labels,
            sample_weight=sample_weight,
            measure_weight=measure_weight,
        )
    )
    model = result.as_trainable()

    assert jnp.allclose(model.support_weight, sample_weight * measure_weight)
    assert jnp.array_equal(model.support_mask, jnp.array([True, True, False]))
    assert jnp.allclose(
        model.probabilities(jnp.array([[1.0]])), jnp.array([[4.0 / 7.0, 3.0 / 7.0]])
    )
    assert model.predict(jnp.array([[1.0]]))[0] == 0

    with pytest.raises(ValueError, match="integer label"):
        KNeighborsClassifierRecipe(1, class_count=2).fit_batch(
            MLBatch(features, labels.astype(float))
        )
    with pytest.raises(Exception, match="class capacity"):
        NearestCentroidRecipe(class_count=2).fit_batch(MLBatch(features, labels))


def test_kernel_density_normalization_capacity_and_weight_gradients():
    singleton = KernelDensityRecipe(2.0).fit_batch(
        MLBatch(jnp.array([[0.0]]), measure_weight=jnp.array([3.0]))
    )
    expected = 1.0 / (2.0 * jnp.sqrt(2.0 * jnp.pi))
    assert jnp.allclose(singleton.as_trainable()(jnp.array([[0.0]]))[0], expected)

    features, _, _, weights = _cluster_data()
    query = jnp.array([0.15, -0.1])

    def density_loss(x, measure_weight, bandwidth):
        return (
            KernelDensityRecipe(bandwidth)
            .fit_batch(MLBatch(x, measure_weight=measure_weight))
            .as_trainable()(query)
        )

    gradients = jax.grad(density_loss, argnums=(0, 1, 2))(
        features, weights, jnp.asarray(0.55)
    )
    _assert_finite(gradients)
    result = KernelDensityRecipe(0.55).fit_batch(
        MLBatch(features, measure_weight=weights)
    )
    contract = result.gradient_contract
    assert (
        contract.fit_features,
        contract.fit_weights,
        contract.fit_hyperparameters,
    ) == ("smooth", "smooth", "smooth")
    _assert_prediction_parameter_gradient(result.as_trainable(), query[None, :])

    exhausted = KernelDensityRecipe(0.55, capacity=3).fit_batch(MLBatch(features))
    assert not exhausted.valid
    assert exhausted.status == ML_CAPACITY_EXHAUSTED
    empty = KernelDensityRecipe(0.55).fit_batch(
        MLBatch(features, sample_mask=jnp.zeros((features.shape[0],), dtype=bool))
    )
    assert not empty.valid
    assert empty.status == ML_INSUFFICIENT_DATA


def test_smooth_neighbor_and_centroid_fit_gradients_match_contracts():
    features, labels, targets, weights = _cluster_data()
    query = jnp.array([0.15, 0.05])
    reg_base = KernelNeighborsRegressorRecipe(temperature=0.65)

    def regression_loss(x, y, sample_weight, temperature):
        recipe = eqx.tree_at(lambda current: current.temperature, reg_base, temperature)
        return recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight)
        ).as_trainable()(query)

    reg_gradients = jax.grad(regression_loss, argnums=(0, 1, 2, 3))(
        features, targets, weights, reg_base.temperature
    )
    _assert_finite(reg_gradients)
    reg_result = reg_base.fit_batch(MLBatch(features, targets, sample_weight=weights))
    assert (
        reg_result.gradient_contract.fit_features,
        reg_result.gradient_contract.fit_targets,
        reg_result.gradient_contract.fit_weights,
        reg_result.gradient_contract.fit_hyperparameters,
    ) == ("smooth", "smooth", "smooth", "smooth")
    _assert_prediction_parameter_gradient(reg_result.as_trainable(), query[None, :])

    cls_base = KernelNeighborsClassifierRecipe(class_count=2, temperature=0.65)

    def classification_loss(x, sample_weight, temperature):
        recipe = eqx.tree_at(lambda current: current.temperature, cls_base, temperature)
        probability = recipe.fit_batch(
            MLBatch(x, labels, sample_weight=sample_weight)
        ).as_trainable()(query)
        return probability[1]

    cls_gradients = jax.grad(classification_loss, argnums=(0, 1, 2))(
        features, weights, cls_base.temperature
    )
    _assert_finite(cls_gradients)
    cls_result = cls_base.fit_batch(MLBatch(features, labels, sample_weight=weights))
    assert cls_result.gradient_contract.fit_targets == "none"
    assert cls_result.gradient_contract.fit_features == "smooth"
    assert cls_result.gradient_contract.fit_weights == "smooth"
    assert cls_result.gradient_contract.fit_hyperparameters == "smooth"

    centroid_base = NearestCentroidRecipe(class_count=2, temperature=0.6)

    def centroid_loss(x, sample_weight):
        probability = centroid_base.fit_batch(
            MLBatch(x, labels, sample_weight=sample_weight)
        ).as_trainable()(query)
        return probability[1]

    centroid_gradients = jax.grad(centroid_loss, argnums=(0, 1))(features, weights)
    _assert_finite(centroid_gradients)
    centroid_result = centroid_base.fit_batch(
        MLBatch(features, labels, sample_weight=weights)
    )
    assert centroid_result.gradient_contract.fit_features == "smooth"
    assert centroid_result.gradient_contract.fit_weights == "conditional"
    _assert_prediction_parameter_gradient(centroid_result.as_trainable(), query[None, :])


def test_metric_learning_exercises_declared_fit_and_parameter_gradients():
    features, labels, _, weights = _cluster_data()
    query = jnp.array([[-0.2, 0.1], [0.9, 0.8]])
    nca_base = NeighborhoodComponentsAnalysisRecipe(
        component_count=2,
        iterations=2,
        learning_rate=0.008,
        temperature=0.9,
        ridge=1e-3,
    )

    def nca_loss(x, sample_weight, learning_rate, temperature, ridge):
        recipe = eqx.tree_at(
            lambda current: (
                current.learning_rate,
                current.temperature,
                current.ridge,
            ),
            nca_base,
            (learning_rate, temperature, ridge),
        )
        model = recipe.fit_batch(
            MLBatch(x, labels, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model(query)))

    nca_gradients = jax.grad(nca_loss, argnums=(0, 1, 2, 3, 4))(
        features,
        weights,
        nca_base.learning_rate,
        nca_base.temperature,
        nca_base.ridge,
    )
    _assert_finite(nca_gradients)
    nca_result = nca_base.fit_batch(MLBatch(features, labels, sample_weight=weights))
    assert (
        nca_result.gradient_contract.fit_features,
        nca_result.gradient_contract.fit_targets,
        nca_result.gradient_contract.fit_weights,
        nca_result.gradient_contract.fit_hyperparameters,
    ) == ("smooth", "none", "conditional", "smooth")
    _assert_prediction_parameter_gradient(nca_result.as_trainable(), query)

    metric_base = MahalanobisMetricRecipe(ridge=0.05, component_count=2)

    def metric_loss(x, sample_weight, ridge):
        recipe = eqx.tree_at(lambda current: current.ridge, metric_base, ridge)
        model = recipe.fit_batch(
            MLBatch(x, labels, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model(query)))

    metric_gradients = jax.grad(metric_loss, argnums=(0, 1, 2))(
        features, weights, metric_base.ridge
    )
    _assert_finite(metric_gradients)
    metric_result = metric_base.fit_batch(
        MLBatch(features, labels, sample_weight=weights)
    )
    assert metric_result.gradient_contract.fit_features == "conditional"
    assert metric_result.gradient_contract.fit_weights == "conditional"
    assert metric_result.gradient_contract.fit_hyperparameters == "conditional"
    _assert_prediction_parameter_gradient(metric_result.as_trainable(), query)


def test_sparse_inputs_fail_closed_without_implicit_densification():
    dense = jnp.array([[-1.0, -0.5], [-0.7, -1.1], [0.8, 0.6], [1.2, 1.0]])
    sparse = SparseFeatures(
        dense,
        jnp.broadcast_to(jnp.array([0, 1]), dense.shape),
        feature_count=2,
    )
    labels = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    targets = jnp.array([-0.8, -0.2, 0.5, 1.1])
    recipes_and_targets = (
        (KNeighborsRegressorRecipe(1), targets),
        (KNeighborsClassifierRecipe(1, class_count=2), labels),
        (KernelNeighborsRegressorRecipe(), targets),
        (KernelNeighborsClassifierRecipe(class_count=2), labels),
        (RadiusNeighborsRegressorRecipe(1.0), targets),
        (RadiusNeighborsClassifierRecipe(1.0, class_count=2), labels),
        (NearestCentroidRecipe(class_count=2), labels),
        (NeighborhoodComponentsAnalysisRecipe(iterations=2), labels),
        (MahalanobisMetricRecipe(), labels),
    )
    for recipe, target in recipes_and_targets:
        with pytest.raises(TypeError, match="SparseFeatures|sparse|dense"):
            recipe.fit_batch(MLBatch(sparse, target))
    for recipe in (KernelDensityRecipe(0.7), LocalOutlierFactorRecipe(1)):
        with pytest.raises(TypeError, match="SparseFeatures|sparse|dense"):
            recipe.fit_batch(MLBatch(sparse))


def test_complex_geometry_follows_each_family_contract():
    dense = jnp.array([[-1.0, -0.5], [-0.7, -1.1], [0.8, 0.6], [1.2, 1.0]])
    labels = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    complex_features = dense.astype(jnp.complex64) * (1.0 + 0.4j)
    complex_targets = jnp.array([1.0 + 0.2j, 0.5j, -0.7 + 0.1j, 1.2 - 0.3j])
    complex_neighbor = (
        KNeighborsRegressorRecipe(1)
        .fit_batch(MLBatch(complex_features, complex_targets))
        .as_trainable()
    )
    assert jnp.iscomplexobj(complex_neighbor(complex_features[:2]))
    complex_density = (
        KernelDensityRecipe(0.7).fit_batch(MLBatch(complex_features)).as_trainable()
    )
    assert jnp.all(jnp.isfinite(complex_density(complex_features[:2])))

    with pytest.raises(TypeError, match="real feature"):
        NeighborhoodComponentsAnalysisRecipe(iterations=2).fit_batch(
            MLBatch(complex_features, labels)
        )
    with pytest.raises(TypeError, match="real features"):
        MahalanobisMetricRecipe().fit_batch(MLBatch(complex_features, labels))


def test_hard_neighbor_failures_and_case_query_geometry_are_explicit():
    features, labels, targets, _ = _cluster_data()
    insufficient = KNeighborsRegressorRecipe(3).fit_batch(
        MLBatch(
            features,
            targets,
            sample_mask=jnp.array([True, False, False, True, False, False]),
        )
    )
    assert not insufficient.valid
    assert insufficient.status == ML_INSUFFICIENT_DATA

    lof = LocalOutlierFactorRecipe(2, chunk_size=2).fit_batch(
        MLBatch(
            features,
            sample_mask=jnp.array([True, False, False, True, False, False]),
        )
    )
    assert not lof.valid
    assert lof.status == ML_INSUFFICIENT_DATA
    lof_model = (
        LocalOutlierFactorRecipe(2, chunk_size=2)
        .fit_batch(MLBatch(features))
        .as_trainable()
    )
    query = jnp.array([[0.15, 0.05], [0.9, 0.9]])
    _assert_finite(jax.grad(lambda point: lof_model(point) ** 2)(query[0]))
    _assert_prediction_parameter_gradient(lof_model, query)

    radius = RadiusNeighborsRegressorRecipe(0.8).fit_batch(MLBatch(features, targets))
    assert radius.gradient_contract.prediction_inputs == "almost-everywhere"
    _assert_finite(
        jax.grad(lambda point: jnp.nan_to_num(radius.as_trainable()(point)) ** 2)(
            query[0]
        )
    )

    cases = jnp.stack((features, features + 0.1), axis=0)
    case_targets = jnp.stack((targets, targets + 0.2), axis=0)
    case_model = (
        KNeighborsRegressorRecipe(2)
        .fit_batch(MLBatch(cases, case_targets))
        .as_trainable()
    )
    assert case_model(features[:2]).shape == (2,)
    with pytest.raises(ValueError, match="begin with fitted case shape"):
        case_model(features[:3])
    with pytest.raises(ValueError, match="unbatched fitted case"):
        case_model.predict_chunked(cases[:, :2], chunk_size=1)
