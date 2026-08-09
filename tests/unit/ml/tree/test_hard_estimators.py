#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import (
    FeatureSchema,
    ML_CAPACITY_EXHAUSTED,
    ML_INSUFFICIENT_DATA,
    MLBatch,
    SparseFeatures,
    TargetSchema,
)
from phydrax.ml.tree import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    convergence_diagnostics,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RandomTreeClassifier,
    RandomTreeRegressor,
    TreeEnsemble,
    XGBoostClassifier,
    XGBoostRanker,
    XGBoostRegressor,
)


def _regression_batch():
    x0 = jnp.linspace(-3.0, 3.0, 10)
    features = jnp.stack((x0, jnp.square(x0) - 2.0), axis=-1)
    targets = 1.5 * x0 - 0.25
    return MLBatch(
        features,
        targets,
        feature_schema=FeatureSchema(("position", "curvature")),
        target_schema=TargetSchema("continuous", names=("response",)),
    )


def _classification_batch(classes=2):
    x0 = jnp.linspace(-3.0, 3.0, 12)
    features = jnp.stack((x0, jnp.sin(x0)), axis=-1)
    if classes == 2:
        targets = (x0 > 0.0).astype(jnp.int32)
        schema = TargetSchema("binary", class_labels=("left", "right"))
    else:
        targets = jnp.where(x0 < -1.0, 0, jnp.where(x0 > 1.0, 2, 1))
        schema = TargetSchema("multiclass", class_labels=("low", "middle", "high"))
    return MLBatch(
        features,
        targets,
        feature_schema=FeatureSchema(("position", "wave")),
        target_schema=schema,
    )


def test_recipe_configuration_is_immutable_while_fit_returns_a_frozen_executable_model():
    recipe = DecisionTreeRegressor(max_depth=1)
    result = recipe.fit_batch(_regression_batch())
    new_rows = jnp.array([[-5.0, 23.0], [5.0, 23.0]])

    with pytest.raises(AttributeError):
        recipe.max_depth = 9
    with pytest.raises(AttributeError):
        result.model.base_score = jnp.array([999.0])
    assert result.model(new_rows).shape == (2,)
    assert jnp.all(jnp.isfinite(result.model(new_rows)))


_REGRESSORS = (
    pytest.param(lambda: DecisionTreeRegressor(max_depth=2), False, id="cart"),
    pytest.param(lambda: RandomTreeRegressor(max_depth=2), True, id="random-tree"),
    pytest.param(lambda: ExtraTreeRegressor(max_depth=2), True, id="extra-tree"),
    pytest.param(
        lambda: RandomForestRegressor(n_estimators=3, max_depth=2),
        True,
        id="random-forest",
    ),
    pytest.param(
        lambda: ExtraTreesRegressor(n_estimators=3, max_depth=2),
        True,
        id="extra-trees",
    ),
    pytest.param(
        lambda: AdaBoostRegressor(n_estimators=3, max_depth=1), False, id="adaboost"
    ),
    pytest.param(
        lambda: GradientBoostingRegressor(n_estimators=3, max_depth=1),
        False,
        id="gradient-boosting",
    ),
    pytest.param(
        lambda: HistGradientBoostingRegressor(n_estimators=3, max_depth=1, max_bins=3),
        False,
        id="hist-gradient-boosting",
    ),
    pytest.param(
        lambda: XGBoostRegressor(n_estimators=3, max_depth=1, min_child_weight=0.0),
        False,
        id="second-order-boosting",
    ),
)


@pytest.mark.parametrize("factory,requires_key", _REGRESSORS)
def test_every_hard_regression_family_fits_frozen_executable_ensembles(
    factory, requires_key
):
    batch = _regression_batch()
    recipe = factory()
    if requires_key:
        with pytest.raises(ValueError, match="explicit JAX key"):
            recipe.fit_batch(batch)
    result = recipe.fit_batch(batch, key=jax.random.key(23))
    repeated = recipe.fit_batch(batch, key=jax.random.key(23))
    model = result.as_trainable()
    probes = jnp.array([[-4.0, 14.0], [-0.5, -1.75], [4.0, 14.0]])

    assert isinstance(model, TreeEnsemble)
    assert jnp.array_equal(model(probes), repeated.as_trainable()(probes))
    assert model(probes).shape == (3,)
    assert jnp.all(jnp.isfinite(model(probes)))
    assert bool(result.valid)
    assert bool(result.diagnostics.converged)
    assert result.gradient_contract.prediction_inputs == "none"
    assert result.gradient_contract.prediction_parameters == "almost-everywhere"
    assert result.gradient_contract.fit_mode == "stopped"
    assert result.gradient_contract.fit_features == "none"
    assert result.gradient_contract.fit_targets == "none"
    assert result.gradient_contract.fit_weights == "none"
    assert result.gradient_contract.fit_hyperparameters == "none"


_CLASSIFIERS = (
    pytest.param(lambda: DecisionTreeClassifier(max_depth=2), False, id="cart"),
    pytest.param(lambda: RandomTreeClassifier(max_depth=2), True, id="random-tree"),
    pytest.param(lambda: ExtraTreeClassifier(max_depth=2), True, id="extra-tree"),
    pytest.param(
        lambda: RandomForestClassifier(n_estimators=3, max_depth=2),
        True,
        id="random-forest",
    ),
    pytest.param(
        lambda: ExtraTreesClassifier(n_estimators=3, max_depth=2),
        True,
        id="extra-trees",
    ),
    pytest.param(lambda: AdaBoostClassifier(n_estimators=3), False, id="adaboost"),
    pytest.param(
        lambda: GradientBoostingClassifier(n_estimators=3, max_depth=1),
        False,
        id="gradient-boosting",
    ),
    pytest.param(
        lambda: HistGradientBoostingClassifier(n_estimators=3, max_depth=1, max_bins=3),
        False,
        id="hist-gradient-boosting",
    ),
    pytest.param(
        lambda: XGBoostClassifier(n_estimators=3, max_depth=1, min_child_weight=0.0),
        False,
        id="second-order-boosting",
    ),
)


@pytest.mark.parametrize("factory,requires_key", _CLASSIFIERS)
def test_every_hard_classifier_family_exposes_probabilities_labels_and_diagnostics(
    factory, requires_key
):
    batch = _classification_batch()
    recipe = factory()
    if requires_key:
        with pytest.raises(ValueError, match="explicit JAX key"):
            recipe.fit_batch(batch)
    result = recipe.fit_batch(batch, key=jax.random.key(9))
    model = result.as_trainable()
    probabilities = model(batch.features)
    labels = model.predict_labels(batch.features)

    assert probabilities.shape in {(12, 2), (12,)}
    assert labels.shape == (12,)
    assert jnp.all((labels == 0) | (labels == 1))
    assert result.diagnostics.trees_built == model.tree_capacity
    assert result.diagnostics.nodes_used >= result.diagnostics.leaves_used
    assert (
        convergence_diagnostics(result.diagnostics).iterations
        == result.diagnostics.iterations
    )


def test_auto_boosting_objective_selects_binary_logistic_and_multiclass_softmax():
    for recipe_type in (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        XGBoostClassifier,
    ):
        binary = (
            recipe_type(n_estimators=2, max_depth=1, min_child_weight=0.0)
            .fit_batch(_classification_batch(2))
            .as_trainable()
        )
        multiclass = (
            recipe_type(n_estimators=2, max_depth=1, min_child_weight=0.0)
            .fit_batch(_classification_batch(3))
            .as_trainable()
        )
        assert binary.objective_transform == "sigmoid"
        assert binary(_classification_batch(2).features).shape == (12,)
        assert multiclass.objective_transform == "softmax"
        assert multiclass(_classification_batch(3).features).shape == (12, 3)
        assert jnp.allclose(
            jnp.sum(multiclass(_classification_batch(3).features), axis=-1), 1.0
        )
        assert multiclass.predict_labels(_classification_batch(3).features).shape == (12,)


def test_classical_histogram_second_order_and_adaboost_keep_distinct_semantics():
    batch = _regression_batch()
    exact = GradientBoostingRegressor(n_estimators=3, max_depth=1).fit_batch(batch)
    histogram = HistGradientBoostingRegressor(
        n_estimators=3, max_depth=1, max_bins=2
    ).fit_batch(batch)
    second_order = XGBoostRegressor(
        n_estimators=3, max_depth=1, min_child_weight=0.0
    ).fit_batch(batch)
    adaptive = AdaBoostRegressor(n_estimators=3, max_depth=1).fit_batch(batch)
    poisson_batch = MLBatch(batch.features, jnp.abs(batch.targets) + 0.5)
    poisson = XGBoostRegressor(
        objective="poisson", n_estimators=3, max_depth=1, min_child_weight=0.0
    ).fit_batch(poisson_batch)

    assert exact.diagnostics.method == "gradient_boosting_regressor"
    assert exact.diagnostics.split_search == "exact"
    assert histogram.diagnostics.method == "hist_gradient_boosting_regressor"
    assert histogram.diagnostics.split_search == "histogram"
    assert second_order.diagnostics.method == "xgboost_regressor"
    poisson_model = poisson.as_trainable()
    adaptive_model = adaptive.as_trainable()
    assert poisson_model.objective_transform == "exponential"
    assert jnp.all(poisson_model(poisson_batch.features) > 0.0)
    assert adaptive_model.aggregation == "weighted_median"
    assert jnp.all(adaptive_model.tree_weight >= 0.0)


def test_cart_leaf_values_use_statistical_sample_weights():
    features = jnp.array([[0.0], [1.0]])
    targets = jnp.array([0.0, 10.0])
    weighted = (
        DecisionTreeRegressor(max_depth=0)
        .fit_batch(MLBatch(features, targets, sample_weight=jnp.array([9.0, 1.0])))
        .model
    )
    unweighted = (
        DecisionTreeRegressor(max_depth=0).fit_batch(MLBatch(features, targets)).model
    )

    assert jnp.allclose(weighted(features), 1.0)
    assert jnp.allclose(unweighted(features), 5.0)
    assert not jnp.allclose(weighted(features), unweighted(features))


def test_masks_statistical_weights_measure_policy_target_and_case_axes_are_preserved():
    x = jnp.stack(
        (
            jnp.stack((jnp.arange(8.0), jnp.arange(8.0) ** 2), axis=-1),
            jnp.stack((jnp.arange(8.0), -jnp.arange(8.0)), axis=-1),
        )
    )
    y = jnp.stack(
        (
            jnp.stack((x[0, :, 0], -x[0, :, 0]), axis=-1),
            jnp.stack((2.0 * x[1, :, 0], x[1, :, 0] + 1.0), axis=-1),
        )
    )
    feature_mask = jnp.ones_like(x, dtype=bool).at[:, 3, 1].set(False)
    target_mask = jnp.ones_like(y, dtype=bool).at[:, 2, 1].set(False)
    batch = MLBatch(
        x,
        y,
        feature_mask=feature_mask,
        target_mask=target_mask,
        sample_mask=jnp.array([True, True, True, True, True, True, True, False]),
        sample_weight=jnp.array([1.0, 2.0, 1.0, 1.0, 0.0, 3.0, 1.0, 100.0]),
        measure_weight=jnp.array([100.0, 1.0, 9.0, 2.0, 7.0, 1.0, 3.0, 1.0]),
        feature_schema=FeatureSchema(("x", "aux")),
        target_schema=TargetSchema("continuous", names=("first", "second")),
    )
    changed_measure = MLBatch(
        x,
        y,
        feature_mask=feature_mask,
        target_mask=target_mask,
        sample_mask=batch.sample_mask,
        sample_weight=batch.sample_weight,
        measure_weight=jnp.ones((8,)),
        feature_schema=batch.feature_schema,
        target_schema=batch.target_schema,
    )
    first = DecisionTreeRegressor(max_depth=2).fit_batch(batch)
    second = DecisionTreeRegressor(max_depth=2).fit_batch(changed_measure)

    assert first.valid.shape == (2,)
    assert first.diagnostics.effective_samples.tolist() == [5, 5]
    assert first.model(x).shape == (2, 8, 2)
    assert jnp.array_equal(first.model(x), second.model(x))


def test_sparse_training_and_inference_require_explicit_dense_conversion():
    batch = _regression_batch()
    columns = jnp.broadcast_to(jnp.arange(batch.feature_count), batch.features.shape)
    sparse = SparseFeatures(batch.features, columns, feature_count=batch.feature_count)
    sparse_batch = MLBatch(
        sparse,
        batch.targets,
        feature_schema=batch.feature_schema,
        target_schema=batch.target_schema,
    )

    with pytest.raises(TypeError):
        DecisionTreeRegressor(max_depth=2).fit_batch(sparse_batch)

    dense_result = DecisionTreeRegressor(max_depth=2).fit_batch(batch)
    explicitly_dense = DecisionTreeRegressor(max_depth=2).fit_batch(
        MLBatch(
            sparse.to_dense(),
            batch.targets,
            feature_schema=batch.feature_schema,
            target_schema=batch.target_schema,
        )
    )
    assert jnp.array_equal(
        dense_result.model(batch.features),
        explicitly_dense.model(batch.features),
    )
    with pytest.raises(TypeError):
        explicitly_dense.model(sparse)


def test_categorical_fit_records_membership_and_routes_unseen_and_missing_rows_out_of_sample():
    features = jnp.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]])
    targets = jnp.array([5.0, 5.0, -2.0, -2.0, -2.0, -2.0])
    result = DecisionTreeRegressor(max_depth=1).fit_batch(
        MLBatch(
            features,
            targets,
            feature_schema=FeatureSchema(("category",), kinds=("categorical",)),
        )
    )
    model = result.as_trainable()
    root_category = model.category_values[0, 0][model.category_mask[0, 0]][0]
    member = model(jnp.array([[root_category]]))[0]
    nonmember = model(jnp.array([[99.0]]))[0]
    missing = model(jnp.array([[jnp.nan]]))[0]

    assert model.split_kind[0, 0] == 1
    assert member != nonmember
    expected_missing = member if model.default_left[0, 0] else nonmember
    assert missing == expected_missing


def test_exact_split_ties_are_deterministic_and_choose_the_first_feature():
    feature = jnp.arange(8.0)
    batch = MLBatch(jnp.stack((feature, feature), axis=-1), (feature > 3).astype(float))
    first = DecisionTreeRegressor(max_depth=1).fit_batch(batch)
    second = DecisionTreeRegressor(max_depth=1).fit_batch(batch)
    first_model = first.as_trainable()
    second_model = second.as_trainable()
    assert first_model.feature_index[0, 0] == 0
    assert jnp.array_equal(first_model.threshold, second_model.threshold)
    assert jnp.array_equal(first_model(batch.features), second_model(batch.features))


def test_monotonic_interaction_constraints_and_pairwise_ranking_are_enforced():
    batch = _regression_batch()
    constrained = XGBoostRegressor(
        n_estimators=4,
        max_depth=2,
        min_child_weight=0.0,
        monotonic_constraints=(1, 0),
        interaction_constraints=((0,), (1,)),
    ).fit_batch(batch)
    ordered = jnp.stack((jnp.linspace(-4.0, 4.0, 25), jnp.zeros((25,))), axis=-1)
    assert jnp.all(jnp.diff(constrained.model(ordered)) >= -1e-6)

    ranking_batch = MLBatch(
        jnp.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]]),
        jnp.array([0.0, 2.0, 1.0, 0.0, 1.0, 3.0]),
        groups=jnp.array([0, 0, 0, 1, 1, 1]),
        target_schema=TargetSchema("ranking"),
    )
    ranking = XGBoostRanker(n_estimators=3, max_depth=1, min_child_weight=0.0).fit_batch(
        ranking_batch
    )
    ranking_model = ranking.as_trainable()
    assert ranking_model(ranking_batch.features).shape == (6,)
    assert ranking_model.target_schema.kind == "ranking"
    isolated_groups = MLBatch(
        ranking_batch.features,
        ranking_batch.targets,
        groups=jnp.arange(6),
        target_schema=ranking_batch.target_schema,
    )
    isolated = XGBoostRanker(n_estimators=3, max_depth=1, min_child_weight=0.0).fit_batch(
        isolated_groups
    )
    assert not jnp.allclose(
        ranking_model(ranking_batch.features),
        isolated.as_trainable()(isolated_groups.features),
    )
    with pytest.raises(ValueError, match="requires batch groups"):
        XGBoostRanker(n_estimators=1).fit_batch(
            MLBatch(ranking_batch.features, ranking_batch.targets)
        )
    with pytest.raises(ValueError, match="align with the feature axis"):
        DecisionTreeRegressor(monotonic_constraints=(1,)).fit_batch(batch)
    with pytest.raises(ValueError, match="out-of-range"):
        DecisionTreeRegressor(interaction_constraints=((0, 7),)).fit_batch(batch)


def test_capacity_exhaustion_and_insufficient_weight_return_structured_failure_values():
    exhausted = DecisionTreeRegressor(max_depth=2, max_nodes=2).fit_batch(
        _regression_batch()
    )
    assert not bool(exhausted.valid)
    assert exhausted.status == ML_CAPACITY_EXHAUSTED
    assert bool(exhausted.diagnostics.capacity_exhausted)
    assert not bool(exhausted.diagnostics.converged)
    assert bool(exhausted.as_trainable().structure_diagnostics().capacity_exhausted)

    batch = _regression_batch()
    empty = DecisionTreeRegressor(max_depth=2).fit_batch(
        MLBatch(batch.features, batch.targets, sample_weight=jnp.zeros((10,)))
    )
    assert not bool(empty.valid)
    assert empty.status == ML_INSUFFICIENT_DATA
    assert empty.diagnostics.effective_samples == 0
    assert not bool(empty.diagnostics.converged)


def test_invalid_weights_complex_features_and_recipe_parameters_fail_explicitly():
    batch = _regression_batch()
    with pytest.raises(ValueError, match="nonnegative sample weights"):
        DecisionTreeRegressor().fit_batch(
            MLBatch(
                batch.features, batch.targets, sample_weight=jnp.array([1.0] * 9 + [-1.0])
            )
        )
    with pytest.raises(TypeError, match="complex"):
        DecisionTreeRegressor().fit_batch(
            MLBatch(batch.features.astype(complex) + 1j, batch.targets)
        )
    with pytest.raises(ValueError, match="Monotonic constraints"):
        DecisionTreeRegressor(monotonic_constraints=(2, 0))
    with pytest.raises(ValueError, match="non-empty sets"):
        DecisionTreeRegressor(interaction_constraints=((0, 0),))
    with pytest.raises(ValueError, match="nonnegative targets"):
        XGBoostRegressor(objective="poisson", n_estimators=1).fit_batch(
            MLBatch(batch.features, -jnp.ones((10,)))
        )
