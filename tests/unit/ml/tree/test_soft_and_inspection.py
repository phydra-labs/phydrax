#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import FeatureSchema, ML_INSUFFICIENT_DATA, MLBatch, TargetSchema
from phydrax.ml.tree import (
    capacity_diagnostics,
    convergence_diagnostics,
    DecisionTreeRegressor,
    export_tree,
    feature_importance,
    FeatureImportance,
    GradientAttribution,
    partial_dependence,
    PartialDependenceResult,
    soft_tree_gradient_attribution,
    SoftDecisionTree,
    SoftDecisionTreeRecipe,
    SoftGradientBoostedTrees,
    SoftGradientBoostedTreesRecipe,
    SoftRandomForest,
    SoftRandomForestRecipe,
    tree_shap,
    TreeConvergenceDiagnostics,
    TreeEnsemble,
    TreeExport,
    TreeFitDiagnostics,
    TreeSHAPExplanation,
    TreeStructureDiagnostics,
)


def _soft_model(model_type=SoftDecisionTree, *, tree_count=1, temperature=0.2):
    logits = jnp.broadcast_to(jnp.array([[[8.0, -8.0]]]), (tree_count, 1, 2))
    thresholds = jnp.broadcast_to(jnp.array([[[0.0, 20.0]]]), logits.shape)
    leaves = jnp.stack(
        [jnp.array([[-2.0 - index], [2.0 + index]]) for index in range(tree_count)]
    )
    weights = (
        jnp.full((tree_count,), 1.0 / tree_count)
        if model_type is SoftRandomForest
        else jnp.full(
            (tree_count,), 0.1 if model_type is SoftGradientBoostedTrees else 1.0
        )
    )
    return model_type(
        feature_logits=logits,
        threshold=thresholds,
        missing_left_logit=jnp.full((tree_count, 1), 12.0),
        leaf_value=leaves,
        tree_weight=weights,
        base_score=jnp.array([0.5]),
        temperature=temperature,
        feature_schema=FeatureSchema(("signal", "noise")),
        target_schema=TargetSchema("continuous", names=("response",)),
        out_size="scalar",
    )


def _hard_inspection_tree():
    return TreeEnsemble(
        feature_index=jnp.array([[0, -1, -1]]),
        threshold=jnp.array([[0.0, 0.0, 0.0]]),
        left_child=jnp.array([[1, -1, -1]]),
        right_child=jnp.array([[2, -1, -1]]),
        default_left=jnp.array([[True, False, False]]),
        leaf_value=jnp.array([[[0.0], [2.0], [6.0]]]),
        node_mask=jnp.ones((1, 3), dtype=bool),
        leaf_mask=jnp.array([[False, True, True]]),
        tree_mask=jnp.array([True]),
        tree_weight=jnp.array([1.0]),
        node_gain=jnp.array([[4.0, 0.0, 0.0]]),
        node_cover=jnp.array([[6.0, 3.0, 3.0]]),
        base_score=jnp.array([1.0]),
        feature_schema=FeatureSchema(("signal", "unused")),
        target_schema=TargetSchema("continuous", names=("response",)),
        max_steps=2,
    )


def _soft_batch(case=False):
    features = jnp.array(
        [
            [-2.0, 0.3],
            [-1.1, -0.2],
            [-0.2, 0.7],
            [0.4, -0.8],
            [1.3, 0.1],
            [2.2, 0.5],
        ]
    )
    targets = jnp.array([-2.3, -1.4, -0.2, 0.7, 1.8, 3.1])
    if case:
        features = jnp.stack((features, features.at[:, 0].multiply(-1.0)))
        targets = jnp.stack((targets, -targets))
    return MLBatch(
        features,
        targets,
        feature_schema=FeatureSchema(("signal", "noise")),
        target_schema=TargetSchema("continuous", names=("response",)),
    )


@pytest.mark.parametrize(
    "model_type,tree_count",
    (
        pytest.param(SoftDecisionTree, 1, id="soft-tree"),
        pytest.param(SoftRandomForest, 3, id="soft-forest"),
        pytest.param(SoftGradientBoostedTrees, 3, id="soft-boosted"),
    ),
)
def test_every_soft_model_family_is_jittable_vmappable_and_smooth(model_type, tree_count):
    model = _soft_model(model_type, tree_count=tree_count)
    points = jnp.array([[-2.0, 3.0], [-0.5, -4.0], [1.0, 9.0]])
    predictions = model(points)

    assert predictions.shape == (3,)
    assert model.predict_trees(points).shape == (3, tree_count)
    assert jnp.allclose(jax.jit(model)(points), predictions)
    assert jnp.allclose(jax.vmap(model)(points), predictions)
    input_gradient = jax.grad(lambda value: jnp.sum(model(value)))(points)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.any(jnp.abs(input_gradient[..., 0]) > 0.0)
    assert jnp.allclose(jnp.sum(model.feature_probabilities(), axis=-1), 1.0)
    assert jnp.all(model.feature_probabilities()[..., 0] > 0.99)
    hardened = model.harden()
    assert isinstance(hardened, TreeEnsemble)
    assert hardened.tree_capacity == tree_count
    assert hardened(points).shape == predictions.shape


def test_soft_temperature_missing_routing_and_hardening_are_explicit_relaxations():
    soft = _soft_model(temperature=0.02)
    hard = soft.harden()
    points = jnp.array([[-2.0, 100.0], [2.0, -100.0], [jnp.nan, 7.0]])

    assert isinstance(hard, TreeEnsemble)
    assert jnp.allclose(soft(points), hard(points), atol=2e-3)
    assert hard.feature_index[0, 0] == 0
    assert bool(hard.default_left[0, 0])
    assert hard(jnp.array([[jnp.nan, -999.0]]))[0] == hard(jnp.array([[-2.0, 999.0]]))[0]

    def hardened_prediction(left_leaf):
        model = SoftDecisionTree(
            feature_logits=soft.feature_logits,
            threshold=soft.threshold,
            missing_left_logit=soft.missing_left_logit,
            leaf_value=soft.leaf_value.at[0, 0, 0].set(left_leaf),
            tree_weight=soft.tree_weight,
            base_score=soft.base_score,
            temperature=soft.temperature,
            feature_schema=soft.feature_schema,
        )
        return model.harden()(jnp.array([-2.0, 0.0]))

    assert jax.grad(hardened_prediction)(jnp.array(-2.0)) == 0.0


def test_soft_model_parameter_and_input_prediction_gradients_match_declared_smoothness():
    model = _soft_model()
    point = jnp.array([0.3, -1.0])
    input_gradient = jax.grad(model)(point)

    def parameterized_prediction(threshold, leaf):
        changed = SoftDecisionTree(
            feature_logits=model.feature_logits,
            threshold=model.threshold.at[0, 0, 0].set(threshold),
            missing_left_logit=model.missing_left_logit,
            leaf_value=model.leaf_value.at[0, 1, 0].set(leaf),
            tree_weight=model.tree_weight,
            base_score=model.base_score,
            temperature=model.temperature,
            feature_schema=model.feature_schema,
        )
        return changed(point)

    parameter_gradient = jax.grad(parameterized_prediction, argnums=(0, 1))(0.0, 2.0)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.all(jnp.isfinite(jnp.asarray(parameter_gradient)))
    assert jnp.any(jnp.abs(jnp.asarray(parameter_gradient)) > 0.0)


_SOFT_RECIPES = (
    pytest.param(
        lambda: SoftDecisionTreeRecipe(depth=1, iterations=4), SoftDecisionTree, id="tree"
    ),
    pytest.param(
        lambda: SoftRandomForestRecipe(n_estimators=2, depth=1, iterations=4),
        SoftRandomForest,
        id="forest",
    ),
    pytest.param(
        lambda: SoftGradientBoostedTreesRecipe(n_estimators=2, depth=1, iterations=4),
        SoftGradientBoostedTrees,
        id="boosted",
    ),
)


@pytest.mark.parametrize("factory,model_type", _SOFT_RECIPES)
def test_every_soft_recipe_requires_keys_is_deterministic_and_declares_unrolled_gradients(
    factory, model_type
):
    recipe = factory()
    batch = _soft_batch(case=True)
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(batch)
    first = recipe.fit_batch(batch, key=jax.random.key(11))
    second = recipe.fit_batch(batch, key=jax.random.key(11))

    first_model = first.as_trainable()
    second_model = second.as_trainable()
    assert isinstance(first_model, model_type)
    assert isinstance(first.diagnostics, TreeFitDiagnostics)
    assert first.valid.shape == (2,)
    assert first_model(batch.features).shape == (2, 6)
    assert jnp.array_equal(first_model(batch.features), second_model(batch.features))
    assert first.gradient_contract.prediction_inputs == "smooth"
    assert first.gradient_contract.prediction_parameters == "smooth"
    assert first.gradient_contract.fit_features == "conditional"
    assert first.gradient_contract.fit_targets == "conditional"
    assert first.gradient_contract.fit_weights == "conditional"
    assert first.gradient_contract.fit_hyperparameters == "conditional"
    assert first.gradient_contract.fit_mode == "unrolled"
    assert "hardened structure" in first.gradient_contract.nondifferentiable_outputs
    diagnostics = convergence_diagnostics(first.diagnostics)
    assert isinstance(diagnostics, TreeConvergenceDiagnostics)
    assert jnp.all(diagnostics.valid)
    assert jnp.all(diagnostics.converged)
    assert jnp.all(diagnostics.iterations == 4)


def test_soft_fit_feature_target_weight_and_hyperparameter_gradients_are_finite():
    batch = _soft_batch()
    features = batch.features
    targets = batch.targets
    weights = jnp.array([0.7, 1.1, 0.9, 1.3, 0.8, 1.2])
    probe = jnp.array([0.25, -0.4])
    key = jax.random.key(5)
    base_recipe = SoftDecisionTreeRecipe(
        depth=1,
        iterations=3,
        learning_rate=0.02,
        initial_temperature=0.8,
        final_temperature=0.4,
        temperature_schedule="linear",
    )

    def fit_prediction(feature_values, target_values, sample_weights, learning_rate):
        recipe = eqx.tree_at(
            lambda candidate: candidate.learning_rate,
            base_recipe,
            learning_rate,
        )
        result = recipe.fit_batch(
            MLBatch(feature_values, target_values, sample_weight=sample_weights),
            key=key,
        )
        return result.as_trainable()(probe)

    gradients = jax.grad(fit_prediction, argnums=(0, 1, 2, 3))(
        features, targets, weights, jnp.array(0.02)
    )
    for gradient in gradients:
        assert jnp.all(jnp.isfinite(gradient))
    assert all(bool(jnp.any(jnp.abs(gradient) > 0.0)) for gradient in gradients)


def test_soft_fit_respects_masks_zero_statistical_weight_and_ignores_measure_weight():
    batch = _soft_batch()
    feature_mask = jnp.ones_like(batch.features, dtype=bool).at[5, 1].set(False)
    common = dict(
        feature_mask=feature_mask,
        sample_mask=jnp.array([True, True, True, True, True, False]),
        sample_weight=jnp.array([1.0, 2.0, 0.0, 1.0, 3.0, 100.0]),
    )
    first = MLBatch(
        batch.features,
        batch.targets,
        measure_weight=jnp.arange(1.0, 7.0),
        **common,
    )
    changed_features = batch.features.at[5].set(jnp.array([999.0, -999.0]))
    changed_targets = batch.targets.at[5].set(-10000.0)
    second = MLBatch(
        changed_features,
        changed_targets,
        measure_weight=jnp.ones((6,)),
        **common,
    )
    recipe = SoftDecisionTreeRecipe(depth=1, iterations=4)
    first_result = recipe.fit_batch(first, key=jax.random.key(17))
    second_result = recipe.fit_batch(second, key=jax.random.key(17))

    assert first_result.diagnostics.effective_samples == 4
    assert jnp.allclose(
        first_result.model(batch.features[:4]), second_result.model(batch.features[:4])
    )


def test_soft_classification_objectives_labels_and_failure_status_are_observable():
    binary_batch = MLBatch(
        jnp.linspace(-2.0, 2.0, 8)[:, None],
        jnp.array([0, 0, 0, 0, 1, 1, 1, 1]),
        target_schema=TargetSchema("binary", class_labels=(0, 1)),
    )
    logistic = SoftDecisionTreeRecipe(
        depth=1, iterations=4, objective="logistic", num_classes=2
    ).fit_batch(binary_batch, key=jax.random.key(2))
    logistic_model = logistic.as_trainable()
    assert logistic_model.objective_transform == "sigmoid"
    assert logistic_model.predict_labels(binary_batch.features).shape == (8,)

    multiclass_batch = MLBatch(
        jnp.linspace(-3.0, 3.0, 9)[:, None],
        jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        target_schema=TargetSchema("multiclass", class_labels=(0, 1, 2)),
    )
    softmax = SoftDecisionTreeRecipe(
        depth=1, iterations=4, objective="softmax", num_classes=3
    ).fit_batch(multiclass_batch, key=jax.random.key(3))
    softmax_model = softmax.as_trainable()
    probabilities = softmax_model(multiclass_batch.features)
    assert probabilities.shape == (9, 3)
    assert jnp.allclose(jnp.sum(probabilities, axis=-1), 1.0)
    assert softmax_model.predict_labels(multiclass_batch.features).shape == (9,)

    empty = SoftDecisionTreeRecipe(depth=1, iterations=2).fit_batch(
        MLBatch(
            binary_batch.features,
            binary_batch.targets.astype(float),
            sample_weight=jnp.zeros((8,)),
        ),
        key=jax.random.key(4),
    )
    assert not bool(empty.valid)
    assert empty.status == ML_INSUFFICIENT_DATA
    assert not bool(empty.diagnostics.converged)


def test_soft_temperatures_categorical_and_complex_feature_paths_fail_explicitly():
    with pytest.raises(ValueError, match="finite and positive"):
        _soft_model(temperature=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        SoftDecisionTreeRecipe(initial_temperature=1.0, final_temperature=0.0)
    with pytest.raises(ValueError, match="constant schedule"):
        SoftDecisionTreeRecipe(
            initial_temperature=1.0,
            final_temperature=0.5,
            temperature_schedule="constant",
        )
    categorical = MLBatch(
        jnp.array([[0.0], [1.0]]),
        jnp.array([0.0, 1.0]),
        feature_schema=FeatureSchema(("category",), kinds=("categorical",)),
    )
    with pytest.raises(ValueError, match="categorical relaxation"):
        SoftDecisionTreeRecipe(iterations=2).fit_batch(categorical, key=jax.random.key(0))
    with pytest.raises(TypeError, match="real features"):
        _soft_model()(jnp.array([[1.0 + 1.0j, 0.0]]))


def test_feature_importance_export_and_capacity_diagnostics_report_stored_structure():
    model = _hard_inspection_tree()
    importance = feature_importance(model)
    capacity = capacity_diagnostics(model)
    exported = export_tree(model, 0)

    assert isinstance(importance, FeatureImportance)
    assert isinstance(capacity, TreeStructureDiagnostics)
    assert jnp.array_equal(importance.gain, jnp.array([4.0, 0.0]))
    assert jnp.array_equal(importance.cover, jnp.array([6.0, 0.0]))
    assert jnp.array_equal(importance.frequency, jnp.array([1.0, 0.0]))
    assert jnp.array_equal(importance.normalized_gain, jnp.array([1.0, 0.0]))
    assert capacity.valid
    assert capacity.used_trees == 1
    assert capacity.used_nodes == 3
    assert capacity.used_leaves == 2
    assert isinstance(exported, TreeExport)
    assert exported.feature_index[0] == 0
    assert exported.gain[0] == 4.0
    case_model = (
        DecisionTreeRegressor(max_depth=1)
        .fit_batch(_soft_batch(case=True))
        .as_trainable()
    )
    assert feature_importance(case_model).gain.shape == (2, 2)
    assert export_tree(case_model, 0, case_index=1).leaf_value.shape[-1] == 1
    with pytest.raises(IndexError, match="identify every case axis"):
        export_tree(case_model, 0)
    with pytest.raises(IndexError, match="outside"):
        export_tree(model, 1)
    with pytest.raises(TypeError, match="hard TreeEnsemble"):
        feature_importance(_soft_model())


def test_partial_dependence_preserves_samples_weights_and_rejects_invalid_domains():
    model = _hard_inspection_tree()
    samples = jnp.array([[-3.0, 10.0], [4.0, -7.0], [9.0, 1.0]])
    result = partial_dependence(
        model,
        samples,
        0,
        jnp.array([-1.0, 1.0]),
        sample_weight=jnp.array([1.0, 2.0, 4.0]),
        return_individual=True,
    )

    assert isinstance(result, PartialDependenceResult)
    assert result.feature_index == 0
    assert jnp.array_equal(result.average, jnp.array([3.0, 7.0]))
    assert result.individual.shape == (2, 3)
    soft_result = partial_dependence(_soft_model(), samples, 0, jnp.array([-1.0, 1.0]))
    assert soft_result.average.shape == (2,)
    assert jnp.all(jnp.isfinite(soft_result.average))
    weighted_unused = partial_dependence(
        model,
        samples,
        1,
        jnp.array([0.0]),
        sample_weight=jnp.array([1.0, 2.0, 4.0]),
    )
    assert jnp.allclose(weighted_unused.average, jnp.array([45.0 / 7.0]))
    case_batch = _soft_batch(case=True)
    case_model = DecisionTreeRegressor(max_depth=1).fit_batch(case_batch).as_trainable()
    case_result = partial_dependence(
        case_model, case_batch.features, 0, jnp.array([-1.0, 1.0])
    )
    assert case_result.average.shape == (2, 2)
    with pytest.raises(IndexError, match="out of range"):
        partial_dependence(model, samples, 2, jnp.array([0.0]))
    with pytest.raises(ValueError, match="nonnegative"):
        partial_dependence(
            model,
            samples,
            0,
            jnp.array([0.0]),
            sample_weight=jnp.array([1.0, -1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="positive mass"):
        partial_dependence(
            model,
            samples,
            0,
            jnp.array([0.0]),
            sample_weight=jnp.zeros((3,)),
        )


def test_exact_tree_shap_is_additive_case_independent_and_explicitly_bounded():
    model = _hard_inspection_tree()
    points = jnp.array([[-2.0, 100.0], [3.0, -100.0]])
    baseline = jnp.array([[-1.0, 0.0], [-1.0, 0.0]])
    explanation = tree_shap(model, points, baseline)

    assert isinstance(explanation, TreeSHAPExplanation)
    assert explanation.values.shape == (2, 2)
    assert jnp.array_equal(explanation.values[:, 1], jnp.zeros((2,)))
    assert jnp.allclose(
        explanation.base_values + jnp.sum(explanation.values, axis=-1),
        explanation.predictions,
    )
    assert jnp.array_equal(explanation.predictions, model(points))
    with pytest.raises(ValueError, match="bounded to 1 features"):
        tree_shap(model, points, baseline, max_features=1)
    with pytest.raises(TypeError, match="hard TreeEnsemble"):
        tree_shap(_soft_model(), points, baseline)
    case_model = (
        DecisionTreeRegressor(max_depth=1)
        .fit_batch(_soft_batch(case=True))
        .as_trainable()
    )
    with pytest.raises(ValueError, match="case-independent"):
        tree_shap(
            case_model,
            _soft_batch(case=True).features,
            jnp.zeros_like(_soft_batch(case=True).features),
        )


def test_soft_gradient_attribution_matches_autodiff_and_baseline_displacement():
    model = _soft_model()
    points = jnp.array([[-1.0, 0.5], [0.4, -0.2]])
    baseline = jnp.zeros_like(points)
    explanation = soft_tree_gradient_attribution(model, points, baseline=baseline)
    expected = jax.vmap(jax.grad(model))(points)

    assert isinstance(explanation, GradientAttribution)
    assert jnp.allclose(explanation.gradients, expected)
    assert jnp.allclose(explanation.attributions, expected * points)
    assert jnp.array_equal(explanation.baseline, baseline)
    raw_gradient = soft_tree_gradient_attribution(model, points)
    assert jnp.array_equal(raw_gradient.gradients, raw_gradient.attributions)
    with pytest.raises(TypeError, match="soft-tree model"):
        soft_tree_gradient_attribution(_hard_inspection_tree(), points)
    case_model = (
        SoftDecisionTreeRecipe(depth=1, iterations=2)
        .fit_batch(_soft_batch(case=True), key=jax.random.key(19))
        .as_trainable()
    )
    with pytest.raises(ValueError, match="case-independent"):
        soft_tree_gradient_attribution(case_model, _soft_batch(case=True).features)
