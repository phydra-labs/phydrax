#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import (
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    MLBatch,
    SparseFeatures,
    TargetSchema,
)
from phydrax.ml.multiclass import (
    ClassifierChainModel,
    ClassifierChainRecipe,
    MultilabelModel,
    MultilabelRecipe,
    OneVsOneModel,
    OneVsOneRecipe,
    OneVsRestModel,
    OneVsRestRecipe,
    OutputCodeModel,
    OutputCodeRecipe,
    SmoothClassifierChainModel,
    SmoothClassifierChainRecipe,
)
from phydrax.ml.naive_bayes import GaussianNaiveBayesRecipe


_FEATURES = jnp.array(
    [
        [-2.4, -1.8],
        [-2.0, -2.5],
        [-1.6, -1.7],
        [-2.2, -2.1],
        [1.6, -2.2],
        [2.3, -1.8],
        [1.8, -1.5],
        [2.4, -2.5],
        [-0.4, 1.8],
        [0.3, 2.5],
        [0.6, 1.7],
        [-0.2, 2.3],
    ]
)
_TARGETS = jnp.repeat(jnp.arange(3, dtype=jnp.int32), 4)
_SCHEMA = TargetSchema(
    "multiclass",
    names=("species",),
    class_labels=("setosa", "versicolor", "virginica"),
)
_MULTILABEL_FEATURES = jnp.array(
    [
        [-2.0, -1.0],
        [-1.5, 0.5],
        [-1.0, 1.5],
        [-0.5, -1.5],
        [-0.1, 0.8],
        [0.2, -0.8],
        [0.6, 1.3],
        [1.0, -1.2],
        [1.5, 0.4],
        [2.0, 1.0],
    ]
)
_MULTILABEL_TARGETS = jnp.stack(
    (
        (_MULTILABEL_FEATURES[:, 0] > 0.0).astype(jnp.int32),
        (_MULTILABEL_FEATURES[:, 1] > 0.0).astype(jnp.int32),
        ((_MULTILABEL_FEATURES[:, 0] + _MULTILABEL_FEATURES[:, 1]) > 0.3).astype(
            jnp.int32
        ),
    ),
    axis=-1,
)
_MULTILABEL_SCHEMA = TargetSchema("multilabel", names=("warm", "wet", "favorable"))


def _base():
    return GaussianNaiveBayesRecipe(var_smoothing=0.03)


def _sparse(values):
    columns = jnp.broadcast_to(jnp.arange(values.shape[-1]), values.shape)
    return SparseFeatures(values, columns, feature_count=values.shape[-1])


@pytest.mark.parametrize(
    ("recipe", "model_type", "method", "component_count"),
    [
        (OneVsRestRecipe(_base()), OneVsRestModel, "one-vs-rest", 4),
        (OneVsOneRecipe(_base()), OneVsOneModel, "one-vs-one", 4),
        (OutputCodeRecipe(_base()), OutputCodeModel, "error-correcting-output-code", 7),
    ],
)
def test_multiclass_compositions_preserve_labels_normalization_and_distinct_evidence(
    recipe, model_type, method, component_count
):
    result = recipe.fit_batch(
        MLBatch(_FEATURES, _TARGETS, target_schema=_SCHEMA),
        key=jax.random.key(4),
    )
    model = result.as_trainable()
    probability = result.model(_FEATURES)

    assert isinstance(model, model_type)
    assert bool(result.valid)
    assert result.method == method
    assert result.diagnostics.method == method
    assert result.diagnostics.component_count == component_count
    assert probability.shape == (12, 3)
    assert jnp.allclose(jnp.sum(probability, axis=-1), 1.0, atol=1e-6)
    assert model.predict(_FEATURES).shape == (12,)
    assert jnp.array_equal(model.labels, jnp.arange(3))
    assert model.target_schema.class_labels == _SCHEMA.class_labels
    assert jax.jit(model)(_FEATURES[:2]).shape == (2, 3)
    assert jax.vmap(model)(_FEATURES[:2]).shape == (2, 3)
    assert result.gradient_contract.prediction_inputs == "smooth"
    if isinstance(model, OneVsOneModel):
        votes = model.vote_counts(_FEATURES)
        assert votes.dtype == jnp.int32
        assert jnp.all(jnp.sum(votes, axis=-1) == len(model.pairs))
        assert model.pairwise_decision_function(_FEATURES).shape == (12, 3)
    if isinstance(model, OutputCodeModel):
        code = jnp.asarray(model.codebook)
        distances = jnp.sum(code[:, None, :] != code[None, :, :], axis=-1)
        assert jnp.min(distances + 100 * jnp.eye(3, dtype=jnp.int32)) >= 3
        assert model.code_decision_function(_FEATURES).shape[-1] == 6


def test_one_vs_rest_preserves_case_masks_weights_and_keys_and_rejects_sparse_input():
    features = jnp.stack((_FEATURES, 1.1 * _FEATURES + 0.2))
    targets = jnp.stack((_TARGETS, _TARGETS))
    feature_mask = jnp.ones_like(features, dtype=bool).at[:, 3, 0].set(False)
    target_mask = jnp.ones_like(targets, dtype=bool).at[:, 7].set(False)
    batch = MLBatch(
        features,
        targets,
        feature_mask=feature_mask,
        target_mask=target_mask,
        sample_mask=jnp.arange(12) != 11,
        sample_weight=jnp.linspace(0.6, 1.5, 12),
        measure_weight=jnp.linspace(1.4, 0.8, 12),
        target_schema=_SCHEMA,
    )
    recipe = OneVsRestRecipe(
        GaussianNaiveBayesRecipe(weight_policy="product", var_smoothing=0.03)
    )
    first = recipe.fit_batch(batch, key=jax.random.key(9))
    repeated = recipe.fit_batch(batch, key=jax.random.key(9))
    with pytest.raises(TypeError, match="requires dense features"):
        OneVsRestRecipe(_base(), num_classes=3).fit_batch(
            MLBatch(_sparse(_FEATURES), _TARGETS), key=jax.random.key(10)
        )

    assert first.valid.shape == (2,)
    assert jnp.all(first.valid)
    assert first.model(features).shape == (2, 12, 3)
    assert jnp.allclose(first.model(features), repeated.model(features))


@pytest.mark.parametrize(
    ("recipe", "model_type", "method", "prediction_inputs"),
    [
        (
            MultilabelRecipe(_base()),
            MultilabelModel,
            "multilabel-binary-relevance",
            "smooth",
        ),
        (
            ClassifierChainRecipe(_base()),
            ClassifierChainModel,
            "classifier-chain",
            "none",
        ),
        (
            SmoothClassifierChainRecipe(_base()),
            SmoothClassifierChainModel,
            "smooth-classifier-chain",
            "smooth",
        ),
    ],
)
def test_multilabel_and_chain_families_preserve_target_axis_masks_and_probabilities(
    recipe, model_type, method, prediction_inputs
):
    target_mask = jnp.ones_like(_MULTILABEL_TARGETS, dtype=bool).at[1, 2].set(False)
    batch = MLBatch(
        _MULTILABEL_FEATURES,
        _MULTILABEL_TARGETS,
        target_mask=target_mask,
        sample_mask=jnp.arange(10) != 8,
        sample_weight=jnp.linspace(0.7, 1.4, 10),
        measure_weight=jnp.linspace(1.3, 0.8, 10),
        target_schema=_MULTILABEL_SCHEMA,
    )
    result = recipe.fit_batch(batch, key=jax.random.key(12))
    model = result.as_trainable()
    probability = model(_MULTILABEL_FEATURES)

    assert isinstance(model, model_type)
    assert bool(result.valid)
    assert result.method == method
    assert probability.shape == _MULTILABEL_TARGETS.shape
    assert jnp.all((probability >= 0.0) & (probability <= 1.0))
    assert model.predict(_MULTILABEL_FEATURES).shape == _MULTILABEL_TARGETS.shape
    assert model.target_schema.names == _MULTILABEL_SCHEMA.names
    assert jax.jit(model)(_MULTILABEL_FEATURES[:2]).shape == (2, 3)
    assert jax.vmap(model)(_MULTILABEL_FEATURES[:2]).shape == (2, 3)
    assert result.gradient_contract.prediction_inputs == prediction_inputs
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"


def test_exact_and_smooth_classifier_chains_use_hard_and_smooth_link_outputs():
    batch = MLBatch(
        _MULTILABEL_FEATURES,
        _MULTILABEL_TARGETS,
        target_schema=_MULTILABEL_SCHEMA,
    )
    hard_result = ClassifierChainRecipe(_base()).fit_batch(batch, key=jax.random.key(2))
    smooth_result = SmoothClassifierChainRecipe(_base()).fit_batch(
        batch, key=jax.random.key(2)
    )
    hard = hard_result.as_trainable()
    smooth = smooth_result.as_trainable()
    points = jnp.array([[0.05, 0.05], [-0.05, -0.05]])
    hard_first = hard.models[0].decision_function(points)
    smooth_first = smooth.models[0].decision_function(points)
    hard_link = (hard_first[..., 1] - hard_first[..., 0] >= 0.0).astype(points.dtype)
    smooth_link = jax.nn.sigmoid(smooth_first[..., 1] - smooth_first[..., 0]).astype(
        points.dtype
    )
    hard_augmented = jnp.concatenate((points, hard_link[..., None]), axis=-1)
    smooth_augmented = jnp.concatenate((points, smooth_link[..., None]), axis=-1)

    assert jnp.all((hard_link == 0.0) | (hard_link == 1.0))
    assert jnp.all((smooth_link > 0.0) & (smooth_link < 1.0))
    assert jnp.allclose(
        hard.decision_function(points)[..., 1],
        hard.models[1].decision_function(hard_augmented)[..., 1]
        - hard.models[1].decision_function(hard_augmented)[..., 0],
    )
    assert jnp.allclose(
        smooth.decision_function(points)[..., 1],
        smooth.models[1].decision_function(smooth_augmented)[..., 1]
        - smooth.models[1].decision_function(smooth_augmented)[..., 0],
    )
    assert not jnp.allclose(
        hard.decision_function(points), smooth.decision_function(points)
    )
    assert hard_result.gradient_contract.prediction_inputs == "none"
    assert smooth_result.gradient_contract.prediction_inputs == "smooth"


@pytest.mark.parametrize(
    ("recipe", "features", "targets", "probe", "prediction_gradient"),
    [
        (
            OneVsRestRecipe(_base(), num_classes=3),
            _FEATURES,
            _TARGETS,
            jnp.array([0.2, -0.1]),
            True,
        ),
        (
            OneVsOneRecipe(_base(), num_classes=3),
            _FEATURES,
            _TARGETS,
            jnp.array([0.2, -0.1]),
            True,
        ),
        (
            OutputCodeRecipe(_base(), num_classes=3),
            _FEATURES,
            _TARGETS,
            jnp.array([0.2, -0.1]),
            True,
        ),
        (
            MultilabelRecipe(_base()),
            _MULTILABEL_FEATURES,
            _MULTILABEL_TARGETS,
            jnp.array([0.2, -0.1]),
            True,
        ),
        (
            ClassifierChainRecipe(_base()),
            _MULTILABEL_FEATURES,
            _MULTILABEL_TARGETS,
            jnp.array([0.2, -0.1]),
            False,
        ),
        (
            SmoothClassifierChainRecipe(_base()),
            _MULTILABEL_FEATURES,
            _MULTILABEL_TARGETS,
            jnp.array([0.2, -0.1]),
            True,
        ),
    ],
)
def test_every_composition_has_declared_fit_and_prediction_parameter_gradients(
    recipe, features, targets, probe, prediction_gradient
):
    weights = jnp.linspace(0.8, 1.3, features.shape[0])

    def fit_loss(values, sample_weight, smoothing):
        configured = eqx.tree_at(
            lambda item: item.base_recipe.var_smoothing,
            recipe,
            smoothing,
        )
        model = configured.fit_batch(
            MLBatch(values, targets, sample_weight=sample_weight),
            key=jax.random.key(3),
        ).as_trainable()
        return jnp.sum(jnp.square(model.decision_function(probe)))

    fit_gradients = jax.grad(fit_loss, argnums=(0, 1, 2))(
        features, weights, jnp.array(0.03)
    )
    result = recipe.fit_batch(
        MLBatch(features, targets, sample_weight=weights), key=jax.random.key(3)
    )
    model = result.as_trainable()
    parameter_gradient = jax.grad(
        lambda means: jnp.sum(
            jnp.square(
                eqx.tree_at(
                    lambda item: item.models[0].means,
                    model,
                    means,
                ).decision_function(probe)
            )
        )
    )(model.models[0].means)

    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in fit_gradients)
    assert jnp.all(jnp.isfinite(parameter_gradient))
    if prediction_gradient:
        input_gradient = jax.grad(
            lambda point: jnp.sum(jnp.square(model.decision_function(point)))
        )(probe)
        assert jnp.all(jnp.isfinite(input_gradient))
    else:
        assert result.gradient_contract.prediction_inputs == "none"


def test_composition_failures_report_vocabulary_support_multilabel_domain_and_capacity():
    invalid_labels = OneVsRestRecipe(_base(), num_classes=3).fit_batch(
        MLBatch(_FEATURES, _TARGETS.at[0].set(9))
    )
    single_class = OneVsRestRecipe(_base(), num_classes=3).fit_batch(
        MLBatch(_FEATURES, jnp.zeros(12, dtype=jnp.int32))
    )
    invalid_multilabel = MultilabelRecipe(_base()).fit_batch(
        MLBatch(_MULTILABEL_FEATURES, _MULTILABEL_TARGETS.at[0, 1].set(2))
    )
    empty_multilabel = MultilabelRecipe(_base()).fit_batch(
        MLBatch(
            _MULTILABEL_FEATURES,
            _MULTILABEL_TARGETS,
            sample_mask=jnp.zeros(10, dtype=bool),
        )
    )

    assert int(invalid_labels.status) == ML_INFEASIBLE
    assert int(single_class.status) == ML_INSUFFICIENT_DATA
    assert int(invalid_multilabel.status) == ML_INFEASIBLE
    assert int(empty_multilabel.status) == ML_INSUFFICIENT_DATA
    with pytest.raises(ValueError, match="Hamming distance"):
        OutputCodeRecipe(
            _base(),
            ((0, 0, 0), (0, 1, 1), (1, 0, 1)),
            num_classes=3,
        )
    with pytest.raises(ValueError, match="scalar multiclass targets"):
        OneVsOneRecipe(_base(), num_classes=3).fit_batch(
            MLBatch(_FEATURES, jax.nn.one_hot(_TARGETS, 3))
        )
    with pytest.raises(ValueError, match="one label axis"):
        ClassifierChainRecipe(_base()).fit_batch(
            MLBatch(_MULTILABEL_FEATURES, _MULTILABEL_TARGETS[..., None])
        )
