#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax._model import AbstractArrayModel
from phydrax.ml import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    GradientContract,
    MLBatch,
    TargetSchema,
)
from phydrax.ml.semi_supervised import (
    HardLabelPropagationModel,
    HardLabelPropagationRecipe,
    HardOneClassCompositionModel,
    HardOneClassCompositionRecipe,
    HardSelfTrainingModel,
    HardSelfTrainingRecipe,
    LabelPropagationModel,
    LabelPropagationRecipe,
    LabelSpreadingRecipe,
    SoftOneClassCompositionModel,
    SoftOneClassCompositionRecipe,
    SoftSelfTrainingModel,
    SoftSelfTrainingRecipe,
)


def _result(model, batch, method):
    valid = jnp.ones(batch.case_shape or (), dtype=bool)
    status = jnp.zeros(batch.case_shape or (), dtype=jnp.int32)
    return FitResult(
        model,
        FitDiagnostics(
            valid=valid,
            status=status,
            effective_samples=jnp.sum(batch.sample_mask, axis=-1),
            method=method,
        ),
        valid=valid,
        status=status,
        method=method,
        gradient_contract=GradientContract.direct(),
    )


class _PriorModel(AbstractArrayModel):
    prior: jax.Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, prior, case_shape, in_size):
        self.prior = jnp.asarray(prior)
        self.case_shape = tuple(case_shape)
        self.in_size = int(in_size)
        self.out_size = int(self.prior.shape[-1])

    def __call__(self, x, /, *, key=None):
        del key
        values = jnp.asarray(x)
        sample_shape = values.shape[len(self.case_shape) : -1]
        prior = self.prior.reshape(
            self.case_shape + (1,) * len(sample_shape) + (self.out_size,)
        )
        return jnp.broadcast_to(prior, values.shape[:-1] + (self.out_size,))


class _PriorRecipe(AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        del key
        targets = batch.require_targets()
        weight = batch.effective_weight() * jnp.all(batch.target_mask, axis=-1)
        prior = jnp.sum(weight[..., None] * targets, axis=-2) / jnp.maximum(
            jnp.sum(weight, axis=-1, keepdims=True), 1e-6
        )
        prior = prior / jnp.maximum(jnp.sum(prior, axis=-1, keepdims=True), 1e-6)
        return _result(
            _PriorModel(prior, batch.case_shape, batch.feature_count), batch, "prior"
        )


class _ScoreModel(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, in_size):
        self.in_size = int(in_size)
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.asarray(x)[..., 0]


class _ScoreRecipe(AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        del key
        return _result(_ScoreModel(batch.feature_count), batch, "score")


class _ConstantModel(AbstractArrayModel):
    value: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, value, in_size):
        self.value = jnp.asarray(value)
        self.in_size = int(in_size)
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.broadcast_to(self.value, jnp.asarray(x).shape[:-1])


class _ConstantRecipe(AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        del key
        return _result(_ConstantModel(2.0, batch.feature_count), batch, "constant")


def _graph_batch(case=True):
    x = jnp.array([[-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0]])
    y = jnp.array([10, 10, 10, 20, 20, 20])
    target_mask = jnp.array([True, False, True, True, False, True])
    if case:
        x = jnp.stack((x, x + 0.1))
        y = jnp.stack((y, y))
    return MLBatch(
        x,
        y,
        target_mask=target_mask,
        sample_weight=jnp.array([2.0, 1.0, 1.0, 1.0, 1.0, 2.0]),
        target_schema=TargetSchema("multiclass", class_labels=(10, 20)),
    )


def _soft_batch():
    x = jnp.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    targets = jnp.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 1.0]])
    target_mask = jnp.array(
        [[True, True], [False, False], [False, False], [False, False], [True, True]]
    )
    return MLBatch(
        x,
        targets,
        target_mask=target_mask,
        target_schema=TargetSchema("multiclass", class_labels=(10, 20)),
    )


def test_label_propagation_spreading_schema_masks_weights_cases_jit_and_grad():
    batch = _graph_batch(case=True)
    propagation = LabelPropagationRecipe(iterations=80, tolerance=1e-3).fit_batch(batch)
    spreading = LabelSpreadingRecipe(alpha=0.7, iterations=80, tolerance=1e-3).fit_batch(
        batch
    )
    model = propagation.as_trainable()
    assert isinstance(model, LabelPropagationModel)
    spreading_model = spreading.as_trainable()
    assert isinstance(spreading_model, LabelPropagationModel)
    assert model.class_labels.tolist() == [10, 20]
    probabilities = model(batch.dense_features())
    assert probabilities.shape == (2, 6, 2)
    assert jnp.allclose(jnp.sum(probabilities, axis=-1), 1.0)
    assert spreading_model(batch.dense_features()).shape == (2, 6, 2)
    assert model.distributions.shape == (2, 6, 2)
    assert spreading_model.distributions.shape == (2, 6, 2)
    assert propagation.diagnostics.labelled_samples.shape == (2,)
    assert jnp.allclose(jax.jit(model)(batch.dense_features()), probabilities)
    gradient = jax.grad(lambda point: jnp.sum(model(point)[..., 0]))(
        batch.dense_features()
    )
    assert gradient.shape == batch.dense_features().shape
    assert jnp.all(jnp.isfinite(gradient))


def test_hard_label_reporting_is_distinct_and_preserves_external_vocabulary():
    batch = _graph_batch(case=False)
    result = HardLabelPropagationRecipe(
        LabelPropagationRecipe(iterations=80, tolerance=1e-3)
    ).fit_batch(batch)
    model = result.as_trainable()
    assert isinstance(model, HardLabelPropagationModel)
    labels = model(batch.dense_features())
    assert labels.shape == (6,)
    assert jnp.all((labels == 10) | (labels == 20))
    probabilities = model.soft_model(batch.dense_features())
    assert probabilities.shape == (6, 2)
    assert jnp.issubdtype(probabilities.dtype, jnp.inexact)
    assert result.gradient_contract.prediction_inputs == "none"


def test_graph_models_fail_closed_for_complex_features_vocabularies_and_partial_masks():
    batch = _graph_batch(case=False)
    with pytest.raises(TypeError, match="real-valued features"):
        LabelPropagationRecipe().fit_batch(
            MLBatch(
                batch.dense_features().astype(jnp.complex64),
                batch.require_targets(),
                target_mask=batch.target_mask,
                target_schema=batch.target_schema,
            )
        )
    with pytest.raises(ValueError, match="class vocabulary"):
        LabelPropagationRecipe(num_classes=3).fit_batch(batch)
    partial = _soft_batch()
    bad_mask = partial.target_mask.at[1, 0].set(True)
    with pytest.raises(Exception, match="every class or no class"):
        LabelSpreadingRecipe().fit_batch(
            MLBatch(
                partial.dense_features(),
                partial.require_targets(),
                target_mask=bad_mask,
                target_schema=partial.target_schema,
            )
        )


def test_soft_and_hard_self_training_are_distinct_keyed_and_deterministic():
    batch = _soft_batch()
    soft_recipe = SoftSelfTrainingRecipe(_PriorRecipe(), iterations=2, blend=0.5)
    hard_recipe = HardSelfTrainingRecipe(
        _PriorRecipe(), iterations=2, confidence_threshold=0.4
    )
    with pytest.raises(ValueError, match="explicit JAX key"):
        soft_recipe.fit_batch(batch)
    soft = soft_recipe.fit_batch(batch, key=jax.random.key(4))
    soft_again = soft_recipe.fit_batch(batch, key=jax.random.key(4))
    hard = hard_recipe.fit_batch(batch, key=jax.random.key(4))
    assert isinstance(soft.as_trainable(), SoftSelfTrainingModel)
    assert isinstance(hard.as_trainable(), HardSelfTrainingModel)
    assert jnp.allclose(
        soft.model(batch.dense_features()), soft_again.model(batch.dense_features())
    )
    assert soft.diagnostics.child_status.shape == (3,)
    assert hard.gradient_contract.nondifferentiable_outputs == (
        "pseudo_label",
        "pseudo_label_acceptance",
    )
    assert jax.jit(soft.as_trainable())(batch.dense_features()).shape == (5, 2)


def test_soft_and_hard_one_class_compositions_gate_natively():
    x = jnp.array([[-2.0], [-0.5], [0.5], [2.0]])
    batch = MLBatch(x, jnp.array([0.0, 0.0, 1.0, 1.0]))
    soft_recipe = SoftOneClassCompositionRecipe(
        _ScoreRecipe(), _ConstantRecipe(), threshold=0.0, temperature=0.5
    )
    hard_recipe = HardOneClassCompositionRecipe(
        _ScoreRecipe(), _ConstantRecipe(), threshold=0.0
    )
    with pytest.raises(ValueError, match="explicit JAX key"):
        hard_recipe.fit_batch(batch)
    soft = soft_recipe.fit_batch(batch, key=jax.random.key(8))
    hard = hard_recipe.fit_batch(batch, key=jax.random.key(8))
    soft_model = soft.as_trainable()
    hard_model = hard.as_trainable()
    assert isinstance(soft_model, SoftOneClassCompositionModel)
    assert isinstance(hard_model, HardOneClassCompositionModel)
    assert jnp.all((soft_model.acceptance(x) > 0.0) & (soft_model.acceptance(x) < 1.0))
    assert jnp.all(hard_model(x[:2]) == 0.0)
    assert jnp.all(hard_model(x[2:]) == 2.0)
    assert jnp.allclose(
        jax.grad(lambda value: jnp.sum(soft_model(value)))(x),
        2.0
        * soft_model.acceptance(x)[:, None]
        * (1.0 - soft_model.acceptance(x)[:, None])
        / 0.5,
    )
