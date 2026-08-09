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
)
from phydrax.ml.ensemble import (
    BaggingRecipe,
    HardVotingModel,
    HardVotingRecipe,
    HeterogeneousEnsembleModel,
    HomogeneousEnsembleModel,
    MixtureOfExpertsModel,
    MixtureOfExpertsRecipe,
    RandomSubspaceRecipe,
    SoftVotingModel,
    SoftVotingRecipe,
    StackingModel,
    StackingRecipe,
)


def _result(model, batch, method):
    valid = jnp.ones(batch.case_shape or (), dtype=bool)
    status = jnp.zeros(batch.case_shape or (), dtype=jnp.int32)
    diagnostics = FitDiagnostics(
        valid=valid,
        status=status,
        effective_samples=jnp.sum(batch.sample_mask, axis=-1),
        method=method,
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        gradient_contract=GradientContract.direct(),
    )


class _ConstantModel(AbstractArrayModel):
    value: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, value, in_size):
        self.value = jnp.asarray(value, dtype=float)
        self.in_size = int(in_size)
        self.out_size = 1

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.broadcast_to(self.value, jnp.asarray(x).shape[:-1] + (1,))


class _ConstantRecipe(AbstractRecipe):
    value: float = eqx.field(static=True)

    def __init__(self, value):
        self.value = float(value)

    def fit_batch(self, batch, /, *, key=None):
        del key
        return _result(_ConstantModel(self.value, batch.feature_count), batch, "constant")


class _CountRecipe(AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        del key
        return _result(
            _ConstantModel(float(batch.sample_count), batch.feature_count),
            batch,
            "sample-count",
        )


class _FeatureMeanRecipe(AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        del key
        value = jnp.mean(batch.dense_features()[..., 0])
        return _result(_ConstantModel(value, batch.feature_count), batch, "feature-mean")


class _GateModel(AbstractArrayModel):
    logits: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, logits, in_size):
        self.logits = jnp.asarray(logits)
        self.in_size = int(in_size)
        self.out_size = int(self.logits.shape[0])

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.broadcast_to(
            self.logits, jnp.asarray(x).shape[:-1] + self.logits.shape
        )


class _GateRecipe(AbstractRecipe):
    count: int = eqx.field(static=True)

    def __init__(self, count):
        self.count = int(count)

    def fit_batch(self, batch, /, *, key=None):
        del key
        return _result(
            _GateModel(jnp.zeros((self.count,)), batch.feature_count), batch, "gate"
        )


def _batch(case=False):
    x = jnp.arange(24.0).reshape(2, 6, 2) if case else jnp.arange(12.0).reshape(6, 2)
    y = jnp.sum(x, axis=-1, keepdims=True)
    return MLBatch(
        x,
        y,
        sample_mask=jnp.array([True, True, True, False, True, True]),
        sample_weight=jnp.array([1.0, 2.0, 1.0, 0.0, 1.0, 3.0]),
    )


def test_bagging_homogeneous_uq_keys_case_axes_jit_vmap_and_grad():
    batch = _batch(case=True)
    recipe = BaggingRecipe(_CountRecipe(), num_members=3, sample_fraction=0.5)
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(batch)

    first = recipe.fit_batch(batch, key=jax.random.key(7))
    second = recipe.fit_batch(batch, key=jax.random.key(7))
    model = first.as_trainable()
    assert isinstance(model, HomogeneousEnsembleModel)
    points = jnp.ones((2, 4, 2))
    assert jnp.allclose(model(points), 3.0)
    assert jnp.allclose(first.model(points), second.model(points))
    assert first.valid.shape == (2,)
    assert jnp.all(first.valid)

    predictive = model.predictive(points, key=jax.random.key(2))
    assert predictive.samples.data.shape == (3, 2, 4, 1)
    assert jnp.allclose(predictive.mean().data, model(points))
    assert jnp.allclose(jax.jit(model)(points), model(points))
    assert jnp.allclose(
        jax.grad(lambda value: jnp.sum(model(value)))(points), jnp.zeros_like(points)
    )
    assert jax.vmap(lambda point: model(point))(points[0]).shape == (4, 1)


def test_random_subspace_is_heterogeneous_deterministic_and_capacity_checked():
    batch = _batch()
    recipe = RandomSubspaceRecipe(_CountRecipe(), num_members=4, feature_count=1)
    first = recipe.fit_batch(batch, key=jax.random.key(11))
    second = recipe.fit_batch(batch, key=jax.random.key(11))
    model = first.as_trainable()
    assert isinstance(model, HeterogeneousEnsembleModel)
    assert jnp.allclose(
        model(batch.dense_features()), second.model(batch.dense_features())
    )
    assert (
        model.predictive(
            batch.dense_features(), key=jax.random.key(1)
        ).samples.data.shape[0]
        == 4
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        RandomSubspaceRecipe(_CountRecipe(), num_members=2, feature_count=3).fit_batch(
            batch, key=jax.random.key(0)
        )


def test_soft_and_hard_voting_are_distinct_and_fail_closed_on_weights():
    batch = _batch()
    soft = SoftVotingRecipe(
        (_ConstantRecipe(1.0), _ConstantRecipe(3.0)),
        member_weights=jnp.array([1.0, 3.0]),
    ).fit_batch(batch, key=jax.random.key(0))
    hard = HardVotingRecipe(
        (_ConstantRecipe(2.0), _ConstantRecipe(2.0), _ConstantRecipe(9.0))
    ).fit_batch(batch, key=jax.random.key(0))
    assert isinstance(soft.as_trainable(), SoftVotingModel)
    assert isinstance(hard.as_trainable(), HardVotingModel)
    assert jnp.allclose(soft.model(batch.dense_features()), 2.5)
    assert jnp.all(hard.model(batch.dense_features()) == 2.0)
    assert hard.gradient_contract.prediction_inputs == "none"
    with pytest.raises(TypeError, match="real-valued"):
        SoftVotingRecipe((_ConstantRecipe(1),), member_weights=jnp.array([1.0j]))
    with pytest.raises(Exception, match="nonnegative"):
        SoftVotingRecipe(
            (_ConstantRecipe(1), _ConstantRecipe(2)),
            member_weights=jnp.array([1.0, -1.0]),
        )


def test_stacking_meta_features_are_strictly_out_of_fold():
    batch = _batch()
    result = StackingRecipe(
        (_CountRecipe(),), _FeatureMeanRecipe(), num_folds=3
    ).fit_batch(batch, key=jax.random.key(5))
    model = result.as_trainable()
    assert isinstance(model, StackingModel)
    # Each fold trains on four of six samples. A leaky full-data base fit would produce six.
    assert jnp.allclose(model(batch.dense_features()), 4.0)
    assert result.diagnostics.auxiliary_status.shape == (1, 3)
    assert result.gradient_contract.nondifferentiable_outputs == ("fold_assignment",)
    with pytest.raises(ValueError, match="cannot exceed"):
        StackingRecipe((_CountRecipe(),), _FeatureMeanRecipe(), num_folds=7).fit_batch(
            batch, key=jax.random.key(0)
        )


def test_mixture_of_experts_uses_smooth_gate_and_structured_diagnostics():
    batch = _batch(case=True)
    result = MixtureOfExpertsRecipe(
        (_ConstantRecipe(1.0), _ConstantRecipe(3.0)),
        _GateRecipe(2),
        temperature=0.5,
    ).fit_batch(batch, key=jax.random.key(9))
    model = result.as_trainable()
    assert isinstance(model, MixtureOfExpertsModel)
    assert jnp.allclose(model.gating_weights(batch.dense_features()), 0.5)
    assert jnp.allclose(model(batch.dense_features()), 2.0)
    assert result.valid.shape == (2,)
    assert result.diagnostics.member_status.shape == (3, 2)
    assert jnp.allclose(
        jax.grad(lambda x: jnp.sum(model(x)))(batch.dense_features()),
        jnp.zeros_like(batch.dense_features()),
    )


def test_ensemble_models_reject_misaligned_members():
    with pytest.raises(ValueError, match="identical input and output sizes"):
        HomogeneousEnsembleModel((_ConstantModel(1.0, 1), _ConstantModel(2.0, 2)))
    with pytest.raises(ValueError, match="identical input and output sizes"):
        HeterogeneousEnsembleModel((_ConstantModel(1.0, 1), _ConstantModel(2.0, 2)))
