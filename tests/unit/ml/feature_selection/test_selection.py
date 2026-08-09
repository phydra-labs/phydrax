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
    ML_SUCCESS,
    MLBatch,
)
from phydrax.ml.feature_selection import (
    ContinuousFeatureGateModel,
    ContinuousSparseGateRecipe,
    ExactFeatureSelectorModel,
    ModelBasedSelectionRecipe,
    MutualInformationFilterRecipe,
    RecursiveFeatureEliminationRecipe,
    ScoreFilterRecipe,
    SequentialFeatureSelectionRecipe,
    VarianceFilterRecipe,
)


class _LinearModel(AbstractArrayModel):
    coefficients: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, coefficients):
        self.coefficients = jnp.asarray(coefficients)
        self.in_size = int(self.coefficients.shape[0])
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.einsum("...f,f->...", jnp.asarray(x), self.coefficients)


class _ImportanceRecipe(AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        del key
        x = batch.dense_features().reshape((-1, batch.feature_count))
        y = batch.require_targets().reshape((-1,))
        weight = batch.effective_weight().reshape((-1,))
        design = jnp.sqrt(weight)[:, None] * x
        target = jnp.sqrt(weight) * y
        coefficients = jnp.linalg.pinv(design) @ target
        valid = jnp.asarray(True)
        diagnostics = FitDiagnostics(
            valid=valid,
            status=ML_SUCCESS,
            effective_samples=jnp.sum(weight > 0),
            method="test-linear",
        )
        return FitResult(
            _LinearModel(coefficients),
            diagnostics,
            valid=valid,
            status=ML_SUCCESS,
            method="test-linear",
            gradient_contract=GradientContract.direct(),
        )


def _linear_importance(model):
    return model.coefficients


def _batch(case=False):
    base = jnp.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 0.0],
            [4.0, 1.0, 2.0],
            [5.0, 1.0, 1.0],
        ]
    )
    x = jnp.stack((base, base + jnp.array([1.0, 0.0, 0.0]))) if case else base
    y = 2.0 * x[..., 0] + 0.1 * x[..., 2]
    feature_mask = jnp.ones_like(x, dtype=bool).at[..., 5, 2].set(False)
    return MLBatch(
        x,
        y,
        feature_mask=feature_mask,
        sample_mask=jnp.array([True, True, True, True, True, False]),
        sample_weight=jnp.array([1.0, 2.0, 1.0, 1.0, 3.0, 9.0]),
    )


def test_variance_and_score_filters_preserve_case_axes_masks_weights_and_gradients():
    batch = _batch(case=True)
    variance = VarianceFilterRecipe(1e-6, max_features=2).fit_batch(batch)
    score = ScoreFilterRecipe(threshold=0.2, max_features=1).fit_batch(batch)
    variance_model = variance.as_trainable()
    assert isinstance(variance_model, ExactFeatureSelectorModel)
    assert variance_model(batch.dense_features()).shape == (2, 6, 2)
    assert variance.diagnostics.selection.indices.shape == (2,)
    assert jnp.all(variance.diagnostics.selection.selected)
    assert score.diagnostics.selection.indices[0] == 0

    model = score.as_trainable()
    point = jnp.array([3.0, 1.0, 2.0])
    assert jnp.allclose(jax.jit(model)(point), jnp.array([3.0]))
    assert jnp.allclose(
        jax.grad(lambda x: jnp.sum(model(x)))(point), jnp.array([1.0, 0.0, 0.0])
    )
    assert jax.vmap(model)(batch.dense_features()[0]).shape == (6, 1)
    assert score.gradient_contract.nondifferentiable_outputs == (
        "selected_indices",
        "selected_mask",
    )


def test_mutual_information_is_deterministic_fixed_capacity_and_fail_closed():
    batch = _batch()
    recipe = MutualInformationFilterRecipe(num_bins=3, threshold=0.0, max_features=2)
    first = recipe.fit_batch(batch, key=jax.random.key(1))
    second = recipe.fit_batch(batch, key=jax.random.key(9))
    assert jnp.array_equal(
        first.diagnostics.selection.indices, second.diagnostics.selection.indices
    )
    assert first.diagnostics.selection.indices.shape == (2,)
    complex_batch = MLBatch(
        batch.dense_features().astype(jnp.complex64), batch.require_targets()
    )
    with pytest.raises(TypeError, match="undefined for complex"):
        recipe.fit_batch(complex_batch)


def test_recursive_sequential_and_model_based_selection_find_signal():
    batch = _batch()
    estimator = _ImportanceRecipe()
    recursive = RecursiveFeatureEliminationRecipe(
        estimator,
        num_features=1,
        importance_getter=_linear_importance,
    ).fit_batch(batch, key=jax.random.key(2))
    sequential = SequentialFeatureSelectionRecipe(
        estimator,
        num_features=1,
        validation_fraction=1 / 3,
    ).fit_batch(batch, key=jax.random.key(3))
    model_based = ModelBasedSelectionRecipe(
        estimator,
        threshold=0.2,
        max_features=1,
        importance_getter=_linear_importance,
    ).fit_batch(batch)

    assert recursive.diagnostics.selection.indices[0] == 0
    assert sequential.diagnostics.selection.indices[0] == 0
    assert model_based.diagnostics.selection.indices[0] == 0
    assert recursive.diagnostics.estimator_status.shape[0] == 3
    assert sequential.diagnostics.estimator_status.shape[0] == batch.feature_count
    with pytest.raises(ValueError, match="explicit JAX key"):
        SequentialFeatureSelectionRecipe(estimator, num_features=1).fit_batch(batch)


def test_continuous_sparse_gate_is_distinct_smooth_jittable_and_vmap_safe():
    batch = _batch(case=True)
    result = ContinuousSparseGateRecipe(temperature=0.2, sparsity=0.4).fit_batch(batch)
    model = result.as_trainable()
    assert isinstance(model, ContinuousFeatureGateModel)
    assert result.diagnostics.selection is None
    assert result.diagnostics.relaxed_gates.shape == (3,)
    assert jnp.all((model.gates > 0.0) & (model.gates < 1.0))
    point = jnp.array([2.0, 1.0, 3.0])
    assert jnp.allclose(jax.jit(model)(point), model(point))
    assert jnp.allclose(jax.grad(lambda x: jnp.sum(model(x)))(point), model.gates)
    assert jax.vmap(model)(batch.dense_features()[0]).shape == (6, 3)
    assert result.gradient_contract.fit_mode == "relaxed"


def test_selector_capacity_scores_weights_and_importance_fail_closed():
    batch = _batch()
    with pytest.raises(ValueError, match="positive"):
        VarianceFilterRecipe(max_features=0)
    with pytest.raises(ValueError, match="one score per feature"):
        ScoreFilterRecipe(lambda _: jnp.ones((2,))).fit_batch(batch)
    with pytest.raises(TypeError, match="real-valued scores"):
        ScoreFilterRecipe(lambda _: jnp.ones((3,), dtype=jnp.complex64)).fit_batch(batch)
    bad_weight = MLBatch(
        batch.dense_features(),
        batch.require_targets(),
        sample_weight=jnp.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0]),
    )
    with pytest.raises(Exception, match="nonnegative"):
        VarianceFilterRecipe().fit_batch(bad_weight)

    with pytest.raises(TypeError, match="importance_getter"):
        ModelBasedSelectionRecipe(_ImportanceRecipe())
