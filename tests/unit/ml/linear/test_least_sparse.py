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
    ML_NONCONVERGED,
    ML_RANK_DEFICIENT,
    MLBatch,
    SparseFeatures,
)
from phydrax.ml.linear import (
    ElasticNetModel,
    ElasticNetRecipe,
    GroupLassoModel,
    GroupLassoRecipe,
    LassoModel,
    LassoRecipe,
    OLSModel,
    OLSRecipe,
    RidgeModel,
    RidgeRecipe,
    SparseGroupLassoModel,
    SparseGroupLassoRecipe,
    TikhonovModel,
    TikhonovRecipe,
)


def _regression_data():
    features = jnp.array(
        [
            [-2.0, 0.0, 1.0],
            [-1.2, 1.0, 0.0],
            [-0.5, 0.0, -1.0],
            [0.2, -1.0, 0.0],
            [0.8, 0.0, 1.5],
            [1.3, 1.2, 0.0],
            [1.9, 0.0, -0.7],
            [2.5, -0.8, 0.0],
        ]
    )
    coefficients = jnp.array([[1.5, -0.3], [-0.4, 1.2], [0.7, 0.5]])
    targets = features @ coefficients + jnp.array([0.2, -0.1])
    return features, targets


def _operator_sparse(features):
    columns = jnp.argsort(jnp.where(features != 0.0, 0, 1), axis=-1)[:, :2]
    values = jnp.take_along_axis(features, columns, axis=-1)
    valid = jnp.take_along_axis(features != 0.0, columns, axis=-1)
    return SparseFeatures(values, columns, feature_count=features.shape[-1], valid=valid)


def _assert_model_gradients(model, inputs):
    input_gradient = jax.grad(lambda value: jnp.sum(jnp.square(model(value))))(inputs)
    coefficient_gradient = jax.grad(
        lambda value: jnp.sum(
            jnp.square(eqx.tree_at(lambda item: item.coefficients, model, value)(inputs))
        )
    )(model.coefficients)
    intercept_gradient = jax.grad(
        lambda value: jnp.sum(
            jnp.square(eqx.tree_at(lambda item: item.intercept, model, value)(inputs))
        )
    )(model.intercept)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.all(jnp.isfinite(coefficient_gradient))
    assert jnp.all(jnp.isfinite(intercept_gradient))


def test_ols_ridge_tikhonov_dense_case_multioutput_masks_weights_and_gradients():
    features, targets = _regression_data()
    cases = jnp.stack((features, 0.7 * features + 0.1), axis=0)
    case_targets = jnp.stack(
        (
            targets,
            (0.7 * features + 0.1) @ jnp.array([[1.5, -0.3], [-0.4, 1.2], [0.7, 0.5]])
            + jnp.array([0.2, -0.1]),
        ),
        axis=0,
    )
    feature_mask = jnp.ones_like(cases, dtype=bool).at[1, 2, 1].set(False)
    target_mask = jnp.ones_like(case_targets, dtype=bool).at[0, 4, 1].set(False)
    weights = jnp.linspace(0.5, 2.0, features.shape[0])

    recipes_and_types = (
        (OLSRecipe(), OLSModel),
        (RidgeRecipe(1e-3), RidgeModel),
        (TikhonovRecipe(jnp.array([1.0, 2.0, 0.5]), strength=1e-3), TikhonovModel),
    )
    for recipe, model_type in recipes_and_types:
        result = recipe.fit_batch(
            MLBatch(
                cases,
                case_targets,
                feature_mask=feature_mask,
                target_mask=target_mask,
                sample_weight=weights,
            )
        )
        model = result.as_trainable()
        assert isinstance(model, model_type)
        assert model(cases).shape == case_targets.shape
        assert result.diagnostics.rank.shape == (2,)
        assert result.diagnostics.condition.shape == (2,)
        assert jax.jit(model)(cases).shape == case_targets.shape
        _assert_model_gradients(model, cases)

    base = RidgeRecipe(0.1)

    def fit_loss(x, y, sample_weight, alpha):
        recipe = eqx.tree_at(lambda item: item.alpha, base, alpha)
        model = recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3))(
        features, targets, weights, base.alpha
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


def test_direct_solvers_match_weighted_normal_equations_and_tikhonov_hypergradients():
    features, multioutput = _regression_data()
    targets = multioutput[:, 0]
    weights = jnp.linspace(0.5, 1.5, features.shape[0])
    augmented = jnp.concatenate((features, jnp.ones((features.shape[0], 1))), axis=-1)
    weighted_gram = augmented.T @ (weights[:, None] * augmented)
    weighted_rhs = augmented.T @ (weights * targets)
    operator = jnp.array([[1.0, 0.2, 0.0], [0.0, 2.0, 0.1]])
    penalties_and_recipes = (
        (jnp.zeros((4, 4)), OLSRecipe()),
        (
            jnp.diag(jnp.array([0.2, 0.2, 0.2, 0.0])),
            RidgeRecipe(0.2),
        ),
        (
            jnp.diag(jnp.array([0.2, 0.2, 0.2, 0.2])),
            RidgeRecipe(0.2, regularize_intercept=True),
        ),
        (
            jnp.pad(0.2 * operator.T @ operator, ((0, 1), (0, 1))),
            TikhonovRecipe(operator, strength=0.2),
        ),
    )
    for penalty, recipe in penalties_and_recipes:
        expected = jnp.linalg.pinv(weighted_gram + penalty) @ weighted_rhs
        model = recipe.fit_batch(
            MLBatch(features, targets, sample_weight=weights)
        ).as_trainable()
        assert jnp.allclose(model.coefficients, expected[:-1], rtol=1e-4, atol=1e-5)
        assert jnp.allclose(model.intercept, expected[-1], rtol=1e-4, atol=1e-5)

    base = TikhonovRecipe(jnp.eye(features.shape[-1]), strength=0.1)

    def fit_loss(x, y, sample_weight, penalty, strength):
        recipe = eqx.tree_at(
            lambda item: (item.penalty, item.strength),
            base,
            (penalty, strength),
        )
        fitted = recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(fitted(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3, 4))(
        features, targets, weights, base.penalty, base.strength
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


@pytest.mark.parametrize(
    "dtype",
    (
        jnp.float32,
        pytest.param(
            jnp.float64,
            marks=pytest.mark.skipif(
                not bool(jax.config.read("jax_enable_x64")),
                reason="JAX x64 is disabled",
            ),
        ),
    ),
)
def test_linear_families_preserve_real_precision_and_intercept_policy(dtype):
    features, targets = _regression_data()
    features = features.astype(dtype)
    targets = targets.astype(dtype)
    for recipe in (
        OLSRecipe(fit_intercept=False),
        RidgeRecipe(jnp.asarray(0.1, dtype=dtype), fit_intercept=False),
        LassoRecipe(
            jnp.asarray(0.1, dtype=dtype),
            fit_intercept=False,
            max_iterations=2,
            tolerance=1e6,
        ),
    ):
        result = recipe.fit_batch(MLBatch(features, targets))
        model = result.as_trainable()
        assert result.valid.shape == result.status.shape == ()
        assert all(
            value.shape == ()
            for value in (
                result.diagnostics.valid,
                result.diagnostics.status,
                result.diagnostics.objective,
                result.diagnostics.iterations,
                result.diagnostics.effective_samples,
                result.diagnostics.rank,
                result.diagnostics.condition,
            )
        )
        assert model.coefficients.dtype == dtype
        assert model.intercept.dtype == dtype
        assert jnp.all(model.intercept == 0.0)


def test_direct_solvers_operator_sparse_complex_and_failure_diagnostics():
    features, targets = _regression_data()
    sparse = _operator_sparse(features)
    for recipe, model_type in (
        (OLSRecipe(), OLSModel),
        (RidgeRecipe(0.1), RidgeModel),
        (TikhonovRecipe(jnp.eye(3), strength=0.1), TikhonovModel),
    ):
        result = recipe.fit_batch(MLBatch(sparse, targets))
        model = result.as_trainable()
        assert isinstance(model, model_type)
        assert model(sparse).shape == targets.shape
        assert jnp.all(jnp.isfinite(model(sparse)))

    complex_features = features.astype(jnp.complex64) * (1.0 + 0.2j)
    complex_targets = complex_features @ jnp.array([0.7 - 0.1j, 0.2j, -0.3 + 0.4j])
    complex_model = (
        RidgeRecipe(0.1)
        .fit_batch(MLBatch(complex_features, complex_targets))
        .as_trainable()
    )
    assert jnp.iscomplexobj(complex_model.coefficients)
    assert jnp.all(jnp.isfinite(complex_model(complex_features)))
    complex_sparse_model = (
        LassoRecipe(0.01, max_iterations=3, tolerance=1e6)
        .fit_batch(MLBatch(complex_features, complex_targets))
        .as_trainable()
    )
    assert jnp.iscomplexobj(complex_sparse_model.coefficients)
    assert jnp.all(jnp.isfinite(complex_sparse_model(complex_features)))

    empty = OLSRecipe().fit_batch(
        MLBatch(
            features, targets, sample_mask=jnp.zeros((features.shape[0],), dtype=bool)
        )
    )
    assert empty.status == ML_INSUFFICIENT_DATA
    infeasible = OLSRecipe().fit_batch(
        MLBatch(features, targets, sample_weight=-jnp.ones((features.shape[0],)))
    )
    assert infeasible.status == ML_INFEASIBLE
    rank_deficient = OLSRecipe().fit_batch(
        MLBatch(jnp.stack((features[:, 0], features[:, 0]), axis=-1), targets[:, 0])
    )
    assert rank_deficient.status == ML_RANK_DEFICIENT
    ill_conditioned = RidgeRecipe(1e-12).fit_batch(
        MLBatch(
            jnp.stack(
                (
                    features[:, 0],
                    features[:, 0] + 1e-7 * features[:, 1],
                ),
                axis=-1,
            ),
            targets[:, 0],
        )
    )
    assert jnp.isfinite(ill_conditioned.diagnostics.condition)
    assert ill_conditioned.diagnostics.condition > 1.0


def test_sparse_fits_preserve_duplicate_entries_as_additive_feature_mass():
    features, targets = _regression_data()
    duplicate_sparse = SparseFeatures(
        jnp.stack(
            (
                0.4 * features[:, 0],
                0.6 * features[:, 0],
                features[:, 1],
                features[:, 2],
            ),
            axis=-1,
        ),
        jnp.broadcast_to(
            jnp.array([0, 0, 1, 2], dtype=jnp.int32),
            (features.shape[0], 4),
        ),
        feature_count=features.shape[-1],
    )
    for recipe in (
        RidgeRecipe(0.1),
        LassoRecipe(0.1, learning_rate=1e-3, max_iterations=3, tolerance=1e6),
    ):
        dense_model = recipe.fit_batch(MLBatch(features, targets)).as_trainable()
        sparse_model = recipe.fit_batch(MLBatch(duplicate_sparse, targets)).as_trainable()
        assert jnp.allclose(
            sparse_model(duplicate_sparse),
            dense_model(features),
            rtol=1e-4,
            atol=1e-5,
        )


def test_lasso_one_step_matches_weighted_proximal_gradient_equation():
    model = (
        LassoRecipe(
            0.5,
            learning_rate=0.1,
            fit_intercept=False,
            max_iterations=1,
            tolerance=1e6,
        )
        .fit_batch(
            MLBatch(
                jnp.array([[2.0]]),
                jnp.array([3.0]),
                sample_weight=jnp.array([2.0]),
            )
        )
        .as_trainable()
    )
    # Gradient is 2 * 2 * (0 - 3) = -12; soft-threshold(1.2, 0.05) = 1.15.
    assert jnp.allclose(model.coefficients, jnp.array([1.15]))
    assert jnp.all(model.intercept == 0.0)


@pytest.mark.parametrize(
    ("recipe", "model_type", "replace_strength", "strength"),
    (
        (
            LassoRecipe(0.05, max_iterations=4, tolerance=1e6),
            LassoModel,
            lambda recipe, value: eqx.tree_at(lambda item: item.alpha, recipe, value),
            jnp.asarray(0.05),
        ),
        (
            ElasticNetRecipe(0.05, max_iterations=4, tolerance=1e6),
            ElasticNetModel,
            lambda recipe, value: eqx.tree_at(
                lambda item: item.l1_strength, recipe, value
            ),
            jnp.asarray(0.025),
        ),
        (
            GroupLassoRecipe((0, 0, 1), alpha=0.05, max_iterations=4, tolerance=1e6),
            GroupLassoModel,
            lambda recipe, value: eqx.tree_at(lambda item: item.alpha, recipe, value),
            jnp.asarray(0.05),
        ),
        (
            SparseGroupLassoRecipe(
                (0, 0, 1), alpha=0.05, max_iterations=4, tolerance=1e6
            ),
            SparseGroupLassoModel,
            lambda recipe, value: eqx.tree_at(
                lambda item: item.l1_strength, recipe, value
            ),
            jnp.asarray(0.025),
        ),
    ),
)
def test_sparse_penalty_families_dense_operator_sparse_jit_vmap_and_fit_gradients(
    recipe, model_type, replace_strength, strength
):
    features, targets = _regression_data()
    weights = jnp.linspace(0.5, 1.5, features.shape[0])
    dense_result = recipe.fit_batch(MLBatch(features, targets, sample_weight=weights))
    dense_model = dense_result.as_trainable()
    assert isinstance(dense_model, model_type)
    assert dense_model(features).shape == targets.shape
    assert jax.jit(dense_model)(features).shape == targets.shape
    assert jax.vmap(dense_model)(features).shape == targets.shape
    _assert_model_gradients(dense_model, features)
    sparse_model = recipe.fit_batch(
        MLBatch(_operator_sparse(features), targets, sample_weight=weights)
    ).as_trainable()
    assert sparse_model(_operator_sparse(features)).shape == targets.shape

    def fit_loss(x, y, sample_weight, hyperparameter):
        fitted = (
            replace_strength(recipe, hyperparameter)
            .fit_batch(MLBatch(x, y, sample_weight=sample_weight))
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3))(
        features, targets, weights, strength
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


def test_sparse_penalty_nonconvergence_and_group_capacity_fail_closed():
    features, targets = _regression_data()
    result = LassoRecipe(0.1, max_iterations=1, tolerance=0.0).fit_batch(
        MLBatch(features, targets)
    )
    assert result.status == ML_NONCONVERGED
    with pytest.raises(ValueError, match="one group id per feature"):
        GroupLassoRecipe((0, 1), max_iterations=2).fit_batch(MLBatch(features, targets))
