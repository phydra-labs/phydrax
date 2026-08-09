#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import (
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_RANK_DEFICIENT,
    MLBatch,
    SparseFeatures,
    TargetSchema,
)
from phydrax.ml.discriminant import (
    LinearDiscriminantModel,
    LinearDiscriminantRecipe,
    QuadraticDiscriminantModel,
    QuadraticDiscriminantRecipe,
    RegularizedDiscriminantRecipe,
    ShrinkageDiscriminantRecipe,
)


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


def _sparse(values):
    columns = jnp.broadcast_to(jnp.arange(values.shape[-1]), values.shape)
    return SparseFeatures(values, columns, feature_count=values.shape[-1])


def _case_batch():
    features = jnp.stack((_FEATURES, 1.2 * _FEATURES + jnp.array([0.3, -0.2])))
    targets = jnp.stack((_TARGETS, _TARGETS))
    feature_mask = jnp.ones_like(features, dtype=bool).at[:, 3, 0].set(False)
    target_mask = jnp.ones_like(targets, dtype=bool).at[:, 7].set(False)
    return MLBatch(
        features,
        targets,
        feature_mask=feature_mask,
        target_mask=target_mask,
        sample_mask=jnp.arange(12) != 11,
        sample_weight=jnp.linspace(0.5, 1.6, 12),
        measure_weight=jnp.linspace(1.4, 0.7, 12),
        target_schema=_SCHEMA,
    )


@pytest.mark.parametrize(
    ("recipe", "model_type", "method"),
    [
        (
            LinearDiscriminantRecipe(weight_policy="product"),
            LinearDiscriminantModel,
            "weighted-lda",
        ),
        (
            QuadraticDiscriminantRecipe(weight_policy="product"),
            QuadraticDiscriminantModel,
            "weighted-qda",
        ),
        (
            ShrinkageDiscriminantRecipe(shrinkage=0.2, weight_policy="product"),
            LinearDiscriminantModel,
            "shrinkage-lda",
        ),
        (
            RegularizedDiscriminantRecipe(regularization=0.05, weight_policy="product"),
            QuadraticDiscriminantModel,
            "regularized-qda",
        ),
    ],
)
def test_discriminant_families_preserve_case_label_probability_and_weight_contracts(
    recipe, model_type, method
):
    batch = _case_batch()
    result = recipe.fit_batch(batch)
    model = result.as_trainable()
    probability = result.model(batch.dense_features())

    assert isinstance(model, model_type)
    assert jnp.all(result.valid)
    assert result.method == method
    assert result.diagnostics.method == method
    assert probability.shape == (2, 12, 3)
    assert jnp.allclose(jnp.sum(probability, axis=-1), 1.0, atol=1e-6)
    assert model.predict(batch.dense_features()).shape == (2, 12)
    assert model.target_schema.class_labels == _SCHEMA.class_labels
    assert jnp.array_equal(model.labels, jnp.arange(3))
    assert result.diagnostics.class_mass.shape == (2, 3)
    assert result.diagnostics.effective_samples.shape == (2,)
    assert recipe.weight_policy == "product"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"
    assert result.gradient_contract.fit_targets == "none"
    assert set(result.gradient_contract.nondifferentiable_outputs) == {
        "predict",
        "predict_indices",
    }


def test_discriminant_dense_complex_and_new_sample_execution_reject_sparse_input():
    recipe = LinearDiscriminantRecipe(num_classes=3, regularization=0.05)
    dense = recipe.fit_batch(MLBatch(_FEATURES, _TARGETS)).as_trainable()
    with pytest.raises(TypeError, match="requires dense features"):
        recipe.fit_batch(MLBatch(_sparse(_FEATURES), _TARGETS))
    complex_features = _FEATURES.astype(jnp.complex64) * (1.0 + 0.2j)
    complex_model = recipe.fit_batch(MLBatch(complex_features, _TARGETS)).as_trainable()
    probes = jnp.array([[-2.1, -2.0], [2.0, -2.0], [0.0, 2.1]])

    assert jnp.all(jnp.isfinite(complex_model(complex_features[:3])))
    assert jax.jit(dense)(probes).shape == (3, 3)
    assert jax.vmap(dense)(probes).shape == (3, 3)
    with pytest.raises(ValueError, match="Expected 2 input features"):
        dense(jnp.ones((4, 3)))


@pytest.mark.parametrize(
    ("recipe", "replace_hyperparameter", "initial"),
    [
        (
            LinearDiscriminantRecipe(num_classes=3, regularization=0.08),
            lambda item, value: eqx.tree_at(
                lambda fitted: fitted.regularization, item, value
            ),
            jnp.array(0.08),
        ),
        (
            QuadraticDiscriminantRecipe(num_classes=3, regularization=0.08),
            lambda item, value: eqx.tree_at(
                lambda fitted: fitted.regularization, item, value
            ),
            jnp.array(0.08),
        ),
        (
            ShrinkageDiscriminantRecipe(
                num_classes=3, shrinkage=0.2, regularization=0.04
            ),
            lambda item, value: eqx.tree_at(lambda fitted: fitted.shrinkage, item, value),
            jnp.array(0.2),
        ),
        (
            RegularizedDiscriminantRecipe(num_classes=3, regularization=0.08),
            lambda item, value: eqx.tree_at(
                lambda fitted: fitted.regularization, item, value
            ),
            jnp.array(0.08),
        ),
    ],
)
def test_every_discriminant_fit_family_has_declared_feature_weight_and_hyperparameter_gradients(
    recipe, replace_hyperparameter, initial
):
    weights = jnp.linspace(0.8, 1.4, 12)
    probe = jnp.array([0.4, -0.3])

    def loss(features, sample_weight, hyperparameter):
        configured = replace_hyperparameter(recipe, hyperparameter)
        model = configured.fit_batch(
            MLBatch(features, _TARGETS, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model.decision_function(probe)))

    gradients = jax.grad(loss, argnums=(0, 1, 2))(_FEATURES, weights, initial)
    model = recipe.fit_batch(
        MLBatch(_FEATURES, _TARGETS, sample_weight=weights)
    ).as_trainable()
    input_gradient = jax.grad(
        lambda point: jnp.sum(jnp.square(model.decision_function(point)))
    )(probe)
    if isinstance(model, LinearDiscriminantModel):
        parameter_gradient = jax.grad(
            lambda coefficients: jnp.sum(
                jnp.square(probe @ coefficients.T + model.intercepts)
            )
        )(model.coefficients)
    else:
        parameter_gradient = jax.grad(
            lambda means: jnp.sum(
                jnp.square(
                    model.log_priors
                    - 0.5
                    * (
                        model.log_determinants
                        + jnp.einsum(
                            "cf,cfg,cg->c",
                            probe - means,
                            model.precisions,
                            probe - means,
                        )
                    )
                )
            )
        )(model.means)

    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.all(jnp.isfinite(parameter_gradient))


def test_discriminant_failures_report_empty_single_class_nonfinite_and_rank_statuses():
    empty = LinearDiscriminantRecipe(num_classes=3, regularization=0.1).fit_batch(
        MLBatch(
            _FEATURES,
            _TARGETS,
            sample_mask=jnp.zeros(12, dtype=bool),
        )
    )
    single_class = LinearDiscriminantRecipe(num_classes=3, regularization=0.1).fit_batch(
        MLBatch(_FEATURES, jnp.zeros(12, dtype=jnp.int32))
    )
    nonfinite_features = _FEATURES.at[0, 0].set(jnp.nan)
    nonfinite = LinearDiscriminantRecipe(num_classes=3, regularization=0.1).fit_batch(
        MLBatch(nonfinite_features, _TARGETS)
    )
    collinear = jnp.stack((jnp.arange(12.0), 2.0 * jnp.arange(12.0)), axis=-1)
    rank_deficient = LinearDiscriminantRecipe(num_classes=3).fit_batch(
        MLBatch(collinear, _TARGETS)
    )
    resolved = ShrinkageDiscriminantRecipe(num_classes=3, shrinkage=0.3).fit_batch(
        MLBatch(collinear, _TARGETS)
    )
    quadratic_rank_deficient = QuadraticDiscriminantRecipe(num_classes=3).fit_batch(
        MLBatch(collinear, _TARGETS)
    )
    quadratic_resolved = RegularizedDiscriminantRecipe(
        num_classes=3, regularization=0.05
    ).fit_batch(MLBatch(collinear, _TARGETS))

    assert int(empty.status) == ML_INSUFFICIENT_DATA
    assert int(single_class.status) == ML_INSUFFICIENT_DATA
    assert int(nonfinite.status) == ML_NONFINITE
    assert int(rank_deficient.status) == ML_RANK_DEFICIENT
    assert bool(rank_deficient.diagnostics.raw_singular)
    assert int(quadratic_rank_deficient.status) == ML_RANK_DEFICIENT
    assert jnp.all(quadratic_rank_deficient.diagnostics.raw_singular)
    assert bool(resolved.valid)
    assert bool(resolved.diagnostics.raw_singular)
    assert bool(quadratic_resolved.valid)
    assert jnp.all(quadratic_resolved.diagnostics.raw_singular)
    assert jnp.all(jnp.isfinite(resolved.model(collinear)))


def test_discriminant_rejects_invalid_schema_priors_and_target_axes():
    with pytest.raises(ValueError, match="sum to one"):
        LinearDiscriminantRecipe(num_classes=3, priors=(0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="conflicts"):
        LinearDiscriminantRecipe(num_classes=2).fit_batch(
            MLBatch(_FEATURES, _TARGETS, target_schema=_SCHEMA)
        )
    with pytest.raises(ValueError, match="scalar class label"):
        QuadraticDiscriminantRecipe(num_classes=3).fit_batch(
            MLBatch(_FEATURES, jax.nn.one_hot(_TARGETS, 3))
        )
