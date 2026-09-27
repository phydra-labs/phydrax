#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax import DerivativeRoute, DerivativeSurface, GradientLevel
from phydrax.ml import ML_INSUFFICIENT_DATA, MLBatch
from phydrax.ml.manifold import (
    FuzzyGraphEmbeddingModel,
    FuzzyGraphEmbeddingRecipe,
    TSNEModel,
    TSNERecipe,
)


def _features() -> Any:
    return jnp.array(
        [
            [-2.0, -0.3, 0.4],
            [-1.3, 0.8, -0.5],
            [-0.6, -0.7, 0.9],
            [0.1, 0.3, -0.8],
            [0.9, 1.0, 0.2],
            [1.7, -0.4, 0.7],
            [2.5, 0.5, -0.2],
        ]
    )


def _embedding_loss(model: Any) -> Any:
    coefficients = jnp.arange(1.0, model.embedding.shape[-2] + 1.0)
    return jnp.sum(coefficients[:, None] * jnp.square(model.embedding))


def test_stochastic_embeddings_scenario_1() -> None:
    features = _features()
    recipe = TSNERecipe(
        2,
        perplexity=2.0,
        iterations=4,
        learning_rate=0.2,
        tolerance=1e6,
        max_samples=7,
    )

    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(MLBatch(features))

    key = jax.random.key(13)
    first = recipe.fit_batch(MLBatch(features), key=key)
    repeated = recipe.fit_batch(MLBatch(features), key=key)
    changed = recipe.fit_batch(MLBatch(features), key=jax.random.key(14))
    model = first.as_trainable()

    assert isinstance(model, TSNEModel)
    assert model.embedding.shape == (7, 2)
    assert model.training_features.shape == features.shape
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(model.embedding, repeated.as_trainable().embedding)
    # ty: ignore[unresolved-attribute]
    assert not jnp.allclose(model.embedding, changed.as_trainable().embedding)
    assert first.diagnostics.eigenvalues.shape == (0,)
    assert first.diagnostics.iterations == 4
    assert first.diagnostics.minimum_degree == 6
    assert first.diagnostics.maximum_degree == 6
    assert first.derivative_contract.level(DerivativeSurface.INPUT) is GradientLevel.NONE
    assert (
        first.derivative_contract.level(DerivativeSurface.MODEL_PARAMETER)
        is GradientLevel.NONE
    )
    assert (
        first.derivative_contract.level(DerivativeSurface.FIT_FEATURES)
        is GradientLevel.CONDITIONAL
    )
    assert (
        first.derivative_contract.level(DerivativeSurface.FIT_WEIGHTS)
        is GradientLevel.CONDITIONAL
    )
    assert (
        first.derivative_contract.level(DerivativeSurface.FIT_HYPERPARAMETERS)
        is GradientLevel.CONDITIONAL
    )
    assert (
        first.derivative_contract.level(DerivativeSurface.FIT_TARGETS)
        is GradientLevel.NONE
    )
    assert first.derivative_contract.route is DerivativeRoute.UNROLLED
    with pytest.raises(ValueError, match="transductive"):
        model(features[0])
    with pytest.raises(ValueError, match="capacity exceeded"):
        TSNERecipe(
            1,
            perplexity=2.0,
            iterations=2,
            learning_rate=0.1,
            max_samples=6,
        ).fit_batch(MLBatch(features), key=key)
    base = _features()
    features = jnp.stack((base, 1.1 * base), axis=0)
    targets = jnp.stack(
        (jnp.sum(features, axis=-1), jnp.prod(features, axis=-1)), axis=-1
    )
    recipe = TSNERecipe(
        1,
        perplexity=2.0,
        iterations=3,
        learning_rate=0.1,
        tolerance=1e6,
    )
    key = jax.random.key(20)
    batch = MLBatch(
        features,
        targets,
        target_mask=jnp.ones_like(targets, dtype="bool").at[:, 0, 1].set(False),
        sample_weight=jnp.array([1.0, 1.3, 0.8, 1.1, 1.4, 0.9, 1.2]),
    )

    result = recipe.fit_batch(batch, key=key)
    compiled_embedding = jax.jit(
        lambda value, random_key: (
            # ty: ignore[unresolved-attribute]
            recipe.fit_batch(MLBatch(value, targets), key=random_key)
            .as_trainable()
            .embedding
        )
    )(features, key)

    # ty: ignore[unresolved-attribute]
    assert result.as_trainable().embedding.shape == (2, 7, 1)
    assert result.diagnostics.objective.shape == (2,)
    assert result.diagnostics.residual.shape == (2,)
    assert compiled_embedding.shape == (2, 7, 1)
    features = _features()
    weights = jnp.array([1.0, 1.3, 0.9, 1.5, 1.1, 0.8, 1.2])
    recipe = FuzzyGraphEmbeddingRecipe(
        2,
        n_neighbors=3,
        iterations=4,
        learning_rate=0.1,
        tolerance=1e6,
        max_samples=7,
    )

    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(MLBatch(features))

    key = jax.random.key(31)
    first = recipe.fit_batch(MLBatch(features, sample_weight=weights), key=key)
    repeated = recipe.fit_batch(MLBatch(features, sample_weight=weights), key=key)
    changed = recipe.fit_batch(
        MLBatch(features, sample_weight=weights), key=jax.random.key(32)
    )
    model = first.as_trainable()
    points = jnp.array([[0.2, -0.1, 0.4], [1.2, 0.3, -0.4]])

    assert isinstance(model, FuzzyGraphEmbeddingModel)
    assert model.embedding.shape == (7, 2)
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(model.embedding, repeated.as_trainable().embedding)
    # ty: ignore[unresolved-attribute]
    assert not jnp.allclose(model.embedding, changed.as_trainable().embedding)
    assert jax.jit(model)(points).shape == (2, 2)
    assert jax.vmap(model)(points).shape == (2, 2)
    input_gradient = jax.grad(lambda point: jnp.sum(jnp.square(model(point))))(points[0])
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.any(jnp.abs(input_gradient) > 1e-8)

    parameter_gradient = eqx.filter_grad(
        lambda candidate: jnp.sum(jnp.square(candidate(points[0])))
    )(model)
    leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(jnp.abs(leaf) > 1e-8) for leaf in leaves)

    contract = first.derivative_contract
    # The fuzzy barycentric transform jumps when the hard neighbor set changes.
    assert contract.level(DerivativeSurface.INPUT) is GradientLevel.NONE
    assert contract.level(DerivativeSurface.MODEL_PARAMETER) is (
        GradientLevel.ALMOST_EVERYWHERE
    )
    # ty: ignore[unresolved-attribute]
    assert contract.regularity.continuity == -1
    assert contract.level(DerivativeSurface.FIT_FEATURES) is GradientLevel.CONDITIONAL
    assert contract.level(DerivativeSurface.FIT_WEIGHTS) is GradientLevel.CONDITIONAL
    assert (
        contract.level(DerivativeSurface.FIT_HYPERPARAMETERS) is GradientLevel.CONDITIONAL
    )
    assert contract.level(DerivativeSurface.FIT_TARGETS) is GradientLevel.NONE
    assert contract.route is DerivativeRoute.UNROLLED

    with pytest.raises(ValueError, match="capacity exceeded"):
        FuzzyGraphEmbeddingRecipe(
            1,
            n_neighbors=2,
            iterations=2,
            max_samples=6,
        ).fit_batch(MLBatch(features), key=key)
    features = _features()
    mask = jnp.array([True, True, False, False, False, False, False])
    key = jax.random.key(51)

    tsne = TSNERecipe(
        1,
        perplexity=2.0,
        iterations=2,
        learning_rate=0.1,
        tolerance=1e6,
    ).fit_batch(MLBatch(features, sample_mask=mask), key=key)
    fuzzy = FuzzyGraphEmbeddingRecipe(
        1,
        n_neighbors=2,
        iterations=2,
        learning_rate=0.1,
        tolerance=1e6,
    ).fit_batch(MLBatch(features, sample_mask=mask), key=key)

    assert not tsne.valid
    assert tsne.status == ML_INSUFFICIENT_DATA
    assert not fuzzy.valid
    assert fuzzy.status == ML_INSUFFICIENT_DATA


def test_stochastic_manifold_fit_feature_and_weight_gradients_use_fixed_keys() -> None:
    for recipe in [
        TSNERecipe(
            1,
            perplexity=2.0,
            iterations=3,
            learning_rate=0.1,
            tolerance=1e6,
        ),
        FuzzyGraphEmbeddingRecipe(
            1,
            n_neighbors=3,
            iterations=3,
            learning_rate=0.1,
            tolerance=1e6,
        ),
    ]:
        features = _features()
        weights = jnp.array([1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3])
        key = jax.random.key(41)

        def feature_loss(value: Any) -> Any:
            model = recipe.fit_batch(
                MLBatch(value, sample_weight=weights), key=key
            ).as_trainable()
            return _embedding_loss(model)

        def weight_loss(value: Any) -> Any:
            model = recipe.fit_batch(
                MLBatch(features, sample_weight=value), key=key
            ).as_trainable()
            return _embedding_loss(model)

        feature_gradient = jax.grad(feature_loss)(features)
        weight_gradient = jax.grad(weight_loss)(weights)
        assert jnp.all(jnp.isfinite(feature_gradient))
        assert jnp.all(jnp.isfinite(weight_gradient))
