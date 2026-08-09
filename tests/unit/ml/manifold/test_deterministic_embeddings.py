#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import ML_INFEASIBLE, ML_NONCONVERGED, MLBatch
from phydrax.ml.manifold import (
    build_neighbor_graph,
    IsomapModel,
    IsomapRecipe,
    LocallyLinearEmbeddingModel,
    LocallyLinearEmbeddingRecipe,
    MultidimensionalScalingModel,
    MultidimensionalScalingRecipe,
    SpectralEmbeddingModel,
    SpectralEmbeddingRecipe,
)


def _features():
    return jnp.array(
        [
            [-2.3, -0.4, 0.8],
            [-1.6, 0.7, -0.2],
            [-0.8, -0.9, 0.5],
            [-0.1, 0.2, -0.7],
            [0.7, 1.1, 0.3],
            [1.4, -0.5, 1.0],
            [2.0, 0.6, -0.4],
            [2.8, -0.2, 0.1],
        ]
    )


def _transductive_loss(model):
    embedding = model.training_embedding
    coefficients = jnp.arange(1.0, embedding.shape[-2] + 1.0)
    return jnp.sum(coefficients[:, None] * jnp.square(jnp.abs(embedding)))


def _assert_finite_nonzero(array):
    assert jnp.all(jnp.isfinite(array))
    assert jnp.any(jnp.abs(array) > 1e-8)


@pytest.mark.parametrize("variant", ["standard", "modified", "hessian", "ltsa"])
def test_every_lle_variant_has_declared_schema_and_exact_transform_support(variant):
    recipe = LocallyLinearEmbeddingRecipe(1, n_neighbors=3, variant=variant)
    result = recipe.fit_batch(MLBatch(_features()))
    model = result.as_trainable()

    assert isinstance(model, LocallyLinearEmbeddingModel)
    assert model.training_features.shape == (8, 3)
    assert model.training_embedding.shape == (8, 1)
    assert result.diagnostics.eigenvalues.shape == (1,)
    assert result.diagnostics.method == f"lle-{variant}"
    assert result.gradient_contract.fit_mode == "spectral"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"
    assert result.gradient_contract.fit_targets == "none"

    if variant in ("standard", "modified"):
        assert model(jnp.array([0.2, -0.1, 0.4])).shape == (1,)
        assert result.model(jnp.zeros((3, 3))).shape == (3, 1)
        assert jax.jit(model)(jnp.zeros((3, 3))).shape == (3, 1)
        assert jax.vmap(model)(jnp.zeros((3, 3))).shape == (3, 1)
        assert result.gradient_contract.prediction_inputs == "conditional"
        assert result.gradient_contract.prediction_parameters == "conditional"
    else:
        assert result.gradient_contract.prediction_inputs == "none"
        assert result.gradient_contract.prediction_parameters == "none"
        with pytest.raises(ValueError, match="not mathematically defined"):
            model(jnp.array([0.2, -0.1, 0.4]))


def test_lle_preserves_case_sample_feature_and_target_axes_and_statistical_weight_policy():
    base = _features()
    features = jnp.stack((base, base * jnp.array([1.2, 0.8, 1.1])), axis=0)
    targets = jnp.stack(
        (jnp.arange(16.0).reshape(2, 8), -jnp.arange(16.0).reshape(2, 8)), axis=-1
    )
    feature_mask = jnp.ones_like(features, dtype=bool).at[:, 3, 2].set(False)
    sample_mask = jnp.array([True, True, True, True, True, True, True, False])
    sample_weight = jnp.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 7.0])
    recipe = LocallyLinearEmbeddingRecipe(1, n_neighbors=2)
    common = dict(
        feature_mask=feature_mask,
        target_mask=jnp.ones_like(targets, dtype=bool).at[:, 0, 1].set(False),
        sample_mask=sample_mask,
        sample_weight=sample_weight,
    )

    first = recipe.fit_batch(MLBatch(features, targets, measure_weight=1.0, **common))
    second = recipe.fit_batch(MLBatch(features, targets, measure_weight=99.0, **common))
    model = first.as_trainable()
    queries = features[:, :3] + 0.05

    assert model.case_shape == (2,)
    assert model.training_embedding.shape == (2, 8, 1)
    assert model(queries).shape == (2, 3, 1)
    assert model(jnp.array([0.1, 0.2, -0.3])).shape == (2, 1)
    assert first.diagnostics.effective_samples.shape == (2,)
    assert jnp.array_equal(first.diagnostics.effective_samples, jnp.array([6, 6]))
    assert jnp.all(model.training_embedding[:, 3] == 0.0)
    assert jnp.all(model.training_embedding[:, 7] == 0.0)
    assert jnp.allclose(
        jnp.abs(model.training_embedding),
        jnp.abs(second.as_trainable().training_embedding),
    )


def test_spectral_embedding_is_jittable_vmappable_and_conditionally_differentiable():
    features = _features()
    weights = jnp.array([1.0, 1.4, 0.8, 1.7, 1.2, 0.9, 1.5, 1.1])
    recipe = SpectralEmbeddingRecipe(2, n_neighbors=3, bandwidth=1.4)
    result = recipe.fit_batch(MLBatch(features, sample_weight=weights))
    model = result.as_trainable()
    points = jnp.array([[0.1, -0.2, 0.3], [1.1, 0.4, -0.5]])

    assert isinstance(model, SpectralEmbeddingModel)
    assert model.eigenvectors.shape == (8, 2)
    assert model.eigenvalues.shape == (2,)
    assert model.degrees.shape == (8,)
    assert jax.jit(model)(points).shape == (2, 2)
    assert jax.vmap(model)(points).shape == (2, 2)
    _assert_finite_nonzero(
        jax.grad(lambda point: jnp.sum(jnp.square(model(point))))(points[0])
    )

    parameter_gradient = eqx.filter_grad(
        lambda candidate: jnp.sum(jnp.square(candidate(points[0])))
    )(model)
    parameter_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]
    assert parameter_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in parameter_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 1e-8) for leaf in parameter_leaves)
    contract = result.gradient_contract
    assert contract.prediction_inputs == "conditional"
    assert contract.prediction_parameters == "conditional"
    assert contract.fit_features == "conditional"
    assert contract.fit_weights == "conditional"
    assert contract.fit_hyperparameters == "conditional"
    assert contract.fit_targets == "none"
    assert contract.fit_mode == "spectral"


def test_classical_mds_preserves_planar_distances_and_smacof_rejects_transform():
    planar = _features()[:, :2]
    classical = MultidimensionalScalingRecipe(2, method="classical").fit_batch(
        MLBatch(planar)
    )
    classical_model = classical.as_trainable()
    embedded = classical_model.training_embedding
    original_distances = jnp.linalg.norm(planar[:, None] - planar[None, :], axis=-1)
    embedded_distances = jnp.linalg.norm(embedded[:, None] - embedded[None, :], axis=-1)

    assert isinstance(classical_model, MultidimensionalScalingModel)
    assert embedded.shape == (8, 2)
    assert jnp.allclose(original_distances, embedded_distances, atol=2e-4)
    assert jax.jit(classical_model)(planar[:3]).shape == (3, 2)
    assert jax.vmap(classical_model)(planar[:3]).shape == (3, 2)
    assert classical.gradient_contract.prediction_inputs == "smooth"
    assert classical.gradient_contract.prediction_parameters == "smooth"
    assert classical.gradient_contract.fit_mode == "spectral"

    smacof = MultidimensionalScalingRecipe(
        2, method="smacof", iterations=3, tolerance=1e6
    ).fit_batch(MLBatch(planar))
    smacof_model = smacof.as_trainable()
    assert smacof_model.training_embedding.shape == (8, 2)
    assert smacof.gradient_contract.prediction_inputs == "none"
    assert smacof.gradient_contract.prediction_parameters == "none"
    assert smacof.gradient_contract.fit_mode == "unrolled"
    with pytest.raises(ValueError, match="transductive"):
        smacof_model(planar[0])
    with pytest.raises(TypeError):
        MultidimensionalScalingRecipe(2, metric="precomputed")
    nonconverged = MultidimensionalScalingRecipe(
        1, method="smacof", iterations=1, tolerance=1e-30
    ).fit_batch(MLBatch(planar))
    assert not nonconverged.valid
    assert nonconverged.status == ML_NONCONVERGED


def test_isomap_exposes_geodesic_invariants_capacity_and_connectivity_status():
    features = _features()
    result = IsomapRecipe(2, n_neighbors=3, max_samples=8).fit_batch(MLBatch(features))
    model = result.as_trainable()

    assert isinstance(model, IsomapModel)
    assert model.training_embedding.shape == (8, 2)
    assert model.geodesic_distances.shape == (8, 8)
    assert jnp.allclose(model.geodesic_distances, model.geodesic_distances.T)
    assert jnp.allclose(jnp.diag(model.geodesic_distances), 0.0)
    assert jnp.all(jnp.isfinite(model.geodesic_distances))
    assert model(features[:3]).shape == (3, 2)
    assert jax.jit(model)(features[:3]).shape == (3, 2)
    assert jax.vmap(model)(features[:3]).shape == (3, 2)
    assert result.diagnostics.connected_components == 1
    assert result.gradient_contract.prediction_inputs == "conditional"
    assert result.gradient_contract.prediction_parameters == "conditional"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"

    with pytest.raises(ValueError, match="capacity exceeded"):
        IsomapRecipe(1, n_neighbors=2, max_samples=7).fit_batch(MLBatch(features))

    disconnected = jnp.array([[0.0], [1.0], [10.0], [11.0]])
    invalid = IsomapRecipe(1, n_neighbors=1).fit_batch(MLBatch(disconnected))
    assert not invalid.valid
    assert invalid.status == ML_INFEASIBLE
    assert invalid.diagnostics.connected_components == 2


def test_complex_manifold_coordinates_use_hermitian_geometry():
    real = _features()
    complex_features = real + 0.2j * jnp.flip(real, axis=-1)
    graph = build_neighbor_graph(
        complex_features, jnp.ones((8,), dtype=bool), n_neighbors=3
    )
    result = LocallyLinearEmbeddingRecipe(1, n_neighbors=3).fit_batch(
        MLBatch(complex_features)
    )
    transformed = result.model(complex_features[:2] + 0.01j)

    assert jnp.isrealobj(graph.distances)
    assert jnp.all(jnp.isfinite(graph.distances))
    assert transformed.shape == (2, 1)
    assert jnp.all(jnp.isfinite(jnp.real(transformed)))
    assert jnp.all(jnp.isfinite(jnp.imag(transformed)))


@pytest.mark.parametrize(
    "recipe",
    [
        LocallyLinearEmbeddingRecipe(1, n_neighbors=3, variant="standard"),
        LocallyLinearEmbeddingRecipe(1, n_neighbors=3, variant="modified"),
        LocallyLinearEmbeddingRecipe(1, n_neighbors=3, variant="hessian"),
        LocallyLinearEmbeddingRecipe(1, n_neighbors=3, variant="ltsa"),
        SpectralEmbeddingRecipe(1, n_neighbors=3, bandwidth=1.3),
        MultidimensionalScalingRecipe(1, method="classical"),
        MultidimensionalScalingRecipe(1, method="smacof", iterations=3, tolerance=1e6),
        IsomapRecipe(1, n_neighbors=3),
    ],
)
def test_deterministic_manifold_fit_feature_and_weight_gradients_match_contract(recipe):
    features = _features()
    weights = jnp.array([1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.05])

    def feature_loss(value):
        fitted = recipe.fit_batch(MLBatch(value, sample_weight=weights)).as_trainable()
        if isinstance(fitted, SpectralEmbeddingModel):
            embedding = fitted.eigenvectors
            coefficients = jnp.arange(1.0, embedding.shape[-2] + 1.0)
            return jnp.sum(coefficients[:, None] * jnp.square(jnp.abs(embedding)))
        return _transductive_loss(fitted)

    def weight_loss(value):
        fitted = recipe.fit_batch(MLBatch(features, sample_weight=value)).as_trainable()
        if isinstance(fitted, SpectralEmbeddingModel):
            embedding = fitted.eigenvectors
            coefficients = jnp.arange(1.0, embedding.shape[-2] + 1.0)
            return jnp.sum(coefficients[:, None] * jnp.square(jnp.abs(embedding)))
        return _transductive_loss(fitted)

    feature_gradient = jax.grad(feature_loss)(features)
    weight_gradient = jax.grad(weight_loss)(weights)
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))
    contract = recipe.fit_batch(MLBatch(features)).gradient_contract
    assert contract.fit_features == "conditional"
    assert contract.fit_weights == "conditional"
    assert contract.fit_hyperparameters == "conditional"
