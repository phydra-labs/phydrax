#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.ml import MLBatch
from phydrax.ml.decomposition import IncrementalPCA, PCA, POD, TruncatedSVD


def _pivot_values(rows):
    indices = jnp.argmax(jnp.abs(rows), axis=-1)
    return jnp.take_along_axis(rows, indices[..., None], axis=-1)[..., 0]


def test_pca_preserves_case_sample_mask_weight_and_canonicalization_contracts():
    base = jnp.array(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, -1.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    )
    features = jnp.stack((base, 2.0 * base), axis=0)
    mask = jnp.ones_like(features, dtype=bool).at[:, 2, 2].set(False)
    batch = MLBatch(
        features,
        feature_mask=mask,
        sample_mask=jnp.array([True, True, False, True, True]),
        sample_weight=jnp.array([1.0, 2.0, 9.0, 2.0, 1.0]),
    )
    result = PCA(2, differentiate="basis").fit_batch(batch)
    model = result.as_trainable()

    assert model.offset.shape == (2, 3)
    assert model.transform(features).shape == (2, 5, 2)
    assert model.inverse_transform(model.transform(features)).shape == features.shape
    assert result.diagnostics.singular_values.shape == (2, 2)
    assert result.diagnostics.explained_energy.shape == (2, 2)
    assert result.diagnostics.numerical_rank.shape == (2,)
    assert result.diagnostics.weighted_orthogonality_error.shape == (2,)
    assert result.diagnostics.projector_gradient_supported.shape == (2,)
    assert result.diagnostics.basis_gradient_supported.shape == (2,)
    pivots = _pivot_values(model.weighted_components)
    assert jnp.allclose(jnp.imag(pivots), 0.0)
    assert jnp.all(jnp.real(pivots) >= 0.0)
    assert result.gradient_contract.fit_mode == "spectral"
    assert "basis representatives" in " ".join(result.gradient_contract.conditions)


def test_pca_projector_prediction_fit_feature_and_fit_weight_gradients_are_finite():
    features = jnp.array(
        [
            [-2.0, 0.2, 1.0],
            [-1.0, 1.1, 0.4],
            [0.1, -0.3, 0.8],
            [1.2, -1.0, -0.2],
            [2.1, 0.4, -1.1],
        ]
    )
    point = jnp.array([0.3, -0.4, 1.2])
    model = PCA(2).fit_batch(MLBatch(features)).as_trainable()

    prediction_gradient = jax.grad(lambda value: jnp.sum(jnp.square(model(value))))(point)

    def feature_loss(value):
        fitted = PCA(2).fit_batch(MLBatch(value)).as_trainable()
        return jnp.sum(jnp.square(fitted.project(point)))

    def weight_loss(weight):
        fitted = PCA(2).fit_batch(MLBatch(features, sample_weight=weight)).as_trainable()
        return jnp.sum(jnp.square(fitted.project(point)))

    feature_gradient = jax.grad(feature_loss)(features)
    weight_gradient = jax.grad(weight_loss)(jnp.arange(1.0, 6.0))
    assert jnp.all(jnp.isfinite(prediction_gradient))
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))


def test_truncated_svd_is_origin_anchored_jittable_and_vmappable():
    features = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    result = TruncatedSVD(2).fit_batch(MLBatch(features))
    model = result.as_trainable()
    points = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    assert jnp.allclose(model.offset, 0.0)
    assert jnp.allclose(model.project(features), features, atol=1e-5)
    assert jax.jit(model.transform)(points).shape == (2, 2)
    assert jax.vmap(model.transform)(points).shape == (2, 2)
    assert model.input_binding().batch_mode == "blockwise"


def test_physical_pod_centering_mask_complex_phase_and_inverse_are_metric_correct():
    coefficients = jnp.array(
        [[-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 1.0], [2.0, -1.0]]
    )
    modes = jnp.array([[1.0 + 1.0j, 0.0, 1.0], [0.0, 1.0 - 0.5j, -1.0]])
    mean = jnp.array([3.0 + 0.2j, -2.0, 1.0])
    values = coefficients @ modes + mean
    feature_mask = jnp.ones_like(values, dtype=bool).at[:, 2].set(False)
    metric = jnp.array([0.5, 2.0, 0.0])
    result = POD(
        2,
        physical_weights=metric,
        centered=True,
        weight_policy="product",
        query_layout_provenance=("grid:x", "channels:scalar"),
        differentiate="basis",
    ).fit_batch(
        MLBatch(
            values,
            feature_mask=feature_mask,
            sample_weight=jnp.array([1.0, 2.0, 1.0, 2.0, 1.0]),
            measure_weight=jnp.array([0.5, 0.5, 1.0, 1.0, 1.0]),
        )
    )
    model = result.as_trainable()
    gram = model.components @ jnp.diag(metric) @ jnp.conj(model.components).T

    assert jnp.allclose(gram, jnp.eye(2), atol=2e-5)
    assert jnp.allclose(model.project(values)[..., :2], values[..., :2], atol=3e-5)
    assert jnp.allclose(model.project(values)[..., 2], model.offset[..., 2])
    assert result.diagnostics.query_layout_provenance == (
        "grid:x",
        "channels:scalar",
    )
    assert result.diagnostics.centering_provenance == "masked-weighted-feature-mean"
    assert "feature:physical" in result.diagnostics.weighting_provenance
    pivots = _pivot_values(model.weighted_components)
    assert jnp.allclose(jnp.imag(pivots), 0.0, atol=1e-6)
    assert jnp.all(jnp.real(pivots) >= 0.0)


def test_repeated_spectrum_disables_canonical_basis_gradient_diagnostic():
    features = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    result = PCA(1, differentiate="basis").fit_batch(MLBatch(features))

    assert result.diagnostics.repeated_spectrum
    assert not result.diagnostics.canonicalization_valid
    assert not result.diagnostics.basis_gradient_supported
    assert jnp.isclose(result.diagnostics.minimum_eigengap, 0.0, atol=1e-6)


def test_incremental_pca_merges_immutable_chunks_and_matches_batch_projector():
    features = jnp.array(
        [
            [-3.0, -1.0, -4.0],
            [-2.0, 1.0, -1.0],
            [-1.0, -1.0, -2.0],
            [0.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
            [2.0, 1.0, 3.0],
            [3.0, -1.0, 2.0],
            [4.0, 1.0, 5.0],
        ]
    )
    batch = MLBatch(features, sample_weight=jnp.arange(1.0, 9.0))
    incremental = IncrementalPCA(2, chunk_size=3).fit_batch(batch)
    exact = PCA(2).fit_batch(batch)
    model = incremental.as_trainable()

    assert model.chunks_seen == 3
    assert jnp.allclose(model.projector(), exact.as_trainable().projector(), atol=2e-4)
    assert jnp.allclose(model.total_weight, jnp.sum(batch.sample_weight))
    assert model.update_recipe(chunk_size=4).previous is model
    assert jax.jit(model.transform)(features).shape == (8, 2)
