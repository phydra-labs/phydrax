#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from phydrax import DerivativeRoute
from phydrax.linalg.svd import RandomizedSVD, SVDTolerancePolicy
from phydrax.ml import MLBatch
from phydrax.ml._contracts import ML_INFEASIBLE
from phydrax.ml.decomposition import IncrementalPCA, PCA, POD, SubspaceModel, TruncatedSVD


def _pivot_values(rows: Array) -> Array:
    indices = jnp.argmax(jnp.abs(rows), axis=-1)
    return jnp.take_along_axis(rows, indices[..., None], axis=-1)[..., 0]


def test_subspaces_scenario_1() -> None:
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
    mask = jnp.ones_like(features, dtype="bool").at[:, 2, 2].set(False)
    batch = MLBatch(
        features,
        feature_mask=mask,
        sample_mask=jnp.array([True, True, False, True, True]),
        sample_weight=jnp.array([1.0, 2.0, 9.0, 2.0, 1.0]),
    )
    result = PCA(2, differentiate="basis").fit_batch(batch)
    model = result.as_trainable()

    # ty: ignore[unresolved-attribute]
    assert model.offset.shape == (2, 3)
    # ty: ignore[unresolved-attribute]
    assert model.transform(features).shape == (2, 5, 2)
    # ty: ignore[unresolved-attribute]
    assert model.inverse_transform(model.transform(features)).shape == features.shape
    assert result.diagnostics.singular_values.shape == (2, 2)
    assert result.diagnostics.explained_energy.shape == (2, 2)
    assert result.diagnostics.rank_evidence.lower_bound.shape == (2,)
    assert result.diagnostics.weighted_orthogonality_error.shape == (2,)
    assert result.diagnostics.projector_gradient_supported.shape == (2,)
    assert result.diagnostics.basis_gradient_supported.shape == (2,)
    # ty: ignore[unresolved-attribute]
    pivots = _pivot_values(model.weighted_components)
    assert jnp.allclose(jnp.imag(pivots), 0.0)
    assert jnp.all(jnp.real(pivots) >= 0.0)
    assert result.derivative_contract.route is DerivativeRoute.SPECTRAL
    features = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    result = TruncatedSVD(2).fit_batch(MLBatch(features))
    model = result.as_trainable()
    points = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(model.offset, 0.0)
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(model.project(features), features, atol=1e-5)
    # ty: ignore[unresolved-attribute]
    assert jax.jit(model.transform)(points).shape == (2, 2)
    # ty: ignore[unresolved-attribute]
    assert jax.vmap(model.transform)(points).shape == (2, 2)
    assert model.input_binding().batch_mode == "blockwise"
    coefficients = jnp.array(
        [[-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 1.0], [2.0, -1.0]]
    )
    modes = jnp.array([[1.0 + 1.0j, 0.0, 1.0], [0.0, 1.0 - 0.5j, -1.0]])
    mean = jnp.array([3.0 + 0.2j, -2.0, 1.0])
    values = coefficients @ modes + mean
    feature_mask = jnp.ones_like(values, dtype="bool").at[:, 2].set(False)
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
    # ty: ignore[unresolved-attribute]
    gram = model.components @ jnp.diag(metric) @ jnp.conj(model.components).T

    assert jnp.allclose(gram, jnp.eye(2), atol=2e-5)
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(model.project(values)[..., :2], values[..., :2], atol=3e-5)
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(model.project(values)[..., 2], model.offset[..., 2])
    assert result.diagnostics.query_layout_provenance == (
        "grid:x",
        "channels:scalar",
    )
    assert result.diagnostics.centering_provenance == "masked-weighted-feature-mean"
    assert "feature:physical" in result.diagnostics.weighting_provenance
    # ty: ignore[unresolved-attribute]
    pivots = _pivot_values(model.weighted_components)
    assert jnp.allclose(jnp.imag(pivots), 0.0, atol=1e-6)
    assert jnp.all(jnp.real(pivots) >= 0.0)


def test_pca_projector_prediction_fit_feature_and_fit_weight_gradients_are_finite() -> (
    None
):
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

    def feature_loss(value: Array) -> Array:
        fitted = PCA(2).fit_batch(MLBatch(value)).as_trainable()
        # ty: ignore[unresolved-attribute]
        return jnp.sum(jnp.square(fitted.project(point)))

    def weight_loss(weight: Array) -> Array:
        fitted = PCA(2).fit_batch(MLBatch(features, sample_weight=weight)).as_trainable()
        # ty: ignore[unresolved-attribute]
        return jnp.sum(jnp.square(fitted.project(point)))

    feature_gradient = jax.grad(feature_loss)(features)
    weight_gradient = jax.grad(weight_loss)(jnp.arange(1.0, 6.0))
    assert jnp.all(jnp.isfinite(prediction_gradient))
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))


def test_subspaces_scenario_2() -> None:
    features = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    result = PCA(1, differentiate="basis").fit_batch(MLBatch(features))

    assert result.diagnostics.cutoff_gap <= 0.0
    assert not result.diagnostics.canonicalization_valid
    assert not result.diagnostics.basis_gradient_supported
    assert jnp.isclose(result.diagnostics.minimum_eigengap, 0.0, atol=1e-6)
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

    # ty: ignore[unresolved-attribute]
    assert model.chunks_seen == 3
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(model.projector(), exact.as_trainable().projector(), atol=2e-4)
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(model.total_weight, jnp.sum(batch.sample_weight))
    # ty: ignore[unresolved-attribute]
    assert model.update_recipe(chunk_size=4).previous is model
    assert jax.jit(model.transform)(features).shape == (8, 2)


def test_repeated_retained_projector_feature_and_weight_response() -> None:
    features = jnp.diag(jnp.array([3.0, 3.0, 1.0, 0.5]))
    point = jnp.array([0.2, -0.7, 0.8, 0.1])
    probe = jnp.array([0.4, 0.3, -0.6, 0.2])
    weights = jnp.ones(4)
    direction = jnp.arange(16.0).reshape((4, 4)) / 19.0

    def loss(x: Array, w: Array) -> Array:
        model = TruncatedSVD(2).fit_batch(MLBatch(x, sample_weight=w)).as_trainable()
        assert isinstance(model, SubspaceModel)
        return jnp.vdot(probe, model.project(point)).real

    dx = jax.jvp(lambda x: loss(x, weights), (features,), (direction,))[1]
    dw_direction = jnp.array([0.3, -0.2, 0.7, 0.1])
    dw = jax.jvp(lambda w: loss(features, w), (weights,), (dw_direction,))[1]
    h = 1e-5
    assert jnp.allclose(
        dx,
        (
            loss(features + h * direction, weights)
            - loss(features - h * direction, weights)
        )
        / (2 * h),
        atol=2e-7,
    )
    assert jnp.allclose(
        dw,
        (
            loss(features, weights + h * dw_direction)
            - loss(features, weights - h * dw_direction)
        )
        / (2 * h),
        atol=2e-7,
    )


def test_current_basis_projection_and_parameter_response() -> None:
    x = jnp.array(
        [[-3.0, 0.2, 0.1], [-1.0, 1.0, -0.4], [1.0, -0.3, 0.2], [3.0, 0.1, 0.5]]
    )
    model = PCA(1).fit_batch(MLBatch(x)).as_trainable()
    assert isinstance(model, SubspaceModel)
    point = jnp.array([0.3, -0.4, 0.9])
    basis = model.weighted_components + jnp.array([[0.1, -0.2, 0.3]])
    current = eqx.tree_at(lambda value: value.weighted_components, model, basis)
    assert jnp.allclose(
        current.project(point),
        current.inverse_transform(current.transform(point)),
        atol=1e-12,
    )
    probe = jnp.array([0.6, -0.2, 0.4])

    def loss(rows: Array) -> Array:
        updated = eqx.tree_at(lambda value: value.weighted_components, model, rows)
        return jnp.vdot(probe, updated.project(point)).real

    direction = jnp.array([[0.2, 0.4, -0.1]])
    tangent = jax.jvp(loss, (basis,), (direction,))[1]
    h = 1e-5
    assert jnp.allclose(
        tangent,
        (loss(basis + h * direction) - loss(basis - h * direction)) / (2 * h),
        atol=1e-8,
    )


def test_none_stops_fit_but_preserves_prediction_input() -> None:
    x = jnp.array([[-3.0, 0.2], [-1.0, 1.0], [1.0, -0.3], [3.0, 0.1]])
    point = jnp.array([0.4, -0.7])

    def loss(features: Array) -> Array:
        model = PCA(1, differentiate="none").fit_batch(MLBatch(features)).as_trainable()
        assert isinstance(model, SubspaceModel)
        return jnp.sum(model.project(point))

    assert jnp.array_equal(jax.grad(loss)(x), jnp.zeros_like(x))
    model = PCA(1, differentiate="none").fit_batch(MLBatch(x)).as_trainable()
    assert isinstance(model, SubspaceModel)
    assert jnp.allclose(jax.jacfwd(model.project)(point), model.projector(), atol=1e-12)


def test_randomized_projection_energy_uses_original_operator() -> None:
    values = jnp.diag(jnp.array([8.0, 3.0, 0.3, 0.2, 0.1, 0.08, 0.04, 0.01]))
    result = TruncatedSVD(
        2,
        differentiate="none",
        method=RandomizedSVD(oversampling=0, power_iterations=0),
        tolerance=SVDTolerancePolicy(residual=1.0, orthogonality=1e-7),
    ).fit_batch(MLBatch(values), key=jax.random.key(19))
    model = result.as_trainable()
    assert isinstance(model, SubspaceModel)
    weighted = values / jnp.sqrt(8.0)
    energies = jnp.sum(
        jnp.abs(weighted @ jnp.conj(model.weighted_components).T) ** 2, axis=0
    )
    total = jnp.sum(jnp.abs(weighted) ** 2)
    assert jnp.allclose(result.diagnostics.explained_energy, energies / total, atol=1e-12)
    assert jnp.allclose(
        result.diagnostics.residual_energy,
        jnp.sum(jnp.abs(weighted - model.project(values) / jnp.sqrt(8.0)) ** 2),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "weights",
    [
        jnp.array([1.0, jnp.nan, 1.0]),
        jnp.array([1.0, -1.0, 1.0]),
        jnp.zeros(3),
    ],
)
def test_pca_weight_owner_preserves_infeasible_precedence(weights: Array) -> None:
    values = jnp.array([[2.0, 0.1], [-1.0, 0.8], [0.3, -0.4]])
    result = PCA(1, differentiate="none").fit_batch(
        MLBatch(values, sample_weight=weights)
    )
    assert int(result.status) == ML_INFEASIBLE
    assert not bool(result.diagnostics.rank_evidence.available)


def test_prediction_only_reconstruction_clears_fit_response() -> None:
    values = jnp.array([[-3.0, 0.2], [-1.0, 1.0], [1.0, -0.3], [3.0, 0.1]])
    model = PCA(1).fit_batch(MLBatch(values)).as_trainable()
    assert isinstance(model, SubspaceModel)
    replacement = model.weighted_components + jnp.array([[0.1, -0.2]])
    prediction = model.prediction_only(weighted_components=replacement)
    assert not prediction.fit_response_provenance
    query = jnp.array([0.3, -0.8])
    assert jnp.allclose(
        prediction.project(query),
        prediction.inverse_transform(prediction.transform(query)),
        atol=1e-12,
    )
