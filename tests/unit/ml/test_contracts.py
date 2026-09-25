#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel, FrozenModel, ModelBinding


_INPUT = phx.DerivativeSurface.INPUT
_PARAMETER = phx.DerivativeSurface.MODEL_PARAMETER
_FIT_TARGETS = phx.DerivativeSurface.FIT_TARGETS
_FIT_HYPERPARAMETERS = phx.DerivativeSurface.FIT_HYPERPARAMETERS


class _ScaleModel(AbstractArrayModel):
    scale: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale):
        self.scale = jnp.asarray(scale)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, x, /, *, key=None):
        del key
        return self.scale * x


class _BlockClassifier(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding = ModelBinding.blockwise()

    def __init__(self):
        self.in_size = 1
        self.out_size = 2

    def __call__(self, x, /, *, key=None):
        del key
        return self.decision_function(x)

    def decision_function(self, x):
        value = jnp.asarray(x)
        return jnp.concatenate((-value, value), axis=-1)

    def predict_proba(self, x):
        return jax.nn.softmax(self.decision_function(x), axis=-1)

    def predict(self, x):
        return jnp.argmax(self.decision_function(x), axis=-1)


class _ScaleRecipe(phx.ml.AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        del key
        targets = batch.require_targets()
        scale = jnp.sum(batch.effective_weight() * targets) / jnp.sum(
            batch.effective_weight()
        )
        diagnostics = phx.ml.FitDiagnostics(
            valid=True,
            status=phx.ml.ML_SUCCESS,
            effective_samples=jnp.sum(batch.effective_weight() > 0),
            method="weighted-mean-scale",
        )
        return phx.ml.FitResult(
            _ScaleModel(scale),
            diagnostics,
            valid=True,
            status=phx.ml.ML_SUCCESS,
            method="weighted-mean-scale",
            derivative_contract=phx.DerivativeContract(
                (
                    phx.SurfaceDerivative(_INPUT, phx.GradientLevel.SMOOTH),
                    phx.SurfaceDerivative(_PARAMETER, phx.GradientLevel.SMOOTH),
                    phx.SurfaceDerivative(_FIT_TARGETS, phx.GradientLevel.CONDITIONAL),
                ),
                route=phx.DerivativeRoute.DIRECT,
            ),
        )


def test_batch_preserves_case_sample_output_and_weight_semantics():
    features = jnp.arange(24.0).reshape(2, 4, 3)
    targets = jnp.arange(16.0).reshape(2, 4, 2)
    batch = phx.ml.MLBatch(
        features,
        targets,
        sample_mask=jnp.array([True, True, False, True]),
        sample_weight=jnp.array([1.0, 2.0, 7.0, 3.0]),
        measure_weight=jnp.array([0.5, 0.5, 0.5, 0.5]),
    )

    assert batch.case_shape == (2,)
    assert batch.target_shape == (2,)
    assert batch.feature_schema.names == ("feature_0", "feature_1", "feature_2")
    assert jnp.allclose(
        batch.effective_weight("product"),
        jnp.array([[0.5, 1.0, 0.0, 1.5], [0.5, 1.0, 0.0, 1.5]]),
    )
    selected = batch.take_samples(jnp.array([3, 0]))
    assert selected.features.shape == (2, 2, 3)
    assert jnp.allclose(selected.targets, targets[:, jnp.array([3, 0])])


def test_sparse_features_are_explicit_and_preserve_duplicate_entries():
    sparse = phx.ml.SparseFeatures(
        jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        jnp.array([[0, 2], [1, 1]]),
        feature_count=3,
    )
    dense = sparse.to_dense()

    assert jnp.allclose(dense, jnp.array([[1.0, 0.0, 2.0], [0.0, 7.0, 0.0]]))
    assert jnp.allclose(
        sparse.right_matmul(jnp.arange(6.0).reshape(3, 2)),
        dense @ jnp.arange(6.0).reshape(3, 2),
    )
    with pytest.raises(ValueError, match="feature_mask is unsupported"):
        phx.ml.MLBatch(sparse, feature_mask=jnp.ones((2, 3), dtype="bool"))


def test_fit_is_pure_frozen_and_remains_differentiable_when_called():
    recipe = _ScaleRecipe()
    features = jnp.ones((3, 1))
    targets = jnp.array([1.0, 2.0, 5.0])
    result = phx.ml.fit(
        recipe,
        features,
        targets,
        sample_weight=jnp.array([1.0, 1.0, 0.0]),
    )

    assert isinstance(result.model, FrozenModel)
    assert jnp.allclose(result.model(jnp.array([3.0])), jnp.array([4.5]))
    assert result.as_trainable() is result.model.model
    gradient = jax.grad(lambda value: jnp.sum(result.model(value)))(jnp.array([2.0]))
    assert jnp.allclose(gradient, jnp.array([1.5]))


def test_fit_result_admits_declared_derivatives_and_rejects_undeclared_surfaces():
    result = phx.ml.fit(_ScaleRecipe(), jnp.ones((3, 1)), jnp.array([1.0, 2.0, 5.0]))

    mixed = phx.DifferentiationRequest((_FIT_TARGETS, _PARAMETER, _INPUT))
    admission = result.require_derivative(mixed)
    assert admission.status == phx.DERIVATIVE_SUPPORTED
    assert admission.route is phx.DerivativeRoute.DIRECT
    assert admission.level(_INPUT) is phx.GradientLevel.SMOOTH
    assert admission.level(_FIT_TARGETS) is phx.GradientLevel.CONDITIONAL
    assert "regularity-undeclared" in admission.conditions

    surrogate = phx.DifferentiationRequest(
        (_INPUT,), authority=phx.ComponentAuthority.SURROGATE
    )
    assert not result.derivative_admission(surrogate).supported
    permitted = result.derivative_admission(
        surrogate, policy=phx.RegularityPolicy(allow_undeclared=True)
    )
    assert permitted.level(_INPUT) is phx.GradientLevel.SMOOTH

    undeclared = phx.DifferentiationRequest((_PARAMETER, _FIT_HYPERPARAMETERS))
    rejected = result.derivative_admission(undeclared)
    assert rejected.status == phx.DERIVATIVE_UNSUPPORTED
    assert rejected.level(_PARAMETER) is phx.GradientLevel.SMOOTH
    assert rejected.level(_FIT_HYPERPARAMETERS) is phx.GradientLevel.NONE
    assert rejected.reasons == ("surface-unsupported:fit-hyperparameters",)
    with pytest.raises(ValueError, match="derivative-unsupported"):
        result.require_derivative(undeclared)
    with pytest.raises(ValueError, match="derivative-unsupported"):
        phx.ml.fit(
            _ScaleRecipe(),
            jnp.ones((3, 1)),
            jnp.array([1.0, 2.0, 5.0]),
            derivative_request=undeclared,
        )


def test_frozen_model_preserves_binding_and_prediction_capabilities():
    frozen = FrozenModel(_BlockClassifier())
    values = jnp.asarray(((1.0,), (-2.0,)))

    assert frozen.input_binding() == ModelBinding.blockwise()
    assert jnp.array_equal(frozen.predict(values), jnp.asarray((1, 0)))
    assert jnp.allclose(
        frozen.predict_proba(values),
        jax.nn.softmax(frozen.decision_function(values), axis=-1),
    )

    with pytest.raises(AttributeError):
        FrozenModel(_ScaleModel(1.0)).predict(values)


def test_existing_batch_rejects_duplicate_metadata():
    batch = phx.ml.MLBatch(jnp.ones((3, 1)), jnp.ones((3,)))
    with pytest.raises(ValueError, match="cannot accompany"):
        phx.ml.fit(_ScaleRecipe(), batch, sample_weight=jnp.ones((3,)))
