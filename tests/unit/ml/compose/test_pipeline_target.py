#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import FrozenInstanceError

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel
from phydrax.ml.compose import Pipeline, TransformedTargetRegressor
from phydrax.ml.preprocessing import StandardScaler


class _AuditDiagnostics(eqx.Module):
    feature_mask: jax.Array
    sample_mask: jax.Array
    sample_weight: jax.Array
    measure_weight: jax.Array
    groups: jax.Array | None
    targets: jax.Array | None


class _IdentityModel(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, width):
        self.in_size = int(width)
        self.out_size = int(width)

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.asarray(x)


class _AuditRecipe(phx.ml.AbstractRecipe):
    gradient_contract: phx.ml.GradientContract

    def __init__(self, gradient_contract=None):
        self.gradient_contract = (
            phx.ml.GradientContract.direct()
            if gradient_contract is None
            else gradient_contract
        )

    def fit_batch(self, batch, /, *, key=None):
        del key
        diagnostics = _AuditDiagnostics(
            batch.feature_mask,
            batch.sample_mask,
            batch.sample_weight,
            batch.measure_weight,
            batch.groups,
            batch.targets,
        )
        return phx.ml.FitResult(
            _IdentityModel(batch.feature_count),
            diagnostics,
            valid=True,
            status=phx.ml.ML_SUCCESS,
            method="audit",
            gradient_contract=self.gradient_contract,
        )


class _LeakageRejectingShift(AbstractArrayModel):
    shift: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, shift, width):
        self.shift = jnp.asarray(shift)
        self.in_size = int(width)
        self.out_size = int(width)

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.asarray(x) + self.shift

    def transform_batch(self, batch, /, *, key=None):
        del key
        if batch.targets is not None:
            raise AssertionError("Fitted feature transforms must not receive targets.")
        return batch.with_features(
            self(batch.features),
            feature_schema=batch.feature_schema,
            feature_mask=batch.feature_mask,
        )


class _KeyedShiftRecipe(phx.ml.AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        if key is None:
            raise ValueError("_KeyedShiftRecipe requires an explicit key.")
        shift = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
        diagnostics = phx.ml.FitDiagnostics(
            valid=True,
            status=phx.ml.ML_SUCCESS,
            effective_samples=jnp.sum(batch.sample_mask),
            method="keyed-shift",
        )
        return phx.ml.FitResult(
            _LeakageRejectingShift(shift, batch.feature_count),
            diagnostics,
            valid=True,
            status=phx.ml.ML_SUCCESS,
            method="keyed-shift",
            gradient_contract=phx.ml.GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="conditional",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="direct",
            ),
        )


class _MeanModel(AbstractArrayModel):
    mean: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, mean, in_size):
        self.mean = jnp.asarray(mean).reshape(())
        self.in_size = int(in_size)
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        values = jnp.asarray(x)
        return self.mean + jnp.asarray(0, dtype=values.dtype) * jnp.sum(values)


class _MeanRegressor(phx.ml.AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        del key
        targets = jnp.asarray(batch.require_targets())
        if targets.ndim != len(batch.case_shape) + 1:
            raise ValueError("_MeanRegressor expects scalar targets.")
        weights = batch.effective_weight()
        mean = jnp.sum(weights * targets, axis=-1) / jnp.sum(weights, axis=-1)
        if batch.case_shape:
            raise ValueError("The test mean regressor uses a single case.")
        diagnostics = phx.ml.FitDiagnostics(
            valid=True,
            status=phx.ml.ML_SUCCESS,
            effective_samples=jnp.sum(weights > 0),
            method="mean-regressor",
        )
        return phx.ml.FitResult(
            _MeanModel(mean, batch.feature_count),
            diagnostics,
            valid=True,
            status=phx.ml.ML_SUCCESS,
            method="mean-regressor",
            gradient_contract=phx.ml.GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="smooth",
                fit_targets="conditional",
                fit_weights="conditional",
                fit_hyperparameters="none",
                fit_mode="direct",
                conditions=("Positive total sample weight is held fixed.",),
            ),
        )


def _batch():
    return phx.ml.MLBatch(
        jnp.array([[1.0, 2.0], [3.0, 5.0], [8.0, 13.0]]),
        jnp.array([2.0, 4.0, 9.0]),
        feature_mask=jnp.array([[True, True], [True, False], [True, True]]),
        sample_mask=jnp.array([True, True, False]),
        sample_weight=jnp.array([1.0, 3.0, 7.0]),
        measure_weight=jnp.array([0.5, 0.25, 2.0]),
        groups=jnp.array([4, 4, 9]),
        feature_schema=phx.ml.FeatureSchema(("x", "z")),
    )


def test_pipeline_is_leakage_safe_deterministic_and_preserves_batch_metadata():
    batch = _batch()
    contract = phx.ml.GradientContract(
        prediction_inputs="almost-everywhere",
        prediction_parameters="smooth",
        fit_features="none",
        fit_targets="none",
        fit_weights="none",
        fit_hyperparameters="none",
        fit_mode="direct",
    )
    recipe = Pipeline((("shift", _KeyedShiftRecipe()), ("audit", _AuditRecipe(contract))))

    first = recipe.fit_batch(batch, key=jax.random.key(17))
    repeated = recipe.fit_batch(batch, key=jax.random.key(17))
    different = recipe.fit_batch(batch, key=jax.random.key(18))
    fitted = first.as_trainable()
    audit = fitted.fit_results[1].diagnostics

    assert jnp.allclose(fitted(batch.features), repeated.as_trainable()(batch.features))
    assert not jnp.allclose(
        fitted(batch.features), different.as_trainable()(batch.features)
    )
    assert jnp.array_equal(audit.feature_mask, batch.feature_mask)
    assert jnp.array_equal(audit.sample_mask, batch.sample_mask)
    assert jnp.array_equal(audit.sample_weight, batch.sample_weight)
    assert jnp.array_equal(audit.measure_weight, batch.measure_weight)
    assert jnp.array_equal(audit.groups, batch.groups)
    assert jnp.array_equal(audit.targets, batch.targets)
    assert fitted.feature_schema.names == ("x", "z")
    assert fitted.final_feature_schema.names == ("x", "z")
    assert first.gradient_contract.prediction_inputs == "almost-everywhere"
    assert first.gradient_contract.fit_features == "none"
    assert first.gradient_contract.fit_mode == "direct"
    assert first.diagnostics.names == ("shift", "audit")
    assert len(fitted.fit_results) == 2
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'steps'"):
        fitted.steps = ()

    point = jnp.array([2.0, 3.0])
    assert jnp.allclose(jax.jit(lambda value: fitted(value))(point), fitted(point))
    gradient = jax.grad(lambda value: jnp.sum(fitted(value)))(point)
    assert jnp.allclose(gradient, jnp.ones_like(point))


def test_transformed_target_regressor_uses_fitted_inverse_and_composes_contracts():
    features = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    targets = jnp.array([2.0, 4.0, 8.0, 10.0])
    weights = jnp.array([1.0, 1.0, 2.0, 0.0])
    batch = phx.ml.MLBatch(
        features,
        targets,
        sample_weight=weights,
        feature_schema=phx.ml.FeatureSchema(("x",)),
    )
    result = TransformedTargetRegressor(_MeanRegressor(), StandardScaler()).fit_batch(
        batch, key=jax.random.key(3)
    )
    fitted = result.as_trainable()
    expected = jnp.sum(weights * targets) / jnp.sum(weights)

    assert jnp.allclose(fitted(jnp.array([7.0])), expected)
    assert jnp.allclose(jax.jit(lambda value: fitted(value))(jnp.array([5.0])), expected)
    assert jnp.allclose(
        jax.grad(lambda value: fitted(value))(jnp.array([5.0])),
        jnp.array([0.0]),
    )
    assert fitted.transformer_result.method == "standard_scaler"
    assert fitted.regressor_result.method == "mean-regressor"
    assert result.diagnostics.names == ("transformer", "regressor")
    assert result.gradient_contract.prediction_inputs == "smooth"
    assert result.gradient_contract.fit_features == "smooth"
    assert result.gradient_contract.fit_targets == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert "inverse_transform" in result.gradient_contract.conditions[-1]


def test_transformed_target_regressor_rejects_non_regression_target_semantics():
    batch = phx.ml.MLBatch(
        jnp.ones((3, 1)),
        jnp.array([0, 1, 0]),
        target_schema=phx.ml.TargetSchema("binary", class_labels=(0, 1)),
    )
    with pytest.raises(ValueError, match="continuous or count"):
        TransformedTargetRegressor(_MeanRegressor(), StandardScaler()).fit_batch(
            batch, key=jax.random.key(0)
        )
