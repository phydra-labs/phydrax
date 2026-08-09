#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import (
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    MLBatch,
    SparseFeatures,
    TargetSchema,
)
from phydrax.ml.calibration import (
    CalibratedClassifierModel,
    CalibratedClassifierRecipe,
    IsotonicCalibrationModel,
    IsotonicCalibrationRecipe,
    MatrixCalibrationModel,
    MatrixCalibrationRecipe,
    MulticlassCalibrationModel,
    MulticlassCalibrationRecipe,
    PlattCalibrationModel,
    PlattCalibrationRecipe,
    SmoothIsotonicCalibrationModel,
    SmoothIsotonicCalibrationRecipe,
    TemperatureCalibrationModel,
    TemperatureCalibrationRecipe,
    VectorCalibrationModel,
    VectorCalibrationRecipe,
)
from phydrax.ml.naive_bayes import GaussianNaiveBayesRecipe


_BINARY_SCORES = jnp.array(
    [[-3.0], [-2.0], [-1.5], [-1.0], [-0.2], [0.2], [0.8], [1.2], [2.0], [3.0]]
)
_BINARY_TARGETS = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 5)
_BINARY_SCHEMA = TargetSchema(
    "binary", names=("event",), class_labels=("absent", "present")
)
_MULTICLASS_LOGITS = jnp.array(
    [
        [3.0, 0.2, -1.0],
        [2.2, 0.8, -0.5],
        [2.8, -0.3, 0.4],
        [1.8, 0.4, 0.1],
        [0.2, 3.0, -0.8],
        [0.7, 2.3, 0.2],
        [-0.4, 2.8, 0.5],
        [0.3, 1.9, 0.8],
        [-0.7, 0.2, 3.1],
        [0.1, 0.8, 2.4],
        [0.6, -0.4, 2.8],
        [0.4, 0.5, 2.0],
    ]
)
_MULTICLASS_TARGETS = jnp.repeat(jnp.arange(3, dtype=jnp.int32), 4)
_MULTICLASS_SCHEMA = TargetSchema(
    "multiclass",
    names=("phase",),
    class_labels=("solid", "liquid", "gas"),
)


def _sparse(values):
    columns = jnp.broadcast_to(jnp.arange(values.shape[-1]), values.shape)
    return SparseFeatures(values, columns, feature_count=values.shape[-1])


def _platt():
    return PlattCalibrationRecipe(max_iterations=2, tolerance=1e3)


def _temperature():
    return TemperatureCalibrationRecipe(num_classes=3, max_iterations=2, tolerance=1e3)


def _vector():
    return VectorCalibrationRecipe(num_classes=3, max_iterations=2, tolerance=1e3)


def _matrix():
    return MatrixCalibrationRecipe(num_classes=3, max_iterations=2, tolerance=1e3)


def _multiclass():
    return MulticlassCalibrationRecipe(num_classes=3, max_iterations=2, tolerance=1e3)


@pytest.mark.parametrize(
    ("recipe", "features", "targets", "schema", "model_type", "method"),
    [
        (
            _platt(),
            _BINARY_SCORES,
            _BINARY_TARGETS,
            _BINARY_SCHEMA,
            PlattCalibrationModel,
            "platt",
        ),
        (
            _temperature(),
            _MULTICLASS_LOGITS,
            _MULTICLASS_TARGETS,
            _MULTICLASS_SCHEMA,
            TemperatureCalibrationModel,
            "temperature",
        ),
        (
            _vector(),
            _MULTICLASS_LOGITS,
            _MULTICLASS_TARGETS,
            _MULTICLASS_SCHEMA,
            VectorCalibrationModel,
            "vector",
        ),
        (
            _matrix(),
            _MULTICLASS_LOGITS,
            _MULTICLASS_TARGETS,
            _MULTICLASS_SCHEMA,
            MatrixCalibrationModel,
            "matrix",
        ),
        (
            _multiclass(),
            _MULTICLASS_LOGITS,
            _MULTICLASS_TARGETS,
            _MULTICLASS_SCHEMA,
            MulticlassCalibrationModel,
            "multiclass",
        ),
    ],
)
def test_every_smooth_calibration_family_normalizes_labels_and_jit_vmap_outputs(
    recipe, features, targets, schema, model_type, method
):
    result = recipe.fit_batch(MLBatch(features, targets, target_schema=schema))
    model = result.as_trainable()
    probability = result.model(features)
    classes = len(schema.class_labels)

    assert isinstance(model, model_type)
    assert bool(result.valid)
    assert result.method == method
    assert result.diagnostics.method == method
    assert bool(result.diagnostics.converged)
    assert probability.shape == (features.shape[0], classes)
    assert jnp.allclose(jnp.sum(probability, axis=-1), 1.0, atol=1e-6)
    assert jnp.allclose(
        jnp.exp(model.predict_log_proba(features)), probability, atol=1e-6
    )
    assert model.predict(features).shape == targets.shape
    assert jnp.array_equal(model.labels, jnp.arange(classes))
    assert model.target_schema.class_labels == schema.class_labels
    assert jax.jit(model)(features[:2]).shape == (2, classes)
    assert jax.vmap(model)(features[:2]).shape == (2, classes)
    assert result.gradient_contract.prediction_inputs == "smooth"
    assert result.gradient_contract.prediction_parameters == "smooth"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"
    assert result.gradient_contract.fit_mode == "unrolled"


def test_platt_calibration_preserves_case_masks_product_weights_and_frozen_execution():
    features = jnp.stack((_BINARY_SCORES, 0.7 * _BINARY_SCORES + 0.1))
    targets = jnp.stack((_BINARY_TARGETS, _BINARY_TARGETS))
    target_mask = jnp.ones_like(targets, dtype=bool).at[:, 1].set(False)
    sample_mask = jnp.arange(10) != 8
    sample_weight = jnp.linspace(0.5, 1.4, 10)
    measure_weight = jnp.linspace(1.3, 0.8, 10)
    batch = MLBatch(
        features,
        targets,
        target_mask=target_mask,
        sample_mask=sample_mask,
        sample_weight=sample_weight,
        measure_weight=measure_weight,
        target_schema=_BINARY_SCHEMA,
    )
    recipe = PlattCalibrationRecipe(
        max_iterations=2, tolerance=1e3, weight_policy="product"
    )
    result = recipe.fit_batch(batch)
    active = target_mask[0] & sample_mask
    effective = jnp.where(active, sample_weight * measure_weight, 0.0)
    expected_mass = jnp.sum(
        effective[:, None] * jax.nn.one_hot(_BINARY_TARGETS, 2), axis=0
    )

    assert result.valid.shape == (2,)
    assert jnp.all(result.valid)
    assert result.model(features).shape == (2, 10, 2)
    assert jnp.allclose(result.diagnostics.class_mass[0], expected_mass)
    assert recipe.weight_policy == "product"
    assert recipe.max_iterations == 2


def test_calibration_rejects_sparse_and_complex_logits_and_checks_new_sample_width():
    dense = (
        _temperature()
        .fit_batch(MLBatch(_MULTICLASS_LOGITS, _MULTICLASS_TARGETS))
        .as_trainable()
    )
    with pytest.raises(TypeError, match="requires dense features"):
        _temperature().fit_batch(
            MLBatch(_sparse(_MULTICLASS_LOGITS), _MULTICLASS_TARGETS)
        )
    probes = jnp.array([[1.0, 0.0, -1.0], [-0.5, 0.5, 1.5]])

    assert dense(probes).shape == (2, 3)
    with pytest.raises(ValueError, match="align with the class vocabulary"):
        dense(jnp.ones((2, 4)))
    with pytest.raises(TypeError, match="real-valued logits"):
        _temperature().fit_batch(
            MLBatch(
                _MULTICLASS_LOGITS.astype(jnp.complex64),
                _MULTICLASS_TARGETS,
            )
        )


@pytest.mark.parametrize(
    ("recipe", "features", "targets", "probe"),
    [
        (_platt(), _BINARY_SCORES, _BINARY_TARGETS, jnp.array([0.3])),
        (
            _temperature(),
            _MULTICLASS_LOGITS,
            _MULTICLASS_TARGETS,
            jnp.array([0.2, -0.3, 0.7]),
        ),
        (_vector(), _MULTICLASS_LOGITS, _MULTICLASS_TARGETS, jnp.array([0.2, -0.3, 0.7])),
        (_matrix(), _MULTICLASS_LOGITS, _MULTICLASS_TARGETS, jnp.array([0.2, -0.3, 0.7])),
        (
            _multiclass(),
            _MULTICLASS_LOGITS,
            _MULTICLASS_TARGETS,
            jnp.array([0.2, -0.3, 0.7]),
        ),
    ],
)
def test_every_smooth_calibrator_has_declared_fit_input_and_parameter_gradients(
    recipe, features, targets, probe
):
    weights = jnp.linspace(0.8, 1.3, features.shape[0])

    def fit_loss(values, sample_weight, learning_rate):
        configured = eqx.tree_at(lambda item: item.learning_rate, recipe, learning_rate)
        model = configured.fit_batch(
            MLBatch(values, targets, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model.decision_function(probe)))

    fit_gradients = jax.grad(fit_loss, argnums=(0, 1, 2))(
        features, weights, recipe.learning_rate
    )
    model = recipe.fit_batch(
        MLBatch(features, targets, sample_weight=weights)
    ).as_trainable()
    input_gradient = jax.grad(
        lambda value: jnp.sum(jnp.square(model.decision_function(value)))
    )(probe)
    if isinstance(model, PlattCalibrationModel):
        parameter_gradient = jax.grad(
            lambda value: jnp.sum(
                jnp.square(
                    eqx.tree_at(lambda item: item.slope, model, value).decision_function(
                        probe
                    )
                )
            )
        )(model.slope)
    elif isinstance(model, TemperatureCalibrationModel):
        parameter_gradient = jax.grad(
            lambda value: jnp.sum(
                jnp.square(
                    eqx.tree_at(
                        lambda item: item.temperature, model, value
                    ).decision_function(probe)
                )
            )
        )(model.temperature)
    elif isinstance(model, VectorCalibrationModel):
        parameter_gradient = jax.grad(
            lambda value: jnp.sum(
                jnp.square(
                    eqx.tree_at(lambda item: item.scale, model, value).decision_function(
                        probe
                    )
                )
            )
        )(model.scale)
    elif isinstance(model, MatrixCalibrationModel):
        parameter_gradient = jax.grad(
            lambda value: jnp.sum(
                jnp.square(
                    eqx.tree_at(lambda item: item.matrix, model, value).decision_function(
                        probe
                    )
                )
            )
        )(model.matrix)
    else:
        parameter_gradient = jax.grad(
            lambda value: jnp.sum(
                jnp.square(
                    eqx.tree_at(lambda item: item.slope, model, value).decision_function(
                        probe
                    )
                )
            )
        )(model.slope)

    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in fit_gradients)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.all(jnp.isfinite(parameter_gradient))


def test_exact_and_smooth_isotonic_are_monotone_distinct_and_extrapolate_constantly():
    batch = MLBatch(
        _BINARY_SCORES,
        _BINARY_TARGETS,
        sample_weight=jnp.linspace(0.7, 1.3, 10),
        target_schema=_BINARY_SCHEMA,
    )
    exact_result = IsotonicCalibrationRecipe().fit_batch(batch)
    smooth_result = SmoothIsotonicCalibrationRecipe(bandwidth=0.2).fit_batch(batch)
    exact = exact_result.as_trainable()
    smooth = smooth_result.as_trainable()
    probes = jnp.linspace(-4.0, 4.0, 41)
    exact_positive = exact.positive_probability(probes)
    smooth_positive = smooth.positive_probability(probes)
    last = int(exact.block_count) - 1
    boundary = exact.positive_probability(jnp.array([-1e6, 1e6]))

    assert isinstance(exact, IsotonicCalibrationModel)
    assert isinstance(smooth, SmoothIsotonicCalibrationModel)
    assert bool(exact_result.valid)
    assert bool(smooth_result.valid)
    assert jnp.all(jnp.diff(exact_positive) >= 0.0)
    assert jnp.all(jnp.diff(smooth_positive) >= -1e-7)
    assert jnp.allclose(boundary, jnp.array([exact.values[0], exact.values[last]]))
    assert not jnp.allclose(exact_positive, smooth_positive)
    assert exact.predict_indices(probes).dtype == jnp.int32
    assert exact_result.gradient_contract.prediction_inputs == "none"
    assert exact_result.gradient_contract.prediction_parameters == "almost-everywhere"
    assert exact_result.gradient_contract.fit_mode == "stopped"
    assert exact_result.gradient_contract.fit_features == "none"
    assert smooth_result.gradient_contract.prediction_inputs == "smooth"
    assert smooth_result.gradient_contract.prediction_parameters == "smooth"
    assert smooth_result.gradient_contract.fit_mode == "stopped"
    assert jax.jit(exact)(probes).shape == (41, 2)
    assert jax.vmap(smooth)(probes).shape == (41, 2)


def test_isotonic_prediction_gradients_match_exact_and_smooth_contracts():
    exact = (
        IsotonicCalibrationRecipe()
        .fit_batch(MLBatch(_BINARY_SCORES, _BINARY_TARGETS))
        .as_trainable()
    )
    smooth = (
        SmoothIsotonicCalibrationRecipe(bandwidth=0.2)
        .fit_batch(MLBatch(_BINARY_SCORES, _BINARY_TARGETS))
        .as_trainable()
    )
    point = jnp.array(-0.2)
    exact_parameter_gradient = jax.grad(
        lambda values: eqx.tree_at(
            lambda item: item.values, exact, values
        ).positive_probability(point)
    )(exact.values)
    smooth_parameter_gradient = jax.grad(
        lambda values: eqx.tree_at(
            lambda item: item.values, smooth, values
        ).positive_probability(point)
    )(smooth.values)
    smooth_input_gradient = jax.grad(smooth.positive_probability)(point)
    bandwidth_gradient = jax.grad(
        lambda bandwidth: eqx.tree_at(
            lambda item: item.bandwidth, smooth, bandwidth
        ).positive_probability(point)
    )(smooth.bandwidth)

    assert jnp.all(jnp.isfinite(exact_parameter_gradient))
    assert jnp.all(jnp.isfinite(smooth_parameter_gradient))
    assert jnp.isfinite(smooth_input_gradient)
    assert jnp.isfinite(bandwidth_gradient)


def test_isotonic_masks_are_equivalent_to_removing_samples():
    mask = jnp.arange(10) != 9
    masked = (
        IsotonicCalibrationRecipe()
        .fit_batch(
            MLBatch(
                _BINARY_SCORES,
                _BINARY_TARGETS,
                sample_mask=mask,
                sample_weight=jnp.linspace(0.7, 1.3, 10),
            )
        )
        .as_trainable()
    )
    removed = (
        IsotonicCalibrationRecipe()
        .fit_batch(
            MLBatch(
                _BINARY_SCORES[:-1],
                _BINARY_TARGETS[:-1],
                sample_weight=jnp.linspace(0.7, 1.3, 10)[:-1],
            )
        )
        .as_trainable()
    )
    probes = jnp.linspace(-4.0, 4.0, 25)

    assert jnp.allclose(masked(probes), removed(probes))


def test_calibrated_classifier_composes_frozen_base_and_calibrator_with_explicit_key():
    recipe = CalibratedClassifierRecipe(
        GaussianNaiveBayesRecipe(var_smoothing=0.03),
        PlattCalibrationRecipe(max_iterations=2, tolerance=1e3),
        num_classes=2,
    )
    batch = MLBatch(
        _BINARY_SCORES,
        _BINARY_TARGETS,
        target_schema=_BINARY_SCHEMA,
    )
    first = recipe.fit_batch(batch, key=jax.random.key(17))
    repeated = recipe.fit_batch(batch, key=jax.random.key(17))
    model = first.as_trainable()
    probability = first.model(_BINARY_SCORES)

    assert isinstance(model, CalibratedClassifierModel)
    assert bool(first.valid)
    assert bool(first.diagnostics.base_valid)
    assert bool(first.diagnostics.calibration_valid)
    assert probability.shape == (10, 2)
    assert jnp.allclose(jnp.sum(probability, axis=-1), 1.0, atol=1e-6)
    assert jnp.allclose(probability, repeated.model(_BINARY_SCORES))
    assert model.target_schema.class_labels == _BINARY_SCHEMA.class_labels
    assert jax.jit(model)(_BINARY_SCORES[:2]).shape == (2, 2)
    assert jax.vmap(model)(_BINARY_SCORES[:2]).shape == (2, 2)


def test_calibration_failures_report_empty_single_class_nonfinite_and_nonconvergence():
    empty = _platt().fit_batch(
        MLBatch(
            _BINARY_SCORES,
            _BINARY_TARGETS,
            sample_mask=jnp.zeros(10, dtype=bool),
        )
    )
    single_class = _platt().fit_batch(
        MLBatch(_BINARY_SCORES, jnp.zeros(10, dtype=jnp.int32))
    )
    nonfinite = _platt().fit_batch(
        MLBatch(_BINARY_SCORES.at[0, 0].set(jnp.nan), _BINARY_TARGETS)
    )
    negative_weight = _platt().fit_batch(
        MLBatch(
            _BINARY_SCORES,
            _BINARY_TARGETS,
            sample_weight=jnp.ones(10).at[2].set(-1.0),
        )
    )
    nonconverged = PlattCalibrationRecipe(max_iterations=1, tolerance=0.0).fit_batch(
        MLBatch(_BINARY_SCORES, _BINARY_TARGETS)
    )
    isotonic_single_class = IsotonicCalibrationRecipe().fit_batch(
        MLBatch(_BINARY_SCORES, jnp.zeros(10, dtype=jnp.int32))
    )

    assert int(empty.status) == ML_INSUFFICIENT_DATA
    assert int(single_class.status) == ML_INSUFFICIENT_DATA
    assert int(nonfinite.status) == ML_NONFINITE
    assert int(negative_weight.status) == ML_NONFINITE
    assert int(nonconverged.status) == ML_NONCONVERGED
    assert int(isotonic_single_class.status) == ML_INSUFFICIENT_DATA
    assert not bool(nonconverged.diagnostics.converged)
    assert jnp.all(jnp.isfinite(nonconverged.model(_BINARY_SCORES)))


def test_calibration_rejects_rank_capacity_schema_and_complex_composition_mismatches():
    with pytest.raises(ValueError, match="exactly one score feature"):
        _platt().fit_batch(MLBatch(jnp.ones((10, 2)), _BINARY_TARGETS))
    with pytest.raises(ValueError, match="one logit per class"):
        _temperature().fit_batch(MLBatch(jnp.ones((12, 2)), _MULTICLASS_TARGETS))
    with pytest.raises(ValueError, match="one score per class"):
        _vector().fit_batch(MLBatch(jnp.ones((12, 2)), _MULTICLASS_TARGETS))
    with pytest.raises(ValueError, match="one score per class"):
        _matrix().fit_batch(MLBatch(jnp.ones((12, 2)), _MULTICLASS_TARGETS))
    with pytest.raises(ValueError, match="one score per class"):
        _multiclass().fit_batch(MLBatch(jnp.ones((12, 2)), _MULTICLASS_TARGETS))
    with pytest.raises(ValueError, match="exactly one score feature"):
        IsotonicCalibrationRecipe().fit_batch(MLBatch(jnp.ones((10, 2)), _BINARY_TARGETS))
    with pytest.raises(ValueError, match="minimum_temperature"):
        TemperatureCalibrationRecipe(minimum_temperature=0.0)
    with pytest.raises(ValueError, match="finite positive scalar"):
        SmoothIsotonicCalibrationRecipe(bandwidth=0.0)
    with pytest.raises(ValueError, match="conflicts"):
        CalibratedClassifierRecipe(
            GaussianNaiveBayesRecipe(var_smoothing=0.03),
            TemperatureCalibrationRecipe(num_classes=3, max_iterations=2, tolerance=1e3),
            num_classes=2,
        ).fit_batch(
            MLBatch(
                _BINARY_SCORES,
                _BINARY_TARGETS,
                target_schema=_BINARY_SCHEMA,
            )
        )
    with pytest.raises(ValueError, match="scalar class label"):
        _temperature().fit_batch(
            MLBatch(
                _MULTICLASS_LOGITS,
                jax.nn.one_hot(_MULTICLASS_TARGETS, 3),
            )
        )
