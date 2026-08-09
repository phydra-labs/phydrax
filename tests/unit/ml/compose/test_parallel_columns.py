#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel, ModelBinding
from phydrax.ml.compose import ColumnTransformer, FeatureUnion


class _DenseScaleModel(AbstractArrayModel):
    factor: jax.Array
    input_schema: phx.ml.FeatureSchema = eqx.field(static=True)
    output_schema: phx.ml.FeatureSchema = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, factor, schema):
        self.factor = jnp.asarray(factor)
        self.input_schema = schema
        self.output_schema = schema
        self.in_size = len(schema.names)
        self.out_size = len(schema.names)

    def __call__(self, x, /, *, key=None):
        del key
        return self.factor * jnp.asarray(x)

    def transform_batch(self, batch, /, *, key=None):
        del key
        if batch.targets is not None:
            raise AssertionError("Fitted feature transforms must not receive targets.")
        return batch.with_features(
            self(batch.features),
            feature_schema=self.output_schema,
            feature_mask=batch.feature_mask,
        )


class _DenseScaleRecipe(phx.ml.AbstractRecipe):
    factor: float = eqx.field(static=True)

    def __init__(self, factor):
        self.factor = float(factor)

    def fit_batch(self, batch, /, *, key=None):
        del key
        diagnostics = phx.ml.FitDiagnostics(
            valid=True,
            status=phx.ml.ML_SUCCESS,
            effective_samples=jnp.sum(batch.sample_mask),
            method="dense-scale",
        )
        return phx.ml.FitResult(
            _DenseScaleModel(self.factor, batch.feature_schema),
            diagnostics,
            valid=True,
            status=phx.ml.ML_SUCCESS,
            method="dense-scale",
            gradient_contract=phx.ml.GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="smooth",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="direct",
            ),
        )


class _SparseFirstModel(AbstractArrayModel):
    factor: jax.Array
    input_schema: phx.ml.FeatureSchema = eqx.field(static=True)
    output_schema: phx.ml.FeatureSchema = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding = ModelBinding.blockwise("flat")

    def __init__(self, factor, input_schema, output_name):
        self.factor = jnp.asarray(factor)
        self.input_schema = input_schema
        self.output_schema = phx.ml.FeatureSchema((output_name,))
        self.in_size = len(input_schema.names)
        self.out_size = 1

    def __call__(self, x, /, *, key=None):
        del key
        values = jnp.asarray(x)
        if values.ndim < 2:
            raise ValueError("Sparse block transforms require a sample axis.")
        case_shape = tuple(int(size) for size in values.shape[:-2])
        sparse_values = self.factor * values[..., :1]
        return phx.ml.SparseFeatures(
            sparse_values,
            jnp.zeros(sparse_values.shape, dtype=jnp.int32),
            feature_count=1,
            case_shape=case_shape,
        )

    def transform_batch(self, batch, /, *, key=None):
        del key
        if isinstance(batch.features, phx.ml.SparseFeatures):
            raise TypeError("The test sparse transform expects dense input.")
        sparse_values = self.factor * batch.features[..., :1]
        sparse = phx.ml.SparseFeatures(
            sparse_values,
            jnp.zeros(sparse_values.shape, dtype=jnp.int32),
            feature_count=1,
            valid=batch.feature_mask[..., :1],
            case_shape=batch.case_shape,
        )
        return batch.with_features(sparse, feature_schema=self.output_schema)


class _SparseFirstRecipe(phx.ml.AbstractRecipe):
    factor: float = eqx.field(static=True)
    output_name: str = eqx.field(static=True)

    def __init__(self, factor, output_name):
        self.factor = float(factor)
        self.output_name = str(output_name)

    def fit_batch(self, batch, /, *, key=None):
        del key
        diagnostics = phx.ml.FitDiagnostics(
            valid=True,
            status=phx.ml.ML_SUCCESS,
            effective_samples=jnp.sum(batch.sample_mask),
            method="sparse-first",
        )
        return phx.ml.FitResult(
            _SparseFirstModel(self.factor, batch.feature_schema, self.output_name),
            diagnostics,
            valid=True,
            status=phx.ml.ML_SUCCESS,
            method="sparse-first",
            gradient_contract=phx.ml.GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="smooth",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="direct",
            ),
        )


def _dense_batch():
    return phx.ml.MLBatch(
        jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        feature_mask=jnp.array([[True, True, False], [True, True, True]]),
        sample_weight=jnp.array([2.0, 3.0]),
        measure_weight=jnp.array([0.25, 0.5]),
        groups=jnp.array([7, 8]),
        feature_schema=phx.ml.FeatureSchema(
            ("a", "b", "c"),
            kinds=("continuous", "ordinal", "boolean"),
            layout_id="source",
        ),
    )


def test_feature_union_dense_outputs_are_ordered_prefixed_and_differentiable():
    batch = _dense_batch()
    result = FeatureUnion(
        (("left", _DenseScaleRecipe(2.0)), ("right", _DenseScaleRecipe(-1.0)))
    ).fit_batch(batch, key=jax.random.key(0))
    fitted = result.as_trainable()
    transformed = fitted.transform_batch(batch)

    assert transformed.feature_schema.names == (
        "left__a",
        "left__b",
        "left__c",
        "right__a",
        "right__b",
        "right__c",
    )
    assert transformed.feature_schema.kinds == batch.feature_schema.kinds * 2
    assert jnp.allclose(
        transformed.features,
        jnp.concatenate((2.0 * batch.features, -batch.features), axis=-1),
    )
    assert jnp.array_equal(
        transformed.feature_mask,
        jnp.concatenate((batch.feature_mask, batch.feature_mask), axis=-1),
    )
    assert jnp.array_equal(transformed.sample_weight, batch.sample_weight)
    assert jnp.array_equal(transformed.measure_weight, batch.measure_weight)
    assert jnp.array_equal(transformed.groups, batch.groups)
    point = jnp.array([2.0, 3.0, 4.0])
    assert jnp.allclose(
        jax.jit(lambda value: fitted(value))(point),
        jnp.concatenate((2.0 * point, -point)),
    )
    assert jnp.allclose(
        jax.grad(lambda value: jnp.sum(fitted(value)))(point),
        jnp.ones_like(point),
    )
    assert result.gradient_contract.prediction_inputs == "smooth"
    assert len(fitted.fit_results) == 2


def test_feature_union_supports_all_sparse_and_rejects_mixed_joins():
    batch = _dense_batch()
    sparse_result = FeatureUnion(
        (
            ("one", _SparseFirstRecipe(1.0, "one")),
            ("two", _SparseFirstRecipe(2.0, "two")),
        )
    ).fit_batch(batch, key=jax.random.key(1))
    fitted = sparse_result.as_trainable()
    transformed = fitted.transform_batch(batch)
    expected = jnp.concatenate(
        (batch.features[..., :1], 2.0 * batch.features[..., :1]),
        axis=-1,
    )

    assert fitted.input_binding().batch_mode == "blockwise"
    assert isinstance(transformed.features, phx.ml.SparseFeatures)
    assert transformed.features.feature_count == 2
    assert jnp.allclose(transformed.features.to_dense(), expected)
    deployed = fitted(batch.features)
    assert isinstance(deployed, phx.ml.SparseFeatures)
    assert jnp.allclose(deployed.to_dense(), expected)
    assert transformed.feature_schema.names == ("one__one", "two__two")

    mixed = FeatureUnion(
        (
            ("dense", _DenseScaleRecipe(1.0)),
            ("sparse", _SparseFirstRecipe(1.0, "sparse")),
        )
    )
    with pytest.raises(TypeError, match="sparse and dense"):
        mixed.fit_batch(batch, key=jax.random.key(2))


def test_column_transformer_resolves_names_indices_and_remainder_schema():
    batch = _dense_batch()
    recipe = ColumnTransformer(
        (
            ("named", _DenseScaleRecipe(10.0), ("b",)),
            ("indexed", _DenseScaleRecipe(-1.0), (0,)),
        ),
        remainder="passthrough",
    )
    result = recipe.fit_batch(batch, key=jax.random.key(3))
    fitted = result.as_trainable()
    transformed = fitted.transform_batch(batch)

    assert fitted.transformers[0][2] == (1,)
    assert fitted.transformers[1][2] == (0,)
    assert fitted.remainder_indices == (2,)
    assert transformed.feature_schema.names == (
        "named__b",
        "indexed__a",
        "remainder__c",
    )
    assert transformed.feature_schema.kinds == (
        "ordinal",
        "continuous",
        "boolean",
    )
    assert jnp.allclose(
        transformed.features,
        jnp.stack(
            (
                10.0 * batch.features[:, 1],
                -batch.features[:, 0],
                batch.features[:, 2],
            ),
            axis=-1,
        ),
    )
    assert jnp.array_equal(
        transformed.feature_mask,
        jnp.stack(
            (
                batch.feature_mask[:, 1],
                batch.feature_mask[:, 0],
                batch.feature_mask[:, 2],
            ),
            axis=-1,
        ),
    )
    assert jnp.array_equal(transformed.sample_weight, batch.sample_weight)
    assert jnp.array_equal(transformed.groups, batch.groups)
    assert jnp.allclose(fitted(batch.features), transformed.features)


def test_column_transformer_rejects_duplicate_unknown_and_duplicate_names():
    batch = _dense_batch()
    duplicate_columns = ColumnTransformer((("bad", _DenseScaleRecipe(1.0), ("a", "a")),))
    with pytest.raises(ValueError, match="cannot select a feature twice"):
        duplicate_columns.fit_batch(batch, key=jax.random.key(4))

    unknown = ColumnTransformer((("bad", _DenseScaleRecipe(1.0), ("missing",)),))
    with pytest.raises(KeyError, match="Unknown feature name"):
        unknown.fit_batch(batch, key=jax.random.key(5))

    with pytest.raises(ValueError, match="names must be unique"):
        ColumnTransformer(
            (
                ("same", _DenseScaleRecipe(1.0), (0,)),
                ("same", _DenseScaleRecipe(1.0), (1,)),
            )
        )
