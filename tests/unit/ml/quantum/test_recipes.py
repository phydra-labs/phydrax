#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.ml import FeatureSchema, MLBatch, TargetSchema
from phydrax.ml.quantum import (
    CircuitFeatureTransformRecipe,
    data_reuploading_feature_map,
    projected_iqp_feature_map,
    VariationalCircuitClassifierRecipe,
)
from phydrax.operators.quantum import HilbertRegisterLayout


def test_circuit_feature_recipe_binds_schema_and_preserves_batch_metadata():
    layout = HilbertRegisterLayout(("a", "b"), (2, 2))
    feature_model = projected_iqp_feature_map(
        layout,
        entanglement_edges=(("a", "b"),),
        axes=("Z",),
    )
    recipe = CircuitFeatureTransformRecipe(
        feature_model,
        output_names=("z_a", "z_b"),
    )
    batch = MLBatch(
        jnp.asarray([[0.1, 0.2], [0.3, -0.4]], dtype=jnp.float64),
        jnp.asarray([0.0, 1.0]),
        sample_weight=jnp.asarray([1.0, 2.0]),
        groups=jnp.asarray([3, 4]),
        feature_schema=FeatureSchema(("x0", "x1")),
    )
    result = recipe.fit_batch(batch)
    model = result.as_trainable()
    transformed = model.transform(batch.features)

    assert result.valid
    assert model.input_schema.names == ("x0", "x1")
    assert model.output_schema.names == ("z_a", "z_b")
    assert transformed.shape == (2, 2)
    assert jnp.all(jnp.isfinite(transformed))


def test_variational_circuit_classifier_fit_returns_finite_probabilities():
    layout = HilbertRegisterLayout(("q",), (2,))
    feature_model = data_reuploading_feature_map(
        1,
        layout,
        1,
        jr.key(1),
        readout_wire_ids=("q",),
        gradient_method="parameter-shift",
    )
    recipe = VariationalCircuitClassifierRecipe(
        feature_model,
        max_iterations=8,
        learning_rate=0.05,
        tolerance=0.0,
    )
    features = jnp.asarray([[-0.8], [-0.2], [0.2], [0.8]], dtype=jnp.float64)
    batch = MLBatch(
        features,
        jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float64),
        feature_schema=FeatureSchema(("x",)),
        target_schema=TargetSchema("binary", class_labels=(0.0, 1.0)),
    )
    result = recipe.fit_batch(batch, key=jr.key(9))
    model = result.as_trainable()
    probabilities = jax.vmap(model)(features)

    assert result.valid
    assert result.diagnostics.logical_program_evaluations == 224
    assert probabilities.shape == (4,)
    assert jnp.all(jnp.isfinite(probabilities))
    assert jnp.all((probabilities > 0.0) & (probabilities < 1.0))


def test_variational_circuit_classifier_requires_key_and_single_case():
    layout = HilbertRegisterLayout(("q",), (2,))
    feature_model = data_reuploading_feature_map(1, layout, 1, jr.key(2))
    recipe = VariationalCircuitClassifierRecipe(feature_model)
    batch = MLBatch(
        jnp.ones((2, 3, 1), dtype=jnp.float64),
        jnp.zeros((2, 3), dtype=jnp.float64),
    )

    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(
            MLBatch(
                jnp.ones((3, 1), dtype=jnp.float64),
                jnp.asarray([0.0, 1.0, 1.0]),
            )
        )
    with pytest.raises(ValueError, match="one ML case"):
        recipe.fit_batch(batch, key=jr.key(0))
