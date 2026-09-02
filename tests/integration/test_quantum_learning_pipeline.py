#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.kernels import SquaredExponentialKernel
from phydrax.ml import FeatureSchema, MLBatch, TargetSchema
from phydrax.ml.compose import Pipeline
from phydrax.ml.kernel_methods import SupportVectorClassifierRecipe
from phydrax.ml.preprocessing import MinMaxScaler
from phydrax.ml.quantum import (
    CircuitFeatureTransformRecipe,
    projected_iqp_feature_map,
)
from phydrax.operators.quantum import HilbertRegisterLayout


def test_projected_quantum_features_fit_inside_native_pipeline():
    layout = HilbertRegisterLayout(("a", "b"), (2, 2))
    quantum_features = projected_iqp_feature_map(
        layout,
        entanglement_edges=(("a", "b"),),
        axes=("Z",),
    )
    recipe = Pipeline(
        (
            ("scale", MinMaxScaler((-jnp.pi, jnp.pi))),
            (
                "quantum",
                CircuitFeatureTransformRecipe(
                    quantum_features,
                    output_names=("z_a", "z_b"),
                ),
            ),
            (
                "classifier",
                SupportVectorClassifierRecipe(
                    SquaredExponentialKernel(length_scale=1.0),
                    iterations=20,
                ),
            ),
        )
    )
    features = jnp.asarray(
        [
            [-1.0, -0.8],
            [-0.7, -0.4],
            [-0.3, -0.9],
            [0.4, 0.7],
            [0.8, 0.2],
            [1.0, 0.9],
        ],
        dtype=jnp.float64,
    )
    batch = MLBatch(
        features,
        jnp.asarray([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
        feature_schema=FeatureSchema(("x0", "x1")),
        target_schema=TargetSchema("binary", class_labels=(-1.0, 1.0)),
    )
    result = recipe.fit_batch(batch)
    model = result.as_trainable()
    predictions = jax.vmap(model)(features)

    assert result.valid
    assert predictions.shape == (6,)
    assert jnp.all(jnp.isfinite(predictions))
    assert model.stage_output_schemas[1].names == ("z_a", "z_b")
