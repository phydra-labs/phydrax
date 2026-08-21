#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax._trainable import partition_trainable


def _axes():
    source_nodes = jnp.asarray([0.0, 0.12, 0.31, 0.55, 0.79, 1.0])
    query_nodes = jnp.asarray([0.04, 0.24, 0.49, 0.73, 0.96])
    source_axis = phx.nn.operator.OperatorAxis(
        "x_support",
        source_nodes,
        quadrature_weights=jnp.asarray([0.08, 0.16, 0.2, 0.21, 0.2, 0.15]),
    )
    query_axis = phx.nn.operator.OperatorAxis(
        "x_query",
        query_nodes,
        quadrature_weights=jnp.full(query_nodes.shape, 1.0 / query_nodes.size),
    )
    return source_axis, query_axis


def _dataset(cases=4):
    source_axis, query_axis = _axes()
    coefficients = jnp.stack(
        (
            jnp.linspace(0.3, 1.0, cases),
            jnp.linspace(-0.7, 0.4, cases),
            jnp.linspace(0.2, 0.8, cases),
        ),
        axis=-1,
    )
    source = (
        coefficients[:, :1]
        + coefficients[:, 1:2] * source_axis.nodes[None, :]
        + coefficients[:, 2:3] * source_axis.nodes[None, :] ** 2
    )
    transformed = jnp.stack(
        (
            0.5 + 1.2 * coefficients[:, 0],
            -0.3 * coefficients[:, 1] + 0.1,
            coefficients[:, 2] + 0.4 * coefficients[:, 0],
        ),
        axis=-1,
    )
    target = (
        transformed[:, :1]
        + transformed[:, 1:2] * query_axis.nodes[None, :]
        + transformed[:, 2:3] * query_axis.nodes[None, :] ** 2
    )
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"source": source},
        {"solution": target},
        source_axes={"source": (source_axis,)},
        query_axes=(query_axis,),
    )


def _model(seed=0):
    source_key, target_key, map_key = jr.split(jr.key(seed), 3)
    source_frame = phx.nn.operator.architectures.LearnedFunctionFrame(
        basis_model=phx.nn.layers.Linear(
            in_size=1,
            out_size=2,
            rwf=False,
            key=source_key,
        ),
        rank=2,
        coord_dim=1,
        frame_id="integration-source-frame",
    )
    target_frame = phx.nn.operator.architectures.LearnedFunctionFrame(
        basis_model=phx.nn.layers.Linear(
            in_size=1,
            out_size=2,
            rwf=False,
            key=target_key,
        ),
        rank=2,
        coord_dim=1,
        frame_id="integration-target-frame",
    )
    coefficient_map = phx.nn.layers.Linear(
        in_size=2,
        out_size=2,
        rwf=False,
        use_bias=False,
        key=map_key,
    )
    return phx.nn.operator.architectures.FunctionFrameReconstructor(
        source_frame=source_frame,
        target_frame=target_frame,
        coefficient_map=coefficient_map,
        policy=phx.nn.operator.architectures.FunctionProjectionPolicy(
            ridge=1e-5,
            rank_policy="regularized",
            require_physical_quadrature=True,
        ),
    )


def _task():
    return phx.nn.operator.OperatorTask(
        "function-frame-polynomial-map",
        dimension_basis=("length",),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "source-field",
                role="source",
                source_name="source",
                physical_dimension=(0.0,),
            ),
            phx.nn.operator.OperatorFieldSpec(
                "solution",
                role="target",
                query_name="query",
                physical_dimension=(0.0,),
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="independent",
            query_is_fixed=False,
        ),
    )


def _trained(model):
    return phx.nn.operator.training.TrainedOperator(
        model,
        _task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "solution"},
        provenance={"dataset": "function-frame-integration"},
    )


def _trainable_arrays(model):
    parameters, _ = partition_trainable(model)
    return tuple(jax.tree.leaves(parameters))


def test_function_frame_reconstructor_trains_through_fit_operator():
    dataset = _dataset()
    model = _model(seed=1)
    initial = _trainable_arrays(model)

    result = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        task=_task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "solution"},
        learning_rate=2e-3,
        steps=1,
        batch_size=2,
    )
    updated = _trainable_arrays(result.last_execution_model)
    prediction = result.trained_operator.predict(dataset.batch)

    assert result.completed_steps == 1
    assert jnp.isfinite(result.final_loss)
    assert prediction.field("solution").values.shape == (4, 5)
    assert jnp.all(jnp.isfinite(prediction.field("solution").values))
    assert any(
        not bool(jnp.array_equal(before, after))
        for before, after in zip(initial, updated, strict=True)
    )


def test_function_frame_artifact_round_trips_in_a_fresh_process(tmp_path):
    dataset = _dataset(cases=2)
    trained = _trained(_model(seed=2))
    expected = trained.predict(dataset.batch).field("solution").values
    destination = phx.nn.operator.training.save_operator_artifact(tmp_path, trained)
    manifest = phx.nn.operator.training.load_operator_artifact_manifest(destination)
    recipe = json.dumps(manifest.execution_model_recipe, sort_keys=True)
    restored = phx.nn.operator.training.load_trained_operator(destination)

    assert manifest.execution_model_architecture_id == (
        "phydrax.operator.architecture:FunctionFrameReconstructor"
    )
    assert "phydrax.operator.function_frame:LearnedFunctionFrame@1" in recipe
    assert "phydrax.nn.operator.architectures.conditioning" not in recipe
    assert jnp.allclose(
        restored.predict(dataset.batch).field("solution").values,
        expected,
    )

    script = r"""
import json
import sys

import jax.numpy as jnp
import numpy as np

import phydrax as phx

source_nodes = jnp.asarray([0.0, 0.12, 0.31, 0.55, 0.79, 1.0])
query_nodes = jnp.asarray([0.04, 0.24, 0.49, 0.73, 0.96])
source_axis = phx.nn.operator.OperatorAxis(
    "x_support",
    source_nodes,
    quadrature_weights=jnp.asarray([0.08, 0.16, 0.2, 0.21, 0.2, 0.15]),
)
query_axis = phx.nn.operator.OperatorAxis(
    "x_query",
    query_nodes,
    quadrature_weights=jnp.full(query_nodes.shape, 1.0 / query_nodes.size),
)
coefficients = jnp.stack(
    (
        jnp.linspace(0.3, 1.0, 2),
        jnp.linspace(-0.7, 0.4, 2),
        jnp.linspace(0.2, 0.8, 2),
    ),
    axis=-1,
)
source = (
    coefficients[:, :1]
    + coefficients[:, 1:2] * source_nodes[None, :]
    + coefficients[:, 2:3] * source_nodes[None, :] ** 2
)
batch = phx.nn.operator.OperatorBatch(
    inputs={
        "source": phx.nn.operator.FunctionSamples(values=source, axes=(source_axis,))
    },
    queries={
        "query": phx.nn.operator.FunctionSamples(values=None, axes=(query_axis,))
    },
    case_axes=("case",),
    case_shape=(2,),
)
manifest = phx.nn.operator.training.load_operator_artifact_manifest(sys.argv[1])
restored = phx.nn.operator.training.load_trained_operator(sys.argv[1])
values = restored.predict(batch).field("solution").values
print(
    json.dumps(
        {
            "architecture_id": manifest.execution_model_architecture_id,
            "values": np.asarray(values).tolist(),
        }
    )
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["architecture_id"] == (
        "phydrax.operator.architecture:FunctionFrameReconstructor"
    )
    assert jnp.allclose(jnp.asarray(payload["values"]), expected)
