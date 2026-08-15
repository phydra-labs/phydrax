#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import subprocess
import sys

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _batch(*, cases=2, size=8):
    axis = phx.nn.operator.OperatorAxis(
        "x", jnp.arange(size, dtype=float) / size, periodic=True
    )
    return phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=jnp.arange(cases * size, dtype=float).reshape(cases, size),
                axes=(axis,),
            )
        },
        queries={
            "solution-query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
        },
        case_axes=("case",),
        case_shape=(cases,),
    )


def _task(*, revision="1"):
    return phx.nn.operator.OperatorTask(
        "periodic-map",
        revision=revision,
        dimension_basis=("length",),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "input",
                role="source",
                source_name="u",
                physical_dimension=(0.0,),
                scale=2.0,
                offset=1.0,
            ),
            phx.nn.operator.OperatorFieldSpec(
                "solution",
                role="target",
                query_name="solution-query",
                physical_dimension=(0.0,),
                scale=3.0,
                offset=4.0,
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "solution-query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
            ),
        ),
        metadata={"source": {"family": "synthetic"}, "tags": ["periodic"]},
    )


def _fixed_task():
    base = _task()
    return phx.nn.operator.OperatorTask(
        base.task_id,
        revision=base.revision,
        dimension_basis=base.dimension_basis,
        fields=base.fields,
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "solution-query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
                fixed_geometry=True,
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=True,
        ),
        metadata=base.metadata,
    )


def _trained(
    *,
    revision="1",
    output_pipeline=None,
    compilation_strategy="eager",
    dtype_policy=None,
):
    model = phx.nn.operator.architectures.FNO(
        n_modes=(3,),
        width=4,
        depth=1,
        coordinate_embedding=False,
        source_key="u",
        key=jr.key(12),
    )
    return phx.nn.operator.training.TrainedOperator(
        model,
        _task(revision=revision),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        output_field_map={"output": "solution"},
        output_pipeline=output_pipeline,
        dtype_policy=dtype_policy,
        compilation_strategy=compilation_strategy,
        provenance={"dataset": "immutable-dataset"},
    )


def test_operator_task_is_canonical_and_rejects_unknown_sources():
    task = _task()
    restored = phx.nn.operator.OperatorTask.from_dict(task.to_dict())

    assert restored.fingerprint == task.fingerprint
    assert len(task.fingerprint) == 64
    task.validate_batch(_batch())

    batch = _batch()
    with pytest.raises(ValueError, match="absent from the task"):
        task.validate_batch(
            phx.nn.operator.OperatorBatch(
                inputs={**batch.inputs, "undeclared": batch.input("u")},
                queries=batch.queries,
                case_axes=batch.case_axes,
                case_shape=batch.case_shape,
            )
        )


def test_trained_operator_applies_physical_transforms_around_model_execution():
    trained = _trained()
    batch = _batch()
    prepared = trained.prepare(batch)
    assert trained.execution_plan.execution_model is trained.execution_model
    assert trained.execution_plan.compilation_strategy == "eager"
    assert trained.execution_plan.padding_policy == "explicit_mask"
    assert prepared.plan_fingerprint == trained.execution_plan.fingerprint
    nondimensional = (batch.input("u").values - 1.0) / 2.0

    assert jnp.allclose(prepared.execution_batch.input("u").values, nondimensional)
    raw = trained.execution_model.predict(prepared.execution_batch).field("output").values
    prediction = trained.predict_prepared(prepared).field("solution").values
    assert jnp.allclose(prediction, raw * 3.0 + 4.0)

    compiled = _trained(compilation_strategy="compiled")
    compiled_prediction = compiled.predict(batch).field("solution").values
    assert jnp.allclose(compiled_prediction, prediction)

    other = _trained(revision="2")
    with pytest.raises(ValueError, match="different runtime contract"):
        other.predict_prepared(prepared)

    other_dtype = _trained(
        dtype_policy=phx.nn.operator.training.OperatorDTypePolicy(compute_dtype="float64")
    )
    with pytest.raises(ValueError, match="different runtime contract"):
        other_dtype.predict_prepared(prepared)


def test_normalized_output_pipeline_enforces_physical_conservation(tmp_path):
    pipeline = phx.nn.operator.training.OperatorOutputPipeline(
        phx.nn.operator.training.ConservationProjection("solution", source_name="u")
    )
    trained = _trained(output_pipeline=pipeline)
    batch = _batch()
    prediction = trained.predict(batch)
    source_total = phx.nn.operator.training.operator_integral(
        batch.input("u").values,
        batch.input("u"),
        case_shape=batch.case_shape,
    )
    predicted_total = phx.nn.operator.training.operator_integral(
        prediction.field("solution").values,
        batch.query("solution-query"),
        case_shape=batch.case_shape,
    )

    assert jnp.allclose(predicted_total, source_total)
    destination = phx.nn.operator.training.save_operator_artifact(tmp_path, trained)
    restored = phx.nn.operator.training.load_trained_operator(destination)
    restored_total = phx.nn.operator.training.operator_integral(
        restored.predict(batch).field("solution").values,
        batch.query("solution-query"),
        case_shape=batch.case_shape,
    )
    manifest = phx.nn.operator.training.load_operator_artifact_manifest(destination)
    assert jnp.allclose(restored_total, source_total)
    assert "format_version" not in manifest.to_dict()
    assert manifest.output_pipeline_fingerprint == pipeline.fingerprint
    assert restored.output_pipeline is not None
    assert restored.output_pipeline.fingerprint == pipeline.fingerprint


def test_trained_operator_preserves_multiple_named_outputs_and_queries():
    source_coordinates = jnp.linspace(0.0, 1.0, 4)[:, None]
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=jnp.linspace(-1.0, 1.0, 4),
                coordinates=source_coordinates,
            )
        },
        queries={
            "spatial": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.linspace(0.0, 1.0, 3)[:, None],
            ),
            "sensors": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.asarray([[0.25], [0.75]]),
            ),
        },
    )
    targets = (
        ("state", "spatial", phx.nn.operator.OperatorOutputSpec("scalar")),
        (
            "flux",
            "sensors",
            phx.nn.operator.OperatorOutputSpec(2, component_names=("x", "y")),
        ),
    )
    graph = phx.nn.operator.OperatorBranchGraph(
        tuple(
            phx.nn.operator.OperatorBranchSpec(
                name,
                role="both",
                geometry_kind="point_cloud",
                source_name="u",
                query_name=query_name,
                output_spec=spec,
            )
            for name, query_name, spec in targets
        )
    )
    model = phx.nn.operator.architectures.ABUPT(
        graph,
        input_channels={"state": "scalar", "flux": "scalar"},
        coord_dims={"state": 1, "flux": 1},
        anchor_counts={"state": 2, "flux": 2},
        width=4,
        depth=1,
        num_heads=1,
        key=jr.key(41),
    )
    task = phx.nn.operator.OperatorTask(
        "coupled-map",
        revision="1",
        dimension_basis=("length",),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "input",
                role="source",
                source_name="u",
                physical_dimension=(0.0,),
            ),
            *tuple(
                phx.nn.operator.OperatorFieldSpec(
                    name,
                    role="target",
                    query_name=query_name,
                    channels=spec.channels,
                    component_names=spec.component_names,
                    physical_dimension=(0.0,),
                )
                for name, query_name, spec in targets
            ),
        ),
        queries=tuple(
            phx.nn.operator.OperatorQuerySpec(
                name,
                geometry_kind="point_cloud",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
            )
            for name in ("spatial", "sensors")
        ),
    )
    trained = phx.nn.operator.training.TrainedOperator(
        model,
        task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
    )

    prediction = trained.predict(batch)

    assert tuple(prediction.fields) == ("state", "flux")
    assert tuple(prediction.queries) == ("spatial", "sensors")
    assert prediction.field("state").values.shape == (3,)
    assert prediction.field("flux").values.shape == (2, 2)


def test_fixed_query_geometry_is_shared_and_persistently_bound(tmp_path):
    batch = _batch()
    task = _fixed_task()
    fingerprint = batch.query("solution-query").geometry_fingerprint()
    trained = phx.nn.operator.training.TrainedOperator(
        _trained().execution_model,
        task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "solution"},
        fixed_query_fingerprints={"solution-query": fingerprint},
    )

    coordinates = jnp.broadcast_to(
        batch.query("solution-query").coordinates_array(
            case_shape=batch.case_shape,
        ),
        batch.case_shape + batch.query("solution-query").sample_shape + (1,),
    )
    case_dependent = phx.nn.operator.OperatorBatch(
        inputs=batch.inputs,
        queries={
            "solution-query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=coordinates.at[1].add(0.125),
            )
        },
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    case_task = phx.nn.operator.OperatorTask(
        "case-dependent-fixed-query",
        dimension_basis=("length",),
        fields=task.fields,
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "solution-query",
                geometry_kind="point_cloud",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
                fixed_geometry=True,
            ),
        ),
        problem=task.problem,
    )
    with pytest.raises(ValueError, match="shared by every case"):
        case_task.validate_batch(case_dependent)

    altered_axis = phx.nn.operator.OperatorAxis(
        "x",
        batch.query("solution-query").axes[0].nodes + 0.01,
        periodic=True,
    )
    altered = phx.nn.operator.OperatorBatch(
        inputs=batch.inputs,
        queries={
            "solution-query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(altered_axis,),
            )
        },
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    with pytest.raises(ValueError, match="different physical geometry"):
        trained.predict(altered)

    destination = phx.nn.operator.training.save_operator_artifact(tmp_path, trained)
    restored = phx.nn.operator.training.load_trained_operator(destination)
    assert dict(restored.fixed_query_fingerprints) == {"solution-query": fingerprint}
    with pytest.raises(ValueError, match="different physical geometry"):
        restored.predict(altered)


def test_operator_task_serialization_rejects_noncanonical_fields():
    payload = _task().to_dict()
    payload["schema_version"] = 1

    with pytest.raises(ValueError, match="current canonical fields"):
        phx.nn.operator.OperatorTask.from_dict(payload)


def test_operator_artifact_manifest_rejects_noncanonical_fields(tmp_path):
    phx.nn.operator.training.save_operator_artifact(tmp_path, _trained())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="current canonical fields"):
        phx.nn.operator.training.load_trained_operator(tmp_path)


def test_operator_artifact_rejects_unknown_architecture_codec(tmp_path):
    phx.nn.operator.training.save_operator_artifact(tmp_path, _trained())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_model_architecture_id"] = "unknown.operator:FNO@1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown operator architecture ID"):
        phx.nn.operator.training.load_trained_operator(tmp_path)


def test_portable_operator_artifact_round_trips_inference_and_training_state(tmp_path):
    trained = _trained()
    batch = _batch()
    destination = phx.nn.operator.training.save_operator_artifact(
        tmp_path,
        trained,
        training_state={"step": jnp.asarray(7), "loss": jnp.asarray(0.25)},
        training_metadata={"epoch": 2},
    )
    manifest = phx.nn.operator.training.load_operator_artifact_manifest(destination)
    assert manifest.format == "phydrax-operator-artifact"
    assert manifest.version == 3
    assert manifest.execution_model_architecture_id == "phydrax.operator.architecture:FNO"
    recipe = json.dumps(manifest.execution_model_recipe, sort_keys=True)
    assert "phydrax.nn." not in recipe
    assert "phydrax.operator.architecture:FNO" in recipe
    assert "phydrax.artifact:Linear@1" in recipe

    restored = phx.nn.operator.training.load_trained_operator(destination)
    resume = phx.nn.operator.training.load_operator_training_state(destination)

    assert restored.task.fingerprint == trained.task.fingerprint
    assert restored.contract_fingerprint == trained.contract_fingerprint
    assert jnp.allclose(
        restored.predict(batch).field("solution").values,
        trained.predict(batch).field("solution").values,
    )
    assert int(resume.state["step"]) == 7
    assert resume.metadata == {"epoch": 2}


def test_wavelet_operator_artifacts_round_trip_without_model_templates(tmp_path):
    models = (
        phx.nn.operator.architectures.WaveletNeuralOperator(
            1,
            in_channels="scalar",
            out_channels="scalar",
            levels=2,
            wavelet="db2",
            width=4,
            depth=1,
            source_key="u",
            key=jr.key(31),
        ),
        phx.nn.operator.architectures.MultiwaveletOperator(
            in_channels="scalar",
            out_channels="scalar",
            order=2,
            levels=2,
            width=4,
            depth=1,
            source_key="u",
            key=jr.key(32),
        ),
    )
    batch = _batch()

    for model in models:
        trained = phx.nn.operator.training.TrainedOperator(
            model,
            _task(),
            training_evidence=phx.nn.operator.OperatorTrainingEvidence(
                regime="task_specific"
            ),
            output_field_map={"output": "solution"},
        )
        expected = trained.predict(batch).field("solution").values
        destination = phx.nn.operator.training.save_operator_artifact(
            tmp_path / type(model).__name__,
            trained,
        )

        restored = phx.nn.operator.training.load_trained_operator(destination)
        actual = restored.predict(batch).field("solution").values
        manifest = phx.nn.operator.training.load_operator_artifact_manifest(destination)

        assert manifest.execution_model_architecture_id == (
            f"phydrax.operator.architecture:{type(model).__name__}"
        )
        assert jnp.allclose(actual, expected)


def test_sfno_artifact_round_trips_s2fft_plan_without_model_template(tmp_path):
    plan = phx.nn.operator.architectures.SphericalHarmonicPlan(3)
    axes = (
        phx.nn.operator.OperatorAxis(
            "theta",
            plan.theta,
            quadrature_weights=plan.theta_quadrature_weights,
            basis="sphere",
        ),
        phx.nn.operator.OperatorAxis(
            "phi",
            plan.phi,
            quadrature_weights=plan.phi_quadrature_weights,
            basis="fourier",
            periodic=True,
        ),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=jnp.cos(plan.theta)[:, None] * jnp.ones((1, plan.phi.size)),
                axes=axes,
            )
        },
        queries={
            "solution-query": phx.nn.operator.FunctionSamples(values=None, axes=axes)
        },
    )
    task = phx.nn.operator.OperatorTask(
        "spherical-map",
        dimension_basis=(),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "input",
                role="source",
                source_name="u",
                physical_dimension=(),
            ),
            phx.nn.operator.OperatorFieldSpec(
                "solution",
                role="target",
                query_name="solution-query",
                physical_dimension=(),
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "solution-query",
                geometry_kind="sphere",
                coordinate_components=("theta", "phi"),
                quadrature="physical_required",
                fixed_geometry=True,
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=True,
        ),
    )
    model = phx.nn.operator.architectures.SFNO(
        plan,
        width=4,
        depth=1,
        source_key="u",
        key=jr.key(33),
    )
    trained = phx.nn.operator.training.TrainedOperator(
        model,
        task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        output_field_map={"output": "solution"},
        fixed_query_fingerprints={
            "solution-query": batch.query("solution-query").geometry_fingerprint()
        },
    )
    expected = trained.predict(batch).field("solution").values

    destination = phx.nn.operator.training.save_operator_artifact(tmp_path, trained)
    restored = phx.nn.operator.training.load_trained_operator(destination)
    assert isinstance(restored.execution_model, phx.nn.operator.architectures.SFNO)
    actual = restored.predict(batch).field("solution").values

    assert restored.execution_model.plan.fingerprint == plan.fingerprint
    assert jnp.allclose(actual, expected)


def test_portable_operator_artifact_loads_in_fresh_process(tmp_path):
    trained = _trained()
    expected = trained.predict(_batch()).field("solution").values
    destination = phx.nn.operator.training.save_operator_artifact(tmp_path, trained)
    script = """
import json
import sys

import jax.numpy as jnp
import numpy as np

import phydrax as phx

size = 8
axis = phx.nn.operator.OperatorAxis(
    "x", jnp.arange(size, dtype=float) / size, periodic=True
)
batch = phx.nn.operator.OperatorBatch(
    inputs={
        "u": phx.nn.operator.FunctionSamples(
            values=jnp.arange(2 * size, dtype=float).reshape(2, size),
            axes=(axis,),
        )
    },
    queries={
        "solution-query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
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

    assert payload["architecture_id"] == "phydrax.operator.architecture:FNO"
    assert jnp.allclose(jnp.asarray(payload["values"]), expected)


def test_operator_artifact_checksum_and_task_fingerprint_fail_closed(tmp_path):
    phx.nn.operator.training.save_operator_artifact(tmp_path, _trained())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = tmp_path / manifest["execution_model_file"]
    model_path.write_bytes(model_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="model checksum"):
        phx.nn.operator.training.load_trained_operator(tmp_path)

    phx.nn.operator.training.save_operator_artifact(tmp_path, _trained())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="task fingerprint"):
        phx.nn.operator.training.load_trained_operator(tmp_path)


def test_external_checkpoint_enters_the_same_task_bound_runtime(tmp_path):
    checkpoint = tmp_path / "external.bin"
    checkpoint.write_bytes(b"verified-external-state")
    manifest = phx.nn.operator.adapters.OperatorCheckpointManifest(
        architecture="FNO",
        model_version="1.0.0",
        source_uri="https://example.test/source",
        checkpoint_uri="https://example.test/checkpoint",
        revision="immutable-revision",
        input_schema={"u": "case_x"},
        output_schema={"solution": "case_x"},
        preprocessing={"layout": "native"},
        normalization={"kind": "task-owned"},
        dataset_provenance=("immutable-dataset",),
        code_license="Apache-2.0",
        weights_license="CC-BY-4.0",
        checkpoint_sha256=phx.nn.operator.adapters.checkpoint_sha256(checkpoint),
    )
    manifest_path = tmp_path / "external.json"
    phx.nn.operator.adapters.save_operator_manifest(manifest_path, manifest)

    trained = phx.nn.operator.training.load_external_trained_operator(
        manifest_path,
        checkpoint,
        lambda external_manifest, checkpoint_path: lambda payload, key: 2.0 * payload,
        _task(),
        phx.nn.operator.OperatorTrainingEvidence(regime="task_specific"),
        input_adapter=lambda batch, external_manifest: batch.input("u").values,
        output_adapter=lambda output, batch, external_manifest: output,
        in_size="scalar",
        out_size="scalar",
        output_field_map={"output": "solution"},
    )
    values = _batch().input("u").values
    assert jnp.allclose(
        trained.predict(_batch()).field("solution").values,
        3.0 * values + 1.0,
    )


def test_training_checkpoint_uses_only_current_manifest(tmp_path):
    batch = _batch()
    target = jnp.zeros_like(batch.input("u").values)
    model = _trained().execution_model
    optimizer_state = {"momentum": jnp.ones((2,), dtype=float)}
    schema = phx.nn.operator.training.operator_batch_schema(
        batch,
        target=phx.nn.operator.OperatorTargetBatch.from_arrays(
            {"solution": target},
            batch,
        ),
    )
    checkpoint = phx.nn.operator.training.save_operator_training_checkpoint(
        tmp_path / "checkpoint",
        model,
        optimizer_state,
        step=7,
        key=jr.key(9),
        schema=schema,
    )

    restored = phx.nn.operator.training.load_operator_training_checkpoint(
        checkpoint,
        model,
        optimizer_state,
        expected_schema=schema,
    )
    assert restored.step == 7
    assert jnp.array_equal(restored.key, jr.key(9))
    assert jnp.array_equal(
        restored.optimizer_state["momentum"], optimizer_state["momentum"]
    )

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "format_version" not in manifest
    manifest["format_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="current canonical fields"):
        phx.nn.operator.training.load_operator_training_checkpoint(
            checkpoint,
            model,
            optimizer_state,
            expected_schema=schema,
        )
