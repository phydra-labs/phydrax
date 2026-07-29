#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _batch(*, cases=2, size=8):
    axis = phx.nn.OperatorAxis("x", jnp.arange(size, dtype=float) / size, periodic=True)
    return phx.nn.OperatorBatch(
        inputs={
            "u": phx.nn.FunctionSamples(
                values=jnp.arange(cases * size, dtype=float).reshape(cases, size),
                axes=(axis,),
            )
        },
        queries={"solution-query": phx.nn.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(cases,),
    )


def _task(*, revision="1"):
    return phx.nn.OperatorTask(
        "periodic-map",
        revision=revision,
        dimension_basis=("length",),
        fields=(
            phx.nn.OperatorFieldSpec(
                "input",
                role="source",
                source_name="u",
                physical_dimension=(0.0,),
                scale=2.0,
                offset=1.0,
            ),
            phx.nn.OperatorFieldSpec(
                "solution",
                role="target",
                query_name="solution-query",
                physical_dimension=(0.0,),
                scale=3.0,
                offset=4.0,
            ),
        ),
        queries=(
            phx.nn.OperatorQuerySpec(
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
    return phx.nn.OperatorTask(
        base.task_id,
        revision=base.revision,
        dimension_basis=base.dimension_basis,
        fields=base.fields,
        queries=(
            phx.nn.OperatorQuerySpec(
                "solution-query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
                fixed_geometry=True,
            ),
        ),
        problem=phx.nn.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=True,
        ),
        metadata=base.metadata,
    )


def _trained(*, revision="1", output_pipeline=None):
    model = phx.nn.FNO(
        n_modes=(3,),
        width=4,
        depth=1,
        coordinate_embedding=False,
        source_key="u",
        key=jr.key(12),
    )
    return phx.nn.TrainedOperator(
        model,
        _task(revision=revision),
        training_evidence=phx.nn.OperatorTrainingEvidence(regime="task_specific"),
        output_field_map={"output": "solution"},
        output_pipeline=output_pipeline,
        provenance={"dataset": "immutable-dataset"},
    )


def test_operator_task_is_canonical_and_rejects_unknown_sources():
    task = _task()
    restored = phx.nn.OperatorTask.from_dict(task.to_dict())

    assert restored.fingerprint == task.fingerprint
    assert len(task.fingerprint) == 64
    task.validate_batch(_batch())

    batch = _batch()
    with pytest.raises(ValueError, match="absent from the task"):
        task.validate_batch(
            phx.nn.OperatorBatch(
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
    nondimensional = (batch.input("u").values - 1.0) / 2.0

    assert jnp.allclose(prepared.execution_batch.input("u").values, nondimensional)
    raw = trained.execution_model.predict(prepared.execution_batch).field("output").values
    prediction = trained.predict_prepared(prepared).field("solution").values
    assert jnp.allclose(prediction, raw * 3.0 + 4.0)

    other = _trained(revision="2")
    with pytest.raises(ValueError, match="different runtime contract"):
        other.predict_prepared(prepared)


def test_normalized_output_pipeline_enforces_physical_conservation(tmp_path):
    pipeline = phx.nn.OperatorOutputPipeline(
        phx.nn.ConservationProjection("solution", source_name="u")
    )
    trained = _trained(output_pipeline=pipeline)
    batch = _batch()
    prediction = trained.predict(batch)
    source_total = phx.nn.operator_integral(
        batch.input("u").values,
        batch.input("u"),
        case_shape=batch.case_shape,
    )
    predicted_total = phx.nn.operator_integral(
        prediction.field("solution").values,
        batch.query("solution-query"),
        case_shape=batch.case_shape,
    )

    assert jnp.allclose(predicted_total, source_total)
    destination = phx.nn.save_operator_artifact(tmp_path, trained)
    restored = phx.nn.load_trained_operator(destination)
    restored_total = phx.nn.operator_integral(
        restored.predict(batch).field("solution").values,
        batch.query("solution-query"),
        case_shape=batch.case_shape,
    )
    manifest = phx.nn.load_operator_artifact_manifest(destination)
    assert jnp.allclose(restored_total, source_total)
    assert "format_version" not in manifest.to_dict()
    assert manifest.output_pipeline_fingerprint == pipeline.fingerprint
    assert restored.output_pipeline is not None
    assert restored.output_pipeline.fingerprint == pipeline.fingerprint


def test_trained_operator_preserves_multiple_named_outputs_and_queries():
    source_coordinates = jnp.linspace(0.0, 1.0, 4)[:, None]
    batch = phx.nn.OperatorBatch(
        inputs={
            "u": phx.nn.FunctionSamples(
                values=jnp.linspace(-1.0, 1.0, 4),
                coordinates=source_coordinates,
            )
        },
        queries={
            "spatial": phx.nn.FunctionSamples(
                values=None,
                coordinates=jnp.linspace(0.0, 1.0, 3)[:, None],
            ),
            "sensors": phx.nn.FunctionSamples(
                values=None,
                coordinates=jnp.asarray([[0.25], [0.75]]),
            ),
        },
    )
    targets = (
        ("state", "spatial", phx.nn.OperatorOutputSpec("scalar")),
        (
            "flux",
            "sensors",
            phx.nn.OperatorOutputSpec(2, component_names=("x", "y")),
        ),
    )
    graph = phx.nn.OperatorBranchGraph(
        tuple(
            phx.nn.OperatorBranchSpec(
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
    model = phx.nn.ABUPT(
        graph,
        input_channels={"state": "scalar", "flux": "scalar"},
        coord_dims={"state": 1, "flux": 1},
        anchor_counts={"state": 2, "flux": 2},
        width=4,
        depth=1,
        num_heads=1,
        key=jr.key(41),
    )
    task = phx.nn.OperatorTask(
        "coupled-map",
        revision="1",
        dimension_basis=("length",),
        fields=(
            phx.nn.OperatorFieldSpec(
                "input",
                role="source",
                source_name="u",
                physical_dimension=(0.0,),
            ),
            *tuple(
                phx.nn.OperatorFieldSpec(
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
            phx.nn.OperatorQuerySpec(
                name,
                geometry_kind="point_cloud",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
            )
            for name in ("spatial", "sensors")
        ),
    )
    trained = phx.nn.TrainedOperator(
        model,
        task,
        training_evidence=phx.nn.OperatorTrainingEvidence(regime="task_specific"),
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
    trained = phx.nn.TrainedOperator(
        _trained().execution_model,
        task,
        training_evidence=phx.nn.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "solution"},
        fixed_query_fingerprints={"solution-query": fingerprint},
    )

    coordinates = jnp.broadcast_to(
        batch.query("solution-query").coordinates_array(
            case_shape=batch.case_shape,
        ),
        batch.case_shape + batch.query("solution-query").sample_shape + (1,),
    )
    case_dependent = phx.nn.OperatorBatch(
        inputs=batch.inputs,
        queries={
            "solution-query": phx.nn.FunctionSamples(
                values=None,
                coordinates=coordinates.at[1].add(0.125),
            )
        },
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    case_task = phx.nn.OperatorTask(
        "case-dependent-fixed-query",
        dimension_basis=("length",),
        fields=task.fields,
        queries=(
            phx.nn.OperatorQuerySpec(
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

    altered_axis = phx.nn.OperatorAxis(
        "x",
        batch.query("solution-query").axes[0].nodes + 0.01,
        periodic=True,
    )
    altered = phx.nn.OperatorBatch(
        inputs=batch.inputs,
        queries={
            "solution-query": phx.nn.FunctionSamples(
                values=None,
                axes=(altered_axis,),
            )
        },
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    with pytest.raises(ValueError, match="different physical geometry"):
        trained.predict(altered)

    destination = phx.nn.save_operator_artifact(tmp_path, trained)
    restored = phx.nn.load_trained_operator(destination)
    assert dict(restored.fixed_query_fingerprints) == {"solution-query": fingerprint}
    with pytest.raises(ValueError, match="different physical geometry"):
        restored.predict(altered)


def test_operator_task_serialization_rejects_noncanonical_fields():
    payload = _task().to_dict()
    payload["schema_version"] = 1

    with pytest.raises(ValueError, match="current canonical fields"):
        phx.nn.OperatorTask.from_dict(payload)


def test_operator_artifact_manifest_rejects_noncanonical_fields(tmp_path):
    phx.nn.save_operator_artifact(tmp_path, _trained())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="current canonical fields"):
        phx.nn.load_trained_operator(tmp_path)


def test_portable_operator_artifact_round_trips_inference_and_training_state(tmp_path):
    trained = _trained()
    batch = _batch()
    destination = phx.nn.save_operator_artifact(
        tmp_path,
        trained,
        training_state={"step": jnp.asarray(7), "loss": jnp.asarray(0.25)},
        training_metadata={"epoch": 2},
    )

    restored = phx.nn.load_trained_operator(destination)
    resume = phx.nn.load_operator_training_state(destination)

    assert restored.task.fingerprint == trained.task.fingerprint
    assert restored.contract_fingerprint == trained.contract_fingerprint
    assert jnp.allclose(
        restored.predict(batch).field("solution").values,
        trained.predict(batch).field("solution").values,
    )
    assert int(resume.state["step"]) == 7
    assert resume.metadata == {"epoch": 2}


def test_operator_artifact_checksum_and_task_fingerprint_fail_closed(tmp_path):
    phx.nn.save_operator_artifact(tmp_path, _trained())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = tmp_path / manifest["execution_model_file"]
    model_path.write_bytes(model_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="model checksum"):
        phx.nn.load_trained_operator(tmp_path)

    phx.nn.save_operator_artifact(tmp_path, _trained())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="task fingerprint"):
        phx.nn.load_trained_operator(tmp_path)


def test_external_checkpoint_enters_the_same_task_bound_runtime(tmp_path):
    checkpoint = tmp_path / "external.bin"
    checkpoint.write_bytes(b"verified-external-state")
    manifest = phx.nn.OperatorCheckpointManifest(
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
        checkpoint_sha256=phx.nn.checkpoint_sha256(checkpoint),
    )
    manifest_path = tmp_path / "external.json"
    phx.nn.save_operator_manifest(manifest_path, manifest)

    trained = phx.nn.load_external_trained_operator(
        manifest_path,
        checkpoint,
        lambda external_manifest, checkpoint_path: lambda payload, key: 2.0 * payload,
        _task(),
        phx.nn.OperatorTrainingEvidence(regime="task_specific"),
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
    schema = phx.nn.operator_batch_schema(
        batch,
        target=phx.nn.OperatorTargetBatch.from_arrays(
            {"solution": target},
            batch,
        ),
    )
    checkpoint = phx.nn.save_operator_training_checkpoint(
        tmp_path / "checkpoint",
        model,
        optimizer_state,
        step=7,
        key=jr.key(9),
        schema=schema,
    )

    restored = phx.nn.load_operator_training_checkpoint(
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
        phx.nn.load_operator_training_checkpoint(
            checkpoint,
            model,
            optimizer_state,
            expected_schema=schema,
        )
