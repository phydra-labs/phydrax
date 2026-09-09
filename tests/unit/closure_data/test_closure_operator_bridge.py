from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


class _StressOperator(phx.nn.operator.AbstractOperatorModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 2
        self.out_size = 9

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("features").values
        assert values is not None
        diagonal = values[..., 0]
        zero = jnp.zeros_like(diagonal)
        stress = jnp.stack(
            (
                jnp.stack((diagonal, zero, zero), axis=-1),
                jnp.stack((zero, -diagonal, zero), axis=-1),
                jnp.stack((zero, zero, zero), axis=-1),
            ),
            axis=-2,
        )
        return stress.reshape(stress.shape[:-2] + (9,))

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _task_and_template():
    axis = phx.nn.operator.OperatorAxis("x", jnp.asarray((0.0, 1.0)))
    template = phx.nn.operator.OperatorBatch(
        inputs={
            "features": phx.nn.operator.FunctionSamples(
                values=jnp.zeros((2, 2), dtype=jnp.float32),
                axes=(axis,),
            )
        },
        queries={"grid": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
    )
    task = phx.nn.operator.OperatorTask(
        "closure-stress",
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "features",
                channels=2,
                role="source",
                component_names=("s_xx", "s_xy"),
            ),
            phx.nn.operator.OperatorFieldSpec(
                "stress",
                channels=9,
                role="target",
                query_name="grid",
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "grid",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
            ),
        ),
    )
    return task, template


def _closure_inputs():
    node = phx.closure_data.ClosureAnalysisNode(
        "reynolds_sgs_stress",
        ("resolved",),
        output_name="stress",
        output_units="(m/s)^2",
    )
    dag = phx.closure_data.ClosureAnalysisDAG(("resolved",), (node,))
    cases = []
    extents = []
    chunks = []
    assignments = []
    plan = phx.closure_data.LeakageSafePartitionPlan(
        "case",
        train_fraction=0.34,
        validation_fraction=0.33,
        test_fraction=0.33,
        salt="closure-operator-test",
    )
    split_names = ("train", "validation", "test")
    for index, split in enumerate(split_names):
        key = phx.closure_data.ClosureSampleKey(
            case_id=f"case-{index}",
            trajectory_id=f"trajectory-{index}",
            realization_id="realization",
            time_block_id="block",
            time_index=0,
        )
        sample = phx.closure_data.ClosureSample(
            jnp.asarray(
                ((float(index + 1), 0.0), (float(index + 2), 0.0)),
                dtype=jnp.float32,
            ),
            key,
            schema_id="flow-schema",
        )
        diagonal = sample.values[..., 0]
        zero = jnp.zeros_like(diagonal)
        stress = jnp.stack(
            (
                jnp.stack((diagonal, zero, zero), axis=-1),
                jnp.stack((zero, -diagonal, zero), axis=-1),
                jnp.stack((zero, zero, zero), axis=-1),
            ),
            axis=-2,
        )
        target = phx.closure_data.ClosureTarget(
            stress.reshape((2, 9)),
            node,
            target_kind="sgs_stress",
            schema_id="flow-schema",
        )
        cases.append(
            phx.closure_data.ClosureOperatorCase(
                {"features": sample},
                {"stress": target},
            )
        )
        extent = phx.closure_data.DatasetExtent(
            case_id=key.case_id,
            trajectory_id=key.trajectory_id,
            realization_id=key.realization_id,
            time_block_id=key.time_block_id,
            sample_count=1,
        )
        extents.append(extent)
        chunks.append(
            phx.closure_data.ClosureDatasetChunk.from_payload(
                bytes((index,)),
                extent_id=extent.extent_id,
                logical_name=f"case-{index}",
                chunk_index=0,
                sample_start=0,
                sample_stop=1,
                byte_offset=0,
            )
        )
        assignments.append(
            phx.closure_data.PartitionAssignment(
                sample_id=sample.sample_id,
                group_key=key.group_key("case"),
                split=split,
            )
        )
    partition = phx.closure_data.LeakageSafePartition(plan, tuple(assignments))
    manifest = phx.closure_data.ChunkedClosureDatasetManifest(
        dataset_id="closure-dataset",
        schema_id="flow-schema",
        analysis_dag_id=dag.dag_id,
        extents=tuple(extents),
        chunks=tuple(chunks),
    )
    normalizer = phx.closure_data.TrainOnlyNormalizer(
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        phx.closure_data.NormalizerProvenance(
            partition_id=partition.partition_id,
            training_assignment_ids=(assignments[0].assignment_id,),
            training_sample_ids=(cases[0].inputs["features"].sample_id,),
            feature_name="features",
            schema_id="flow-schema",
        ),
        epsilon=1e-6,
    )
    return tuple(cases), dag, manifest, partition, normalizer


def _datasets():
    task, template = _task_and_template()
    cases, dag, manifest, partition, normalizer = _closure_inputs()
    datasets = phx.closure_data.prepare_closure_operator_datasets(
        cases,
        template,
        task,
        manifest,
        dag,
        partition,
        normalizers={"features": normalizer},
    )
    return task, template, cases, normalizer, datasets


def test_closure_partitions_and_provenance_remain_authoritative():
    task, template, cases, _, datasets = _datasets()

    assert datasets.train.size == 1
    assert datasets.validation is not None and datasets.validation.size == 1
    assert datasets.test is not None and datasets.test.size == 1
    assert datasets.task_fingerprint == task.fingerprint
    assert datasets.train.provenance[0].case_id == cases[0].case_id
    assert (
        datasets.validation.provenance[0].identities["closure_partition"]
        == datasets.partition_id
    )
    assert datasets.test.provenance[0].order["time_index"] == 0.0
    np.testing.assert_allclose(
        datasets.train.batch.input("features").axes[0].nodes,
        template.input("features").axes[0].nodes,
    )


def test_closure_preparation_rejects_schema_and_partition_mismatches():
    task, template = _task_and_template()
    cases, dag, manifest, partition, normalizer = _closure_inputs()
    wrong = phx.closure_data.ClosureSample(
        cases[0].inputs["features"].values,
        cases[0].key,
        schema_id="other-schema",
    )
    with pytest.raises(ValueError, match="share one schema"):
        phx.closure_data.ClosureOperatorCase(
            {"features": wrong},
            cases[0].targets,
        )
    with pytest.raises(ValueError, match="normalizer provenance"):
        bad_normalizer = phx.closure_data.TrainOnlyNormalizer(
            normalizer.mean,
            normalizer.scale,
            phx.closure_data.NormalizerProvenance(
                partition_id="other-partition",
                training_assignment_ids=("assignment",),
                training_sample_ids=(wrong.sample_id,),
                feature_name="features",
                schema_id="other-schema",
            ),
            epsilon=1e-6,
        )
        phx.closure_data.prepare_closure_operator_datasets(
            cases,
            template,
            task,
            manifest,
            dag,
            partition,
            normalizers={"features": bad_normalizer},
        )


def test_trained_operator_binds_through_the_existing_stress_policy():
    task, template, _, normalizer, datasets = _datasets()
    resolved_filter = phx.equations.ResolvedLESFilter(
        "cell filter",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="volume-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="unmodeled",
    )
    parameter_provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        "mesh",
        "constant-density-incompressible",
        source_kind="user",
        evidence_ids=(),
    )
    artifact_id = "trained-stress-artifact"
    plan = phx.closure_data.LearnedStressBindingPlan(
        phx.closure_data.LearnedStressFeatureSchema(
            name="features",
            component_names=("s_xx", "s_xy"),
            component_units=("1/s", "1/s"),
            shape=(2, 2),
            dtype=jnp.float32,
            flow_schema_id="flow-schema",
        ),
        phx.closure_data.LearnedStressOutputContract(
            shape=(2, 3, 3),
            dtype=jnp.float32,
            units="(m/s)^2",
            target_id="stress-target",
            filter_id=resolved_filter.filter_id,
            discretization_id="mesh",
            regime="constant-density-incompressible",
            stress_convention="deviatoric",
            symmetry_tolerance=1e-6,
            trace_tolerance=1e-6,
        ),
        resolved_filter,
        parameter_provenance,
        model_artifact_id=artifact_id,
        normalizer_id=normalizer.normalizer_id,
    )
    provenance = {
        "closure_dataset_manifest": datasets.dataset_manifest_id,
        "closure_partition": datasets.partition_id,
        "closure_analysis_dag": datasets.analysis_dag_id,
        "closure_flow_schema": datasets.flow_schema_id,
        "closure_normalizer": normalizer.normalizer_id,
        "closure_preparation": datasets.preparation_id,
    }
    trained = phx.nn.operator.training.TrainedOperator(
        _StressOperator(),
        task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "stress"},
        artifact_id=artifact_id,
        provenance=provenance,
    )
    prepared = phx.closure_data.bind_trained_stress_operator(
        trained,
        plan,
        normalizer,
        template,
        datasets,
        source_name="features",
        target_name="stress",
    )
    features = jnp.asarray(((1.0, 0.0), (2.0, 0.0)), dtype=jnp.float32)
    strain = jnp.asarray(
        (
            ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 0.0)),
            ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 0.0)),
        ),
        dtype=jnp.float32,
    )
    result = eqx.filter_jit(prepared)(features, strain)

    np.testing.assert_allclose(result.stress[..., 0, 0], (1.0, 2.0))
    np.testing.assert_allclose(result.stress[..., 1, 1], (-1.0, -2.0))
    assert bool(result.evidence.valid)

    with pytest.raises(ValueError, match="provenance"):
        phx.closure_data.bind_trained_stress_operator(
            phx.nn.operator.training.TrainedOperator(
                _StressOperator(),
                task,
                training_evidence=phx.nn.operator.OperatorTrainingEvidence(
                    "task_specific"
                ),
                output_field_map={"output": "stress"},
                artifact_id=artifact_id,
                provenance=provenance | {"closure_partition": "other"},
            ),
            plan,
            normalizer,
            template,
            datasets,
            source_name="features",
            target_name="stress",
        )
