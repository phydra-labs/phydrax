#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..nn.operator.data import (
    function_samples_with_values,
    OperatorBatch,
    OperatorCaseProvenance,
    OperatorTargetBatch,
)
from ..nn.operator.task import OperatorTask
from ..nn.operator.training._dataset import (
    operator_dataset_from_cases,
    OperatorDataset,
)
from ..nn.operator.training._execution import PreparedOperatorInput
from ..nn.operator.training._trained_operator import TrainedOperator
from ._analysis import ClosureAnalysisDAG, ClosureTarget
from ._binding import LearnedStressBindingPlan, PreparedLearnedStressBinding
from ._dataset import (
    ChunkedClosureDatasetManifest,
    ClosureSample,
    DatasetSplit,
    LeakageSafePartition,
    TrainOnlyNormalizer,
)
from ._les import LESAnalysisReference


@dataclass(frozen=True)
class ClosureOperatorCase:
    """One explicitly aligned closure-learning case."""

    inputs: Mapping[str, ClosureSample]
    targets: Mapping[str, ClosureTarget]
    references: Mapping[str, LESAnalysisReference] = field(default_factory=dict)

    def __post_init__(self):
        inputs = frozendict({str(name): value for name, value in self.inputs.items()})
        targets = frozendict({str(name): value for name, value in self.targets.items()})
        references = frozendict(
            {str(name): value for name, value in self.references.items()}
        )
        if not inputs or not targets:
            raise ValueError("Closure operator cases require inputs and targets.")
        if any(not name for name in (*inputs, *targets, *references)):
            raise ValueError("Closure operator field names must be non-empty.")
        if any(not isinstance(value, ClosureSample) for value in inputs.values()):
            raise TypeError("Closure operator inputs must be ClosureSample values.")
        if any(not isinstance(value, ClosureTarget) for value in targets.values()):
            raise TypeError("Closure operator targets must be ClosureTarget values.")
        if any(
            not isinstance(value, LESAnalysisReference) for value in references.values()
        ):
            raise TypeError(
                "Closure operator references must be LESAnalysisReference values."
            )
        keys = {value.key.sample_id for value in inputs.values()}
        if len(keys) != 1:
            raise ValueError("All closure operator inputs must share one sample key.")
        schemas = {
            *(value.schema_id for value in inputs.values()),
            *(value.schema_id for value in targets.values()),
        }
        if len(schemas) != 1:
            raise ValueError("Closure operator inputs and targets must share one schema.")
        if set(references) - set(targets):
            raise ValueError(
                "Closure references must name closure targets in the same case."
            )
        for name, reference in references.items():
            if reference.target_id != targets[name].target_id:
                raise ValueError(
                    "Closure target and LES reference identities do not match."
                )
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "references", references)

    @property
    def key(self):
        return next(iter(self.inputs.values())).key

    @property
    def schema_id(self) -> str:
        return next(iter(self.inputs.values())).schema_id

    @property
    def case_id(self) -> str:
        return self.key.sample_id

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "closure-operator-case",
                "key": self.key.sample_id,
                "inputs": {name: value.sample_id for name, value in self.inputs.items()},
                "targets": {
                    name: value.target_id for name, value in self.targets.items()
                },
                "references": {
                    name: value.reference_id for name, value in self.references.items()
                },
            }
        )


@dataclass(frozen=True)
class ClosureOperatorDatasets:
    """Authoritative closure partitions represented as operator datasets."""

    train: OperatorDataset
    validation: OperatorDataset | None
    test: OperatorDataset | None
    dataset_manifest_id: str
    partition_id: str
    analysis_dag_id: str
    flow_schema_id: str
    task_fingerprint: str
    preparation_id: str


def _extent_for_case(
    case: ClosureOperatorCase,
    manifest: ChunkedClosureDatasetManifest,
    /,
):
    key = case.key
    matches = tuple(
        extent
        for extent in manifest.extents
        if (
            extent.case_id,
            extent.trajectory_id,
            extent.realization_id,
            extent.time_block_id,
        )
        == (
            key.case_id,
            key.trajectory_id,
            key.realization_id,
            key.time_block_id,
        )
    )
    if len(matches) != 1:
        raise ValueError("Closure operator case has no unique dataset extent.")
    extent = matches[0]
    if key.time_index >= extent.sample_count:
        raise ValueError("Closure operator case time_index exceeds its dataset extent.")
    return extent


def _case_provenance(
    case: ClosureOperatorCase,
    /,
    *,
    manifest: ChunkedClosureDatasetManifest,
    partition: LeakageSafePartition,
    analysis: ClosureAnalysisDAG,
) -> OperatorCaseProvenance:
    key = case.key
    identities = {
        "closure_case": key.case_id,
        "closure_trajectory": key.trajectory_id,
        "closure_realization": key.realization_id,
        "closure_time_block": key.time_block_id,
        "closure_dataset_manifest": manifest.dataset_id,
        "closure_partition": partition.partition_id,
        "closure_analysis_dag": analysis.dag_id,
        "flow_schema": case.schema_id,
    }
    identities.update(
        {
            f"closure_target:{name}": target.target_id
            for name, target in case.targets.items()
        }
    )
    identities.update(
        {
            f"closure_reference:{name}": reference.reference_id
            for name, reference in case.references.items()
        }
    )
    return OperatorCaseProvenance(
        case.case_id,
        identities=identities,
        order={"time_index": float(key.time_index)},
    )


def _materialize_operator_case(
    case: ClosureOperatorCase,
    template: OperatorBatch,
    task: OperatorTask,
    /,
    *,
    normalizers: Mapping[str, TrainOnlyNormalizer],
) -> tuple[OperatorBatch, OperatorTargetBatch]:
    if set(case.inputs) != {field.name for field in task.source_fields}:
        raise ValueError("Closure inputs must match the task source fields exactly.")
    if set(case.targets) != {field.name for field in task.target_fields}:
        raise ValueError("Closure targets must match the task target fields exactly.")
    inputs = dict(template.inputs)
    for field_spec in task.source_fields:
        assert field_spec.source_name is not None
        if field_spec.source_name not in inputs:
            raise KeyError(
                f"Operator template is missing task source {field_spec.source_name!r}."
            )
        sample = case.inputs[field_spec.name]
        values = sample.values
        if field_spec.name in normalizers:
            values = normalizers[field_spec.name].normalize(values)
        inputs[field_spec.source_name] = function_samples_with_values(
            inputs[field_spec.source_name], values
        )
    batch = OperatorBatch(
        inputs=inputs,
        queries=template.queries,
        case_axes=template.case_axes,
        case_shape=template.case_shape,
    )
    task.validate_batch(batch)
    target_values = {name: target.values for name, target in case.targets.items()}
    query_names = {}
    specs = {}
    for field_spec in task.target_fields:
        assert field_spec.query_name is not None
        assert field_spec.output_spec is not None
        query_names[field_spec.name] = field_spec.query_name
        specs[field_spec.name] = field_spec.output_spec
    targets = OperatorTargetBatch.from_arrays(
        target_values,
        batch,
        query_names=query_names,
        specs=specs,
    )
    return batch, targets


def prepare_closure_operator_datasets(
    cases: Sequence[ClosureOperatorCase],
    template: OperatorBatch,
    task: OperatorTask,
    manifest: ChunkedClosureDatasetManifest,
    analysis: ClosureAnalysisDAG,
    partition: LeakageSafePartition,
    /,
    *,
    normalizers: Mapping[str, TrainOnlyNormalizer] | None = None,
) -> ClosureOperatorDatasets:
    """Bind closure cases to their authoritative split and operator geometry."""

    values = tuple(cases)
    if not values or any(not isinstance(case, ClosureOperatorCase) for case in values):
        raise ValueError("At least one ClosureOperatorCase is required.")
    if not isinstance(template, OperatorBatch) or template.case_shape:
        raise ValueError("Closure operator templates must have no case axes.")
    if not isinstance(task, OperatorTask):
        raise TypeError("task must be an OperatorTask.")
    if not isinstance(manifest, ChunkedClosureDatasetManifest):
        raise TypeError("manifest must be a ChunkedClosureDatasetManifest.")
    if not isinstance(analysis, ClosureAnalysisDAG):
        raise TypeError("analysis must be a ClosureAnalysisDAG.")
    if not isinstance(partition, LeakageSafePartition):
        raise TypeError("partition must be a LeakageSafePartition.")
    if len({case.case_id for case in values}) != len(values):
        raise ValueError("Closure operator cases must have unique sample keys.")
    if any(case.schema_id != manifest.schema_id for case in values):
        raise ValueError("Closure operator case schema does not match the manifest.")
    if manifest.analysis_dag_id != analysis.dag_id:
        raise ValueError("Closure manifest and analysis DAG identities do not match.")
    source_names = {field.name for field in task.source_fields}
    target_names = {field.name for field in task.target_fields}
    if any(set(case.inputs) != source_names for case in values):
        raise ValueError("Closure inputs must match the task source fields exactly.")
    if any(set(case.targets) != target_names for case in values):
        raise ValueError("Closure targets must match the task target fields exactly.")
    assignment_by_sample = {
        assignment.sample_id: assignment for assignment in partition.assignments
    }
    task.validate_batch(template)
    analysis_nodes = {node.node_id for node in analysis.nodes}
    normalizer_map = frozendict(
        {}
        if normalizers is None
        else {str(name): value for name, value in normalizers.items()}
    )
    if set(normalizer_map) - set(task.field_by_name):
        raise KeyError("Closure normalizers contain fields absent from the task.")
    for name, normalizer in normalizer_map.items():
        if not isinstance(normalizer, TrainOnlyNormalizer):
            raise TypeError(
                "Closure input normalizers must be TrainOnlyNormalizer values."
            )
        if not task.field_by_name[name].is_source:
            raise ValueError("Closure normalizers may only bind source fields.")
        field_spec = task.field_by_name[name]
        if any(value != 1.0 for value in field_spec.scale) or any(
            value != 0.0 for value in field_spec.offset
        ):
            raise ValueError(
                "Closure-normalized inputs require identity task affine scaling."
            )
        if (
            normalizer.provenance.partition_id != partition.partition_id
            or normalizer.provenance.schema_id != manifest.schema_id
            or normalizer.provenance.feature_name != name
        ):
            raise ValueError("Closure normalizer provenance does not match preparation.")
        field_assignments = tuple(
            assignment_by_sample[case.inputs[name].sample_id] for case in values
        )
        expected_assignments = tuple(
            assignment for assignment in field_assignments if assignment.split == "train"
        )
        if set(normalizer.provenance.training_sample_ids) != {
            assignment.sample_id for assignment in expected_assignments
        } or set(normalizer.provenance.training_assignment_ids) != {
            assignment.assignment_id for assignment in expected_assignments
        }:
            raise ValueError(
                "Closure normalizer was not fitted on the complete authoritative "
                "training split."
            )

    split_batches: dict[DatasetSplit, list[OperatorBatch]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_targets: dict[DatasetSplit, list[OperatorTargetBatch]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_provenance: dict[DatasetSplit, list[OperatorCaseProvenance]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    case_fingerprints = []
    for case in values:
        _extent_for_case(case, manifest)
        if any(
            target.node.node_id not in analysis_nodes for target in case.targets.values()
        ):
            raise ValueError("Closure target is absent from the supplied analysis DAG.")
        if any(
            reference.analysis_dag_id != analysis.dag_id
            for reference in case.references.values()
        ):
            raise ValueError("LES reference analysis DAG does not match preparation.")
        assignments = tuple(
            assignment_by_sample[sample.sample_id] for sample in case.inputs.values()
        )
        split_names = {assignment.split for assignment in assignments}
        if len(split_names) != 1:
            raise ValueError("Aligned closure inputs cannot cross dataset partitions.")
        split = next(iter(split_names))
        batch, targets = _materialize_operator_case(
            case,
            template,
            task,
            normalizers=normalizer_map,
        )
        split_batches[split].append(batch)
        split_targets[split].append(targets)
        split_provenance[split].append(
            _case_provenance(
                case,
                manifest=manifest,
                partition=partition,
                analysis=analysis,
            )
        )
        case_fingerprints.append(case.fingerprint)

    def materialize(split: DatasetSplit) -> OperatorDataset | None:
        if not split_batches[split]:
            return None
        return operator_dataset_from_cases(
            split_batches[split],
            split_targets[split],
            provenance=split_provenance[split],
        )

    train = materialize("train")
    if train is None:
        raise ValueError("Closure operator preparation produced an empty train split.")
    preparation_id = canonical_fingerprint(
        {
            "kind": "closure-operator-datasets",
            "cases": case_fingerprints,
            "template": {
                "inputs": {
                    name: value.support_id for name, value in template.inputs.items()
                },
                "queries": {
                    name: value.support_id for name, value in template.queries.items()
                },
            },
            "task": task.fingerprint,
            "manifest": manifest.dataset_id,
            "partition": partition.partition_id,
            "analysis": analysis.dag_id,
            "normalizers": {
                name: value.normalizer_id for name, value in normalizer_map.items()
            },
        }
    )
    return ClosureOperatorDatasets(
        train=train,
        validation=materialize("validation"),
        test=materialize("test"),
        dataset_manifest_id=manifest.dataset_id,
        partition_id=partition.partition_id,
        analysis_dag_id=analysis.dag_id,
        flow_schema_id=manifest.schema_id,
        task_fingerprint=task.fingerprint,
        preparation_id=preparation_id,
    )


class TrainedClosureOperatorPredictor(StrictModule, NonTrainableState):
    """Prepared fixed-geometry operator exposed through the stress-predictor ABI."""

    trained: TrainedOperator
    prepared: PreparedOperatorInput
    source_name: str
    target_name: str
    output_shape: tuple[int, ...] | None
    predictor_id: str

    def __init__(
        self,
        trained: TrainedOperator,
        template: OperatorBatch,
        /,
        *,
        source_name: str,
        target_name: str,
        output_shape: Sequence[int] | None = None,
    ):
        if not isinstance(trained, TrainedOperator):
            raise TypeError("trained must be a TrainedOperator.")
        if not trained.artifact_id:
            raise ValueError("Trained closure operators require an artifact identity.")
        if trained.normalization is not None:
            raise ValueError(
                "Closure-trained operators must not apply a second normalization."
            )
        if trained.output_pipeline is not None:
            raise ValueError(
                "Closure-trained operators must leave stress policy enforcement "
                "to the learned-stress binding."
            )
        if template.case_shape:
            raise ValueError("Closure predictor templates must have no case axes.")
        source = str(source_name)
        target = str(target_name)
        if len(trained.task.source_fields) != 1:
            raise ValueError(
                "Closure predictors initially support exactly one source field."
            )
        source_field = trained.task.source_fields[0]
        if source_field.name != source or source_field.source_name not in template.inputs:
            raise ValueError(
                "Closure predictor source does not match its task or template."
            )
        if target not in {field.name for field in trained.task.target_fields}:
            raise ValueError("Closure predictor target is absent from the operator task.")
        shape = (
            None if output_shape is None else tuple(int(size) for size in output_shape)
        )
        if shape is not None and (not shape or any(size <= 0 for size in shape)):
            raise ValueError(
                "Closure predictor output_shape must contain positive sizes."
            )
        self.trained = trained
        self.prepared = trained.prepare(template)
        self.source_name = source
        self.target_name = target
        self.output_shape = shape
        self.predictor_id = canonical_fingerprint(
            {
                "kind": "trained-closure-operator-predictor",
                "artifact": trained.artifact_id,
                "task": trained.task_fingerprint,
                "contract": trained.contract_fingerprint,
                "source": source,
                "target": target,
                "output_shape": shape,
                "physical_support": self.prepared.physical_batch.input(
                    source_field.source_name
                ).support_id,
            }
        )

    def __call__(self, normalized: Any, args: Any = None, /):
        if args is not None:
            raise ValueError(
                "Trained closure operator predictors do not accept hidden runtime args."
            )
        source_field = self.trained.task.field_by_name[self.source_name]
        assert source_field.source_name is not None
        value = jnp.asarray(normalized)
        physical_source = self.prepared.physical_batch.input(source_field.source_name)
        execution_source = self.prepared.execution_batch.input(source_field.source_name)
        if physical_source.values is None or execution_source.values is None:
            raise ValueError("Closure predictor templates require source values.")
        if value.shape != physical_source.values.shape:
            raise ValueError(
                "Closure predictor input does not match the bound feature shape."
            )
        physical_inputs = dict(self.prepared.physical_batch.inputs)
        physical_inputs[source_field.source_name] = function_samples_with_values(
            physical_source, value
        )
        execution_value = source_field.nondimensionalize(value).astype(
            execution_source.values.dtype
        )
        execution_inputs = dict(self.prepared.execution_batch.inputs)
        execution_inputs[source_field.source_name] = function_samples_with_values(
            execution_source, execution_value
        )
        physical_batch = OperatorBatch(
            inputs=physical_inputs,
            queries=self.prepared.physical_batch.queries,
            case_axes=(),
            case_shape=(),
        )
        execution_batch = OperatorBatch(
            inputs=execution_inputs,
            queries=self.prepared.execution_batch.queries,
            case_axes=(),
            case_shape=(),
        )
        prediction = self.trained.predict_prepared(
            PreparedOperatorInput(
                physical_batch,
                execution_batch,
                plan_fingerprint=self.prepared.plan_fingerprint,
            )
        )
        values = prediction.field(self.target_name).values
        return values if self.output_shape is None else values.reshape(self.output_shape)


def bind_trained_stress_operator(
    trained: TrainedOperator,
    plan: LearnedStressBindingPlan,
    normalizer: TrainOnlyNormalizer,
    template: OperatorBatch,
    datasets: ClosureOperatorDatasets,
    /,
    *,
    source_name: str,
    target_name: str,
) -> PreparedLearnedStressBinding:
    """Bind a trained operator to learned stress after exact provenance checks."""

    if not isinstance(plan, LearnedStressBindingPlan):
        raise TypeError("plan must be a LearnedStressBindingPlan.")
    if not isinstance(normalizer, TrainOnlyNormalizer):
        raise TypeError("normalizer must be a TrainOnlyNormalizer.")
    if not isinstance(datasets, ClosureOperatorDatasets):
        raise TypeError("datasets must be ClosureOperatorDatasets.")
    if trained.artifact_id != plan.model_artifact_id:
        raise ValueError("Trained operator artifact does not match the stress plan.")
    if trained.task_fingerprint != datasets.task_fingerprint:
        raise ValueError("Trained operator task does not match the closure datasets.")
    expected = {
        "closure_dataset_manifest": datasets.dataset_manifest_id,
        "closure_partition": datasets.partition_id,
        "closure_analysis_dag": datasets.analysis_dag_id,
        "closure_flow_schema": datasets.flow_schema_id,
        "closure_normalizer": normalizer.normalizer_id,
        "closure_preparation": datasets.preparation_id,
    }
    if any(trained.provenance.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "Trained operator provenance does not match the closure preparation."
        )
    predictor = TrainedClosureOperatorPredictor(
        trained,
        template,
        source_name=source_name,
        target_name=target_name,
        output_shape=plan.output_contract.shape,
    )
    return plan.prepare(
        predictor,
        normalizer,
        model_artifact_id=trained.artifact_id,
        target_id=plan.output_contract.target_id,
        output_units=plan.output_contract.units,
    )


__all__ = [
    "ClosureOperatorCase",
    "ClosureOperatorDatasets",
    "TrainedClosureOperatorPredictor",
    "bind_trained_stress_operator",
    "prepare_closure_operator_datasets",
]
