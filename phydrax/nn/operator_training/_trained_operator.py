#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ..models.core._base import _AbstractOperatorModel
from ..models.core._keys import EvalKey, split_eval_key
from ..models.core._operator import OperatorBatch, OperatorPrediction
from ..models.core._operator_capabilities import (
    ConfiguredOperatorContract,
    OperatorTrainingEvidence,
)
from ..models.core._operator_task import _freeze_json, OperatorTask
from ._dtype import OperatorDTypePolicy
from ._execution import nondimensionalize_batch, physicalize_prediction
from ._normalization import OperatorNormalizationPolicy
from ._physics import OperatorOutputPipeline


def _canonical_hash(value: Any, /) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def operator_contract_fingerprint(contract: ConfiguredOperatorContract, /) -> str:
    """Return a stable digest of one configured runtime/training contract."""
    return _canonical_hash(
        {
            "architecture": contract.architecture,
            "configuration": [list(item) for item in contract.configuration],
            "capabilities": asdict(contract.capabilities),
            "training": asdict(contract.training),
            "field_specs": [field.to_dict() for field in contract.field_specs],
        }
    )


def operator_normalization_fingerprint(
    normalization: OperatorNormalizationPolicy | None, /
) -> str:
    """Return a stable digest of normalization semantics, including the null policy."""
    return _canonical_hash(
        None if normalization is None else normalization.to_dict()
    )



class PreparedOperatorInput(StrictModule):
    """Validated physical and normalized execution batches for a trained operator."""

    physical_batch: OperatorBatch
    execution_batch: OperatorBatch
    task_fingerprint: str
    contract_fingerprint: str
    normalization_fingerprint: str

    def __init__(
        self,
        physical_batch: OperatorBatch,
        execution_batch: OperatorBatch,
        /,
        *,
        task_fingerprint: str,
        contract_fingerprint: str,
        normalization_fingerprint: str,
    ):
        self.physical_batch = physical_batch
        self.execution_batch = execution_batch
        self.task_fingerprint = str(task_fingerprint)
        self.contract_fingerprint = str(contract_fingerprint)
        self.normalization_fingerprint = str(normalization_fingerprint)


class TrainedOperator(StrictModule):
    """Task-bound operator with validated physical preprocessing and prediction."""

    execution_model: _AbstractOperatorModel
    task: OperatorTask
    contract: ConfiguredOperatorContract
    output_field_map: frozendict[str, str]
    fixed_query_fingerprints: frozendict[str, str]
    output_pipeline: OperatorOutputPipeline | None
    normalization: OperatorNormalizationPolicy | None
    dtype_policy: OperatorDTypePolicy
    training_evidence: OperatorTrainingEvidence
    artifact_id: str
    provenance: frozendict[str, Any]
    calibration: frozendict[str, Any]

    def __init__(
        self,
        execution_model: _AbstractOperatorModel,
        task: OperatorTask,
        /,
        *,
        training_evidence: OperatorTrainingEvidence,
        output_field_map: Mapping[str, str] | None = None,
        fixed_query_fingerprints: Mapping[str, str] | None = None,
        output_pipeline: OperatorOutputPipeline | None = None,
        normalization: OperatorNormalizationPolicy | None = None,
        dtype_policy: OperatorDTypePolicy | None = None,
        artifact_id: str = "",
        provenance: dict[str, Any] | None = None,
        calibration: dict[str, Any] | None = None,
    ):
        if not isinstance(execution_model, _AbstractOperatorModel):
            raise TypeError("TrainedOperator requires a PhydraX execution model.")
        if not isinstance(task, OperatorTask):
            raise TypeError("TrainedOperator requires an OperatorTask.")
        if not isinstance(training_evidence, OperatorTrainingEvidence):
            raise TypeError("training_evidence must be an OperatorTrainingEvidence.")
        if normalization is not None and not isinstance(
            normalization, OperatorNormalizationPolicy
        ):
            raise TypeError("normalization must be an OperatorNormalizationPolicy.")
        if output_pipeline is not None and not isinstance(
            output_pipeline, OperatorOutputPipeline
        ):
            raise TypeError("output_pipeline must be an OperatorOutputPipeline.")
        policy = OperatorDTypePolicy() if dtype_policy is None else dtype_policy
        if not isinstance(policy, OperatorDTypePolicy):
            raise TypeError("dtype_policy must be an OperatorDTypePolicy.")
        targets = task.target_fields
        declared = execution_model.operator_output_specs
        target_names = tuple(field.name for field in targets)
        if output_field_map is None:
            resolved_output_field_map = {
                name: name for name in declared if name in target_names
            }
        else:
            resolved_output_field_map = {
                str(model_name): str(target_name)
                for model_name, target_name in output_field_map.items()
            }
        if set(resolved_output_field_map) != set(declared):
            raise ValueError(
                "output_field_map must name every model output exactly; "
                f"expected {tuple(declared)!r}, got "
                f"{tuple(resolved_output_field_map)!r}."
            )
        if (
            len(set(resolved_output_field_map.values()))
            != len(resolved_output_field_map)
            or set(resolved_output_field_map.values()) != set(target_names)
        ):
            raise ValueError(
                "output_field_map must map bijectively onto the task target fields; "
                f"expected {target_names!r}, got "
                f"{tuple(resolved_output_field_map.values())!r}."
            )
        model_name_by_target = {
            target_name: model_name
            for model_name, target_name in resolved_output_field_map.items()
        }
        declared_specs = tuple(
            declared[model_name_by_target[field.name]] for field in targets
        )
        for target, model_spec in zip(targets, declared_specs, strict=True):
            target_spec = target.output_spec
            assert target_spec is not None
            if (
                model_spec.channels != target_spec.channels
                or model_spec.component_names != target_spec.component_names
            ):
                raise ValueError(
                    f"Model output contract for {target.name!r} disagrees with the task."
                )
        if output_pipeline is not None:
            unknown_pipeline_fields = {
                transform.field_name
                for transform in output_pipeline.transforms
                if transform.field_name not in target_names
            }
            if unknown_pipeline_fields:
                raise ValueError(
                    "Output pipeline transforms reference unknown task targets: "
                    f"{tuple(sorted(unknown_pipeline_fields))!r}."
                )
        cast_model = policy.cast_model(execution_model)
        resolved_fixed_queries = {
            str(name): str(fingerprint)
            for name, fingerprint in (
                {} if fixed_query_fingerprints is None else fixed_query_fingerprints
            ).items()
        }
        unknown_fixed_queries = set(resolved_fixed_queries) - set(task.query_by_name)
        if unknown_fixed_queries:
            raise ValueError(
                "Fixed query fingerprints reference unknown task queries: "
                f"{tuple(sorted(unknown_fixed_queries))!r}."
            )
        if task.problem.query_is_fixed is True and set(resolved_fixed_queries) != set(
            task.query_by_name
        ):
            raise ValueError(
                "Fixed-query tasks require a geometry fingerprint for every query."
            )
        contract = cast_model.operator_contract
        self.execution_model = cast_model
        self.task = task
        self.contract = contract
        self.output_field_map = frozendict(resolved_output_field_map)
        self.fixed_query_fingerprints = frozendict(resolved_fixed_queries)
        self.output_pipeline = output_pipeline
        self.normalization = normalization
        self.dtype_policy = policy
        self.training_evidence = training_evidence
        self.artifact_id = str(artifact_id)
        self.provenance = _freeze_json({} if provenance is None else provenance)
        self.calibration = _freeze_json({} if calibration is None else calibration)

    @property
    def task_fingerprint(self) -> str:
        return self.task.fingerprint

    @property
    def contract_fingerprint(self) -> str:
        return _canonical_hash(
            {
                "operator_contract": operator_contract_fingerprint(self.contract),
                "output_field_map": dict(self.output_field_map),
                "fixed_query_fingerprints": dict(self.fixed_query_fingerprints),
                "output_pipeline": (
                    None
                    if self.output_pipeline is None
                    else self.output_pipeline.fingerprint
                ),
            }
        )

    @property
    def normalization_fingerprint(self) -> str:
        return operator_normalization_fingerprint(self.normalization)

    def _nondimensionalize_batch(self, batch: OperatorBatch, /) -> OperatorBatch:
        return nondimensionalize_batch(batch, self.task)

    def prepare_prevalidated(self, batch: OperatorBatch, /) -> PreparedOperatorInput:
        """Transform a batch whose task and runtime contracts were checked on the host."""
        execution = self._nondimensionalize_batch(batch)
        if self.normalization is not None:
            execution = self.normalization.normalize_batch(execution)
        execution = self.dtype_policy.cast_batch(execution)
        return PreparedOperatorInput(
            batch,
            execution,
            task_fingerprint=self.task_fingerprint,
            contract_fingerprint=self.contract_fingerprint,
            normalization_fingerprint=self.normalization_fingerprint,
        )


    def prepare(self, batch: OperatorBatch, /) -> PreparedOperatorInput:
        """Validate and transform one physical batch outside the compiled hot path."""
        for name, expected in self.fixed_query_fingerprints.items():
            actual = batch.query(name).geometry_fingerprint()
            if actual != expected:
                raise ValueError(
                    f"Fixed query {name!r} has a different physical geometry."
                )
        self.task.validate_batch(batch)
        report = self.contract.validate(
            batch,
            problem=self.task.problem,
            training_evidence=self.training_evidence,
            fields=self.task.fields,
        )
        report.require()
        return self.prepare_prevalidated(batch)

    def predict_prepared(
        self,
        prepared: PreparedOperatorInput,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        """Execute one already-validated batch and restore physical output units."""
        if not isinstance(prepared, PreparedOperatorInput):
            raise TypeError("predict_prepared requires a PreparedOperatorInput.")
        expected = (
            self.task_fingerprint,
            self.contract_fingerprint,
            self.normalization_fingerprint,
        )
        actual = (
            prepared.task_fingerprint,
            prepared.contract_fingerprint,
            prepared.normalization_fingerprint,
        )
        if actual != expected:
            raise ValueError("Prepared operator input belongs to a different runtime contract.")
        model_key = key
        pipeline_key = key
        if self.output_pipeline is not None:
            model_key, pipeline_key = split_eval_key(key, 2)
        raw = self.execution_model.predict_prevalidated(
            prepared.execution_batch,
            key=model_key,
        )
        prediction = physicalize_prediction(
            raw,
            prepared.physical_batch,
            self.task,
            self.output_field_map,
            self.normalization,
        )
        if self.output_pipeline is not None:
            prediction = self.output_pipeline(
                prediction,
                prepared.physical_batch,
                key=pipeline_key,
            )
            self.task.validate_prediction(prediction)
        return prediction

    def predict(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        """Validate, execute, and return a physical metadata-bearing prediction."""
        return self.predict_prepared(self.prepare(batch), key=key)

    def __call__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        return self.predict(batch, key=key)


__all__ = [
    "PreparedOperatorInput",
    "TrainedOperator",
    "operator_contract_fingerprint",
    "operator_normalization_fingerprint",
]
