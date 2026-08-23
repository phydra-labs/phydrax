#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...._doc import DOC_KEY0
from ...._frozendict import frozendict
from ...._strict import StrictModule
from ..._keys import EvalKey
from ..capabilities import ConfiguredOperatorContract, OperatorTrainingEvidence
from ..data import OperatorBatch, OperatorPrediction
from ..engine import AbstractOperatorModel
from ..sharding import OperatorShardingPolicy
from ..task import _freeze_json, OperatorTask
from ._dtype import OperatorDTypePolicy, OperatorPrecisionEvidence
from ._execution import (
    operator_contract_fingerprint,
    operator_normalization_fingerprint,
    OperatorCompilationStrategy,
    OperatorExecutionPlan,
    OperatorPaddingPolicy,
    PreparedOperatorInput,
)
from ._normalization import OperatorNormalizationPolicy
from ._physics import OperatorOutputPipeline


class TrainedOperator(StrictModule):
    """Artifact identity around one fully prepared operator execution plan."""

    execution_plan: OperatorExecutionPlan
    artifact_id: str
    provenance: frozendict[str, Any]
    calibration: frozendict[str, Any]

    def __init__(
        self,
        execution_model: AbstractOperatorModel,
        task: OperatorTask,
        /,
        *,
        training_evidence: OperatorTrainingEvidence,
        output_field_map: Mapping[str, str] | None = None,
        fixed_query_fingerprints: Mapping[str, str] | None = None,
        output_pipeline: OperatorOutputPipeline | None = None,
        normalization: OperatorNormalizationPolicy | None = None,
        dtype_policy: OperatorDTypePolicy | None = None,
        sharding_policy: OperatorShardingPolicy | None = None,
        compilation_strategy: OperatorCompilationStrategy = "eager",
        padding_policy: OperatorPaddingPolicy = "explicit_mask",
        artifact_id: str = "",
        provenance: dict[str, Any] | None = None,
        calibration: dict[str, Any] | None = None,
    ):
        self.execution_plan = OperatorExecutionPlan(
            execution_model,
            task,
            training_evidence=training_evidence,
            output_field_map=output_field_map,
            fixed_query_fingerprints=fixed_query_fingerprints,
            output_pipeline=output_pipeline,
            normalization=normalization,
            dtype_policy=dtype_policy,
            sharding_policy=sharding_policy,
            compilation_strategy=compilation_strategy,
            padding_policy=padding_policy,
        )
        self.artifact_id = str(artifact_id)
        self.provenance = _freeze_json({} if provenance is None else provenance)
        self.calibration = _freeze_json({} if calibration is None else calibration)

    @property
    def execution_model(self) -> AbstractOperatorModel:
        return self.execution_plan.execution_model

    @property
    def task(self) -> OperatorTask:
        return self.execution_plan.task

    @property
    def contract(self) -> ConfiguredOperatorContract:
        return self.execution_plan.contract

    @property
    def output_field_map(self) -> frozendict[str, str]:
        return self.execution_plan.output_field_map

    @property
    def fixed_query_fingerprints(self) -> frozendict[str, str]:
        return self.execution_plan.fixed_query_fingerprints

    @property
    def output_pipeline(self) -> OperatorOutputPipeline | None:
        return self.execution_plan.output_pipeline

    @property
    def normalization(self) -> OperatorNormalizationPolicy | None:
        return self.execution_plan.normalization

    @property
    def dtype_policy(self) -> OperatorDTypePolicy:
        return self.execution_plan.dtype_policy

    @property
    def precision_evidence(self) -> OperatorPrecisionEvidence:
        return self.execution_plan.precision_evidence

    @property
    def training_evidence(self) -> OperatorTrainingEvidence:
        return self.execution_plan.training_evidence

    @property
    def sharding_policy(self) -> OperatorShardingPolicy | None:
        return self.execution_plan.sharding_policy

    @property
    def compilation_strategy(self) -> OperatorCompilationStrategy:
        return self.execution_plan.compilation_strategy

    @property
    def padding_policy(self) -> OperatorPaddingPolicy:
        return self.execution_plan.padding_policy

    @property
    def task_fingerprint(self) -> str:
        return self.execution_plan.task_fingerprint

    @property
    def contract_fingerprint(self) -> str:
        return self.execution_plan.contract_fingerprint

    @property
    def normalization_fingerprint(self) -> str:
        return self.execution_plan.normalization_fingerprint

    def prepare_prevalidated(self, batch: OperatorBatch, /) -> PreparedOperatorInput:
        return self.execution_plan.prepare_prevalidated(batch)

    def prepare(self, batch: OperatorBatch, /) -> PreparedOperatorInput:
        return self.execution_plan.prepare(batch)

    def predict_prepared(
        self,
        prepared: PreparedOperatorInput,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        return self.execution_plan.predict_prepared(prepared, key=key)

    def predict(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        return self.execution_plan.predict(batch, key=key)

    def __call__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        return self.predict(batch, key=key)


__all__ = [
    "OperatorCompilationStrategy",
    "OperatorExecutionPlan",
    "OperatorPaddingPolicy",
    "PreparedOperatorInput",
    "TrainedOperator",
    "operator_contract_fingerprint",
    "operator_normalization_fingerprint",
]
