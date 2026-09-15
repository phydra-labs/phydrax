#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...particle_physics import (
    HEPRunContext,
    ProcessNormalization,
    SystematicConfiguration,
)


class ColliderStageEvidence(StrictModule, NonTrainableState):
    stage: str = eqx.field(static=True)
    input_artifact_id: str = eqx.field(static=True)
    output_artifact_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    input_event_count: int = eqx.field(static=True)
    output_event_count: int = eqx.field(static=True)
    rejected_event_count: int = eqx.field(static=True)
    overflow_event_count: int = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        stage: str,
        input_artifact_id: str,
        output_artifact_id: str,
        provider_id: str,
        /,
        *,
        input_event_count: int,
        output_event_count: int,
        rejected_event_count: int,
        overflow_event_count: int,
        evidence_ids: Sequence[str],
        accepted: bool,
    ):
        values = tuple(
            str(value).strip()
            for value in (stage, input_artifact_id, output_artifact_id, provider_id)
        )
        evidence = tuple(sorted(str(value).strip() for value in evidence_ids))
        counts = tuple(
            map(
                int,
                (
                    input_event_count,
                    output_event_count,
                    rejected_event_count,
                    overflow_event_count,
                ),
            )
        )
        if (
            any(not value for value in values)
            or not evidence
            or any(not value for value in evidence)
            or len(set(evidence)) != len(evidence)
        ):
            raise ValueError("Collider stage identity and evidence are required.")
        if (
            any(value < 0 for value in counts)
            or counts[1] + counts[2] != counts[0]
            or counts[3] > counts[2]
        ):
            raise ValueError("Collider stage event accounting is inconsistent.")
        self.stage, self.input_artifact_id, self.output_artifact_id, self.provider_id = (
            values
        )
        (
            self.input_event_count,
            self.output_event_count,
            self.rejected_event_count,
            self.overflow_event_count,
        ) = counts
        self.evidence_ids = evidence
        self.accepted = bool(accepted)
        self.stage_id = canonical_fingerprint(
            {
                "kind": "collider-stage-evidence",
                "values": list(values),
                "counts": list(counts),
                "evidence": list(evidence),
                "accepted": bool(accepted),
            }
        )


class ColliderProductionRecord(StrictModule, NonTrainableState):
    run_context: HEPRunContext
    normalization: ProcessNormalization
    systematics: SystematicConfiguration
    stages: tuple[ColliderStageEvidence, ...]
    final_artifact_id: str = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        run_context: HEPRunContext,
        normalization: ProcessNormalization,
        systematics: SystematicConfiguration,
        stages: Sequence[ColliderStageEvidence],
        /,
    ):
        if (
            not isinstance(run_context, HEPRunContext)
            or not isinstance(normalization, ProcessNormalization)
            or not isinstance(systematics, SystematicConfiguration)
        ):
            raise TypeError(
                "Collider production requires typed run, normalization, and systematics records."
            )
        stages_ = tuple(stages)
        if not stages_ or any(
            not isinstance(value, ColliderStageEvidence) for value in stages_
        ):
            raise TypeError("stages must contain typed non-empty evidence.")
        stage_names = tuple(value.stage for value in stages_)
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("Collider production stage names must be unique.")
        for previous, current in zip(stages_[:-1], stages_[1:], strict=True):
            if previous.output_artifact_id != current.input_artifact_id:
                raise ValueError("Collider production artifact lineage is discontinuous.")
            if previous.output_event_count != current.input_event_count:
                raise ValueError("Collider production event accounting is discontinuous.")
        closed = (
            run_context.scientifically_admitted
            and all(value.accepted for value in stages_)
            and all(value.overflow_event_count == 0 for value in stages_)
        )
        self.run_context = run_context
        self.normalization = normalization
        self.systematics = systematics
        self.stages = stages_
        self.final_artifact_id = stages_[-1].output_artifact_id
        self.closed = closed
        self.record_id = canonical_fingerprint(
            {
                "kind": "collider-production-record",
                "run_context": run_context.context_id,
                "normalization": normalization.normalization_id,
                "systematics": systematics.configuration_id,
                "stages": [value.stage_id for value in stages_],
                "closed": closed,
            }
        )


__all__ = ["ColliderProductionRecord", "ColliderStageEvidence"]
