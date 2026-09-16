#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...measurement import ExposureKind, ExposureRecord
from ...particle_physics import HEPProviderBinding


class FixedTargetPlan(StrictModule, NonTrainableState):
    beam_pdg_id: int = eqx.field(static=True)
    beam_energy: float = eqx.field(static=True)
    beam_energy_unit_id: str = eqx.field(static=True)
    target_material_id: str = eqx.field(static=True)
    target_areal_density: float = eqx.field(static=True)
    target_areal_density_unit_id: str = eqx.field(static=True)
    exposure: ExposureRecord
    production_provider: HEPProviderBinding
    transport_provider: HEPProviderBinding
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        beam_pdg_id: int,
        beam_energy: float,
        beam_energy_unit_id: str,
        target_material_id: str,
        target_areal_density: float,
        target_areal_density_unit_id: str,
        exposure: ExposureRecord,
        production_provider: HEPProviderBinding,
        transport_provider: HEPProviderBinding,
    ):
        energy = float(beam_energy)
        density = float(target_areal_density)
        labels = tuple(
            str(value).strip()
            for value in (
                beam_energy_unit_id,
                target_material_id,
                target_areal_density_unit_id,
            )
        )
        if (
            not math.isfinite(energy)
            or energy <= 0.0
            or not math.isfinite(density)
            or density <= 0.0
            or any(not value for value in labels)
        ):
            raise ValueError("Fixed-target beam and material declarations are invalid.")
        if (
            not isinstance(exposure, ExposureRecord)
            or exposure.kind is not ExposureKind.PROTONS_ON_TARGET
        ):
            raise ValueError("Fixed-target plans require protons-on-target exposure.")
        if not isinstance(
            production_provider, HEPProviderBinding
        ) or not production_provider.supports("hep.fixed-target.production"):
            raise ValueError("Production provider lacks hep.fixed-target.production.")
        if not isinstance(
            transport_provider, HEPProviderBinding
        ) or not transport_provider.supports("hep.detector.transport"):
            raise ValueError("Transport provider lacks hep.detector.transport.")
        self.beam_pdg_id = int(beam_pdg_id)
        self.beam_energy = energy
        (
            self.beam_energy_unit_id,
            self.target_material_id,
            self.target_areal_density_unit_id,
        ) = labels
        self.target_areal_density = density
        self.exposure = exposure
        self.production_provider = production_provider
        self.transport_provider = transport_provider
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-target-plan",
                "beam": [self.beam_pdg_id, energy, labels[0]],
                "target": [labels[1], density, labels[2]],
                "exposure": exposure.exposure_id,
                "providers": [
                    production_provider.binding_id,
                    transport_provider.binding_id,
                ],
            }
        )


class FixedTargetStageRecord(StrictModule, NonTrainableState):
    stage: str = eqx.field(static=True)
    input_artifact_id: str = eqx.field(static=True)
    output_artifact_id: str = eqx.field(static=True)
    weighted_input_count: float = eqx.field(static=True)
    weighted_output_count: float = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        stage: str,
        input_artifact_id: str,
        output_artifact_id: str,
        /,
        *,
        weighted_input_count: float,
        weighted_output_count: float,
        evidence_ids: Sequence[str],
    ):
        labels = tuple(
            str(value).strip() for value in (stage, input_artifact_id, output_artifact_id)
        )
        evidence = tuple(sorted(str(value).strip() for value in evidence_ids))
        input_count = float(weighted_input_count)
        output_count = float(weighted_output_count)
        if (
            any(not value for value in labels)
            or not evidence
            or any(not value for value in evidence)
            or len(set(evidence)) != len(evidence)
        ):
            raise ValueError("Fixed-target stage identity and evidence are required.")
        if (
            not math.isfinite(input_count)
            or not math.isfinite(output_count)
            or input_count < 0.0
            or output_count < 0.0
            or output_count > input_count
        ):
            raise ValueError("Fixed-target weighted accounting is invalid.")
        self.stage, self.input_artifact_id, self.output_artifact_id = labels
        self.weighted_input_count = input_count
        self.weighted_output_count = output_count
        self.evidence_ids = evidence
        self.stage_id = canonical_fingerprint(
            {
                "kind": "fixed-target-stage-record",
                "labels": list(labels),
                "weighted_counts": [input_count, output_count],
                "evidence": list(evidence),
            }
        )


class FixedTargetChainRecord(StrictModule, NonTrainableState):
    plan: FixedTargetPlan
    stages: tuple[FixedTargetStageRecord, ...]
    closed: bool = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self, plan: FixedTargetPlan, stages: Sequence[FixedTargetStageRecord], /
    ):
        if not isinstance(plan, FixedTargetPlan):
            raise TypeError("plan must be FixedTargetPlan.")
        stages_ = tuple(stages)
        if not stages_ or any(
            not isinstance(value, FixedTargetStageRecord) for value in stages_
        ):
            raise TypeError("stages must contain typed non-empty records.")
        for previous, current in zip(stages_[:-1], stages_[1:], strict=True):
            if (
                previous.output_artifact_id != current.input_artifact_id
                or previous.weighted_output_count != current.weighted_input_count
            ):
                raise ValueError(
                    "Fixed-target chain lineage/accounting is discontinuous."
                )
        self.plan = plan
        self.stages = stages_
        self.closed = True
        self.record_id = canonical_fingerprint(
            {
                "kind": "fixed-target-chain-record",
                "plan": plan.plan_id,
                "stages": [value.stage_id for value in stages_],
            }
        )


__all__ = ["FixedTargetChainRecord", "FixedTargetPlan", "FixedTargetStageRecord"]
