#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.random as jr
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver._photon_transport import PhotonTransportPlan, PhotonTransportResult
from ._detector import PlanarXRayDetectorPlan, PlanarXRayDetectorResult
from ._sources import DiagnosticXRaySourcePlan, PhotonSourceBatch


class DiagnosticXRayExperimentResult(StrictModule, NonTrainableState):
    source: PhotonSourceBatch
    transport: PhotonTransportResult
    detector: PlanarXRayDetectorResult
    successful: Array
    plan_id: str = eqx.field(static=True)


class DiagnosticXRayExperimentPlan(StrictModule, NonTrainableState):
    source: DiagnosticXRaySourcePlan
    transport: PhotonTransportPlan
    detector: PlanarXRayDetectorPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: DiagnosticXRaySourcePlan,
        transport: PhotonTransportPlan,
        detector: PlanarXRayDetectorPlan,
        /,
    ):
        if not isinstance(source, DiagnosticXRaySourcePlan):
            raise TypeError("source must be DiagnosticXRaySourcePlan.")
        if not isinstance(transport, PhotonTransportPlan):
            raise TypeError("transport must be PhotonTransportPlan.")
        if not isinstance(detector, PlanarXRayDetectorPlan):
            raise TypeError("detector must be PlanarXRayDetectorPlan.")
        if (
            source.spectrum.energy_unit.unit_id
            != transport.cross_sections.energy_unit.unit_id
        ):
            raise ValueError("Source and transport energy units must match exactly.")
        if not bool(transport.geometry.locate(source.position).inside):
            raise ValueError("Diagnostic source must lie inside the transport universe.")
        self.source = source
        self.transport = transport
        self.detector = detector
        self.plan_id = canonical_fingerprint(
            {
                "kind": "diagnostic-xray-experiment",
                "source": source.source_id,
                "transport": transport.plan_id,
                "detector": detector.detector_id,
                "dose_semantics": "kerma-not-absorbed-dose",
            }
        )

    def simulate(
        self,
        key: Array,
        history_count: int,
        /,
        *,
        first_history_id: int = 0,
    ) -> DiagnosticXRayExperimentResult:
        source = self.source.sample(
            jr.fold_in(key, 0),
            history_count,
            first_history_id=first_history_id,
        )
        transport = self.transport.simulate(
            source.origins,
            source.directions,
            source.energies,
            jr.fold_in(key, 1),
            history_ids=source.history_ids,
            weights=source.weights,
        )
        detector = self.detector.score(transport)
        successful = source.successful & transport.all_successful & detector.successful
        return DiagnosticXRayExperimentResult(
            source, transport, detector, successful, self.plan_id
        )


__all__ = ["DiagnosticXRayExperimentPlan", "DiagnosticXRayExperimentResult"]
