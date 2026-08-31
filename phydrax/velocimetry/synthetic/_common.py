#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PIVScenarioKind(StrEnum):
    """Two-dimensional particle-image motion family."""

    NO_MOTION = "no-motion"
    TRANSLATION = "translation"
    AFFINE = "affine"
    SHEAR = "shear"
    ROTATION = "rotation"
    SPATIAL_FREQUENCY = "spatial-frequency"


class PTVScenarioKind(StrEnum):
    """Three-dimensional particle-tracking stress family."""

    BASELINE = "baseline"
    CALIBRATION = "calibration"
    REFRACTION = "refraction"
    DEGENERATE_RAYS = "degenerate-rays"
    CROSSINGS = "crossings"
    OCCLUSION = "occlusion"
    BIRTHS_DEATHS = "births-deaths"
    DENSE = "dense"


class SyntheticEvidence(StrictModule, NonTrainableState):
    """Static identity and finite-capacity evidence for a generated scenario."""

    capacity: int = eqx.field(static=True)
    generated_count: int = eqx.field(static=True)
    finite: bool = eqx.field(static=True)
    status: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        capacity: int,
        generated_count: int,
        /,
        *,
        finite: bool,
        status: str,
        source_id: str,
    ):
        capacity_ = int(capacity)
        count = int(generated_count)
        status_ = str(status)
        source = str(source_id)
        if capacity_ <= 0:
            raise ValueError("capacity must be positive.")
        if count < 0 or count > capacity_:
            raise ValueError("generated_count must lie between zero and capacity.")
        if not status_ or not source:
            raise ValueError("status and source_id must be non-empty.")
        self.capacity = capacity_
        self.generated_count = count
        self.finite = bool(finite)
        self.status = status_
        self.source_id = source
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "synthetic-scenario-evidence",
                "source": source,
                "capacity": capacity_,
                "generated_count": count,
                "finite": bool(finite),
                "status": status_,
            }
        )


__all__ = [
    "PIVScenarioKind",
    "PTVScenarioKind",
    "SyntheticEvidence",
]
