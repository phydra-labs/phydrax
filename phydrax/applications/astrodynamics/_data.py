#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


AstrodynamicsDataDifferentiability: TypeAlias = Literal[
    "native-parameter", "coordinate-only", "constant"
]


class AstrodynamicsDataProvenance(StrictModule, NonTrainableState):
    producer: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    differentiability: AstrodynamicsDataDifferentiability = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        producer: str,
        producer_version: str,
        source_id: str,
        checksum: str,
        license_id: str,
        frame_id: str,
        epoch_id: str,
        scale_id: str,
        differentiability: AstrodynamicsDataDifferentiability,
    ):
        values = tuple(
            str(value).strip()
            for value in (
                producer,
                producer_version,
                source_id,
                checksum,
                license_id,
                frame_id,
                epoch_id,
                scale_id,
            )
        )
        if any(not value for value in values):
            raise ValueError("Astrodynamics data provenance fields must be non-empty.")
        if differentiability not in ("native-parameter", "coordinate-only", "constant"):
            raise ValueError("Unknown astrodynamics data differentiability contract.")
        (
            self.producer,
            self.producer_version,
            self.source_id,
            self.checksum,
            self.license_id,
            self.frame_id,
            self.epoch_id,
            self.scale_id,
        ) = values
        self.differentiability = differentiability
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-data-provenance",
                "values": list(values),
                "differentiability": differentiability,
            }
        )


__all__ = ["AstrodynamicsDataDifferentiability", "AstrodynamicsDataProvenance"]
