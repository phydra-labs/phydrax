#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

from ...particle_physics import HEPCapabilityContract
from ._core import DetectorConditions


@dataclass(frozen=True, slots=True)
class DetectorProviderBinding:
    capability: HEPCapabilityContract
    conditions: DetectorConditions
    geometry_mapping_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.capability, HEPCapabilityContract):
            raise TypeError("capability must be HEPCapabilityContract.")
        if not isinstance(self.conditions, DetectorConditions):
            raise TypeError("conditions must be DetectorConditions.")
        if not self.geometry_mapping_id.strip():
            raise ValueError("geometry_mapping_id must be non-empty.")

    def require(self, capability: str, /) -> None:
        if not self.capability.supports(capability):
            raise ValueError("Detector provider lacks the required capability.")


__all__ = ["DetectorProviderBinding"]
