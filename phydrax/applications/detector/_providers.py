#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

from ...particle_physics import HEPProviderBinding
from ._core import DetectorConditions


@dataclass(frozen=True, slots=True)
class DetectorProviderBinding:
    capability: HEPProviderBinding
    conditions: DetectorConditions
    geometry_mapping_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.capability, HEPProviderBinding):
            raise TypeError("capability must be HEPProviderBinding.")
        if not isinstance(self.conditions, DetectorConditions):
            raise TypeError("conditions must be DetectorConditions.")
        if not self.geometry_mapping_id.strip():
            raise ValueError("geometry_mapping_id must be non-empty.")

    def require(self, capability: str, /) -> None:
        if not self.capability.supports(capability):
            raise ValueError("Detector provider lacks the required capability.")


__all__ = ["DetectorProviderBinding"]
