#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from enum import StrEnum


class HybridSensitivityMode(StrEnum):
    SHARP_BRANCHWISE = "sharp_branchwise"
    SMOOTH_SURROGATE = "smooth_surrogate"
    HYBRID_EVENT_AWARE = "hybrid_event_aware"


__all__ = ["HybridSensitivityMode"]
