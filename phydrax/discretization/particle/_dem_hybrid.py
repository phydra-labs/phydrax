#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from enum import StrEnum


class DEMHybridEventKind(StrEnum):
    CONTACT_ONSET = "contact_onset"
    CONTACT_SEPARATION = "contact_separation"
    STICK_TO_SLIP = "stick_to_slip"
    SLIP_TO_STICK = "slip_to_stick"
    COHESION_BIRTH = "cohesion_birth"
    COHESION_RUPTURE = "cohesion_rupture"
    ROLLING_YIELD = "rolling_yield"
    TORSIONAL_YIELD = "torsional_yield"
    USER = "user"


__all__ = ["DEMHybridEventKind"]
