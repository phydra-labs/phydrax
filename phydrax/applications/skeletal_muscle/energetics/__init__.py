#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-named skeletal-muscle energetic observation models."""

from ._retained_heat import (
    RETAINED_HEAT_CATEGORIES,
    RetainedHeatLedger,
    uchida_umberger_retained_heat,
)
from ._uchida_umberger_2010 import (
    integrate_metabolic_energy_joule,
    UCHIDA_UMBERGER_2010_SOURCE_REVISION,
    UCHIDA_UMBERGER_2010_VALIDATION_DOI,
    UchidaUmberger2010Evidence,
    UchidaUmberger2010Parameters,
    UchidaUmberger2010Plan,
    UchidaUmberger2010Result,
)


__all__ = [
    "RETAINED_HEAT_CATEGORIES",
    "RetainedHeatLedger",
    "uchida_umberger_retained_heat",
    "UCHIDA_UMBERGER_2010_SOURCE_REVISION",
    "UCHIDA_UMBERGER_2010_VALIDATION_DOI",
    "UchidaUmberger2010Evidence",
    "UchidaUmberger2010Parameters",
    "UchidaUmberger2010Plan",
    "UchidaUmberger2010Result",
    "integrate_metabolic_energy_joule",
]
