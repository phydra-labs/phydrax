#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Layered, frequency-domain, time-domain, MT, and GPR electromagnetics."""

from ._frequency_domain import (
    ConductiveEMMaterial,
    FrequencyDomainEMPlan,
    FrequencyDomainEMResult,
    FrequencyDomainEMSurvey,
)
from ._gpr import DispersiveFullWaveGPRPlan, GaussianDerivativeWaveform, GPRResult
from ._layered import (
    DigitalHankelTransformPlan,
    LayeredEarthModel,
    VACUUM_PERMEABILITY_H_M,
    VACUUM_PERMITTIVITY_F_M,
)
from ._magnetotelluric import (
    GalvanicDistortion,
    MagnetotelluricResponse,
    MagnetotelluricResponsePlan,
    RemoteReferenceMTPlan,
)
from ._time_domain import ImplicitTimeDomainEMPlan, TimeDomainEMResult, TimeDomainEMState


__all__ = [
    "ConductiveEMMaterial",
    "DigitalHankelTransformPlan",
    "DispersiveFullWaveGPRPlan",
    "FrequencyDomainEMPlan",
    "FrequencyDomainEMResult",
    "FrequencyDomainEMSurvey",
    "GPRResult",
    "GalvanicDistortion",
    "GaussianDerivativeWaveform",
    "ImplicitTimeDomainEMPlan",
    "LayeredEarthModel",
    "MagnetotelluricResponse",
    "MagnetotelluricResponsePlan",
    "RemoteReferenceMTPlan",
    "TimeDomainEMResult",
    "TimeDomainEMState",
    "VACUUM_PERMEABILITY_H_M",
    "VACUUM_PERMITTIVITY_F_M",
]
