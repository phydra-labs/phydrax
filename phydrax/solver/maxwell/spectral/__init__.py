#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pseudo-spectral analytical time-domain (PSATD) Maxwell for PIC.

Cartesian (`SpectralMaxwellPlan`) and quasi-cylindrical azimuthal-mode
(`QuasiCylindricalMaxwellPlan`) solvers share one exact interval propagator.
"""

from ._antenna import PreparedSpectralPlaneAntenna
from ._fbpic_provider import (
    fbpic_laser_wakefield,
    FBPICProvider,
    FBPICWakefieldResult,
)
from ._galilean import SpectralOperators
from ._nci import (
    godfrey_vay_growth_rate,
    GodfreyVayReference,
    NCIGrowthFit,
    SpectralNCIMonitorPlan,
    SpectralNCISample,
)
from ._pml import SpectralPMLPlan
from ._psatd import (
    modified_wavenumber,
    PreparedSpectralHuygensBox,
    PreparedSpectralMaxwell,
    SpectralAbsorber,
    SpectralChargeConservation,
    SpectralDecomposition,
    SpectralGrid,
    SpectralHuygensBoxPlan,
    SpectralLocalUpdate,
    SpectralMaxwellDiagnostics,
    SpectralMaxwellPlan,
    SpectralMaxwellSource,
    SpectralMaxwellState,
    SpectralMaxwellVariant,
    SpectralStencil,
    SpectralTimeDependency,
    stencil_coefficients,
)
from ._quasi_cylindrical import (
    PreparedQuasiCylindricalMaxwell,
    QuasiCylindricalAbsorber,
    QuasiCylindricalAntennaPlan,
    QuasiCylindricalHuygensPlan,
    QuasiCylindricalMaxwellDiagnostics,
    QuasiCylindricalMaxwellPlan,
    QuasiCylindricalMaxwellState,
    QuasiCylindricalSource,
    RadialDampingPlan,
)


__all__ = [
    "FBPICProvider",
    "FBPICWakefieldResult",
    "GodfreyVayReference",
    "NCIGrowthFit",
    "PreparedQuasiCylindricalMaxwell",
    "PreparedSpectralHuygensBox",
    "PreparedSpectralMaxwell",
    "PreparedSpectralPlaneAntenna",
    "QuasiCylindricalAbsorber",
    "QuasiCylindricalAntennaPlan",
    "QuasiCylindricalHuygensPlan",
    "QuasiCylindricalMaxwellDiagnostics",
    "QuasiCylindricalMaxwellPlan",
    "QuasiCylindricalMaxwellState",
    "QuasiCylindricalSource",
    "RadialDampingPlan",
    "SpectralAbsorber",
    "SpectralChargeConservation",
    "SpectralDecomposition",
    "SpectralGrid",
    "SpectralHuygensBoxPlan",
    "SpectralLocalUpdate",
    "SpectralMaxwellDiagnostics",
    "SpectralMaxwellPlan",
    "SpectralMaxwellSource",
    "SpectralMaxwellState",
    "SpectralMaxwellVariant",
    "SpectralNCIMonitorPlan",
    "SpectralNCISample",
    "SpectralOperators",
    "SpectralPMLPlan",
    "SpectralStencil",
    "SpectralTimeDependency",
    "fbpic_laser_wakefield",
    "godfrey_vay_growth_rate",
    "modified_wavenumber",
    "stencil_coefficients",
]
