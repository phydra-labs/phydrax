#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Free-space and spherical-harmonic potential-field models."""

from ._corrections import (
    FourierContinuationPlan,
    RegionalTrendPlan,
    TerrainCorrectionPlan,
)
from ._gravity import (
    FreeSpaceGravityPlan,
    GRAVITATIONAL_CONSTANT_M3_KG_S2,
    GravityQuadratureSource,
    GravityResult,
)
from ._magnetics import (
    FreeSpaceMagneticPlan,
    MagneticMaterial,
    MagneticResult,
    VACUUM_PERMEABILITY_H_M,
)
from ._spherical_harmonics import (
    SphericalGravityResult,
    SphericalHarmonicGravityPlan,
    SphericalHarmonicMagneticPlan,
    SphericalMagneticResult,
)


__all__ = [
    "FourierContinuationPlan",
    "FreeSpaceGravityPlan",
    "FreeSpaceMagneticPlan",
    "GRAVITATIONAL_CONSTANT_M3_KG_S2",
    "GravityQuadratureSource",
    "GravityResult",
    "MagneticMaterial",
    "MagneticResult",
    "RegionalTrendPlan",
    "SphericalGravityResult",
    "SphericalHarmonicGravityPlan",
    "SphericalHarmonicMagneticPlan",
    "SphericalMagneticResult",
    "TerrainCorrectionPlan",
    "VACUUM_PERMEABILITY_H_M",
]
