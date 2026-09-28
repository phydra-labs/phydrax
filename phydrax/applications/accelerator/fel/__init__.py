#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-independent period-averaged free-electron lasers."""

from ._averaged import (
    FELEnergyLedger,
    FELEvidence,
    FELGainEvidence,
    FELPlan,
    FELResult,
    FELSeed,
    FELStatus,
    FELTransverseModel,
    FELWakeLoss,
)
from ._lattice import (
    FELUndulatorLattice,
    FELUndulatorSegment,
    undulator_coupling_factors,
)
from ._slices import FELBeamSlices, FELLoading, FELParticles, FELShotNoise
from ._theory import fel_scaling_estimate, FELScalingEstimate


__all__ = [
    "FELBeamSlices",
    "FELEnergyLedger",
    "FELEvidence",
    "FELGainEvidence",
    "FELLoading",
    "FELParticles",
    "FELPlan",
    "FELResult",
    "FELScalingEstimate",
    "FELSeed",
    "FELShotNoise",
    "FELStatus",
    "FELTransverseModel",
    "FELUndulatorLattice",
    "FELUndulatorSegment",
    "FELWakeLoss",
    "fel_scaling_estimate",
    "undulator_coupling_factors",
]
