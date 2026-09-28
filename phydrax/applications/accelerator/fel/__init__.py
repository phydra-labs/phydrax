#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Free-electron lasers: period-averaged (time-independent, time-dependent) and full-wave."""

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
from ._full_wave import (
    FELFullWaveBeam,
    FELFullWaveBeamSpecies,
    FELFullWaveEvidence,
    FELFullWaveFrameEvidence,
    FELFullWaveHuygens,
    FELFullWaveLedger,
    FELFullWavePlan,
    FELFullWaveResult,
    FELFullWaveSeed,
    FELFullWaveStatus,
    FELFullWaveTracks,
    PreparedFELFullWave,
)
from ._genesis4 import (
    genesis4_input,
    Genesis4GaussianSeed,
    Genesis4Result,
    run_genesis4,
)
from ._lattice import (
    FELUndulatorLattice,
    FELUndulatorSegment,
    undulator_coupling_factors,
)
from ._prebunching import FELModulator, FELPrebunching
from ._puffin import (
    puffin_input,
    PuffinGaussianSeed,
    PuffinProvider,
    PuffinResult,
    read_puffin_power,
    run_puffin,
)
from ._slices import FELBeamSlices, FELLoading, FELParticles, FELShotNoise
from ._space_charge import FELSpaceCharge, FELTransverseSpaceCharge
from ._theory import fel_scaling_estimate, FELScalingEstimate
from ._time_dependent import (
    FELPulseSeed,
    FELSlippageRoute,
    FELSpectrum,
    FELTimeDependentEvidence,
    FELTimeDependentLedger,
    FELTimeDependentPlan,
    FELTimeDependentResult,
    FELWindowBoundary,
)


__all__ = [
    "FELBeamSlices",
    "FELEnergyLedger",
    "FELEvidence",
    "FELFullWaveBeam",
    "FELFullWaveBeamSpecies",
    "FELFullWaveEvidence",
    "FELFullWaveFrameEvidence",
    "FELFullWaveHuygens",
    "FELFullWaveLedger",
    "FELFullWavePlan",
    "FELFullWaveResult",
    "FELFullWaveSeed",
    "FELFullWaveStatus",
    "FELFullWaveTracks",
    "FELGainEvidence",
    "FELLoading",
    "FELModulator",
    "FELParticles",
    "FELPlan",
    "FELPrebunching",
    "FELPulseSeed",
    "FELResult",
    "FELScalingEstimate",
    "FELSeed",
    "FELShotNoise",
    "FELSlippageRoute",
    "FELSpaceCharge",
    "FELSpectrum",
    "FELStatus",
    "FELTimeDependentEvidence",
    "FELTimeDependentLedger",
    "FELTimeDependentPlan",
    "FELTimeDependentResult",
    "FELTransverseModel",
    "FELTransverseSpaceCharge",
    "FELUndulatorLattice",
    "FELUndulatorSegment",
    "FELWakeLoss",
    "FELWindowBoundary",
    "Genesis4GaussianSeed",
    "Genesis4Result",
    "PreparedFELFullWave",
    "PuffinGaussianSeed",
    "PuffinProvider",
    "PuffinResult",
    "fel_scaling_estimate",
    "genesis4_input",
    "puffin_input",
    "read_puffin_power",
    "run_genesis4",
    "run_puffin",
    "undulator_coupling_factors",
]
