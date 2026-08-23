#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Structured conservative finite-volume discretizations."""

from ._amr import (
    ConservativeAMRSynchronizationPlan,
    ConservativeAMRSynchronizationResult,
    flux_register_from_accepted_steps,
)
from ._boundary import (
    AbstractFiniteVolumeBoundary,
    ConstantStateBoundary,
    ExtrapolationBoundary,
    FiniteVolumeBoundaryPair,
    FiniteVolumeBoundarySet,
    PrescribedNormalFluxBoundary,
    PrescribedStateBoundary,
    ReflectiveBoundary,
)
from ._diffusion import (
    AdvectionForm,
    AdvectionReconstruction,
    ConservativeAdvectionPlan,
    ConservativeBoundaryCondition,
    ConservativeBoundaryKind,
    ConservativeDiffusionPlan,
    FaceCoefficientPlan,
    FaceInterpolationKind,
    PreparedConservativeAdvection,
    PreparedConservativeDiffusion,
)
from ._distributed import (
    FiniteVolumeDecompositionPlan,
    FiniteVolumeHaloRoute,
    FiniteVolumeShardingReport,
    PreparedFiniteVolumeDecomposition,
)
from ._dynamics import (
    ConvexStateLimiterPlan,
    DifferentiabilityPolicy,
    FiniteVolumeMethodPlan,
    FiniteVolumeResidualDiagnostics,
    PreparedFiniteVolumeDynamics,
)
from ._halo import (
    FiniteVolumeGhostedAxis,
    FiniteVolumeHaloPlan,
    PreparedFiniteVolumeHaloPlan,
    reconstruction_ghost_width,
)
from ._high_resolution import (
    CharacteristicReconstructionPlan,
    CharacteristicSystem,
    HighResolutionMethod,
    HighResolutionReconstructionPlan,
    NonuniformWENOReconstructionPlan,
)
from ._incompressible import (
    FunctionalPressureCorrectionPlan,
    MACPressureProjectionPlan,
    PressureCorrectionResult,
    PressureProjectionResult,
)
from ._mapped import (
    evaluate_mapped_finite_volume_geometry,
    MappedFiniteVolumeDiscretization,
    MappedFiniteVolumePlan,
)
from ._multiblock import (
    ConservativeMultiblockFluxResult,
    ConservativeMultiblockInterfacePlan,
)
from ._physical_boundaries import (
    CharacteristicInflowBoundary,
    CharacteristicOutflowBoundary,
    FarFieldBoundary,
    NoSlipAdiabaticWallBoundary,
    NoSlipIsothermalWallBoundary,
    PrescribedHeatFluxWallBoundary,
    SlipWallBoundary,
    SupersonicInflowBoundary,
    SupersonicOutflowBoundary,
)
from ._positivity import (
    EinfeldtHLLFluxPlan,
    FiniteVolumeAdmissibilityReport,
    FluxPositivityPlan,
    PositivityBlendResult,
)
from ._reconstruction import (
    AbstractFaceReconstructionPlan,
    AbstractSlopeLimiter,
    MCLimiter,
    MinmodLimiter,
    MUSCLReconstruction,
    PiecewiseConstantReconstruction,
    SuperbeeLimiter,
    UnlimitedLimiter,
    VanLeerLimiter,
)
from ._riemann import (
    AbstractNumericalFluxPlan,
    EntropyConservativeEulerFluxPlan,
    EntropyStableEulerFluxPlan,
    HLLCFluxPlan,
    HLLFluxPlan,
    NumericalFluxResult,
    RoeFluxPlan,
    RusanovFluxPlan,
)
from ._structured import FiniteVolumeDiscretization, FiniteVolumePlan
from ._viscous import ViscousFluxPlan, ViscousStabilityReport
from ._wave import (
    AbstractWavePropagationPlan,
    FWaveShallowWaterPlan,
    RoeWavePropagationPlan,
    TransverseWaveSolverPlan,
    WaveDecomposition,
    WaveFamilyLimiterPlan,
    WaveLimiterKind,
)
from ._weno import WENOOrder, WENOReconstructionPlan


__all__ = [
    "AdvectionForm",
    "AdvectionReconstruction",
    "ConservativeAdvectionPlan",
    "ConservativeBoundaryCondition",
    "ConservativeBoundaryKind",
    "ConservativeDiffusionPlan",
    "ViscousFluxPlan",
    "ViscousStabilityReport",
    "FaceCoefficientPlan",
    "FaceInterpolationKind",
    "PreparedConservativeAdvection",
    "PreparedConservativeDiffusion",
    "AbstractFaceReconstructionPlan",
    "AbstractFiniteVolumeBoundary",
    "AbstractNumericalFluxPlan",
    "AbstractSlopeLimiter",
    "AbstractWavePropagationPlan",
    "ConservativeMultiblockFluxResult",
    "ConservativeMultiblockInterfacePlan",
    "CharacteristicReconstructionPlan",
    "CharacteristicSystem",
    "ConstantStateBoundary",
    "ConvexStateLimiterPlan",
    "DifferentiabilityPolicy",
    "EntropyConservativeEulerFluxPlan",
    "ConservativeAMRSynchronizationPlan",
    "ConservativeAMRSynchronizationResult",
    "flux_register_from_accepted_steps",
    "EntropyStableEulerFluxPlan",
    "ExtrapolationBoundary",
    "FWaveShallowWaterPlan",
    "FiniteVolumeBoundaryPair",
    "FiniteVolumeBoundarySet",
    "FiniteVolumeHaloPlan",
    "FiniteVolumeGhostedAxis",
    "FiniteVolumeDecompositionPlan",
    "FiniteVolumeHaloRoute",
    "FiniteVolumeShardingReport",
    "PreparedFiniteVolumeDecomposition",
    "PreparedFiniteVolumeHaloPlan",
    "reconstruction_ghost_width",
    "FiniteVolumeDiscretization",
    "FiniteVolumeMethodPlan",
    "MappedFiniteVolumeDiscretization",
    "MappedFiniteVolumePlan",
    "evaluate_mapped_finite_volume_geometry",
    "FiniteVolumePlan",
    "FunctionalPressureCorrectionPlan",
    "MACPressureProjectionPlan",
    "PressureCorrectionResult",
    "PressureProjectionResult",
    "FiniteVolumeResidualDiagnostics",
    "HLLCFluxPlan",
    "HLLFluxPlan",
    "EinfeldtHLLFluxPlan",
    "FiniteVolumeAdmissibilityReport",
    "FluxPositivityPlan",
    "PositivityBlendResult",
    "HighResolutionMethod",
    "HighResolutionReconstructionPlan",
    "MCLimiter",
    "MUSCLReconstruction",
    "MinmodLimiter",
    "NonuniformWENOReconstructionPlan",
    "NumericalFluxResult",
    "PiecewiseConstantReconstruction",
    "PreparedFiniteVolumeDynamics",
    "PrescribedNormalFluxBoundary",
    "PrescribedStateBoundary",
    "ReflectiveBoundary",
    "CharacteristicInflowBoundary",
    "CharacteristicOutflowBoundary",
    "FarFieldBoundary",
    "NoSlipAdiabaticWallBoundary",
    "NoSlipIsothermalWallBoundary",
    "PrescribedHeatFluxWallBoundary",
    "SlipWallBoundary",
    "SupersonicInflowBoundary",
    "SupersonicOutflowBoundary",
    "RoeFluxPlan",
    "RoeWavePropagationPlan",
    "RusanovFluxPlan",
    "SuperbeeLimiter",
    "TransverseWaveSolverPlan",
    "UnlimitedLimiter",
    "VanLeerLimiter",
    "WENOOrder",
    "WENOReconstructionPlan",
    "WaveDecomposition",
    "WaveFamilyLimiterPlan",
    "WaveLimiterKind",
]
