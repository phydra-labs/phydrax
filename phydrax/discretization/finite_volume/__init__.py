#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Structured conservative finite-volume discretizations."""

from ._amr import (
    ConservativeAMRSynchronizationPlan,
    ConservativeAMRSynchronizationResult,
    flux_register_from_accepted_steps,
)
from ._automatic_remap import (
    build_unstructured_conservative_remap,
    UnstructuredConservativeRemapBuildResult,
    UnstructuredConservativeRemapEvidence,
    UnstructuredConservativeRemapStatus,
)
from ._boundary import (
    AbstractFiniteVolumeBoundary,
    ALEBoundaryContext,
    ConstantStateBoundary,
    ExtrapolationBoundary,
    FiniteVolumeBoundaryPair,
    FiniteVolumeBoundarySet,
    PrescribedNormalFluxBoundary,
    PrescribedStateBoundary,
    ReflectiveBoundary,
)
from ._capillarity import (
    BalancedCapillaryOperator,
    CapillaryFaceRateBlock,
    CurvatureEvidence,
    CurvatureGeometryError,
    CurvatureStatus,
    CurvatureUncertaintyError,
    SurfaceTensionPolicy,
)
from ._cell_polynomial import (
    CellPolynomialBasis,
    CellPolynomialReconstructionPlan,
    CellPolynomialReconstructionReport,
    PreparedCellPolynomialReconstruction,
)
from ._closure import ConservativeFaceClosurePlan
from ._contact_angle import (
    ContactAngleCondition,
    ContactAngleEvidence,
    ContactAngleReconstructionResult,
    ContactAngleStatus,
    EmbeddedBoundaryContactAngleSet,
    reconstruct_wall_interface_normal,
)
from ._coupling import (
    PreparedUnstructuredFiniteVolumeCoupling,
    UnstructuredFiniteVolumeCouplingPlan,
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
from ._embedded_dynamics import (
    lower_embedded_stage_metrics,
    UnstructuredEmbeddedBoundarySet,
)
from ._entropy import (
    FiniteVolumeEntropyDiagnostics,
    integrated_finite_volume_relative_entropy,
)
from ._flux_ledger import (
    FiniteVolumeAcceptedFluxIntegralBlock,
    FiniteVolumeAcceptedFluxIntegralLedger,
    FiniteVolumeStageFluxRateBlock,
    FiniteVolumeStageFluxRateLedger,
)
from ._geometry_protocol import (
    ALEGeometryConsistencyPolicy,
    ExplicitFaceBlockGeometry,
    FiniteVolumeFaceBlock,
    FiniteVolumeGeometryStatus,
    FiniteVolumeStageFaceBlock,
    FiniteVolumeStageFaceLayout,
    FiniteVolumeStageGeometryEvidence,
    FiniteVolumeStageMetrics,
    lower_static_unstructured_stage_metrics,
    PreparedFiniteVolumeGeometry,
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
from ._high_resolution_extended import (
    ExplicitStabilizationPlan,
    FilterADPolicy,
    TENOQualification,
)
from ._incompressible import (
    FaceVelocity,
    MACOperatorPlan,
    MACOperatorReport,
    PreparedMACOperators,
)
from ._mapped import (
    evaluate_mapped_finite_volume_geometry,
    MappedFiniteVolumeDiscretization,
    MappedFiniteVolumePlan,
)
from ._mhd_ct import MHDCTRateResult, UpwindConstrainedTransportPlan
from ._multiblock import (
    ConservativeMultiblockFluxResult,
    ConservativeMultiblockInterfacePlan,
)
from ._physical_boundaries import (
    CharacteristicInflowBoundary,
    CharacteristicOutflowBoundary,
    FarFieldBoundary,
    MovingSlipWallBoundary,
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
from ._precision import FiniteVolumePrecisionPolicy, PrecisionDType
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
    AbstractArbitraryNormalNumericalFluxPlan,
    AbstractNumericalFluxPlan,
    AbstractSymmetricTwoPointFluxPlan,
    EntropyConservativeEulerFluxPlan,
    EntropyStableEulerFluxPlan,
    HLLCFluxPlan,
    HLLDFluxPlan,
    HLLFluxPlan,
    NumericalFluxResult,
    RoeFluxPlan,
    RusanovFluxPlan,
)
from ._small_cell import (
    ConservativeSmallCellRedistributionEvidence,
    ConservativeSmallCellRedistributionPlan,
    ConservativeSmallCellRedistributionReport,
    ConservativeSmallCellRedistributionResult,
)
from ._structured import FiniteVolumeDiscretization, FiniteVolumePlan
from ._triangle_archive import (
    read_triangle_fv_archive,
    write_triangle_fv_archive,
)
from ._triangle_dynamics import (
    PreparedTriangleFiniteVolumeDynamics,
    TriangleFiniteVolumeBoundarySet,
    TriangleFiniteVolumeDiagnostics,
    TriangleFiniteVolumeMethodPlan,
)
from ._triangle_fv import (
    evaluate_triangle_fv_geometry,
    TriangleFiniteVolumeDiscretization,
    TriangleFiniteVolumePlan,
    TriangleFiniteVolumeQualityReport,
)
from ._triangle_polynomial import (
    evaluate_triangle_second_moments,
    PreparedTriangleQuadratic,
    TriangleKExactReconstructionPlan,
    TriangleQuadraticReport,
)
from ._triangle_reconstruction import (
    PreparedTriangleWLSQ,
    TriangleLimiterKind,
    TriangleMUSCLReconstructionPlan,
    TriangleWLSQReport,
)
from ._triangle_viscous import (
    TriangleViscousFluxPlan,
    TriangleViscousStabilityReport,
)
from ._unstructured import (
    evaluate_unstructured_fv_geometry,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
    UnstructuredFiniteVolumeQualityReport,
)
from ._unstructured_amr import (
    UnstructuredAMRFluxRegister,
    UnstructuredAMRHierarchyPlan,
    UnstructuredAMRSelection,
)
from ._unstructured_archive import (
    read_unstructured_fv_archive,
    write_unstructured_fv_archive,
)
from ._unstructured_dynamics import (
    PreparedUnstructuredFiniteVolumeDynamics,
    UnstructuredFiniteVolumeBoundarySet,
    UnstructuredFiniteVolumeDiagnostics,
    UnstructuredFiniteVolumeMethodPlan,
)
from ._unstructured_embedded_boundary import (
    EmbeddedBoundaryEvidence,
    EmbeddedBoundaryMetrics,
    EmbeddedBoundaryPlan,
    EmbeddedBoundaryReport,
    EmbeddedBoundaryStabilizationPolicy,
    EmbeddedBoundaryStatus,
)
from ._unstructured_incompressible import (
    PreparedUnstructuredCollocatedOperators,
    UnstructuredCollocatedOperatorReport,
)
from ._unstructured_motion import (
    FixedConnectivityMotionPlan,
    UnstructuredALEStepGeometry,
    UnstructuredFiniteVolumeGeometryState,
    UnstructuredMotionMetrics,
    UnstructuredMotionReport,
)
from ._unstructured_overset import (
    PeriodicSlidingCoupling,
    PeriodicSlidingInterfacePlan,
    PeriodicSlidingRefreshArtifact,
    UnstructuredOversetPlan,
    UnstructuredOversetReport,
)
from ._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
    UnstructuredRemapReport,
)
from ._unstructured_vof import (
    JAXPLICStageReconstruction,
    PLICFaceApertures,
    PLICInterfaceStatus,
    PLICReconstruction,
    UnstructuredVOFPlan,
)
from ._unstructured_weno import (
    PreparedUnstructuredWENOZReconstruction,
    UnstructuredWENOLimiter,
    UnstructuredWENOZReconstructionPlan,
)
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
    "ALEBoundaryContext",
    "MovingSlipWallBoundary",
    "ALEGeometryConsistencyPolicy",
    "FiniteVolumeFaceBlock",
    "FiniteVolumeGeometryStatus",
    "FiniteVolumeStageFaceBlock",
    "FiniteVolumeStageFaceLayout",
    "FiniteVolumeStageGeometryEvidence",
    "FiniteVolumeStageMetrics",
    "ExplicitFaceBlockGeometry",
    "PreparedFiniteVolumeGeometry",
    "lower_static_unstructured_stage_metrics",
    "UnstructuredConservativeRemapBuildResult",
    "UnstructuredConservativeRemapEvidence",
    "UnstructuredConservativeRemapStatus",
    "build_unstructured_conservative_remap",
    "BalancedCapillaryOperator",
    "CapillaryFaceRateBlock",
    "CurvatureEvidence",
    "CurvatureGeometryError",
    "CurvatureStatus",
    "CurvatureUncertaintyError",
    "SurfaceTensionPolicy",
    "ContactAngleCondition",
    "ContactAngleEvidence",
    "ContactAngleReconstructionResult",
    "ContactAngleStatus",
    "EmbeddedBoundaryContactAngleSet",
    "reconstruct_wall_interface_normal",
    "FiniteVolumeAcceptedFluxIntegralBlock",
    "FiniteVolumeAcceptedFluxIntegralLedger",
    "FiniteVolumeStageFluxRateBlock",
    "FiniteVolumeStageFluxRateLedger",
    "PreparedUnstructuredFiniteVolumeCoupling",
    "UnstructuredFiniteVolumeCouplingPlan",
    "UnstructuredFiniteVolumeDiscretization",
    "UnstructuredFiniteVolumePlan",
    "UnstructuredFiniteVolumeQualityReport",
    "evaluate_unstructured_fv_geometry",
    "ConservativeSmallCellRedistributionEvidence",
    "ConservativeSmallCellRedistributionPlan",
    "ConservativeSmallCellRedistributionReport",
    "ConservativeSmallCellRedistributionResult",
    "EmbeddedBoundaryMetrics",
    "EmbeddedBoundaryPlan",
    "EmbeddedBoundaryReport",
    "UnstructuredEmbeddedBoundarySet",
    "lower_embedded_stage_metrics",
    "EmbeddedBoundaryEvidence",
    "EmbeddedBoundaryStabilizationPolicy",
    "EmbeddedBoundaryStatus",
    "JAXPLICStageReconstruction",
    "PLICFaceApertures",
    "PLICInterfaceStatus",
    "PLICReconstruction",
    "UnstructuredVOFPlan",
    "UnstructuredAMRFluxRegister",
    "UnstructuredAMRHierarchyPlan",
    "UnstructuredAMRSelection",
    "read_unstructured_fv_archive",
    "write_unstructured_fv_archive",
    "PreparedUnstructuredFiniteVolumeDynamics",
    "UnstructuredFiniteVolumeBoundarySet",
    "UnstructuredFiniteVolumeDiagnostics",
    "UnstructuredFiniteVolumeMethodPlan",
    "PreparedUnstructuredCollocatedOperators",
    "UnstructuredCollocatedOperatorReport",
    "CellPolynomialBasis",
    "CellPolynomialReconstructionPlan",
    "FixedConnectivityMotionPlan",
    "UnstructuredALEStepGeometry",
    "UnstructuredFiniteVolumeGeometryState",
    "UnstructuredMotionMetrics",
    "UnstructuredMotionReport",
    "UnstructuredConservativeRemapPlan",
    "UnstructuredRemapReport",
    "PeriodicSlidingCoupling",
    "PeriodicSlidingRefreshArtifact",
    "PeriodicSlidingInterfacePlan",
    "UnstructuredOversetPlan",
    "UnstructuredOversetReport",
    "CellPolynomialReconstructionReport",
    "PreparedCellPolynomialReconstruction",
    "PreparedUnstructuredWENOZReconstruction",
    "UnstructuredWENOLimiter",
    "UnstructuredWENOZReconstructionPlan",
    "PreparedTriangleFiniteVolumeDynamics",
    "TriangleFiniteVolumeBoundarySet",
    "TriangleFiniteVolumeDiagnostics",
    "TriangleFiniteVolumeDiscretization",
    "TriangleFiniteVolumeMethodPlan",
    "TriangleFiniteVolumePlan",
    "TriangleFiniteVolumeQualityReport",
    "evaluate_triangle_fv_geometry",
    "read_triangle_fv_archive",
    "write_triangle_fv_archive",
    "PreparedTriangleWLSQ",
    "evaluate_triangle_second_moments",
    "PreparedTriangleQuadratic",
    "TriangleKExactReconstructionPlan",
    "TriangleQuadraticReport",
    "TriangleLimiterKind",
    "TriangleMUSCLReconstructionPlan",
    "TriangleWLSQReport",
    "TriangleViscousFluxPlan",
    "TriangleViscousStabilityReport",
    "AdvectionForm",
    "AdvectionReconstruction",
    "ConservativeAdvectionPlan",
    "ConservativeBoundaryCondition",
    "ConservativeBoundaryKind",
    "ConservativeDiffusionPlan",
    "FiniteVolumePrecisionPolicy",
    "PrecisionDType",
    "ViscousFluxPlan",
    "ViscousStabilityReport",
    "FaceCoefficientPlan",
    "FaceInterpolationKind",
    "PreparedConservativeAdvection",
    "PreparedConservativeDiffusion",
    "AbstractFaceReconstructionPlan",
    "AbstractFiniteVolumeBoundary",
    "AbstractArbitraryNormalNumericalFluxPlan",
    "AbstractNumericalFluxPlan",
    "AbstractSymmetricTwoPointFluxPlan",
    "AbstractSlopeLimiter",
    "AbstractWavePropagationPlan",
    "ConservativeMultiblockFluxResult",
    "ConservativeMultiblockInterfacePlan",
    "CharacteristicReconstructionPlan",
    "CharacteristicSystem",
    "ConstantStateBoundary",
    "ConvexStateLimiterPlan",
    "ConservativeFaceClosurePlan",
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
    "MHDCTRateResult",
    "UpwindConstrainedTransportPlan",
    "FiniteVolumePlan",
    "FaceVelocity",
    "MACOperatorPlan",
    "MACOperatorReport",
    "PreparedMACOperators",
    "FiniteVolumeEntropyDiagnostics",
    "FiniteVolumeResidualDiagnostics",
    "integrated_finite_volume_relative_entropy",
    "HLLCFluxPlan",
    "HLLDFluxPlan",
    "HLLFluxPlan",
    "EinfeldtHLLFluxPlan",
    "FiniteVolumeAdmissibilityReport",
    "FluxPositivityPlan",
    "PositivityBlendResult",
    "HighResolutionMethod",
    "HighResolutionReconstructionPlan",
    "ExplicitStabilizationPlan",
    "FilterADPolicy",
    "TENOQualification",
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
