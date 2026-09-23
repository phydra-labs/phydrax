#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import TYPE_CHECKING

from ._compressible_adaptivity import (
    KineticAMRTransferEvidence,
    KineticAMRTransferPlan,
    KineticAMRTransferResult,
    KineticMultiblockInterfacePlan,
    MappedKineticGridPlan,
    MovingKineticGeometryEvidence,
    MovingKineticGeometryPlan,
    MovingKineticGeometryResult,
    PredictiveKineticRefinementPlan,
    PredictiveRefinementEvidence,
)
from ._compressible_contracts import (
    CompressibleKineticConservationEvidence,
    CompressibleKineticMacroscopicState,
    CompressibleKineticModelKind,
    CompressibleKineticPopulationState,
    CompressibleKineticStepResult,
    CompressibleKineticSupportTuple,
    KineticPopulationFieldSpec,
    KineticPopulationLayout,
    KineticPopulationRole,
)
from ._compressible_execution import (
    CompressibleKineticPrecisionPolicy,
    IntegerLatticeTransportEvidence,
    IntegerLatticeTransportPlan,
    KineticStorageLayout,
    KineticStoragePlan,
    KineticVelocityPartitionEvidence,
    KineticVelocityPartitionPlan,
    KineticWorksetPlan,
    ScaledPopulationField,
)
from ._compressible_frame import (
    AdaptiveGaugePlan,
    IntegerKineticFramePlan,
    KineticFrameRemapEvidence,
    KineticFrameRemapResult,
    remap_kinetic_frame,
)
from ._compressible_multiphysics import (
    EquilibratingKineticSourceLiftPlan,
    KineticAuxiliaryState,
    KineticEffectiveTransportEvidence,
    KineticEffectiveTransportPlan,
    KineticRadiationAblationEvidence,
    KineticRadiationAblationPlan,
    KineticSourceEvidence,
    KineticSpeciesTransportEvidence,
    KineticSpeciesTransportPlan,
    KineticSpectralAnalysisPlan,
    KineticSpectrum,
)
from ._compressible_rules import (
    CompressibleVelocityRule,
    d3q33_filtered_rule,
    d3q39_guided_rule,
    d3q343_entropic_rule,
    shift_compressible_velocity_rule,
)
from ._compressible_runtime import (
    CompressibleKineticRuntimeEvidence,
    CompressibleKineticRuntimePlan,
    CompressibleKineticRuntimeResult,
    CompressibleKineticRuntimeState,
)
from ._energy_equilibrium import (
    EnergyEquilibriumEvidence,
    EnergyEquilibriumResult,
    EnergyEquilibriumStatus,
    PositiveEnergyEquilibriumPlan,
)
from ._filtered_d3q33 import FilteredD3Q33Evidence, FilteredD3Q33Plan
from ._positive_kinetic import (
    entropic_d3q343_plan,
    guided_d3q39_plan,
    PositiveCompressibleKineticPlan,
    PositiveKineticCollisionKind,
)
from ._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
    QuadratureMomentCertification,
    VelocityTransportKind,
)
from ._quasi_equilibrium import (
    FullRangeQuasiEquilibriumPlan,
    QuasiEquilibriumEvidence,
    QuasiEquilibriumSlowFamily,
)
from ._semi_lagrangian import (
    CoupledD2V37TransportEvidence,
    CoupledD2V37TransportResult,
    CoupledD2V37TransportStatus,
    DeclaredPopulationMomentMap,
    PeriodicUniformGridDepartureTransfer,
    PreparedCoupledD2V37OffLatticeTransport,
    PreparedOffLatticeSemiLagrangianDVM,
    SemiLagrangianTransferRequirements,
    SemiLagrangianTransportEvidence,
)


if TYPE_CHECKING:
    from ._checkpoint import (
        read_smooth_compressible_d2v_checkpoint,
        SmoothCompressibleD2VCheckpoint,
        SmoothCompressibleD2VCheckpointPlan,
        write_smooth_compressible_d2v_checkpoint,
    )
    from ._hybrid import (
        AtomicHybridUpdateEvidence,
        AtomicHybridUpdateResult,
        CommonFVKineticFluxEvidence,
        ConformingFVKineticState,
        FixedConformingFVKineticInterfacePlan,
        KineticShockSensorEvidence,
        KineticShockSensorPlan,
    )
    from ._hybrid_runtime import (
        DynamicHybridCheckpoint,
        DynamicHybridCompositeState,
        DynamicHybridMigrationEvidence,
        DynamicHybridMigrationResult,
        DynamicHybridOwnershipDecision,
        DynamicHybridOwnershipPlan,
        DynamicHybridOwnershipState,
        FixedHybridStageEvidence,
        FixedPartitionHybridAdvanceEvidence,
        FixedPartitionHybridAdvanceResult,
        FixedPartitionHybridAudit,
        FixedPartitionHybridCheckpoint,
        FixedPartitionHybridState,
        FixedPartitionHybridStatus,
        PreparedFixedPartitionHybridRuntime,
    )
    from ._learned_thermal_research import (
        ExtendedParticleEquilibriumEvidence,
        ExtendedParticleEquilibriumResult,
        IntegerVelocityFrameAdmissibilityEvidence,
        IntegerVelocityFrameShiftEvidence,
        IntegerVelocityFrameShiftPlan,
        IntegerVelocityFrameShiftResult,
        LearnedThermalEnergyEvidence,
        LearnedThermalEnergyResult,
        LearnedThermalResearchStatus,
        MatchedThermalCrossRelaxationPlan,
        PositiveLearnedThermalEnergyPlan,
        PressureExtendedParticleEquilibriumPlan,
        ThermalCrossRelaxationEvidence,
        ThermalCrossRelaxationResult,
        ThermalFrameMoments,
        ThermalQuasiEquilibriumEvidence,
        ThermalQuasiEquilibriumResult,
    )
    from ._smooth_compressible import (
        smooth_compressible_d2v17_method,
        smooth_compressible_d2v37_off_lattice_method,
        SmoothCompressibleCollisionEvidence,
        SmoothCompressibleD2VKineticMethod,
        SmoothCompressibleEquilibriumEvidence,
        SmoothCompressibleKineticState,
        SmoothCompressibleLearnedCollisionResult,
        SmoothCompressibleLearnedEquilibriumEvidence,
        SmoothCompressibleMoments,
        SmoothCompressibleRealizabilityEvidence,
    )
    from ._spatial import (
        D2V17PeriodicTransportPlan,
        PreparedSmoothCompressibleD2V17SpatialDynamics,
        SmoothCompressibleD2V17SpatialPlan,
        SmoothCompressibleD2VConservationEvidence,
        SmoothCompressibleD2VStepEvidence,
        SmoothCompressibleD2VStepResult,
        SmoothCompressibleD2VStepStatus,
    )
    from ._spatial_boundary import (
        AbstractSmoothCompressibleD2VBoundaryPlan,
        CompiledD2V17BoundaryTopology,
        EquilibriumReservoirD2VBoundaryPlan,
        MaxwellThermalD2VBoundaryPlan,
        OutwardExtrapolationD2VBoundaryPlan,
        PeriodicD2VBoundaryPlan,
        SmoothCompressibleD2VBoundaryCorner,
        SmoothCompressibleD2VBoundaryFace,
        SmoothCompressibleD2VBoundaryHistory,
        SmoothCompressibleD2VBoundaryResult,
        SmoothCompressibleD2VBoundaryStatus,
        SmoothCompressibleD2VLinkOwner,
        SmoothCompressibleD2VReservoirParameters,
        SpecularAdiabaticD2VBoundaryPlan,
    )
    from ._spatial_forcing import (
        SmoothCompressibleD2VBodyForcingPlan,
        SmoothCompressibleD2VForcingEvidence,
        SmoothCompressibleD2VForcingResult,
        SmoothCompressibleD2VForcingStatus,
        ZeroSmoothCompressibleD2VForcingPlan,
    )


_SMOOTH_COMPRESSIBLE_EXPORTS = frozenset(
    {
        "SmoothCompressibleCollisionEvidence",
        "SmoothCompressibleD2VKineticMethod",
        "SmoothCompressibleEquilibriumEvidence",
        "SmoothCompressibleKineticState",
        "SmoothCompressibleMoments",
        "SmoothCompressibleRealizabilityEvidence",
        "SmoothCompressibleLearnedCollisionResult",
        "SmoothCompressibleLearnedEquilibriumEvidence",
        "smooth_compressible_d2v17_method",
        "smooth_compressible_d2v37_off_lattice_method",
    }
)
_HYBRID_EXPORTS = frozenset(
    {
        "AtomicHybridUpdateEvidence",
        "AtomicHybridUpdateResult",
        "CommonFVKineticFluxEvidence",
        "ConformingFVKineticState",
        "FixedConformingFVKineticInterfacePlan",
        "KineticShockSensorEvidence",
        "KineticShockSensorPlan",
    }
)
_HYBRID_RUNTIME_EXPORTS = frozenset(
    {
        "DynamicHybridCheckpoint",
        "DynamicHybridCompositeState",
        "DynamicHybridMigrationEvidence",
        "DynamicHybridMigrationResult",
        "DynamicHybridOwnershipDecision",
        "DynamicHybridOwnershipPlan",
        "DynamicHybridOwnershipState",
        "FixedHybridStageEvidence",
        "FixedPartitionHybridAdvanceEvidence",
        "FixedPartitionHybridAdvanceResult",
        "FixedPartitionHybridAudit",
        "FixedPartitionHybridCheckpoint",
        "FixedPartitionHybridState",
        "FixedPartitionHybridStatus",
        "PreparedFixedPartitionHybridRuntime",
    }
)
_SPATIAL_EXPORTS = frozenset(
    {
        "D2V17PeriodicTransportPlan",
        "PreparedSmoothCompressibleD2V17SpatialDynamics",
        "SmoothCompressibleD2V17SpatialPlan",
        "SmoothCompressibleD2VConservationEvidence",
        "SmoothCompressibleD2VStepEvidence",
        "SmoothCompressibleD2VStepResult",
        "SmoothCompressibleD2VStepStatus",
    }
)
_BOUNDARY_EXPORTS = frozenset(
    {
        "AbstractSmoothCompressibleD2VBoundaryPlan",
        "CompiledD2V17BoundaryTopology",
        "EquilibriumReservoirD2VBoundaryPlan",
        "MaxwellThermalD2VBoundaryPlan",
        "OutwardExtrapolationD2VBoundaryPlan",
        "PeriodicD2VBoundaryPlan",
        "SmoothCompressibleD2VBoundaryCorner",
        "SmoothCompressibleD2VBoundaryFace",
        "SmoothCompressibleD2VBoundaryHistory",
        "SmoothCompressibleD2VBoundaryResult",
        "SmoothCompressibleD2VBoundaryStatus",
        "SmoothCompressibleD2VLinkOwner",
        "SmoothCompressibleD2VReservoirParameters",
        "SpecularAdiabaticD2VBoundaryPlan",
    }
)
_FORCING_EXPORTS = frozenset(
    {
        "SmoothCompressibleD2VBodyForcingPlan",
        "SmoothCompressibleD2VForcingEvidence",
        "SmoothCompressibleD2VForcingResult",
        "SmoothCompressibleD2VForcingStatus",
        "ZeroSmoothCompressibleD2VForcingPlan",
    }
)
_CHECKPOINT_EXPORTS = frozenset(
    {
        "SmoothCompressibleD2VCheckpoint",
        "SmoothCompressibleD2VCheckpointPlan",
        "read_smooth_compressible_d2v_checkpoint",
        "write_smooth_compressible_d2v_checkpoint",
    }
)
_THERMAL_RESEARCH_EXPORTS = frozenset(
    {
        "ExtendedParticleEquilibriumEvidence",
        "ExtendedParticleEquilibriumResult",
        "IntegerVelocityFrameAdmissibilityEvidence",
        "IntegerVelocityFrameShiftEvidence",
        "IntegerVelocityFrameShiftPlan",
        "IntegerVelocityFrameShiftResult",
        "LearnedThermalEnergyEvidence",
        "LearnedThermalEnergyResult",
        "LearnedThermalResearchStatus",
        "MatchedThermalCrossRelaxationPlan",
        "PositiveLearnedThermalEnergyPlan",
        "PressureExtendedParticleEquilibriumPlan",
        "ThermalCrossRelaxationEvidence",
        "ThermalCrossRelaxationResult",
        "ThermalFrameMoments",
        "ThermalQuasiEquilibriumEvidence",
        "ThermalQuasiEquilibriumResult",
    }
)


def __getattr__(name: str) -> object:
    if name in _SMOOTH_COMPRESSIBLE_EXPORTS:
        from . import _smooth_compressible

        return getattr(_smooth_compressible, name)
    if name in _HYBRID_EXPORTS:
        from . import _hybrid

        return getattr(_hybrid, name)
    if name in _HYBRID_RUNTIME_EXPORTS:
        from . import _hybrid_runtime

        return getattr(_hybrid_runtime, name)
    if name in _SPATIAL_EXPORTS:
        from . import _spatial

        return getattr(_spatial, name)
    if name in _BOUNDARY_EXPORTS:
        from . import _spatial_boundary

        return getattr(_spatial_boundary, name)
    if name in _FORCING_EXPORTS:
        from . import _spatial_forcing

        return getattr(_spatial_forcing, name)
    if name in _CHECKPOINT_EXPORTS:
        from . import _checkpoint

        return getattr(_checkpoint, name)
    if name in _THERMAL_RESEARCH_EXPORTS:
        from . import _learned_thermal_research

        return getattr(_learned_thermal_research, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return list(__all__)


__all__ = [
    "AbstractSmoothCompressibleD2VBoundaryPlan",
    "AtomicHybridUpdateEvidence",
    "AtomicHybridUpdateResult",
    "CertifiedDiscreteVelocityQuadrature",
    "CommonFVKineticFluxEvidence",
    "CompiledD2V17BoundaryTopology",
    "ConformingFVKineticState",
    "CoupledD2V37TransportEvidence",
    "CoupledD2V37TransportResult",
    "CoupledD2V37TransportStatus",
    "D2V17PeriodicTransportPlan",
    "DeclaredPopulationMomentMap",
    "DynamicHybridCheckpoint",
    "DynamicHybridCompositeState",
    "DynamicHybridMigrationEvidence",
    "DynamicHybridMigrationResult",
    "DynamicHybridOwnershipDecision",
    "DynamicHybridOwnershipPlan",
    "DynamicHybridOwnershipState",
    "EnergyEquilibriumEvidence",
    "EnergyEquilibriumResult",
    "EnergyEquilibriumStatus",
    "EquilibriumReservoirD2VBoundaryPlan",
    "ExtendedParticleEquilibriumEvidence",
    "ExtendedParticleEquilibriumResult",
    "FixedConformingFVKineticInterfacePlan",
    "FixedHybridStageEvidence",
    "FixedPartitionHybridAdvanceEvidence",
    "FixedPartitionHybridAdvanceResult",
    "FixedPartitionHybridAudit",
    "FixedPartitionHybridCheckpoint",
    "FixedPartitionHybridState",
    "FixedPartitionHybridStatus",
    "IntegerVelocityFrameAdmissibilityEvidence",
    "IntegerVelocityFrameShiftEvidence",
    "IntegerVelocityFrameShiftPlan",
    "IntegerVelocityFrameShiftResult",
    "KineticShockSensorEvidence",
    "KineticShockSensorPlan",
    "LearnedThermalEnergyEvidence",
    "LearnedThermalEnergyResult",
    "LearnedThermalResearchStatus",
    "MatchedThermalCrossRelaxationPlan",
    "MaxwellThermalD2VBoundaryPlan",
    "OutwardExtrapolationD2VBoundaryPlan",
    "PeriodicD2VBoundaryPlan",
    "PeriodicUniformGridDepartureTransfer",
    "PositiveEnergyEquilibriumPlan",
    "PositiveLearnedThermalEnergyPlan",
    "PreparedCoupledD2V37OffLatticeTransport",
    "PreparedFixedPartitionHybridRuntime",
    "PreparedOffLatticeSemiLagrangianDVM",
    "PreparedSmoothCompressibleD2V17SpatialDynamics",
    "PressureExtendedParticleEquilibriumPlan",
    "QuadratureMomentCertification",
    "SemiLagrangianTransferRequirements",
    "SemiLagrangianTransportEvidence",
    "SmoothCompressibleCollisionEvidence",
    "SmoothCompressibleD2V17SpatialPlan",
    "SmoothCompressibleD2VBodyForcingPlan",
    "SmoothCompressibleD2VBoundaryCorner",
    "SmoothCompressibleD2VBoundaryFace",
    "SmoothCompressibleD2VBoundaryHistory",
    "SmoothCompressibleD2VBoundaryResult",
    "SmoothCompressibleD2VBoundaryStatus",
    "SmoothCompressibleD2VCheckpoint",
    "SmoothCompressibleD2VCheckpointPlan",
    "SmoothCompressibleD2VConservationEvidence",
    "SmoothCompressibleD2VForcingEvidence",
    "SmoothCompressibleD2VForcingResult",
    "SmoothCompressibleD2VForcingStatus",
    "SmoothCompressibleD2VKineticMethod",
    "SmoothCompressibleD2VLinkOwner",
    "SmoothCompressibleD2VReservoirParameters",
    "SmoothCompressibleD2VStepEvidence",
    "SmoothCompressibleD2VStepResult",
    "SmoothCompressibleD2VStepStatus",
    "SmoothCompressibleEquilibriumEvidence",
    "SmoothCompressibleKineticState",
    "SmoothCompressibleLearnedCollisionResult",
    "SmoothCompressibleLearnedEquilibriumEvidence",
    "SmoothCompressibleMoments",
    "SmoothCompressibleRealizabilityEvidence",
    "SpecularAdiabaticD2VBoundaryPlan",
    "ThermalCrossRelaxationEvidence",
    "ThermalCrossRelaxationResult",
    "ThermalFrameMoments",
    "ThermalQuasiEquilibriumEvidence",
    "ThermalQuasiEquilibriumResult",
    "VelocityTransportKind",
    "ZeroSmoothCompressibleD2VForcingPlan",
    "d2v17_quadrature",
    "d2v37_off_lattice_quadrature",
    "read_smooth_compressible_d2v_checkpoint",
    "smooth_compressible_d2v17_method",
    "smooth_compressible_d2v37_off_lattice_method",
    "write_smooth_compressible_d2v_checkpoint",
    "AdaptiveGaugePlan",
    "CompressibleKineticRuntimeEvidence",
    "CompressibleKineticRuntimePlan",
    "CompressibleKineticRuntimeResult",
    "CompressibleKineticRuntimeState",
    "CompressibleKineticConservationEvidence",
    "CompressibleKineticMacroscopicState",
    "CompressibleKineticModelKind",
    "CompressibleKineticPopulationState",
    "CompressibleKineticPrecisionPolicy",
    "CompressibleKineticStepResult",
    "CompressibleKineticSupportTuple",
    "CompressibleVelocityRule",
    "EquilibratingKineticSourceLiftPlan",
    "FilteredD3Q33Evidence",
    "FilteredD3Q33Plan",
    "FullRangeQuasiEquilibriumPlan",
    "IntegerKineticFramePlan",
    "IntegerLatticeTransportEvidence",
    "IntegerLatticeTransportPlan",
    "KineticAMRTransferEvidence",
    "KineticAMRTransferPlan",
    "KineticAMRTransferResult",
    "KineticAuxiliaryState",
    "KineticEffectiveTransportEvidence",
    "KineticEffectiveTransportPlan",
    "KineticFrameRemapEvidence",
    "KineticFrameRemapResult",
    "KineticMultiblockInterfacePlan",
    "KineticPopulationFieldSpec",
    "KineticPopulationLayout",
    "KineticPopulationRole",
    "KineticRadiationAblationEvidence",
    "KineticRadiationAblationPlan",
    "KineticSourceEvidence",
    "KineticSpeciesTransportEvidence",
    "KineticSpeciesTransportPlan",
    "KineticSpectralAnalysisPlan",
    "KineticSpectrum",
    "KineticStorageLayout",
    "KineticStoragePlan",
    "KineticWorksetPlan",
    "KineticVelocityPartitionEvidence",
    "KineticVelocityPartitionPlan",
    "MappedKineticGridPlan",
    "MovingKineticGeometryEvidence",
    "MovingKineticGeometryPlan",
    "MovingKineticGeometryResult",
    "PositiveCompressibleKineticPlan",
    "PositiveKineticCollisionKind",
    "PredictiveKineticRefinementPlan",
    "PredictiveRefinementEvidence",
    "QuasiEquilibriumEvidence",
    "QuasiEquilibriumSlowFamily",
    "ScaledPopulationField",
    "d3q33_filtered_rule",
    "d3q39_guided_rule",
    "d3q343_entropic_rule",
    "entropic_d3q343_plan",
    "guided_d3q39_plan",
    "remap_kinetic_frame",
    "shift_compressible_velocity_rule",
]
