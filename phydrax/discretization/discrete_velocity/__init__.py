#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._energy_equilibrium import (
    EnergyEquilibriumEvidence,
    EnergyEquilibriumResult,
    EnergyEquilibriumStatus,
    PositiveEnergyEquilibriumPlan,
)
from ._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
    QuadratureMomentCertification,
    VelocityTransportKind,
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


__all__ = [
    "EnergyEquilibriumEvidence",
    "CoupledD2V37TransportEvidence",
    "CoupledD2V37TransportResult",
    "CoupledD2V37TransportStatus",
    "DeclaredPopulationMomentMap",
    "EnergyEquilibriumResult",
    "EnergyEquilibriumStatus",
    "CertifiedDiscreteVelocityQuadrature",
    "PreparedOffLatticeSemiLagrangianDVM",
    "PeriodicUniformGridDepartureTransfer",
    "PreparedCoupledD2V37OffLatticeTransport",
    "QuadratureMomentCertification",
    "SemiLagrangianTransferRequirements",
    "SemiLagrangianTransportEvidence",
    "VelocityTransportKind",
    "PositiveEnergyEquilibriumPlan",
    "d2v17_quadrature",
    "d2v37_off_lattice_quadrature",
]
