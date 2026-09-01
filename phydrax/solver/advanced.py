"""Advanced compressible multiphysics plans and evidence types."""

from ..discretization.finite_volume._mhd_boundary import (
    AbstractConstrainedMHDBoundary,
    ConstrainedMHDBoundarySet,
    MHDBoundaryTrace,
    MHDOutflowBoundary,
    PerfectlyConductingWallBoundary,
    PrescribedMHDInflowBoundary,
)
from ..discretization.finite_volume._mhd_closure import (
    ConstrainedMHDClosurePlan,
    LearnedClosureDiagnostics,
    MultiresolutionMHDClosurePlan,
    StructurePreservingFaceClosurePlan,
)
from ..discretization.finite_volume._mhd_ct import ConstrainedMagneticStateLayout
from ..discretization.finite_volume._mhd_reconstruction import (
    MHDPrimitiveReconstructionPlan,
)
from ..discretization.finite_volume._uct import (
    FluxCTElectromotivePlan,
    HLLUCTElectromotivePlan,
    UCTElectromotiveResult,
)
from ._amr_multiphysics import (
    AMRTopologyEpoch,
    AMRTopologyReplayPlan,
    CompositeAMRGravityDiagnostics,
    CompositeAMRGravityPlan,
)
from ._balance_law import (
    AbstractPreparedAcceptedStepCoupling,
    BalanceLawAcceptedStepContext,
    BalanceLawAcceptedStepCouplingAdvance,
)
from ._balance_law_composition import (
    AdditiveIMEXTableau,
    BalanceLawCompositionPlan,
    BalanceLawIntegrationMode,
)
from ._constrained_mhd import ConstrainedMHDAcceptedIntegralLedger
from ._distributed_mhd import (
    DegreeAwareEntityOwnership,
    DistributedGravitySolvePlan,
    DistributedGravitySolveResult,
    DistributedMHDReconciliationDiagnostics,
    DistributedMultiphysicsSynchronizationPlan,
    DistributedMultiphysicsSynchronizationResult,
    reconcile_distributed_mhd_entities,
)
from ._isolated_gravity import (
    IsolatedCartesianGravityPlan,
    IsolatedGravityDiagnostics,
)
from ._mapped_mhd import (
    MappedALEConstrainedTransportPlan,
    MappedCochainGeometry,
    MappedFaradayDiagnostics,
)
from ._mhd_advanced import (
    DualEnergyMHDPlan,
    DualEnergyMHDState,
    LocalMHDPositivityPlan,
    LocalMHDPositivityResult,
    MHDCharacteristicReconstructionPlan,
    MHDCTUPredictorPlan,
)
from ._mhd_amr import (
    ConstrainedMHDAMRSynchronizationPlan,
    DivergenceFreeMagneticTransferPlan,
    ElectromotiveForceRegister,
    MagneticAMRTransferDiagnostics,
)
from ._modal_forcing import (
    ModalForcingBasis,
    ModalOUForcingDiagnostics,
    ModalOUForcingPlan,
    PreparedModalOUForcing,
)
from ._multiphysics_inference import (
    FieldObservationPlan,
    ParticleMarginalLikelihoodPlan,
    SimulationSensitivityReport,
    WhitenedFieldInferencePlan,
)
from ._nonideal_mhd import (
    AnisotropicThermalTransportDiagnostics,
    AnisotropicThermalTransportPlan,
    NonIdealMHDDiagnostics,
    NonIdealMHDPlan,
)
from ._radiation import (
    GrayRadiationDiffusionPlan,
    RadiationDiffusionDiagnostics,
    RadiationMatterState,
)
from ._self_gravity import (
    ConservativeGravityEnergyCoupling,
    ConservativeGravityEnergyDiagnostics,
)
from ._thermochemistry import (
    PreparedThermochemistryProcess,
    ThermochemistryDiagnostics,
    ThermochemistryProcessPlan,
)
from ._unstructured_mhd import (
    UnstructuredConstrainedTransportPlan,
    UnstructuredFaradayDiagnostics,
    UnstructuredMagneticState,
)


__all__ = [
    "AdditiveIMEXTableau",
    "AnisotropicThermalTransportDiagnostics",
    "AnisotropicThermalTransportPlan",
    "DistributedGravitySolvePlan",
    "DistributedGravitySolveResult",
    "DistributedMultiphysicsSynchronizationPlan",
    "DistributedMultiphysicsSynchronizationResult",
    "IsolatedCartesianGravityPlan",
    "IsolatedGravityDiagnostics",
    "MultiresolutionMHDClosurePlan",
    "AbstractPreparedAcceptedStepCoupling",
    "BalanceLawAcceptedStepContext",
    "BalanceLawAcceptedStepCouplingAdvance",
    "BalanceLawCompositionPlan",
    "BalanceLawIntegrationMode",
    "ConservativeGravityEnergyCoupling",
    "ConservativeGravityEnergyDiagnostics",
    "ConstrainedMHDAcceptedIntegralLedger",
    "AMRTopologyEpoch",
    "AMRTopologyReplayPlan",
    "AbstractConstrainedMHDBoundary",
    "CompositeAMRGravityDiagnostics",
    "CompositeAMRGravityPlan",
    "ConstrainedMHDBoundarySet",
    "ConstrainedMHDAMRSynchronizationPlan",
    "ConstrainedMHDClosurePlan",
    "ConstrainedMagneticStateLayout",
    "DegreeAwareEntityOwnership",
    "DivergenceFreeMagneticTransferPlan",
    "DistributedMHDReconciliationDiagnostics",
    "DualEnergyMHDPlan",
    "DualEnergyMHDState",
    "ElectromotiveForceRegister",
    "FieldObservationPlan",
    "FluxCTElectromotivePlan",
    "GrayRadiationDiffusionPlan",
    "HLLUCTElectromotivePlan",
    "LearnedClosureDiagnostics",
    "LocalMHDPositivityPlan",
    "LocalMHDPositivityResult",
    "MHDBoundaryTrace",
    "MHDCharacteristicReconstructionPlan",
    "MHDCTUPredictorPlan",
    "MHDOutflowBoundary",
    "MHDPrimitiveReconstructionPlan",
    "MagneticAMRTransferDiagnostics",
    "MappedALEConstrainedTransportPlan",
    "MappedCochainGeometry",
    "MappedFaradayDiagnostics",
    "ModalForcingBasis",
    "ModalOUForcingDiagnostics",
    "ModalOUForcingPlan",
    "NonIdealMHDDiagnostics",
    "NonIdealMHDPlan",
    "ParticleMarginalLikelihoodPlan",
    "PerfectlyConductingWallBoundary",
    "PreparedModalOUForcing",
    "PreparedThermochemistryProcess",
    "PrescribedMHDInflowBoundary",
    "RadiationDiffusionDiagnostics",
    "RadiationMatterState",
    "SimulationSensitivityReport",
    "StructurePreservingFaceClosurePlan",
    "ThermochemistryDiagnostics",
    "ThermochemistryProcessPlan",
    "UCTElectromotiveResult",
    "UnstructuredConstrainedTransportPlan",
    "UnstructuredFaradayDiagnostics",
    "UnstructuredMagneticState",
    "WhitenedFieldInferencePlan",
    "reconcile_distributed_mhd_entities",
]
