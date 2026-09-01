#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# ruff: noqa: F401

"""Commercially auditable material-point methods and runtime contracts."""

from ._boundary import PrescribedGridVelocityPlan, PrescribedGridVelocityResult
from ._commercial import (
    MPMClaimOutcome,
    MPMClaimTuple,
    MPMCommercialFailure,
    MPMDerivativeEvidence,
    MPMDerivativeKind,
    MPMEventJournal,
    MPMIntendedUse,
    MPMOperationalStatus,
    MPMReleaseEvidenceBundle,
    MPMReleaseGate,
    MPMReleaseGateEvidence,
    MPMRunProvenance,
    MPMSupportDecision,
    MPMSupportMatrix,
    MPMTopologyJournal,
)
from ._contact import (
    AbstractMPMFrictionPlan,
    MPMGridConstraintResult,
    RigidMPMContactPlan,
    SharpCoulombMPMFrictionPlan,
    SmoothCoulombMPMFrictionPlan,
)
from ._contact_kway import (
    apply_rigid_actor_reactions,
    KWayMPMContactPlan,
    MPMContactGraph,
    MPMKWayContactResult,
    MPMRigidActorState,
)
from ._derivatives import (
    branchwise_gradient,
    generalized_contact_derivative,
    locate_event,
    MPMEventLocalizationResult,
    MPMGradientResult,
    nondifferentiable_result,
    saltation_action,
    smooth_surrogate_gradient,
    stochastic_derivative_estimate,
)
from ._distributed import (
    distributed_global_transaction,
    distributed_p2g_reduce,
    exchange_block_halo,
    migrate_particles,
    MPMDistributedEvidence,
    MPMDistributedPlan,
    MPMDistributedTransaction,
    MPMParticleMigration,
    MPMShardCheckpointManifest,
    particle_owners,
)
from ._domain import MPMParticleDomainPlan
from ._dynamics import ExternalMPMAcceleration, PreparedMPMDynamics
from ._execution import (
    deterministic_global_sum,
    fused_contact_projection,
    fused_route_reduction,
    MPMCapacityCertificate,
    MPMDeterminismMode,
    MPMExecutionPlan,
    MPMKernelRealization,
)
from ._fields import (
    MPMMaterialBank,
    MPMMaterialBankEntry,
    MPMMaterialBankState,
    MPMNodalFieldPlan,
)
from ._fracture import (
    CPICCompatibilityState,
    CPICFracturePlan,
    MPMFieldPartitionFracturePlan,
    MPMFractureTopologyState,
)
from ._lifecycle_amr import (
    MPMAMRPlan,
    MPMAMRTopologyJournal,
    MPMCapacityBucketPlan,
    MPMLifecycleEvidence,
    MPMLifecycleResult,
    MPMLifecycleState,
    MPMPageTablePlan,
    MPMPageTableState,
    MPMParticleLifecyclePlan,
)
from ._method import ExplicitMPMMethodPlan, MPMResourcePolicy
from ._qualification_commercial import (
    assess_release,
    MPMCommercialProfile,
    MPMCommercialProfileKind,
    MPMIndependentReview,
    MPMReleaseAssessment,
    MPMStandardsTrace,
    MPMStandardsTraceabilityMatrix,
)
from ._rigid_coupling import (
    PreparedRigidMPMCoupling,
    RigidMPMConstraintPayload,
    RigidMPMCouplingEvaluation,
    RigidMPMCouplingMode,
    RigidMPMCouplingPlan,
    RigidMPMCouplingState,
    RigidMPMCouplingStepResult,
    RigidMPMRouteCacheCertificate,
)
from ._schedule import (
    AbstractExplicitMPMSchedule,
    AffineMUSLMPMSchedule,
    MUSLMPMSchedule,
    PostAdvectionMUSLMPMSchedule,
    USFMPMSchedule,
    USLMPMSchedule,
)
from ._storage import (
    AbstractMPMNodalStoragePlan,
    BlockSparseMPMNodalStoragePlan,
    DenseMPMNodalStoragePlan,
    MPMActiveBlockPlan,
    MPMActiveBlockState,
)
from ._transfer import APICGatherResult
from ._types import (
    MPMDiagnostics,
    MPMEnergyLedger,
    MPMGridState,
    MPMLimitingProcess,
    MPMParticleState,
    MPMPreparationEvidence,
    MPMRejectionReason,
    MPMRunStatus,
    MPMRuntimeState,
    MPMScheduleEvidence,
    MPMStepRestriction,
    MPMStepResult,
    MPMTransferEvidence,
)
from ._velocity_transfer import (
    AbstractMPMAdvectionPlan,
    AbstractMPMVelocityTransferPlan,
    APICTransferPlan,
    apply_velocity_transfer,
    FLIPTransferPlan,
    MidpointAdvectionPlan,
    MPMVelocityTransferResult,
    PICAdvectionPlan,
    PICFLIPTransferPlan,
    PICTransferPlan,
    TransferredVelocityAdvectionPlan,
)


__all__ = [name for name in globals() if not name.startswith("_")]
