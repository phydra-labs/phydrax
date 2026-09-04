# Soft robotics applications

The public surface is split by ownership. See the
[soft robotics guide](../../guides_soft_robotics.md) for capability boundaries,
composition rules, and qualified profiles.

## Shared four-space geometry and true duals

::: phydrax.metrix.AbstractStateGeometry

---

::: phydrax.metrix.LocalRetraction

---

::: phydrax.metrix.StateChartEvidence

---

::: phydrax.metrix.StateTransportEvidence

---

::: phydrax.linalg.DualSpace

---

::: phydrax.linalg.dual_transpose

## Complete plant lifecycle and codecs

::: phydrax.dynamics
    options:
      members:
        - AbstractDiscretePlant
        - PlantRuntimeState
        - PlantParameters
        - PlantStepContext
        - PlantProposal
        - PlantResetResult
        - PlantStepResult
        - PlantCheckpoint
        - PlantReplayResult
        - PlantStateVectorCodec
        - ControlVectorCodec
        - EncodedPlantState
        - EncodedPlantVector
        - EncodedControl
        - PlantVectorRole
        - PlantPowerEvidence
      show_root_heading: false
      members_order: source

## Rod foundation, basis, reconstruction, dynamics, and plants

::: phydrax.applications.solid_mechanics
    options:
      members:
        - RodPlan
        - RodState
        - PreparedRod
        - prepare_rod
        - RodEvaluation
        - evaluate_rod
        - RodDynamicsPlan
        - PreparedRodDynamics
        - prepare_rod_dynamics
        - RodStrainBasisPlan
        - RodStrainBasisEvidence
        - PreparedRodStrainBasis
        - prepare_rod_strain_basis
        - piecewise_constant_rod_strain_basis
        - shifted_legendre_rod_strain_basis
        - explicit_rod_strain_basis
        - ReducedRodPlan
        - ReducedRodState
        - PreparedReducedRod
        - prepare_reduced_rod
        - lift_configuration
        - lift_velocity_operator
        - lift_effort_pullback_operator
        - lift_reduced_rod_state
        - lift_reduced_rod_velocity
        - pullback_reduced_rod_loads
        - ReducedRodPowerEvidence
        - reduced_rod_power_evidence
        - RodFrameQueryPlan
        - RodReconstructionPlan
        - PreparedRodReconstruction
        - prepare_rod_reconstruction
        - RodReconstructionEvaluation
        - evaluate_rod_reconstruction
        - RodNativeDiscretizationDiscrepancy
        - RodDiscretizationComparison
        - compare_reduced_rod_discretizations
        - RodMaterialWorkset
        - RodConstitutiveControl
        - RodConstitutiveTrial
        - LinearElasticRodMaterialPlan
        - KelvinVoigtRodMaterialPlan
        - ReducedRodDirectLoad
        - ReducedRodMaterialState
        - ReducedRodMaterialControl
        - ReducedRodDenseCholeskyPlan
        - ReducedRodMatrixFreeCGPlan
        - PreparedReducedRodDynamics
        - prepare_reduced_rod_dynamics
        - reduced_rod_mass
        - reduced_rod_inverse_mass
        - reduced_rod_bias
        - reduced_rod_energy
        - evaluate_reduced_rod_dynamics
        - reduced_rod_forward_dynamics
        - reduced_rod_inverse_dynamics
        - ReducedRodIntegrationState
        - ReducedRodSemiImplicitVelocityEuler
        - ReducedRodImplicitMidpoint
        - initialize_reduced_rod_integration_state
        - integrate_reduced_rod_step
        - ReducedRodEnergyWorkLedger
        - PreparedReducedRodPlant
        - prepare_reduced_rod_plant
        - ReducedRodPlantState
        - ReducedRodPlantEvidence
        - ReducedRodMassResponseRevision
      show_root_heading: false
      members_order: source

## Tendons and advanced actuators

::: phydrax.applications.solid_mechanics
    options:
      members:
        - RodMaterialStation
        - TendonRoutePlan
        - PreparedTendonRoute
        - prepare_tendon_route
        - FrictionlessElasticTendonPlan
        - PreparedFrictionlessElasticTendon
        - prepare_frictionless_elastic_tendon
        - TendonActuatorState
        - TendonPayoutCommand
        - TendonActuationEvaluation
        - integrate_tendon_payout
        - evaluate_tendon_actuation
        - TendonActuatorStateBank
        - TendonDrivenRodPlant
        - PreparedTendonDrivenRodPlant
        - prepare_tendon_driven_rod_plant
        - TendonDrivenRodPlantState
        - TendonDrivenRodPlantCommand
        - TendonDrivenRodCommandBounds
        - TendonDrivenRodActuationLedger
        - TendonDrivenRodPlantStatus
        - TendonDrivenRodPlantEvidence
        - TendonDrivenRodMassResponseRevision
        - CapstanTendonFrictionPlan
        - PreparedCapstanTendonFriction
        - CapstanTendonFrictionState
        - CapstanTendonFrictionEvaluation
        - RodTubeStation
        - ReducedTubeChamberPlan
        - RegulatedReducedTubePressurePlan
        - SealedReducedTubePressurePlan
        - IntrinsicStrainActuationPlan
        - VariableStiffnessActuationPlan
        - AffineMagneticActuationPlan
      show_root_heading: false
      members_order: source

## Tasks, observations, control, and inference

::: phydrax.applications.robotics
    options:
      members:
        - ContinuumPositionTask
        - ContinuumOrientationTask
        - ContinuumPoseTask
        - ContinuumShapeTask
        - ContinuumPostureTask
        - ContinuumInverseKinematicsPlan
        - ContinuumInverseKinematicsResult
        - ContinuumDifferentialIKPlan
        - ContinuumDifferentialIKResult
        - SmoothReducedRodTrajectoryPlan
        - SmoothReducedRodTrajectoryResult
        - SmoothReducedRodReplay
        - SoftObservationLayout
        - SoftRobotObservation
        - SoftReducedStateQueryPlan
        - SoftFrameQueryPlan
        - SoftStrainQueryPlan
        - SoftTendonQueryPlan
        - SoftEnergyLoadQueryPlan
        - SoftSensorPlan
        - SoftSensorState
        - SoftObservationPlan
        - PreparedSoftObservationPlan
        - prepare_soft_observation_plan
        - PositiveParameterMap
        - BoundedParameterMap
        - SPDParameterMap
        - ReducedRodParameterization
        - CalibrationExperiment
        - CalibrationAcceptance
        - ReducedRodCalibrationProblem
        - ReducedRodCalibrationResult
        - calibrate_reduced_rod
        - FixedModeDerivativeEvidence
        - SoftCoDesignConstraint
        - CoDesignHeldOutScenario
        - SoftRobotCoDesignProblem
        - SoftRobotCoDesignResult
        - SoftPlantMPCPlan
        - SoftPlantMPCResult
        - build_soft_plant_mpc
      show_root_heading: false
      members_order: source

## Capsule contact and atomic rod contact

::: phydrax.applications.contact
    options:
      members:
        - RodCapsuleGeometryPlan
        - PreparedRodCapsuleGeometry
        - ReducedRodCapsuleContactParticipant
        - prepare_reduced_rod_contact_participant
        - RodCapsuleDualityEvidence
        - RodContactSearchPlan
        - PreparedRodContactSearch
        - RodContactSearchResult
        - RodContactSearchEvidence
        - RodContactWitnessBatch
        - RodContactManifoldState
        - RodContactManifoldTransition
        - RodContactCCDPlan
        - RodContactCCDResult
        - RodContactCCDEvidence
        - prepare_composite_contact_block
        - build_rod_contact_velocity_operator
        - build_capsule_contact_velocity_operator
        - CompositeContactResponse
        - CompositeContactResult
      show_root_heading: false
      members_order: source

---

::: phydrax.applications.solid_mechanics
    options:
      members:
        - FRICTIONLESS_ROD_CONTACT_CAPABILITY
        - ISOTROPIC_COULOMB_ROD_CONTACT_CAPABILITY
        - PreparedReducedRodContactPlant
        - prepare_reduced_rod_contact_plant
        - ReducedRodContactPlantState
        - ReducedRodContactPlantStatus
        - ReducedRodContactPlantStepEvidence
        - ReducedRodContactEnergyEvidence
        - ReducedRodContactConservationEvidence
      show_root_heading: false
      members_order: source

## Floating and hybrid plants

::: phydrax.applications.solid_mechanics
    options:
      members:
        - FloatingReducedRodPlan
        - FloatingReducedRodState
        - PreparedFloatingReducedRod
        - prepare_floating_reduced_rod
        - floating_reduced_rod_mass
        - floating_reduced_rod_inverse_mass
        - floating_reduced_rod_bias
        - floating_reduced_rod_gravity
        - evaluate_floating_reduced_rod
        - floating_reduced_rod_forward_dynamics
        - floating_reduced_rod_inverse_dynamics
        - FloatingReducedRodPlant
        - FloatingReducedRodPlantState
        - FloatingReducedRodPlantControl
        - FloatingReducedRodPlantEvidence
      show_root_heading: false
      members_order: source

---

::: phydrax.applications.robotics
    options:
      members:
        - AttachmentFrameState
        - FrameWrench
        - RigidFrameAttachmentPlan
        - SoftEndpointAttachmentPlan
        - RigidSoftAttachmentPlan
        - AttachmentWrenchCommand
        - AttachmentKinematics
        - AttachmentWrenchRoute
        - transform_attachment_frame
        - evaluate_attachment_kinematics
        - route_attachment_wrench
        - AbstractHybridPlantPort
        - PreparedReducedRodPlantPort
        - FloatingReducedRodPlantPort
        - TendonDrivenRodPlantPort
        - SynchronizedStepPolicy
        - HybridRigidSoftState
        - HybridRigidSoftCommands
        - HybridRigidSoftStatus
        - HybridResetEvidence
        - HybridStepEvidence
        - HybridRigidSoftPlant
      show_root_heading: false
      members_order: source

## FEM and MPM soft plants

::: phydrax.applications.robotics
    options:
      members:
        - FEMSoftLoadLayout
        - FEMSoftCommand
        - FEMSoftLoads
        - FEMSoftParameters
        - FEMSoftState
        - FEMSoftSensorLayout
        - FEMSoftObservation
        - FEMSoftCapabilityManifest
        - FEMSoftResetEvidence
        - FEMSoftStepEvidence
        - FEMSoftPlant
        - FEM_LINEAR_ELASTICITY_CAPABILITY_ID
        - FEM_HYPERELASTICITY_CAPABILITY_ID
        - FEM_VISCOELASTICITY_CAPABILITY_ID
        - FEM_PRESSURE_ACTUATION_CAPABILITY_ID
        - FEM_FIBER_ACTUATION_CAPABILITY_ID
        - FEM_BODY_FORCE_ACTUATION_CAPABILITY_ID
        - FEM_REGION_DISPLACEMENT_SENSOR_CAPABILITY_ID
        - FEM_REGION_FORCE_SENSOR_CAPABILITY_ID
        - FEM_EXACT_STATE_CODEC_CAPABILITY_ID
        - FEM_EXACT_CONTROL_CODEC_CAPABILITY_ID
        - FEM_ATOMIC_REPLAY_CAPABILITY_ID
        - FEM_REMESH_CAPABILITY_ID
        - FEM_FRACTURE_CAPABILITY_ID
        - FEM_CONTACT_CAPABILITY_ID
        - MPMSoftState
        - MPMSoftCommand
        - MPMSoftParameters
        - MPMSoftFeatureManifest
        - MPMSoftResolutionRequirement
        - MPMSoftResolutionEvidence
        - MPMSoftResetEvidence
        - MPMSoftStepEvidence
        - MPMSoftObservationRequest
        - MPMParticleRegionObservation
        - MPMGridSurfaceObservation
        - MPMSoftObservation
        - MPMSoftPlant
      show_root_heading: false
      members_order: source

## MJX complete-state plant

::: phydrax.applications.robotics
    options:
      members:
        - mjx_availability
        - MJX_JAX_BACKEND_CAPABILITIES
        - MJX_JAX_PROFILE
        - MJX_WARP_PROFILE
        - MJXPreparedModelManifest
        - MJXState
        - MJXObservationRequest
        - MJXObservation
        - MJXRefreshResult
        - MJXAdapter
        - prepare_mjx_adapter
      show_root_heading: false
      members_order: source
