"""Differentiable cosmological geometry, products, simulation, and observations."""
# ruff: noqa: F401

from ._background import FLRWBackground
from ._closure import (
    CoordinateLayout,
    CorrelatedGaussianPlan,
    CorrelatedGaussianResult,
    CosmologyPhysicalState,
    CosmologyRealizationSignature,
    DifferentiationContract,
    LinearObservationPlan,
    PhysicalDependencyProjection,
    PrecisionCovarianceAction,
    ScientificArtifactEnvelope,
    TheoryVector,
)
from ._cmb import (
    CMB_FIELDS,
    CMB_MODES,
    CmbBandpowerResponsePlan,
    CmbBandpowerResponseResult,
    CmbSpectrumTable,
    CmbSpectrumTransformPlan,
    PrimordialPowerLaw,
)
from ._cmb_instrument import (
    CmbBandpowerHandoff,
    CmbBeamProduct,
    CmbIngressEvidence,
    CmbIngressPlan,
    CmbMapmakingEvidence,
    CmbMapmakingPlan,
    CmbMapmakingResult,
    CmbPointingProduct,
    CmbSkyMapProduct,
    CmbTodProduct,
    CmbTodSimulationPlan,
    HarmonicSkySynthesisPlan,
)
from ._corrections import (
    CorrectionModelCard,
    MatterPowerCorrectionEvidence,
    MatterPowerCorrectionResult,
    MultiplicativeMatterPowerCorrectionPlan,
)
from ._cosmological_amr import (
    AMREpochResult,
    AMRParticleDepositResult,
    AMRParticleGatherResult,
    AMRParticleLevelAssignment,
    BlockAMREpochPlan,
    BlockAMRGravityPlan,
    BlockAMRGravityResult,
    BlockAMRParticleGravityResult,
    BlockAMRParticleRoutingPlan,
)
from ._coupled import (
    ComovingEulerDiagnostics,
    ComovingEulerPlan,
    ComovingEulerState,
    CosmologicalGasParticleDiagnostics,
    CosmologicalGasParticleGravityPlan,
    CosmologicalGasParticleResult,
    CosmologicalGasParticleState,
    SharedGasParticleGravityResult,
)
from ._curvature_validity import (
    LocalCurvatureValidityPlan,
    LocalCurvatureValidityResult,
)
from ._dark_matter_inference import (
    ConstantExternalDarkMatterProduct,
    DarkMatterCoordinateContract,
    DarkMatterDiscrepancyPlan,
    DarkMatterDiscrepancyProduct,
    DarkMatterEmulatorCalibrationPlan,
    DarkMatterEmulatorCalibrationProduct,
    DarkMatterErrorBudget,
    DarkMatterInferenceEvaluation,
    ExternalDarkMatterEmulatorProduct,
    FixedTapeStochasticEvaluation,
    FixedTapeStochasticSensitivityPlan,
    SmoothDarkMatterKind,
    SmoothFixedGridDarkMatterInferencePlan,
    SmoothFixedGridSensitivityProduct,
    StochasticSensitivityEvidence,
)
from ._dark_matter_observables import (
    apply_dark_matter_observation,
    component_force_work_ledger_from_mixed_diagnostics,
    ComponentForceWorkLedger,
    ComponentForceWorkLedgerPlan,
    DarkMatterDensityConvention,
    DarkMatterHaloCompositionProduct,
    DarkMatterLensingCompositionProduct,
    DarkMatterPeriodicRadialShellProduct,
    DarkMatterSpatialContract,
    DarkMatterSpectrumProduct,
    DarkMatterSurfaceDensityProduct,
    find_dark_matter_halos,
    gravothermal_sidm_observables,
    GravothermalSIDMObservableProduct,
    MixedComponentSpectrumPlan,
    MixedComponentSpectrumProduct,
    ParticleAngularMomentProduct,
    ParticleDarkMatterObservablePlan,
    ParticleDarkMatterObservableProduct,
    ParticleGravothermalProduct,
    project_dark_matter_lensing_plane,
    RadialCoreProfileProduct,
    select_dark_matter_periodic_radial_shells,
    sidm_angular_moments,
    sidm_collision_observables,
    SIDMAngularMomentProduct,
    SIDMCollisionObservableProduct,
    VortexCirculationProduct,
    WaveDarkMatterObservablePlan,
    WaveDarkMatterObservableProduct,
    weighted_particle_statistics,
    weighted_sidm_collision_observables,
    weighted_sidm_packet_observables,
    WeightedParticleStatistics,
    WeightedSIDMCollisionObservableProduct,
    WeightedSIDMPacketObservableProduct,
)
from ._dark_matter_qualification import (
    bind_dark_matter_promotion,
    dark_matter_claim_criteria,
    dark_matter_claim_metric_ids,
    DarkMatterClaimName,
    DarkMatterQualificationLevel,
    differential_sidm_claim_profile,
    frequent_sidm_claim_profile,
    gravothermal_sidm_claim_profile,
    inelastic_sidm_claim_profile,
    mixed_wave_particle_claim_profile,
    mixed_wave_particle_gas_claim_profile,
    periodic_wave_amr_claim_profile,
    periodic_wave_claim_profile,
    PromotedDarkMatterClaim,
    rare_equal_sidm_claim_profile,
    SUPPORTED_DARK_MATTER_CLAIM_PROFILES,
    weighted_sidm_claim_profile,
)
from ._dark_radiation import (
    DarkRadiationExportEvidence,
    DarkRadiationExportResult,
    DarkRadiationLedger,
    DarkRadiationLedgerPlan,
    DarkRadiationPacket,
    DarkRadiationStatus,
)
from ._dark_sector_species import DarkSectorSpeciesPlan
from ._distances import FLRWDistancePlan, FLRWDistanceResult
from ._distributed_mixed import (
    DistributedMixedCheckpointEvidence,
    DistributedMixedCollectiveEvidence,
    DistributedMixedDensity,
    DistributedMixedEvolutionResult,
    DistributedMixedExecutionPlan,
    DistributedMixedGravityResult,
    DistributedMixedPlacementEvidence,
    DistributedMixedPreparationEvidence,
    DistributedMixedPreparationResult,
    DistributedMixedPreparationStatus,
    DistributedMixedState,
    DistributedMixedStepResult,
    PreparedDistributedMixedExecution,
)
from ._early_universe import (
    BbnReactionNetworkPlan,
    BbnResult,
    RecombinationPlan,
    RelicBackgroundPlan,
    RelicBackgroundResult,
)
from ._energy_deposition import (
    CascadeKernelEvidence,
    CascadeKernelProduct,
    DepositionSourceKind,
    EnergyDepositionEvidence,
    EnergyDepositionLedger,
    EnergyDepositionStatus,
    ExternalEnergyDepositionProviderResult,
    InjectionSpectrum,
    project_to_thermodynamics_history,
    ProviderExecution,
    SpeciesResolvedThermodynamicsHistory,
    ThermodynamicsHistoryEvidence,
)
from ._feedback import (
    CosmologicalPopulationPlan,
    CosmologicalPopulationState,
    FeedbackEventLedger,
    POPULATION_BLACK_HOLE,
    POPULATION_DARK_MATTER,
    POPULATION_INACTIVE,
    POPULATION_STAR,
    StarFormationResult,
    StochasticStarFormationPlan,
    StochasticThermalFeedbackPlan,
    ThermalFeedbackResult,
)
from ._force_resolution import PeriodicForceQualificationResult, PeriodicImageForcePlan
from ._force_scalability import (
    CosmologySnapshotProduct,
    DistributedPMFeasibilityEvidence,
    MeshMatchedNearFieldGate,
    PeriodicEwaldEvidence,
    PeriodicEwaldForcePlan,
    PeriodicEwaldResult,
)
from ._growth import FLRWGrowthPlan
from ._halo_finder import (
    DensityPeakSubstructurePlan,
    DirectHaloUnbindingPlan,
    FoFFinderEvidence,
    FoFFinderResult,
    FoFRealization,
    HaloPropertyPlan,
    HaloPropertyResult,
    HaloUnbindingResult,
    MergerMatchResult,
    ParticleCoreOverlapTreePlan,
    PeriodicFoFFinderPlan,
    SubstructureCandidateResult,
)
from ._halo_lineage import (
    HaloLifecycleState,
    HaloLineageEventKind,
    HaloLineageEventLedger,
    HaloLineageProduct,
    HaloTracerEvidence,
    HaloTrackSnapshot,
    ParticleCoreLineagePlan,
)
from ._halo_models import (
    HaloCatalog,
    HaloTripletResult,
    MatterHaloModel200mPlan,
    MatterHaloModelResult,
    SmoothComponentSphericalCollapsePlan,
    SmoothSphericalCollapseResult,
    TinkerDuffy200mPlan,
    Zheng07OccupationExpectation200m,
    Zheng07OccupationResult,
)
from ._halos import (
    LinearVariancePlan,
    NFWProfile,
    SphericalCollapseEdS,
    SphericalOverdensityMassDefinition,
)
from ._initial_conditions import (
    LagrangianDealiasing,
    LagrangianInitialConditionResult,
    LagrangianPerturbationInitialConditionPlan,
)
from ._inverse_realization import (
    ParticleFieldRealizationEvaluation,
    ParticleFieldRealizationPlan,
    ParticleTargetKind,
)
from ._linear_theory import (
    CosmologyModelRequest,
    CosmologyModelResult,
    MassiveNeutrinoSpecies,
    SubprocessCosmologyModelBackend,
)
from ._matter_power_emulator import (
    EmulatorSupportEvidence,
    ExternalMatterPowerResult,
    MatterPowerEvaluationRequest,
    MatterPowerProcessEvidence,
    MatterPowerProviderError,
    SubprocessMatterPowerBackend,
)
from ._microphysics import (
    PRIMORDIAL_PROCESSES,
    PRIMORDIAL_SPECIES,
    PrimordialMicrophysicsLedger,
    PrimordialMicrophysicsPlan,
    PrimordialMicrophysicsResult,
    PrimordialRateTable,
    PrimordialSpeciesState,
)
from ._mixed_initial_conditions import (
    ComponentModeRealization,
    ComponentTransferMatrixProduct,
    ImportedComplexFieldEvidence,
    ImportedComplexFieldValidationPlan,
    ImportedComplexFieldValidationResult,
    LocalizedWaveSeedEvidence,
    MixedInitialConditionPlan,
    ParticleInitialConditionProjection,
    PrimordialModeRealization,
    SolitonSeedPlan,
    SolitonSeedResult,
    VortexSeedEvidence,
    VortexSeedPlan,
    VortexSeedResult,
    WavePhaseSeedEvidence,
    WavePhaseSeedPlan,
    WavePhaseSeedResult,
)
from ._mixed_matter import (
    MixedCosmologyDiagnostics,
    MixedDensityAssembler,
    MixedDensityAssembly,
    PreparedWaveParticleCosmology,
    PreparedWaveParticleGasCosmology,
    SharedPeriodicGravityPlan,
    SharedPeriodicGravityResult,
    WaveParticleCosmologyPlan,
    WaveParticleCosmologyResult,
    WaveParticleCosmologyState,
    WaveParticleGasCosmologyPlan,
    WaveParticleGasCosmologyResult,
    WaveParticleGasCosmologyState,
)
from ._native_boltzmann import (
    ApproximationTransitionPolicy,
    FlatRadialKernelPlan,
    LineOfSightSpectraPlan,
    LineOfSightSpectraResult,
    NativeThermodynamicsPlan,
    NativeThermodynamicsResult,
    PreparedScalarEinsteinBoltzmann,
    RestrictedScalarTransferPlan,
    ScalarEinsteinBoltzmannEvidence,
    ScalarEinsteinBoltzmannPlan,
    ScalarEinsteinBoltzmannResult,
    ScalarEvolutionOperatorTable,
    ScalarHierarchyLayout,
    ScalarTransferResult,
    ThermodynamicsRateTable,
)
from ._nonlinear_closure import (
    BaryonicFeedbackPlan,
    CmbLensingPlan,
    HaloMassFunctionPlan,
    HaloModelPlan,
    HaloModelResult,
    LensingPlanePlan,
    LightConePlan,
    LightConeResult,
)
from ._observables import (
    LensingConvergenceTracer,
    LimberAngularPowerPlan,
    LinearDensityTracer,
    LinearRSDMultipolePlan,
    ObservablePrediction,
    RadialGrid,
    RedshiftDistribution,
    RSDMultipoleResult,
)
from ._parity import ParityEvidence, ParityProfile
from ._particle_mesh import (
    CosmologicalParticleMeshDiagnostics,
    CosmologicalParticleMeshPlan,
    CosmologicalParticleMeshResult,
)
from ._particles import (
    CosmologicalKDKPlan,
    CosmologicalParticleDiagnostics,
    CosmologicalParticleState,
)
from ._precision_backends import (
    BackendBuildManifest,
    CambLinearTheoryBackend,
    ClassLinearTheoryBackend,
    compare_precision_backends,
    LinearTheoryOutputPolicy,
    LinearTheoryPhysicsPolicy,
    LinearTheoryResourcePolicy,
    PrecisionBackendOverlapEvidence,
    PrecisionLinearTheoryResult,
)
from ._production_profiles import (
    AbstractScheduledCosmologyProductionMethod,
    PeriodicWaveProductionMethod,
    RareSIDMProductionMethod,
    RareSIDMProductionState,
)
from ._products import (
    combine_differentiation,
    cosmology_product_content_id,
    CosmologyProductProvenance,
    CosmologyProductSource,
    ExpansionHistory,
    LagrangianGrowthHistory,
    LinearTransferDescriptor,
    LinearTransferTable,
    MatterField,
    MatterPowerDescriptor,
    MatterPowerStage,
    MatterPowerTable,
    reconstruct_total_matter_power,
    ShotNoiseConvention,
    ThermodynamicsHistory,
    TransferGauge,
)
from ._s3_dynamics import (
    S3GeodesicKDKPlan,
    S3HarmonicBasisPlan,
    S3KDKResult,
    S3ManifoldPlan,
    S3ParticleMeshPlan,
    S3ParticleMeshResult,
    S3ParticleState,
    S3PoissonPlan,
    S3PoissonResult,
)
from ._scales import CODE_COSMOLOGY_SCALE, CosmologyScaleContract
from ._sidm import (
    CosmologicalSIDMDiagnostics,
    CosmologicalSIDMPlan,
    CosmologicalSIDMResult,
    SIDMCollisionDiagnostics,
    SIDMCollisionPolicy,
    SIDMCollisionResult,
    SIDMCrossSectionPlan,
)
from ._sidm_frequent import (
    FrequentSmallAngleSIDMDiagnostics,
    FrequentSmallAngleSIDMPlan,
    FrequentSmallAngleSIDMResult,
)
from ._sidm_gravothermal import (
    gravothermal_calibration_payload,
    GravothermalSIDMDiagnostics,
    GravothermalSIDMPlan,
    GravothermalSIDMResult,
    GravothermalSIDMState,
)
from ._sidm_kernels import (
    angles_from_direction,
    AngularSample,
    DifferentialKernelEvaluation,
    directions_from_angles,
    IdenticalParticleConvention,
    ScreeningConvention,
    SmallAngleSplitEvaluation,
    SmallAngleSplitPlan,
    TwoBodyDifferentialKernelPlan,
    TwoBodyKernelMoments,
)
from ._sidm_reactions import (
    DarkRadiationStateResult,
    DarkRadiationTransactionEvidence,
    DarkReactionEvidence,
    DarkReactionPartialRates,
    DarkReactionResult,
    DarkReactionStatus,
    DarkTwoBodyReactionPlan,
    InelasticSIDMPlan,
    InelasticSIDMState,
    read_inelastic_sidm_checkpoint,
    write_inelastic_sidm_checkpoint,
)
from ._sidm_weighted import (
    WeightedPacketLedger,
    WeightedPacketResamplingDiagnostics,
    WeightedPacketResamplingPlan,
    WeightedPacketResamplingResult,
    WeightedPMForceResult,
    WeightedPMIntervalDiagnostics,
    WeightedPMIntervalResult,
    WeightedSIDMCollisionDiagnostics,
    WeightedSIDMCollisionResult,
    WeightedSIDMPacketState,
    WeightedSIDMPlan,
    WeightedSIDMRolloutDiagnostics,
    WeightedSIDMRolloutResult,
)
from ._simulation_products import (
    CommonGravitySimulationSnapshot,
    CommonGravitySnapshotEvidence,
    CosmologyOutputBundle,
    DarkMatterCheckpointContract,
    DarkMatterCheckpointPayload,
    DarkMatterCheckpointRecoveryEvidence,
    DarkMatterRestartSnapshot,
    GasSimulationSnapshot,
    GasSnapshotEvidence,
    ParticleSimulationSnapshot,
    ParticleSnapshotEvidence,
    SimulationProductStatusEvidence,
    WaveSimulationSnapshot,
    WaveSnapshotEvidence,
)
from ._spectral_statistics import (
    CosmologicalFieldSpectrumPlan,
    FieldDensityConvention,
    MatterPowerEstimate,
    SpectralFieldDiscrepancyPlan,
    SpectralFieldDiscrepancyResult,
    stack_matter_power_estimates,
)
from ._spt import OneLoopEdSSPTPlan, OneLoopSPTEvidence, OneLoopSPTResult
from ._survey_framework import (
    desi_full_shape_slice,
    joint_survey_slice,
    spin2_pseudocl_slice,
    SurveyCoordinate,
    SurveyFrameworkPlan,
    SurveyTheoryProduct,
    SurveyVerticalSliceManifest,
)
from ._survey_likelihood import (
    DesiFullShapeLikelihoodPlan,
    SurveyReleaseManifest,
    SurveyReleaseProduct,
)
from ._tree_gravity import (
    BarnesHutGravityPlan,
    CartesianExpansionSpace,
    CartesianFMMOperators,
    CartesianFMMResourceEvidence,
    DistributedParticleLayout,
    MeshComplementCalibrationEvidence,
    MeshComplementCalibrationPlan,
    ParticleOctreePlan3D,
    PeriodicBarnesHutPlan,
    PreparedParticleOctree3D,
    TreeGravityEvidence,
    TreeGravityResult,
    TreePMPlan,
    TreePMResult,
    TreePMShortRangeKernel,
    TreePMSplitPolicy,
    UniformFMMPlan,
)
from ._wave_amr import (
    PreparedWaveAMR,
    WaveAMRAdaptivityPlan,
    WaveAMRDiagnostics,
    WaveAMRDiscretizationPlan,
    WaveAMRDistributedPreparation,
    WaveAMRPhysicsPlan,
    WaveAMRResult,
    WaveAMRState,
    WaveAMRTopologyIndicators,
    WaveAMRTopologyProposal,
    WaveAMRTransferEvidence,
    WaveAMRTransitionResult,
)
from ._wave_boundaries import (
    AbsorbingWaveBoundaryPolicy,
    AbsorbingWaveResult,
    IsolatedPotentialGauge,
    IsolatedWaveBoundaryDescriptor,
    PeriodicWaveBoundaryDescriptor,
    WaveBoundaryDescriptor,
)
from ._wave_dark_matter import (
    PreparedPeriodicWaveDarkMatter,
    WaveDarkMatterDiagnostics,
    WaveDarkMatterDifferentiability,
    WaveDarkMatterPlan,
    WaveDarkMatterPoissonResult,
    WaveDarkMatterResult,
    WaveDarkMatterState,
    WaveDarkMatterStepPolicy,
)
from ._wave_finite_difference import (
    PeriodicWaveFiniteDifferencePlan,
    PreparedPeriodicWaveFiniteDifference,
    WaveContactActionResult,
    WaveContactSelfInteractionPlan,
    WaveFiniteDifferenceDiagnostics,
    WaveFiniteDifferencePolicy,
    WaveFiniteDifferenceResult,
    WaveFiniteDifferenceState,
)


__all__ = [name for name in globals() if not name.startswith("_")]
