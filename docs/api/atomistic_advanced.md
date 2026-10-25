# Advanced atomistic methods

## Committee uncertainty

::: phydrax.atomistic.CommitteeAtomisticPotential

::: phydrax.atomistic.CommitteeReductionPolicy

::: phydrax.atomistic.ConservativeUncertaintyBlend
::: phydrax.atomistic.CommitteeAcquisitionScorePolicy


::: phydrax.atomistic.AcquisitionPlan

Provider labels enter `AtomisticLabelSet` with an immutable `AtomisticLabelSplit`
membership (`"train"` or `"validation"`). `AtomisticLabelSet.source` exposes them as a
canonical `AtomisticStructureSource`, and `training_problem` lowers them to the streamed
`AtomisticTrainingProblem`; see the
[adaptive-learning guide](../guides_atomistic_active_learning.md).

::: phydrax.atomistic.AtomisticLabelSet

::: phydrax.atomistic.AtomisticLearningCampaignPlan

::: phydrax.atomistic.AtomisticLearningCampaignState
::: phydrax.atomistic.AtomisticCampaignLifecycle


::: phydrax.atomistic.run_atomistic_campaign_round


::: phydrax.atomistic.SegmentFallbackPolicy

::: phydrax.atomistic.SegmentFallbackDecision


## Ensembles and rigid dynamics

Executable thermostats and cell methods (`NoseHooverChainPlan`,
`StochasticVelocityRescalingPlan`, `VelocityRescalingPlan`,
`GeneralizedLangevinPlan`, and the MTK methods) are documented with
[atomistic dynamics](atomistic.md#dynamics-constraints-and-thermodynamics).
`AtomisticSplittingPlan` describes operator schedules over `SplittingOperatorKind`
operators, including `CELL_THERMOSTAT`.

::: phydrax.atomistic.AtomisticSplittingPlan

::: phydrax.atomistic.SplittingOperatorKind

::: phydrax.atomistic.AnisotropicPressurePlan

::: phydrax.atomistic.RigidAtomisticCoordinateMap

::: phydrax.atomistic.BrownianDynamicsPlan

::: phydrax.atomistic.RotationalAtomisticState


## Polarization and solvent

::: phydrax.atomistic.PermanentMultipoleSiteData

::: phydrax.atomistic.PolarizationPlan

::: phydrax.atomistic.FiniteMultipoleRoute

::: phydrax.atomistic.EwaldMultipoleRoute

::: phydrax.atomistic.ParticleMeshMultipoleRoute

::: phydrax.atomistic.ImplicitSolventPlan

`NeighborhoodPairs` is the scalable multipole real-space pair sum over a bound
image-pair-sum neighborhood view; its own stale, capacity, and scientific
evidence gates the result. `PolarizationPredictorPlan` is the two-frame
induced-dipole warm start of the PCG solver.

::: phydrax.atomistic.NeighborhoodPairs

::: phydrax.atomistic.PolarizationPredictorPlan

## Quantum nuclei

::: phydrax.atomistic.RingPolymerNormalModePlan

::: phydrax.atomistic.StagingCoordinatePlan

::: phydrax.atomistic.ThermostattedRPMDPlan

::: phydrax.atomistic.ConstantPressureRingPolymerPlan

PIGLET ring-polymer normal modes keep the Euler-Maruyama kernel of continuous
drift and diffusion matrices, `EulerMaruyamaGeneralizedLangevinPlan`. Its
discretization is not FDT-exact and is never presented as a canonical sampler;
atomistic dynamics executes GLE only through the discrete FDT-checked
`GeneralizedLangevinPlan`.

::: phydrax.atomistic.EulerMaruyamaGeneralizedLangevinPlan

::: phydrax.atomistic.PIGLETPlan

## Soft matter and many-body potentials

Cutoff many-body terms declare classical pair and directed image-graph
requirements served by the Hamiltonian's neighborhood bundle and execute through
an explicit particle `AtomisticGraphExecutionPlan`. Directed neighbor slot routes
are prepared once on `AtomisticGraph` and reused by EAM, Stillinger--Weber, and
Tersoff evaluations. `maximum_neighbors` is therefore a correctness resource:
overflow invalidates the evaluation instead of falling back to dense all-pairs
work.

::: phydrax.atomistic.ScalarWallPotential

::: phydrax.atomistic.ManifoldConstraintPlan

::: phydrax.atomistic.ManifoldProjection


::: phydrax.atomistic.DissipativeParticleDynamicsPlan

::: phydrax.atomistic.ActiveForcePlan

::: phydrax.atomistic.ActiveForceEvaluation


::: phydrax.atomistic.EAMPotential

::: phydrax.atomistic.StillingerWeberPotential

::: phydrax.atomistic.TersoffPotential

## Distributed execution

`AtomisticDomainPlan` with one `AtomisticDomainRolePlan` per neighborhood role is the
canonical spatial-ownership contract. `prepare_atomistic_domain` returns a
`PreparedAtomisticDomainRuntime` whose `ingress_global`/`ingress_process_local`,
`evaluate`, `rebuild`, `rebind_model`, and `loss_gradient` own halo or
graph-partition execution, certificates, atomic migration, and reverse cotangents
returned exactly once. Reciprocal-space work is computed by
`AtomisticReciprocalDistributionPlan` owners on slab or pencil FFT layouts, never
supplied as certified evidence. See the
[distributed guide](../guides_atomistic_distributed_execution.md), including its
old-to-new API map. Lane-reference owners provide a numerical oracle, not hardware
qualification, and no multi-host or multi-GPU tuple is qualified by these pages.

### Domain plan, state, and runtime

::: phydrax.atomistic.AtomisticDomainPlan

::: phydrax.atomistic.AtomisticDomainReductionPolicy

::: phydrax.atomistic.AtomisticDomainRolePlan

::: phydrax.atomistic.AtomisticDomainConstraintState

::: phydrax.atomistic.AtomisticDomainForceCache

::: phydrax.atomistic.AtomisticDomainState

::: phydrax.atomistic.AtomisticDomainStrategy

::: phydrax.atomistic.AtomisticDomainTopology

::: phydrax.atomistic.AtomisticDomainTopologyEvidence

::: phydrax.atomistic.owner_directory_lookup

::: phydrax.atomistic.prepare_domain_relations

::: phydrax.atomistic.AtomisticDomainCertificate

::: phydrax.atomistic.AtomisticDomainEvaluation

::: phydrax.atomistic.AtomisticDomainExceptionInputs

::: phydrax.atomistic.AtomisticDomainExceptionRows

::: phydrax.atomistic.AtomisticDomainExecutionStatus

::: phydrax.atomistic.AtomisticDomainForceRecord

::: phydrax.atomistic.AtomisticDomainLossGradient

::: phydrax.atomistic.AtomisticDomainOwnedRows

::: phydrax.atomistic.AtomisticDomainOwnerPredictions

::: phydrax.atomistic.AtomisticDomainOwnerView

::: phydrax.atomistic.AtomisticDomainPacket

::: phydrax.atomistic.AtomisticDomainReplicatedInputs

::: phydrax.atomistic.AtomisticDomainTermInputs

::: phydrax.atomistic.AtomisticDomainTermRows

::: phydrax.atomistic.AtomisticDomainTermView

::: phydrax.atomistic.AtomisticDomainTransition

::: phydrax.atomistic.owner_parameter_partials

::: phydrax.atomistic.owner_predictions

::: phydrax.atomistic.prepare_atomistic_domain

::: phydrax.atomistic.PreparedAtomisticDomainRuntime

### Domain terms

::: phydrax.atomistic.AbstractAtomisticDomainTerm

::: phydrax.atomistic.AtomisticDomainEwaldTerm

::: phydrax.atomistic.AtomisticDomainExceptionTable

::: phydrax.atomistic.AtomisticDomainExceptionTopology

::: phydrax.atomistic.AtomisticDomainLennardJonesTerm

::: phydrax.atomistic.AtomisticDomainLennardJonesPMETerm

::: phydrax.atomistic.AtomisticDomainFourierD3Term

::: phydrax.atomistic.AtomisticDomainPolarizationTerm

::: phydrax.atomistic.domain_exception_topology

::: phydrax.atomistic.owner_ewald_exception_share

::: phydrax.atomistic.owner_ewald_real_share

::: phydrax.atomistic.owner_lennard_jones_share

::: phydrax.atomistic.owner_lennard_jones_pme_real_share

::: phydrax.atomistic.owner_lennard_jones_pme_reciprocal_share

::: phydrax.atomistic.owner_fourier_d3_share

::: phydrax.atomistic.owner_polarization_share

::: phydrax.atomistic.owner_reciprocal_share

::: phydrax.atomistic.OwnerExceptionRoute

::: phydrax.atomistic.OwnerTermResult

::: phydrax.atomistic.OwnerTermRoute

::: phydrax.atomistic.PreparedAtomisticDomainTerm

::: phydrax.atomistic.region_halo

::: phydrax.atomistic.replicated_halo

::: phydrax.atomistic.require_domain_requirement

### Layered owner execution and graph partition

::: phydrax.atomistic.AtomisticColumnExchange

::: phydrax.atomistic.AtomisticExchangeCapacityError

::: phydrax.atomistic.AtomisticLayeredModel

::: phydrax.atomistic.AtomisticLayerUpdate

::: phydrax.atomistic.AtomisticSystemStage

::: phydrax.atomistic.HaloColumnExchange

::: phydrax.atomistic.owner_atom_energies

::: phydrax.atomistic.owner_edge_geometry

::: phydrax.atomistic.owner_layer_gradients

::: phydrax.atomistic.owner_system_sum

::: phydrax.atomistic.OwnerConditioning

::: phydrax.atomistic.OwnerEdgeGeometry

::: phydrax.atomistic.OwnerFieldTerm

::: phydrax.atomistic.OwnerGlobalTerm

::: phydrax.atomistic.OwnerLayerForward

::: phydrax.atomistic.OwnerLayerGradients

::: phydrax.atomistic.OwnerLayerTopology

::: phydrax.atomistic.AtomisticGraphPartitionMemoryEvidence

::: phydrax.atomistic.AtomisticGraphPartitionPlan

::: phydrax.atomistic.AtomisticGraphPartitionResourceError

::: phydrax.atomistic.graph_partition_memory_evidence

::: phydrax.atomistic.prepare_graph_partition_topology

::: phydrax.atomistic.ReplicatedColumnExchange

### Distributed reciprocal space and polarization

Literal selectors: `AtomisticReciprocalStrategy` (`"distributed-fft"`,
`"replicated-mesh"`) and `AtomisticReciprocalSchedule` (`"slab"`, `"pencil"`; a
pencil plan also declares `pencil_grid=(px, py)`, and owner `r` holds the
physical block `(r % px, r // px)` and modal z-slab `r`).
`ReciprocalDistributionReport` carries the layout identities (`schedule`,
`spectral_grid`, `owner_coordinates`, `owner_devices`, physical and modal layout
ids and block shapes, `spectral_topology_id`, `collective_order`).

::: phydrax.atomistic.AtomisticReciprocalDistributionPlan

::: phydrax.atomistic.AtomisticReciprocalKind

::: phydrax.atomistic.owner_multipole_bilinear

::: phydrax.atomistic.owner_reciprocal_actions

::: phydrax.atomistic.owner_reciprocal_energy

::: phydrax.atomistic.OwnerMultipoleRelation

::: phydrax.atomistic.OwnerMultipoleRoutes

::: phydrax.atomistic.OwnerPolarizationPlan

::: phydrax.atomistic.OwnerPolarizationRelations

::: phydrax.atomistic.OwnerPolarizationResult

::: phydrax.atomistic.OwnerPolarizationSites

::: phydrax.atomistic.OwnerReciprocalActions

::: phydrax.atomistic.OwnerReciprocalTerm

::: phydrax.atomistic.prepare_owner_polarization

::: phydrax.atomistic.prepare_reciprocal_distribution

::: phydrax.atomistic.PreparedOwnerPolarization

::: phydrax.atomistic.PreparedReciprocalDistribution

::: phydrax.atomistic.ReciprocalDistributionCapacityError

::: phydrax.atomistic.ReciprocalDistributionEvidence

::: phydrax.atomistic.ReciprocalDistributionReport

::: phydrax.atomistic.ReciprocalSitePayload

::: phydrax.atomistic.solve_owner_polarization

### Distributed dynamics and constraints

::: phydrax.atomistic.AtomisticDomainBarostatEvaluation

::: phydrax.atomistic.AtomisticDomainConstraintEvidence

::: phydrax.atomistic.AtomisticDomainConstraintPlan

::: phydrax.atomistic.AtomisticDomainConstraintRows

::: phydrax.atomistic.AtomisticDomainConstraintSolve

::: phydrax.atomistic.AtomisticDomainConstraintTransition

::: phydrax.atomistic.AtomisticDomainDynamicsDiagnostics

::: phydrax.atomistic.AtomisticDomainDynamicsPlan

::: phydrax.atomistic.AtomisticDomainDynamicsReductions

::: phydrax.atomistic.AtomisticDomainEnsemblePlan

::: phydrax.atomistic.AtomisticDomainPrimeEvaluation

::: phydrax.atomistic.AtomisticDomainStepEvaluation

::: phydrax.atomistic.domain_constraint_rows

::: phydrax.atomistic.export_domain_constraints

::: phydrax.atomistic.import_domain_constraints

::: phydrax.atomistic.owner_count

::: phydrax.atomistic.owner_degrees_of_freedom

::: phydrax.atomistic.owner_inner_product

::: phydrax.atomistic.owner_kinetic_energy

::: phydrax.atomistic.owner_kinetic_tensor

::: phydrax.atomistic.owner_linear_momentum

::: phydrax.atomistic.owner_masked_sum

::: phydrax.atomistic.PreparedAtomisticDomainDynamics

::: phydrax.atomistic.rebuild_domain_constraints

::: phydrax.atomistic.solve_domain_constraints

### Durable domain checkpoints

::: phydrax.atomistic.AtomisticDomainCheckpoint

::: phydrax.atomistic.AtomisticDomainCheckpointError

::: phydrax.atomistic.AtomisticDomainCheckpointRecord

::: phydrax.atomistic.AtomisticDomainConstraintStatics

::: phydrax.atomistic.AtomisticDomainMigrationCapacityError

::: phydrax.atomistic.AtomisticDomainMigrationEvidence

::: phydrax.atomistic.AtomisticDomainPublication

::: phydrax.atomistic.AtomisticDomainRestore

::: phydrax.atomistic.checkpoint_domain_state

::: phydrax.atomistic.domain_execution_id

::: phydrax.atomistic.migrate_domain_checkpoint

::: phydrax.atomistic.publish_domain_checkpoint

::: phydrax.atomistic.restore_domain_checkpoint

::: phydrax.atomistic.restore_domain_state

### Data-parallel and data x domain training

::: phydrax.atomistic.atomistic_domain_training_objectives

::: phydrax.atomistic.AtomisticDataParallelReduction

::: phydrax.atomistic.AtomisticDomainKernelObjective

::: phydrax.atomistic.AtomisticDomainTrainingShard

::: phydrax.atomistic.AtomisticTrainingDistributionPlan

::: phydrax.atomistic.run_distributed_training_attempt

### Stage transport and stage x domain campaigns

`AtomisticStageTransportPlan` is the only cross-stage communication: one fixed
root-group schedule per round (control all-gather, fixed-slot payload all-to-all
with SHA-256 per fragment, receipt all-gather) with atomic group transfers,
credits, and FIFO parking. An `AtomisticTransportEdge` `mode` is `"replicated"`
(the source stage coordinator's packet to every destination member) or
`"process-local"` (one fragment per source member process).
`AtomisticDomainCampaignPlan` runs native distributed domain dynamics on each
stage's child group between rounds; `checkpoint` commits the complete
continuation, `restore` resumes it bitwise on the same topology, and `migrate`
continues it on another topology within tolerance. See
[stage x domain campaigns](../guides_atomistic_distributed_execution.md#stage-x-domain-campaigns).

::: phydrax.atomistic.AtomisticStageTransportPlan

::: phydrax.atomistic.AtomisticTransportStage

::: phydrax.atomistic.AtomisticTransportEdge

::: phydrax.atomistic.PreparedAtomisticStageTransport

::: phydrax.atomistic.AtomisticStageTransportState

::: phydrax.atomistic.AtomisticTransportRound

::: phydrax.atomistic.AtomisticTransportRoundEvidence

::: phydrax.atomistic.AtomisticTransportDelivery

::: phydrax.atomistic.AtomisticTransportSink

::: phydrax.atomistic.AtomisticPacketCodec

::: phydrax.atomistic.AtomisticTransportCapacityError

::: phydrax.atomistic.AtomisticTransportClosedError

::: phydrax.atomistic.AtomisticTransportDivergenceError

::: phydrax.atomistic.AtomisticTransportSchemaError

::: phydrax.atomistic.process_subset_group_spec

::: phydrax.atomistic.stage_transport_probe_rank

::: phydrax.atomistic.AtomisticDomainCampaignPlan

::: phydrax.atomistic.AtomisticDomainStage

::: phydrax.atomistic.AtomisticDomainStageProgram

::: phydrax.atomistic.AtomisticDomainCampaignTask

::: phydrax.atomistic.PreparedAtomisticDomainCampaign

::: phydrax.atomistic.AtomisticDomainCampaignState

::: phydrax.atomistic.AtomisticDomainResident

::: phydrax.atomistic.AtomisticDomainCampaignRound

::: phydrax.atomistic.AtomisticDomainStageReport

::: phydrax.atomistic.AtomisticDomainCampaignCheckpointPolicy

::: phydrax.atomistic.AtomisticDomainCampaignPublication

::: phydrax.atomistic.AtomisticDomainCampaignMigration

::: phydrax.atomistic.AtomisticDomainCampaignCheckpointError

## Nanoflow observables

::: phydrax.geometry.PlanarWallFramePlan

::: phydrax.atomistic.PlanarWallProfileObserverPlan

::: phydrax.atomistic.MultiOriginCorrelationObserverPlan

::: phydrax.atomistic.DrivenSlipFitPlan

::: phydrax.atomistic.WallForceCorrelationPlan

::: phydrax.atomistic.DiffusionTensorFitPlan

::: phydrax.atomistic.AtomisticNanoflowClosureArtifact

## Hydrodynamics and driven flow

::: phydrax.atomistic.driven_flow
    options:
      members: true
      show_root_heading: true
      show_source: false
