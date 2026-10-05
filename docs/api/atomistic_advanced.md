# Advanced atomistic methods

## Committee uncertainty

::: phydrax.atomistic.CommitteeAtomisticPotential

::: phydrax.atomistic.CommitteeReductionPolicy

::: phydrax.atomistic.ConservativeUncertaintyBlend
::: phydrax.atomistic.CommitteeAcquisitionScorePolicy


::: phydrax.atomistic.AcquisitionPlan
::: phydrax.atomistic.AtomisticLabelSet

::: phydrax.atomistic.AtomisticLearningCampaignPlan

::: phydrax.atomistic.AtomisticLearningCampaignState
::: phydrax.atomistic.AtomisticCampaignLifecycle


::: phydrax.atomistic.run_atomistic_campaign_round


::: phydrax.atomistic.SegmentFallbackPolicy

::: phydrax.atomistic.SegmentFallbackDecision


## Ensembles and rigid dynamics

::: phydrax.atomistic.AtomisticSplittingPlan

::: phydrax.atomistic.BussiThermostatPlan

::: phydrax.atomistic.NoseHooverChainPlan

::: phydrax.atomistic.GeneralizedLangevinPlan

::: phydrax.atomistic.AnisotropicPressurePlan

::: phydrax.atomistic.RigidAtomisticCoordinateMap

::: phydrax.atomistic.BrownianDynamicsPlan

::: phydrax.atomistic.RotationalAtomisticState


## Polarization and solvent

::: phydrax.atomistic.PermanentMultipoleSiteData

::: phydrax.atomistic.PolarizationPlan

::: phydrax.atomistic.MultipolePMEPlan

::: phydrax.atomistic.ImplicitSolventPlan

## Quantum nuclei

::: phydrax.atomistic.RingPolymerNormalModePlan

::: phydrax.atomistic.StagingCoordinatePlan

::: phydrax.atomistic.ThermostattedRPMDPlan

::: phydrax.atomistic.ConstantPressureRingPolymerPlan


::: phydrax.atomistic.PIGLETPlan

## Soft matter and many-body potentials

Cutoff many-body terms consume the prepared particle neighborhood through an
explicit particle `AtomisticGraphExecutionPlan`. Directed neighbor slot routes
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

`DistributedAtomisticPlan` is the slab runtime with transactional migration,
reductions, PME, and polarization contracts. `halo_short_range_evaluate`
evaluates a classical program globally and then masks the outputs by owner, and
is kept as a labeled reference. Owner-local learned execution
(`OwnerLocalAtomisticPlan`) evaluates each owner's receivers from source-halo
features layer by layer. It returns reverse cotangents, including the shared
cell contribution, exactly once. Migration commits the complete accepted state
as one transaction. See the
[distributed guide](../guides_atomistic_distributed_execution.md#owner-local-learned-execution).
Every distributed MACE support tuple is an unreleased candidate. Lane-reference
owners provide a numerical oracle, not hardware qualification.

::: phydrax.atomistic.DistributedAtomisticPlan

::: phydrax.atomistic.DistributedAtomisticState

::: phydrax.atomistic.halo_short_range_evaluate

### Owner-local learned execution

::: phydrax.atomistic.OwnerLocalAtomisticPlan

::: phydrax.atomistic.prepare_owner_local_atomistic

::: phydrax.atomistic.OwnerLocalAtomisticState

::: phydrax.atomistic.OwnerLocalAtomisticTopology

::: phydrax.atomistic.OwnerLocalTopologyEvidence

::: phydrax.atomistic.evaluate_owner_local_atomistic

::: phydrax.atomistic.OwnerLocalAtomisticEvaluation

::: phydrax.atomistic.OwnerLocalExecutionStatus

::: phydrax.atomistic.owner_local_loss_gradient

::: phydrax.atomistic.OwnerLocalLossGradient

::: phydrax.atomistic.rebuild_owner_local_atomistic

::: phydrax.atomistic.OwnerLocalTransition

::: phydrax.atomistic.rebind_owner_local_model

::: phydrax.atomistic.checkpoint_owner_local_atomistic

::: phydrax.atomistic.restore_owner_local_atomistic

::: phydrax.atomistic.OwnerLocalAtomisticCheckpoint

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
