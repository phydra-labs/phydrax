# Atomistic learning and dynamics

`phydrax.atomistic` owns finite and periodic atomistic learning, prediction, training,
conservative dynamics, and native model artifacts. `phydrax.nn.atomistic` owns the
PaiNN, low-degree Cartesian NequIP, and standard MACE model architectures together with
MACE's radial, symmetric-contraction, preparation, and accelerated-coupling owners.
Material particles, sparse and streamed relations, image neighborhoods, graph IR,
precision, replay, and qualification remain shared native substrates.

Classical force fields, interaction sites, trajectory interoperability, MACE
checkpoint conversion, ASE and i-PI deployment, enhanced sampling, uncertainty,
advanced ensembles, polarization, quantum nuclei, many-body methods, and domain-owned
distributed execution are documented on the split
[force-field](atomistic_force_fields.md), [interoperability](atomistic_interop.md),
[sampling](atomistic_sampling.md), [crystal prediction](atomistic_crystal.md), and
[advanced-method](atomistic_advanced.md) API pages.
Image-aware particle neighborhoods are on the
[particle API page](discretization/particle.md), and frozen atomistic IREE export is on
the [export API page](export.md). The [atomistic learning guide](../guides_atomistic.md)
[training-data guide](../guides_atomistic_training_data.md),
[distillation guide](../guides_atomistic_distillation.md), and
[native MACE execution guide](../guides_mace_execution.md) describe the contracts.
Every MACE route is an unreleased candidate.

## Structures, units, systems, and topology

`AtomicStructure` and `AtomisticBatch` accept `mass_accumulation_dtype`, defaulting
to float64 independently of coordinate storage. Select float32 explicitly for
an all-float32 Metal workflow. Batches also accept one accumulation dtype per case
so heterogeneous CPU structures retain their choices. This controls prepared
mass arithmetic, not the physical definition of the weights; copies, position
updates, and structure-to-batch conversion preserve it.

::: phydrax.atomistic.AtomisticScaleContract

::: phydrax.atomistic.AtomisticUnitSystem

::: phydrax.atomistic.AtomicStructure

::: phydrax.atomistic.AtomisticBatch

::: phydrax.atomistic.AtomisticSystemPlan

::: phydrax.atomistic.PreparedAtomisticSystem

::: phydrax.atomistic.MolecularTopologyPlan

::: phydrax.atomistic.PreparedMolecularTopology

## Graph layouts, payloads, and binding

`AtomisticGraphLayout` owns every static executable choice of a graph (scale,
execution plan, capacities, coordinate dtype, cutoff, skin, periodic axes, cells,
and route-search envelope). `layout.prepare(batch)` is the host boundary over
concrete positions and cells: it discovers directed routes, explicit integer image
shifts, stable receiver-major route IDs, and the streamed schedule once per
topology epoch, and returns an `AtomisticGraphPayload` whose material identities
are dynamic leaves. `payload.graph(...)` and `bind_atomistic_graph` only compute
`d_e = x[receiver] - x[sender] + n_e @ H[case_e]`, cutoff masks, and certificates,
and are safe under `jit` and differentiation. The image candidate radius is
`cutoff + skin` rounded outward by one value of the coordinate dtype
(`layout.search_radius`); the physical cutoff is unchanged.
`AtomisticGraphExecutionPlan.backend` is `"dense"` (the bounded named reference)
or `"particle"` (case-batched cell-list image search); `search_cell` and
`search_margin` declare the reusable search envelope. Capacity exhaustion is
`topology.overflow`; scientific or material search failure is
`topology.search_failure`.

::: phydrax.atomistic.AtomisticGraphExecutionPlan

::: phydrax.atomistic.AtomisticGraphLayout

::: phydrax.atomistic.AtomisticGraphPayload

::: phydrax.atomistic.AtomisticGraphTopology

::: phydrax.atomistic.bind_atomistic_graph

::: phydrax.atomistic.AtomisticGraph

::: phydrax.atomistic.particle_atomistic_graph_topology

::: phydrax.atomistic.realize_particle_atomistic_graph

## Neighborhood bundles

`AtomisticNeighborhoodBundlePlan` is the canonical runtime resource owner of
potential programs and dynamics. It binds explicit sources, search policies, and
capacity ladders to every declared `AtomisticNeighborhoodRole` requirement; there
is no implicit dense fallback. Refresh and selection are outside the
differentiated scalar, and capacity failure is distinct from scientific or
domain failure.

Image search policies are explicit resource choices: `AtomisticCellListImageSearch`
(fractional cell list), `AtomisticClusterTileImageSearch` (Morton cluster tiles
with an independently charged tile capacity), `AtomisticDenseImageSearch` (the
bounded dense reference), and `AtomisticTiledImageSearch(*, tiles,
deformation_margin=0.0)`. The tiled policy tests every receiver, source, and image
candidate of the dense reference by traversing tiles of an `ImageTilePolicy(*,
receiver_width, source_width, maximum_tile_candidates)`: one step materializes
`case * receiver_width * source_width * images` candidates, never the all-pairs
product, and that workspace is admitted against `maximum_tile_candidates` at
preparation. Width one on both axes is the scalar reference.

::: phydrax.atomistic.AbstractAtomisticImageSearchPolicy

::: phydrax.atomistic.AtomisticCellListImageSearch

::: phydrax.atomistic.AtomisticClusterTileImageSearch

::: phydrax.atomistic.AtomisticCompletePairSource

::: phydrax.atomistic.AtomisticCompletePairState

::: phydrax.atomistic.AtomisticDenseImageSearch

::: phydrax.atomistic.AtomisticNeighborhoodBundlePlan

::: phydrax.atomistic.AtomisticNeighborhoodBundleState

::: phydrax.atomistic.AtomisticNeighborhoodGeometry

::: phydrax.atomistic.AtomisticNeighborhoodRequirement

::: phydrax.atomistic.AtomisticNeighborhoodRole

::: phydrax.atomistic.AtomisticNeighborhoodSource

::: phydrax.atomistic.AtomisticNeighborhoodSourceProof

::: phydrax.atomistic.AtomisticNeighborhoodView

::: phydrax.atomistic.AtomisticPairCounting

::: phydrax.atomistic.AtomisticPairExclusions

::: phydrax.atomistic.AtomisticRoleCapacity

::: phydrax.atomistic.AtomisticRoleCapacityLadder

::: phydrax.atomistic.AtomisticRoleEvidence

::: phydrax.atomistic.AtomisticTiledImageSearch

::: phydrax.discretization.ImageTilePolicy

::: phydrax.atomistic.PreparedAtomisticCompletePairs

::: phydrax.atomistic.PreparedAtomisticNeighborhoodBundle

## Typed model boundary, conditions, and outputs

`model.evaluate_graph(AtomisticGraphInputs(graph, scale, species_kind=...,
conditioning=None, key=None), request)` is the numerical model boundary.
`AtomisticObservableRequest` names requested outputs and typed derivative
requests; `AtomisticModelOutputs` returns energies (absent for property-only
models), atomic contributions with their completeness, success, and typed
additional observables. `AtomisticConditioningSpec` binds named fields to
`ValuePort` identities, levels, kinds, dtypes, units, and domains;
`TOTAL_CHARGE_PORT` and `SPIN_MULTIPLICITY_PORT` are the standard physical
charge and spin-multiplicity ports. `AtomisticPotentialRequirements` declares a
model or term's neighborhoods, field dependencies, bonded geometry, and
reciprocal grid.

::: phydrax.atomistic.AtomisticGraphInputs

::: phydrax.atomistic.AtomisticConditionDType

::: phydrax.atomistic.AtomisticConditionField

::: phydrax.atomistic.AtomisticConditioning

::: phydrax.atomistic.AtomisticConditioningSpec

::: phydrax.atomistic.AtomisticConditionKind

::: phydrax.atomistic.AtomisticConditionLevel

::: phydrax.atomistic.AtomisticConditionStatus

::: phydrax.atomistic.AtomisticFieldReference

::: phydrax.atomistic.AtomisticModelOutputs

::: phydrax.atomistic.AtomisticObservable

::: phydrax.atomistic.AtomisticObservableRequest

::: phydrax.atomistic.AtomisticObservableValue

::: phydrax.atomistic.AbstractAtomisticModel

::: phydrax.atomistic.AbstractPreparedAtomisticModel

::: phydrax.atomistic.AtomisticPotentialRequirements

## Potentials, prediction, and stress

::: phydrax.atomistic.AbstractAtomisticPotential

::: phydrax.atomistic.AbstractPreparedAtomisticPotential

::: phydrax.atomistic.AtomisticPotentialCapabilities

::: phydrax.atomistic.atomistic_model_revision

::: phydrax.nn.atomistic.PaiNNPotential

::: phydrax.nn.atomistic.NequIPPotential

::: phydrax.atomistic.AtomisticPrediction

::: phydrax.atomistic.AtomisticProvenance

::: phydrax.atomistic.energy_and_forces

::: phydrax.atomistic.AtomisticEnergyDerivatives

::: phydrax.atomistic.atomistic_energy_derivatives

::: phydrax.atomistic.AtomisticStressConvention

## Coordinate Hessian actions

The coordinate Hessian of a conservative scalar potential is the matrix-free
operator `H v = -DF[v]` over one prepared force linearization
(`phydrax.linalg.prepare_linearization` on `F = -grad E`); forward, transpose,
and adjoint actions reuse that linearization. `bind_atomistic_hessian(potential,
payload, positions, *, request=None, policy=None, ...)` is the pure, traceable
numerical binding on a layout-bound graph payload, usable inside `jit` and
parameter differentiation. `prepare_atomistic_hessian(potential, batch,
execution, *, ...)` is the host boundary: it refuses graph capacity failure
(grow the resource and prepare again), records an `AtomisticHessianIdentity` of
every bound input (model definition and revision, derivative admission, batch,
graph, geometry, cell, conditioning, key, image counts, resource epochs), and
returns a `PreparedAtomisticHessian` whose `require_current` refuses reuse after
any of them changes.

Admission is the model's declared `DerivativeContract` under an explicit
`RegularityPolicy`; property-only and nonconservative models, models without a
contract, and strain, cell, or conditioning responses are refused.
`ATOMISTIC_HESSIAN_REQUEST` is the coordinate order-two request; a parameter
derivative of a Hessian action needs an admitted mixed order-three request.
`ATOMISTIC_NONCOINCIDENT_SUPPORT` and `atomistic_edge_distance_locus(radius)`
name the loci the binding certifies at the actual geometry; a violated locus
poisons only its case, and loci it cannot certify stay in
`AtomisticHessianOperator.unresolved_conditions`. Dense output is never the full
matrix: `AtomisticHessianOperator.blocks(selection, policy)` materializes
case-major `(case, output_atom, 3, input_atom, 3)` blocks for an explicit
`AtomisticHessianBlockSelection` under a `MaterializationPolicy`, and never forms
cross-case zeros.

::: phydrax.atomistic.AtomisticHessianOperator

::: phydrax.atomistic.AtomisticHessianBlockSelection

::: phydrax.atomistic.AtomisticHessianIdentity

::: phydrax.atomistic.PreparedAtomisticHessian

::: phydrax.atomistic.bind_atomistic_hessian

::: phydrax.atomistic.prepare_atomistic_hessian

::: phydrax.atomistic.atomistic_hessian_identity

::: phydrax.atomistic.atomistic_edge_distance_locus

## Native MACE

`MACEArchitecture.interactions` admits `"real-agnostic"`, `"real-agnostic-residual"`,
`"real-agnostic-density"`, and `"real-agnostic-density-residual"`. Other literal
selectors:

- `distance_transform`: `"none"` or `"agnesi"`;
- `cutoff_placement`: `"embedding"` or `"weights"`;
- `energy_scaling`: `"unscaled"` or `"scale-shift"`;
- `prepare_mace_potential(radial=...)`: `"exact"` or `"tabulated"`.

`MACEPotential` is trainable. `PreparedMACEPotential` is inference-only and bound to
one parameter revision. Acceleration is an execution policy that changes neither
`architecture_id` nor the numeric revision.

::: phydrax.nn.atomistic.MACEArchitecture

::: phydrax.nn.atomistic.MACEPotential

::: phydrax.nn.atomistic.MACELayerUpdate

::: phydrax.nn.atomistic.MACESourceRows

::: phydrax.nn.atomistic.mace_potential_from_source

::: phydrax.nn.atomistic.prepare_mace_potential

::: phydrax.nn.atomistic.PreparedMACEPotential

::: phydrax.nn.atomistic.StaleMACEPreparation

### Radial realization and tables

`RadialMLP` postprocessing is `"none"` or `"tanh-square"`. Table layouts are
`"projected-width"` or `"embedding-width"`, and table objectives are
`"minimum-table-bytes"` or `"minimum-edge-work"`. Tables admit first coordinate
derivatives only; Hessians and force/stress training use the exact radial network.

::: phydrax.nn.atomistic.BesselRadialBasis

::: phydrax.nn.atomistic.PolynomialCutoff

::: phydrax.nn.atomistic.AgnesiTransform

::: phydrax.nn.atomistic.RadialEmbedding

::: phydrax.nn.atomistic.RadialMLP

::: phydrax.nn.atomistic.normalized_silu_scale

::: phydrax.nn.atomistic.RadialTableDeclaration

::: phydrax.nn.atomistic.RadialSpeciesBinding

::: phydrax.nn.atomistic.prepare_radial_tables

::: phydrax.nn.atomistic.PreparedRadialTables

::: phydrax.nn.atomistic.radial_source_revision

::: phydrax.nn.atomistic.radial_table_resources

::: phydrax.nn.atomistic.RadialTableResources

::: phydrax.nn.atomistic.select_radial_table_layout

::: phydrax.nn.atomistic.RadialTableLayoutChoice

::: phydrax.nn.atomistic.RadialTableQualificationPolicy

::: phydrax.nn.atomistic.qualify_radial_tables

::: phydrax.nn.atomistic.RadialTableQualification

::: phydrax.nn.atomistic.RadialTableQualificationError

::: phydrax.nn.atomistic.StaleRadialTableBinding

### Symmetric contraction

A `SymmetricContractionBasis` origin is `"native-coupling"` or `"imported-source"`.
The trainable contraction keeps the original weights `W`, and merged coefficients are
fixed data bound to one `W` revision.

Basis construction, native generation, and source-tensor import accept `dtype`
(float64 by default); native plans propagate it to their bases. MACE selects its
compute dtype. Stored coefficients are dynamic arrays, and their actual dtype
and rounded values determine identity. Native certification remains host
float64. The requested `equivariance_tolerance`, measured
`equivariance_residual`, and derived `equivariance_roundoff_bound` remain separate;
restoration rechecks all three rather than trusting recorded labels.

::: phydrax.nn.atomistic.SymmetricContractionBasis

::: phydrax.nn.atomistic.SymmetricContractionPlan

::: phydrax.nn.atomistic.SymmetricContraction

::: phydrax.nn.atomistic.symmetric_contraction_revision

::: phydrax.nn.atomistic.MergedSymmetricContraction

::: phydrax.nn.atomistic.StaleSymmetricContractionBinding

### Accelerated edge coupling

Target admission (`"cuda"` with compute capability 8.0, 9.0, 10.0, 10.3, 12.0, or 12.1, or the
`"cpu_interpret"` Mosaic GPU interpreter used only for semantic reference
verification) is owned by the [backend API](backends.md). See the
[MACE execution guide](../guides_mace_execution.md) for architecture-specific
compiled correctness evidence and qualification limits.

::: phydrax.nn.atomistic.MACEKernelPlan

::: phydrax.nn.atomistic.MACEAcceleratedCoupling

::: phydrax.nn.atomistic.MACEEdgeCouplingSpec

::: phydrax.nn.atomistic.MACECouplingRow

::: phydrax.nn.atomistic.MACELayoutBlock

::: phydrax.nn.atomistic.MACEFragmentExtent

::: phydrax.nn.atomistic.fragment_extent

::: phydrax.nn.atomistic.require_fragment_routing

::: phydrax.nn.atomistic.MACEFragmentResources

::: phydrax.nn.atomistic.MACEKernelEvidence

::: phydrax.nn.atomistic.MACECouplingDerivativeError

## Potential programs and classical terms

::: phydrax.atomistic.AtomisticPotentialProgram

::: phydrax.atomistic.PreparedAtomisticPotentialProgram

::: phydrax.atomistic.LearnedGraphPotentialTerm

::: phydrax.atomistic.HarmonicBondPotential

::: phydrax.atomistic.HarmonicAnglePotential

::: phydrax.atomistic.PeriodicTorsionPotential

::: phydrax.atomistic.LennardJonesPotential

::: phydrax.atomistic.DirectCoulombPotential

::: phydrax.atomistic.EwaldReferencePotential

::: phydrax.atomistic.ParticleMeshEwaldPotential

## Dynamics, constraints, and thermodynamics

`AtomisticDynamicsPlan(system, potential, integrator, *, constraints=None,
strain_response=None, force_ports=())` takes every neighborhood resource from
`potential.neighborhood_bundle`; there is no second neighborhood argument.
`strain_response` requests `dE/d strain` with every force evaluation and is
mandatory for integrator-owned cell methods. Force ports
(`AbstractAtomisticForcePort`) are typed nonconservative contributions; a runtime
with ports has no complete scalar force definition and refuses integrator-owned
cell methods. The thermodynamic table admits `VelocityVerletPlan` for NVE;
`BAOABLangevinPlan`, `NoseHooverChainPlan`, `StochasticVelocityRescalingPlan`, and
`GeneralizedLangevinPlan` for NVT (and NPT with a scheduled Monte Carlo volume
move); the deterministic, noncanonical `VelocityRescalingPlan` for NVT only;
`MTKNPTPlan` for NPT; and `MTKNPHPlan` for NPH. MTK cell methods require atomic
volume scaling and refuse constraints. `MTKNPHPlan` has no thermostat of any
kind; `MTKBarostatPlan.inertia_temperature` only sets the cell inertia `W` and is
required there. Per-particle stochastic thermostats refuse a center-of-mass
momentum restriction.

`AtomisticForceState` caches `dE/d strain` (row-cell convention) and the
tension-positive Cauchy stress `sym(dE/d strain) / V`, both NaN unless strain is
available; the finite position moment is not a periodic virial and is not
cached. `PreparedDistanceConstraints` declares constant rank, certified at an
explicit geometry by `ConstraintRankEvidence`; dynamics refuses a trajectory that
reaches a rank-changing configuration. `CAPACITY_REJECTIONS` is the mask of
capacity rejection bits of `AtomisticStepRejectionReason`.

::: phydrax.atomistic.VelocityVerletPlan

::: phydrax.atomistic.BAOABLangevinPlan

::: phydrax.atomistic.NoseHooverChainParameters

::: phydrax.atomistic.NoseHooverChainPlan

::: phydrax.atomistic.StochasticVelocityRescalingPlan

::: phydrax.atomistic.VelocityRescalingPlan

::: phydrax.atomistic.GeneralizedLangevinPlan

::: phydrax.atomistic.AtomisticCellChart

::: phydrax.atomistic.MTKBarostatPlan

::: phydrax.atomistic.MTKNPTPlan

::: phydrax.atomistic.MTKNPHPlan

::: phydrax.atomistic.DistanceConstraintPlan

::: phydrax.atomistic.PreparedDistanceConstraints

::: phydrax.atomistic.ConstraintRankEvidence

::: phydrax.atomistic.AtomisticDynamicsPlan

::: phydrax.atomistic.PreparedAtomisticDynamics

::: phydrax.atomistic.AtomisticDynamicsState

::: phydrax.atomistic.AtomisticForceState

::: phydrax.atomistic.AtomisticControllerState

::: phydrax.atomistic.AtomisticEnergyLedgerState

::: phydrax.atomistic.AtomisticDynamicsDiagnostics

::: phydrax.atomistic.AtomisticStepEvaluation

::: phydrax.atomistic.AtomisticPrimeEvaluation

::: phydrax.atomistic.AtomisticInitialStateEvaluation

::: phydrax.atomistic.AtomisticDynamicsHamiltonianTransition

::: phydrax.atomistic.AtomisticStepRejectionReason

::: phydrax.atomistic.retry_atomistic_step_with_capacity

::: phydrax.atomistic.neighborhood_route_work

::: phydrax.atomistic.AbstractAtomisticForcePort

::: phydrax.atomistic.AtomisticTermStateBindings

::: phydrax.atomistic.ThermodynamicAccumulator

::: phydrax.atomistic.AtomisticPhaseSpaceMeasurePlan

::: phydrax.atomistic.AtomisticThermodynamicStatePlan

::: phydrax.atomistic.PreparedThermodynamicStateTable

::: phydrax.atomistic.RadialDistributionPlan

## Relaxation

`AtomisticRelaxationPlan` binds one prepared scalar Hamiltonian, one coordinate or
cell chart, one native force-driven optimizer (`FIRE`, `FIRE2`, or
`TrustScaledLBFGS`), and one convergence policy; `PreparedAtomisticRelaxation`
runs a batch of independent relaxations of the same program as optimizer groups.
Each round realizes pending proposals through the chart, refreshes candidate
resources from the accepted bundle state, and differentiates the objective
(energy, plus `P V` for a variable cell) in one reverse pass. A capacity-only
failure is not consumed: the group is `CAPACITY_PENDING` until the host `grow`
lifecycle replays the identical proposal at larger declared capacities.
Scientific, numerical, and cell failures terminate the group with an explicit
status. The projected residual (chart units) and the full residual (Cartesian
forces and `sigma + P I`) are reported separately, and
`AtomisticRelaxationConvergence.require_full` selects which terminates a group.
`AtomisticCartesianChart` moves unwrapped Cartesian coordinates,
`AtomisticDeformationChart` adds an affine or logarithmic symmetric strain
subspace, and `AtomisticSymmetryChart` moves an atomic or rigid-molecular crystal
chart by its retraction. Holonomic topology constraints are refused.

::: phydrax.atomistic.AtomisticRelaxationPlan

::: phydrax.atomistic.PreparedAtomisticRelaxation

::: phydrax.atomistic.AtomisticRelaxationState

::: phydrax.atomistic.AtomisticRelaxationStatus

::: phydrax.atomistic.AtomisticRelaxationConvergence

::: phydrax.atomistic.AtomisticRelaxationSensitivity

::: phydrax.atomistic.AtomisticSensitivityStatus

::: phydrax.atomistic.AbstractAtomisticRelaxationChart

::: phydrax.atomistic.AtomisticCartesianChart

::: phydrax.atomistic.AtomisticDeformationChart

::: phydrax.atomistic.AtomisticSymmetryChart

::: phydrax.atomistic.AtomisticSymmetryEvidence

::: phydrax.atomistic.AtomisticCellChartKind

::: phydrax.atomistic.AtomisticStrainSubspace

::: phydrax.atomistic.AtomisticCoordinateConstraint

::: phydrax.atomistic.AtomisticGaugePolicy

::: phydrax.atomistic.AtomisticStationaryPolicy

::: phydrax.atomistic.AtomisticStationaryEvidence

::: phydrax.atomistic.atomistic_stationary_inertia

::: phydrax.atomistic.AtomisticCartesianTrial

::: phydrax.atomistic.AtomisticTrialOutcome

::: phydrax.atomistic.evaluate_cartesian_trial

::: phydrax.atomistic.trial_geometry

::: phydrax.atomistic.program_energy

::: phydrax.atomistic.initialize_capacity

::: phydrax.atomistic.replay_capacity

::: phydrax.atomistic.AtomisticHamiltonianTransition

::: phydrax.atomistic.AtomisticHamiltonianTransitionKind

::: phydrax.atomistic.AtomisticHamiltonianTransitionResult

::: phydrax.atomistic.hamiltonian_transition_kind

## Rollout, replay, checkpoints, restarts, and stress

A runtime checkpoint requires an explicitly matching prepared model. A restart
additionally bundles the pickle-free native model artifact for fresh-process
continuation.

::: phydrax.atomistic.AtomisticTrajectoryPlan

::: phydrax.atomistic.AtomisticRolloutPlan

::: phydrax.atomistic.AtomisticReplayPolicy

::: phydrax.atomistic.AtomisticCheckpointPlan

::: phydrax.atomistic.write_atomistic_checkpoint

::: phydrax.atomistic.read_atomistic_checkpoint

::: phydrax.atomistic.AtomisticCheckpoint

::: phydrax.atomistic.AtomisticCheckpointResources

::: phydrax.atomistic.read_atomistic_checkpoint_resources

::: phydrax.atomistic.write_atomistic_restart

::: phydrax.atomistic.read_atomistic_restart_models

::: phydrax.atomistic.AtomisticRestartModelBinding

::: phydrax.atomistic.read_atomistic_restart

::: phydrax.atomistic.atomistic_cell_energy_and_stress

A Monte Carlo volume move accepts with
`ln alpha = -beta (U' - U + P (V' - V)) + (N + 1) ln(V'/V)`; its draws are
addressed by `state.controllers.move_counter`, which advances once per completed
Metropolis decision.

::: phydrax.atomistic.IsotropicMonteCarloBarostatPlan

::: phydrax.atomistic.AtomisticBarostatEvaluation

::: phydrax.atomistic.apply_isotropic_monte_carlo_barostat

## Staged campaigns

An `AtomisticCampaignPlan(campaign_id, stages, tasks, max_pending_publications=16)`
composes native scientific transitions into `AtomisticStagePlan` stages of bounded lane
banks over the existing execution, workset, and lifecycle owners. The transitions are
molecular dynamics, relaxation, grouped reaction paths, generation, and labeling. The
campaign is not another scheduler.

An `AtomisticStagePlan` maps each system family to one stage operation. Families are how
mixed sizes are expressed: each family gets fixed-capacity banks at its own system
capacity. The stage plan also declares:

- a per-visit `budget`, required for dynamics and generation;
- `AtomisticStageRouting`, which holds one `AtomisticStageRoute` per terminal outcome;
- `lane_capacity`, which bounds resident lanes, and `queue_capacity`, which bounds tasks
  waiting to enter the stage;
- an `AtomisticEntryPolicy`, which controls reuse of an incoming evaluation;
- an `AtomisticRevisionPolicy`, which controls in-flight lanes when an operation is
  rebound;
- an `AtomisticContinuationPolicy`, which decides whether a dynamics stage continues an
  incoming walker (`CONTINUE`, the default) or explicitly begins a new run from its
  phase point (`RESTART`).

A stage operation (`AbstractAtomisticStageOperation`) is a shared lane executor bound to
one prepared native runtime. The concrete operations are:

- `AtomisticDynamicsStageOperation`;
- `AtomisticRelaxationStageOperation` (groups of a `PreparedAtomisticRelaxation`);
- `AtomisticReactionPathStageOperation` (whole bands of
  `phydrax.chemistry.reaction.PreparedAtomisticNudgedElasticBand`; see
  [chemistry](chemistry.md));
- `AtomisticGenerationStageOperation`;
- `AtomisticLabelStageOperation`.

An operation's static `AtomisticStageSignature` holds only layout identities, while its
scientific and revision identities are dynamic `IdentityWords`. Signature grouping
therefore equals compilation. Stages that reference the same operation share one lane
bank.

`AtomisticCampaignTask` holds stable task, material, run, and replica IDs, an optional
`group_id`, and an `AtomisticTaskPacket` of accepted host state. A group must fit in one
bank and is admitted, staged, and moved as a whole. A reaction-path lane is one complete
band, and its image slots are never refilled individually.

An evaluation is reused across stages only under `AtomisticEntryPolicy.SHARE`, by an
operation that supports shared entry (currently labeling with `reuse=True`), and only
when the operation's `AtomisticEvaluationKey` is served by the packet's key. Capacity
repair follows `AtomisticCapacityMode`:

- `REBUCKET` (dynamics, label, generation) re-primes only the offending task into a bank
  at grown declared capacities, consuming no step, draw, or budget;
- `GROW` (relaxation, paths) grows the whole bank in place and keeps optimizer state.

`PreparedAtomisticCampaign.rebind_operation` moves a stage family to another numeric
revision. Under `DRAIN`, admitted lanes finish under the old binding. Under `REPRIME`,
each admitted lane passes through its owner's explicit `transition_hamiltonian`.

A stage boundary is an observation cadence, not a new run. A dynamics lane exports its
complete accepted state as an `AtomisticDynamicsContinuation`: thermostat chains,
barostat, GLE memory and Monte Carlo counters, step, time, RNG key, images, term states,
force cache, and neighborhood resources, bound to the exporting operation's
`binding_id`. The next dynamics stage continues it:

- a state of the same runtime and thermodynamic table is adopted bitwise, so observing
  a walker every `n` steps reproduces one uninterrupted run;
- a state of the same runtime at grown capacities moves to that capacity bucket;
- another row of the same table is entered by the owner's warm
  `rebase_thermodynamic_state`;
- a resource-only change of the same scientific Hamiltonian uses the owner's
  `rebind_hamiltonian` and `prime_state`;
- a changed Hamiltonian of the same method uses the source runtime's
  `transition_hamiltonian`; its lineage is recorded and the energy jump is booked as
  external work. The source operation must be in the plan or resident in a bank;
  otherwise the walker is refused with `IDENTITY_MISMATCH`.

Any other change (integrator, table, constraints, strain response, force ports, or
system) is refused with `CONTINUATION_REFUSED`, unless the stage declares
`AtomisticContinuationPolicy.RESTART`. A restart is initialized by the owner from the
incoming positions, momenta, cell, and time with the campaign's semantic key, and the
visit records `DYNAMICS_RESTARTED`. Label stages evaluate a configuration without moving
it, so a walker's continuation passes through them, and their outputs carry its momenta
and time. Relaxation and path stages move the configuration and end the run. A refused
walker keeps its continuation as evidence.

`PreparedAtomisticCampaign.terminal_packets(state)` returns the final continuation
packet of every terminal walker, whatever its outcome (`state.task(task_id).outcome`).
Such a packet may be the initial packet of an `AtomisticCampaignTask` in a later
campaign, which continues it. To continue a walker under a changed model while the old
runtime is still available, call `old_operation.transition_packet(packet,
new_operation)`; it applies the same owner transition and raises `ValueError` with the
campaign and owner reasons on refusal. `operation.continuation(state)` and
`operation.continuation_packet(continuation)` authenticate a state against an
operation's runtime, table, and resource bundle. `operation.state_template()` gives the
state structure for restoring a stored continuation. Sink outputs never carry
continuations. The campaign envelope persists queued, observed, and terminal walker
continuations, one per task.

Each round runs `evaluate`, `stage_outputs`, `publish(state, sink)`,
`commit(state, store)`, and `refill`, and `run_round` composes them.

- `stage_outputs` moves a whole unit only when both the next stage queue and the bounded
  pending-publication bank admit it. Otherwise the unit's lanes park, and no physics
  advances.
- `publish` stops at the first sink backpressure.
- `commit` atomically replaces the `AtomisticCampaignStore` envelope.
- `resume(store, operations=())` restores the last durable commit in a fresh process and
  refills deterministically.

An `AtomisticCampaignOutput` is idempotent by its `semantic_key`, which is derived from
the campaign, task, stage, and stage entry. Its `payload_id` fingerprints the complete
content. Sinks (`AbstractAtomisticCampaignSink`, `AtomisticDirectorySink`) return an
existing `AtomisticOutputReceipt` on replay and refuse a different payload under the same
key, so outputs replayed after an interrupted commit are published exactly once.

`AtomisticStageOutcome` keeps six results distinct: scientific convergence, completion
of a fixed budget, exhaustion before convergence, failure, pending capacity, and refused
capacity. By default, converged and completed lanes continue to the next stage, while
exhausted, failed, and capacity-refused lanes terminate and publish their outcome.
Every output records the method `reasons`, the `AtomisticCampaignReason` bits, its
counters, numeric revisions, Hamiltonian lineage, and RNG words. Energy, force, and
stress ports are present only for a finite accepted evaluation.

The recipe in the
[dynamics cookbook](../cookbook/atomistic_dynamics.md#staged-heterogeneous-campaigns)
follows `examples/atomistic_relaxation_campaign.py`. That example exits normally after
publishing and before committing, then resumes in a fresh process. It is not a
`SIGKILL` test.

Stage x domain campaigns over execution groups have a separate owner,
`AtomisticDomainCampaignPlan`, which is documented with
[distributed execution](atomistic_advanced.md#stage-transport-and-stage-x-domain-campaigns).

::: phydrax.atomistic.AtomisticCampaignPlan

::: phydrax.atomistic.PreparedAtomisticCampaign

::: phydrax.atomistic.AtomisticCampaignState

::: phydrax.atomistic.AtomisticCampaignTask

::: phydrax.atomistic.AtomisticCampaignOutput

::: phydrax.atomistic.AtomisticCampaignReason

::: phydrax.atomistic.AtomisticCampaignStore

::: phydrax.atomistic.AtomisticStagePlan

::: phydrax.atomistic.AtomisticStageRoute

::: phydrax.atomistic.AtomisticStageRouting

::: phydrax.atomistic.AtomisticStageSignature

::: phydrax.atomistic.AtomisticStageOperationKind

::: phydrax.atomistic.AtomisticStageOutcome

::: phydrax.atomistic.AbstractAtomisticStageOperation

::: phydrax.atomistic.AtomisticDynamicsStageOperation

::: phydrax.atomistic.AtomisticRelaxationStageOperation

::: phydrax.atomistic.AtomisticReactionPathStageOperation

::: phydrax.atomistic.AtomisticGenerationStageOperation

::: phydrax.atomistic.AtomisticLabelStageOperation

::: phydrax.atomistic.AtomisticLaneBank

::: phydrax.atomistic.AtomisticLaneAdmission

::: phydrax.atomistic.AtomisticLaneAdvance

::: phydrax.atomistic.AtomisticDynamicsLanes

::: phydrax.atomistic.AtomisticLabelLanes

::: phydrax.atomistic.AtomisticEntryPolicy

::: phydrax.atomistic.AtomisticRevisionPolicy

::: phydrax.atomistic.AtomisticContinuationPolicy

::: phydrax.atomistic.AtomisticDynamicsContinuation

::: phydrax.atomistic.AtomisticCapacityMode

::: phydrax.atomistic.AtomisticCapacityGrowth

::: phydrax.atomistic.AtomisticCapacityTransfer

::: phydrax.atomistic.AtomisticEvaluationKey

::: phydrax.atomistic.AtomisticEvaluationRecord

::: phydrax.atomistic.AtomisticTaskPacket

::: phydrax.atomistic.AtomisticTaskRecord

::: phydrax.atomistic.AtomisticTaskStatus

::: phydrax.atomistic.AtomisticRoundEvidence

::: phydrax.atomistic.AbstractAtomisticCampaignSink

::: phydrax.atomistic.AtomisticDirectorySink

::: phydrax.atomistic.AtomisticOutputReceipt

::: phydrax.atomistic.AtomisticCommitReceipt

::: phydrax.atomistic.atomistic_hamiltonian_revision

::: phydrax.atomistic.write_atomistic_campaign_checkpoint

::: phydrax.atomistic.read_atomistic_campaign_checkpoint

## Native model artifacts

Archives are bounded by `ATOMISTIC_MODEL_ARTIFACT_LIMITS` before allocation.
Restoring an archive reruns the registered model's scientific validators and
recomputes every identity. No optional provider package is imported.

::: phydrax.atomistic.write_atomistic_model_artifact

::: phydrax.atomistic.read_atomistic_model_artifact

::: phydrax.atomistic.AtomisticModelArtifact

::: phydrax.atomistic.AtomisticModelArtifactManifest

::: phydrax.atomistic.AtomisticModelArtifactError

::: phydrax.atomistic.atomistic_model_identity

::: phydrax.atomistic.register_atomistic_model_artifact

## Controlled Hamiltonians, providers, and specialized methods

`NativeAtomisticProviderPlan(model, graph_execution, *, finite_resources=None,
periodic_resources=None, name="learned")` prepares one learned model as a native
provider. Its resources are explicit complete `AtomisticNeighborhoodBundlePlan`
values for finite and periodic systems; a system kind without declared resources
is refused, with no dense or implicit fallback. Energy, forces, and stress come
from one evaluation of the same scalar energy over one neighborhood bundle state.
Stress is available exactly when the system is a fully periodic 3D cell and every
program term owns cell derivatives. Dynamics binds the prepared program, never a
provider neighborhood.

::: phydrax.atomistic.AlchemicalControlSchedulePlan

::: phydrax.atomistic.AlchemicalInteractionPartitionPlan

::: phydrax.atomistic.ControlledHamiltonianPlan

::: phydrax.atomistic.PreparedControlledHamiltonian

::: phydrax.atomistic.RegionMaskedPotential

::: phydrax.atomistic.RESPAPlan

::: phydrax.atomistic.AbstractExternalAtomisticProvider

::: phydrax.atomistic.NativeAtomisticProviderPlan

::: phydrax.atomistic.NativeAtomisticProvider

::: phydrax.atomistic.NativeAtomisticRequest

::: phydrax.atomistic.NativeAtomisticEvaluation

::: phydrax.atomistic.BornOppenheimerVelocityVerletPlan

::: phydrax.atomistic.RingPolymerPlan

::: phydrax.atomistic.PreparedRingPolymerDynamics

::: phydrax.atomistic.VarianceConstrainedSemiGrandPlan

## Typed training

`fit_atomistic_potential` runs every loader global batch as one accumulation
window and one attempt of the shared accepted-update training kernel, with
`MODEL` root authority and one kernel objective per declared atomistic objective.
Numerators and supports merge before normalization. A failed window (nonfinite
or capacity) ends the run with `NONFINITE` or `NEIGHBOR_OVERFLOW` at the last
committed boundary with its loader cursor; a zero-support window is a committed
skip, and a continuation replays from that boundary.
`AtomisticTrainingResult.training_state` holds the committed kernel state
(parameters, optimizer and plateau rule state, EMA target, open window, root
key, and cursors); training checkpoints and restarts persist it without pickle.
Literal selectors: `AtomisticObjectiveWeighting` (`"literal"`, `"normalized"`),
`AtomisticEvaluationView` (`"current"`, `"target"`), `AtomisticResidualKind`
(`"squared"`, `"absolute"`, `"huber"`, `"euclidean"`), `AtomisticScheduleKind`,
`AtomisticEnergyReduction` (`"per-atom-structure"`, `"total-structure"`,
`"per-atom-atom-weighted"`), `AtomisticForceReduction` (`"component"`, `"atom"`,
`"structure"`), `AtomisticTensorReduction` (`"component"`, `"structure"`), and
`AtomisticLabelLevel` (`"structure"`, `"atom"`). `ATOMISTIC_CARTESIAN_AXIS` and
`ATOMISTIC_CELL_VECTOR_AXIS` are the canonical label axes.

::: phydrax.atomistic.atomistic_training_schema

::: phydrax.atomistic.atomistic_training_source

::: phydrax.atomistic.atomistic_training_loader

::: phydrax.atomistic.AtomisticTrainingFields

::: phydrax.atomistic.AtomisticBatchPreparation

::: phydrax.atomistic.AtomisticTrainingProblem

::: phydrax.atomistic.AtomisticTrainingNormalization

::: phydrax.atomistic.fit_atomistic_normalization

::: phydrax.atomistic.AtomisticParameterSelection

::: phydrax.atomistic.AtomisticIrrepLowRankAdaptation

::: phydrax.atomistic.AtomisticIrrepLowRankTree

::: phydrax.atomistic.atomistic_irrep_linear_sites

::: phydrax.atomistic.AtomisticTrainedTree

::: phydrax.atomistic.AtomisticPlateauPolicy

::: phydrax.atomistic.AtomisticOptimizerState

::: phydrax.atomistic.AtomisticTrainingPolicy

::: phydrax.atomistic.AtomisticTrainingResult

::: phydrax.atomistic.AtomisticObjectiveEvaluation

::: phydrax.atomistic.evaluate_atomistic_objectives

::: phydrax.atomistic.fit_atomistic_potential

::: phydrax.atomistic.write_atomistic_training_checkpoint

::: phydrax.atomistic.read_atomistic_training_checkpoint

::: phydrax.atomistic.AtomisticTrainingPrecision

::: phydrax.atomistic.AtomisticResidualLoss

::: phydrax.atomistic.AtomisticObjectiveSchedule

::: phydrax.atomistic.AbstractAtomisticObjective

::: phydrax.atomistic.AtomisticEnergyObjective

::: phydrax.atomistic.AtomisticForceObjective

::: phydrax.atomistic.AtomisticStressObjective

::: phydrax.atomistic.AtomisticPropertyObjective

::: phydrax.atomistic.AtomisticAtomEnergyObjective

::: phydrax.atomistic.AtomisticReferenceEnergyFit

::: phydrax.atomistic.fit_atomistic_reference_energies

::: phydrax.atomistic.AtomisticTrainingBatch

::: phydrax.atomistic.AtomisticHostLabels

::: phydrax.atomistic.AtomisticObjectiveTerm

::: phydrax.atomistic.AbstractAtomisticObjectiveReduction

::: phydrax.atomistic.AtomisticLocalObjectiveReduction

::: phydrax.atomistic.atomistic_training_derivatives

::: phydrax.atomistic.atomistic_training_outputs

::: phydrax.atomistic.atomistic_training_hessian

::: phydrax.atomistic.energy_objective_term

::: phydrax.atomistic.force_objective_term

::: phydrax.atomistic.stress_objective_term

::: phydrax.atomistic.case_atom_sum

::: phydrax.atomistic.atomistic_stress_unit

::: phydrax.atomistic.write_atomistic_training_restart

::: phydrax.atomistic.read_atomistic_training_restart

## Training data loaders

Mixtures bind canonical source order, draw probabilities and the sampled or
importance-corrected objective; buckets bound cases, rows of every counted level,
and bytes; cursors bind source content, mixture, bindings, buckets, and
placement. Literal selectors: `AtomisticMixtureDraw` (`"epoch"`,
`"replacement"`), `AtomisticMixtureObjective` (`"sampled"`,
`"importance-corrected"`), `AtomisticLevelLayout` (`"per-case"`, `"packed"`),
`AtomisticLoaderRemainder` (`"short"`, `"drop"`), `AtomisticLoaderTarget`
(`"host"`, `"device"`), and `AtomisticLoaderFailure`.

::: phydrax.atomistic.AtomisticMixtureComponent

::: phydrax.atomistic.AtomisticSourceMixture

::: phydrax.atomistic.AtomisticSplitEvidence

::: phydrax.atomistic.AtomisticFieldBinding

::: phydrax.atomistic.AtomisticLevelCapacity

::: phydrax.atomistic.AtomisticWorkloadBucket

::: phydrax.atomistic.AtomisticWorkloadBuckets

::: phydrax.atomistic.AtomisticLoaderPlacement

::: phydrax.atomistic.AtomisticLoaderStaging

::: phydrax.atomistic.AtomisticStructureLoader

::: phydrax.atomistic.AtomisticLoaderEpoch

::: phydrax.atomistic.AtomisticLoaderCursor

::: phydrax.atomistic.AtomisticLoaderItem

::: phydrax.atomistic.AtomisticPayloadLayout

::: phydrax.atomistic.AtomisticPayloadLevel

::: phydrax.atomistic.AtomisticLoaderPayload

::: phydrax.atomistic.AtomisticPayloadProvenance

::: phydrax.atomistic.AtomisticLoaderError

## Structure store

Immutable structure sources and the canonical content-addressed Zarr store
(optional `atomistic-data` extra). Literal selectors: `AtomisticStoreLevelKind`,
`AtomisticStoreFieldRole`, `AtomisticStoreValueKind`, `AtomisticStoreExtensivity`,
`AtomisticStoreAvailability`, `AtomisticStoreDType`, `AtomisticStoreIndexBasis`,
`AtomisticStoreCompression`, `AtomisticStoreOperation`, and
`AtomisticStoreFailure`.

::: phydrax.atomistic.AtomisticStoreLevel

::: phydrax.atomistic.AtomisticStoreIndex

::: phydrax.atomistic.AtomisticStoreField

::: phydrax.atomistic.AtomisticStructureStoreSchema

::: phydrax.atomistic.AtomisticStructureRecord

::: phydrax.atomistic.AtomisticStructureBatch

::: phydrax.atomistic.AbstractAtomisticStructureSource

::: phydrax.atomistic.AtomisticStructureSource

::: phydrax.atomistic.AtomisticStoredStructureSource

::: phydrax.atomistic.AtomisticStoreLayout

::: phydrax.atomistic.AtomisticStoreLimits

::: phydrax.atomistic.AtomisticStoreShard

::: phydrax.atomistic.AtomisticStoreRetraction

::: phydrax.atomistic.AtomisticStoreCommit

::: phydrax.atomistic.AtomisticStructureStoreWriter

::: phydrax.atomistic.open_atomistic_structure_store

::: phydrax.atomistic.AtomisticStoreError

## Distillation

Native teachers, typed teacher labels, distillation objectives, experience replay,
and the on-policy transaction. See the
[distillation guide](../guides_atomistic_distillation.md). The teacher slot names
(`TEACHER_ENERGY_SLOT`, `TEACHER_FORCES_SLOT`, `TEACHER_STRESS_SLOT`,
`TEACHER_ATOM_ENERGY_SLOT`, `TEACHER_EMBEDDING_SLOT`, `HESSIAN_PROBE_SLOT`,
`TEACHER_HVP_SLOT`, `BOLTZMANN_GROUP_SLOT`, `BOLTZMANN_GENERATOR_SLOT`,
`BOLTZMANN_PV_SLOT`), `ATOMISTIC_DISTILLATION_PHASES`, and the mixed third-order
`ATOMISTIC_HVP_TRAINING_REQUEST` are module constants; `AtomisticHessianProbeKind`
(`"rademacher"`, `"gaussian"`), `AtomisticDistillationPhase`, and
`AtomisticDistillationDrive` (`"promoted"`, `"candidate"`) are literal selectors.

::: phydrax.atomistic.AtomisticTeacher

::: phydrax.atomistic.AtomisticLabelRequest

::: phydrax.atomistic.AtomisticHessianProbe

::: phydrax.atomistic.atomistic_label_id

::: phydrax.atomistic.atomistic_distillation_schema

::: phydrax.atomistic.label_atomistic_structures

::: phydrax.atomistic.require_atomistic_labels

::: phydrax.atomistic.AtomisticTeacherLabelError

::: phydrax.atomistic.AtomisticEmbeddingProjector

::: phydrax.atomistic.AtomisticEmbeddingObjective

::: phydrax.atomistic.AtomisticHessianVectorObjective

::: phydrax.atomistic.AtomisticGeneratorState

::: phydrax.atomistic.AtomisticGeneratorSamples

::: phydrax.atomistic.atomistic_generator_source

::: phydrax.atomistic.AtomisticGeneratorMixture

::: phydrax.atomistic.AtomisticBoltzmannGroup

::: phydrax.atomistic.atomistic_boltzmann_source

::: phydrax.atomistic.AtomisticBoltzmannObjective

::: phydrax.atomistic.AtomisticExperienceAdmission

::: phydrax.atomistic.AtomisticReplayDraw

::: phydrax.atomistic.AtomisticExperienceBuffer

::: phydrax.atomistic.AtomisticReplayRevision

::: phydrax.atomistic.AtomisticExperienceFrame

::: phydrax.atomistic.AtomisticFrameLabel

::: phydrax.atomistic.AtomisticEvictionRecord

::: phydrax.atomistic.AtomisticRefusalRecord

::: phydrax.atomistic.AtomisticDistillationRollout

::: phydrax.atomistic.AtomisticDistillationQualification

::: phydrax.atomistic.AtomisticQualificationOutcome

::: phydrax.atomistic.AtomisticOnPolicyDistillationPlan

::: phydrax.atomistic.AtomisticOnPolicyDistillationState

::: phydrax.atomistic.run_atomistic_distillation_segment

::: phydrax.atomistic.AtomisticDistillationSegmentResult

::: phydrax.atomistic.AtomisticDistillationReport

::: phydrax.atomistic.write_atomistic_distillation_restart

::: phydrax.atomistic.read_atomistic_distillation_restart

## rMD17 archives

::: phydrax.atomistic.RMD17Dataset

::: phydrax.atomistic.load_rmd17_npz

::: phydrax.atomistic.split_rmd17

## Generation streams

Supports, sampler lanes, condition schedules, filters, ordinal routing, sinks, and
the ordinal-addressed generation stream. See the
[generation guide](../guides_atomistic_generation.md). Proposals carry no
likelihood, Boltzmann weight, or physical qualification. The `policy` of
`AtomisticConditionSchedule` is `"cycle"` or `"bounded"`;
`GENERATION_ADDRESS_NAMESPACE` is the semantic RNG namespace of every proposal
key.

::: phydrax.atomistic.AbstractAtomisticGenerationSink

::: phydrax.atomistic.AbstractConditionSource

::: phydrax.atomistic.AbstractOrdinalRoute

::: phydrax.atomistic.AbstractPreparedAtomisticSampler

::: phydrax.atomistic.AbstractProposalFilter

::: phydrax.atomistic.atomistic_proposal_id

::: phydrax.atomistic.AtomisticCellAdmission

::: phydrax.atomistic.AtomisticCellBounds

::: phydrax.atomistic.AtomisticChargeBalanceQualifier

::: phydrax.atomistic.AtomisticChemistryAdmission

::: phydrax.atomistic.AtomisticCompositionBounds

::: phydrax.atomistic.AtomisticConditionSchedule

::: phydrax.atomistic.AtomisticConditionTransform

::: phydrax.atomistic.AtomisticCoordinateRepresentation

::: phydrax.atomistic.AtomisticFilterEvidence

::: phydrax.atomistic.AtomisticFixedRoute

::: phydrax.atomistic.AtomisticGeneratedProposal

::: phydrax.atomistic.AtomisticGenerationCursor

::: phydrax.atomistic.AtomisticGenerationReceipt

::: phydrax.atomistic.AtomisticGenerationState

::: phydrax.atomistic.AtomisticGenerationStopReason

::: phydrax.atomistic.AtomisticGenerationStreamPlan

::: phydrax.atomistic.AtomisticGenerationSupport

::: phydrax.atomistic.AtomisticMemoryGenerationSink

::: phydrax.atomistic.AtomisticMinimumDistanceFilter

::: phydrax.atomistic.AtomisticPayloadSpec

::: phydrax.atomistic.AtomisticProposalBank

::: phydrax.atomistic.AtomisticProposalBlock

::: phydrax.atomistic.AtomisticProposalStatus

::: phydrax.atomistic.AtomisticSamplerFamily

::: phydrax.atomistic.AtomisticSamplerLaneState

::: phydrax.atomistic.AtomisticStructureStoreGenerationSink

::: phydrax.atomistic.AtomisticTopologyAdmission

::: phydrax.atomistic.AtomisticWeightedRoute

::: phydrax.atomistic.map_admitted_lanes

::: phydrax.atomistic.ordinal_words

::: phydrax.atomistic.PreparedAtomisticGenerationStream

::: phydrax.atomistic.proposal_address

::: phydrax.atomistic.proposal_keys

::: phydrax.atomistic.role_keys

::: phydrax.atomistic.screen_atomistic_proposals

### Distributed quotas

`AtomisticGenerationPlacement` places one stream on the participants of a live
`ExecutionGroup` (`AtomisticGenerationStreamPlan.prepare(placement=...)`): one
global ordinal window, accepted target, attempt budget, replicated pending bank,
and receipt chain, with group-local rank 0 as the publication authority. A placed
stream persists through `PreparedAtomisticGenerationStream.publish_checkpoint` /
`restore_checkpoint` (same stream profile and placement only); see
[distributed quotas](../guides_atomistic_generation.md#distributed-quotas).

::: phydrax.atomistic.AtomisticGenerationPlacement

::: phydrax.atomistic.AtomisticGenerationCapacityError

::: phydrax.atomistic.AtomisticGenerationCheckpointError

::: phydrax.atomistic.AtomisticGenerationDivergenceError

### Periodic coordinate, cell, and species diffusion

::: phydrax.atomistic.atomistic_score_equivariance

::: phydrax.atomistic.AtomisticCoordinateDiffusion

::: phydrax.atomistic.AtomisticEquivarianceReport

::: phydrax.atomistic.AtomisticHybridDiffusion

::: phydrax.atomistic.AtomisticLogCholeskyCellChart

::: phydrax.atomistic.AtomisticLogCholeskyCellChartEvidence

::: phydrax.atomistic.AtomisticPeriodicCoordinateDiffusion

::: phydrax.atomistic.AtomisticPeriodicPerturbation

::: phydrax.atomistic.AtomisticPeriodicSamplerMethod

::: phydrax.atomistic.AtomisticSpeciesDiffusion

::: phydrax.atomistic.AtomisticWrappedNormalScore

::: phydrax.atomistic.PreparedPeriodicDiffusionSampler

::: phydrax.atomistic.PreparedPeriodicFlowSampler

## Qualification

::: phydrax.atomistic.AtomisticDynamicsQualificationClaim

::: phydrax.atomistic.AtomisticDynamicsQualificationProfile

::: phydrax.atomistic.AtomisticDynamicsQualificationResult
