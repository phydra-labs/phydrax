# Incompressible flow

## Spectral problems and periodic projection

Periodic compilation retains a full-complex live velocity state. Optional
`HermitianSpectralCoordinates` validate the real-field boundary and can encode
selected checkpoint leaves without changing solver callbacks. Constant-power and OU
forcing are explicit application plans; neither is injected automatically.


::: phydrax.equations.IncompressibleFlowProblem

---

::: phydrax.discretization.PeriodicLerayProjector

---

::: phydrax.equations.compile_periodic_incompressible_flow

---

::: phydrax.equations.CompiledIncompressibleSpectralDynamics

---

::: phydrax.discretization.IncompressibleSpectralDiagnostics


## Large-eddy simulation
Periodic static algebraic LES compiles into the named incompressible stage and uses
a current-state ETDRK guard. Periodic dynamic LES compiles with its exact coarser
test discretization and gains transactional ETDRK production; periodic-uniform MAC
dynamic LES compiles with a projected explicit method. Distributed slab/pencil
full-flow production, channel mixed traction, MAC stochastic inflow, KSGS variants,
learned stress, pressure-stepped unstructured flow, Favre transport, and immersed
execution retain their route-specific contracts. The normative
[LES equations](../equations/les.md) define signs, trace, filters, formulas, identity,
and AD; the [LES guide](../../guides_large_eddy_simulation.md) owns workflows and
candidate/refusal status.

::: phydrax.equations.PeriodicAlgebraicLESPlan

---

::: phydrax.equations.PeriodicDynamicLESPlan

---

::: phydrax.equations.MACAlgebraicLESPlan

---

::: phydrax.equations.MACDynamicLESPlan

---

::: phydrax.solver.LESStabilityGuardedETDRKMethod

---

::: phydrax.solver.PreparedLESStabilityGuardedETDRKMethod

---

## Fourier–Chebyshev–Fourier channel flow

`ChannelStokesPlan` defaults to the fixed-band, pressure-eliminated
`ultraspherical_banded` route while returning primitive velocity, pressure, and
pressure-gradient fields. Its zero horizontal mode owns pressure-gradient or
two-component bulk-flux control; nonzero modes use wall-normal
velocity/vorticity elimination. `dense_reference` is an explicit oracle and carries
no `ultraspherical_banded` production or qualification evidence. Preparation reports
the route-specific ranks, storage/workspace, pivot margin, and
unsharded wall-normal axis.

`PreparedChannelSBDF2Method` binds backward-Euler and BDF2 Stokes solves to one
exact step. `ChannelSBDF2State` is the restart state and includes both velocity and
nonlinear histories, pressure, affine pressure gradient, and history count. The
method forbids step reduction, including output alignment and robust retry reduction.


::: phydrax.discretization.ChannelMeanConstraint

---

::: phydrax.discretization.ChannelStokesPlan

---

::: phydrax.discretization.PreparedChannelStokesSolver

---

::: phydrax.discretization.ChannelStokesSolveResult

---

::: phydrax.equations.compile_channel_flow

---

::: phydrax.equations.CompiledChannelFlowDynamics

---

::: phydrax.equations.ChannelVelocityDiagnostics

---

::: phydrax.solver.ChannelSBDF2Method

---

::: phydrax.solver.solve_channel_sbdf2

---

::: phydrax.solver.ChannelFlowSolution

---

::: phydrax.solver.PreparedChannelSBDF2Method

---

::: phydrax.solver.ChannelSBDF2State


### Channel LES

`compile_channel_les` adds retained negative SGS-stress divergence. Its variable
wall-normal implicit filter is noncommuting with no correction.
`ChannelLESExplicitRestriction` is the enforced complete channel-SBDF2
advection-plus-SGS budget. The default remains wall resolved; optional
`PreparedVectorEquilibriumWallStressChannel` replaces tangential essential data
with equilibrium traction while retaining normal constraints. It evaluates the
retained Chebyshev expansion at declared lower/upper off-wall sample distances,
zeroes normal velocity, and requires stationary walls and zero prescribed pressure
gradient.

::: phydrax.equations.channel_les_filter

---

::: phydrax.equations.compile_channel_les

---

::: phydrax.equations.CompiledChannelLESDynamics

---



::: phydrax.applications.incompressible_flow.PreparedVectorEquilibriumWallStressChannel

---

## Structured finite-volume MAC dynamics and projection

`MACPressureOperatorSpec` freezes the generalized action
`A p = -D(beta G_h p)` with one positive coefficient, boundary preparation, optional
static Robin traces, route request, resource bound, and geometry epoch. Uniform
constant coefficients can use the exact tensor `transform` route; three-dimensional
all-Neumann constant or line-structured coefficients can use exact `hybrid`.
Execution of either direct route requires its matching prepared transform action.

Symmetric general coefficients and mixed/Robin closures select PCG with a frozen
constant preconditioner; stabilized nonsymmetric traction selects FGMRES. Preparation
and execution expose distinct coefficient, lift, symmetry, derivative, resource,
residual, gauge, and boundary-power evidence. `MACDistributedProjectionPlan` remains
a separate collective matrix-free CG owner; no direct route silently gathers shards.


::: phydrax.discretization.MACOperatorPlan

---

::: phydrax.discretization.PreparedMACOperators

---

::: phydrax.discretization.MACBoundaryPlan

---

::: phydrax.discretization.MACMomentumPlan

---

::: phydrax.discretization.PreparedMACMomentumOperators

---

::: phydrax.discretization.MACMomentumReport

---

::: phydrax.discretization.MACMomentumDiagnostics

---

::: phydrax.solver.MACPressureProjectionPlan

---

::: phydrax.solver.MACPressureProjectionResult

---

::: phydrax.solver.MACRateProjectionResult

---

::: phydrax.equations.compile_mac_incompressible_flow

---

::: phydrax.equations.CompiledMACIncompressibleDynamics

---

::: phydrax.equations.MACIncompressibleDiagnostics

---

::: phydrax.equations.MACLESStepRestriction

---

::: phydrax.solver.MACPressureRobinSide

---

::: phydrax.solver.MACPressureOperatorSpec

---

::: phydrax.solver.PreparedMACPressureOperator

---

::: phydrax.solver.MACPressureSolveResult

---

::: phydrax.solver.MACWeightedPressureAction

---


## Incompressible production and statistics

`ProductionRunPlan` binds an `AbstractFixedStepMethod`--including prepared ETDRK,
channel SBDF2, or a compatible MAC method--to an absolute end time, absolute
accepted-step capacity, checkpoint cadence, compiled segment
length, optional exact output schedule, streaming moments, and typed
checkpoint/publish/stop triggers. `PreparedProductionRun` carries accepted PyTree
state and all continuation cursors across segment and checkpoint boundaries.
Checkpoint IDs include stored content; immutable generations restore only for
identical case, runtime, tree, and encoding identities. Full-complex Hermitian state
leaves may use independent-real checkpoint encoding.

`PeriodicModalTurbulenceStatisticsPlan(dynamics, bin_edges, /, *, ...)` binds one compiled
equation and reports conservative full-complex energy, molecular dissipation,
advective transfer, SGS transfer, forcing injection, resolved spectral flux,
derived scales/tails, static-SGS energy/projection evidence, and current step limits.
`evaluate(time, velocity, args, *, stage=..., additive_forcing_rate=...,
step_restriction=...)` rejects foreign stages or restrictions.
`SpectralChannelStatisticsPlan` provides
homogeneous-plane raw/central moments and separate-wall shear/friction semantics.
`StreamingMomentPlan` places those or other real-valued observables in sample- or
time-weighted windows with optional fixed-capacity completed blocks.

`PeriodicSpectralProductionPlan(dynamics, method, statistics, case, /, *, ...)`
verifies matching compiled dynamics, a `PeriodicSpectralProductionCase` bound to the
initial modal velocity, Hermitian ETDRK coordinates, statistics, static LES action,
and forcing identities. Compiled constant-power wiring is supported; adapter wiring
is unavailable for guarded LES. Mutually exclusive OU forcing and realization inputs
produce a `PreparedOUForcedETDRKMethod` whose accepted state transactionally couples
velocity and exact OU continuation. `SpectralChannelProductionPlan` derives the exact
step from prepared SBDF2 and requires its horizon, outputs, and statistics window on
that lattice.
`StructuredMACProductionPlan` binds a fixed-step method, compiled MAC dynamics, and
`MACPlaneWallStatisticsPlan`; optional `MACConstantPressureGradientForcing` must
already be compiled with the same identity. General feedback is instead owned by
`MACFlowControlPlan`: prepared SSPRK, IMEX-Euler, or SBDF2 steps control a prescribed
pressure gradient, bulk velocity, or frozen-density mass flux through a finite
method-stage response map, with complete continuation state and atomic rollback.
Each production route prepares a checkpoint root before initialization.

::: phydrax.applications.incompressible_flow.MACFlowControlTarget

---

::: phydrax.applications.incompressible_flow.MACFlowControlPlan

---

::: phydrax.applications.incompressible_flow.PreparedMACFlowControl

---

::: phydrax.applications.incompressible_flow.MACFlowControlState

---

::: phydrax.applications.incompressible_flow.MACFlowControlStepResult

---


::: phydrax.applications.incompressible_flow.PeriodicSpectralProductionPlan

---

::: phydrax.applications.incompressible_flow.PeriodicSpectralProductionCase

---
::: phydrax.applications.incompressible_flow.PreparedOUForcedETDRKMethod

---


::: phydrax.applications.incompressible_flow.SpectralChannelProductionPlan

---

::: phydrax.applications.incompressible_flow.StructuredMACProductionPlan

---

::: phydrax.applications.incompressible_flow.MACPlaneWallStatisticsPlan

---

::: phydrax.solver.ProductionRunPlan

---

::: phydrax.solver.PreparedProductionRun

---

::: phydrax.solver.ProductionTriggerBinding

---

::: phydrax.solver.StreamingMomentPlan

---

::: phydrax.applications.incompressible_flow.PeriodicModalTurbulenceStatisticsPlan

---

::: phydrax.applications.incompressible_flow.SpectralChannelStatisticsPlan

---

## Route-qualified candidate evidence

`tools/incompressible_flow_qualification.py` owns three independent route labels and
artifact paths:

| Route | Candidate artifact | Required case metrics |
| --- | --- | --- |
| `periodic-spectral` | `benchmarks/incompressible_periodic_qualification.json` | `taylor-green-decay`; `manufactured-forcing-refinement-restart` |
| `spectral-channel` | `benchmarks/incompressible_channel_qualification.json` | `couette`; `poiseuille`; `manufactured-sbdf2-refinement-restart` |
| `mac` | `benchmarks/mac_incompressible_qualification.json` | `periodic-taylor-green`; `stretched-couette`; `stretched-poiseuille`; `full-hybrid-iterative-comparison` |

Generated artifacts bind exact support, input, reference, and configuration
identities to raw metrics, gates, status, and explicit failure/inconclusive reasons.
They always retain `release_ready=false`. Assembly requires passed existing artifacts
and produces an unsigned candidate with `signed=false` and
`CapabilityProfile.released=false`. Neither an artifact path nor the assembled
candidate is a universal DNS or distributed-support claim.

### LES campaign status

`tools/large_eddy_simulation_qualification.py` produces route-exact LES candidate
evidence through the existing `QualificationEvidence`, `QualificationMatrix`,
`SupportTuple`, `CapabilityProfile`, `ReferenceArtifactManifest`, and
`ResolvedRunSpec` contracts. Generated profiles remain candidate/unreleased.
The base incompressible profile is an external release dependency; an LES campaign
cannot manufacture or waive it. No support tuple, signature, release decision, or
artifact status is implied by the producer or a path alone.

## Scalar and variable-density MAC dynamics

::: phydrax.discretization.MACScalarProblem

---

::: phydrax.discretization.PreparedMACScalarTransport

---

::: phydrax.equations.MACBuoyancyLaw

---

::: phydrax.equations.compile_mac_scalar_buoyancy

---

::: phydrax.equations.CompiledMACScalarBuoyancyDynamics

---

::: phydrax.discretization.MACVariableDensityPlan

---

::: phydrax.solver.MACVariableDensityProjectionPlan

---

::: phydrax.equations.compile_mac_variable_density_flow

---

::: phydrax.equations.CompiledMACVariableDensityDynamics

## Implicit, adaptive, and sensitivity execution

Static algebraic MAC LES with a positive coefficient activates frozen
variable-viscosity profiles in `MACIMEXEulerMethod` and `MACSBDF2Method`; those
methods intentionally refuse dynamic continuation. Compiled periodic-uniform dynamic
MAC instead uses `PreparedMACDynamicExplicitMethod`, with projected candidate state,
current combined restriction, and transactional Lagrangian history. See
[MAC and scalar LES](../../guides_large_eddy_simulation.md#mac-and-scalar-les).

::: phydrax.solver.MACHelmholtzSolvePlan

---

::: phydrax.solver.MACIMEXEulerMethod

---

::: phydrax.solver.MACSBDF2Method

---

::: phydrax.solver.MACAdaptiveRolloutPlan

---

::: phydrax.solver.MACFrozenGridReplayPlan

---


### Integrated dynamic, distributed, learned, and unstructured LES

::: phydrax.applications.incompressible_flow.PreparedPeriodicDynamicETDRKMethod

---

::: phydrax.applications.incompressible_flow.PreparedMACDynamicExplicitMethod

---

::: phydrax.applications.incompressible_flow.DistributedPeriodicLESProductionPlan

---

::: phydrax.equations.PeriodicLearnedStressPlan

---

::: phydrax.equations.MACLearnedStressPlan

---

::: phydrax.solver.UnstructuredLowMachLESFixedStepMethod

---

::: phydrax.solver.MACFixedGridSensitivityPlan

---

::: phydrax.solver.MACSegmentedShadowingPlan

## Resolved, distributed, and moving-geometry execution

### Fixed immersed MAC LES

`FixedImmersedMACLESPlan` binds a static masked SGS action and optional attached
equilibrium tangential wall traction to the prescribed-marker pressure constraint.
It is single-device, unit-density, 3-D, stationary, fixed-route/topology, and limited
to periodic/free-slip/symmetry outer boundaries. Its admission reuses the existing
prescribed-marker owner but retains distinct filter/model/geometry/action evidence.
See [Immersed-boundary coupling](../../guides_immersed_boundary.md#fixed-immersed-mac-les).

::: phydrax.applications.incompressible_flow.FixedImmersedMACLESPlan

---

::: phydrax.applications.incompressible_flow.PreparedFixedImmersedMACLES

---

::: phydrax.applications.incompressible_flow.compile_fixed_immersed_mac_les_flow

---

::: phydrax.discretization.MACMarkerTransferPlan

---

::: phydrax.solver.MACImmersedBoundaryProjectionPlan

---

::: phydrax.solver.MACImmersedBoundaryIMEXEulerMethod

---

::: phydrax.solver.MACImmersedBoundarySBDF2Method

---

::: phydrax.equations.MACPenaltyIBCFDEMCouplingPlan

---

::: phydrax.solver.advance_mac_penalty_ib_cfd_dem_window

---

::: phydrax.solver.MACRigidImmersedEulerMethod

---

::: phydrax.solver.MACDeformableImmersedBackwardEulerMethod

---

::: phydrax.discretization.MACDistributedTopologyPlan

---

::: phydrax.solver.MACDistributedProjectionPlan

---

::: phydrax.discretization.MappedMACGeometryPlan

---

::: phydrax.solver.MACALEGeometryPlan

---

::: phydrax.solver.MACRemeshEpochPlan

## Marker-flow stage, mechanics, and topology closure

::: phydrax.solver.MACHelmholtzStageInverseMomentum

---

::: phydrax.solver.MACVariableDensityStageInverseMomentum

---

::: phydrax.solver.MACVariableViscosityStagePlan

---

::: phydrax.solver.MACRigidImmersedBackwardEulerMethod

---

::: phydrax.solver.MACRigidImmersedMidpointMethod

---

::: phydrax.solver.MACRigidImmersedContactMethod

---

::: phydrax.solver.MACRigidImmersedJointMethod

---

::: phydrax.solver.MACDeformableImmersedNewmarkMethod

---

::: phydrax.solver.DeformableContactResidualPlan

---

::: phydrax.discretization.ResolvedLubricationCorrectionPlan

---

::: phydrax.discretization.CompositeMACMarkerTransferPlan

---

::: phydrax.solver.CompositeMACProjectionPlan

---

::: phydrax.discretization.DistributedMACMarkerTransfer

---

::: phydrax.discretization.MarkerEpochTransferPlan

## Divergence-free, sharp, and stochastic families

::: phydrax.solver.MACDivergenceFreeMarkerTransfer

---

::: phydrax.solver.MACDFIBProjectionPlan

---

::: phydrax.solver.MACSharpInterfaceProjectionPlan

---

::: phydrax.solver.MACSharpOperatorEvidence

---

::: phydrax.solver.MACMovingSharpInterfaceEpochPlan

---

::: phydrax.solver.MACPassiveTracerContinuationState

---

::: phydrax.solver.MACPassiveTracerFixedStepMethod

---

::: phydrax.solver.MACImmersedInterfaceProjectionPlan

---

::: phydrax.solver.MACInterfaceMethodSelector

---

::: phydrax.solver.MACDiscreteStochasticStressPlan

---

::: phydrax.solver.MACFluctuatingHydrodynamicsPlan

---

::: phydrax.solver.MACInertialStochasticStepPlan

---

::: phydrax.solver.FIBOverdampedPlan

## Marker-flow runtime and evidence

::: phydrax.solver.MarkerFlowCheckpointPlan

---

::: phydrax.solver.MarkerFlowReplayPlan

---

::: phydrax.solver.MarkerFlowOutputPlan

---

::: phydrax.solver.HydrodynamicLoadPlan

---

::: phydrax.solver.MarkerFlowAdaptiveStepPlan

---

::: phydrax.solver.MarkerFlowTrajectoryAdapter

---

::: phydrax.solver.MarkerFlowCompiledExportPlan

---

::: phydrax.solver.MarkerFlowQualificationPlan

## Immersed candidate profile and admission

`ImmersedDNSQualificationProfile` is an unsigned, unreleased candidate covering six
exact owner regimes: prescribed MAC markers, free rigid MAC markers, fixed-topology
sharp MAC, deformable/contact MAC, LBM body forcing, and resolved CFD–DEM. These
support tuples are not interchangeable. In particular, sharp and LBM-body routes do
not admit distributed execution, contact changes derivative scope, and owner-computes
marker reductions do not imply a general distributed DNS profile.

`ImmersedBodyRegimePlan` binds an already prepared numerical owner to marker, geometry,
route, motion, and topology epochs. `ImmersedReferenceCampaignPlan` consumes measured
reference evidence without executing another solve. `ImmersedRuntimeAdmissionPlan`
then has a two-phase `prepare(preflight)` / `prepared.admit(runtime)` boundary for
qualification, rank, conditioning, resources, derivative mode, ownership, epochs,
support truncation, distributed reduction, gap regime, sharp certificate, and load
provenance. Admission does not change or retry the owner route.

::: phydrax.applications.incompressible_flow.ImmersedDNSQualificationProfile

---

::: phydrax.applications.incompressible_flow.ImmersedBodyRegimePlan

---

::: phydrax.applications.incompressible_flow.ImmersedReferenceCampaignPlan

---

::: phydrax.applications.incompressible_flow.ImmersedRuntimeAdmissionPlan

---

::: phydrax.applications.incompressible_flow.PreparedImmersedRuntimeAdmission

---

::: phydrax.applications.incompressible_flow.ImmersedRuntimeAdmissionResult
