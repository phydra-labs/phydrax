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

## Structured finite-volume MAC dynamics and projection

The constant-density projection supports iterative, uniform tensor-transform, and
hybrid transform-line routes. Hybrid pressure is eligible only for a three-dimensional
all-Neumann closure with one explicit nonperiodic physical line and two uniform
transform-compatible transverse axes. Transverse analysis precedes the batched
nonuniform tridiagonal line solves; synthesis is applied in reverse axis order.
Preparation certifies physical-action identity and resource capacity.

The all-zero transverse mode is volume-compatibility projected, uses one pinned row
only in the factorization, and returns a volume-zero-mean pressure. A runtime
inverse-momentum diagonal, variable density, mixed/open closure, or ineligible tensor
uses the iterative route. There is no distributed hybrid line and no MAC
fixed-bulk-flux controller.


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

::: phydrax.equations.MACStepRestriction

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

`PeriodicModalTurbulenceStatisticsPlan` provides conservative full-complex energy,
dissipation, nonlinear-transfer, and forcing-injection shells with explicit validity
for derived scales and tails. `SpectralChannelStatisticsPlan` provides
homogeneous-plane raw/central moments and separate-wall shear/friction semantics.
`StreamingMomentPlan` places those or other real-valued observables in sample- or
time-weighted windows with optional fixed-capacity completed blocks.

`PeriodicSpectralProductionPlan` verifies matching prepared ETDRK coordinates and
statistics. Compiled constant-power wiring verifies the forcing identity; adapter
wiring adds forcing and therefore requires an otherwise unforced drift. Mutually
exclusive OU forcing and realization inputs produce a
`PreparedOUForcedETDRKMethod` whose accepted state transactionally couples velocity
and exact OU coefficient continuation. `SpectralChannelProductionPlan` derives the
exact step from prepared SBDF2 and requires its horizon, outputs, and statistics
window on that lattice.
`StructuredMACProductionPlan` binds a fixed-step method, compiled MAC dynamics, and
`MACPlaneWallStatisticsPlan`; optional `MACConstantPressureGradientForcing` must
already be compiled with the same identity and is not feedback control. Each route
prepares a checkpoint root before initialization.

::: phydrax.applications.incompressible_flow.PeriodicSpectralProductionPlan

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

::: phydrax.solver.MACFixedGridSensitivityPlan

---

::: phydrax.solver.MACSegmentedShadowingPlan

## Resolved, distributed, and moving-geometry execution

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
