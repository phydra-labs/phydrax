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


## Spectral viscosity and Courant policy

`compile_periodic_incompressible_flow(..., spectral_viscosity=plan)` binds the
[spectral viscosity operator](../discretization/spectral.md#spectral-vanishing-viscosity)
into the diagonal linear term, so ETDRK integrates molecular and numerical decay
exactly. SVV cannot be composed with algebraic or dynamic LES. Compilation,
linear, nonlinear, drift, bundle, and method identities are unchanged when the
option is absent; an active option contributes its prepared scientific identity.

Stage records retain molecular and SVV rates separately. Absent SVV contributes
`spectral_viscosity_rate=None`, with no modal zero array in compiled stages.
Diagnostics always include zero-valued `spectral_viscosity_energy_rate` and
`spectral_viscosity_dissipation` when absent and a static `spectral_viscosity_id`.
The energy defect uses forcing minus molecular, SGS, and SVV dissipation.

DNS `advective_courant_restriction(state, maximum_courant_number=C_max)` computes
f=max_native_grid Σ_a |u_a|/(L_a/N_a), with bound C_max/f (infinity for zero f).
`CourantGuardedETDRKMethod(ETDRKMethod(4), maximum_courant_number=C_max)` admits
0<C_max≤2√2/π, the classical RK4 frozen-uniform-advection reference.
It is **not** a nonlinear stability certificate. LES and ETDRK2 are refused.
`PeriodicAdvectiveCourantRestriction` requires an explicit non-empty canonical
`compilation_id`, binding the evidence to its compiled dynamics. Its
`reference_model="frozen_uniform_advection_rk4"` and
`nonlinear_stability_certified=False` are fixed by the equation owner, not
constructor options. The same owner validates C_max for both dynamics and
the guard: strings and booleans raise `TypeError`, while nonfinite or
out-of-range values raise `ValueError`.
Preparation requires velocity-compatible Hermitian coordinates; every transition
rejects above-bound or nonfinite evidence without changing the accepted state.
The existing LES guard uses the same transactional acceptance helper and retains
its original identities and numerical semantics.

::: phydrax.equations.PeriodicAdvectiveCourantRestriction

::: phydrax.solver.CourantGuardedETDRKMethod

::: phydrax.solver.PreparedCourantGuardedETDRKMethod

## Large-eddy simulation

Periodic static algebraic LES compiles into the named incompressible stage and uses
a current-state ETDRK guard. Periodic dynamic LES compiles with its exact coarser
test discretization and gains transactional ETDRK production; periodic-uniform MAC
dynamic LES compiles with a projected explicit method.

Distributed slab/pencil full-flow production is scientific-owner-bound. Its
full-complex FFT plan derives the exact `SpectralPrecisionPolicy`, admits only scalar,
three-vector, and 3-by-3 tensor payloads, and records separate `numerical_id`,
`execution_id`, and owner-bound `plan_id`. FFT logical storage/workspace/traffic,
LES closure and method liveness, and production checkpoint capacity retain separate
owners. Distributed restart requires the exact execution identity; one-device parity
does not establish physical multi-device support or scaling.

Channel mixed traction, MAC stochastic inflow, KSGS variants, learned stress,
pressure-stepped unstructured flow, Favre transport, and immersed execution retain
their route-specific contracts. The normative
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
equation and reports conservative full-complex energy, molecular and spectral
viscosity dissipation, advective transfer, SGS transfer, forcing injection,
resolved spectral flux, derived scales/tails, SGS evidence, and current step limits.
`evaluate(time, velocity, args, *, stage=..., additive_forcing_rate=...,
step_restriction=...)` rejects foreign stages or restrictions.
SVV shells, total and volume-normalized mean dissipation are always present and
zero when unavailable; `spectral_viscosity_available` and `spectral_viscosity_id`
declare the operator binding. `numerical_dissipation_fraction` is SVV dissipation
divided by molecular + SVV + resolved SGS dissipation (SGS is minus its energy
transfer). A nonpositive denominator yields zero with
`numerical_dissipation_fraction_valid=False`. Taylor/Kolmogorov validity is false
whenever SVV is active. The production vector stores seven shell arrays followed
by 54 scalars; the added scalars are SVV total, mean, and numerical fraction, after
molecular total/mean. Availability and fraction validity remain snapshot evidence,
not extra vector slots. Periodic manifests bind the evaluator ID and value shape
through the runtime inventory's `program` identity. A changed shell layout or
evaluator payload raises `StaleRuntimeCheckpointError` with `roles=("program",)`
before any checkpoint arrays are decoded. Manifests, topology/geometry digests,
stores, and production-plan identities change at this cutover; checkpoints without
the inventory are also refused with typed stale-role evidence. There is no
old-layout decoder or compatibility shim.
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
No-LES production also accepts `PreparedCourantGuardedETDRKMethod`, retaining its
guarded execution route. Courant guards are explicitly incompatible with OU
forcing and constant-power adapter wiring; compiled forcing does not drop a guard.
`prepare_constant_power_periodic_method(method: PreparedETDRKMethod, forcing, /)`
accepts only plain ETDRK. `PreparedOUForcedETDRKMethod(base, forcing, realization, /)`
and `prepare_ou_forced_periodic_method(method, forcing, realization, /)` accept
`PreparedETDRKMethod | PreparedLESStabilityGuardedETDRKMethod`. Unsupported guarded
adapter variants raise `TypeError`, including explicit Courant refusals.
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

`tools/incompressible_flow_qualification.py` separates independent route labels and
candidate artifact paths:

| Route | Candidate artifact | Required case metrics |
| --- | --- | --- |
| `periodic-spectral` | `benchmarks/incompressible_periodic_qualification.json` | `taylor-green-decay`; `manufactured-forcing-refinement-restart` |
| `periodic-spectral-vanishing-viscosity` | `benchmarks/incompressible_svv_qualification.json` | `shear-decay-2d`; `shear-decay-3d`; `energy-identity-2d`; `energy-identity-3d`; `fixed-low-mode-vanishing` |
| `periodic-statistical-solution` | `benchmarks/incompressible_statistical_qualification.json` | `coupled-campaign`: default 16/32/64 nested grids; paired-jackknife significant W1 decrease, consistent non-abstained scaling, runtime gates; per-resolution dynamics/method/ensemble/discretization/SVV provenance |
| `spectral-channel` | `benchmarks/incompressible_channel_qualification.json` | `couette`; `poiseuille`; `manufactured-sbdf2-refinement-restart` |
| `mac` | `benchmarks/mac_incompressible_qualification.json` | `periodic-taylor-green`; `stretched-couette`; `stretched-poiseuille`; `full-hybrid-iterative-comparison` |

Generated artifacts bind exact support, input, reference, and configuration
identities to raw metrics, gates, status, and explicit failure/inconclusive reasons.
They always retain `release_ready=false`. Assembly requires passed existing artifacts
and produces an unsigned candidate with `signed=false` and
`CapabilityProfile.released=false`. Neither an artifact path nor the assembled
candidate is a universal DNS or distributed-support claim.

### Spectral viscosity operator qualification

The SVV route runs unforced inviscid single-mode solutions in both two and three
dimensions with ordinary ETDRK4. An independent oracle computes `K=N/2-1` for
even mode counts and `K=(N-1)/2` for odd counts directly from `N` and evaluates
the declared plan's analytic rate. It chooses the smallest retained mode `n`
with a negative **oracle** rate, not from the prepared decay array, and a unit
polarization `a` perpendicular to `n`. Thus `u(0)=a*cos(n.x)` is an exact steady
Euler solution even for diagonal modes. Gates independently check the prepared
resolution equals `K` and the prepared rate equals the oracle before comparing
with `u(t)=exp(lambda_n*t)u(0)`.
Initial and final diagnostics are compared with
`E(t)=(2*pi)^d exp(2*lambda_n*t)/4` and
`D_SVV(t)=-2*lambda_n*E(t)`. Gates cover velocity error, divergence, successful
integration, positive numerical dissipation, energy/rate agreement, finiteness,
and the semidiscrete energy-balance defect.

The fixed-low-mode case compares one retained integer mode across at least three
strictly increasing resolutions. It records the actual and analytic rates,
monotone decay magnitudes, and the vanishing bound
`|lambda_n| <= strength*K^(1-2*order)*|n|^(2*order)` on this `2*pi` box.
These are finite-campaign operator checks, not solution-convergence evidence.

Every case, including each low-mode resolution row, reports
`operator_scaling_admissible`, `cited_convergence_supported=false`, and
`support_reason`. Only `tadmor_step` with the cited strict cutoff scaling can
be operator-admissible. `exponential_highpass` is practical exponential high-pass
hyperviscosity, not the cited theorem's operator. Neither profile certifies the
initial-data mollification or full-method hypotheses for rough-data convergence.
The scaling reference is [Rohner–Mishra (2024), Section 2](https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2024/2024-04_rev1.pdf).
Strength, order, cutoff scale, cutoff exponent, and kernel must be explicitly
supplied; the exponential profile additionally requires its high-pass exponent.

For example, this declared test configuration produces an unreleased artifact:

```bash
.venv/bin/python -m tools.incompressible_flow_qualification \
  periodic-spectral-vanishing-viscosity \
  --strength 0.1 --order 1 --cutoff-scale 0.5 --cutoff-exponent 0.25 \
  --kernel tadmor_step --mode-count 12 --low-mode-counts 12 24 48 \
  --output benchmarks/incompressible_svv_qualification.json
```

The displayed values are campaign inputs, not universal scientific defaults.
Passing these gates cannot release support or certify nonlinear stability.

### Finite statistical refinement qualification

The statistical route requires at least three nested resolutions (default
`16 32 64`, coarsest count greater than eight). Consecutive coupled ensembles
use the same ordered members, final save time, and each pair's coarse injection
nodes. Componentwise empirical W1 distances are comparable spatial integrals
over the same physical domain, with the same component-mean convention and
domain quadrature measure; no fine retained modes are truncated for sampling.
Deleting the same member at every resolution gives jackknife SEs of each
distance `d_j` and, using the paired deletions, each `d_j-d_(j+1)`.
The declared threshold `z` defaults to two: every consecutive difference above
`z*SE` passes the refinement gate, any difference below `-z*SE` fails, and
otherwise refinement is inconclusive.
Inconsistent scaling or any abstained scaling fit is inconclusive and cannot
yield a passed route. Runtime failures return guard/invariant evidence without
statistical metrics; finite metric evidence is also a gate.

Inputs bind measure/coupling IDs and per-resolution compilation, prepared
temporal method, prepared/result ensemble, discretization, and SVV plan/prepared
IDs. Support binds kernel, two spatial dimensions, `fourier-pseudospectral`,
`courant-guarded-etdrk4`, and `convergence_certified=false`. The power-law exponent
is a declared parameter: the family is motivated by Herde et al. (2024),
Appendix B.2.2, Eq. (42); LMPP (arXiv:1909.06615) motivates only metrics and
structure functions. Passing finite-campaign gates is not a continuum claim.

For example:

```bash
.venv/bin/python -m tools.incompressible_flow_qualification \
  periodic-statistical-solution \
  --strength 0.1 --order 1 --cutoff-scale 0.5 --cutoff-exponent 0.25 \
  --kernel tadmor_step --power-law-exponent 1.5 --velocity-scale 0.02 \
  --maximum-courant-number 0.5 --mode-counts 16 32 64 --members 6 --steps 2 \
  --refinement-z 2 \
  --output benchmarks/incompressible_statistical_qualification.json
```

### LES campaign status

`tools/large_eddy_simulation_qualification.py` produces route-exact LES candidate
evidence through the existing `QualificationEvidence`, `QualificationMatrix`,
`SupportTuple`, `CapabilityProfile`, `ReferenceArtifactManifest`, and
`ResolvedRunSpec` contracts. Generated profiles remain candidate/unreleased.
The base incompressible profile is an external release dependency; an LES campaign
cannot manufacture or waive it. No support tuple, signature, release decision, or
artifact status is implied by the producer or a path alone.

### Distributed spectral dependency

Distributed periodic LES depends on the separate unreleased
`distributed-spectral-execution` capability. Its exact candidate support is
full-complex C2C JAX global arrays on regular divisible slab/pencil shards and the
current horizontal Fourier--Chebyshev--Fourier channel action. The candidate gates
independent forward/inverse, JVP/adjoint, precision, payload, ordered-stage,
process-qualified topology, no-gather, resource/compiler-memory, physical-device, and
separate multi-host evidence. It makes no R2C/C2R, vendor/rank-local provider, uneven
shard, mixed-transform, dynamic-scheduling, or implicit-gather claim.

Forced-device and same-host evidence is inconclusive for the physical gates, and the
phase-separated `benchmarks/distributed_spectral.py` driver is performance evidence,
not release authority. LES also retains its own scientific, workflow-resource, and
exact-restart qualification boundaries.

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
::: phydrax.solver.MACEnthalpyPorosityIMEXEulerMethod

---

::: phydrax.solver.MACEnthalpyPorositySBDF2Method

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

## Volume-fraction immersed CFD–DEM

This first-order direct-forcing route is distinct from regularized marker
penalty coupling and sharp cut-cell interface traction. Its candidate
`volume-fraction-cfd-dem` support tuple is not part of the existing six-regime
immersed DNS reference portfolio.

`MACSDFCutFractionPlan` binds canonical prepared rigid-shape signed queries
to fixed cell/subbox/query capacities. It returns per-body estimates, union
lower/upper bounds, an explicitly bounded midpoint estimate, unresolved
fraction, sign reliability, domain support, and overflow evidence. The
unresolved threshold applies **per cell**. Unsupported active queries,
superquadrics without certified bounds, capacity overflow, and uncontained
bodies are not silently repaired. Periodic fluid boundaries do not imply
periodic posed-shape images.

`MACVolumeFractionIBCFDEMCouplingPlan` requires identical MAC operator
identities for its fluid, geometry, and projection owners, positive scalar
intrinsic densities, and solid density consistent with active DEM mass/volume.
Mixture inertia is \(\rho_m=(1-\phi)\rho_f+\phi\rho_s\) and solid mass
fraction is \(\psi=\phi\rho_s/\rho_m\). Centered geometric face fractions
define direct forcing
\(\mathbf f_{\rm IB}=\rho_m\psi(\mathbf u_s-\mathbf u^*)/\Delta t\).
Normalized per-body shares partition overlaps. Body load density is
\(\alpha_b[\psi\mathbf b_\sigma-(1-\psi)\mathbf f_{\rm IB}]\).
The load/rigid-velocity pullback uses actual face locations and face-dual
measures, retaining discrete force, torque, and work identities while
approximating volume integration.

`MACVolumeFractionIBEvaluation` separates pressure/viscous loads from direct
reaction. Its impulse residual balances **external-fluid**
\((1-\psi)\mathbf f_{\rm IB}\), not total mixture forcing; the internal
impulse is separately reported. Torque residual is about the world origin,
while body torque is about its COM.

`MACVolumeFractionIBState.initialize` accepts the fluid owner's native packed
density/momentum state and optional pressure. An advance builds unprojected
native transport/viscous prediction including the accepted pressure gradient,
evaluates direct forcing, performs **one** variable-density pressure-increment
projection, and evaluates final total-pressure/viscous loads. It then
advances native compliant DEM, and admits candidate-pose geometry. Accepted
fluid coordinates use candidate geometric mixture density and preserve the
projected physical velocity through native donor-density momentum. Geometry
owns next-step density; the miscible-density transport result does not replace
it. Fluid, DEM, pressure, impulse ledgers, and counters commit atomically.

Invalid step size, geometry/capacity/support, transport, force/torque/work
identity, projection, DEM, or candidate-pose admission failure retains the
entire previous accepted state. `MACVolumeFractionIBStepResult` preserves
candidate state, `evaluation`, `pressure_projection`, `dem_step`,
`candidate_cut_fraction`, and status. No retries or capacity growth are
performed. See the [immersed-boundary guide](../../guides_immersed_boundary.md#volume-fraction-direct-forcing-cfddem)
for the discrete formulas, reference attribution, real sphere-settling
candidate tolerance, and limitations.

::: phydrax.equations.MACSDFCutFractionPlan

---

::: phydrax.equations.MACSDFCutFractionEvaluation

---

::: phydrax.equations.MACVolumeFractionIBCFDEMCouplingPlan

---

::: phydrax.equations.MACVolumeFractionIBEvaluation

---

::: phydrax.equations.MACVolumeFractionIBStatus

---

::: phydrax.equations.evaluate_mac_volume_fraction_ib_cfd_dem

---

::: phydrax.solver.MACVolumeFractionIBState

---

::: phydrax.solver.MACVolumeFractionIBStepResult

---

::: phydrax.solver.MACVolumeFractionIBStepStatus

---

::: phydrax.solver.advance_mac_volume_fraction_ib_cfd_dem
## Periodic incompressible initial measures

Families synthesize orthonormal modal coefficients on periodic Fourier boxes.
Taylor–Green is direct modal synthesis; patch and cell vorticities use analytic
integrals with explicit mean neutralization. A patch is a circular patch **plus
a neutralizing uniform background**, not an isolated nonzero-mean torus vortex.
The displaced shear layer combines the layer profile from
[Lanthaler–Mishra–Parés-Pulido §5.2, Eq. (42)](https://arxiv.org/abs/1909.06615)
with the mode-weighted displacement law from
[Poseidon Appendix B.2.3, Eq. (45)](https://arxiv.org/html/2405.19101v2#A2.SS2.SSS3).
It is not LMP Eq. (43)'s single-frequency displacement law, nor equivalent to
Poseidon Eq. (44)'s layer (whose tanh argument omits displacement).
The profile is evaluated at `mod(y + sigma(x), 1)` before both the branch
comparison and tanh argument, defining its periodic extension on the torus.
Periodicity requires no displacement bound. Thickness remains strictly positive
for smooth layers, and perturbation amplitude nonnegative as the declared
amplitude law. An explicit oversampling policy and separate discarded-tail/
Leray-projection energies accompany the hybrid family.

The sinusoidal sheet follows Poseidon Eqs. (46)–(50): vectorized host
Gauss–Legendre radial Hankel quadrature, split at the mollifier's polynomial
break, resolves the largest retained |k|ρ and checks normalization. A bounded
type-1 NUFFT computes
the **exact discrete sum over trapezoid quadrature**, never exact continuum
coefficients. Each state reports its Q-versus-2Q residual (not a certified
continuum error bound). Point quadrature and member workspaces are refused
before execution when the declared byte cap cannot accommodate them.

Member parameters depend on the measure identity and member index, not batch
or chunk size. Resource policies have a distinct preparation identity.
`restrict_periodic_flow_initial_state(state, target, maximum_bytes=...)` is the
single restriction API: canonical retained-band gather plus target Nyquist
zeroing, without reprojection. Nested restrictions are bitwise transitive.
Scientific identities and root-key/coupling identities persist across
restrictions. Byte-budget refusals raise `MemoryBudgetError` with
`budget="maximum_bytes"`, required bytes, and limit bytes; wrong scalar kinds
(including booleans and strings) raise `TypeError`.

`velocity_from_vorticity` refuses nonfinite input, nonzero mean, and 3D
longitudinal vorticity with traced `eqx.error_if` validation, including under
JIT. For `curl(u)=∂x u_y−∂y u_x`, its 2D convention is
`psi_hat=omega_hat/|k|²`, `u_hat=(i k_y,−i k_x)*psi_hat`.
Planar embedding scales coefficients by `sqrt(L_invariant)`: integrated energy
scales by invariant length, while energy per unit length and planar dynamics
are preserved. Embedded measure and coupling identities depend on the source
measure, invariant axis/domain/length, and zero-mode embedding policy, never on
the target resolution. `restrict_periodic_flow_initial_state` owns retained-band
restriction of embedded states (and ordinary initial states), preserving those
identities so coupled multi-resolution planar 3D campaigns can align.

::: phydrax.discretization.spectral.PeriodicLerayProjector.velocity_from_vorticity

::: phydrax.discretization.spectral.PeriodicLerayProjector.vorticity_from_velocity

::: phydrax.applications.incompressible_flow.TaylorGreenFamily

::: phydrax.applications.incompressible_flow.ShearLayerProfile

::: phydrax.applications.incompressible_flow.ShearLayerFamily

::: phydrax.applications.incompressible_flow.VortexPatchFamily

::: phydrax.applications.incompressible_flow.PiecewiseConstantVorticityFamily

::: phydrax.applications.incompressible_flow.SinusoidalVortexSheetFamily

::: phydrax.applications.incompressible_flow.PowerLawVelocityFamily

::: phydrax.applications.incompressible_flow.PeriodicFlowFamily

::: phydrax.applications.incompressible_flow.PeriodicFlowInitialMeasure

::: phydrax.applications.incompressible_flow.PreparedPeriodicFlowInitialMeasure

::: phydrax.applications.incompressible_flow.PeriodicFlowInitialState

::: phydrax.applications.incompressible_flow.embed_planar_velocity

::: phydrax.applications.incompressible_flow.restrict_periodic_flow_initial_state

## Statistical periodic-flow evidence

The [statistical-flow guide](../../guides_statistical_incompressible_flow.md)
connects initial measures, SVV, Courant-guarded ETDRK4, the solver-owned
fixed-step ensemble, radial increment moments, and grouped operator data.
These APIs report finite-campaign evidence, not convergence, compactness, or
exact multidimensional correlation-marginal Wasserstein distance.

`prepare_periodic_flow_ensemble(...) -> PreparedPeriodicFlowEnsemble` binds the
native prepared executable to its initial measure; `.execute()` returns a
`PeriodicFlowEnsembleResult`. That result validates scientific member path labels
and the problem's discretization identity, not shape coincidence.
`CoupledEnsemblePair` validates measure/coupling identities, ordered paths, member
indices, weights and saves on construction. Alignment preserves the original
owners for identity permutations and otherwise gathers the fine solutions once.

Metrics alias-sum fine modes onto the coarse lattice before the coarse inverse
transform, exactly reproducing nested fine-grid injection without reconstructing
the fine physical grid or truncating its unresolved modes. All three accept
`member_chunk_size: int = 8` and `maximum_workspace_bytes: int = 2**28`, admitting
live shared-node, modal/transform and transport buffers with `MemoryBudgetError`.
The sliced metric uses this common cap, not a tuple-only cap. Componentwise
Wasserstein is exact only for scalar marginals; paired norm cost is a coupling
upper bound; sliced distances report projection Monte Carlo uncertainty.

Scaling uses a declared resolved radius window and paired member jackknife
replicates across radii and resolutions. Fit status and abstention reason own
validity; a reported exponent is not a theorem certificate. Retained-band
dataset transfer is a separate observation policy, not statistical metric
alignment. `periodic_flow_operator_dataset(result: PeriodicFlowEnsembleResult, /,
..., member_chunk_size: int = 8, maximum_dataset_bytes: int = 2**31)` owns chunked
conversion with retained-plus-transient byte admission. Vorticity uses the
projector-owned curl, inverse to Biot--Savart for admissible mean-zero solenoidal
velocities. Leakage grouping is coupling/member-based: it shares groups across
coupled resolutions but distinguishes different measures using the same key.

::: phydrax.applications.incompressible_flow.prepare_periodic_flow_ensemble

::: phydrax.applications.incompressible_flow.PreparedPeriodicFlowEnsemble

::: phydrax.applications.incompressible_flow.PeriodicFlowEnsembleResult

::: phydrax.applications.incompressible_flow.CoupledEnsemblePair

::: phydrax.applications.incompressible_flow.align_coupled_ensembles

::: phydrax.applications.incompressible_flow.componentwise_empirical_wasserstein

::: phydrax.applications.incompressible_flow.paired_coupling_upper_bound

::: phydrax.applications.incompressible_flow.sliced_joint_marginal_distance

::: phydrax.applications.incompressible_flow.StructureFunctionScalingPlan

::: phydrax.applications.incompressible_flow.ComponentwiseEmpiricalWassersteinResult

::: phydrax.applications.incompressible_flow.PairedCouplingUpperBoundResult

::: phydrax.applications.incompressible_flow.SlicedJointMarginalDistanceResult

::: phydrax.applications.incompressible_flow.ScalingEvidenceStatus

::: phydrax.applications.incompressible_flow.StructureFunctionScalingFit

::: phydrax.applications.incompressible_flow.InstantaneousStructureFunctionScalingEvidence

Time quadrature returns a separate finite time-integrated evidence record,
retaining the explicit weights rather than labeling an integrated result as an
instantaneous snapshot.

::: phydrax.applications.incompressible_flow.TimeIntegratedStructureFunctionScalingEvidence

## Periodic-flow operator data

`periodic_flow_operator_dataset` produces member-major input/target cases on a
declared observation grid. Its `initial` input is the selected input-time field;
`conditions` supplies scalar lead time; queries and targets use the selected
`velocity` or `vorticity` field. Coarser observation grids use modal retained-band
restriction before reconstruction, not physical-node injection. Only selected
saves must be valid and finite; selected invalid saves are refused and target
times must be later than the input time. Dataset byte admission precedes
materialization. Use the default grouped
`phydrax.nn.operator.training.split_operator_dataset` policy to keep every saved
time and coupled resolution of one realization in the same partition.

::: phydrax.applications.incompressible_flow.PeriodicFlowDatasetField

::: phydrax.applications.incompressible_flow.periodic_flow_operator_dataset
