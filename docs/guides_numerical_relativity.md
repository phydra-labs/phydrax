# Numerical relativity, horizons, and radiation

`phydrax.applications.numerical_relativity` composes metrix geometry, physical
equations, solver runtimes, spherical transforms, block AMR, lifecycle, and execution
owners. It supplies no monolithic simulation object. Plans fix topology, capacities,
conventions, tolerances, and numerical policies; results retain candidate and accepted
states plus orthogonal numerical/scientific evidence.

## Z4c state and equations

`Z4cState` is the canonical 25-channel component-first state:
conformal factor, six conformal-metric components, trace extrinsic curvature, six
trace-free conformal-extrinsic-curvature components, three conformal connections,
constraint variable $\Theta$, lapse, three shifts, and three shift-driver components.
Symmetric tensors use `xx, xy, xz, yy, yz, zz` ordering.

`Z4cSystem` implements the canonical mostly-plus conformal Z4c method-of-lines equations
with explicit constraint damping and matter coupling. It requires the shared
$R^\rho{}_{\sigma\mu\nu}=+\partial_\mu\Gamma^\rho{}_{\nu\sigma}-\cdots$ and
$K_{ij}=-\tfrac12\mathcal L_n\gamma_{ij}$ convention. `evaluate_z4c_rhs` returns the
rates, `ADMGridGeometry`, and Hamiltonian, momentum, connection, determinant,
trace-free, and $\Theta$ constraint evidence. `vacuum_z4c_rhs` is the explicit
source-free path; it does not manufacture a matter record.

`FourthOrderDerivatives` owns fourth-order centered first/second derivatives,
velocity-sign-selected fourth-order upwind advection, and sixth-derivative
Kreiss--Oliger damping on a uniform `FixedGridGeometry`. Boundaries are either periodic
or explicit one-sided stencils. At least five periodic or six one-sided nodes are
required per spatial axis. `GeodesicGauge`, `HarmonicGauge`, and
`MovingPunctureGauge` are distinct gauge plans. `PeriodicBoundary`,
`AnalyticBoundary`, and `CharacteristicRadiativeBoundary` are distinct outer-boundary
policies. `Z4cAlgebraicEnforcement` projects $\det\tilde\gamma=1$ and
$\operatorname{tr}\tilde A=0$ only at the accepted-step boundary and reports its
correction.

## Initial data and fixed-grid evolution

The analytic initial-data surface contains Cartesian Minkowski, isotropic exterior
Schwarzschild, and horizon-penetrating Cartesian Kerr--Schild slices. Constraint and
finite-surface ADM charge diagnostics use pointwise automatic differentiation and
explicit surface quadrature. Bowen--York and Brill--Lindquist puncture data and the
fixed-resolution tensor-Chebyshev `TwoPunctureHamiltonianPlan` add nonlinear
Hamiltonian correction and tuning evidence; they do not stand in for arbitrary
elliptic initial-data families.

`FixedGridZ4cRuntime` performs fixed-grid SSPRK(3,3) or the declared fixed-step
integrator without topology changes. `evaluate_z4c_step` produces an uncommitted
candidate and all evidence. `accept_z4c_step` commits or rolls back atomically. The
`NumericalRelativityStatus` is a composable bitset: `NONFINITE_STATE`,
`NONPOSITIVE_LAPSE`, `NONPOSITIVE_CONFORMAL_FACTOR`,
`SINGULAR_CONFORMAL_METRIC`, `CONSTRAINT_TOLERANCE_EXCEEDED`,
`BOUNDARY_FAILURE`, `ENFORCEMENT_FAILURE`, `DERIVATIVE_INVALID`,
`TIME_GRID_MISMATCH`, `SOURCE_INVALID`, and `STEP_REJECTED`. `SUCCESS` is zero.
`ScientificStatus` keeps that numerical status orthogonal to finite, converged,
physical, qualification, and derivative predicates.

## Matter coupling

`Z4cMatterCoupledRuntime` coordinates Z4c with GRHD, GRMHD, or GRRMHD proposals
through matching stage callbacks. `GRRMHDZ4cStageAdapter` obtains each stage geometry,
combines material and radiation stress-energy, and returns one atomic GRRMHD proposal.
Every exchange carries one static `geometry_lineage_id` and the exact stage
`snapshot_token`; a projection from another stage fails compatibility.

`SourceExchangeLedger`, `ConstraintLedger`, `ConservationLedger`, `FloorLedger`, and
`HorizonFluxLedger` remain separate. `CoupledBudget` accumulates accepted work only;
rejected attempts never enter it. The authoritative `CoupledEvolutionState` commits
the spacetime/matter pair all-or-nothing and records consecutive-failure and terminal
policies.

`CoupledEvolutionStatus` independently records `INVALID_STEP`,
`STAGE_IDENTITY_MISMATCH`, `Z4C_REJECTED`, `MATTER_REJECTED`, `NONFINITE`,
`LEDGER_LIMIT_EXCEEDED`, `FAILURE_LIMIT_REACHED`, `TERMINAL`, and
`DERIVATIVE_INVALID`; multiple bits may be present. Resistive and force-free models
remain explicit outer evolution/transition plans; they are not silently inserted.

## Four different horizon statements

The horizon APIs intentionally prevent one result type from acquiring a stronger name.

1. **Stationary Killing horizon.** `StationaryKillingHorizonResult` belongs to exact
   stationary Kerr thermodynamics in `compact_objects`. It is not produced from an
   evolving slice.
2. **MOTS and apparent horizon.** `MOTSSolvePlan` solves
   $\Theta_{(+)}[r]=0$ on a fixed spherical spectral surface and returns a
   `MOTSSolveResult`, explicitly not a horizon. The result retains residual and
   finite-dimensional stability-operator evidence and deliberately does not claim an
   implicit derivative of the solved surface. `ApparentHorizonSearchPlan` runs every
   fixed seed. An apparent horizon is certified only with explicit complete search,
   resolved exclusions, finite nested candidates, and one enclosing stable
   representative. A failed solve is never evidence that no MOTS exists.
3. **Tracked and isolated/dynamical horizon.** `HorizonTrackerPlan` predicts from the
   last surface and gates time and area jumps; every accepted tracked candidate still
   needs recertification. `QuasilocalHorizonWorldtube` joins qualified marginal slices.
   `DynamicalHorizonBalancePlan` classifies each slice as isolated, dynamical,
   transitional, or invalid using worldtube signature, flux, and area/mass rates and
   audits integrated energy/angular-momentum balance. It is a quasilocal statement,
   never a global event-horizon statement.
4. **Event horizon.** `OfflineEventHorizonTracingPlan` works only on an explicitly
   completed `CompletedSpacetimeHistory`. The history stores the full time-batched ADM
   geometry and does not accept a prescribed outward-normal field. The plan normalizes
   caller-supplied terminal spatial null covectors and evolves both positions and
   covectors backward with the time-dependent 3+1 Hamilton equations and RK4
   step-doubling. It checks chart/support coverage, null residual, position and
   covector convergence, geodesic transport, terminal-surface qualification, and
   caustic/crease entry. The product is a global offline candidate. No live slice,
   MOTS, or tracker is relabeled as an event horizon.

The status families preserve those distinctions. `MOTSStatus` reports success,
nonfinite, nonconverged, nonphysical-surface, or invalid-stability evidence.
`ApparentHorizonSearchStatus` reports invalid, incomplete, no-surface, unique,
multiple, or non-nested search results. `HorizonTrackingStatus` reports an accepted
candidate needing recertification, inactivity, nonmonotone time, MOTS/geometry failure,
or a tracking jump. `DynamicalHorizonStatus` is a bitset for nonfinite/unqualified
slices, failed balance, nonphysical energy flux, timelike worldtube, area decrease,
and invalid derivatives. `EventHorizonStatus` is a separate bitset for nonfinite or
nonconverged flow, missing coverage, nonnull generators, invalid geodesic transport,
caustics, unqualified terminal surface, and invalid derivatives.

## Wave extraction, characteristic evolution, and BMS products

`vacuum_weyl_curvature` produces electric/magnetic Weyl tensors with trace, symmetry,
metric, and convention evidence. `Psi4ExtractionPlan` uses an explicit null-tetrad
convention and spin-$-2$ exact-sampling spherical transform. `FixedFrequencyStrainPlan`
performs the declared twice-integrated fixed-frequency high-pass operation, and
`FiniteRadiusExtrapolationPlan` fits a bounded polynomial in $1/r$. Every operation
retains reconstruction or radial convergence evidence; extraction at one radius is not
future-null-infinity data.

`CharacteristicEvolutionPlan` evolves the linearized Bondi--Sachs hypersurface equation
for $q=rJ$ from a completed finite-radius worldtube to $1/r=0$. At scri the module
uses $h=q$, news $N=\partial_u h$, and $\Psi_4=\partial_u N$ under its declared
convention. It returns those modes, energy flux, radiated energy, and the independent
hypersurface residual. Callers may supply already evaluated nonlinear sources, but the
plan does not claim a nonlinear characteristic field solver. Endpoint one-sided time
derivatives are excluded from the reported derivative mask.

`BMSQuadraturePlan` requires finite unit directions, positive weights summing to
$4\pi$, a declared `supported_bandlimit`, and an analytic `charge_basis_gram`.
The first four charge-basis rows must be $(1,n_x,n_y,n_z)$; preparation checks weighted
$\ell=0,1$ first/second moments and the full declared Gram matrix. Evaluation repeats
those checks with dtype roundoff before computing Bondi four-momentum, Lorentz charges,
sampled supermomenta, and news energy flux from completed `BMSScriData`. The
Lorentz-charge aspect is caller supplied and is not synthesized from the mass aspect.
`BMSFrameTransformation` implements a proper orthochronous Lorentz boost followed by
an origin translation and retains causal, quadrature, frame, and derivative evidence.

`CharacteristicStatus` is a bitset for nonfinite output, unconverged hypersurface,
nonphysical mode set, invalid derivatives, and invalid energy balance. `BMSStatus`
separately records nonfinite values, invalid quadrature, acausal four-momentum, invalid
frame, and invalid derivatives.

## Uncertainty, multifidelity, and learned corrections

`NumericalErrorRecord` and `ModelDiscrepancyRecord` keep numerical error distinct from
model bias and low-rank discrepancy covariance. `RelativisticMultifidelityPlan`
constructs an explicit four-level additive result—baseline, perturbative, ROM, and
full correction—with no implicit substitution when a fidelity is missing or invalid.

`smooth_nr_inverse_adapter` binds fixed-grid, fixed-step parameters without topology or
event selection. `smooth_grhd_inverse_adapter` requires a shock-free trajectory and
fixed recovery/flux branch. Their model evaluators preserve the original scientific
status rather than declaring every JAX trace differentiable.

`LearnedClosureCandidate` is only an additive correction. `admit_learned_closure`
requires independent conservation, admissibility, intended-use rights, support, and
identity evidence; `apply_admitted_learned_closure` retains both native value and
correction and requires the native model to remain present. There is no learned native-
replacement API, hidden model loading, or qualification transfer from training loss.

## Latest block-AMR architecture

Numerical relativity reuses the native host-compiled, fixed-capacity block-AMR
architecture described in [Block AMR](guides_block_amr.md); it does not maintain a
second hierarchy.

- `NumericalRelativityAMRDistributionPlan` layers the exact formulation identity over
  the authoritative Morton-contiguous `BlockAMRPartitionPlan`.
  `PreparedNumericalRelativityAMRDistribution` owns canonical owner-computes packed
  blocks and FillPatch route transposes.
- `NumericalRelativityAMRTopologyEpoch` joins one immutable compiled block epoch to its
  prepared distribution and stable ownership. `NumericalRelativityAMRState` stores
  fixed-capacity fields for exactly one of `z4c`, `grhd`, `grmhd`, `grrmhd`,
  `z4c-grhd`, `z4c-grmhd`, or `z4c-grrmhd`.
- `NumericalRelativityAMRHaloPlan` reuses source-classified `FDAMRFillPatchPlan` for
  Z4c, material, radiation-moment, and oriented magnetic fields. Same-level, periodic,
  coarse-time, and caller-owned physical-boundary sources remain visible.
- `Z4cAMRTransferPlan` applies cell transfer followed by algebraic constraint
  projection. `RelativisticMaterialTransferPlan` audits volume-integrated
  $(D,S_i,\tau)$ conservation. `RelativisticRadiationTransferPlan` does the same for
  densitized M1 energy-momentum. `RelativisticMagneticAMRTransferPlan` preserves and
  audits face-flux divergence through the cochain transfer family.
- `RelativisticMaterialSubcyclingPlan` reuses the N-level AMR schedule and accepted
  conservation ledgers. Material reflux uses the accepted face-flux mismatch;
  `RelativisticMagneticRefluxPlan` applies the accepted edge-EMF mismatch through the
  exact cochain curl.
- `NumericalRelativityAMRTopologyTransition` commits a consecutive compiled epoch only
  if capacity, finiteness, and every formulation-specific transfer record pass.
  Failure, unchanged topology, or capacity overflow retains the predecessor. Field
  shapes, dtypes, and capacities cannot change during commit.

All AMR transfer and topology-transition records currently report
`derivative_valid=False`. Within a frozen prepared epoch the numerical array routes may
be differentiated under their own contracts, but no derivative crosses tagging,
compilation, stable-ID allocation, partition selection, projection/reflux transaction,
or epoch adoption.

## Distributed state and restart

`NumericalRelativityDistributedPlan` provides three-dimensional named sharding for the
same seven formulations. Z4c is component-replicated and spatially sharded; material
and radiation moments are cell-sharded; oriented CT cochains retain separate component
shapes rather than being flattened. `single_device_authority` says only that the
prepared plan has one device; it is not multi-device parity evidence.

`NumericalRelativityRestartState` carries formulation, runtime, geometry, topology,
topology epoch, time/step, exact field names, and the complete typed PyTree. Its
constructors cover Z4c, GRHD, GRMHD, GRRMHD, and coupled states.
`NumericalRelativityCheckpointPlan` additionally binds analysis, numeric revision,
execution, exact CT where applicable, and one `state_template`; the template tree and
every leaf shape/dtype are part of the plan.

Local archives and distributed shards use the existing pickle-free lifecycle owners.
Distributed publication must be durably repository-committed before assembly.
`restore_distributed_numerical_relativity_checkpoint` reconstructs the complete typed
state and runtime arguments directly into the template shardings, validates bounded
reconstruction metadata and all committed shard/rank/artifact identities, applies the
restart relation, and returns `DistributedNumericalRelativityRestart` containing the
typed checkpoint and evidence. Exact and tolerance restart remain separate
`NumericalRelativityRestartPolicy` relations; neither permits topology change.

## Timelike conformal AdS references

The metric conformal Einstein owner is disjoint from Z4c and characteristic
null infinity. It currently provides a full local vacuum zero-quantity
evaluator, exact constant-curvature AdS sign control, generalized-wave gauge
evidence, and Cartesian timelike-boundary/corner evidence. Covariant
derivatives and the unphysical Riemann tensor are explicit inputs with one
source identity; they are not reconstructed silently.

The only evolution path in this support tuple is the fixed-background
Einstein-cylinder scalar reference. Nonlinear metric-conformal evolution,
coupled Einstein–matter backreaction, conformal block AMR, and distributed AdS
execution are not implemented support tuples and must not be inferred from the
zero-quantity or scalar APIs.

Holographic scalar and stress-tensor routines audit caller-declared asymptotic
data. They require exponents, Fefferman–Graham coefficients, counterterms,
normalization, trace target, and divergence. This preserves a strict boundary
between a finite classical bulk calculation and a renormalized boundary-CFT
claim.

## Scope and derivative boundary

Landed capabilities establish bounded implementation and evidence paths for the exact
plans above. They do not by themselves establish continuum convergence for a new case,
long-duration binary-black-hole fidelity, multi-host scaling, astrophysical validity,
or deployment authorization. Fixed-grid and fixed-epoch smooth kernels retain their
reported derivatives. Gauge/boundary selection, nonlinear root branches, apparent-
horizon completeness, horizon regime changes, caustics, extraction masks, topology,
partition, restart relation, failure/rollback, and production admission are discrete
boundaries.