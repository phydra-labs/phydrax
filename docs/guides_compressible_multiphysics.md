# Differentiable compressible multiphysics

Phydrax composes conservative finite-volume transport with fixed temporal programs,
transactional source processes, particle-grid transfer, and compatible magnetic
cochains. The fixed program is the differentiated model: timesteps, topology,
process order, and stochastic realization are explicit and nontrainable.

## Fixed replay

`AdaptiveFiniteVolumeRolloutPlan` records an accepted `RealizedTemporalMesh`.
`AdaptiveBalanceLawRolloutPlan` additionally uses every process step limit and records
each accepted or rejected attempt in a `BalanceLawDecisionJournal`.
`ScheduledFiniteVolumeRolloutPlan` consumes an all-active internal `TemporalMesh` and
attempts every interval exactly. It never accepts a CFL clamp or retry.

```text
mesh = phx.discretization.TemporalMesh.uniform(
    0.0, 0.01, 10, role="internal"
)
rollout = phx.solver.ScheduledFiniteVolumeRolloutPlan(
    runtime,
    mesh,
    replay=phx.solver.FiniteVolumeReplayPolicy("block", block_size=4),
)
result = rollout.rollout(initial_state, args)
```

A failed prescribed interval retains the incoming state. `stable_step_limits` and
`stability_margins` expose the schedule validity boundary. Replay policies control
reverse-mode storage independently from final/checkpoint/trajectory retention.

`ScheduledBalanceLawRolloutPlan.from_realized_mesh` converts the accepted prefix of an
adaptive balance-law result into an all-active internal mesh. Replaying that mesh fixes
the discrete schedule for forward- and reverse-mode differentiation. Rejected adaptive
attempts never enter the replay program.

## Prepared transport adapters

`prepare_balance_law_transport` converts either `PreparedFiniteVolumeRuntime` or
`ConstrainedMHDSSPRK3Plan` into the same narrow transport contract. The adapter owns
prescribed advancement, source views, adapter-specific auxiliary state, and checkpoint
arrays. It does not form a general multiphysics graph.

Finite-volume source views expose ordinary cell averages. Constrained-MHD source views
reconstruct cell magnetic fields from authoritative face flux while permitting ordinary
processes to modify only density, momentum, and total energy. A process must declare its
modified component names; undeclared changes reject the complete interval.

## Transactional source processes

`PreparedBalanceLawRuntime` applies declared processes in forward order over the first
half-interval, advances the adapter-owned transport interval once, then applies processes
in reverse order over the second half. `BalanceLawCompositionPlan` specifies the number
of equal subintervals for each process in each half. It does not choose a process's
integrator: `process.advance(start, end, ...)` owns the complete finite update.
Symmetric ordering alone does not raise a first-order process method to second order.
The conservative outer step limit for process `i` is its reported `step_limit` multiplied
by its subcycle count; `process_step_limits` and adaptive-controller evidence use those
outer-interval limits.

`BalanceLawProcessAdvance.source_change` is a cell-average increment, not a tendency
or a volume-integrated source. It must have exactly the incoming view's shape, be finite,
agree with `cell_average - incoming` to storage-precision roundoff, and vanish in every
undeclared component. Invalid shape is a contract error; inconsistent or nonfinite
increments reject the complete interval with balance status 4. Candidate ownership
violations retain status 3. Process state is provisional until every source half-step,
transport step, and accepted-step coupling succeeds.

A failure rolls back cell state, magnetic cochains, process state, transport auxiliary
state, and cumulative accepted budgets. The returned transport's accepted integrals are
zeroed and its state is restored on outer rejection, even if transport itself succeeded.
Its native diagnostics and status describe the transport proposal; they are not an
outer acceptance certificate.
Random drivers come from immutable `WienerRealization`,
`OrnsteinUhlenbeckRealization`, or `CompositeStochasticRealization` values; no hidden
key is consumed. OU innovations query one global transformed Brownian clock and obey
the exact OU semigroup when an interval is subdivided.

`BalanceLawRuntimeState.accepted_budget` is a `BalanceLawAcceptedBudget`. It retains
only initial component integrals, cumulative source rows (process declaration order),
net transport component integrals, coupling rows (coupling declaration order), and the
accepted outer-step count. Source/coupling increments use the source view's active-cell
mask, actual effective volumes, and native reduction precision. Transport totals are
the measured change in content across transport, including changes in effective
measures; they are not independent boundary-flux estimates. Use native accepted flux
ledgers for face-resolved conservation evidence. `budget.total_change` sums the three
contributions and can be compared with current content minus `budget.initial_integrals`.
No extra per-cell or per-step budget history is retained. Scheduled replay, adaptive
retries, and checkpoints carry the same committed budget and stochastic process state.

Built-in processes:

- `NewtonianSelfGravityPlan`: periodic Poisson gravity with zero-mean gauge and an
  internal-energy-preserving momentum/energy kick;
- `SpectralOUForcingPlan`: Hermitian, band-limited turbulence driven by a replayable
  `OrnsteinUhlenbeckRealization`;
- `RadiativeCoolingProcessPlan`: material-owned temperature and an implicitly
  differentiated local cooling solve.

`BalanceLawCheckpointPlan` archives the adapter-owned transport continuation, exact
process-state inventory, and compact accepted budget in a checksum-validated pickle-free
array archive. MHD checkpoints include reduced cell state, face magnetic flux, time,
proposed step, status, and accepted-step count. Runtime states are normally created by
`runtime.initialize_state`; direct `BalanceLawRuntimeState` construction requires
`(transport_state, process_states, accepted_budget)`.

`tools/balance_law_transport_qualification.py` compares adaptive constrained-MHD
execution with full, step-rematerialized, and block-rematerialized balance-law replay,
including cell, magnetic-cochain, retention, and divergence evidence.

## Particle mesh gravity

`ParticleMeshGravityPlan` reuses one `PreparedParticleGridSplat` for conservative mass
deposition and grid-field gathering. Its ordinary-time kick-drift-kick update preserves
stable material particle identities. `ParticleMeshGravityForceResult` exposes the
deposited field, Poisson convergence, gathered acceleration, support, mass balance, and
net force without imposing an evolution coordinate. Particle routing is piecewise
differentiable; cell-route changes remain discrete.

`CosmologicalParticleMeshPlan` composes that same acceleration evaluation with
`CosmologicalKDKPlan`; it never nests the ordinary-time PM step or creates a second
deposition/Poisson path. The cosmological plan advances canonical momentum over an
explicit scale-factor schedule, recomputes endpoint force, reuses one authoritative
particle discretization, and rolls back on failed force or state evidence.

`ComovingEulerPlan` reuses a prepared finite-volume Euler residual while changing the
evolution coordinate to scale factor. It applies the exact `1/(a^2 H)` transport scaling,
Hubble momentum/internal-energy work, and shared rescaled-potential gravity source.
`CosmologicalGasParticleGravityPlan` predicts and corrects gas transport, deposits DM,
solves one total-density periodic potential, gathers the same field to particles, and
atomically accepts or rolls back the complete epoch.

`PrimordialMicrophysicsPlan` closes the first deterministic source layer with named
HI/HII/HeI/HeII/HeIII/electron state, immutable temperature/scale rate tables,
photoionization/heating, collisional ionization, recombination, cooling, a fixed Newton
solve, nuclei/charge/energy ledgers, and atomic gas-energy commit.
`CosmologicalPopulationPlan` extends this path with fixed-capacity dynamic gravitating
star/BH slots, generation-safe IDs, replayable event ledgers, conservative gas→star
transfer, and stochastic thermal reservoir coupling. H2, metals, winds, black-hole
accretion, radiation, and calibrated production models remain profile-specific.

`PeriodicImageForcePlan` remains a cheap diagnostic and `PeriodicEwaldForcePlan` remains
the periodic reference. `ParticleOctreePlan3D` prepares one sparse Morton
hierarchy consumed by isolated Barnes--Hut, occupied-level Cartesian FMM, and
BH-short-range TreePM; none introduces a second particle state or PM path.
`TwoLevelAMRPlan` supplies ratio-two prolong/restrict,
reflux, average-down, fine-authoritative composite gravity, particle level routing, and
atomic epoch commit. Multi-level partial patches, CT/radiation synchronization, and
distributed communication remain later parity profiles rather than silent modes.

## Constrained transport MHD

`StructuredCochainBridge.pack_face_flux` stores integrated magnetic flux as a degree-two
cochain. `pack_edge_circulation` stores integrated electromotive circulation as a
degree-one cochain. `UpwindConstrainedTransportPlan` updates magnetic flux through the
cochain exterior derivative, so the discrete divergence change is zero by construction.

`ConstrainedMHDSSPRK3Plan` advances reduced cell conservation and face magnetic flux in
the same SSPRK stages. `PreparedConstrainedMHDBalanceLawTransport` then composes that
advance with the ordinary gravity, cooling, and OU process contracts under scheduled or
adaptive balance-law replay. A global convex stage blend preserves conservation and the
magnetic constraint while enforcing ideal-MHD admissibility. `HLLDFluxPlan` uses HLL
fallback for degenerate or inadmissible intermediate fans.

Initial qualification is deliberately narrow:

- stationary Cartesian three-dimensional topology;
- all axes periodic;
- piecewise-constant MHD face traces;
- gravity, cooling, and OU forcing may modify only their declared nonmagnetic cell
  components; face magnetic flux remains transport-owned;
- no AMR, mapped grids, physical MHD boundaries, or distributed CT.

## Learned closures

`ConservativeFaceClosurePlan` adds one correction to each shared baseline face flux.
Trainable parameters arrive through runtime `args`; the static closure callable and ID
remain part of the numerical method identity. Equal-state consistency and finite output
are enforced. Existing positivity machinery blends an unsafe corrected flux toward the
uncorrected monotone fallback.

Cell-face closures are rejected for constrained MHD until a closure also supplies a
compatible edge-electromotive correction.

## Differentiability boundary

The fixed discrete program is differentiable. Hard limiter decisions, HLL/HLLD wave
regions, fallback masks, positivity activation, table intervals, particle routes, and
schedule validity are branchwise. Failed transport, elliptic, nonlinear, or stochastic
primals do not define valid gradients.

Favre LES is differentiable only on the admissible interior with fixed species,
filter, discretization, and trace policy. Positivity/mass-fraction gates,
eddy-viscosity bounds, algebraic zero branches, and any shock/limiter selection are
branchwise. Prepared coefficients and provenance are nontrainable.

## Compressible-flow candidate ownership

`phydrax.applications.compressible_flow` is the application facade for the current
smooth and shock-resolving compressible candidates. `CompressibleFlowCaseSpec` binds
dimension, Euler or Navier–Stokes physics, route, one canonical
`HomogeneousHelmholtzPlan`, reference scales, density/pressure floors, thermal solve
capacity, and an optional finite-x boundary-layer case independently of a
discretization. Its `fidelity="dns-candidate"` value is a candidate identity:
`claims_dns` remains false.

The canonical all-species state is

```text
U = (rho_1, ..., rho_S, rho u_1, ..., rho u_d, rho E),
rho = sum_s rho_s .
```

Primitive state is `(rho_1, ..., rho_S, u_1, ..., u_d, T)`.
`ChemicalComponentCatalog`, phase-specific `ChemicalSpeciesSchema` with explicit gas
standard pressure, species calorics, `IdealGasReferenceHelmholtzTerm`, and an ideal or
residual Helmholtz term form one model identity. `HomogeneousMixtureEulerSystem` and
`HomogeneousMixtureCompressibleNavierStokesSystem` delegate pressure, temperature,
entropy, frozen-composition sound speed, state recovery, characteristics, and transport
calorics to that model. Peng–Robinson roots, stability, and flash remain separate
solver-owned equilibrium operations; they are never selected inside an Euler flux.

### Favre LES transport and SGS energy

`HomogeneousMixtureCompressibleNavierStokesSystem(..., favre_les=model)` adds
physical SGS transport with exact gas species, SI units, filter/provenance,
Prandtl/Schmidt numbers, viscosity bound, SGS-energy dissipation coefficient, and
SGS-energy Schmidt number.

The `provided-sgs-kinetic-energy` policy appends `rho*k_sgs` after total energy in
conserved state and `k_sgs` after temperature in primitive state. Total energy
includes SGS energy. Isotropic SGS pressure participates in hyperbolic flux and
sound speed; deviatoric work, heat/species transport, and SGS-energy diffusion are
diffusive. `FavreLESCoupledRate` applies production/dissipation only to the
SGS-energy component, exposes its positivity step, and reports zero total-energy
source. The `neglected` policy retains the smaller state.

Both policies require conserved state/gradients and refuse the
primitive-gradient-only convenience. Favre transport remains separate from shock
sensors, Riemann dissipation, limiting, bulk viscosity, and artificial viscosity.
Binding it does not qualify every application route. See
[LES equations](api/equations/les.md#favre-effective-transport).

### Smooth, all-speed, and shock routes

`SmoothCompressibleProductionPlan` owns tensor DGSEM split volume flux, system-specific
sampled entropy compatibility evidence, and entropy-BR1 viscosity.
`NodalDGCompressibleProductionPlan` is a separate overintegrated nodal-DG route with
LDG traces; evidence from one is not evidence for the other. Both bind prepared spatial
dynamics through `prepare_explicit`, while the tensor route can bind an already
constructed additive IMEX method through `prepare_imex`.

`StructuredFVCompressibleProductionPlan` owns structured or mapped high-resolution
finite volume with WENO-Z, TENO, or MP5 reconstruction and stage positivity.
`ShockAwareAllSpeedFluxPlan` is the primary interface flux. Its
`AllSpeedHLLFluxPlan` scales the HLL dissipative acoustic half-width with relative
Mach, including ALE grid velocity, and uses the symmetric central limit when the
scaled wave width is zero. `NumericalFluxResult.max_speed` retains the unscaled
physical acoustic bound used by FV timestep admission. A pressure-jump sensor,
inadmissible state, or nonfinite primary flux selects the canonical arbitrary-normal
HLL fallback and records that decision. Stage positivity uses the same fallback.
This remains a numerical shock model, never a hidden fallback or a smooth-DNS
fidelity claim.

```python
from phydrax.applications import compressible_flow as cflow
from phydrax.equations import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
)

system = HomogeneousMixtureCompressibleNavierStokesSystem(
    homogeneous_thermodynamics,
    mixture_transport,
    3,
)
case = cflow.CompressibleFlowCaseSpec(
    "channel-candidate",
    system,
    "structured-fv",
    fidelity="dns-candidate",
)
shock = cflow.ShockResolvingPolicy(
    "weno_z",
    all_speed=cflow.AllSpeedCompressiblePolicy(reference_mach=0.2),
)
route = cflow.StructuredFVCompressibleProductionPlan(
    "structured",
    shock=shock,
    viscous=viscous_flux_plan,
)
production = route.prepare_explicit(prepared_fv_dynamics)
step_result = production.step(step_index, time, state, step_size, runtime_args)
```

The case stores the exact immutable system; its case identity therefore includes
thermodynamics, transport, auxiliary physics, and state layout. FV dynamics must be
prepared with `route.method`, and diffusive systems must bind `ViscousFluxPlan`.
`prepare_explicit` checks those identities. `PreparedCompressibleProduction.checkpoint`
binds method, route, topology, time, step, tree structure, and content. `restore`
requires the same topology identity.

`CharacteristicNonreflectingBoundaryPlan` freezes incoming characteristics to a far
field and passes outgoing waves. `CompressibleSpongePlan` relaxes conserved variables
with mass, momentum, energy, and entropy ledgers. `FiniteXBoundaryLayerCaseSpec` owns
finite streamwise extent, inflow, characteristic outflow, and no-slip thermal or
adiabatic wall semantics; it is distinct from the slow-growth model below.

## Slow-growth source model

`CompressiblePlaneBaseflowPlan.evaluate()` forms one immutable,
Favre-consistent wall-normal baseflow snapshot from homogeneous-plane statistics. A
temporal model prepares the primitive source

```text
S_q^temporal(y) = -g (y - y_0) partial_y q_bar(y),
```

while `SpatialSlowGrowthModelPlan` prepares the explicitly modeled spatial source

```text
S_q^modeled-spatial(y) = -U_c partial_x q_bar(y).
```

The spatial form requires the caller to supply `streamwise_base_derivative` in the
snapshot; it does not compute a finite-x streamwise solution. Both plans can impose
declared displacement/momentum-thickness rates and adiabatic or isothermal wall source
conditions. Conversion to conservative mass, momentum, total-energy, internal-energy,
temperature, and entropy rates is exposed with algebraic, wall, integral, energy, and
entropy evidence.

```python
baseflow = cflow.CompressiblePlaneBaseflowPlan(
    case,
    wall_normal_coordinates,
    wall_normal_axis=1,
)
snapshot = baseflow.evaluate(conserved, sample_index=accepted_step)
continuation = cflow.SlowGrowthContinuation(snapshot)
source = cflow.TemporalSlowGrowthModelPlan(growth_rate).prepare(
    snapshot,
    continuation=continuation,
)
evaluation = source.evaluate(conserved)
comparison = source.compare_finite_x(
    conserved,
    finite_x_reference_source,
    reference_id="finite-x-reference",
    relative_tolerance=0.05,
)
```

One `PreparedSlowGrowthSource` is frozen before the parent step and reused unchanged by
every RK or IMEX stage. `SlowGrowthContinuation.accept()` advances only with a new
snapshot from the same baseflow plan; `reject()` preserves the exact parent snapshot
and continuation identity. `compare_finite_x` reports L2, relative L2, maximum error,
the admission threshold, and `admitted`; both the prepared model and comparison retain
`claims_spatial_dns=False`. Supplied finite-x data are external evidence, not a
fidelity relabel.

## High-speed aerodynamic surface contract

`PreparedFiniteVolumeDynamics.boundary_trace()` exposes the same reconstructed
interior/exterior states, inviscid numerical flux, diffusive flux, normals, and
measures used by the residual. `CompressibleSurfaceObservationPlan` converts those
traces into dimensional pressure, wall heat flux, pressure/viscous forces, moments,
and nondimensional coefficients. Its total force is the discrete outward momentum
flux; `force_balance_defect` separates numerical wall-normal transport from the
pressure-plus-viscous interpretation.

`NormalShockReferencePlan`, `ObliqueShockReferencePlan`, and
`PrandtlMeyerReferencePlan` are calorically-perfect-gas references. They reject
detached, sonic, or out-of-bracket conditions rather than applying those formulas to
thermally perfect or reacting states.

## Transonic RANS and reduced models

`SpalartAllmarasCompressibleSystem` appends one conservative SA-neg-noft2 working
variable to the canonical mixture Navier--Stokes state. `SpalartAllmarasArguments`
supplies an exact positive wall-distance field. `SpalartAllmarasWallBoundary` imposes
zero face working variable while delegating gas velocity and thermal semantics to an
existing no-slip wall. The former algebraic SA/SST transport hooks were removed:
they did not represent complete turbulence transport equations.

`AirfoilSectionPlan` and `AirfoilOGridPlan` provide a fixed-topology mapped airfoil
route. `RAE2822CasePlan` requires an explicit rights-bearing reference manifest and
exact Mach, Reynolds, temperature, and lift conditions. `TransonicFixedLiftPlan`
solves only a caller-supplied deterministic lift residual.

`TransonicSmallDisturbancePlan` is a separate inviscid low-fidelity model. It is not
a `CompressibleFlowCaseSpec` route and carries no RANS, separation, buffet, or
hypersonic claim. Native panel flows accept only explicit incompressible,
Prandtl--Glauert, or Karman--Tsien pressure postprocessing; sonic and supersonic
postprocessing is rejected.

## Nonequilibrium high-enthalpy states

`TwoTemperatureMixtureEulerSystem` and
`TwoTemperatureMixtureNavierStokesSystem` retain every species density, total energy,
and explicit thermal-mode energy. Heavy-particle and mode calorics are disjoint, so
equilibrium vibrational/electronic energy is not counted twice. Pressure and frozen
sound speed use the heavy-particle temperature; mode temperatures come from
fixed-capacity implicit energy inversions.

The initial supported model is neutral multi-species gas with explicit mode pools.
Ionization, electron pressure, radiation, catalysis, ablation, and DSMC are not
implied.

## Buffet and operator-learning observations

`CompressibleShockTrackPlan` binds a surface interval, compression direction, and
minimum pressure gradient. `CompressibleSnapshotMetricPlan` applies declared field
scales and square-root cell-volume weighting before DMD or POD; temporal
`TrajectoryData.weights` are not repurposed as spatial weights.

`CompressibleOperatorDatasetPlan` admits only cases whose artifact manifests permit
training. It maps exact geometry/system/method/qualification identities into
`OperatorCaseProvenance`; its default split keeps each geometry in one partition.
