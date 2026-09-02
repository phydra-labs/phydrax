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

`PreparedBalanceLawRuntime` applies declared processes symmetrically around one exact
adapter-owned transport interval. Process state is provisional until every source
half-step and the transport step succeeds. A failure rolls back cell state, magnetic
cochains, process state, and transport auxiliary state.
Random drivers come from immutable `WienerRealization`,
`OrnsteinUhlenbeckRealization`, or `CompositeStochasticRealization` values; no hidden
key is consumed. OU innovations query one global transformed Brownian clock and obey
the exact OU semigroup when an interval is subdivided.

Built-in processes:

- `NewtonianSelfGravityPlan`: periodic Poisson gravity with zero-mean gauge and an
  internal-energy-preserving momentum/energy kick;
- `SpectralOUForcingPlan`: Hermitian, band-limited turbulence driven by a replayable
  `OrnsteinUhlenbeckRealization`;
- `RadiativeCoolingProcessPlan`: material-owned temperature and an implicitly
  differentiated local cooling solve.

`BalanceLawCheckpointPlan` archives the adapter-owned transport continuation and exact
process-state inventory in a checksum-validated pickle-free array archive. MHD
checkpoints include reduced cell state, face magnetic flux, time, proposed step, status,
and accepted-step count.

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
the periodic reference. `ParticleOctreePlan3D` prepares one Morton hierarchy consumed by
isolated Barnes--Hut, uniform Cartesian FMM, and BH-short-range TreePM; none introduces
a second particle state or PM path. `TwoLevelAMRPlan` supplies ratio-two prolong/restrict,
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

## Compressible-flow candidate ownership

`phydrax.applications.compressible_flow` is the application facade for the current
smooth and shock-resolving compressible candidates. `CompressibleFlowCaseSpec` binds
dimension, Euler or Navier–Stokes physics, route, material, reference scales, and an
optional finite-x boundary-layer case independently of a discretization. Its
`fidelity="dns-candidate"` value is a candidate identity: `claims_dns` remains false,
and no route, shock policy, slow-growth model, or passed local diagnostic turns it into
a released DNS claim.

The conservative state is

```text
U = (rho, rho u_1, ..., rho u_d, rho E)
rho E = rho e(rho, p) + 1/2 rho |u|^2 .
```

`IdealGasMaterial` remains the standard material. `ThermallyPerfectGasMaterial`
provides polynomial heat capacity with bounded caloric inversion.
`ResearchRealGasMaterial` is restricted to non-characteristic structured or mapped
finite volume and requires caller-supplied pressure/internal-energy/sound-speed
providers plus exact derivative and convexity certificates. Phydrax does not download
or silently substitute an external EOS.

### Smooth, all-speed, and shock routes

`SmoothCompressibleProductionPlan` owns tensor DGSEM split volume flux, an
entropy-stable interface, and entropy-BR1 viscosity.
`NodalDGCompressibleProductionPlan` is a separate overintegrated nodal-DG route with
LDG traces; evidence from one is not evidence for the other. Both bind prepared spatial
dynamics through `prepare_explicit`, while the tensor route can bind an already
constructed additive IMEX method through `prepare_imex`.

`StructuredFVCompressibleProductionPlan` owns structured or mapped high-resolution
finite volume with WENO-Z, TENO, or MP5 reconstruction and stage positivity.
`ShockAwareAllSpeedFluxPlan` is the route's primary interface flux. Its
`AllSpeedHLLFluxPlan` applies `AllSpeedCompressiblePolicy` to the HLL acoustic
half-width as `min(1, max(M_min, M_relative / M_ref))`; ALE uses velocity relative
to the moving grid. This is an O(M) smooth low-Mach correction, not an
incompressible projection. A pressure-jump sensor, inadmissible primary state, or
nonfinite primary flux selects the declared robust fallback and records it in the flux
result. Ideal-gas Euler/Navier–Stokes routes use `EinfeldtHLLFluxPlan`; general
certified material systems use the arbitrary-normal `HLLFluxPlan` because they do not
expose an ideal-gas Roe eigensystem. The finite-volume stage positivity route can also
invoke its declared fallback. This remains a numerical shock model, never a hidden
fallback or a smooth-DNS fidelity claim.

```python
from phydrax.applications import compressible_flow as cflow

case = cflow.CompressibleFlowCaseSpec(
    "channel-candidate",
    3,
    "navier_stokes",
    "structured-fv",
    material,
    fidelity="dns-candidate",
)
shock = cflow.ShockResolvingPolicy(
    "weno_z",
    all_speed=cflow.AllSpeedCompressiblePolicy(reference_mach=0.2),
)
route = cflow.StructuredFVCompressibleProductionPlan(
    "structured",
    shock=shock,
)
production = route.prepare_explicit(prepared_fv_dynamics)
step_result = production.step(step_index, time, state, step_size, runtime_args)
```

The FV dynamics in the example must have been prepared with `route.method`;
`prepare_explicit` checks that identity. `PreparedCompressibleProduction.checkpoint`
binds method, route, topology, time, step, tree structure, and content. `restore`
requires the same topology identity. These application assemblies are local fixed-step
owners; distributed execution and topology-changing restart require separate exact
plans and evidence.

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
