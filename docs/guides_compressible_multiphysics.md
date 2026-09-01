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
atomically accepts or rolls back the complete epoch. Its first scope is adiabatic ideal
gas plus collisionless DM; it has no cooling, chemistry, feedback, or tree-force claim.

`PeriodicImageForcePlan` is a small-N softened periodic image-shell qualification tool.
It reports absolute/relative force error and net-force evidence for a supplied candidate.
It is not an Ewald solver, short-range correction, Barnes--Hut tree, FMM, or production
TreePM implementation.

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
