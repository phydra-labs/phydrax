# Lattice Boltzmann and discrete-velocity methods

Phydrax provides native JAX kinetic methods with explicit capability, ownership,
precision, conservation, and replay contracts. The default lattice-Boltzmann path is
a weakly compressible, trailing-population, pull-streaming method on uniform Cartesian
cell grids. Research extensions reuse the same prepared-state and fixed-step runtime
without silently claiming qualification outside their evidence envelope.

## Status vocabulary

- **Qualified**: covered by reproducible benchmark or invariant artifacts.
- **Implemented research path**: executable, differentiable where stated, and covered
  by focused contract tests, but not a production stability guarantee.
- **Experimental infrastructure**: explicit data model and transfer/update contract;
  users must provide application-specific evidence before relying on it.

Evidence is graduated rather than inferred: **invariant-complete**,
**physics-qualified**, **differentiation-qualified**, **execution-qualified**, and
**deployment-qualified** are independent claims. A capability advances only when the
corresponding named artifact records its configuration and tolerances.

These labels describe numerical evidence, not release status. The commercial profiles
described below are constructed with `signed=False` and `released=False`; a passing
profile evaluation remains a candidate for one exact operating/deployment tuple.

## Capability matrix

| Capability | Status | Contract |
| --- | --- | --- |
| D2Q9, D3Q19; BGK/TRT; Guo forcing | Qualified baseline | Uniform Cartesian grid, low Mach, float64 |
| D3Q27 | Implemented research path | Certified opposite map and isotropy evidence |
| MRT, regularized, central-moment, cumulant, KBC, entropic, Smagorinsky | Implemented research path | Collision diagnostics expose conservation, positivity, entropy, and root residuals |
| Float32 and mixed storage/compute precision | Implemented research path | Explicit population, compute, accumulation, and nonlinear-solve dtypes |
| Stationary and moving halfway wall | Qualified baseline | Write-once blocked-link routing |
| Interpolated bounce-back and moving curved walls | Implemented research path | Link fraction and wall velocity are geometry data; force ledger is explicit |
| Periodic, velocity, pressure, convective open boundaries | Implemented research path | Compiled link ownership and fixed stream → wall → open stages |
| Static SDF, refreshed SDF, moving bodies, immersed forcing | Implemented research path | Geometry epochs, conservative newborn-cell initialization, and force ledgers |
| Multiblock Cartesian coupling | Implemented research path | Same-step halo schedule with explicit transfer operators |
| Ratio-2 block refinement | Experimental infrastructure | Conservative restriction/prolongation and fixed subcycling schedule |
| Collision-aware ratio-2 transfer | Experimental research path | Equilibrium/nonequilibrium transfer, acoustic scaling, half-time interface data, and local defects |
| Phase-field, colour-gradient, free-energy, thermal, passive species | Implemented research path | Separate distributions and explicit mass/energy/species ledgers |
| Single-source binary thermodynamics | Implemented research path | Energy, variational derivative, symmetric stress, and selected force share one closure |
| Reactive species with Strang splitting | Implemented research path | Atomic rollback across flow, thermal, and species states |
| D2V17 and off-lattice D2V37 smooth compressible kinetic methods | Implemented research path | Total energy is a kinetic population; off-lattice transport is explicit |
| Fixed FV/kinetic interface | Experimental infrastructure | One common conservative flux, FV shock ownership, atomic update |
| Mapped-grid kinetic method | Experimental research path | Metric-identity/free-stream residuals and injected geometric source |
| Sharded execution and halo exchange | Implemented execution path | Fixed decomposition, deterministic ownership, no dynamic repartitioning |
| Prepared production sharding | Implemented execution path | Actual prepared hydrodynamics runs under fixed spatial NamedSharding with global reference equivalence |
| AA/even-odd memory layout and fused step | Implemented execution path | Logical-state equivalence evidence and checkpoint parity metadata |
| Block reverse replay | Implemented execution path | Fixed-size replay blocks; no adaptive recomputation policy |
| Static block-sparse fluid cells | Implemented research path | Explicit fluid logical IDs, compact block populations, periodic pull streaming, and halfway bounce-back on missing fluid neighbors |
| IREE export | Forward inference only | Stable tuple ABI; gradients remain in JAX |
| Portable kinetic checkpoint | Implemented execution path | Full array PyTree, runtime controls, program identity, topology, parity, and checksums |

## Core state and update order

The baseline stored state has shape `grid.shape + (Q,)` and is post-stream,
boundary-closed, and pre-collision. One accepted step:

1. reconstructs density and raw momentum;
2. evaluates explicit acceleration or solves the declared local velocity-dependent
   force problem;
3. converts physical quantities to lattice units and applies the half-force shift;
4. evaluates equilibrium and the selected collision family;
5. applies the collision-compatible forcing representation;
6. pulls populations according to the compiled stream ownership;
7. applies wall reconstruction exactly once on wall-owned links;
8. applies velocity, pressure, or convective reconstruction exactly once on
   open-owned links;
9. accepts the complete candidate or retains the complete previous state.

Macroscopic velocity follows `rho * u = sum_i(c_i * f_i) + 0.5 * dt * F`.
Pressure is derived from density for the athermal weakly compressible path. No API
labels this path as an exact incompressible projection.

## Static sparse fluid domains

`SparseLatticeBoltzmannPlan` binds a `PreparedTensorIndexSpace`, one certified
velocity set, collision plan, block shape, and block capacity. Preparation
accepts explicit fluid-cell logical IDs. It constructs a
`SparseBlockTopologyState`, stores populations on the compact storage-cell
axis, and lowers every pull-streaming source to a compact slot. A missing
nonperiodic or nonfluid source is owned by halfway bounce-back.

The path supports all unforced collision families accepted by
`collide_detailed`, periodic axes, fixed walls, JIT execution, and explicit
geometry refresh. `refresh_geometry` aligns retained populations by logical
cell ID, initializes activated cells from declared density/velocity, reports
activated and retired mass, and advances the topology generation.

This is an explicit realization. It does not silently scan a virtual grid to
discover fluid cells and does not fall back to dense execution. Moving
arbitrary SDF discovery, staged open boundaries, multiphase stencil closures,
AA addressing, multiblock, and distributed sparse execution remain outside
this realization and must use an existing qualified dense path.

Logical cell count and storage-cell capacity are separate resource facts.
Sparse storage is appropriate when a fluid region occupies a minority of its
bounding box; the dense realization remains appropriate above the measured
crossover.

## Collision, moments, and forcing

`MomentBasisPlan` prepares a full-rank raw-moment transform and exposes round-trip
residual evidence. `RelaxationSpectrumPlan` names conserved and relaxing channels.
MRT collision operates in this prepared basis. Regularized, central-moment,
cumulant, KBC, entropic, and Smagorinsky plans retain their method-specific
diagnostics instead of collapsing to a BGK alias.

`GuoForcingPlan` explicitly certifies compatible collision families. MRT and
central-moment collisions transform the same raw Guo source through their prepared
bases before applying relaxation-dependent factors. `VelocityDependentAccelerationPlan`
makes drag-like force laws an explicit local nonlinear problem and records
convergence and residual evidence; there is no hidden fixed-point loop.

## Boundary ownership

`compile_staged_lattice_boltzmann_boundary` assigns every destination population
exactly one owner from typed face, body, and corner declarations. It also fixes
runtime parameter ordering through `velocity_parameter_ids`,
`pressure_parameter_ids`, and `convective_parameter_ids`; ambiguous mixed-owner
corners are rejected unless a `LatticeBoltzmannCornerRule` selects precedence.
`StagedLatticeBoltzmannBoundaryPlan` then executes a fixed stage order:

1. stream local, periodic, or halo populations;
2. reconstruct wall-owned populations;
3. reconstruct open-owned populations.

No later stage may overwrite another owner's link. Runtime boundary parameters hold
body velocities and velocity/pressure/convective targets. Convective history is part
of `LatticeBoltzmannBoundaryState`, never Python-side mutable state.

The original `LatticeBoltzmannBoundaryPlan` remains the qualified simple path for
periodic, stationary halfway, and fixed axis-aligned moving-wall cases.

## Geometry, topology changes, and transfers

`LatticeBoltzmannGeometryEpoch` separates geometry epoch, topology epoch, and
parameter-only changes. `LatticeBoltzmannGeometryRefresh` reports changed links and
fluid/solid transitions. `LatticeBoltzmannPopulationTransferPlan` preserves
unchanged fluid cells, initializes newly uncovered cells from declared macroscopic
data, and reports mass/momentum transfer defects.
`LatticeBoltzmannGeometryTransaction` commits geometry and populations atomically.

Fixed-SDF curved walls use prepared link fractions. Moving-body geometry may refresh
those fractions every step without changing the tensor shape. Topology-changing
refreshes require the explicit conservative transfer transaction.
`ImmersedBoundaryForcingPlan` is a separate regularized marker coupling with
equal-and-opposite force, torque, work, partition-of-unity, and convergence evidence.

`prepare_lattice_boltzmann_link_geometry` bridges an existing
`CompiledGeometry` with signed-distance and boundary-normal capabilities into
certified per-link fractions and normals; the LBM layer does not duplicate CAD,
mesh, image, or rasterization frontends. Parabolic and Womersley profile plans
provide differentiable runtime velocity targets while leaving link ownership
unchanged.

`LatticeBoltzmannMultiblockCouplingPlan` exchanges all fixed conforming interface
traces from the same source state before committing incoming directions, with exact
orientation/Q-permutation reciprocity. Ratio-2 refinement uses explicit conservative
restriction, prolongation, and a fixed subcycle phase. `MappedLatticeBoltzmannPlan`
requires its prepared metric nodes to coincide with the Cartesian LBM cell centres;
users therefore prepare the companion point-primary metric grid over the cell-centre
coordinate bounds. Mapped free-stream and metric-identity residuals remain explicit
experimental evidence.

## Multiphysics

The multiphysics paths use separate distributions and expose conservation accounting:

- colour-gradient and free-energy phase-field flow;
- thermal energy distribution with sensible-energy conversion;
- independent passive species distributions;
- reactive species with a fixed Strang reaction/transport/reaction schedule.

`AbstractBulkFreeEnergy` and `DoubleWellFreeEnergy` define the local potential.
`BinaryPhaseThermodynamicClosure` combines that potential with differentiable
`BinaryThermodynamicParameters` so bulk energy, chemical potential, symmetric stress,
and both stress-divergence and chemical-potential-gradient force representations
come from one source. `PreparedBinaryKineticThermodynamics` binds the closure to the
certified lattice stencil and records the selected force policy and representation
residual. Application phase-field workflows consume the same closure and parameters.
The same closure also supplies the characteristic interface width and planar
surface tension used by free-energy admissibility and capillary diagnostics.
Chemical-potential-gradient and negative stress-divergence forcing are separate,
explicitly selected discrete representations; qualification records their
mesh-convergent agreement rather than assuming a discrete product rule.

Every ledger separates initial amount, boundary exchange, volumetric source,
reaction exchange where applicable, and residual. A coupled step commits all fields
or rolls all fields back.

## Smooth compressible discrete-velocity methods

`CertifiedDiscreteVelocityQuadrature` records velocities, weights, transport kind,
and exactness certification. D2V17 is the on-lattice research choice. D2V37 is
explicitly off-lattice and must use `PreparedOffLatticeSemiLagrangianDVM` or another
declared transport method; it never reuses integer roll streaming.

`SmoothCompressibleD2VKineticMethod` stores particle and total-energy populations.
Density, momentum, total energy, pressure, and temperature are recovered jointly,
and realizability evidence exposes nonfinite, nonpositive, or population-level
violations. The fixed FV/kinetic interface derives one population flux and maps that
same flux into conservative variables, so both sides see equal and opposite exchange.
Its local program manifest therefore contains moment, equilibrium, collision, and
realizability stages only. Finite-volume DVM transport has a separate reconstruction,
numerical-flux, source, residual, and conservation manifest; neither path claims
integer lattice streaming for off-lattice quadratures.

### Learned total-energy equilibrium

`PositiveEnergyEquilibriumPlan` provides a local positive equilibrium for the
smooth-compressible total-energy population. Phydrax uses weight-absorbed
populations, so the conserved channel is `sum(g) = total_energy_density`; it is
not the `2 rho E` convention used by some thermal lattice formulations. For
dual coordinates `zeta`, the represented population is proportional to
`E * w_i * exp(zeta . c_i)`, normalized with the certified quadrature weights.
This makes total energy exact up to floating-point roundoff. The first velocity
moment remains a constitutive energy flux and is reported separately against
`(E + p) u`.

`solve(total_energy, target_flux)` is the native fixed-iteration oracle. It
checks that the normalized target flux lies strictly inside the discrete
velocity hull and returns root, positivity, and moment evidence.
`evaluate(total_energy, target_flux, dual)` evaluates an externally supplied
dual without claiming that its flux residual vanishes. Nonfinite inputs,
nonpositive energy, targets outside the realizable interior, failed roots, and
represented zero/nonfinite populations have explicit status values; they are
never clipped or replaced with an analytic equilibrium.

`SmoothCompressibleD2VKineticMethod.equilibrium_from_energy_dual_with_evidence`
combines the existing analytic particle equilibrium with this energy channel.
`collide_with_energy_dual_with_evidence` is an all-or-nothing local transaction:
it accepts only a conservative, post-collision-realizable candidate and otherwise
retains the complete input state.

Learned parameters never enter the prepared kinetic method, which is
`NonTrainableState`. `LearnedEnergyEquilibriumBindingPlan` instead binds a
four-component state schema, an exact D2V quadrature and material identity,
leakage-safe dataset preparation, train-only normalization, semantic provenance,
and a numeric model revision. Its prepared deployment freezes a `4 -> 2`
pointwise model and predicts only the dual coordinates.

This is an experimental local equilibrium/collision path. The checked D2V17
artifact does not qualify spatial transport, boundary conditions, forcing,
the fixed FV/kinetic interface, D2V37 transport, shock stability, stage-two
rollout training, entropy dissipation, export, or production deployment. The
qualified athermal D2Q9/D3Q19 baseline is unchanged.

## Program manifest and restart

Every prepared kinetic path exposes a `KineticProgramManifest`. Its field
specifications declare population, macroscopic, source, geometry, history, ledger,
precision, halo, conservation, checkpoint, and differentiability roles. Ordered
stage specifications declare reads, writes, exchanges, reductions, and failure
scope. Preparation rejects unknown or unavailable reads, duplicate stage ordering,
and exchanged fields without halo support. The manifest is metadata over the
existing pure kernels; it is not a runtime model DSL.

`KineticCheckpointPlan` fingerprints the runtime and manifest together with optional
geometry, topology, execution, and replay identities. `write_kinetic_checkpoint` and
`read_kinetic_checkpoint` use the checksummed pickle-free array archive and exact
PyTree templates. Population fields, runtime controls, boundary history, ledgers,
AMR phase, and AA parity therefore retain exact shape and dtype across continuation.

## Runtime parameters and differentiation

`LatticeBoltzmannRuntimeParameters` carries viscosity, force parameters, moving-wall
velocities, staged boundary parameters, and an optional local root solver. Runtime
arrays are normal PyTrees. Lattice choice, grid, collision family, topology,
step count, decomposition, and export ABI are structural.

Differentiation is supported through fixed schedules and fixed ownership. Hard
geometry predicates, topology changes, adaptive level choice, failure transitions,
and shock ownership are nondifferentiable decisions. Reverse execution uses
`FixedStepReplayPolicy`:

- `mode="full"`: ordinary reverse-mode storage;
- `mode="step"`: rematerialize each step;
- `mode="block"`: replay fixed-size blocks.

`FixedStepRolloutPlan` independently controls output retention (`final`,
`checkpoints`, or `trajectory`). Replay policy and output retention are separate.

## Execution and export

`PreparedDistributedLatticeBoltzmannDynamics` binds the actual prepared hydrodynamic
step to `ShardedLatticeBoltzmannExecutionPlan` and
`LatticeBoltzmannHaloSchedule`. The global population tensor keeps JAX semantics
under a fixed spatial mesh, Q remains unpartitioned, and the direction-complete halo
schedule certifies every source route. Qualification compares complete populations,
failures, work, and diagnostics against the unpartitioned production dynamics.
Production may omit the duplicate reference realization. No runtime repartitioning
occurs.

`AALatticeBoltzmannPlan` stores logical even/odd parity and includes parity in
checkpoints. `FusedLatticeBoltzmannExecutionPlan` JIT-compiles the full fixed scan
as one XLA program. Qualification compares populations, failure transitions, work,
and diagnostics; production may omit the duplicate reference realization afterward.

`LatticeBoltzmannIREEContract` has `forward` and `forward-vjp` modes. The
forward realization and transpose are separately callable, checksummed IREE
artifacts with explicit differentiable population/runtime-array inputs. The VJP
transposes the exact fixed-step realization; unsupported compiler primitives fail
before publication and never route back through hidden JAX execution.
Objects, callbacks, dynamic geometry, and undeclared leaves are outside the ABI.

## General AMR and replay

`LatticeBoltzmannAMRPlan` prepares any finite hierarchy of integer spatial ratios
at least two. `LatticeBoltzmannAMRScalingPolicy` selects acoustic r, diffusive r²,
or explicit declared substeps; `LatticeBoltzmannAMRTemporalTracePlan` evaluates
declared polynomial coarse traces at each fine fraction. Recursive advance
prolongs, subcycles, restricts, averages down, reports interface conservation and
positivity, and commits all levels atomically.

`prepare_replay_schedule` builds a static uniform or declared-cost split tree
under explicit checkpoint-byte and schedule-operation caps. A
`FixedStepReplayPolicy(\"scheduled\", schedule=...)` never adapts to observed
runtime cost; full, step, block, and scheduled realizations retain identical
fixed-step semantics.

## Qualification and acceptance

Run `tools/lattice_boltzmann_qualification.py` for the qualified baseline and
`tools/kinetic_expansion_qualification.py` for advanced invariants. Run
`tools/kinetic_scientific_qualification.py` for named physics, differentiation,
checkpoint, collision-aware AMR, curved-geometry, DVM, and actual production-sharding
evidence. The scientific artifact reports each evidence level separately; forward
IREE deployment remains unqualified when the matched compiler/runtime is unavailable.

Run `tools/learned_energy_equilibrium_qualification.py` for the separate
stage-one D2V17 learned-equilibrium artifact. It generates a native oracle
dataset, enforces leakage-safe normalization, trains and freezes a small dual
predictor, checks held-out population/flux accuracy and local transactional
collision, and reports paired compiled oracle-versus-learned runtime. Its
passing result applies only to the recorded state box, quadrature, material,
model revision, dtype, and hardware.

Qualification artifacts record software/hardware identity, parameters, tolerances,
errors, conservation defects, throughput, compiler memory evidence, and explicit
pass/fail status. A passing artifact qualifies only the recorded configuration; it
is not a universal stability theorem.

## Commercial candidate envelope

`LatticeBoltzmannOperatingEnvelopePlan` binds the lattice, collision, optional Guo
forcing, precision policy, exact hardware target, physics/boundary labels, and finite
limits for relaxation, Mach and Knudsen numbers, density and density ratio, force,
interface/wall resolution, viscosity ratio, Cahn/capillary numbers, mass drift, and
spurious currents. `preflight()` counts state, temporary, halo, checkpoint, and output
bytes per device. `prepare(...)` refuses an over-budget allocation; the resulting
`PreparedLatticeBoltzmannOperatingEnvelope.execute(point)` returns every named margin
and predicate without changing the numerical method.
`LatticeBoltzmannQualificationProfile` rejects continuum-DNS method labels: these are
lattice-kinetic operating envelopes, not direct numerical simulation profiles.


The profile factories create only unsigned candidates:

- `c0_guo_baseline_profiles`: D2Q9/D3Q19 crossed with BGK/TRT and
  population-space Guo forcing;
- `c1_collision_native_forcing_profile`: D3Q27 central-moment collision with
  collision-native Guo forcing;
- `c2_binary_interface_profiles`: selected D2Q9/TRT free-energy and colour-gradient
  interfaces, with dynamic wetting only when an explicit
  `ConstitutiveDynamicWettingPlan` is bound;
- `c3_passive_transport_profiles`: passive sensible energy, passive Fickian species,
  and fixed Strang reactive thermal/species models;
- `conjugate_thermal_qualification_profile`: a real `ConjugateThermalPlan` bound to
  its passive-energy candidate.

`LatticeBoltzmannQualificationProfile.evaluate()` conjuncts current evidence coverage,
operating admission, exact deployment compatibility, parity evidence, and support
dependencies. `LatticeBoltzmannCommercialEvidence.require()` fails with the specific
missing gates; it never promotes `profile.released`.

`ConjugateThermalPlan` is passive sensible-energy coupling, not compressible total
energy. Preparation freezes one-to-one interface distance, measure, conductivity, heat
capacity, contact resistance, and conductance. For fluid and solid interface
temperatures it evaluates

```text
q = (T_fluid - T_solid)
    / (d_fluid / k_fluid + R_contact + d_solid / k_solid),
Q = q A,
dE_fluid/dt = -Q,  dE_solid/dt = Q.
```

`PreparedConjugateThermalPlan.execute()` returns candidate and accepted fluid/solid
energy state, equal-and-opposite rates, conservation residual, and atomic rollback.
It does not provide a solid momentum solver, volumetric solid mesh transport, or
compressible energy closure.

`LatticeBoltzmannHardwareTarget` and `LatticeBoltzmannDeploymentRecord` name host and
per-host device counts exactly. The reference factory defaults to one host and one
device. JAX `NamedSharding` supplies the actual fixed multi-device execution route;
the record and envelope alone do not launch hosts or prove multi-host scaling. A
topology-changing restart requires an explicit topology-restart relation rather than an
implicit redistribution.
