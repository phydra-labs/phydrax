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
| Phase-field, colour-gradient, free-energy, thermal, passive species | Implemented research path | Separate distributions and explicit mass/energy/species ledgers |
| Reactive species with Strang splitting | Implemented research path | Atomic rollback across flow, thermal, and species states |
| D2V17 and off-lattice D2V37 smooth compressible kinetic methods | Implemented research path | Total energy is a kinetic population; off-lattice transport is explicit |
| Fixed FV/kinetic interface | Experimental infrastructure | One common conservative flux, FV shock ownership, atomic update |
| Mapped-grid kinetic method | Experimental research path | Metric-identity/free-stream residuals and injected geometric source |
| Sharded execution and halo exchange | Implemented execution path | Fixed decomposition, deterministic ownership, no dynamic repartitioning |
| AA/even-odd memory layout and fused step | Implemented execution path | Logical-state equivalence evidence and checkpoint parity metadata |
| Block reverse replay | Implemented execution path | Fixed-size replay blocks; no adaptive recomputation policy |
| IREE export | Forward inference only | Stable tuple ABI; gradients remain in JAX |

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

`AbstractBulkFreeEnergy` and `DoubleWellFreeEnergy` are shared equation-layer
constitutive contracts; application workflows consume the same source of truth.
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

`ShardedLatticeBoltzmannExecutionPlan` and `LatticeBoltzmannHaloSchedule`
describe a fixed device mesh, an unpartitioned Q axis, and direction-selected
face/edge/corner halo routes. Qualification compares the global reference; production
can omit that duplicate realization. No runtime repartitioning occurs.

`AALatticeBoltzmannPlan` stores logical even/odd parity and includes parity in
checkpoints. `FusedLatticeBoltzmannExecutionPlan` JIT-compiles the full fixed scan
as one XLA program. Qualification compares populations, failure transitions, work,
and diagnostics; production may omit the duplicate reference realization afterward.

`LatticeBoltzmannIREEForwardContract` exports stable tuple inputs and outputs for
forward inference. Objects, callbacks, adaptive geometry rebuilds, and gradients are
not part of the IREE ABI. Training and differentiation remain in JAX.

## Qualification and acceptance

Run `tools/lattice_boltzmann_qualification.py` for the baseline shear-decay,
Poiseuille, Couette, and runtime artifact. Run
`tools/kinetic_expansion_qualification.py` for advanced invariants: collision
conservation/positivity evidence, boundary write-once ownership, geometry-transfer
conservation, multiphysics ledgers, DVM quadrature/energy evidence, single-device
versus sharded equivalence, AA parity, and replay equivalence.

Qualification artifacts record software/hardware identity, parameters, tolerances,
errors, conservation defects, throughput, compiler memory evidence, and explicit
pass/fail status. A passing artifact qualifies only the recorded configuration; it
is not a universal stability theorem.
