# Structured finite volume

Phydrax finite volume is a structured, face-first discretization for Cartesian,
stationary mapped, multiblock, and fixed-capacity adaptive grids. It stores conserved
cell averages on interval entities, constructs one oriented flux per directional face,
and updates neighboring cells from that shared contribution.

The execution invariant is

```text
cell volume × state rate = -oriented integrated face flux + volume source
```

Internal face contributions therefore cancel exactly. Grid geometry, physical systems,
numerical methods, boundaries, and time integration remain separate objects.

## Scalar conservation law

```python
import jax.numpy as jnp
import phydrax as phx

support = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(128, periodic=True),),
    axis_names=("x",),
).prepare(jnp.asarray([[0.0], [1.0]]))

finite_volume = phx.discretization.FiniteVolumePlan(
    support,
    field_name="u",
).prepare()

system = phx.equations.ScalarConservationSystem(
    1,
    lambda state, axis, args: args["speed"] * state,
    lambda left, right, axis, args: jnp.full(
        left.shape[:-1], jnp.abs(args["speed"])
    ),
    system_id="linear-advection",
)

problem = phx.equations.ConservationProblemIR(
    "linear-advection",
    "u",
    system,
    phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
)

method = phx.discretization.FiniteVolumeMethodPlan(
    phx.discretization.MUSCLReconstruction(phx.discretization.MCLimiter()),
    phx.discretization.RusanovFluxPlan(),
)

compiled = phx.equations.compile_conservation_problem(
    problem,
    finite_volume,
    method,
)

x = support.structured_axes[0].interval_centers
state = jnp.sin(2.0 * jnp.pi * x)[:, None]
dt = compiled.stable_step(state, {"speed": jnp.asarray(0.7)}, cfl=0.4)
stepper = phx.solver.UnsplitFiniteVolumeSSPRK3Plan(compiled.dynamics)
result = stepper.advance(
    jnp.asarray(0.0),
    state,
    dt,
    {"speed": jnp.asarray(0.7)},
)
```

`FiniteVolumePlan` accepts only interval-primary `PreparedTensorGrid` support. Runtime
state has shape `cell_shape + (component_count,)`; component names and topological
location remain static discretization metadata.
Callable semantics are always explicit identity boundaries. A
`ConservationProblemIR` with `source=...` also requires `source_id=...`;
callable face interpolation requires `function_id`, and a
`CharacteristicSystem` requires `system_id`. These IDs participate in prepared
program fingerprints; callable `repr` or process-local identity is never used.


## Cell and face geometry

A prepared discretization provides:

- `cell_layout`, `cell_centers`, and `cell_volumes`;
- one directional `face_layout`, center, measure, and area vector per axis;
- a cell-average `DiscreteFieldSpace`;
- directional flux-moment spaces;
- canonical positive-axis orientation;
- preparation provenance and resource counts.

Bounded axes have one more face than cells. Periodic axes store one unique face per
cell, including one unique periodic seam. Face measure excludes the normal-axis dual
measure; it is the physical tangential area used to integrate normal flux density.

`AxisDiscretization.quad_weights` are authoritative interval widths for interval-primary
axes. This supports nonuniform Cartesian cells when widths span the declared bounds and
nodes are the resulting cell centers.

## Reconstruction

The method plan selects cell-average to face-trace reconstruction independently from the
interface solver.

| Reconstruction | Order | Differentiability |
|---|---:|---|
| `PiecewiseConstantReconstruction` | 1 | smooth discrete |
| `MUSCLReconstruction` | 2 in smooth regions | frozen limiter decision |
| `WENOReconstructionPlan` | 3 or 5 | almost everywhere |
| `HighResolutionReconstructionPlan("weno_z")` | 5 | almost everywhere |
| `HighResolutionReconstructionPlan("teno")` | 5 | frozen stencil activation |
| `HighResolutionReconstructionPlan("mp5")` | 5 | frozen limiter decision |
| `NonuniformWENOReconstructionPlan` | high order on prepared edges | method dependent |

MUSCL limiters are explicit: minmod, monotonized central, van Leer, and Superbee.
High-resolution bounded faces use explicit physical exterior states and first-order
boundary traces; the documented interior order does not extend through the wall.

Characteristic reconstruction requires an equation-owned eigensystem. Physical systems
live in `phydrax.equations`; reconstruction never owns Euler, multispecies, shallow-water,
or MHD physics.

## Interface solvers

Conservative numerical-flux plans return one normal flux density and one maximum signal
speed:

- `RusanovFluxPlan`;
- `HLLFluxPlan`;
- `HLLCFluxPlan` for Euler;
- `HLLDFluxPlan` for canonical ideal MHD, with explicit HLL fallback evidence;
- `RoeFluxPlan` for characteristic systems;
- `EntropyConservativeEulerFluxPlan`;
- `EntropyStableEulerFluxPlan`.

Preparation rejects incompatible combinations. HLLC requires Euler state layout, Roe
requires a characteristic system, and positivity limiting requires an admissibility
predicate. Entropy-conservative/stable interface residuals use an explicit
`ConvexEntropyPair`; the pair must target the same conservation system.

Wave propagation is a separate interface family, not an optional extension of a flux
result:

- `RoeWavePropagationPlan` returns waves, speeds, and left/right fluctuations;
- `WaveFamilyLimiterPlan` limits each wave family;
- `TransverseWaveSolverPlan` splits a normal fluctuation in a transverse direction.

Bathymetric wet/dry shallow water is a separate balanced-face family.
`ShallowWaterHydrostaticHLLPlan` returns one shared transport flux plus one-sided
hydrostatic bed corrections. `PreparedFiniteVolumeRuntime` blends the complete face
contribution against the piecewise-constant fallback at every SSPRK stage. See
[Shallow water](guides_shallow_water.md).

## Boundaries

Every bounded axis requires a `FiniteVolumeBoundaryPair`; periodic axes require `None`
and derive their seam from topology.

Available policies:

- `ExtrapolationBoundary`;
- `ConstantStateBoundary`;
- `PrescribedStateBoundary`;
- `ReflectiveBoundary`;
- `PrescribedNormalFluxBoundary`.

A prescribed state callback receives time, adjacent interior state, physical face
coordinates, outward normal, and runtime arguments. Reflective boundaries delegate
component parity to the physical system. Direct normal-flux boundaries bypass exterior
state construction and are converted from outward orientation to the canonical face
orientation.

Physical boundary meaning is distinct from numerical closure. A centered diffusive
Dirichlet ghost formula is not silently reused as an advective inflow state.

## Physical systems

`phydrax.equations` owns:

- `ScalarConservationSystem` in one, two, or three dimensions;
- `EulerSystem` in one, two, or three dimensions;
- `HomogeneousMixtureEulerSystem` in one, two, or three dimensions;
- `IdealMHDSystem` with three-vector momentum and magnetic field;
- `ShallowWaterSystem` in one or two dimensions.

System capabilities are explicit abstract contracts for characteristics and
admissibility. Entropy variables, entropy fluxes, and relative entropy are supplied by
an explicit `ConvexEntropyPair`. Unsupported dimensions or numerical combinations fail
during construction or compilation.

## Positivity and entropy

`ConvexStateLimiterPlan` scales reconstructed face states toward admissible cell
averages using fixed-count bisection. It does not claim that an arbitrary time step is
positive; the time integrator must also respect the method CFL and any source
restriction.

Attach a pair through the standard compiler:

```text
compiled = phx.equations.compile_conservation_problem(
    problem,
    finite_volume,
    method,
    entropy_pair=phx.equations.ideal_gas_euler_entropy_pair(problem.system),
)
```

Compiler-attached entropy diagnostics currently support structured and mapped
structured finite volumes. Triangle and modern unstructured compilation reject a
supplied pair until normal-face and ALE/content-rate entropy accounting are explicit.

`residual_with_diagnostics()` then returns volume-weighted total entropy, semidiscrete
entropy rate, source entropy rate, convective entropy rate, admissibility, and
precision evidence under `diagnostics.entropy`. The pair is rejected alongside a
viscous flux plan because viscous entropy production is not yet represented separately.
For bounded domains, the convective rate includes boundary transport; it is not a
closed entropy-production certificate.

## Conservation accounting

Structured, triangle, and unstructured `residual_with_diagnostics()` paths reduce
cell-rate, source, and outward-boundary contributions with a twofold compensated
sum in the prepared reduction dtype. The conservation defect is one signed
accumulation of the final rounded contributions, not a subtraction of independently
rounded totals. Tiny nonzero defects remain dtype-scale evidence; they are not a
discretization-error or cross-device reproducibility guarantee.

Conservative multiblock interfaces, accepted flux ledgers, remap, overset/sliding
coupling, and small-cell redistribution use the same accounting convention.

## Time execution

Spatial dynamics do not own a stepper.

- `UnsplitFiniteVolumeSSPRK3Plan` advances the complete semidiscretization.
- `DirectionalSplitFiniteVolumePlan` provides Godunov or symmetric Strang composition
  of directional residuals.
- Source and viscous contributions are composed separately from directional hyperbolic
  updates.

The stable-step estimate uses face speed, face measure, effective cell volume, and an
optional capacity field. A zero hyperbolic rate produces an infinite hyperbolic step
limit; source stiffness remains an independent solver concern.

## Diffusion and viscous fluxes

Conservative diffusion is finite-volume owned. `FaceCoefficientPlan` makes arithmetic,
harmonic, upwind, or callable interpolation explicit. `ConservativeDiffusionPlan`
constructs scalar, diagonal-tensor, or full-tensor cell-to-face flux followed by
face-to-cell divergence.

`ViscousFluxPlan` provides Newtonian stress and Fourier heat flux using
equation-owned thermodynamic materials and transport closures. Convective and viscous
face contributions remain separate until conservative divergence.

## Incompressible projection

`MACOperatorPlan` prepares geometry-only normal-face velocity and cell-pressure
operators. `PreparedMACOperators` owns compatible divergence and gradient, the volume
gauge, coefficient interpolation, weighted-adjoint evidence, and transform eligibility.
`MACPressureOperatorSpec` adds the frozen closure-aware action
`A p = -D(beta G_h p)`, where the strictly positive cell coefficient is interpolated
once to faces. Static Robin sides impose `alpha p + beta_r dp/dn = value`; wall,
inflow, outlet, and stabilized traction semantics still come from the same
`MACBoundaryPlan`.

`spec.prepare()` certifies positivity/contrast, linearity, symmetry when applicable,
affine boundary lift, JVP/VJP consistency, resource capacity, coefficient identity,
and geometry epoch. `prepared.solve(...)` then returns the candidate, committed value,
compatible right-hand side, residual, gauge defect, boundary power, convergence
evidence, and the selected route. Frozen coefficient or geometry changes require a new
preparation.

The direct routes remain exact representations, not generic sparse direct solvers.
Uniform constant-coefficient periodic/Neumann tensors may select `transform`; a
three-dimensional all-Neumann operator with a named nonperiodic line and constant or
line-structured coefficient may select `hybrid`. Their zero mode compatibility-projects
the right-hand side, pins one line row only in factorization, and returns a
volume-zero-mean pressure. Execution requires the matching prepared transform action
through `direct_solve`; there is no hidden dense solve.

All other symmetric positive actions—including general positive coefficients and
Robin or mixed closures—select native PCG with a frozen positive constant
preconditioner. Stabilized nonsymmetric traction selects FGMRES without pretending the
preconditioner is an exact inverse. Distributed projection is a separate collective
PCG owner. `StructuredSolveTopologyPlan` and `DistributedLineSolvePlan` expose
partition-aware line algorithms (`partitioned-thomas`, `spike`, or power-of-two
balanced `pcr`), but they operate on caller-provided arrays and do not add a
multi-device transport layer to the MAC hybrid route.

### Runtime liquid masks and atmospheric pressure

`MACFreeSurfaceProjectionPlan` retains the same MAC divergence, gradient,
physical-boundary, pairing, and linear-solver contracts while restricting pressure
to a runtime liquid mask. Air pressure is exactly zero; liquid–air interfaces
therefore provide an atmospheric Dirichlet reference. An all-liquid mask activates
the existing zero-mean gauge. The plan is used by fixed-population
[FLIP](guides_flip.md) and does not claim VOF/PLIC or level-set geometry.

### Boundary and symmetry-preserving momentum

`MACBoundaryPlan` declares each nonperiodic side as no-slip, free-slip/symmetry,
velocity or normal-flux inflow, pressure outlet, or stabilized traction-open.
`MACBoundaryProvider` separates static provider identity from dynamic JAX value/rate
leaves. Stage evaluation rejects nonfinite data and incompatible prescribed flux.
Pressure closure is derived from these velocity declarations rather than configured
independently. Open-boundary diagnostics retain advective, pressure, and nonnegative
backflow-stabilization power.

`MACMomentumPlan` prepares conservative face-momentum transport, its weighted
skew-adjoint action, componentwise viscous diffusion, and skew, diffusion-symmetry,
and dissipation evidence under the MAC face-dual measure. The public physical
velocity remains one differently shaped normal-face array per axis;
`PreparedMACOperators.velocity_space` supplies canonical flat temporal coordinates.
The momentum construction uses the compatible symmetry and conservation identities
described by [Verstappen and Veldman (2003)](https://doi.org/10.1016/S0021-9991(03)00126-8)
and the stage-projection formulation in
[Costa (2018)](https://doi.org/10.1016/j.camwa.2018.07.034). Those references and
small benchmark cases do not qualify every MAC boundary, coefficient, resolution,
backend, or time-integration route.

Run the `mac` route of `tools/incompressible_flow_qualification.py` with
`--output benchmarks/mac_incompressible_qualification.json` for the canonical
candidate artifact. A generated artifact
is valid only for its exact support tuple and input/reference/configuration IDs; it
retains raw metrics, gates, status, failure/inconclusive reasons, and artifact ID.
`release_ready` remains false. Assembly can reference a passed artifact in an
unsigned `CapabilityProfile` candidate with `profile.released=false`, but that does
not qualify other MAC closures, coefficients, grids, integrators, or distributed
routes.

```python
finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
boundaries = phx.discretization.MACBoundaryPlan(mac)
momentum = phx.discretization.MACMomentumPlan(
    mac, boundaries=boundaries
).prepare()
pressure = phx.solver.MACPressureProjectionPlan(
    mac, boundaries=boundaries
)
compiled = phx.equations.compile_mac_incompressible_flow(
    phx.equations.IncompressibleFlowProblem(2, viscosity),
    momentum,
    pressure,
)
initial_state = compiled.project_state(face_velocity)
```

For a generalized coefficient, prepare the operator separately from execution:

```python
pressure_operator = phx.solver.MACPressureOperatorSpec(
    mac,
    inverse_momentum,
    boundaries=boundaries,
    solve_method="iterative",
).prepare()
pressure_result = pressure_operator.solve(pressure_rhs)
```

Use `solve_method="direct"`, `"transform"`, or `"hybrid"` only when preparation
accepts that exact representation, and pass its prepared transform solve as
`direct_solve` at execution.

`compile_mac_incompressible_flow` projects every temporal rate and exposes physical
pressure, energy, boundary, divergence, residual, gauge, and step-restriction
evidence. Explicit SSPRK, implicit-diffusion `MACIMEXEulerMethod`, and fixed-step
`MACSBDF2Method` consume the same compiled state. `MACHelmholtzSolvePlan` supports
iterative, certified uniform-transform, and resource-gated transform-line routes.

`MACFlowControlTarget` distinguishes prescribed pressure gradient, volume-weighted
bulk velocity, and frozen-density mass flux. The last freezes one positive density
field into both its observable and its acceleration for the complete response map; it
is not evolving variable-density control. Constant targets receive content identities,
while callable schedules require `schedule_id`.

`MACFlowControlPlan(method, target).prepare()` accepts compiled MAC SSPRK,
IMEX-Euler, or fixed-step SBDF2 methods. For bulk velocity or frozen-density mass flux,
each attempted step evaluates the zero-control and unit-control method-stage maps,
solves the finite dense response system, and rejects rank loss, poor conditioning,
resource excess, response mismatch, boundary/projection/pressure residuals, or a
failed underlying method. `PreparedMACFlowControl.initialize(...)` creates complete
checkpoint state and `step(...)` atomically commits or rolls back it. Prescribed
pressure-gradient targets bypass feedback but retain the same acceptance diagnostics.

```python
target = phx.applications.incompressible_flow.MACFlowControlTarget.bulk_velocity(
    [1.0],
    axes=(0,),
)
controller = phx.applications.incompressible_flow.MACFlowControlPlan(
    fixed_step_method,
    target,
).prepare()
control_state = controller.initialize(start_time, initial_state)
control_step = controller.step(control_state, step_size=step_size)
```

`MACConstantPressureGradientForcing` remains the simpler fixed compiler-space
acceleration `-pressure_gradient / density`; it has no feedback state and
`StructuredMACProductionPlan` never inserts or retunes it.

`MACPlaneWallStatisticsPlan` is an instantaneous raw-statistics route for a two- or
three-dimensional MAC grid with one nonperiodic wall axis and periodic homogeneous
axes. Each normal-face component is arithmetically centered between adjacent faces
before exact cell-volume-weighted plane means, raw second moments, and Reynolds
stresses are formed. Wall-normal means are face-measure-weighted reductions of the
native boundary-face values. Separate lower/upper tangential shear vectors use
one-sided derivatives from explicitly declared no-slip wall velocities to the
nearest cell center in increasing wall coordinate. Both wall velocities default to
zero and are part of the statistics-plan identity. The statistics plan does not
bind or inspect a `MACBoundaryPlan`, so the caller must supply values matching the
prepared flow boundary contract.
Kinetic energy and forcing power use face-dual measures; bulk velocity, divergence
norm, and the exact conventions are reported with the result. Time or ensemble
averaging belongs to production `StreamingMomentPlan`, not this instantaneous
evaluator.

`StructuredMACProductionPlan` requires a prepared fixed-step method, compiled
constant-density MAC dynamics, matching `MACPlaneWallStatisticsPlan`, absolute
`start_time`/`end_time`, `step_size`, and `checkpoint_interval`. Optional output
targets and statistical windows are bounded by that horizon. `maximum_steps` defaults
to the capacity needed to reach the end; the prepared runtime still enforces any
method-specific exact-step lattice. `plan.prepare(checkpoint_root, ...)` binds the
durable store, and `prepared.initialize(face_velocity)` projects physical staggered
faces into the compiled state. MAC checkpoints retain the native real state rather
than applying Hermitian encoding.

### Scalar, variable-density, and coupled dynamics

`MACScalarProblem` owns named cell scalars, conservative centered/upwind transport,
diffusion, source/reaction ledgers, and scalar boundary conditions.
`compile_mac_scalar_buoyancy` couples selected names through `MACBuoyancyLaw` while
using the exact transport face interpolation for kinetic/potential exchange.

Dynamic miscible density is a separate conservative model:
`MACVariableDensityState` stores positive cell density and face momentum, derives
velocity from one face-density policy, shares one mass flux between density and
momentum, and uses `MACVariableDensityProjectionPlan`. It does not claim an EOS,
low-Mach heat expansion, VOF, or multiphase interface physics.

`MACMarkerTransferPlan` builds fixed local cubic tensor B-spline routes on each
staggered face layout and exposes material-measure adjoint gather/spread, moment,
force, torque, and work evidence. `MACImmersedBoundaryProjectionPlan` solves
pressure and prescribed marker constraints together. Its IMEX-Euler and SBDF2
methods evaluate marker kinematics at the attempted stage. The separate
`MACPenaltyIBCFDEMCouplingPlan` and
`advance_mac_penalty_ib_cfd_dem_window` retain approximate penalty forcing, DEM
contact subcycling, ledgers, and atomic rollback.

### Distribution, mapped geometry, and sensitivity

`MACDistributedTopologyPlan` owns pressure/face shardings, interface-face ownership,
and local halo metadata. `MACDistributedProjectionPlan` supplies globally reduced
compatibility, gauge, matrix-free CG, rank agreement, and atomic rollback. Direct
distributed transforms remain unavailable unless an explicit redistribution plan is
added; no hidden global gather is performed.

`MappedMACGeometryPlan` certifies positive mapped cell/face/dual measures,
free-stream preservation, D/G adjointness, and pressure action. `MACALEGeometryPlan`
adds fixed-connectivity stage geometry, grid velocity, relative flux, GCL,
wall-kinematic evidence, and fail-closed projection. Topology changes occur only
through `MACRemeshEpochPlan`, whose transfer is explicitly nondifferentiable.

`MACAdaptiveRolloutPlan` records bounded transactional attempts and an accepted time
grid. `MACFrozenGridReplayPlan` and `MACFixedGridSensitivityPlan` differentiate that
stopped grid. `MACSegmentedShadowingPlan` exposes QR-stabilized least-squares
shadowing with conditioning/residual gates; it returns failure rather than presenting
an uncertified long-time turbulent derivative.

## Stationary mapped grids

`MappedFiniteVolumePlan` maps a prepared Cartesian control-volume geometry while
retaining fixed tensor topology. Preparation computes mapped vertices, cell volumes,
face centers, face measures, and oriented area vectors in one, two, or three dimensions.
It rejects nonpositive orientation or measure.

Generic mapped conservative-state execution currently accepts Rusanov or HLL fluxes,
which evaluate the physical normal flux against mapped unit normals, and remains
stationary. Time-dependent fixed-connectivity geometry is deliberately separate under
the MAC-specific `MACALEGeometryPlan` described above.

## Multiblock and AMR

`ConservativeMultiblockInterfacePlan` computes one conforming or nested 2:1 interface
flux, orients the opposing trace, evaluates on the fine mortar, sums fine integrated
fluxes to coarse faces, and reports the global conservation defect.

The reported interface defect uses the compensated accounting reduction described
above; AMR synchronization values selected to be elementwise identical remain zero
by construction.
`FiniteVolumeMultiblockRuntimePlan` couples the block candidates and every mortar to
one global secondary positivity factor. The same accepted mortar integral is retained
for both sides; an inadmissible fallback causes atomic rejection and returns the
unchanged base states.


`ConservativeAMRSubcyclingPlan` accumulates time-integrated coarse and fine interface
fluxes. `FluxRegister` records orientation, refinement ratio, accumulation time, and the
interface mask. `ConservativeAMRSynchronizationPlan` executes subcycling, reflux, and
covered-cell restriction in that order.

State and parameter gradients are supported for a fixed hierarchy. Refinement tagging,
slot activation, migration routes, and topology changes are discrete and are not
differentiated.

## Bounded passive-tracer transport

`MACPassiveTracerMacCormackPlan` prepares a two-pass semi-Lagrangian MacCormack
operator for periodic, cell-centered `point_value` scalars on a prepared Cartesian
MAC grid. It traces midpoint characteristics through one frozen face-velocity field,
forms the reverse-advection correction, and clips the corrected value to the original
donor-corner envelope.

This is an explicitly nonconservative side lane. Pairing-weighted integral change is
reported as diagnostics and never reclassified as a conserved balance. The operator
must not transport finite-volume cell averages, phase fraction, density, momentum,
temperature, salinity, charge, or energy. Nonperiodic boundaries, mapped grids,
staggered payloads, vector payloads, and adaptive routes are rejected.

`MACPassiveTracerFixedStepMethod` may compose one tracer with an existing fixed-step
method. Carrier velocity is sampled from the pre-step base state. The base and tracer
candidate are committed or rolled back together.

## Differentiability contract

The substrate differentiates the fixed discrete program. Method metadata distinguishes:

- smooth discrete paths;
- almost-everywhere paths;
- frozen branch decisions;
- explicit smooth surrogates;
- unsupported topology decisions.

`RusanovFluxPlan(smooth_epsilon=...)` is an opt-in smooth wave-speed surrogate. Hard
limiters, TENO activation, positivity bisection, and AMR decisions are never silently
presented as globally smooth.

`PreparedFiniteVolumeDynamics.linearize()` returns the residual, a matrix-free JVP, and
a VJP pullback. Preparation, grid shape, reconstruction order, boundary kind, and AMR
topology remain static.

## Current limitations

- Balanced shallow water supports static Cartesian and mapped geometry through the
  explicit `ShallowWaterBathymetryPlan`; arbitrary-normal/ALE face contributions are
  available to fixed-topology stage lowerings.
- Equilibrium WENO-Z records characteristic use, dry-stencil fallback, and eigenbasis
  condition evidence rather than silently switching methods.
- Initial transverse solver support is a primitive building block, not a complete
  three-dimensional CTU implementation.
- Mapped viscous periodic axes require a certified `MappedPeriodicSeamPlan`; undeclared
  or geometrically mismatched seams fail before residual evaluation.
- Periodic Cartesian constrained MHD is executed by
  `UpwindConstrainedTransportPlan` and `ConstrainedMHDSSPRK3Plan`, then composed with
  gravity, cooling, or OU forcing through the prepared balance-law transport adapter.
  Physical MHD boundaries, AMR reflux-curl, mapped grids, and distributed CT remain
  unsupported.
- Hard shock-capturing decisions produce branchwise, not globally smooth, sensitivities.

Runtime, material, boundary, positivity, checkpoint, rollout, and sharding contracts are
documented in
[Structured finite-volume runtime](guides_finite_volume_runtime.md).
