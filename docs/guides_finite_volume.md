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
- `FWaveShallowWaterPlan` decomposes a bathymetry-balanced flux jump;
- `WaveFamilyLimiterPlan` limits each wave family;
- `TransverseWaveSolverPlan` splits a normal fluctuation in a transverse direction.

The initial f-wave implementation is one-dimensional and preserves lake-at-rest states.

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
- `MultispeciesEulerSystem` in one, two, or three dimensions;
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
operators. `PreparedMACOperators` owns the compatible divergence, gradient,
constant/variable-coefficient pressure actions, volume gauge, coefficient
interpolation, weighted-adjoint evidence, and exact transform eligibility.

`phydrax.solver.MACPressureProjectionPlan` owns closure-aware pressure execution.
Uniform constant-coefficient periodic/Neumann operators may use an exact FFT/DCT
transform after an independent action-identity check; general positive coefficients
and mixed/open closures use prepared `phydrax.linalg` problems. Neumann-only closures
project compatibility and impose a volume gauge. A pressure outlet removes the
constant nullspace and does neither. Every route reports its boundary closure, mass
defect, gauge, pressure residual, pre/post divergence, and atomic commit status.

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
The construction independently follows
[Verstappen and Veldman (2003)](https://doi.org/10.1016/S0021-9991(03)00126-8);
the compatible stage projection is qualified against
[Costa (2018)](https://doi.org/10.1016/j.camwa.2018.07.034).

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

`compile_mac_incompressible_flow` projects every temporal rate and exposes physical
pressure, energy, boundary, divergence, residual, gauge, and step-restriction
evidence. Explicit SSPRK, implicit-diffusion `MACIMEXEulerMethod`, and fixed-step
`MACSBDF2Method` consume the same compiled state. `MACHelmholtzSolvePlan` supports
iterative, certified uniform-transform, and resource-gated transform-line routes.

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

`MACMarkerTransferPlan` gathers MAC face velocity and spreads marker force through
the dual-measure adjoint. `ResolvedMACIBCFDEMCouplingPlan` and
`advance_mac_resolved_ib_window` provide penalty IB force/torque, DEM contact
subcycling, post-forcing projection, work/impulse ledgers, and complete rollback.

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

`ConservativeAMRSubcyclingPlan` accumulates time-integrated coarse and fine interface
fluxes. `FluxRegister` records orientation, refinement ratio, accumulation time, and the
interface mask. `ConservativeAMRSynchronizationPlan` executes subcycling, reflux, and
covered-cell restriction in that order.

State and parameter gradients are supported for a fixed hierarchy. Refinement tagging,
slot activation, migration routes, and topology changes are discrete and are not
differentiated.

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

- No unstructured or polyhedral meshes.
- No moving mapped grids.
- Initial shallow-water f-wave support is one-dimensional.
- Initial transverse solver support is a primitive building block, not a complete
  three-dimensional CTU implementation.
- Mapped fluxes currently use Rusanov or HLL.
- Periodic Cartesian constrained MHD is executed by
  `UpwindConstrainedTransportPlan` and `ConstrainedMHDSSPRK3Plan`, then composed with
  gravity, cooling, or OU forcing through the prepared balance-law transport adapter.
  Physical MHD boundaries, AMR reflux-curl, mapped grids, and distributed CT remain
  unsupported.
- Hard shock-capturing decisions produce branchwise, not globally smooth, sensitivities.

Runtime, material, boundary, positivity, checkpoint, rollout, and sharding contracts are
documented in
[Structured finite-volume runtime](guides_finite_volume_runtime.md).
