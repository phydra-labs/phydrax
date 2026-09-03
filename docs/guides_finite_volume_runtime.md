# Structured finite-volume runtime

The structured finite-volume runtime owns stage positivity, retries, accepted flux
integrals, precision, and continuation state. Compressible flow and Cartesian
bathymetric wet/dry shallow water have separately qualified method contracts. Mapped,
multiblock, AMR, incompressible, MHD, and distributed combinations remain explicitly
qualified rather than being implied by the core API. See
[Shallow water](guides_shallow_water.md) for its balanced-face restrictions.

## Materials and transport

`EulerSystem` owns an immutable `IdealGasMaterial`. `CompressibleNavierStokesSystem`
adds an `AbstractTransportClosure`:

```python
import phydrax as phx

material = phx.equations.IdealGasMaterial(
    1.4,
    287.0,
    density_floor=1e-10,
    pressure_floor=1e-8,
)
transport = phx.equations.SutherlandTransport(
    1.8e-5,
    300.0,
    110.4,
    1004.5,
    0.71,
)
system = phx.equations.CompressibleNavierStokesSystem(
    transport,
    3,
    material=material,
)
```

The equation system owns primitive/conservative conversion, pressure, temperature,
sound speed, admissibility, and physical fluxes. `ViscousFluxPlan` owns only spatial
face-gradient and conservative-divergence mechanics; viscosity and conductivity always
come from the equation's transport closure.

Supported material/transport building blocks:

| Capability | Status |
|---|---|
| Ideal gas | qualified core target |
| Constant viscosity/conductivity | qualified core target |
| Sutherland viscosity + constant Prandtl | qualified core target |
| Stiffened gas | implemented, experimental until separately verified |
| Multispecies transport | not yet supported |

## Physical boundaries and halos

Physical boundaries are equation-aware and separate convective, velocity, and thermal
semantics:

- `SlipWallBoundary`
- `NoSlipAdiabaticWallBoundary`
- `NoSlipIsothermalWallBoundary`
- `PrescribedHeatFluxWallBoundary`
- `SupersonicInflowBoundary`
- `SupersonicOutflowBoundary`
- `CharacteristicInflowBoundary`
- `CharacteristicOutflowBoundary`
- `FarFieldBoundary`

`FiniteVolumeHaloPlan` derives required depth from reconstruction, validates local
extent, and compiles periodic or physical face ownership into one
`PreparedFiniteVolumeHaloPlan`. `materialize_axis()` returns layer-specific ghost states,
mirrored/periodic axis coordinates, and physical ghost centers. High-order,
characteristic, nonuniform, and viscous methods consume this prepared data; they do not
select their own boundary policy.

Characteristic boundaries use the equation-owned eigensystem and remain branchwise
differentiable. No-slip adiabatic, isothermal, and prescribed-heat-flux walls determine
the actual ghost velocity/temperature or accepted boundary energy flux.

## Positivity and retry

`FluxPositivityPlan` compares each high-order SSPRK stage with a monotone
piecewise-constant `EinfeldtHLLFluxPlan` stage. Per-cell admissibility factors are found
by fixed-count bisection and lowered to one shared factor per face. The accepted
time-averaged SSPRK face flux is exposed to diagnostics and AMR flux registers.

`PreparedFiniteVolumeRuntime` applies positivity at every SSPRK3 stage and retries a
rejected step with a reduced timestep according to `FiniteVolumeStepPolicy`.

Statuses are bounded and machine-readable:

- success;
- recovered rejection;
- invalid initial state;
- retry limit reached;
- minimum timestep reached;
- nonfinite state.

Accepted states are never repaired through post-hoc density or pressure clipping.

## Convex entropy diagnostics

Bind an explicit `ConvexEntropyPair` through
`compile_conservation_problem(..., entropy_pair=pair)`. Runtime residual diagnostics
then expose volume-weighted total entropy, semidiscrete entropy rate, source entropy
rate, convective entropy rate, admissibility, and precision evidence.

This compiler-attached path currently covers structured and mapped structured finite
volumes. Triangle and modern unstructured geometry reject an attached pair rather than
silently applying stationary cell-volume formulas to normal-face or ALE content rates.

Entropy-pair diagnostics are rejected with a `ViscousFluxPlan` until viscous entropy
production is represented separately. The current convective rate therefore never
mislabels a viscous contribution. Bounded-domain rates include boundary transport;
periodic, source-free entropy-stable cases are the appropriate sign-check surface.


## Runtime state and case identity

`FiniteVolumeRuntimeState` contains evolving continuation values:

- conservative state, physical time, accepted-step count, step size, and status;
- timestep-controller and integrator state;
- forcing state;
- random state;
- output cursor.

Prepared geometry, equation systems, methods, and callables remain in the compiled
runtime.

`FiniteVolumeCaseSpec` records the canonical case format and
runtime/equation/discretization/method/boundary identities, execution policy, and
`FiniteVolumePrecisionPolicy`. Unknown case fields are rejected rather than ignored.

## Checkpoint and output

`FiniteVolumeCheckpointPlan` binds checkpoint compatibility to one case identity and
precision policy. Checkpoints are pickle-free ZIP containers with:

- a JSON manifest;
- independent NPY array payloads;
- per-array SHA-256 checksums;
- complete payload identity;
- atomic temporary-write replacement.
MAC continuation uses `MACAdaptiveRuntimeState` as the sole restart payload: canonical
velocity coordinates, time, accepted count, requested next step, controller counters,
fixed-capacity accepted-grid history, forcing state, and output cursor.
`MACFiniteVolumeCheckpointPlan` binds the exact dynamics, method, controller, grid,
precision dtype, and leaf shape/dtype template. Checksum, identity, unknown/missing
leaf, truncation, or dtype mismatches fail before `advance` can step.


Visualization output is separate. `FiniteVolumeOutputPlan` writes Cartesian snapshots
to HDF5 and a temporal XDMF collection when the optional `hdf5` extra is installed:

```bash
pip install "phydrax[hdf5]"
```

Mapped XDMF output is not yet exposed because mapped coordinates require a different
XDMF geometry representation.

## Differentiable rollout

`AdaptiveFiniteVolumeRolloutPlan` executes a bounded number of adaptive attempts and
records the accepted prefix as a `RealizedTemporalMesh`.
`ScheduledFiniteVolumeRolloutPlan` consumes an explicit all-active internal
`TemporalMesh`; every interval is either accepted exactly or rejected without changing
state or physical time. It never accepts a CFL clamp or retry.

`FiniteVolumeReplayPolicy` separates reverse storage from output retention:

- `full`: ordinary scan;
- `step`: rematerialize each step body;
- `block`: retain block boundaries and rematerialize each inner block.

Retention remains final state, fixed-stride checkpoints, or full trajectory. Final-only
retention does not materialize a state trajectory.

`gradient_report()` compares forward-mode, reverse-mode, and centered finite-difference
directional derivatives. The fixed temporal mesh and topology are nontrainable.
Positivity activation, hard limiters, shock motion, fallback masks, and schedule
validity remain branchwise.

Stateful self-gravity, stochastic forcing, and cooling use
`PreparedBalanceLawRuntime`, whose prepared transport adapter may own ordinary
finite-volume state or constrained-MHD cell and face-flux state. Symmetric
source/transport composition commits process and adapter state only after the complete
interval succeeds. See
[Differentiable compressible multiphysics](guides_compressible_multiphysics.md).

## Named sharding

`FiniteVolumeDecompositionPlan` validates global shape, split factors, local extent, and
halo reach before constructing `Mesh` and `NamedSharding`. Prepared lower/upper halo
routes record mesh axis, neighbor offset, and depth. The current execution uses
sharding-aware global operations while exposing periodic local halo materialization.

Explicit communication/interior overlap and multi-host qualification remain future
performance work. No `pmap` axis name appears in the public contract.

## Verification

`FiniteVolumeVerificationCase`, physical error norms, convergence reports, and the
`tools/finite_volume_qualification.py` entry point provide machine-readable advection,
Sod, Lax, double-rarefaction, and Woodward–Colella qualification. Couette and Poiseuille
analytic profiles support viscous verification.

Release qualification additionally requires pinned accelerator coverage,
multidimensional references, restart equivalence, performance baselines, and a complete
support matrix. Implemented capability is not automatically supported capability.

## Runtime qualification closure

Face-local positivity is followed by an explicit final-state admissibility check. If
independently limited faces change a cell correction enough to violate admissibility, a
secondary global conservative reduction scales every remaining antidiffusive face flux.
Runtime acceptance requires both fallback and final limited-state validity.

For Navier–Stokes, `ViscousFluxPlan.stability_report()` returns momentum and thermal
diffusion rates and their limiting cell. The runtime step is the minimum of hyperbolic,
viscous, thermal, and user restrictions.

Nonuniform WENO coefficients are re-prepared against the exact periodic or mirrored
ghost-edge geometry during finite-volume dynamics preparation. Runtime ghost values and
prepared coefficient counts must match exactly.

`load_finite_volume_case()` provides an allowlisted portable schema for one-dimensional
ideal-gas Euler and Navier–Stokes cases. Unknown fields and incompatible periodic,
boundary, reconstruction, flux, material, or transport declarations are rejected before
JIT. Checkpoints preserve controller, integrator, forcing, random, and output-cursor
continuation state. The qualification runner advances until the declared final time,
snaps the final step, and reports actual-versus-target time and attempt capacity.
