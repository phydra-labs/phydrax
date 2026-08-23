# Structured finite-volume runtime

The structured finite-volume runtime narrows the supported product contract to
single-phase compressible flow on Cartesian uniform or stretched grids. Mapped,
multiblock, AMR, incompressible, shallow-water, MHD, and distributed capabilities remain
explicitly qualified per feature rather than being implied by the core API.

## Materials and transport

`EulerSystem` owns an immutable `IdealGasMaterial`. `CompressibleNavierStokesSystem`
adds an `AbstractTransportClosure`:

```python
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

## Runtime state and case identity

`FiniteVolumeRuntimeState` contains evolving continuation values:

- conservative state, physical time, accepted-step count, step size, and status;
- timestep-controller and integrator state;
- forcing state;
- random state;
- output cursor.

Prepared geometry, equation systems, methods, and callables remain in the compiled
runtime.

`FiniteVolumeCaseSpec` records schema version, runtime/equation/discretization/method/
boundary identities, execution policy, and `FiniteVolumePrecisionPolicy`. Unknown case
fields are rejected rather than ignored.

## Checkpoint and output

`FiniteVolumeCheckpointPlan` binds checkpoint compatibility to one case identity and
precision policy. Checkpoints are pickle-free ZIP containers with:

- a JSON manifest;
- independent NPY array payloads;
- per-array SHA-256 checksums;
- complete payload identity;
- atomic temporary-write replacement.

Visualization output is separate. `FiniteVolumeOutputPlan` writes Cartesian snapshots
to HDF5 and a temporal XDMF collection when the optional `hdf5` extra is installed:

```bash
pip install "phydrax[hdf5]"
```

Mapped XDMF output is not yet exposed because mapped coordinates require a different
XDMF geometry representation.

## Differentiable rollout

`FiniteVolumeRolloutPlan` executes a fixed number of runtime steps through `jax.lax.scan`.
Retention policies are final state, fixed-stride checkpoints, or full trajectory.
Step-level rematerialization uses `jax.checkpoint`.

`gradient_report()` compares:

- forward-mode directional derivative;
- reverse-mode directional derivative;
- centered finite-difference derivative.

The rollout differentiates the fixed discrete program. Positivity activation, retries,
hard limiters, shock motion, and topology changes remain branchwise or unsupported.

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
